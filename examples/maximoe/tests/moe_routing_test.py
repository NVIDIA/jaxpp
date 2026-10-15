# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Routing kernels against the ``xla`` specification.

``maximoe.kernels.moe_routing.xla`` defines every operation. ``RoutingSpecTest``
checks that the specification's hand-written backward pieces agree with JAX
autodiff of its forward functions. ``XlaRoutingTracingTest`` and
``MosaicGpuRoutingTracingTest`` check tracing and shape contracts on any device.
``XlaRoutingPrecisionTest`` holds numerical precision contracts;
``MosaicGpuRoutingPrecisionTest`` reruns them on a supported GPU and checks
values and gradients against the specification on the rows it defines: rows a
live route targets, and tokens inside live prefixes. Properties only the Mosaic
kernels have are covered by ``moe_routing_mosaic_gpu_test``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import AbstractMesh
from jax.sharding import PartitionSpec as P

from maximoe import test_util
from maximoe.kernels.moe_routing import mosaic_gpu, xla

jax.config.parse_flags_with_absl()

HIDDEN_SIZES = [150, 1024]
EXPERT_MESH = AbstractMesh(
    (2,), ("expert",), axis_types=(jax.sharding.AxisType.Explicit,)
)


def targeted_rows(positions, extent) -> np.ndarray:
    """Rows of an expert-major buffer that live routes write."""
    positions = np.asarray(positions)
    written = np.zeros(extent, dtype=bool)
    written[positions[positions < extent]] = True
    return written


def dead_route_nan(weights, positions, extent):
    """Weights are unspecified at dead routes; NaN there must not leak."""
    return jnp.where(positions < extent, weights, jnp.nan)


def expert_major_routes(hidden, seed=0):
    """Two batches of four tokens with two routes, some dead, extent 24."""
    extent = 24
    positions = jnp.array(
        [
            [[0, 1], [2, 3], [4, extent], [5, 6]],
            [[7, 8], [9, 10], [extent, 11], [12, extent]],
        ],
        jnp.int32,
    )
    buffer_key, weight_key = jax.random.split(jax.random.key(seed))
    buffer = jax.random.normal(buffer_key, (extent, hidden), jnp.bfloat16)
    weights = jax.random.normal(weight_key, positions.shape, jnp.bfloat16)
    weights = dead_route_nan(weights.at[0, 0, 1].set(0), positions, extent)
    return buffer, weights, positions


def batched_routes(hidden, seed=1):
    """Two batches with private buffers of extent 12."""
    extent = 12
    positions = jnp.array(
        [
            [[0, 1], [2, extent], [4, 5], [extent, 7]],
            [[1, 0], [3, 2], [extent, 6], [8, extent]],
        ],
        jnp.int32,
    )
    buffer_key, weight_key = jax.random.split(jax.random.key(seed))
    buffer = jax.random.normal(buffer_key, (2, extent, hidden), jnp.bfloat16)
    weights = jax.random.normal(weight_key, positions.shape, jnp.bfloat16)
    return buffer, dead_route_nan(weights, positions, extent), positions


def compact_routes(hidden, seed=2):
    """Three batches whose live token prefixes are 17, 0, and 31 of 64."""
    batch, sequence, routes = 3, 64, 2
    group_sizes = jnp.array([17, 0, 31], jnp.int32)
    extent = 48 * routes
    positions = jnp.full((batch, sequence, routes), extent, jnp.int32)
    positions = positions.at[0, :17].set(
        jnp.arange(34, dtype=jnp.int32).reshape(17, routes)
    )
    positions = positions.at[2, :31].set(
        jnp.arange(34, extent, dtype=jnp.int32).reshape(31, routes)
    )
    buffer_key, weight_key = jax.random.split(jax.random.key(seed))
    buffer = jax.random.normal(buffer_key, (extent, hidden), jnp.bfloat16)
    weights = jax.random.normal(weight_key, positions.shape, jnp.bfloat16)
    return buffer, weights, positions, group_sizes


def live_tokens(group_sizes, sequence) -> np.ndarray:
    return np.arange(sequence)[None, :] < np.asarray(group_sizes)[:, None]


def trace_shard_map(function, args, in_specs, out_specs):
    mapped = jax.shard_map(
        function,
        mesh=EXPERT_MESH,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=True,
    )
    with jax.sharding.use_abstract_mesh(EXPERT_MESH):
        return jax.jit(mapped).trace(*args)


class RoutingSpecTest(test_util.MaximoeTestCase):
    """Hand-written backward pieces of the specification agree with autodiff."""

    def test_combine_gradient_matches_hand_written_backward_pieces(self):
        buffer, weights, positions = expert_major_routes(150)
        buffer = buffer.astype(jnp.float32)
        weights = weights.astype(jnp.float32)
        extent = buffer.shape[-2]
        grad_out = jax.random.normal(
            jax.random.key(10), (*positions.shape[:2], 150), jnp.float32
        )

        _, vjp = jax.vjp(
            lambda b, w: xla.combine(b, w, positions, False), buffer, weights
        )
        d_buffer, d_weights = vjp(grad_out)

        # This is how ``RaggedDotCombine.vjp_bwd`` composes the backward of
        # ``combine`` around the down projection; the kernels have no other VJP.
        expert_cotangent = xla.combine_transpose(grad_out, positions, extent)
        expected_d_buffer, expected_d_weights = xla.weight_expert_rows(
            expert_cotangent, weights, positions, buffer
        )

        self.assertTrue(np.isfinite(np.asarray(d_buffer)).all())
        self.assertTrue(np.isfinite(np.asarray(d_weights)).all())
        self.assertClose(d_buffer, expected_d_buffer, rtol=1e-5, atol_scale=1e-5)
        self.assertClose(d_weights, expected_d_weights, rtol=1e-5, atol_scale=1e-5)
        untargeted = ~targeted_rows(positions, extent)
        self.assertArraysEqual(np.asarray(d_buffer)[untargeted], 0, check_dtypes=False)
        self.assertArraysEqual(
            np.asarray(d_weights)[np.asarray(positions >= extent)],
            0,
            check_dtypes=False,
        )

    def test_gather_sum_gathers_per_batch(self):
        buffer, _, positions = batched_routes(8)
        buffer = buffer.astype(jnp.float32)
        extent = buffer.shape[1]

        output = xla.gather_sum(buffer, positions)

        expected = np.zeros(output.shape, np.float32)
        for batch, token, route in np.ndindex(positions.shape):
            position = int(positions[batch, token, route])
            if position < extent:
                expected[batch, token] += np.asarray(
                    buffer[batch, position], np.float32
                )
        self.assertAllClose(output, expected, rtol=1e-5, atol=1e-5)

    def test_combine_zeroes_tokens_past_live_prefix(self):
        buffer, weights, positions, group_sizes = compact_routes(16)
        _, sequence, _ = positions.shape

        prefixed = xla.combine(buffer, weights, positions, False, group_sizes)
        dense = xla.combine(buffer, weights, positions, False)

        live = live_tokens(group_sizes, sequence)
        prefixed = np.asarray(prefixed)
        self.assertArraysEqual(prefixed[live], np.asarray(dense)[live])
        self.assertArraysEqual(prefixed[~live], 0, check_dtypes=False)

    def test_combine_transpose_honours_live_prefixes(self):
        _, _, positions, group_sizes = compact_routes(16)
        batch, sequence, routes = positions.shape
        extent = 48 * routes
        grad_out = jax.random.normal(
            jax.random.key(11), (batch, sequence, 16), jnp.float32
        )

        dense = xla.combine_transpose(grad_out, positions, extent)
        prefixed = xla.combine_transpose(grad_out, positions, extent, None, group_sizes)
        self.assertArraysEqual(dense, prefixed)

        live = live_tokens(group_sizes, sequence)
        expected = np.repeat(np.asarray(grad_out)[live][:, None, :], routes, axis=1)
        self.assertArraysEqual(dense, expected.reshape(extent, 16))

        dead_prefix = xla.combine_transpose(
            grad_out, positions, extent, None, jnp.zeros_like(group_sizes)
        )
        self.assertArraysEqual(dead_prefix, 0, check_dtypes=False)

    def test_permute_gradient_matches_gather_sum(self):
        tokens, packed_size, hidden = 4, 16, 8
        positions = jnp.array(
            [[0, 1], [2, packed_size], [4, 5], [packed_size, 7]], jnp.int32
        )
        inputs = jax.random.normal(jax.random.key(12), (tokens, hidden), jnp.float32)
        cotangent = jax.random.normal(
            jax.random.key(13), (packed_size, hidden), jnp.float32
        )

        output, pullback = jax.vjp(
            lambda x: xla.permute(x, positions, packed_size), inputs
        )
        (grad_inputs,) = pullback(cotangent)

        expected_output = np.zeros((packed_size, hidden), np.float32)
        for token, route in np.ndindex(positions.shape):
            if positions[token, route] < packed_size:
                expected_output[int(positions[token, route])] = inputs[token]
        self.assertArraysEqual(output, expected_output)
        self.assertArraysEqual(grad_inputs, xla.gather_sum(cotangent, positions))
        batched = xla.gather_sum(
            jnp.stack((cotangent, cotangent)), jnp.stack((positions, positions))
        )
        self.assertArraysEqual(batched[1], grad_inputs)

    def test_rejects_mismatched_route_shapes(self):
        with self.assertRaisesRegex(ValueError, r"positions.shape\[0\]"):
            xla.permute(jnp.zeros((4, 8)), jnp.zeros((3, 2), jnp.int32), 16)
        with self.assertRaisesRegex(ValueError, "gather_sum expects"):
            xla.gather_sum(jnp.zeros((16, 8)), jnp.zeros((2, 4, 2), jnp.int32))


class XlaRoutingTracingTest(test_util.MaximoeTestCase):
    """Contracts shared by every implementation, checked on the specification."""

    implementation = xla

    # --- Manual-axis contracts ---------------------------------------------

    @parameterized.parameters("positions", "weights")
    def test_combine_output_varies_with_route_metadata(self, varying_metadata):
        """Mosaic must expose the same route-induced variation as the XLA spec."""
        weights_vary = varying_metadata == "weights"
        positions_vary = varying_metadata == "positions"
        weights_spec = P("expert", None, None) if weights_vary else P()
        positions_spec = P("expert", None, None) if positions_vary else P()
        args = (
            jax.ShapeDtypeStruct(
                (16, 32), jnp.bfloat16, sharding=jax.NamedSharding(EXPERT_MESH, P())
            ),
            jax.ShapeDtypeStruct(
                (2 if weights_vary else 1, 4, 2),
                jnp.bfloat16,
                sharding=jax.NamedSharding(EXPERT_MESH, weights_spec),
            ),
            jax.ShapeDtypeStruct(
                (2 if positions_vary else 1, 4, 2),
                jnp.int32,
                sharding=jax.NamedSharding(EXPERT_MESH, positions_spec),
            ),
        )
        in_specs = (P(), weights_spec, positions_spec)

        def combine(buffer, weights, positions):
            return self.implementation.combine(buffer, weights, positions, False)

        with self.assertRaisesRegex(ValueError, "require replication"):
            trace_shard_map(combine, args, in_specs, P())
        trace_shard_map(combine, args, in_specs, P("expert", None, None))

    def test_combine_transpose_output_varies_with_positions(self):
        """Transpose must retain variation introduced by route positions."""
        args = (
            jax.ShapeDtypeStruct(
                (1, 4, 32), jnp.bfloat16, sharding=jax.NamedSharding(EXPERT_MESH, P())
            ),
            jax.ShapeDtypeStruct(
                (2, 4, 2),
                jnp.int32,
                sharding=jax.NamedSharding(EXPERT_MESH, P("expert", None, None)),
            ),
        )
        in_specs = (P(), P("expert", None, None))

        def combine_transpose(grad_out, positions):
            return self.implementation.combine_transpose(grad_out, positions, 8)

        with self.assertRaisesRegex(ValueError, "require replication"):
            trace_shard_map(combine_transpose, args, in_specs, P())
        trace_shard_map(combine_transpose, args, in_specs, P("expert", None))

    def test_gather_sum_output_varies_with_positions(self):
        args = (
            jax.ShapeDtypeStruct(
                (1, 16, 32), jnp.bfloat16, sharding=jax.NamedSharding(EXPERT_MESH, P())
            ),
            jax.ShapeDtypeStruct(
                (2, 4, 2),
                jnp.int32,
                sharding=jax.NamedSharding(EXPERT_MESH, P("expert", None, None)),
            ),
        )
        in_specs = (P(), P("expert", None, None))
        gather_sum = self.implementation.gather_sum

        with self.assertRaisesRegex(ValueError, "require replication"):
            trace_shard_map(gather_sum, args, in_specs, P())
        trace_shard_map(gather_sum, args, in_specs, P("expert", None, None))

    # --- Shape validation --------------------------------------------------

    @parameterized.parameters((1, 4), (2, 1), (3, 4), (2, 5))
    def test_combine_transpose_rejects_mismatched_token_shapes(self, batch, sequence):
        with self.assertRaisesRegex(ValueError, "matching batch and sequence dims"):
            jax.eval_shape(
                lambda g, p: self.implementation.combine_transpose(g, p, 16),
                jax.ShapeDtypeStruct((batch, sequence, 32), jnp.float32),
                jax.ShapeDtypeStruct((2, 4, 2), jnp.int32),
            )

    @parameterized.parameters((1, 2), (2, 1))
    def test_gather_sum_rejects_mismatched_batch_counts(self, buffers, routes):
        with self.assertRaisesRegex(ValueError, r"positions.shape\[0\].*must equal"):
            jax.eval_shape(
                self.implementation.gather_sum,
                jax.ShapeDtypeStruct((buffers, 16, 32), jnp.float32),
                jax.ShapeDtypeStruct((routes, 4, 2), jnp.int32),
            )

    def test_weight_expert_rows_rejects_batched_buffers(self):
        buffer, weights, positions = batched_routes(8)
        with self.assertRaisesRegex(ValueError, r"weight_expert_rows expects \[P, M\]"):
            jax.eval_shape(
                self.implementation.weight_expert_rows, buffer, weights, positions
            )

    def test_weight_expert_rows_rejects_mismatched_dot_values(self):
        buffer, weights, positions = expert_major_routes(8)
        route_dot_values = jnp.zeros((2, *buffer.shape), buffer.dtype)
        with self.assertRaisesRegex(
            ValueError, "route_dot_values shape .* must match buffer"
        ):
            jax.eval_shape(
                self.implementation.weight_expert_rows,
                buffer,
                weights,
                positions,
                route_dot_values,
            )

    def test_combine_rejects_batched_buffers(self):
        buffer, weights, positions = batched_routes(8)
        with self.assertRaisesRegex(ValueError, r"combine expects \[P, M\]"):
            jax.eval_shape(
                lambda b, w, p: self.implementation.combine(b, w, p, False),
                buffer,
                weights,
                positions,
            )


class MosaicGpuRoutingTracingTest(XlaRoutingTracingTest):
    """Mosaic tracing and shape contracts do not require GPU execution."""

    implementation = mosaic_gpu


class XlaRoutingPrecisionTest(test_util.MaximoeTestCase):
    """Numerical precision contracts shared by every implementation."""

    implementation = xla

    def test_persistent_prefixes_are_bounded_by_sequence_length(self):
        buffer, weights, positions = expert_major_routes(150)
        extent = buffer.shape[0]
        group_sizes = jnp.array([-1, positions.shape[1] + 1], jnp.int32)
        bounded_sizes = jnp.clip(group_sizes, 0, positions.shape[1])
        cotangent = jnp.ones((*positions.shape[:2], 150), buffer.dtype)

        actual = self.implementation.combine(
            buffer, weights, positions, False, group_sizes
        )
        expected = xla.combine(buffer, weights, positions, False, bounded_sizes)
        # The empty batch's output is unspecified in the Mosaic implementation.
        self.assertClose(actual[1], expected[1])

        actual_grad = self.implementation.combine_transpose(
            cotangent, positions, extent, None, group_sizes
        )
        expected_grad = xla.combine_transpose(
            cotangent, positions, extent, None, bounded_sizes
        )
        self.assertArraysEqual(
            np.asarray(actual_grad)[targeted_rows(positions[1], extent)],
            np.asarray(expected_grad)[targeted_rows(positions[1], extent)],
        )

    @parameterized.parameters(False, True)
    def test_weight_expert_rows_preserves_fp32_probabilities(self, with_route_dot):
        buffer = jnp.full((2, 150), 1.5, jnp.bfloat16)
        weights = jnp.array([[[1.003, jnp.nan]]], jnp.float32)
        positions = jnp.array([[[0, 2]]], jnp.int32)
        dot_values = jnp.full_like(buffer, 2) if with_route_dot else None

        actual = self.implementation.weight_expert_rows(
            buffer, weights, positions, dot_values
        )

        if with_route_dot:
            actual, route_dot = actual
            self.assertEqual(route_dot.dtype, jnp.float32)
            self.assertArraysEqual(route_dot, np.array([[[450.0, 0.0]]], np.float32))
        # BF16(1.003) == 1, but BF16(1.5 * 1.003) == 1.5078125.
        # Round only after the product; leave untargeted rows unchanged.
        self.assertEqual(actual.dtype, jnp.bfloat16)
        self.assertArraysEqual(actual[0], np.full(150, 1.5078125), check_dtypes=False)
        self.assertArraysEqual(actual[1], np.full(150, 1.5), check_dtypes=False)

    def test_combine_preserves_fp32_probabilities(self):
        buffer = jnp.full((1, 150), 1.5, jnp.bfloat16)
        weights = jnp.array([[[1.003, jnp.nan]]], jnp.float32)
        positions = jnp.array([[[0, 1]]], jnp.int32)

        actual = self.implementation.combine(buffer, weights, positions, False)

        self.assertEqual(actual.dtype, jnp.bfloat16)
        self.assertArraysEqual(
            actual, np.full((1, 1, 150), 1.5078125), check_dtypes=False
        )


class MosaicGpuRoutingPrecisionTest(XlaRoutingPrecisionTest):
    """The Mosaic kernels against the specification on the rows it defines."""

    implementation = mosaic_gpu

    def setUp(self):
        super().setUp()
        self.skip_unless_gpus()

    @parameterized.product(
        hidden=HIDDEN_SIZES,
        activations_in_float32=[False, True],
        weighted=[False, True],
    )
    def test_combine_value_and_grad_match_spec(
        self, hidden, activations_in_float32, weighted
    ):
        """``combine`` and the transpose/weighting composition that
        ``RaggedDotCombine.vjp_bwd`` uses as its backward match autodiff of
        the specification."""
        buffer, weights, positions = expert_major_routes(hidden)
        extent = buffer.shape[-2]
        cotangent = jax.random.normal(
            jax.random.key(3), (*positions.shape[:2], hidden), jnp.float32
        )
        weights = weights if weighted else None

        def loss(buffer, weights):
            output = xla.combine(buffer, weights, positions, activations_in_float32)
            return jnp.sum(output.astype(jnp.float32) * cotangent), output

        argnums = (0, 1) if weighted else (0,)
        (_, expected_output), expected_grads = jax.value_and_grad(
            loss, argnums, has_aux=True
        )(buffer, weights)
        expected = (expected_output, *expected_grads)

        kernels = self.implementation
        output = kernels.combine(buffer, weights, positions, activations_in_float32)
        expert_cotangent = kernels.combine_transpose(
            cotangent.astype(buffer.dtype), positions, extent
        )
        if weighted:
            grads = kernels.weight_expert_rows(
                expert_cotangent, weights, positions, buffer, None
            )
        else:
            grads = (expert_cotangent,)
        actual = (output, *grads)

        self.assertEqual(actual[0].dtype, expected[0].dtype)
        self.assertEqual(
            actual[0].dtype, jnp.float32 if activations_in_float32 else buffer.dtype
        )
        self.assertClose(actual[0], expected[0])
        self.assertClose(actual[1], expected[1], targeted_rows(positions, extent))
        if weighted:
            self.assertEqual(actual[2].dtype, weights.dtype)
            self.assertClose(actual[2], expected[2])
            self.assertArraysEqual(
                np.asarray(actual[2])[np.asarray(positions >= extent)],
                0,
                check_dtypes=False,
            )

    @parameterized.product(hidden=HIDDEN_SIZES, activations_in_float32=[False, True])
    def test_combine_live_prefixes_match_spec(self, hidden, activations_in_float32):
        buffer, weights, positions, group_sizes = compact_routes(hidden)

        expected = xla.combine(
            buffer, weights, positions, activations_in_float32, group_sizes
        )
        actual = self.implementation.combine(
            buffer, weights, positions, activations_in_float32, group_sizes
        )

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertClose(actual, expected, live_tokens(group_sizes, positions.shape[1]))

    @parameterized.parameters(*HIDDEN_SIZES)
    def test_combine_transpose_matches_spec(self, hidden):
        _, _, positions = expert_major_routes(hidden)
        extent = 24
        grad_out = jax.random.normal(
            jax.random.key(4), (*positions.shape[:2], hidden), jnp.bfloat16
        )

        expected = xla.combine_transpose(grad_out, positions, extent)
        actual = self.implementation.combine_transpose(grad_out, positions, extent, 13)

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        written = targeted_rows(positions, extent)
        self.assertArraysEqual(
            np.asarray(actual)[written], np.asarray(expected)[written]
        )

    @parameterized.parameters(*HIDDEN_SIZES)
    def test_combine_transpose_live_prefixes_match_spec(self, hidden):
        _, _, positions, group_sizes = compact_routes(hidden)
        batch, sequence, _ = positions.shape
        extent = 48 * 2
        grad_out = jax.random.normal(
            jax.random.key(6), (batch, sequence, hidden), jnp.bfloat16
        )

        expected = xla.combine_transpose(grad_out, positions, extent, None, group_sizes)
        actual = self.implementation.combine_transpose(
            grad_out, positions, extent, None, group_sizes
        )

        # Every buffer row belongs to a live token here, so compare all of them.
        self.assertTrue(targeted_rows(positions, extent).all())
        self.assertArraysEqual(actual, expected)

    @parameterized.product(hidden=HIDDEN_SIZES, with_route_dot=[False, True])
    def test_weight_expert_rows_matches_spec(self, hidden, with_route_dot):
        buffer, weights, positions = expert_major_routes(hidden)
        route_dot_values = (
            jax.random.normal(jax.random.key(7), buffer.shape, buffer.dtype)
            if with_route_dot
            else None
        )

        expected = xla.weight_expert_rows(
            buffer, weights, positions, route_dot_values, 13
        )
        # The kernel donates and rewrites ``buffer``, so it runs last.
        actual = self.implementation.weight_expert_rows(
            buffer, weights, positions, route_dot_values, 13
        )

        if with_route_dot:
            (actual, actual_route_dot), (expected, expected_route_dot) = (
                actual,
                expected,
            )
            self.assertEqual(actual_route_dot.dtype, weights.dtype)
            self.assertClose(actual_route_dot, expected_route_dot)
            dead = np.asarray(positions >= buffer.shape[-2])
            self.assertArraysEqual(
                np.asarray(actual_route_dot)[dead], 0, check_dtypes=False
            )
        # Untargeted rows keep their original values, so every row must agree.
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertArraysEqual(actual, expected)

    @parameterized.parameters(*HIDDEN_SIZES, 7168)
    def test_permute_value_and_grad_match_spec(self, hidden):
        tokens, packed_size = 4, 16
        positions = jnp.array(
            [[0, 1], [2, packed_size], [4, 5], [packed_size, 7]], jnp.int32
        )
        inputs_key, cotangent_key = jax.random.split(jax.random.key(8))
        inputs = jax.random.normal(inputs_key, (tokens, hidden), jnp.bfloat16)
        cotangent = jax.random.normal(
            cotangent_key, (packed_size, hidden), jnp.bfloat16
        )

        def run(kernels):
            output, pullback = jax.vjp(
                lambda values: kernels.permute(values, positions, packed_size), inputs
            )
            (grad_inputs,) = pullback(cotangent)
            return output, grad_inputs

        actual_output, actual_grad = run(self.implementation)
        expected_output, expected_grad = run(xla)

        written = targeted_rows(positions, packed_size)
        self.assertArraysEqual(
            np.asarray(actual_output)[written], np.asarray(expected_output)[written]
        )
        self.assertClose(actual_grad, expected_grad)

    @parameterized.product(
        hidden=HIDDEN_SIZES + [7168], dtype=[jnp.bfloat16, jnp.float32]
    )
    def test_gather_sum_batched_matches_spec(self, hidden, dtype):
        batch, tokens, packed_size = 2, 4, 16
        positions = jnp.array(
            [
                [[0, 1], [2, packed_size], [4, 5], [packed_size, 7]],
                [[9, 8], [packed_size, packed_size], [3, 2], [10, 11]],
            ],
            jnp.int32,
        )
        buffer = jax.random.normal(
            jax.random.key(9), (batch, packed_size, hidden), dtype
        )

        expected = xla.gather_sum(buffer, positions)
        actual = self.implementation.gather_sum(buffer, positions)

        self.assertEqual(actual.shape, (batch, tokens, hidden))
        self.assertEqual(expected.shape, (batch, tokens, hidden))
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(expected.dtype, dtype)
        self.assertClose(actual, expected)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
