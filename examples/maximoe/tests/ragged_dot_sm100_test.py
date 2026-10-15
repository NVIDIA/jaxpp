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

"""Blackwell-specific behavior of the ragged-dot kernels.

Agreement with the specification on the rows it defines is covered by
``ragged_dot_test``. These tests pin properties only the SM100 kernels have:
masking of unaligned compact groups in the persistent dW pipeline, in-place
accumulation, the fused SwiGLU epilogue and the saved preactivation its
backward replays, and the static cost estimates attached to kernel launches.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import jaxpr_util
from jax._src import test_util as jtu
from jax._src.ad_checkpoint import saved_residuals
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import AbstractMesh
from jax.sharding import PartitionSpec as P

from maximoe import test_util
from maximoe.kernels import gpu, ragged_dot
from maximoe.kernels.ragged_dot import sm100, xla

jax.config.parse_flags_with_absl()

CONFIG = sm100.TuningConfig(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    max_concurrent_steps=2,
    collective=False,
    grid_tile_width=1,
    epilogue_tile_n=64,
)
DW_CONFIG = dataclasses.replace(CONFIG, epilogue_tile_n=128)


def make_dw_inputs(group_sizes, tail_value=None):
    rng = np.random.default_rng(123)
    total = int(np.sum(group_sizes))
    allocated_rows = ((total + 7) // 8 + 1) * 8
    lhs = rng.normal(size=(allocated_rows, 128)).astype(np.float32)
    dout = rng.normal(size=(allocated_rows, 128)).astype(np.float32)
    if tail_value is not None:
        lhs[total:] = tail_value
        dout[total:] = tail_value
    return jnp.asarray(lhs, jnp.bfloat16), jnp.asarray(dout, jnp.bfloat16)


def dw(lhs, dout, group_sizes, accumulator=None):
    groups = jnp.asarray(group_sizes, dtype=jnp.int32)
    return sm100.ragged_dot_dw_kernel(
        lhs, dout, groups, DW_CONFIG, accumulator=accumulator
    )


class Sm100TestCase(test_util.MaximoeTestCase):
    def setUp(self):
        super().setUp()
        self.skip_unless_gpus(architecture=gpu.GpuArchitecture.SM100)


class DwKernelTest(Sm100TestCase):
    def test_ignores_unused_trailing_rows(self):
        groups = [100, 64]
        lhs_zero, dout_zero = make_dw_inputs(groups, tail_value=0.0)
        lhs_nan, dout_nan = make_dw_inputs(groups, tail_value=np.nan)

        expected = jax.jit(lambda x, dy: dw(x, dy, groups))(lhs_zero, dout_zero)
        actual = jax.jit(lambda x, dy: dw(x, dy, groups))(lhs_nan, dout_nan)

        self.assertTrue(jnp.isfinite(actual).all())
        self.assertArraysEqual(actual, expected)

    def test_masks_compact_misaligned_group_tail(self):
        # Both TMA heads and contraction tails must exclude neighboring experts.
        groups = [101, 73]
        lhs, dout = make_dw_inputs(groups, tail_value=np.nan)

        actual = jax.jit(lambda x, dy: dw(x, dy, groups))(lhs, dout)
        expected = xla.ragged_dot_dw(lhs, dout, jnp.asarray(groups, jnp.int32))

        self.assertAllClose(actual, expected, rtol=4e-2, atol=4e-2)

    def test_compact_starts_mask_neighbors_and_accumulate(self):
        # Each repetition advances the start by one modulo eight. Four output tiles
        # per expert exercise persistent slot reuse across empty and nonempty groups.
        pattern = [1, 0, 7, 64, 0, 65, 8, 63, 0, 193, 0, 0, 8]
        sizes = np.asarray(pattern * 8, dtype=np.int32)
        starts = np.cumsum(sizes) - sizes
        allocated_rows = (int(sizes.sum()) + 255) // 256 * 256
        observed = np.arange(len(sizes)) % len(pattern) != len(pattern) - 1
        valid = np.zeros(allocated_rows, dtype=bool)
        for start, count, keep in zip(starts, sizes, observed, strict=True):
            if keep:
                valid[start : start + count] = True

        rng = np.random.default_rng(457)
        lhs = jnp.asarray(rng.normal(0, 0.2, (allocated_rows, 256)), jnp.bfloat16)
        dout = jnp.asarray(rng.normal(0, 0.2, (allocated_rows, 256)), jnp.bfloat16)
        live = jnp.asarray(valid[:, None])
        lhs_zero, dout_zero = jnp.where(live, lhs, 0), jnp.where(live, dout, 0)
        # Poison entire neighboring experts as well as the unused buffer tail.
        # Their own outputs may be NaN; the observed experts must be independent.
        lhs_nan, dout_nan = (
            jnp.where(live, lhs, jnp.nan),
            jnp.where(live, dout, jnp.nan),
        )
        groups = jnp.asarray(sizes)
        zero = sm100.ragged_dot_dw_kernel(lhs_zero, dout_zero, groups, DW_CONFIG)
        poisoned = sm100.ragged_dot_dw_kernel(lhs_nan, dout_nan, groups, DW_CONFIG)
        result = np.asarray(poisoned, dtype=np.float32)
        self.assertTrue(np.isfinite(result[observed]).all())
        self.assertArraysEqual(
            result[observed], np.asarray(zero, dtype=np.float32)[observed]
        )

        lhs_f32 = np.asarray(lhs_zero, np.float32)
        dout_f32 = np.asarray(dout_zero, np.float32)
        expected = np.stack(
            [
                lhs_f32[start : start + count].T @ dout_f32[start : start + count]
                for start, count, keep in zip(starts, sizes, observed, strict=True)
                if keep
            ]
        )
        self.assertAllClose(result[observed], expected, rtol=4e-2, atol=4e-2)
        self.assertArraysEqual(result[sizes == 0], 0, check_dtypes=False)

        initial = jnp.asarray(rng.normal(0, 0.2, zero.shape), jnp.bfloat16)
        expected_accumulated = np.asarray(poisoned + initial)
        accumulated = sm100.ragged_dot_dw_kernel(
            lhs_nan, dout_nan, groups, DW_CONFIG, accumulator=initial
        )
        self.assertArraysEqual(
            np.asarray(accumulated)[observed], expected_accumulated[observed]
        )
        self.assertArraysEqual(
            np.asarray(accumulated)[sizes == 0], np.asarray(initial)[sizes == 0]
        )

    def test_accumulates_into_aliased_array(self):
        groups = [99, 73, 0]
        lhs, dout = make_dw_inputs(groups, tail_value=np.nan)
        initial = jnp.asarray(
            np.random.default_rng(456).normal(size=(3, 128, 128)), dtype=jnp.bfloat16
        )

        plain = jax.jit(lambda x, dy: dw(x, dy, groups))(lhs, dout)

        @jax.jit
        def accumulate(x, dy, accumulator):
            return dw(x, dy, groups, accumulator=accumulator)

        fused = accumulate(lhs, dout, initial)

        self.assertArraysEqual(fused, plain + initial)
        self.assertArraysEqual(fused[2], initial[2])

        @jax.jit
        def accumulate_empty(x, dy, accumulator):
            return dw(x, dy, [0], accumulator=accumulator)

        all_empty = accumulate_empty(lhs[:128], dout[:128], initial[:1])
        self.assertArraysEqual(all_empty, initial[:1])

        twice = accumulate(lhs, dout, accumulate(lhs, dout, initial))
        self.assertArraysEqual(twice, (initial + plain) + plain)

    def test_accumulator_requires_matching_shape(self):
        lhs, dout = make_dw_inputs([128])
        accumulator = jnp.zeros((1, 128, 64), dtype=jnp.bfloat16)

        with self.assertRaisesRegex(ValueError, "dW accumulator shape"):
            dw(lhs, dout, [128], accumulator=accumulator)


class SwigluKernelTest(Sm100TestCase):
    def test_empty_group_matches_spec(self):
        rng = np.random.default_rng(31)
        lhs = jnp.asarray(rng.normal(0, 0.05, (768, 64)), jnp.bfloat16)
        gate = jnp.asarray(rng.normal(0, 0.05, (3, 64, 128)), jnp.bfloat16)
        up = jnp.asarray(rng.normal(0, 0.05, (3, 64, 128)), jnp.bfloat16)
        group_sizes = jnp.array([264, 0, 248], jnp.int32)
        interleaved = xla.interleave_gate_up(gate, up, jnp.bfloat16)

        actual = sm100.ragged_dot_swiglu_kernel(
            lhs, interleaved, group_sizes, sm100.DEFAULT_CONFIG
        )
        expected = xla.ragged_dot_swiglu(lhs, interleaved, group_sizes)

        self.assertAllClose(actual[:512], expected[:512], rtol=0.03, atol=0.01)

    @parameterized.product(tile_n=[64, 128, 192], out_dtype=[jnp.bfloat16, jnp.float32])
    def test_custom_tiles_preserve_odd_spans(self, tile_n, out_dtype):
        config = sm100.TuningConfig(128, tile_n, 64, 2, False, 1)
        rng = np.random.default_rng(58)
        lhs = jnp.asarray(rng.normal(0, 0.5, (256, 64)), jnp.bfloat16)
        lhs = lhs.at[255:].set(jnp.nan)
        rhs = jnp.asarray(rng.normal(0, 0.5, (3, 64, 2 * tile_n)), jnp.bfloat16)
        sizes = jnp.array([129, 0, 126], jnp.int32)

        actual, preact = sm100.ragged_dot_swiglu_kernel(
            lhs, rhs, sizes, config, out_dtype, True
        )
        expected = xla.ragged_dot_swiglu(lhs, rhs, sizes, out_dtype)
        expected_preact = xla.live_rows_dot(
            lhs, rhs, sizes, xla.LHS_RAGGED_DIMENSION_NUMBERS
        ).astype(out_dtype)

        self.assertAllClose(actual[:255], expected[:255], rtol=0.03, atol=0.01)
        self.assertAllClose(
            preact[:255],
            expected_preact[:255],
            rtol=0.03,
            atol=0.01,
            check_dtypes=False,
        )

    @parameterized.parameters(32, 128)
    def test_rejects_incompatible_interleave_width(self, epilogue_tile_n):
        config = sm100.TuningConfig(128, 128, 64, 2, False, 1, epilogue_tile_n)
        args = (
            jax.ShapeDtypeStruct((256, 64), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 64, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        )
        with self.assertRaisesRegex(
            ValueError, "epilogue_tile_n=64.*interleaved gate/up"
        ):
            jax.eval_shape(
                lambda x, w, sizes: sm100.ragged_dot_swiglu_kernel(x, w, sizes, config),
                *args,
            )

    @parameterized.parameters(jnp.bfloat16, jnp.float32)
    def test_backward_matches_spec_and_reuses_donated_buffer(self, dtype):
        preact_key, dy_key = jax.random.split(jax.random.key(27))
        preact = jax.random.normal(preact_key, (512, 512), dtype=dtype)
        dy = jax.random.normal(dy_key, (512, 256), dtype=dtype)
        group_sizes = jnp.array([129, 0, 254], dtype=jnp.int32)
        valid_rows = 383
        # Test backward independently of the fused forward kernel's layout.
        expected = xla.swiglu_backward(preact, dy, group_sizes)
        expected.block_until_ready()
        preact_pointer = preact.unsafe_buffer_pointer()

        actual = sm100.swiglu_backward_in_place(preact, dy, group_sizes)
        actual.block_until_ready()

        self.assertEqual(actual.shape, (512, 512))
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual.unsafe_buffer_pointer(), preact_pointer)
        self.assertTrue(preact.is_deleted())
        self.assertClose(actual[:valid_rows], expected[:valid_rows])

    def test_saved_preactivation_matches_jax_dpreactivation(self):
        lhs_key, rhs_key, dy_key = jax.random.split(jax.random.key(17), 3)
        lhs = (0.05 * jax.random.normal(lhs_key, (512, 256), dtype=jnp.float32)).astype(
            jnp.bfloat16
        )
        rhs = (
            0.05 * jax.random.normal(rhs_key, (3, 256, 512), dtype=jnp.float32)
        ).astype(jnp.bfloat16)
        group_sizes = jnp.array([128, 0, 256], dtype=jnp.int32)
        dy = jax.random.normal(dy_key, (512, 256), dtype=jnp.bfloat16)
        valid_rows = 384

        _, preact = sm100.ragged_dot_swiglu_kernel(
            lhs, rhs, group_sizes, sm100.DEFAULT_CONFIG, None, True
        )

        block = sm100.DEFAULT_CONFIG.epilogue_tile_n // 2
        preact_blocks = preact[:valid_rows].reshape(valid_rows, -1, 2, block)
        gate = preact_blocks[:, :, 0, :].astype(jnp.float32)
        up = preact_blocks[:, :, 1, :].astype(jnp.float32)
        dy_blocks = (
            dy[:valid_rows]
            .astype(preact.dtype)
            .reshape(valid_rows, -1, block)
            .astype(jnp.float32)
        )
        sigmoid_gate = jax.nn.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        dsilu_dgate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        expected = jnp.stack(
            (dy_blocks * up * dsilu_dgate, dy_blocks * silu_gate), axis=2
        )
        expected = expected.reshape(valid_rows, -1).astype(preact.dtype)

        actual = sm100.swiglu_backward_in_place(preact, dy, group_sizes)

        self.assertEqual(actual.shape, (512, 512))
        self.assertEqual(actual.dtype, jnp.bfloat16)
        self.assertArraysEqual(actual[:valid_rows], expected)


class RaggedDotKernelTest(Sm100TestCase):
    def test_collective_accumulator_reuse_waits_for_both_ctas(self):
        config = sm100.TuningConfig(128, 128, 64, 2, True, 8)
        lhs = jnp.ones((4096, 64), jnp.bfloat16)
        rhs = jnp.stack(
            [jnp.full((64, 2048), group + 1, jnp.bfloat16) for group in range(32)]
        )
        group_sizes = jnp.full((32,), 128, jnp.int32)
        expected_rows = np.repeat(np.arange(1, 33, dtype=np.float32) * 64, 128)

        run = jax.jit(lambda x, w, sizes: sm100.ragged_dot_kernel(x, w, sizes, config))
        for _ in range(3):
            actual = np.asarray(run(lhs, rhs, group_sizes), np.float32)
            self.assertArraysEqual(
                actual, np.broadcast_to(expected_rows[:, None], actual.shape)
            )

    def kernel_params(self, jaxpr):
        # warp_map also uses mpmd_map, but its WarpMesh is inside the launch.
        equations = [
            eqn
            for _, eqn in jaxpr_util.all_eqns(jaxpr)
            if eqn.primitive.name == "mpmd_map"
            and any(isinstance(mesh, plgpu.Mesh) for mesh in eqn.params["meshes"])
        ]
        self.assertLen(equations, 1)
        return equations[0].params

    @parameterized.named_parameters(("even", (64, 64)), ("one_empty", (128, 0)))
    def test_cost_uses_static_representative_rows(self, group_sizes):
        lhs = jax.ShapeDtypeStruct((256, 64), jnp.bfloat16)
        rhs = jax.ShapeDtypeStruct((2, 64, 128), jnp.bfloat16)

        traced = jax.make_jaxpr(
            lambda x, w, sizes: sm100.ragged_dot_kernel(x, w, sizes, CONFIG, 128)
        )(lhs, rhs, jnp.asarray(group_sizes, jnp.int32))
        params = self.kernel_params(traced)

        self.assertEqual(
            params["cost_estimate"],
            pl.CostEstimate(
                flops=2 * 128 * 64 * 128,
                transcendentals=0,
                bytes_accessed=(128 * 64 + 2 * 64 * 128 + 128 * 128) * 2,
            ),
        )
        self.assertEqual(
            dict(params["metadata"]),
            {"cost_model": "uniform_unpadded", "cost_estimate_num_rows": "128"},
        )

    def test_dw_cost_matches_logical_tgmm(self):
        lhs = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
        dout = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
        group_sizes = jax.ShapeDtypeStruct((2,), jnp.int32)

        traced = jax.make_jaxpr(
            lambda x, dy, sizes: sm100.ragged_dot_dw_kernel(
                x, dy, sizes, CONFIG, cost_estimate_num_rows=128
            )
        )(lhs, dout, group_sizes)

        self.assertEqual(
            self.kernel_params(traced)["cost_estimate"],
            pl.CostEstimate(
                flops=2 * 128 * 128 * 128,
                transcendentals=0,
                bytes_accessed=(128 * 128 + 128 * 128 + 2 * 128 * 128) * 2,
            ),
        )

    def test_dw_accumulator_cost_includes_read_and_add(self):
        lhs = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
        dout = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
        group_sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
        accumulator = jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16)

        def accumulate(x, dy, sizes, initial):
            return sm100.ragged_dot_dw_kernel(
                x, dy, sizes, CONFIG, accumulator=initial, cost_estimate_num_rows=128
            )

        traced = jax.make_jaxpr(accumulate)(lhs, dout, group_sizes, accumulator)
        output_elements = 2 * 128 * 128
        params = self.kernel_params(traced)
        primitive_names = [eqn.primitive.name for _, eqn in jaxpr_util.all_eqns(traced)]

        # The mutable Ref is the kernel output contract, so mpmd_map has no value
        # output to alias back to its accumulator input.
        self.assertEqual(primitive_names.count("new_ref"), 1)
        self.assertEqual(dict(params["input_output_aliases"]), {})
        self.assertEqual(
            params["cost_estimate"],
            pl.CostEstimate(
                flops=2 * 128 * 128 * 128 + output_elements,
                transcendentals=0,
                bytes_accessed=(
                    128 * 128 + 128 * 128 + output_elements + output_elements
                )
                * 2,
            ),
        )


class SwigluHiJaxTest(test_util.MaximoeTestCase):
    """Trace the shared SwiGLU primitive using the SM100 implementation."""

    def setUp(self):
        super().setUp()
        # Tracing the kernels reads the local GPU's SM count.
        self.skip_unless_gpus()

    def test_backward_reuses_outer_remat_preactivation(self):
        def layer(lhs, rhs, group_sizes, wo):
            intermediate = ragged_dot.ragged_dot_swiglu(
                lhs, rhs, group_sizes, implementation="sm100"
            )
            return intermediate @ wo

        rematted_layer = jax.remat(layer)

        def loss(lhs, rhs, group_sizes, wo):
            return rematted_layer(lhs, rhs, group_sizes, wo).astype(jnp.float32).sum()

        args = (
            jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 256, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((256, 128), jnp.bfloat16),
        )
        residuals = saved_residuals(rematted_layer, *args)
        self.assertEqual(
            [(aval.shape, source) for aval, source in residuals],
            [
                ((256, 256), "from the argument lhs"),
                ((2, 256, 512), "from the argument rhs"),
                ((2,), "from the argument group_sizes"),
                ((256, 128), "from the argument wo"),
            ],
        )

        jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 3)))(*args)
        self.assertEqual(
            test_util.kernel_names(jaxpr).count("swiglu_interleaved_bwd_in_place"), 1
        )

    def test_preserves_expert_vma_through_hijax(self):
        """The HiJAX primitive must carry expert VMA through opaque Mosaic outputs."""
        expert_mesh = AbstractMesh(
            (2,), ("expert",), axis_types=(jax.sharding.AxisType.Explicit,)
        )

        @jax.shard_map(
            mesh=expert_mesh,
            in_specs=(P(), P("expert", None, None), P("expert")),
            out_specs=P("expert", None),
            check_vma=True,
        )
        def primal(lhs, rhs, group_sizes):
            return ragged_dot.ragged_dot_swiglu(
                lhs, rhs, group_sizes, implementation="sm100"
            )

        @jax.shard_map(
            mesh=expert_mesh,
            in_specs=(
                P("expert", None),
                P("expert", None, None),
                P("expert"),
                P("expert", None),
            ),
            out_specs=(P("expert", None), P("expert", None, None)),
            check_vma=True,
        )
        def reverse(lhs, rhs, group_sizes, dout):
            _, pullback = jax.vjp(
                lambda x, w: ragged_dot.ragged_dot_swiglu(
                    x, w, group_sizes, implementation="sm100"
                ),
                lhs,
                rhs,
            )
            return pullback(dout)

        with jax.sharding.use_abstract_mesh(expert_mesh):
            jax.jit(primal).trace(
                jax.ShapeDtypeStruct(
                    (256, 256),
                    jnp.bfloat16,
                    sharding=jax.NamedSharding(expert_mesh, P()),
                ),
                jax.ShapeDtypeStruct(
                    (4, 256, 512),
                    jnp.bfloat16,
                    sharding=jax.NamedSharding(expert_mesh, P("expert", None, None)),
                ),
                jax.ShapeDtypeStruct(
                    (4,),
                    jnp.int32,
                    sharding=jax.NamedSharding(expert_mesh, P("expert")),
                ),
            )
            jax.jit(reverse).trace(
                jax.ShapeDtypeStruct(
                    (512, 256),
                    jnp.bfloat16,
                    sharding=jax.NamedSharding(expert_mesh, P("expert", None)),
                ),
                jax.ShapeDtypeStruct(
                    (4, 256, 512),
                    jnp.bfloat16,
                    sharding=jax.NamedSharding(expert_mesh, P("expert", None, None)),
                ),
                jax.ShapeDtypeStruct(
                    (4,),
                    jnp.int32,
                    sharding=jax.NamedSharding(expert_mesh, P("expert")),
                ),
                jax.ShapeDtypeStruct(
                    (512, 256),
                    jnp.bfloat16,
                    sharding=jax.NamedSharding(expert_mesh, P("expert", None)),
                ),
            )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
