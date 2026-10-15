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

"""Ragged-dot kernels against the ``xla`` specification.

``maximoe.kernels.ragged_dot.xla`` defines every operation. ``RaggedDotSpecTest``
checks the precision contract and that the specification's hand-written
gradient pieces agree with JAX autodiff of its forward functions.
``MosaicRaggedDotTest`` checks that the Mosaic implementation for the local
architecture computes the same values and gradients on the rows the
specification defines.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from maximoe import test_util
from maximoe.kernels import gpu, ragged_dot
from maximoe.kernels.ragged_dot import swiglu, xla

jax.config.parse_flags_with_absl()

ROWS, CONTRACT, OUT, HIDDEN = 512, 256, 256, 128
NUM_GROUPS = 4


@dataclasses.dataclass(frozen=True)
class Groups:
    """Compact expert rows followed by an unused allocation tail."""

    name: str
    sizes: tuple[int, ...]

    @property
    def group_sizes(self) -> jax.Array:
        return jnp.asarray(self.sizes, jnp.int32)

    def live_rows(self) -> np.ndarray:
        return np.arange(ROWS) < sum(self.sizes)

    def poison_tail(self, values: jax.Array) -> jax.Array:
        """Poison unused rows so contraction tails must stay inside each expert."""
        live = jnp.asarray(self.live_rows())
        return jnp.where(live[:, None], values, jnp.nan)


# Nonaligned group boundaries and the unused allocation tail are independent.
PARTIAL = Groups("partial", (130, 0, 197, 96))
FULL = Groups("full", (128, 256, 0, 128))
UNALIGNED = Groups("unaligned", (5, 0, 100, 250))
GROUPS = (PARTIAL, FULL, UNALIGNED)
GROUP_CASES = [(groups.name, groups) for groups in GROUPS]
OUT_DTYPES = [None, jnp.float32]


def make_inputs(seed, *, out=OUT):
    lhs_key, rhs_key, dout_key = jax.random.split(jax.random.key(seed), 3)
    lhs = jax.random.normal(lhs_key, (ROWS, CONTRACT), jnp.bfloat16)
    rhs = jax.random.normal(rhs_key, (NUM_GROUPS, CONTRACT, out), jnp.float32)
    rhs = (rhs / jnp.sqrt(CONTRACT)).astype(jnp.bfloat16)
    dout = jax.random.normal(dout_key, (ROWS, out), jnp.bfloat16)
    return lhs, rhs, dout


class RaggedDotSpecTest(test_util.MaximoeTestCase):
    """Precision contract and self-consistency of the specification."""

    @parameterized.named_parameters(("bf16", None), ("f32", jnp.float32))
    def test_swiglu_rounds_projection_once_at_output_dtype(self, out_dtype):
        lhs_key, rhs_key, dout_key = jax.random.split(jax.random.key(14), 3)
        lhs = jax.random.normal(lhs_key, (8, 4), jnp.bfloat16)
        rhs = jax.random.normal(rhs_key, (2, 4, 64), jnp.bfloat16)
        group_sizes = jnp.array([3, 4], jnp.int32)
        live = jnp.arange(lhs.shape[0]) < 7
        output_dtype = jnp.dtype(lhs.dtype if out_dtype is None else out_dtype)

        def projection(lhs, rhs):
            preactivations = jax.lax.ragged_dot_general(
                jnp.where(live[:, None], lhs, 0),
                rhs,
                group_sizes,
                xla.LHS_RAGGED_DIMENSION_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            return jnp.where(live[:, None], preactivations, 0)

        def explicit_precision_boundary(lhs, rhs):
            preactivations = projection(lhs, rhs).astype(output_dtype)
            return xla.swiglu_interleaved(preactivations, output_dtype)

        def specification(lhs, rhs):
            return xla.ragged_dot_swiglu(lhs, rhs, group_sizes, out_dtype)

        actual, actual_vjp = jax.vjp(specification, lhs, rhs)
        expected, expected_vjp = jax.vjp(explicit_precision_boundary, lhs, rhs)
        dout = jax.random.normal(dout_key, actual.shape, output_dtype)

        self.assertArraysEqual(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_vjp(dout), expected_vjp(dout), strict=True
        ):
            self.assertArraysEqual(actual_grad, expected_grad)

        if output_dtype == lhs.dtype:
            unrounded = xla.swiglu_interleaved(projection(lhs, rhs), output_dtype)
            self.assertFalse(np.array_equal(np.asarray(actual), np.asarray(unrounded)))

    def test_gradient_pieces_match_autodiff(self):
        groups = PARTIAL
        lhs, rhs, dout = make_inputs(8)
        lhs, rhs, dout = (x.astype(jnp.float32) for x in (lhs, rhs, dout))
        lhs = groups.poison_tail(lhs)
        dout = groups.poison_tail(dout)

        out, vjp = jax.vjp(
            lambda x, w: xla.ragged_dot(x, w, groups.group_sizes), lhs, rhs
        )
        dlhs, drhs = vjp(dout)

        live = groups.live_rows()
        self.assertTrue(np.isfinite(np.asarray(out)).all())
        self.assertTrue(np.isfinite(np.asarray(dlhs)).all())
        self.assertArraysEqual(np.asarray(out)[~live], 0, check_dtypes=False)
        self.assertArraysEqual(np.asarray(dlhs)[~live], 0, check_dtypes=False)
        expected_dlhs = xla.ragged_dot_dx(dout, rhs, groups.group_sizes)
        self.assertClose(dlhs, expected_dlhs, live, rtol=1e-5, atol_scale=1e-5)
        expected_drhs = xla.ragged_dot_dw(lhs, dout, groups.group_sizes)
        self.assertClose(drhs, expected_drhs, rtol=1e-5, atol_scale=1e-5)

    def test_dw_accumulates_after_rounding(self):
        groups = PARTIAL
        lhs, _, dout = make_inputs(9)
        accumulator = jax.random.normal(
            jax.random.key(10), (NUM_GROUPS, CONTRACT, OUT), jnp.bfloat16
        )

        plain = xla.ragged_dot_dw(lhs, dout, groups.group_sizes)
        fused = xla.ragged_dot_dw(
            lhs, dout, groups.group_sizes, accumulator=accumulator
        )

        self.assertArraysEqual(fused, accumulator + plain)
        with self.assertRaisesRegex(ValueError, "accumulator shape"):
            xla.ragged_dot_dw(
                lhs, dout, groups.group_sizes, accumulator=accumulator[:1]
            )
        with self.assertRaisesRegex(ValueError, "accumulator dtype"):
            xla.ragged_dot_dw(
                lhs,
                dout,
                groups.group_sizes,
                accumulator=accumulator.astype(jnp.float32),
            )

    def test_swiglu_matches_unfused_gate_and_up(self):
        groups = PARTIAL
        lhs, _, _ = make_inputs(11)
        gate_key, up_key = jax.random.split(jax.random.key(12))
        gate = jax.random.normal(gate_key, (NUM_GROUPS, CONTRACT, HIDDEN), jnp.bfloat16)
        up = jax.random.normal(up_key, (NUM_GROUPS, CONTRACT, HIDDEN), jnp.bfloat16)
        interleaved = xla.interleave_gate_up(gate, up, jnp.bfloat16)

        recovered_gate, recovered_up = xla.deinterleave_gate_up(interleaved)
        self.assertArraysEqual(recovered_gate, gate)
        self.assertArraysEqual(recovered_up, up)

        actual = xla.ragged_dot_swiglu(
            lhs, interleaved, groups.group_sizes, jnp.float32
        )
        gate_out = xla.ragged_dot(lhs, gate, groups.group_sizes)
        up_out = xla.ragged_dot(lhs, up, groups.group_sizes)
        expected = jax.nn.silu(gate_out.astype(jnp.float32)) * up_out.astype(
            jnp.float32
        )

        # The unfused reference rounds gate and up to bf16 before the activation.
        live = groups.live_rows()
        self.assertClose(actual, expected, live)
        self.assertArraysEqual(np.asarray(actual)[~live], 0, check_dtypes=False)


class MosaicRaggedDotTest(test_util.MaximoeTestCase):
    """The Mosaic kernels for the local architecture against the specification."""

    def setUp(self):
        super().setUp()
        self.skip_unless_gpus()
        self.kernels = ragged_dot.get_implementation(gpu.architecture())

    @parameterized.named_parameters(*GROUP_CASES)
    def test_ragged_dot_matches_spec(self, groups):
        lhs, rhs, _ = make_inputs(1)
        lhs = groups.poison_tail(lhs)
        expected = xla.ragged_dot(lhs, rhs, groups.group_sizes)
        actual = self.kernels.ragged_dot(
            lhs, rhs, groups.group_sizes, sum(groups.sizes)
        )

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertClose(actual, expected, groups.live_rows())

    @parameterized.named_parameters(*GROUP_CASES)
    def test_ragged_dot_vjp_matches_spec(self, groups):
        lhs, rhs, dout = make_inputs(2)
        lhs = groups.poison_tail(lhs)
        dout = groups.poison_tail(dout)

        def run(implementation):
            out, vjp = jax.vjp(
                lambda x, w: ragged_dot.ragged_dot(
                    x, w, groups.group_sizes, implementation=implementation
                ),
                lhs,
                rhs,
            )
            return out, *vjp(dout)

        actual_out, actual_dlhs, actual_drhs = run(gpu.architecture().value)
        expected_out, expected_dlhs, expected_drhs = run("xla")

        live = groups.live_rows()
        self.assertClose(actual_out, expected_out, live)
        self.assertClose(actual_dlhs, expected_dlhs, live)
        self.assertClose(actual_drhs, expected_drhs)

    @parameterized.named_parameters(*GROUP_CASES)
    def test_ragged_dot_dx_matches_spec(self, groups):
        _, rhs, dout = make_inputs(3)
        dout = groups.poison_tail(dout)
        expected = xla.ragged_dot_dx(dout, rhs, groups.group_sizes)
        actual = self.kernels.ragged_dot_dx(
            dout, rhs, groups.group_sizes, cost_estimate_num_rows=sum(groups.sizes)
        )

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertClose(actual, expected, groups.live_rows())

    @parameterized.named_parameters(*GROUP_CASES)
    def test_ragged_dot_dw_matches_spec(self, groups):
        lhs, _, dout = make_inputs(4)
        lhs = groups.poison_tail(lhs)
        dout = groups.poison_tail(dout)
        expected = xla.ragged_dot_dw(lhs, dout, groups.group_sizes)
        actual = self.kernels.ragged_dot_dw(
            lhs, dout, groups.group_sizes, cost_estimate_num_rows=sum(groups.sizes)
        )

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertClose(actual, expected)

    def test_ragged_dot_dw_reuses_pipeline_across_uneven_groups(self):
        # Enough tiles to reuse each CTA's pipeline across short, empty, and long
        # groups. The old delayed-release configuration fails on this pattern.
        group_sizes = jnp.asarray([64, 8, 0, 320] * 8, jnp.int32)
        live_rows = jnp.arange(4096) < jnp.sum(group_sizes)
        lhs = jnp.broadcast_to(
            jnp.where(live_rows, jnp.bfloat16(1), jnp.bfloat16(jnp.nan))[:, None],
            (4096, 512),
        )
        dout = jnp.broadcast_to(
            jnp.where(live_rows, jnp.bfloat16(2), jnp.bfloat16(jnp.nan))[:, None],
            (4096, 1024),
        )
        actual = self.kernels.ragged_dot_dw(lhs, dout, group_sizes)
        expected = np.asarray((2 * group_sizes).astype(jnp.bfloat16), np.float32)[
            :, None, None
        ]
        self.assertArraysEqual(
            np.asarray(actual, np.float32), np.broadcast_to(expected, actual.shape)
        )

    def test_ragged_dot_dw_accumulator_matches_spec(self):
        groups = PARTIAL
        lhs, _, dout = make_inputs(5)
        accumulator = jax.random.normal(
            jax.random.key(6), (NUM_GROUPS, CONTRACT, OUT), jnp.bfloat16
        )
        expected = xla.ragged_dot_dw(
            lhs, dout, groups.group_sizes, accumulator=accumulator
        )
        actual = self.kernels.ragged_dot_dw(
            lhs, dout, groups.group_sizes, accumulator=accumulator
        )

        self.assertClose(actual, expected)
        # The empty group leaves its accumulator slice untouched in both.
        self.assertArraysEqual(actual[1], accumulator[1])

    @parameterized.product(groups=GROUPS, out_dtype=OUT_DTYPES)
    def test_ragged_dot_swiglu_vjp_matches_spec(self, groups, out_dtype):
        lhs, rhs, dout = make_inputs(7, out=2 * HIDDEN)
        lhs = groups.poison_tail(lhs)
        dout = groups.poison_tail(dout[:, :HIDDEN])
        if out_dtype is not None:
            dout = dout.astype(out_dtype)

        def run(implementation):
            out, vjp = jax.vjp(
                lambda x, w: ragged_dot.ragged_dot_swiglu(
                    x, w, groups.group_sizes, out_dtype, implementation=implementation
                ),
                lhs,
                rhs,
            )
            return out, *vjp(dout)

        actual_out, actual_dlhs, actual_drhs = run(gpu.architecture().value)
        expected_out, expected_dlhs, expected_drhs = run("xla")

        live = groups.live_rows()
        self.assertEqual(actual_out.dtype, expected_out.dtype)
        self.assertEqual(actual_out.shape, (ROWS, HIDDEN))
        self.assertClose(actual_out, expected_out, live)
        self.assertClose(actual_dlhs, expected_dlhs, live)
        self.assertClose(actual_drhs, expected_drhs)

    @parameterized.product(dtype=[jnp.bfloat16, jnp.float32])
    def test_live_swiglu_runtime_bound_and_gradients(self, dtype):
        preact = jax.random.normal(jax.random.key(21), (512, 256), dtype)
        dy = jax.random.normal(jax.random.key(22), (512, 128), dtype)

        @jax.jit
        def run(preact, dy, num_rows):
            out = swiglu.forward(preact, num_rows, dtype)
            return out, swiglu.backward(preact, num_rows, dy)

        expected_out, pullback = jax.vjp(
            lambda x: xla.swiglu_interleaved(x, dtype), preact
        )
        expected_grad = pullback(dy)[0]
        # Reuse one executable across empty, partial-tile, and full live prefixes.
        executable = run.lower(preact, dy, jnp.int32(0)).compile()
        for rows in (0, 1, 7, 8, 423, 512):
            live = jnp.arange(512) < rows
            poisoned = jnp.where(live[:, None], preact, jnp.nan)
            poisoned_dy = jnp.where(live[:, None], dy, jnp.nan)
            out, grad = executable(poisoned, poisoned_dy, jnp.int32(rows))
            tolerance = 1e-2 if dtype == jnp.bfloat16 else 2e-6
            for actual, expected in ((out, expected_out), (grad, expected_grad)):
                self.assertAllClose(
                    np.asarray(actual[:rows], np.float32),
                    np.asarray(expected[:rows], np.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
