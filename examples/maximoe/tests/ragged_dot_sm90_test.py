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

"""Hopper-specific configuration and dW Ref accumulation of the ragged-dot kernels.

Agreement with the specification is covered by ``ragged_dot_test``.
"""

from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from maximoe import test_util
from maximoe.kernels import gpu, ragged_dot
from maximoe.kernels.ragged_dot import sm90
from maximoe.kernels.ragged_dot import xla as ragged_dot_xla

jax.config.parse_flags_with_absl()


class Sm90RaggedDotTest(test_util.MaximoeTestCase):
    @parameterized.named_parameters(("reserve_one_sm", 1), ("use_one_sm", None))
    def test_odd_sm_margin_supports_forward_and_input_gradient(self, margin):
        self.skip_unless_gpus(architecture=gpu.GpuArchitecture.SM90)
        if margin is None:
            margin = sm90.device_sm_count() - 1
        self.enterContext(
            mock.patch.dict("os.environ", {"RAGGED_DOT_SM_MARGIN": str(margin)})
        )
        sm90.ragged_dot_kernel.clear_cache()
        lhs = jnp.ones((256, 256), jnp.bfloat16)
        rhs = jnp.full((2, 256, 256), 0.125, jnp.bfloat16)
        sizes = jnp.array([127, 129], jnp.int32)
        for operation in ("ragged_dot", "ragged_dot_dx"):
            actual = getattr(sm90, operation)(lhs, rhs, sizes)
            expected = getattr(ragged_dot_xla, operation)(lhs, rhs, sizes)
            self.assertArraysEqual(actual, expected)

    @parameterized.named_parameters(
        ("bf16", None, jnp.bfloat16, sm90.DEFAULT_CONFIG),
        ("f32", jnp.float32, jnp.float32, sm90.FP32_SWIGLU_CONFIG),
    )
    def test_swiglu_projection_uses_output_precision_in_forward_and_backward(
        self, out_dtype, expected_dtype, expected_config
    ):
        self.enterContext(mock.patch.object(sm90, "device_sm_count", return_value=132))
        args = (
            jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 256, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        )

        def loss(lhs, rhs, group_sizes):
            output = ragged_dot.ragged_dot_swiglu(
                lhs, rhs, group_sizes, out_dtype=out_dtype, implementation="sm90"
            )
            return output.sum()

        jaxpr = jax.make_jaxpr(jax.grad(loss, (0, 1)))(*args)
        forward = next(
            eqn
            for eqn in test_util.kernel_launches(jaxpr)
            if eqn.params["name"].startswith("ragged_dot_sm90_fwd")
        )

        self.assertEqual(forward.params["out_avals"][0].dtype, expected_dtype)
        self.assertIn(f"_s{expected_config.num_stages}_", forward.params["name"])

    @parameterized.product(dtype=[jnp.bfloat16, jnp.float16])
    def test_dw_ref_accumulates_multiple_passes(self, dtype):
        """dW passes accumulate into one caller-owned Ref; empty groups keep it."""
        self.skip_unless_gpus(architecture=gpu.GpuArchitecture.SM90)
        rows, contract, out, num_groups = 512, 256, 256, 4
        lhs_key, _, dout_key = jax.random.split(jax.random.key(19), 3)
        lhs = jax.random.normal(lhs_key, (rows, contract), jnp.bfloat16).astype(dtype)
        dout = jax.random.normal(dout_key, (rows, out), jnp.bfloat16).astype(dtype)
        initial = jax.random.normal(
            jax.random.key(20), (num_groups, contract, out), dtype
        )
        sizes = jnp.asarray([[17, 0, 127, 128], [0, 0, 32, 0], [0, 0, 0, 0]])

        @jax.jit
        def accumulate(lhs, dout, initial, sizes):
            ref = jax.new_ref(initial)

            def body(i, _):
                live = jnp.arange(rows) < jnp.sum(sizes[i])
                x = jnp.where(live[:, None], lhs, jnp.nan)
                dy = jnp.where(live[:, None], dout, jnp.nan)
                result = sm90.ragged_dot_dw(x, dy, sizes[i], accumulator=ref)
                self.assertIsNone(result)

            jax.lax.fori_loop(0, sizes.shape[0], body, None)
            return jax.freeze(ref)

        expected = initial
        for group_sizes in sizes:
            expected = expected + sm90.ragged_dot_dw(lhs, dout, group_sizes)
        actual = accumulate(lhs, dout, initial, sizes)
        self.assertArraysEqual(actual, expected)
        self.assertArraysEqual(actual[1], initial[1])

    def test_cluster_config_falls_back_for_unpaired_n_tile(self):
        paired = sm90.fit_cluster_to_output(sm90.DEFAULT_CONFIG, 256)
        unpaired = sm90.fit_cluster_to_output(sm90.DEFAULT_CONFIG, 128)

        self.assertEqual(paired, sm90.DEFAULT_CONFIG)
        self.assertEqual(paired.cluster_n, 2)
        self.assertEqual(unpaired.cluster_n, 1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
