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

"""Mosaic-specific properties of the routing kernels.

Numerical agreement with the specification is covered by ``moe_routing_test``.
These tests check lowering properties, the persistent grid, and varying-axis
propagation that only the Mosaic implementation has.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from maximoe import test_util
from maximoe.kernels.moe_routing import mosaic_gpu, xla

jax.config.parse_flags_with_absl()


class MosaicGpuRoutingKernelTest(test_util.MaximoeTestCase):
    """Lowering, persistent-grid, and varying-axis properties of the kernels."""

    def setUp(self):
        super().setUp()
        self.skip_unless_gpus()

    def expert_mesh(self):
        self.skip_unless_gpus(2)
        return jax.sharding.Mesh(
            np.array(jax.devices()[:2]),
            ("expert",),
            axis_types=(jax.sharding.AxisType.Explicit,),
        )

    @parameterized.parameters(150, 7168)
    def test_combine_forward_is_one_kernel_without_vjp(self, hidden):
        buffer = jax.ShapeDtypeStruct((24, hidden), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((2, 4, 2), jnp.float32)
        positions = jax.ShapeDtypeStruct((2, 4, 2), jnp.int32)

        jaxpr = jax.make_jaxpr(mosaic_gpu.combine, static_argnums=(3,))(
            buffer, weights, positions, False
        )

        self.assertEqual(test_util.kernel_names(jaxpr), ["moe_combine_mosaic_gpu_fwd"])
        # The backward is composed by ``RaggedDotCombine`` from the
        # transpose and weighting kernels, so ``combine`` itself has no VJP: JAX
        # rejects differentiating the kernel launch primitive.
        # HiPrimitive's missing JVP rule raises this specific assertion.
        with self.assertRaisesRegex(AssertionError, "^must override$"):
            jax.make_jaxpr(
                jax.grad(
                    lambda b: jnp.sum(
                        mosaic_gpu.combine(b, weights, positions, False).astype(
                            jnp.float32
                        )
                    )
                )
            )(buffer)

    def test_fp32_combine_does_not_promote_the_expert_buffer(self):
        buffer = jax.ShapeDtypeStruct((24, 150), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((2, 4, 2), jnp.float32)
        positions = jax.ShapeDtypeStruct((2, 4, 2), jnp.int32)

        stablehlo = mosaic_gpu.combine.lower(buffer, weights, positions, True).as_text()

        self.assertNotIn("tensor<24x150xf32>", stablehlo)
        self.assertNotIn("tensor<24x256xf32>", stablehlo)

    def test_persistent_grid_wraps_live_prefixes(self):
        b, k, m = 3, 2, 128
        block_s = mosaic_gpu.COMBINE_FWD_TOKENS_PER_TASK
        num_programs = mosaic_gpu.persistent_grid(mosaic_gpu.COMBINE_FWD_CTAS_PER_SM)[0]
        s = num_programs * block_s
        third_group_size = (num_programs - 1) * block_s
        group_sizes = jnp.array([block_s + 1, 0, third_group_size], dtype=jnp.int32)
        extent = (s + 1) * k
        positions = jnp.full((b, s, k), extent, dtype=jnp.int32)
        positions = positions.at[0, : block_s + 1].set(
            jnp.arange((block_s + 1) * k, dtype=jnp.int32).reshape(block_s + 1, k)
        )
        positions = positions.at[2, :third_group_size].set(
            jnp.arange((block_s + 1) * k, extent, dtype=jnp.int32).reshape(
                third_group_size, k
            )
        )
        buffer = jax.random.normal(jax.random.key(16), (extent, m), dtype=jnp.bfloat16)
        weights = jax.random.normal(
            jax.random.key(15), positions.shape, dtype=jnp.bfloat16
        )

        output = mosaic_gpu.combine(buffer, weights, positions, False, group_sizes)
        expected = xla.combine(buffer, weights, positions, False, group_sizes)

        live = np.arange(s)[None, :] < np.asarray(group_sizes)[:, None]
        self.assertArraysEqual(np.asarray(output)[live], np.asarray(expected)[live])

    def test_combine_transpose_lowering_has_no_early_allocation(self):
        grad_out = jax.ShapeDtypeStruct((2, 4, 150), jnp.bfloat16)
        positions = jax.ShapeDtypeStruct((2, 4, 2), jnp.int32)

        stablehlo = mosaic_gpu.combine_transpose.lower(
            grad_out, positions, 24, 13
        ).as_text()

        self.assertNotIn("@AllocateBuffer", stablehlo)
        self.assertNotIn("tf.aliasing_output", stablehlo)

    @parameterized.parameters("invariant", "reduced")
    def test_combine_transpose_unions_expert_vma_from_positions(self, grad_state):
        mesh = self.expert_mesh()
        extent, m = 4, 32
        grad_out = jnp.arange(2 * m, dtype=jnp.bfloat16).reshape(1, 2, m)
        positions = jnp.array([[[1], [extent]], [[2], [extent]]], dtype=jnp.int32)
        grad_out = jax.device_put(
            grad_out, jax.sharding.NamedSharding(mesh, jax.P(None, None, None))
        )
        positions = jax.device_put(
            positions, jax.sharding.NamedSharding(mesh, jax.P("expert", None, None))
        )

        def transpose_grad(grad, pos):
            if grad_state == "reduced":
                grad = jax.lax.pcast(grad, "expert", to="reduced")
            return mosaic_gpu.combine_transpose(grad, pos, extent)

        expert_cotangent = jax.shard_map(
            transpose_grad,
            mesh=mesh,
            in_specs=(jax.P(None, None, None), jax.P("expert", None, None)),
            out_specs=jax.P("expert", None),
        )(grad_out, positions)

        self.assertArraysEqual(np.asarray(expert_cotangent)[1], grad_out[0, 0])
        self.assertArraysEqual(np.asarray(expert_cotangent)[6], grad_out[0, 0])

    def test_weight_expert_rows_unions_expert_vma_from_routes(self):
        mesh = self.expert_mesh()
        extent, m = 4, 32
        buffer = jnp.arange(extent * m, dtype=jnp.bfloat16).reshape(extent, m)
        weights = jnp.array([[[0.5], [0.0]], [[1.25], [0.0]]], dtype=jnp.bfloat16)
        positions = jnp.array([[[1], [extent]], [[2], [extent]]], dtype=jnp.int32)
        expected_first = np.asarray(buffer[1] * weights[0, 0, 0])
        expected_second = np.asarray(buffer[2] * weights[1, 0, 0])

        with jax.set_mesh(mesh):
            buffer = jax.device_put(buffer, jax.P(None, None))
            weights, positions = jax.device_put(
                (weights, positions), jax.P("expert", None, None)
            )
            weighted = jax.shard_map(
                mosaic_gpu.weight_expert_rows,
                mesh=mesh,
                in_specs=(
                    jax.P(None, None),
                    jax.P("expert", None, None),
                    jax.P("expert", None, None),
                ),
                out_specs=jax.P("expert", None),
            )(buffer, weights, positions)

        self.assertArraysEqual(np.asarray(weighted)[1], expected_first)
        self.assertArraysEqual(np.asarray(weighted)[6], expected_second)

    def test_weight_rows_ref_alias_has_no_copy(self):
        buffer = jax.ShapeDtypeStruct((24, 1024), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((2, 4, 2), jnp.bfloat16)
        positions = jax.ShapeDtypeStruct((2, 4, 2), jnp.int32)

        stablehlo = mosaic_gpu.weight_expert_rows.lower(
            buffer, weights, positions
        ).as_text()

        self.assertIn("tf.aliasing_output = 0", stablehlo)
        self.assertNotIn("@AllocateBuffer", stablehlo)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
