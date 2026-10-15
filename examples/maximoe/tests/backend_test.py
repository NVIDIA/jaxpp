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

"""Implementation selection and shared HiJAX gradient accumulation."""

import inspect
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import jaxpr_util
from jax._src import test_util as jtu

from maximoe import test_util
from maximoe.kernels import backend, gpu, moe_routing, ragged_dot
from maximoe.kernels.moe_routing import mosaic_gpu
from maximoe.kernels.moe_routing import xla as moe_routing_xla
from maximoe.kernels.ragged_dot import sm90, sm100
from maximoe.kernels.ragged_dot import xla as ragged_dot_xla

jax.config.parse_flags_with_absl()


def primitive_names(jaxpr):
    """Names of every primitive reachable from ``jaxpr``."""
    return {eqn.primitive.name for _, eqn in jaxpr_util.all_eqns(jaxpr)}


def xla_operation(operation):
    """The shared primitive and the standalone spec for a weight-consuming op."""
    sizes = jnp.array([2, 2], jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32).reshape(1, 2, 2)
    weights = jnp.full(positions.shape, 0.5, jnp.float32)
    if operation == "down_combine":

        def primitive(x, w):
            return ragged_dot.ragged_dot_combine(
                x, w, sizes, weights, positions, None, False, 4, implementation="xla"
            )

        def spec(x, w):
            return moe_routing_xla.combine(
                ragged_dot_xla.ragged_dot(x, w, sizes), weights, positions, False
            )

        return primitive, spec

    name = "ragged_dot" if operation == "dot" else "ragged_dot_swiglu"

    def primitive(x, w):
        return getattr(ragged_dot, name)(x, w, sizes, implementation="xla")

    def spec(x, w):
        return getattr(ragged_dot_xla, name)(x, w, sizes)

    return primitive, spec


class BackendTest(test_util.MaximoeTestCase):
    def data_mesh(self):
        if jax.device_count() < 2:
            self.skipTest("requires two devices")
        return jax.sharding.Mesh(
            np.array(jax.devices()[:2]),
            ("data",),
            axis_types=(jax.sharding.AxisType.Explicit,),
        )

    def test_every_gpu_architecture_has_an_implementation(self):
        for architecture in gpu.GpuArchitecture:
            implementation = backend.Implementation(architecture.value)
            self.assertIsNot(implementation, backend.Implementation.XLA)

    def test_explicit_selection_overrides_platform_default(self):
        self.enterContext(mock.patch.object(jax, "default_backend", lambda: "gpu"))
        self.enterContext(
            mock.patch.object(
                gpu, "architecture", lambda device=None: gpu.GpuArchitecture.SM100
            )
        )
        self.assertIs(backend.resolve(), backend.Implementation.SM100)
        self.assertIs(backend.resolve("xla"), backend.Implementation.XLA)
        self.assertIs(backend.resolve("sm90"), backend.Implementation.SM90)
        self.assertIs(backend.resolve(), backend.Implementation.SM100)

    @parameterized.product(platform=["cpu", "tpu"])
    def test_selection_defaults_to_xla_on_non_gpu(self, platform):
        self.enterContext(mock.patch.object(jax, "default_backend", lambda: platform))

        def unexpected_architecture_query(device=None):
            del device
            raise AssertionError("non-GPU selection must not query a GPU architecture")

        self.enterContext(
            mock.patch.object(gpu, "architecture", unexpected_architecture_query)
        )
        self.assertIs(backend.resolve(), backend.Implementation.XLA)

    def test_resolve_rejects_unknown_implementation(self):
        with self.assertRaises(ValueError):
            backend.resolve("tpu")

    def test_implementation_modules_are_stable_objects(self):
        self.enterContext(mock.patch.object(jax, "default_backend", lambda: "gpu"))
        self.enterContext(
            mock.patch.object(
                gpu, "architecture", lambda device=None: gpu.GpuArchitecture.SM90
            )
        )
        self.assertIs(ragged_dot.get_implementation(), sm90)
        self.assertIs(moe_routing.get_implementation(), mosaic_gpu)

        self.assertIs(ragged_dot.get_implementation("sm100"), sm100)
        self.assertIs(ragged_dot.get_implementation(gpu.GpuArchitecture.SM100), sm100)
        self.assertIs(
            ragged_dot.get_implementation(backend.Implementation.XLA), ragged_dot_xla
        )
        self.assertIs(moe_routing.get_implementation("sm100"), mosaic_gpu)
        self.assertIs(moe_routing.get_implementation("xla"), moe_routing_xla)

    @parameterized.product(
        operation=[
            "ragged_dot",
            "ragged_dot_dx",
            "ragged_dot_dw",
            "ragged_dot_swiglu",
            "swiglu_backward",
        ]
    )
    def test_ragged_dot_implementations_share_public_parameters(self, operation):
        expected = inspect.signature(getattr(ragged_dot_xla, operation)).parameters
        for implementation in (sm90, sm100):
            parameters = inspect.signature(
                getattr(implementation, operation)
            ).parameters
            self.assertEqual(
                tuple(parameters),
                tuple(expected),
                f"{implementation.__name__}.{operation}",
            )

    def test_xla_selection_traces_every_operation_without_kernels(self):
        lhs = jax.ShapeDtypeStruct((16, 64), jnp.bfloat16)
        rhs = jax.ShapeDtypeStruct((2, 64, 64), jnp.bfloat16)
        group_sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
        buffer = jax.ShapeDtypeStruct((16, 64), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((1, 4, 2), jnp.float32)
        positions = jax.ShapeDtypeStruct((1, 4, 2), jnp.int32)

        def routed(lhs, rhs, group_sizes, buffer, weights, positions):
            expert_output = ragged_dot.ragged_dot(
                lhs, rhs, group_sizes, implementation="xla"
            )
            dw = ragged_dot.ragged_dot_dw(
                lhs, expert_output, group_sizes, implementation="xla"
            )
            dx = ragged_dot.ragged_dot_dx(
                expert_output, rhs, group_sizes, implementation="xla"
            )
            swiglu = ragged_dot.ragged_dot_swiglu(
                lhs, rhs, group_sizes, implementation="xla"
            )
            combined = moe_routing.combine(
                buffer, weights, positions, False, implementation="xla"
            )
            compact = moe_routing.combine(
                buffer, weights, positions, False, group_sizes[:1], implementation="xla"
            )
            transposed = moe_routing.combine_transpose(
                combined, positions, 16, implementation="xla"
            )
            weighted = moe_routing.weight_expert_rows(
                buffer, weights, positions, implementation="xla"
            )
            permuted = moe_routing.permute(
                lhs[:4], positions[0], 16, implementation="xla"
            )
            gathered = moe_routing.gather_sum(
                permuted, positions[0], implementation="xla"
            )
            return dw, dx, swiglu, compact, transposed, weighted, gathered

        jaxpr = jax.jit(routed).trace(lhs, rhs, group_sizes, buffer, weights, positions)

        self.assertIn("ragged_dot_general", primitive_names(jaxpr.lojax.jaxpr))
        self.assertEmpty(test_util.kernel_launches(jaxpr.lojax.jaxpr))

    def test_explicit_implementations_coexist_in_jit_and_backward(self):
        self.skip_unless_gpus()
        mosaic = backend.resolve()
        lhs = jnp.ones((256, 256), jnp.bfloat16)
        rhs = jnp.full((2, 256, 256), 0.125, jnp.bfloat16)
        sizes = jnp.array([128, 128], jnp.int32)

        @jax.jit
        def project(x, w):
            return ragged_dot.ragged_dot(x, w, sizes, implementation="xla") + (
                ragged_dot.ragged_dot(x, w, sizes, implementation=mosaic)
            )

        names = primitive_names(project.trace(lhs, rhs).lojax.jaxpr)
        self.assertIn("ragged_dot_general", names)
        self.assertIn("mpmd_map", names)
        output, pullback = jax.vjp(project, lhs, rhs)
        expected, expected_pullback = jax.vjp(
            lambda x, w: 2 * ragged_dot_xla.ragged_dot(x, w, sizes), lhs, rhs
        )
        actual_grads = jax.jit(pullback)(jnp.ones_like(output))
        expected_grads = expected_pullback(jnp.ones_like(expected))
        for actual, reference in zip(
            (output, *actual_grads), (expected, *expected_grads), strict=True
        ):
            self.assertArraysEqual(actual, reference)

    @parameterized.product(operation=["dot", "swiglu", "down_combine"])
    def test_xla_primitives_vary_with_reduced_weights(self, operation):
        """Reduced weights make the output varying, like the spec on cast weights.

        JAX does not compare a primitive's declared output type with its expansion,
        so the promotion in ``ops.promote_varying`` needs an explicit check. Keep
        the activations invariant so only the weights carry the data axis.
        """
        mesh = self.data_mesh()
        primitive, spec = xla_operation(operation)
        observed = {}
        out_spec = (
            jax.P("data", None, None)
            if operation == "down_combine"
            else jax.P("data", None)
        )

        @jax.shard_map(
            mesh=mesh,
            in_specs=(jax.P(None, None), jax.P(reduced={"data"})),
            out_specs=out_spec,
        )
        def forward(x, w):
            out = primitive(x, w)
            reference = spec(x, jax.lax.pcast(w, ("data",), to="varying"))
            observed["primitive"] = jax.typeof(out).manual_axis_type
            observed["spec"] = jax.typeof(reference).manual_axis_type
            return out

        x = jax.ShapeDtypeStruct(
            (8, 4),
            jnp.float32,
            sharding=jax.sharding.NamedSharding(mesh, jax.P(None, None)),
        )
        w = jax.ShapeDtypeStruct(
            (2, 4, 64),
            jnp.float32,
            sharding=jax.sharding.NamedSharding(mesh, jax.P(reduced={"data"})),
        )
        with jax.set_mesh(mesh):
            jax.jit(forward).trace(x, w)

        primitive_type, spec_type = observed["primitive"], observed["spec"]
        self.assertEqual(primitive_type.varying, {"data"})
        self.assertFalse(primitive_type.reduced)
        self.assertFalse(primitive_type.unreduced)
        self.assertEqual(
            (primitive_type.varying, primitive_type.reduced, primitive_type.unreduced),
            (spec_type.varying, spec_type.reduced, spec_type.unreduced),
        )

    @parameterized.product(implementation=["xla", "sm90", "sm100"])
    def test_shared_swiglu_backward_accepts_caller_owned_weight_ref(
        self, implementation
    ):
        self.skip_unless_gpus()

        @jax.jit
        def accumulate(lhs, rhs, sizes, seed):
            ref = jax.new_ref(seed)
            output, pullback = jax.vjp(
                lambda w: ragged_dot.ragged_dot_swiglu(
                    lhs, w, sizes, implementation=implementation
                ),
                rhs,
            )
            pullback.with_refs(ref)(jnp.ones_like(output))
            return jax.freeze(ref)

        lhs = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
        rhs = jax.ShapeDtypeStruct((2, 256, 512), jnp.bfloat16)
        sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
        traced = accumulate.trace(lhs, rhs, sizes, rhs)
        # Expanding this used to fail on SM100, which wrapped the supplied Ref in
        # another new_ref call instead of passing it to the dW kernel.
        self.assertEqual(traced.lojax.jaxpr.outvars[0].aval.shape, rhs.shape)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
