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

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import effects as jax_effects
from jax._src import test_util as jtu
from jax.interpreters import partial_eval as pe
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpp import array_ops, env_vars
from jaxpp.core import (
    add_deletes,
    finalize_lifetimes,
    infer_donation,
    new_primitive_eqn,
    to_local_jaxprs,
)
from jaxpp.experimental import mpmd
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import (
    communication_effect,
    delete_p,
    local_slice_p,
    local_stack_p,
    recv_done_p,
    reuse_fence_p,
    slice_p,
    stack_p,
    task_p,
    transfer_done_p,
    transfer_p,
    transfer_start_p,
    zeros_p,
)
from tests.fake_devices import make_mesh


def identity(x):
    return x


def duplicate(x):
    return x, x


def add_one(x):
    return x + jnp.asarray(1, dtype=x.dtype)


def multiply_two(x):
    return x * jnp.asarray(2, dtype=x.dtype)


def add_pair(x, y):
    return x + y


class TestExperimentalSendRecvTrace(jtu.JaxTestCase):
    def test_axis_index(self):
        assert array_ops.axis_index(P(), "mpmd") is None
        assert array_ops.axis_index(P(None, "mpmd"), "mpmd") == 1
        assert array_ops.axis_index(P(("x", "mpmd"), None), "mpmd") == 0

    def test_axis_index_named_sharding(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.jax_mesh, P(None, "mpmd"))

        assert array_ops.axis_index(sharding, "mpmd") == 1

    def test_communication_effect_is_registered(self):
        assert jax_effects.lowerable_effects.contains(communication_effect)
        assert jax_effects.control_flow_allowed_effects.contains(communication_effect)

    def test_experimental_task_infers_in_shardings_from_mpmd(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(x):
            return mpmd.task(identity, name="identity", out_shardings=sharding)(x)

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))
        (task_eqn,) = [eqn for eqn in jaxpr.eqns if eqn.primitive is task_p]

        assert task_eqn.params["task_name"] == "identity"
        assert task_eqn.params["mpmd_idx"] == 0
        assert task_eqn.params["in_shardings"] == (sharding,)
        assert task_eqn.params["out_shardings"] == (sharding,)

    def test_experimental_task_partition_spec_out_sharding_infers_input_mesh(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P("x"))

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(x):
            return mpmd.task(add_one, name="add_one", out_shardings=P("x"))(x)

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))
        (task_eqn,) = [eqn for eqn in jaxpr.eqns if eqn.primitive is task_p]

        assert task_eqn.params["mpmd_idx"] == 0
        assert task_eqn.params["in_shardings"] == (sharding,)
        assert task_eqn.params["out_shardings"] == (sharding,)

    def test_experimental_task_partition_specs_use_inferred_mesh_context(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P("x"))

        def constrained_add_one(x):
            return jax.lax.with_sharding_constraint(add_one(x), P("x"))

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(x):
            return mpmd.task(
                constrained_add_one,
                name="constrained_add_one",
                in_shardings=P("x"),
                out_shardings=P("x"),
            )(x)

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))
        (task_eqn,) = [eqn for eqn in jaxpr.eqns if eqn.primitive is task_p]
        constraints = [
            eqn
            for eqn in task_eqn.params["call_jaxpr"].jaxpr.eqns
            if eqn.primitive.name == "sharding_constraint"
        ]

        self.assertLen(constraints, 1)
        assert task_eqn.params["in_shardings"] == (sharding,)
        assert task_eqn.params["out_shardings"] == (sharding,)
        assert constraints[0].params["context_mesh"] == sharding.mesh.abstract_mesh
        assert constraints[0].params["sharding"].mesh == sharding.mesh.abstract_mesh
        assert constraints[0].params["sharding"].spec == P("x")

    def test_experimental_task_partition_spec_can_use_explicit_mesh_argument(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P("x"))

        @mpmd.mpmd(mesh, in_shardings=())
        def foo():
            return mpmd.task(
                lambda: jnp.zeros((8,), jnp.float32),
                name="zeros",
                mesh=sharding.mesh,
                out_shardings=P("x"),
            )()

        jaxpr = jax.make_jaxpr(foo)()
        (task_eqn,) = [eqn for eqn in jaxpr.eqns if eqn.primitive is task_p]

        assert task_eqn.params["mpmd_idx"] == 0
        assert task_eqn.params["in_shardings"] == ()
        assert task_eqn.params["out_shardings"] == (sharding,)

    def test_experimental_task_partition_spec_requires_inferred_or_explicit_mesh(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")

        @mpmd.mpmd(mesh, in_shardings=())
        def bad():
            return mpmd.task(
                lambda: jnp.zeros((8,), jnp.float32), name="zeros", out_shardings=P("x")
            )()

        with self.assertRaisesRegex(RuntimeError, "requires a non-empty mesh"):
            jax.make_jaxpr(bad)()

    def test_experimental_task_out_shardings_are_required(self):
        with self.assertRaisesRegex(
            TypeError, "required keyword-only argument: 'out_shardings'"
        ):
            mpmd.task(lambda x: x)

    def test_experimental_task_explicit_mesh_must_match_argument_mesh(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P("x"))
        sharding_1 = NamedSharding(mesh.unstack[1], P("x"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def bad(x):
            return mpmd.task(
                add_one, name="bad_mesh", mesh=sharding_1.mesh, out_shardings=P("x")
            )(x)

        with self.assertRaisesRegex(ValueError, "task argument shardings"):
            jax.make_jaxpr(bad)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_task_explicit_mesh_must_match_output_mesh(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P("x"))
        sharding_1 = NamedSharding(mesh.unstack[1], P("x"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def bad(x):
            return mpmd.task(
                add_one, name="bad_mesh", mesh=sharding_0.mesh, out_shardings=sharding_1
            )(x)

        with self.assertRaisesRegex(ValueError, "task out_shardings"):
            jax.make_jaxpr(bad)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_task_explicit_mesh_must_belong_to_mpmd_mesh(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        other_mesh = make_mesh((1, 2), ("mpmd", "x"), "mpmd").jax_mesh

        @mpmd.mpmd(mesh, in_shardings=())
        def bad():
            return mpmd.task(
                lambda: jnp.zeros((8,), jnp.float32),
                name="zeros",
                mesh=other_mesh,
                out_shardings=P("x"),
            )()

        with self.assertRaisesRegex(ValueError, "does not belong to the MPMD mesh"):
            jax.make_jaxpr(bad)()

    def test_experimental_task_accepts_multi_index_mesh(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        task_mesh = mesh.mpmd_submesh([0, 1]).jax_mesh
        sharding = NamedSharding(task_mesh, P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(x):
            return mpmd.task(identity, name="identity", out_shardings=sharding)(x)

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))
        (task_eqn,) = [eqn for eqn in jaxpr.eqns if eqn.primitive is task_p]

        assert task_eqn.params["mpmd_idx"] == task_mesh
        assert task_eqn.params["in_shardings"] == (sharding,)
        assert task_eqn.params["out_shardings"] == (sharding,)

    def test_experimental_mpmd_rejects_keyword_arguments(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(*, x):
            return mpmd.task(identity, name="identity", out_shardings=sharding)(x)

        with self.assertRaisesRegex(ValueError, "keyword arguments"):
            jax.make_jaxpr(foo)(x=jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_mpmd_fun_call_traces_global_jaxpr(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def two_stage(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            future = mpmd.transfer(a, out_shardings=sharding_1)
            b = future.done()
            return b

        self.assertIsInstance(two_stage, mpmd.MpmdFunction)
        jaxpr = jax.make_jaxpr(two_stage)(jax.ShapeDtypeStruct((8,), jnp.float32))

        assert [eqn.primitive for eqn in jaxpr.eqns] == [
            task_p,
            transfer_p,
            transfer_done_p,
        ]

    def test_experimental_transfer_future_done_is_traceable(self):
        mesh = make_mesh((3, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())
        sharding_2 = NamedSharding(mesh.unstack[2], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            future_1 = mpmd.transfer(x, out_shardings=sharding_1)
            future_2 = mpmd.transfer(x, out_shardings=sharding_2)
            b = future_1.done()
            c = future_2.done()
            return b, c

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

        assert [eqn.primitive for eqn in jaxpr.eqns] == [
            transfer_p,
            transfer_p,
            transfer_done_p,
            transfer_done_p,
        ]
        assert all(communication_effect in eqn.effects for eqn in jaxpr.eqns)
        assert set(jaxpr.eqns[0].params) == {"src_shardings", "tgt_shardings"}
        assert set(jaxpr.eqns[1].params) == {"src_shardings", "tgt_shardings"}
        assert jaxpr.eqns[2].params == {}
        assert jaxpr.eqns[3].params == {}
        assert jaxpr.eqns[0].outvars[0].aval is jcore.abstract_token
        assert jaxpr.eqns[1].outvars[0].aval is jcore.abstract_token
        assert jaxpr.eqns[2].invars == [
            jaxpr.eqns[0].outvars[0],
            jaxpr.eqns[0].outvars[1],
        ]
        assert jaxpr.eqns[3].invars == [
            jaxpr.eqns[1].outvars[0],
            jaxpr.eqns[1].outvars[1],
        ]
        returned_vars = set(jaxpr.outvars)
        assert all(outvar not in returned_vars for outvar in jaxpr.eqns[0].outvars[1:])
        assert all(outvar not in returned_vars for outvar in jaxpr.eqns[1].outvars[1:])

    def test_experimental_transfer_accepts_multi_index_meshes(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        src_mesh = mesh.mpmd_submesh([0, 1]).jax_mesh
        tgt_mesh = mesh.mpmd_submesh([2, 3]).jax_mesh
        src_sharding = NamedSharding(src_mesh, P())
        tgt_sharding = NamedSharding(tgt_mesh, P())

        @mpmd.mpmd(mesh, in_shardings=(src_sharding,))
        def move(x):
            future = mpmd.transfer(x, out_shardings=tgt_sharding)
            y = future.done()
            return y

        jaxpr = jax.make_jaxpr(move)(jax.ShapeDtypeStruct((8,), jnp.float32))

        assert [eqn.primitive for eqn in jaxpr.eqns] == [transfer_p, transfer_done_p]
        transfer = jaxpr.eqns[0]
        assert transfer.params == {
            "src_shardings": (src_sharding,),
            "tgt_shardings": (tgt_sharding,),
        }
        assert jaxpr.eqns[1].invars == [transfer.outvars[0], transfer.outvars[1]]

    def test_experimental_stack_and_slice_infer_group_shardings(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        mesh_0 = mesh.unstack[0]
        mesh_1_2 = mesh.mpmd_submesh([1, 2]).jax_mesh
        mesh_0_1_2 = mesh.mpmd_submesh([0, 1, 2]).jax_mesh
        sharding_0 = NamedSharding(mesh_0, P("mpmd"))
        sharding_1_2 = NamedSharding(mesh_1_2, P("mpmd"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1_2))
        def regroup(a, b):
            c = mpmd.stack(a, b)
            left, right = mpmd.slice(c, [[0, 1], 2])
            return c, left, right

        jaxpr = jax.make_jaxpr(regroup)(
            jax.ShapeDtypeStruct((1, 8), jnp.float32),
            jax.ShapeDtypeStruct((2, 8), jnp.float32),
        )
        stack_eqn = next(eqn for eqn in jaxpr.eqns if eqn.primitive is stack_p)
        slice_eqn = next(eqn for eqn in jaxpr.eqns if eqn.primitive is slice_p)

        assert set(stack_eqn.params) == {"in_shardings", "mpmd_mesh", "axis"}
        assert stack_eqn.params["in_shardings"] == (sharding_0, sharding_1_2)
        assert stack_eqn.params["mpmd_mesh"].jax_mesh == mesh_0_1_2
        assert stack_eqn.params["axis"] == 0
        assert stack_eqn.outvars[0].aval.shape == (3, 8)
        assert set(slice_eqn.params) == {"in_sharding", "groups", "mpmd_mesh"}
        assert slice_eqn.params["in_sharding"].mesh == mesh_0_1_2
        assert slice_eqn.params["groups"] == ((0, 1), (2,))
        assert slice_eqn.params["mpmd_mesh"] == mesh
        assert [outvar.aval.shape for outvar in slice_eqn.outvars] == [(2, 8), (1, 8)]

    def test_experimental_stack_task_slice_with_inline_all_reduce(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        mesh_0 = mesh.unstack[0]
        mesh_1_2 = mesh.mpmd_submesh([1, 2]).jax_mesh
        mesh_0_1_2 = mesh.mpmd_submesh([0, 1, 2]).jax_mesh
        sharding_0 = NamedSharding(mesh_0, P("mpmd"))
        sharding_1_2 = NamedSharding(mesh_1_2, P("mpmd"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1_2))
        def regroup(a, b):
            a = mpmd.task(lambda x: x, name="stack-input-0", out_shardings=sharding_0)(
                a
            )
            b = mpmd.task(
                lambda x: x, name="stack-input-1-2", out_shardings=sharding_1_2
            )(b)
            c = mpmd.stack(a, b)
            reduced = mpmd.task(
                lambda x: x.sum(0),
                name="cross-mpmd-all-reduce",
                out_shardings=NamedSharding(mesh_0_1_2, P()),
            )(c)
            return mpmd.slice(reduced, [[0, 1], 2])

        jaxpr = jax.make_jaxpr(regroup)(
            jax.ShapeDtypeStruct((1, 9), jnp.float32),
            jax.ShapeDtypeStruct((2, 9), jnp.float32),
        )

        assert [eqn.primitive for eqn in jaxpr.eqns] == [
            task_p,
            task_p,
            stack_p,
            task_p,
            slice_p,
        ]
        input_0_eqn, input_1_2_eqn, stack_eqn, task_eqn, slice_eqn = jaxpr.eqns
        assert input_0_eqn.params["task_name"] == "stack-input-0"
        assert input_1_2_eqn.params["task_name"] == "stack-input-1-2"
        assert task_eqn.params["task_name"] == "cross-mpmd-all-reduce"
        assert task_eqn.params["mpmd_idx"] == mesh_0_1_2
        assert task_eqn.params["in_shardings"][0].mesh == mesh_0_1_2
        assert task_eqn.params["in_shardings"][0].spec == P("mpmd")
        assert task_eqn.params["out_shardings"][0].mesh == mesh_0_1_2
        assert task_eqn.params["out_shardings"][0].spec == P()
        task_body_eqns = task_eqn.params["call_jaxpr"].jaxpr.eqns
        assert [eqn.primitive.name for eqn in task_body_eqns] == ["reduce_sum"]
        assert task_body_eqns[0].params["axes"] == (0,)
        assert stack_eqn.outvars[0].aval.shape == (3, 9)
        assert task_eqn.outvars[0].aval.shape == (9,)
        assert slice_eqn.params["groups"] == ((0, 1), (2,))
        assert slice_eqn.params["mpmd_mesh"] == mesh
        assert [outvar.aval.shape for outvar in slice_eqn.outvars] == [(9,), (9,)]

    def test_experimental_stack_rejects_overlapping_meshes(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P("mpmd"))

        @mpmd.mpmd(mesh, in_shardings=(sharding, sharding))
        def bad(a, b):
            return mpmd.stack(a, b)

        with self.assertRaisesRegex(ValueError, "duplicate MPMD"):
            jax.make_jaxpr(bad)(
                jax.ShapeDtypeStruct((1, 8), jnp.float32),
                jax.ShapeDtypeStruct((1, 8), jnp.float32),
            )

    def test_experimental_stack_rejects_mismatched_dtypes(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P("mpmd"))
        sharding_1 = NamedSharding(mesh.unstack[1], P("mpmd"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1))
        def bad(a, b):
            return mpmd.stack(a, b)

        with self.assertRaisesRegex(ValueError, "same dtype"):
            jax.make_jaxpr(bad)(
                jax.ShapeDtypeStruct((1, 8), jnp.float32),
                jax.ShapeDtypeStruct((1, 8), jnp.int32),
            )

    def test_experimental_stack_rejects_mixed_replicated_and_mpmd_sharded_groups(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        sharding_0_1 = NamedSharding(mesh.mpmd_submesh([0, 1]).jax_mesh, P("mpmd"))
        sharding_2_3 = NamedSharding(mesh.mpmd_submesh([2, 3]).jax_mesh, P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0_1, sharding_2_3))
        def bad(a, b):
            return mpmd.stack(a, b)

        with self.assertRaisesRegex(ValueError, "mix replicated"):
            jax.make_jaxpr(bad)(
                jax.ShapeDtypeStruct((2, 8), jnp.float32),
                jax.ShapeDtypeStruct((8,), jnp.float32),
            )

    def test_experimental_stack_requires_mpmd_axis_as_full_leading_spec_entry(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P(("x", "mpmd")))
        sharding_1 = NamedSharding(mesh.unstack[1], P(("x", "mpmd")))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1))
        def bad(a, b):
            return mpmd.stack(a, b)

        with self.assertRaisesRegex(ValueError, "PartitionSpec axis 0"):
            jax.make_jaxpr(bad)(
                jax.ShapeDtypeStruct((2, 8), jnp.float32),
                jax.ShapeDtypeStruct((2, 8), jnp.float32),
            )

    def test_experimental_slice_requires_mpmd_axis_as_full_leading_spec_entry(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.jax_mesh, P(("x", "mpmd")))

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def bad(x):
            return mpmd.slice(x, [0, 1])

        with self.assertRaisesRegex(ValueError, "leading PartitionSpec axis"):
            jax.make_jaxpr(bad)(jax.ShapeDtypeStruct((4, 8), jnp.float32))

    def test_experimental_transfer_future_done_rejects_double_done(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            future = mpmd.transfer(x, out_shardings=sharding_1)
            b = future.done()
            future.done()
            return b

        with self.assertRaisesRegex(RuntimeError, "may only be called once"):
            jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_transfer_returns_pytree_result_after_token_in_jaxpr(self):
        mesh = make_mesh((3, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            future = mpmd.transfer((x, x), out_shardings=sharding_1)
            assert callable(future.done)
            b, c = future.done()
            return b, c

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

        transfer, transfer_done = jaxpr.eqns
        assert transfer.primitive is transfer_p
        assert transfer_done.primitive is transfer_done_p
        assert transfer.params == {
            "src_shardings": (sharding_0, sharding_0),
            "tgt_shardings": (sharding_1, sharding_1),
        }
        assert transfer.outvars[0].aval is jcore.abstract_token
        assert jaxpr.outvars == transfer_done.outvars
        assert transfer_done.invars == [transfer.outvars[0], *transfer.outvars[1:]]

    def test_experimental_transfer_accepts_mixed_channels(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1))
        def swap(x, y):
            future = mpmd.transfer((x, y), out_shardings=(sharding_1, sharding_0))
            return future.done()

        jaxpr = jax.make_jaxpr(swap)(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        )

        transfer, transfer_done = jaxpr.eqns
        assert transfer.primitive is transfer_p
        assert transfer_done.primitive is transfer_done_p
        assert transfer.params == {
            "src_shardings": (sharding_0, sharding_1),
            "tgt_shardings": (sharding_1, sharding_0),
        }
        assert transfer_done.invars == [transfer.outvars[0], *transfer.outvars[1:]]
        assert jaxpr.outvars == transfer_done.outvars

    def test_experimental_transfer_rejects_overlapping_leaf_channel(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def same_channel(x):
            return mpmd.transfer(x, out_shardings=sharding_0).done()

        with self.assertRaisesRegex(ValueError, "overlap at leaf 0"):
            jax.make_jaxpr(same_channel)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_task_rejects_donating_input_during_transfer(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            future = mpmd.transfer(a, out_shardings=sharding_1)
            reused = mpmd.task(
                add_one,
                name="reuse_before_done",
                out_shardings=sharding_0,
                donate_argnums=0,
            )(a)
            b = future.done()
            return reused, b

        with self.assertRaisesRegex(ValueError, "has been sent"):
            jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_task_rejects_donating_sent_input_after_transfer_done(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            future = mpmd.transfer(a, out_shardings=sharding_1)
            b = future.done()
            reused = mpmd.task(
                add_one,
                name="reuse_after_done",
                out_shardings=sharding_0,
                donate_argnums=0,
            )(a)
            return reused, b

        with self.assertRaisesRegex(ValueError, "has been sent"):
            jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_transfer_rejects_non_equivalent_out_sharding(self):
        mesh = make_mesh((2, 2), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P("x"))
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            b = mpmd.transfer(x, out_shardings=sharding_1).done()
            return b

        with self.assertRaisesRegex(NotImplementedError, "equivalent"):
            jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_transfer_rejects_non_equivalent_out_memory_kind(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P()).with_memory_kind("device")
        sharding_1 = NamedSharding(mesh.unstack[1], P()).with_memory_kind("pinned_host")

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            b = mpmd.transfer(x, out_shardings=sharding_1).done()
            return b

        with self.assertRaisesRegex(NotImplementedError, "equivalent"):
            jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_mpmd_infers_task_and_transfer_params(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b = mpmd.transfer(a, out_shardings=sharding_1).done()
            return mpmd.task(identity, name="consumer", out_shardings=sharding_1)(b)

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

        producer, transfer, transfer_done, consumer = jaxpr.eqns
        assert [eqn.primitive for eqn in jaxpr.eqns] == [
            task_p,
            transfer_p,
            transfer_done_p,
            task_p,
        ]
        assert producer.params["task_name"] == "producer"
        assert producer.params["mpmd_idx"] == 0
        assert producer.params["in_shardings"] == (sharding_0,)
        assert producer.params["out_shardings"] == (sharding_0,)
        assert transfer.params == {
            "src_shardings": (sharding_0,),
            "tgt_shardings": (sharding_1,),
        }
        assert transfer.outvars[0].aval is jcore.abstract_token
        assert transfer_done.invars == [transfer.outvars[0], transfer.outvars[1]]
        assert consumer.params["task_name"] == "consumer"
        assert consumer.params["mpmd_idx"] == 1
        assert consumer.params["in_shardings"] == (sharding_1,)
        assert consumer.params["out_shardings"] == (sharding_1,)

    def test_experimental_mpmd_rejects_task_mesh_change_without_transfer(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            return mpmd.task(identity, name="bad_consumer", out_shardings=sharding_1)(x)

        with self.assertRaisesRegex(ValueError, "same mesh"):
            jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

    def test_experimental_transfer_future_keeps_shared_source_until_last_done(self):
        mesh = make_mesh((3, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())
        sharding_2 = NamedSharding(mesh.unstack[2], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            future_1 = mpmd.transfer(x, out_shardings=sharding_1)
            future_2 = mpmd.transfer(x, out_shardings=sharding_2)
            b = future_1.done()
            c = future_2.done()
            return b, c

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))
        inferred = finalize_lifetimes(jaxpr, donated_invars=(True,))

        transfer_indices = [
            idx for idx, eqn in enumerate(inferred.eqns) if eqn.primitive is transfer_p
        ]
        delete_indices = [
            idx for idx, eqn in enumerate(inferred.eqns) if eqn.primitive is delete_p
        ]

        assert len(transfer_indices) == 2
        assert len(delete_indices) == 1
        assert delete_indices[0] > transfer_indices[1]
        assert inferred.eqns[delete_indices[0]].invars == [jaxpr.invars[0]]


def _trace_same_channel_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
    def foo(x):
        a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
        b_future = mpmd.transfer(a, out_shardings=sharding_1)
        c_future = mpmd.transfer(a, out_shardings=sharding_1)
        b_in = b_future.done()
        c_in = c_future.done()
        b = mpmd.task(identity, name="consumer_0", out_shardings=sharding_1)(b_in)
        c = mpmd.task(identity, name="consumer_1", out_shardings=sharding_1)(c_in)
        return b, c

    return jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))


def _trace_same_channel_reverse_use_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
    def foo(x):
        a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
        b_future = mpmd.transfer(a, out_shardings=sharding_1)
        c_future = mpmd.transfer(a, out_shardings=sharding_1)
        b_in = b_future.done()
        c_in = c_future.done()
        c = mpmd.task(identity, name="consumer_1", out_shardings=sharding_1)(c_in)
        b = mpmd.task(identity, name="consumer_0", out_shardings=sharding_1)(b_in)
        return b, c

    return jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))


def _trace_reusable_recv_buffer_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
    def foo(x):
        a0 = mpmd.task(identity, name="producer_0", out_shardings=sharding_0)(x)
        b0 = mpmd.transfer(a0, out_shardings=sharding_1).done()
        c0 = mpmd.task(identity, name="consumer_0", out_shardings=sharding_1)(b0)
        a1 = mpmd.task(add_one, name="producer_1", out_shardings=sharding_0)(x)
        b1 = mpmd.transfer(a1, out_shardings=sharding_1).done()
        c1 = mpmd.task(identity, name="consumer_1", out_shardings=sharding_1)(b1)
        return c0, c1

    return jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))


def _trace_same_channel_non_fifo_done_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
    def foo(x):
        a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
        b_future = mpmd.transfer(a, out_shardings=sharding_1)
        c_future = mpmd.transfer(a, out_shardings=sharding_1)
        c_in = c_future.done()
        b_in = b_future.done()
        b = mpmd.task(identity, name="consumer_0", out_shardings=sharding_1)(b_in)
        c = mpmd.task(identity, name="consumer_1", out_shardings=sharding_1)(c_in)
        return b, c

    return jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))


def _trace_bidirectional_reverse_use_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1))
    def foo(left_x, right_x):
        left = mpmd.task(identity, name="left", out_shardings=sharding_0)(left_x)
        right = mpmd.task(identity, name="right", out_shardings=sharding_1)(right_x)
        left_to_right_0 = mpmd.transfer(left, out_shardings=sharding_1).done()
        left_to_right_1 = mpmd.transfer(left, out_shardings=sharding_1).done()
        right_to_left_0 = mpmd.transfer(right, out_shardings=sharding_0).done()
        right_to_left_1 = mpmd.transfer(right, out_shardings=sharding_0).done()
        use_left_to_right_1 = mpmd.task(
            identity, name="use_ltr_1", out_shardings=sharding_1
        )(left_to_right_1)
        use_left_to_right_0 = mpmd.task(
            identity, name="use_ltr_0", out_shardings=sharding_1
        )(left_to_right_0)
        use_right_to_left_1 = mpmd.task(
            identity, name="use_rtl_1", out_shardings=sharding_0
        )(right_to_left_1)
        use_right_to_left_0 = mpmd.task(
            identity, name="use_rtl_0", out_shardings=sharding_0
        )(right_to_left_0)
        return (
            use_left_to_right_0,
            use_left_to_right_1,
            use_right_to_left_0,
            use_right_to_left_1,
        )

    return jax.make_jaxpr(foo)(
        jax.ShapeDtypeStruct((8,), jnp.float32), jax.ShapeDtypeStruct((8,), jnp.float32)
    )


def _trace_ping_pong_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
    def foo(x):
        a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
        b = mpmd.transfer(a, out_shardings=sharding_1).done()
        c = mpmd.task(identity, name="middle", out_shardings=sharding_1)(b)
        d = mpmd.transfer(c, out_shardings=sharding_0).done()
        return mpmd.task(identity, name="consumer", out_shardings=sharding_0)(d)

    return jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))


def _trace_send_then_reuse_jaxpr(mesh):
    sharding_0 = NamedSharding(mesh.unstack[0], P())
    sharding_1 = NamedSharding(mesh.unstack[1], P())

    @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
    def foo(x):
        a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
        b = mpmd.transfer(a, out_shardings=sharding_1).done()
        reused = mpmd.task(add_one, name="reuse_sent", out_shardings=sharding_0)(a)
        received = mpmd.task(
            multiply_two, name="use_received", out_shardings=sharding_1
        )(b)
        return reused, received

    return jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))


def _finalized_local_jaxprs(jaxpr, mesh):
    return [
        local_jaxpr.closed_jaxpr.map_jaxpr(
            partial(
                finalize_lifetimes,
                donated_invars=(False,) * len(local_jaxpr.global_invar_indices),
            )
        )
        for local_jaxpr in to_local_jaxprs(jaxpr, mesh)
    ]


def _comm_order(jaxpr):
    return [
        eqn.primitive
        for eqn in jaxpr.eqns
        if eqn.primitive in {transfer_start_p, recv_done_p}
    ]


def _task_eqn_by_name(jaxpr, task_name):
    return next(
        eqn
        for eqn in jaxpr.eqns
        if eqn.primitive is task_p and eqn.params["task_name"] == task_name
    )


class TestSendRecvLifetime(jtu.JaxTestCase):
    def test_lifetime_passes_split_donation_from_deletes(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        jaxpr = _trace_send_then_reuse_jaxpr(mesh)

        donated = infer_donation(jaxpr, donated_invars=(False,))

        assert not any(eqn.primitive is delete_p for eqn in donated.eqns)
        assert _task_eqn_by_name(donated, "reuse_sent").params["donate_invars"] == (
            False,
        )
        assert _task_eqn_by_name(donated, "use_received").params["donate_invars"] == (
            False,
        )

        with_deletes = add_deletes(donated, donated_invars=(False,))

        assert any(eqn.primitive is delete_p for eqn in with_deletes.eqns)
        assert _task_eqn_by_name(with_deletes, "reuse_sent").params[
            "donate_invars"
        ] == (False,)

    def test_explicit_transfer_done_keeps_repeated_sends_distinct(self):
        mesh = make_mesh((3, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())
        sharding_2 = NamedSharding(mesh.unstack[2], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def fanout(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b_in = mpmd.transfer(a, out_shardings=sharding_1).done()
            c_in = mpmd.transfer(a, out_shardings=sharding_2).done()
            b = mpmd.task(identity, name="consumer_1", out_shardings=sharding_1)(b_in)
            c = mpmd.task(identity, name="consumer_2", out_shardings=sharding_2)(c_in)
            return b, c

        cjaxpr = jax.make_jaxpr(fanout)(jax.ShapeDtypeStruct((8,), jnp.float32))

        assert isinstance(cjaxpr, jcore.ClosedJaxpr)
        transfer_eqns = [eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p]
        transfer_done_eqns = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        ]

        assert len(transfer_eqns) == 2
        assert all(
            set(eqn.params) == {"src_shardings", "tgt_shardings"}
            for eqn in transfer_eqns
        )
        assert all(eqn.params == {} for eqn in transfer_done_eqns)
        assert [eqn.invars for eqn in transfer_done_eqns] == [
            [eqn.outvars[0], *eqn.outvars[1:]] for eqn in transfer_eqns
        ]
        assert all(len(eqn.outvars) == 1 for eqn in transfer_done_eqns)
        assert communication_effect in cjaxpr.effects

    def test_transfer_done_tracks_same_channel_transfers_by_token(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_same_channel_jaxpr(mesh)
        transfer_eqns = [eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p]
        transfer_done_eqns = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        ]

        assert [eqn.invars[0] for eqn in transfer_done_eqns] == [
            eqn.outvars[0] for eqn in transfer_eqns
        ]

    def test_to_local_jaxprs_transfer_done_fifo_check_is_optional(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_same_channel_non_fifo_done_jaxpr(mesh)

        to_local_jaxprs(cjaxpr, mesh)

        with self.assertRaisesRegex(ValueError, "transfer_done must be FIFO"):
            to_local_jaxprs(cjaxpr, mesh, check_transfer_done_fifo=True)

    def test_finalize_lifetimes_allows_sent_buffer_reuse_after_transfer(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        jaxpr = _trace_send_then_reuse_jaxpr(mesh)

        inferred = finalize_lifetimes(jaxpr, donated_invars=(False,))

        assert _task_eqn_by_name(inferred, "reuse_sent").params["donate_invars"] == (
            False,
        )
        assert _task_eqn_by_name(inferred, "use_received").params["donate_invars"] == (
            False,
        )

    def test_finalize_lifetimes_allows_source_use_before_done_without_donation(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            future = mpmd.transfer(a, out_shardings=sharding_1)
            reused = mpmd.task(
                add_one, name="reuse_sent_before_done", out_shardings=sharding_0
            )(a)
            b = future.done()
            received = mpmd.task(
                multiply_two, name="use_received", out_shardings=sharding_1
            )(b)
            return reused, received

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

        inferred = finalize_lifetimes(jaxpr, donated_invars=(False,))

        transfer_eqn_idx = next(
            idx for idx, eqn in enumerate(jaxpr.eqns) if eqn.primitive is transfer_p
        )
        reuse_idx = next(
            idx
            for idx, eqn in enumerate(jaxpr.eqns)
            if eqn.primitive is task_p
            and eqn.params["task_name"] == "reuse_sent_before_done"
        )
        transfer_done_idx = next(
            idx
            for idx, eqn in enumerate(jaxpr.eqns)
            if eqn.primitive is transfer_done_p
        )
        transfer = jaxpr.eqns[transfer_eqn_idx]
        reuse = _task_eqn_by_name(inferred, "reuse_sent_before_done")

        # Reading the sent value before future.done() is valid, but donating it would
        # let the task clobber storage that the in-flight transfer still owns.
        assert transfer_eqn_idx < reuse_idx < transfer_done_idx
        assert reuse.invars == [transfer.invars[0]]
        assert reuse.params["donate_invars"] == (False,)

    def test_to_local_jaxprs_rejects_transfer_without_transfer_done(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def foo(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            mpmd.transfer(a, out_shardings=sharding_1)
            return a

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))

        with self.assertRaisesRegex(ValueError, "matching transfer_done"):
            to_local_jaxprs(jaxpr, mesh)

    def test_to_local_jaxprs_rejects_raw_transfer_output(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())
        aval = jcore.ShapedArray((8,), jnp.float32)
        x = jcore.Var(aval)
        transfer = new_primitive_eqn(
            transfer_p, [x], src_shardings=(sharding_0,), tgt_shardings=(sharding_1,)
        )
        _token, raw_received = transfer.outvars
        jaxpr = jcore.Jaxpr(
            constvars=(),
            invars=[x],
            outvars=[raw_received],
            eqns=[transfer],
            effects=transfer.effects,
        )

        with self.assertRaisesRegex(
            ValueError, "transfer output used without transfer_done"
        ):
            to_local_jaxprs(jcore.ClosedJaxpr(jaxpr, ()), mesh)


class TestSendRecvLocalJaxprs(jtu.JaxTestCase):
    def test_experimental_mpmd_returns_local_jaxprs_with_deletes(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def two_stage(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            future = mpmd.transfer(a, out_shardings=sharding_1)
            b = future.done()
            return mpmd.task(identity, name="consumer", out_shardings=sharding_1)(b)

        lowered = two_stage.lower(jax.ShapeDtypeStruct((8,), jnp.float32))

        self.assertIs(lowered.mpmd_mesh, mesh)
        self.assertEqual(lowered.in_shardings, (sharding_0,))
        self.assertLen(lowered.local_jaxprs, 2)
        self.assertEqual(
            [local_jaxpr.global_invar_indices for local_jaxpr in lowered.local_jaxprs],
            [(0,), ()],
        )
        src_jaxpr, tgt_jaxpr = [
            local_jaxpr.closed_jaxpr for local_jaxpr in lowered.local_jaxprs
        ]
        assert [eqn.primitive for eqn in src_jaxpr.eqns] == [
            task_p,
            transfer_start_p,
            delete_p,
        ]
        assert [eqn.primitive for eqn in tgt_jaxpr.eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
            task_p,
            delete_p,
        ]
        self.assertEqual(lowered.out_shardings, sharding_1)
        self.assertEqual(lowered.out_shape.sharding, sharding_1)

    def test_lowered_local_jaxpr_prunes_unused_global_inputs(self):
        mesh = make_mesh((1, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding, sharding))
        def only_uses_first_arg(x, y):
            del y
            return mpmd.task(add_one, out_shardings=sharding)(x)

        lowered = only_uses_first_arg.lower(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        )

        self.assertEqual(lowered.local_jaxprs[0].global_invar_indices, (0,))
        self.assertLen(lowered.local_jaxprs[0].closed_jaxpr.invars, 1)

    def test_experimental_mpmd_lowers_multi_index_task_mesh(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        task_mesh = mesh.mpmd_submesh([0, 1]).jax_mesh
        sharding = NamedSharding(task_mesh, P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,), donate_argnums=0)
        def consume_input(x):
            return mpmd.task(identity, name="consume", out_shardings=sharding)(x)

        lowered = consume_input.lower(jax.ShapeDtypeStruct((8,), jnp.float32))

        self.assertLen(lowered.local_jaxprs, 4)
        for mpmd_idx in (0, 1):
            local_jaxpr = lowered.local_jaxprs[mpmd_idx].closed_jaxpr
            assert [eqn.primitive for eqn in local_jaxpr.eqns] == [task_p, delete_p]
            task, delete = local_jaxpr.eqns
            assert task.params["mpmd_idx"] == task_mesh
        for mpmd_idx in (2, 3):
            assert lowered.local_jaxprs[mpmd_idx].closed_jaxpr.eqns == []
        self.assertEqual(lowered.out_shardings, sharding)

    def test_to_local_jaxprs_expands_stack_and_slice_casts_by_mesh(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        mesh_0 = mesh.unstack[0]
        mesh_1_2 = mesh.mpmd_submesh([1, 2]).jax_mesh
        mesh_0_1 = mesh.mpmd_submesh([0, 1]).jax_mesh
        sharding_0 = NamedSharding(mesh_0, P("mpmd"))
        sharding_1_2 = NamedSharding(mesh_1_2, P("mpmd"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1_2))
        def regroup(a, b):
            c = mpmd.stack(a, b)
            return mpmd.slice(c, [[0, 1], 2])

        cjaxpr = jax.make_jaxpr(regroup)(
            jax.ShapeDtypeStruct((1, 8), jnp.float32),
            jax.ShapeDtypeStruct((2, 8), jnp.float32),
        )

        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        assert [eqn.primitive for eqn in local_jaxprs[0].eqns] == [
            local_stack_p,
            local_slice_p,
        ]
        assert [eqn.primitive for eqn in local_jaxprs[1].eqns] == [
            local_stack_p,
            local_slice_p,
        ]
        assert [eqn.primitive for eqn in local_jaxprs[2].eqns] == [
            local_stack_p,
            local_slice_p,
        ]
        assert local_jaxprs[3].eqns == []
        assert local_jaxprs[0].outvars == [cjaxpr.outvars[0]]
        assert local_jaxprs[1].outvars == [cjaxpr.outvars[0]]
        assert local_jaxprs[2].outvars == [cjaxpr.outvars[1]]
        assert local_jaxprs[0].eqns[0].params["out_sharding"].mesh == (
            mesh.mpmd_submesh([0, 1, 2]).jax_mesh
        )
        assert local_jaxprs[1].eqns[1].params["out_shardings"][0].mesh == mesh_0_1
        assert (
            local_jaxprs[2].eqns[1].params["out_shardings"][0].mesh == mesh.unstack[2]
        )

    def test_to_local_jaxprs_expands_stack_task_slice_by_mesh(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        mesh_0 = mesh.unstack[0]
        mesh_1_2 = mesh.mpmd_submesh([1, 2]).jax_mesh
        mesh_0_1_2 = mesh.mpmd_submesh([0, 1, 2]).jax_mesh
        sharding_0 = NamedSharding(mesh_0, P("mpmd"))
        sharding_1_2 = NamedSharding(mesh_1_2, P("mpmd"))

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1_2))
        def regroup(a, b):
            a = mpmd.task(lambda x: x, name="stack-input-0", out_shardings=sharding_0)(
                a
            )
            b = mpmd.task(
                lambda x: x, name="stack-input-1-2", out_shardings=sharding_1_2
            )(b)
            c = mpmd.stack(a, b)
            reduced = mpmd.task(
                lambda x: x.sum(0),
                name="cross-mpmd-all-reduce",
                out_shardings=NamedSharding(mesh_0_1_2, P()),
            )(c)
            return mpmd.slice(reduced, [[0, 1], 2])

        cjaxpr = jax.make_jaxpr(regroup)(
            jax.ShapeDtypeStruct((1, 9), jnp.float32),
            jax.ShapeDtypeStruct((2, 9), jnp.float32),
        )
        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        for mpmd_idx in (0, 1, 2):
            assert [eqn.primitive for eqn in local_jaxprs[mpmd_idx].eqns] == [
                task_p,
                local_stack_p,
                task_p,
                local_slice_p,
            ]
            expected_input_task = (
                "stack-input-0" if mpmd_idx == 0 else "stack-input-1-2"
            )
            assert (
                local_jaxprs[mpmd_idx].eqns[0].params["task_name"]
                == expected_input_task
            )
            assert (
                local_jaxprs[mpmd_idx].eqns[2].params["task_name"]
                == "cross-mpmd-all-reduce"
            )
            assert local_jaxprs[mpmd_idx].eqns[2].params["mpmd_idx"] == mesh_0_1_2
        assert local_jaxprs[3].eqns == []
        assert local_jaxprs[0].outvars == [cjaxpr.outvars[0]]
        assert local_jaxprs[1].outvars == [cjaxpr.outvars[0]]
        assert local_jaxprs[2].outvars == [cjaxpr.outvars[1]]
        assert local_jaxprs[0].eqns[3].params["out_shardings"][0].mesh == (
            mesh.mpmd_submesh([0, 1]).jax_mesh
        )
        assert (
            local_jaxprs[2].eqns[3].params["out_shardings"][0].mesh == mesh.unstack[2]
        )

    def test_slice_zero_fill_stack_models_mpmd_reshard_patterns(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        rest_mesh = mesh.mpmd_submesh([1, 2, 3]).jax_mesh

        @mpmd.mpmd(mesh, in_shardings=(NamedSharding(mesh.jax_mesh, P("mpmd")),))
        def round_trip(x):
            y = mpmd.task(
                identity,
                name="all-gather",
                out_shardings=NamedSharding(mesh.jax_mesh, P()),
            )(x)  # {0,1,2,3}
            x, rest = mpmd.slice(y, [0, [1, 2, 3]])  # x: {0}, rest: {1,2,3}
            x = mpmd.task(
                lambda x: x[1:2],
                name="take-1",
                # Locally replicated, sharded when stacked.
                out_shardings=NamedSharding(mesh.unstack[0], P("mpmd")),
            )(x)  # {0}
            z = mpmd.task(
                lambda: jnp.zeros_like(rest[:3]),
                name="zero-fill-rest",
                out_shardings=NamedSharding(rest_mesh, P("mpmd")),
            )()  # {1,2,3}
            stacked = mpmd.stack(x, z)  # {0,1,2,3}
            return mpmd.task(
                identity,
                name="shard-back-to-mpmd-axis",
                out_shardings=NamedSharding(mesh.jax_mesh, P("mpmd")),
            )(stacked)  # {0,1,2,3}

        cjaxpr = jax.make_jaxpr(round_trip)(jax.ShapeDtypeStruct((4, 8), jnp.float32))

        assert [eqn.primitive for eqn in cjaxpr.eqns] == [
            task_p,
            slice_p,
            task_p,
            task_p,
            stack_p,
            task_p,
        ]
        identity_task, slice_eqn, take_task, zeros_task, stack_eqn, output_task = (
            cjaxpr.eqns
        )

        assert identity_task.params["task_name"] == "all-gather"
        assert identity_task.params["mpmd_idx"] == mesh.jax_mesh
        assert identity_task.params["in_shardings"][0].mesh == mesh.jax_mesh
        assert identity_task.params["in_shardings"][0].spec == P("mpmd")
        assert identity_task.params["out_shardings"][0].mesh == mesh.jax_mesh
        assert identity_task.params["out_shardings"][0].spec == P()

        assert slice_eqn.params["in_sharding"].mesh == mesh.jax_mesh
        assert slice_eqn.params["in_sharding"].spec == P()
        assert slice_eqn.params["groups"] == ((0,), (1, 2, 3))
        assert slice_eqn.params["mpmd_mesh"] == mesh
        assert [outvar.aval.shape for outvar in slice_eqn.outvars] == [(4, 8), (4, 8)]

        assert take_task.params["task_name"] == "take-1"
        assert take_task.params["mpmd_idx"] == 0
        assert take_task.params["in_shardings"][0].mesh == mesh.unstack[0]
        assert take_task.params["in_shardings"][0].spec == P()
        assert take_task.params["out_shardings"][0].mesh == mesh.unstack[0]
        assert take_task.params["out_shardings"][0].spec == P("mpmd")
        assert take_task.outvars[0].aval.shape == (1, 8)

        assert zeros_task.params["task_name"] == "zero-fill-rest"
        assert zeros_task.params["mpmd_idx"] == rest_mesh
        assert zeros_task.params["in_shardings"][0].mesh == rest_mesh
        assert zeros_task.params["in_shardings"][0].spec == P()
        assert zeros_task.params["out_shardings"][0].mesh == rest_mesh
        assert zeros_task.params["out_shardings"][0].spec == P("mpmd")
        assert zeros_task.outvars[0].aval.shape == (3, 8)

        assert tuple(
            sharding.mesh for sharding in stack_eqn.params["in_shardings"]
        ) == (mesh.unstack[0], rest_mesh)
        assert tuple(
            sharding.spec for sharding in stack_eqn.params["in_shardings"]
        ) == (P("mpmd"), P("mpmd"))
        assert stack_eqn.params["mpmd_mesh"] == mesh
        assert stack_eqn.params["axis"] == 0
        assert stack_eqn.outvars[0].aval.shape == (4, 8)

        assert output_task.params["task_name"] == "shard-back-to-mpmd-axis"
        assert output_task.params["mpmd_idx"] == mesh.jax_mesh
        assert output_task.params["in_shardings"][0].mesh == mesh.jax_mesh
        assert output_task.params["in_shardings"][0].spec == P("mpmd")
        assert output_task.params["out_shardings"][0].mesh == mesh.jax_mesh
        assert output_task.params["out_shardings"][0].spec == P("mpmd")

        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        assert [eqn.primitive for eqn in local_jaxprs[0].eqns] == [
            task_p,
            local_slice_p,
            task_p,
            local_stack_p,
            task_p,
        ]
        assert local_jaxprs[0].eqns[0].params["task_name"] == "all-gather"
        assert local_jaxprs[0].eqns[2].params["task_name"] == "take-1"
        assert local_jaxprs[0].eqns[-1].params["task_name"] == "shard-back-to-mpmd-axis"

        for mpmd_idx in (1, 2, 3):
            assert [eqn.primitive for eqn in local_jaxprs[mpmd_idx].eqns] == [
                task_p,
                local_slice_p,
                task_p,
                local_stack_p,
                task_p,
            ]
            assert local_jaxprs[mpmd_idx].eqns[0].params["task_name"] == "all-gather"
            assert (
                local_jaxprs[mpmd_idx].eqns[2].params["task_name"] == "zero-fill-rest"
            )
            assert (
                local_jaxprs[mpmd_idx].eqns[-1].params["task_name"]
                == "shard-back-to-mpmd-axis"
            )

        for local_jaxpr in local_jaxprs:
            assert local_jaxpr.outvars == cjaxpr.outvars

    def test_experimental_mpmd_donate_argnums_applies_to_local_deletes(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,), donate_argnums=0)
        def consume_input(x):
            return mpmd.task(identity, name="consume", out_shardings=sharding)(x)

        lowered = consume_input.lower(jax.ShapeDtypeStruct((8,), jnp.float32))
        local_jaxpr = lowered.local_jaxprs[0].closed_jaxpr

        assert [eqn.primitive for eqn in local_jaxpr.eqns] == [task_p, delete_p]
        assert local_jaxpr.eqns[-1].invars == [local_jaxpr.invars[0]]

    def test_experimental_mpmd_infer_donation_updates_local_task_donation(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(
            mesh, in_shardings=(sharding,), donate_argnums=0, infer_donation=True
        )
        def consume_input(x):
            return mpmd.task(identity, name="consume", out_shardings=sharding)(x)

        lowered = consume_input.lower(jax.ShapeDtypeStruct((8,), jnp.float32))
        local_jaxpr = lowered.local_jaxprs[0].closed_jaxpr
        task = next(eqn for eqn in local_jaxpr.eqns if eqn.primitive is task_p)

        assert task.params["donate_invars"] == (True,)

    def test_to_local_jaxprs_reports_global_io_indices(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_bidirectional_reverse_use_jaxpr(mesh)

        local_jaxprs = to_local_jaxprs(cjaxpr, mesh)

        self.assertEqual(
            [local_jaxpr.global_invar_indices for local_jaxpr in local_jaxprs],
            [(0,), (1,)],
        )
        self.assertEqual(
            [local_jaxpr.global_outvar_indices for local_jaxpr in local_jaxprs],
            [(2, 3), (0, 1)],
        )

    def test_to_local_jaxprs_closes_used_constvars_without_global_invar_indices(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_bidirectional_reverse_use_jaxpr(mesh)
        constvar = cjaxpr.invars[1]
        const_value = np.arange(8, dtype=np.float32)
        cjaxpr = cjaxpr.replace(
            jaxpr=cjaxpr.jaxpr.replace(constvars=[constvar], invars=[cjaxpr.invars[0]]),
            consts=(const_value,),
        )

        mpmd0_jaxpr, mpmd1_jaxpr = to_local_jaxprs(cjaxpr, mesh)
        mpmd0 = mpmd0_jaxpr.closed_jaxpr
        mpmd1 = mpmd1_jaxpr.closed_jaxpr

        self.assertEqual(mpmd0_jaxpr.global_invar_indices, (0,))
        self.assertEqual(mpmd1_jaxpr.global_invar_indices, ())
        self.assertEmpty(mpmd0.consts)
        self.assertLen(mpmd1.consts, 1)
        self.assertIs(mpmd1.consts[0], const_value)
        self.assertEqual(list(mpmd1.constvars), [constvar])
        self.assertEqual(mpmd1.invars, [])
        finalized = mpmd1.map_jaxpr(partial(finalize_lifetimes, donated_invars=()))
        assert _task_eqn_by_name(finalized, "right").params["donate_invars"] == (False,)

    def test_to_local_jaxprs_threads_send_token_and_recv_done_before_first_use(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def two_stage(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b = mpmd.transfer(a, out_shardings=sharding_1).done()
            return mpmd.task(identity, name="consumer", out_shardings=sharding_1)(b)

        cjaxpr = jax.make_jaxpr(two_stage)(jax.ShapeDtypeStruct((8,), jnp.float32))
        transfer = next(eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p)
        transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        transfer_token, _raw_received = transfer.outvars
        (received,) = transfer_done.outvars

        src_jaxpr, tgt_jaxpr = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        assert [eqn.primitive for eqn in src_jaxpr.eqns] == [task_p, transfer_start_p]
        send_start = src_jaxpr.eqns[1]
        assert len(send_start.outvars) == 1
        assert send_start.outvars == [transfer_token]

        assert [eqn.primitive for eqn in tgt_jaxpr.eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
            task_p,
        ]
        zeros, recv_start, recv_done, scalar_consumer = tgt_jaxpr.eqns
        assert recv_start.invars == zeros.outvars
        assert recv_start.outvars[0].aval is jcore.abstract_token
        assert recv_done.invars == recv_start.outvars
        assert recv_done.outvars == [received]
        assert scalar_consumer.invars == [received]

    def test_to_local_jaxprs_expands_transfer_between_multi_index_meshes(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        src_mesh = mesh.mpmd_submesh([0, 1]).jax_mesh
        tgt_mesh = mesh.mpmd_submesh([2, 3]).jax_mesh
        src_sharding = NamedSharding(src_mesh, P())
        tgt_sharding = NamedSharding(tgt_mesh, P())

        @mpmd.mpmd(mesh, in_shardings=(src_sharding,))
        def transfer_between_groups(x):
            future = mpmd.transfer(x, out_shardings=tgt_sharding)
            y = future.done()
            return y

        cjaxpr = jax.make_jaxpr(transfer_between_groups)(
            jax.ShapeDtypeStruct((8,), jnp.float32)
        )
        transfer = next(eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p)
        transfer_token, raw_received = transfer.outvars
        transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        (received,) = transfer_done.outvars
        assert transfer.params == {
            "src_shardings": (src_sharding,),
            "tgt_shardings": (tgt_sharding,),
        }
        assert transfer_done.invars == [transfer_token, raw_received]

        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        for mpmd_idx in (0, 1):
            local_jaxpr = local_jaxprs[mpmd_idx]
            assert [eqn.primitive for eqn in local_jaxpr.eqns] == [transfer_start_p]
            send_start = local_jaxpr.eqns[0]
            assert send_start.params["send_local_shardings"][0].mesh == src_mesh
            assert send_start.params["send_remote_shardings"][0].mesh == tgt_mesh

        for mpmd_idx in (2, 3):
            local_jaxpr = local_jaxprs[mpmd_idx]
            assert [eqn.primitive for eqn in local_jaxpr.eqns] == [
                zeros_p,
                transfer_start_p,
                recv_done_p,
            ]
            recv_start = local_jaxpr.eqns[1]
            assert recv_start.params["recv_local_shardings"][0].mesh == tgt_mesh
            assert recv_start.params["recv_remote_shardings"][0].mesh == src_mesh
            assert local_jaxpr.outvars == [received]

    def test_to_local_jaxprs_groups_mixed_channel_transfer(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, sharding_1))
        def swap(x, y):
            return mpmd.transfer((x, y), out_shardings=(sharding_1, sharding_0)).done()

        cjaxpr = jax.make_jaxpr(swap)(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        )
        transfer = next(eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p)
        transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )

        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        for mpmd_idx, expected_outvar, remote_sharding in (
            (0, transfer_done.outvars[1], sharding_1),
            (1, transfer_done.outvars[0], sharding_0),
        ):
            local_jaxpr = local_jaxprs[mpmd_idx]
            assert [eqn.primitive for eqn in local_jaxpr.eqns] == [
                zeros_p,
                transfer_start_p,
                recv_done_p,
            ]
            zeros, transfer_start, recv_done = local_jaxpr.eqns
            assert transfer_start.invars == [transfer.invars[mpmd_idx], *zeros.outvars]
            assert transfer_start.params["send_remote_shardings"] == (remote_sharding,)
            assert transfer_start.params["recv_remote_shardings"] == (remote_sharding,)
            assert recv_done.invars == transfer_start.outvars
            assert recv_done.outvars == [expected_outvar]
            assert local_jaxpr.outvars == [expected_outvar]

    def test_to_local_jaxprs_orders_single_then_multi_index_transfer(self):
        mesh = make_mesh((4, 1), ("mpmd", "x"), "mpmd")
        mesh_0 = mesh.unstack[0]
        mesh_3 = mesh.unstack[3]
        src_mesh = mesh.mpmd_submesh([0, 1]).jax_mesh
        tgt_mesh = mesh.mpmd_submesh([2, 3]).jax_mesh
        sharding_0 = NamedSharding(mesh_0, P())
        sharding_3 = NamedSharding(mesh_3, P())
        src_sharding = NamedSharding(src_mesh, P())
        tgt_sharding = NamedSharding(tgt_mesh, P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0, src_sharding))
        def transfer_single_then_group(x0, x1):
            future_0_to_3 = mpmd.transfer(x0, out_shardings=sharding_3)
            received_0_to_3 = future_0_to_3.done()
            future_multi = mpmd.transfer(x1, out_shardings=tgt_sharding)
            received_multi = future_multi.done()
            return received_0_to_3, received_multi

        cjaxpr = jax.make_jaxpr(transfer_single_then_group)(
            jax.ShapeDtypeStruct((8,), jnp.float32),
            jax.ShapeDtypeStruct((8,), jnp.float32),
        )
        transfer_0_to_3, transfer_multi = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p
        ]
        transfer_done_0_to_3, transfer_done_multi = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        ]
        (received_0_to_3,) = transfer_done_0_to_3.outvars
        (received_multi,) = transfer_done_multi.outvars
        assert transfer_0_to_3.params == {
            "src_shardings": (sharding_0,),
            "tgt_shardings": (sharding_3,),
        }
        assert transfer_multi.params == {
            "src_shardings": (src_sharding,),
            "tgt_shardings": (tgt_sharding,),
        }

        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        assert [eqn.primitive for eqn in local_jaxprs[0].eqns] == [
            transfer_start_p,
            transfer_start_p,
        ]
        assert [eqn.primitive for eqn in local_jaxprs[1].eqns] == [transfer_start_p]
        assert [eqn.primitive for eqn in local_jaxprs[2].eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
        ]
        assert [eqn.primitive for eqn in local_jaxprs[3].eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
            transfer_start_p,
            recv_done_p,
        ]

        rank_3_first_recv = local_jaxprs[3].eqns[1]
        rank_3_second_recv = local_jaxprs[3].eqns[3]
        assert rank_3_first_recv.params["recv_remote_shardings"][0].mesh == mesh_0
        assert rank_3_first_recv.params["recv_local_shardings"][0].mesh == mesh_3
        assert rank_3_second_recv.params["recv_remote_shardings"][0].mesh == src_mesh
        assert rank_3_second_recv.params["recv_local_shardings"][0].mesh == tgt_mesh
        assert local_jaxprs[3].outvars == [received_0_to_3, received_multi]

    def test_finalize_lifetimes_deletes_transfer_source_and_received_buffer(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def two_stage(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b = mpmd.transfer(a, out_shardings=sharding_1).done()
            return mpmd.task(identity, name="consumer", out_shardings=sharding_1)(b)

        cjaxpr = jax.make_jaxpr(two_stage)(jax.ShapeDtypeStruct((8,), jnp.float32))
        transfer = next(eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p)
        produced = transfer.invars[0]
        transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        (received,) = transfer_done.outvars

        src_jaxpr, tgt_jaxpr = _finalized_local_jaxprs(cjaxpr, mesh)

        assert [eqn.primitive for eqn in src_jaxpr.eqns] == [
            task_p,
            transfer_start_p,
            delete_p,
        ]
        assert src_jaxpr.eqns[-1].invars == [produced]

        assert [eqn.primitive for eqn in tgt_jaxpr.eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
            task_p,
            delete_p,
        ]
        assert tgt_jaxpr.eqns[-1].invars == [received]

    def test_to_local_jaxprs_materializes_received_jaxpr_output_without_consumer(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def output_only(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b = mpmd.transfer(a, out_shardings=sharding_1).done()
            return b

        cjaxpr = jax.make_jaxpr(output_only)(jax.ShapeDtypeStruct((8,), jnp.float32))
        transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        (received,) = transfer_done.outvars

        _, tgt_jaxpr = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        assert [eqn.primitive for eqn in tgt_jaxpr.eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
        ]
        assert tgt_jaxpr.eqns[1].invars == tgt_jaxpr.eqns[0].outvars
        assert tgt_jaxpr.eqns[-1].outvars == [received]
        assert tgt_jaxpr.outvars == [received]

    def test_to_local_jaxprs_materializes_received_value_before_resending(self):
        mesh = make_mesh((3, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())
        sharding_2 = NamedSharding(mesh.unstack[2], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def chained_transfer(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b = mpmd.transfer(a, out_shardings=sharding_1).done()
            c = mpmd.transfer(b, out_shardings=sharding_2).done()
            return mpmd.task(identity, name="consumer", out_shardings=sharding_2)(c)

        cjaxpr = jax.make_jaxpr(chained_transfer)(
            jax.ShapeDtypeStruct((8,), jnp.float32)
        )
        first_transfer, second_transfer = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p
        ]
        _first_transfer_token, first_received = first_transfer.outvars
        first_transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        (first_done,) = first_transfer_done.outvars

        _, middle_jaxpr, _ = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        assert [eqn.primitive for eqn in middle_jaxpr.eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
            transfer_start_p,
        ]
        zeros = middle_jaxpr.eqns[0]
        recv_start = middle_jaxpr.eqns[1]
        recv_done = middle_jaxpr.eqns[2]
        send_start = middle_jaxpr.eqns[3]
        assert recv_done.params == {}
        assert recv_start.invars == zeros.outvars
        assert recv_done.outvars == [first_done]
        assert recv_done.invars == recv_start.outvars
        assert send_start.params["send_remote_shardings"] == (sharding_2,)
        assert send_start.params["send_local_shardings"] == (sharding_1,)
        assert send_start.params["recv_remote_shardings"] == ()
        assert send_start.params["recv_local_shardings"] == ()
        assert send_start.invars == [first_done]
        assert send_start.outvars == [second_transfer.outvars[0]]

    def test_finalize_lifetimes_deletes_received_buffer_after_resending(self):
        mesh = make_mesh((3, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())
        sharding_2 = NamedSharding(mesh.unstack[2], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def chained_transfer(x):
            a = mpmd.task(identity, name="producer", out_shardings=sharding_0)(x)
            b = mpmd.transfer(a, out_shardings=sharding_1).done()
            c = mpmd.transfer(b, out_shardings=sharding_2).done()
            return mpmd.task(identity, name="consumer", out_shardings=sharding_2)(c)

        cjaxpr = jax.make_jaxpr(chained_transfer)(
            jax.ShapeDtypeStruct((8,), jnp.float32)
        )
        first_transfer, _ = [eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_p]
        first_transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        (first_received,) = first_transfer_done.outvars

        _, middle_jaxpr, _ = _finalized_local_jaxprs(cjaxpr, mesh)

        assert [eqn.primitive for eqn in middle_jaxpr.eqns] == [
            zeros_p,
            transfer_start_p,
            recv_done_p,
            transfer_start_p,
            delete_p,
        ]
        assert middle_jaxpr.eqns[-1].invars == [first_received]

    def test_to_local_jaxprs_batches_recv_zeros_at_local_jaxpr_start_by_default(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_bidirectional_reverse_use_jaxpr(mesh)

        _, mpmd1_jaxpr = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        zeros_eqns = [eqn for eqn in mpmd1_jaxpr.eqns if eqn.primitive is zeros_p]
        assert len(zeros_eqns) == 1
        assert mpmd1_jaxpr.eqns[0] is zeros_eqns[0]
        assert len(zeros_eqns[0].outvars) == 2
        assert [
            eqn.invars[0]
            for eqn in mpmd1_jaxpr.eqns
            if eqn.primitive is transfer_start_p
        ][:2] == zeros_eqns[0].outvars

    def test_to_local_jaxprs_does_not_reuse_received_buffers_when_disabled(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_reusable_recv_buffer_jaxpr(mesh)

        with env_vars.jaxpp_reuse_recv_buffers.set(False):
            _, tgt_jaxpr = [
                local_jaxpr.closed_jaxpr
                for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
            ]

        zeros_eqns = [eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is zeros_p]
        recv_starts = [
            eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is transfer_start_p
        ]
        assert len(zeros_eqns) == 0
        assert recv_starts[0].invars == []
        assert recv_starts[1].invars == []

    def test_finalize_lifetimes_deletes_received_values_not_reusable_buffers(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_reusable_recv_buffer_jaxpr(mesh)
        first_transfer_done, second_transfer_done = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        ]
        (first_received,) = first_transfer_done.outvars
        (second_received,) = second_transfer_done.outvars

        _, tgt_jaxpr = _finalized_local_jaxprs(cjaxpr, mesh)

        zeros_eqns = [eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is zeros_p]
        recv_starts = [
            eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is transfer_start_p
        ]
        reuse_fences = [eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is reuse_fence_p]
        delete_invars = [
            invar
            for eqn in tgt_jaxpr.eqns
            if eqn.primitive is delete_p
            for invar in eqn.invars
        ]

        assert zeros_eqns[0].outvars[0] not in delete_invars
        assert recv_starts[0].outvars[1] not in delete_invars
        assert reuse_fences[0].outvars[0] not in delete_invars
        assert delete_invars == [first_received, second_received]
        assert _task_eqn_by_name(tgt_jaxpr, "consumer_0").params["donate_invars"] == (
            False,
        )
        assert _task_eqn_by_name(tgt_jaxpr, "consumer_1").params["donate_invars"] == (
            False,
        )

    def test_finalize_lifetimes_deletes_received_values_without_reusable_buffers(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_reusable_recv_buffer_jaxpr(mesh)
        first_transfer_done, second_transfer_done = [
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        ]
        (first_received,) = first_transfer_done.outvars
        (second_received,) = second_transfer_done.outvars

        with env_vars.jaxpp_reuse_recv_buffers.set(False):
            _, tgt_jaxpr = _finalized_local_jaxprs(cjaxpr, mesh)

        delete_invars = [
            invar
            for eqn in tgt_jaxpr.eqns
            if eqn.primitive is delete_p
            for invar in eqn.invars
        ]

        assert delete_invars == [first_received, second_received]

    def test_to_local_jaxprs_reuses_received_buffer_after_last_use_by_default(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = _trace_reusable_recv_buffer_jaxpr(mesh)
        first_transfer_done = next(
            eqn for eqn in cjaxpr.eqns if eqn.primitive is transfer_done_p
        )
        (first_received,) = first_transfer_done.outvars

        _, tgt_jaxpr = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        zeros_eqns = [eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is zeros_p]
        recv_starts = [
            eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is transfer_start_p
        ]
        reuse_fences = [eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is reuse_fence_p]
        consumer_0_idx = next(
            idx
            for idx, eqn in enumerate(tgt_jaxpr.eqns)
            if eqn is _task_eqn_by_name(tgt_jaxpr, "consumer_0")
        )
        reuse_fence_idx = next(
            idx for idx, eqn in enumerate(tgt_jaxpr.eqns) if eqn is reuse_fences[0]
        )
        second_transfer_start_idx = next(
            idx for idx, eqn in enumerate(tgt_jaxpr.eqns) if eqn is recv_starts[1]
        )
        assert len(zeros_eqns) == 1
        assert len(zeros_eqns[0].outvars) == 1
        assert len(reuse_fences) == 1
        assert recv_starts[0].invars == zeros_eqns[0].outvars
        assert reuse_fences[0].invars == [first_received]
        assert recv_starts[1].invars == reuse_fences[0].outvars
        assert consumer_0_idx < reuse_fence_idx < second_transfer_start_idx

    def test_to_local_jaxprs_reuses_received_buffers_fifo_by_default(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding_0 = NamedSharding(mesh.unstack[0], P())
        sharding_1 = NamedSharding(mesh.unstack[1], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding_0,))
        def fifo_reusable_recv_buffer(x):
            a0 = mpmd.task(identity, name="producer_0", out_shardings=sharding_0)(x)
            b0 = mpmd.transfer(a0, out_shardings=sharding_1).done()
            a1 = mpmd.task(add_one, name="producer_1", out_shardings=sharding_0)(x)
            b1 = mpmd.transfer(a1, out_shardings=sharding_1).done()
            first_pair = mpmd.task(
                add_pair, name="first_pair", out_shardings=sharding_1
            )(b0, b1)

            a2 = mpmd.task(multiply_two, name="producer_2", out_shardings=sharding_0)(x)
            b2 = mpmd.transfer(a2, out_shardings=sharding_1).done()
            a3 = mpmd.task(identity, name="producer_3", out_shardings=sharding_0)(x)
            b3 = mpmd.transfer(a3, out_shardings=sharding_1).done()
            second_0 = mpmd.task(identity, name="second_0", out_shardings=sharding_1)(
                b2
            )
            second_1 = mpmd.task(identity, name="second_1", out_shardings=sharding_1)(
                b3
            )
            return first_pair, second_0, second_1

        cjaxpr = jax.make_jaxpr(fifo_reusable_recv_buffer)(
            jax.ShapeDtypeStruct((8,), jnp.float32)
        )

        _, tgt_jaxpr = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        recv_starts = [
            eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is transfer_start_p
        ]
        reuse_fences = [eqn for eqn in tgt_jaxpr.eqns if eqn.primitive is reuse_fence_p]

        assert len(reuse_fences) == 2
        assert len(recv_starts) == 4
        assert recv_starts[2].invars == reuse_fences[0].outvars
        assert recv_starts[3].invars == reuse_fences[1].outvars

    def assertDoneTokensFollowStarts(self, jaxpr, start_p, done_p):
        starts = [
            eqn
            for eqn in jaxpr.eqns
            if eqn.primitive is start_p
            and (start_p is not transfer_start_p or len(eqn.outvars) > 1)
        ]
        dones = [eqn for eqn in jaxpr.eqns if eqn.primitive is done_p]
        self.assertEqual(
            [eqn.invars[0] for eqn in dones], [eqn.outvars[0] for eqn in starts]
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "same_channel",
            "trace": _trace_same_channel_jaxpr,
            "expected_orders": (
                [transfer_start_p, transfer_start_p],
                [transfer_start_p, transfer_start_p, recv_done_p, recv_done_p],
            ),
        },
        {
            "testcase_name": "same_channel_reverse_use",
            "trace": _trace_same_channel_reverse_use_jaxpr,
            "expected_orders": (
                [transfer_start_p, transfer_start_p],
                [transfer_start_p, transfer_start_p, recv_done_p, recv_done_p],
            ),
        },
        {
            "testcase_name": "bidirectional_reverse_use",
            "trace": _trace_bidirectional_reverse_use_jaxpr,
            "expected_orders": (
                [
                    transfer_start_p,
                    transfer_start_p,
                    transfer_start_p,
                    recv_done_p,
                    transfer_start_p,
                    recv_done_p,
                ],
                [
                    transfer_start_p,
                    recv_done_p,
                    transfer_start_p,
                    recv_done_p,
                    transfer_start_p,
                    transfer_start_p,
                ],
            ),
        },
        {
            "testcase_name": "ping_pong",
            "trace": _trace_ping_pong_jaxpr,
            "expected_orders": (
                [transfer_start_p, transfer_start_p, recv_done_p],
                [transfer_start_p, recv_done_p, transfer_start_p],
            ),
        },
    )
    def test_to_local_jaxprs_orders_communication(self, trace, expected_orders):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        cjaxpr = trace(mesh)

        local_jaxprs = [
            local_jaxpr.closed_jaxpr for local_jaxpr in to_local_jaxprs(cjaxpr, mesh)
        ]

        self.assertEqual(
            [_comm_order(jaxpr) for jaxpr in local_jaxprs], list(expected_orders)
        )
        for jaxpr in local_jaxprs:
            self.assertDoneTokensFollowStarts(jaxpr, transfer_start_p, recv_done_p)


class TestTaskPrimitive(jtu.JaxTestCase):
    def test_task_dce_filters_task_metadata_from_make_jaxpr(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(x):
            a, _ = mpmd.task(
                duplicate,
                name="duplicate",
                out_shardings=(sharding, sharding),
                donate_argnums=0,
            )(x)
            return a

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32)).jaxpr
        dced_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr, [True])
        (task_eqn,) = [eqn for eqn in dced_jaxpr.eqns if eqn.primitive is task_p]

        assert used_inputs == [True]
        assert len(task_eqn.invars) == 1
        assert len(task_eqn.outvars) == 1
        assert task_eqn.params["donate_invars"] == (True,)
        assert len(task_eqn.params["in_shardings"]) == 1
        assert len(task_eqn.params["out_shardings"]) == 1
        assert len(task_eqn.params["call_jaxpr"].out_avals) == 1

    def test_task_donate_argnums_applies_to_keyword_calls(self):
        mesh = make_mesh((2, 1), ("mpmd", "x"), "mpmd")
        sharding = NamedSharding(mesh.unstack[0], P())

        @mpmd.mpmd(mesh, in_shardings=(sharding,))
        def foo(x):
            return mpmd.task(
                identity, name="identity", out_shardings=sharding, donate_argnums=0
            )(x=x)

        jaxpr = jax.make_jaxpr(foo)(jax.ShapeDtypeStruct((8,), jnp.float32))
        (task_eqn,) = [eqn for eqn in jaxpr.eqns if eqn.primitive is task_p]

        assert task_eqn.params["donate_invars"] == (True,)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
