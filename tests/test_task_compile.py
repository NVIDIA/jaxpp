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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import io_callback
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpp.api import MpmdMesh
from jaxpp.jax_primitives import task_p


def add_index(i, x):
    return (x + i,)


def bind_task(call_jaxpr, *args, in_shardings, out_shardings, donate_invars):
    # Binds outside of tracing, as `jcore.eval_jaxpr` does at runtime.
    return task_p.bind(
        *args,
        call_jaxpr=call_jaxpr,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        donate_invars=donate_invars,
        task_name="task",
        task_info=None,
        mpmd_idx=0,
    )


class TestTaskCompile(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        mesh = jtu.create_mesh((1, 1), ("stage", "model"))
        self.mpmd_mesh = MpmdMesh(mesh, "stage")
        self.replicated = NamedSharding(self.mpmd_mesh.unstack[0], P())
        self.sharded = NamedSharding(self.mpmd_mesh.unstack[0], P("model"))
        # Traced with a Python int, so the index is weakly typed like a loop index.
        self.add_index_jaxpr = jax.make_jaxpr(add_index)(0, jnp.zeros(4, jnp.float32))
        self.index_dtype = self.add_index_jaxpr.in_avals[0].dtype

    def add_index(self, index, x):
        with self.mpmd_mesh:
            return bind_task(
                self.add_index_jaxpr,
                index,
                x,
                in_shardings=(self.replicated, self.sharded),
                out_shardings=(self.sharded,),
                donate_invars=(False, True),
            )

    def arange(self):
        return jax.device_put(jnp.arange(4, dtype=jnp.float32), self.sharded)

    @parameterized.named_parameters(
        ("python_int", lambda i, dtype, sharding: i),
        ("numpy_scalar", lambda i, dtype, sharding: dtype.type(i)),
        ("uncommitted_array", lambda i, dtype, sharding: jnp.asarray(i, dtype)),
        (
            "committed_weak_array",
            lambda i, dtype, sharding: jax.device_put(i, sharding),
        ),
        (
            "committed_array",
            lambda i, dtype, sharding: jax.device_put(dtype.type(i), sharding),
        ),
    )
    def test_index_representations_share_executable(self, make_index):
        committed_index = jax.device_put(self.index_dtype.type(1), self.replicated)
        self.add_index(committed_index, self.arange())

        index = make_index(2, self.index_dtype, self.replicated)
        x = self.arange()
        with jtu.count_jit_and_pmap_lowerings() as lowerings:
            (out,) = self.add_index(index, x)
        self.assertEqual(lowerings(), 0)
        self.assertArraysEqual(out, np.arange(4, dtype=np.float32) + 2)
        self.assertTrue(x.is_deleted())

    def test_index_with_wrong_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "types differ"):
            self.add_index(np.float32(1), self.arange())
        with self.assertRaisesRegex(TypeError, "types differ"):
            self.add_index(np.zeros(2, self.index_dtype), self.arange())

    def test_ordered_effects_run_in_call_order(self):
        seen = []

        def record(x):
            seen.append(int(x))
            return x

        def fn(x):
            out_type = jax.ShapeDtypeStruct((), np.int32)
            return (io_callback(record, out_type, x, ordered=True),)

        call_jaxpr = jax.make_jaxpr(fn)(np.int32(0))
        args = [np.int32(0), jax.device_put(np.int32(1), self.replicated), np.int32(2)]
        with self.mpmd_mesh, jtu.count_jit_and_pmap_lowerings() as lowerings:
            for arg in args:
                bind_task(
                    call_jaxpr,
                    arg,
                    in_shardings=(self.replicated,),
                    out_shardings=(self.replicated,),
                    donate_invars=(False,),
                )
            jax.effects_barrier()
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(lowerings(), 1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
