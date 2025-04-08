# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

# TODO: Fix & remove
pytest.skip(allow_module_level=True)

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import ray
from helper import JaxppUnitTest
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as P

from jaxpp.compilation import pjit_to_serializeable_mesh_computation
from jaxpp.mesh import RemoteMpmdMesh
from jaxpp.ops import CommDesc, PpermuteOp, RecvOp, RunOp, SendOp


class MeshShardedCommunication(JaxppUnitTest):
    def test_sharded_communication(self):
        num_nodes = 2
        mesh_shape = (2, 2)
        mesh = RemoteMpmdMesh(
            num_stages=num_nodes,
            worker_mesh_shape=mesh_shape,
            worker_mesh_axis_names=("outer", "inner"),
            ray_address="local",
        )

        nodes = (0, 1)
        comm_key, ranks = mesh.establish_ad2ad_comms(nodes)

        matrix_shape = (2, 2)

        def remote_sdas():
            """
            0 1   0 1 2 3
            2 3,
            """
            return jnp.arange(4, dtype=jnp.float32).reshape(matrix_shape), jnp.arange(
                4, dtype=jnp.float32
            )

        mesh0 = mesh.node_mesh(0)
        mesh1 = mesh.node_mesh(1)

        s00 = jax.sharding.NamedSharding(mesh0, P("outer", "inner"))
        s01 = jax.sharding.NamedSharding(mesh0, P("inner"))

        comp0 = pjit_to_serializeable_mesh_computation(
            closed_jaxpr=jax.make_jaxpr(remote_sdas)(),
            in_axis_resources=(),
            out_axis_resources=[s00, s01],
        )

        s10 = jax.sharding.NamedSharding(mesh1, P("outer", "inner"))
        s11 = jax.sharding.NamedSharding(mesh1, P("inner"))
        comp1 = pjit_to_serializeable_mesh_computation(
            closed_jaxpr=jax.make_jaxpr(
                lambda: (jnp.zeros(matrix_shape), jnp.zeros((4,)))
            )(),
            in_axis_resources=(),
            out_axis_resources=[s10, s11],
        )

        ray.get(mesh.workers[0].put_mesh_computation.remote(0, comp0))
        ray.get(mesh.workers[1].put_mesh_computation.remote(0, comp1))
        ray.get(mesh.workers[0].execute_instructions.remote([RunOp(0, [], [1, 2])]))
        ray.get(mesh.workers[0].execute_instructions.remote([RunOp(0, [], [3, 4])]))
        ray.get(mesh.workers[1].execute_instructions.remote([RunOp(0, [], [1, 2])]))
        ray.get(mesh.workers[1].execute_instructions.remote([RunOp(0, [], [3, 4])]))

        def dbg(env):
            sda1, sda2 = env[1], env[2]
            bufs = sda1.device_buffers
            for b in bufs:
                print(repr(b.device()))

        ray.get(mesh.workers[0].run_lambda.remote(dbg))
        ray.get(mesh.workers[1].run_lambda.remote(dbg))

        dmap0 = s00.devices_indices_map(matrix_shape)
        dmap1 = s10.devices_indices_map(matrix_shape)

        s_dev_id = []
        d_dev_id = []
        for (d0, i0), (d1, i1) in jax.util.safe_zip(dmap0.items(), dmap1.items()):
            assert i0 == i1
            s_dev_id.append(d0.id)
            d_dev_id.append(d1.id)

        send_op = SendOp((0, 1), [CommDesc(1, d_dev_id)])
        recv_op = RecvOp((0, 1), [CommDesc(1, s_dev_id)])
        f1 = mesh.workers[0].execute_instructions.remote([send_op])
        f2 = mesh.workers[1].execute_instructions.remote([recv_op])

        ray.get([f1, f2])
        self.assertTrue(
            np.all(
                ray.get(mesh.workers[0].get_tensor.remote(1))
                == ray.get(mesh.workers[1].get_tensor.remote(1))
            )
        )

        gpspecs = [P("stages", "outer", "inner")]
        gmesh = jax.sharding.Mesh(
            mesh.mesh.devices[[0, 1]],
            mesh.mesh.axis_names,
        )

        # NOTE: `perm` below are indices into the `gmesh` `stages` axes
        jaxpr = jax.make_jaxpr(
            shard_map.shard_map(
                partial(jax.lax.ppermute, axis_name="stages", perm=[(0, 1)]),
                mesh=gmesh,
                in_specs=(gpspecs,),
                out_specs=gpspecs,
            )
        )([jnp.zeros((2, 2, 2))])

        gshardings = [jax.sharding.NamedSharding(gmesh, gpspec) for gpspec in gpspecs]

        lowering = pjit_to_serializeable_mesh_computation(
            jaxpr,
            in_axis_resources=gshardings,
            out_axis_resources=gshardings,
            name="ppermute(0->1)",
        )

        ray.get(mesh.workers[0].put_mesh_computation.remote(5, lowering))
        ray.get(mesh.workers[1].put_mesh_computation.remote(5, lowering))

        f1 = mesh.workers[0].execute_instructions.remote(
            [PpermuteOp(exec_uid=5, is_sender=True, stage_ids=[0, 1], arrays=[3])]
        )

        f2 = mesh.workers[1].execute_instructions.remote(
            [PpermuteOp(exec_uid=5, is_sender=False, stage_ids=[0, 1], arrays=[3])]
        )

        ray.get([f1, f2])

        self.assertTrue(
            np.all(
                ray.get(mesh.workers[0].get_tensor.remote(3))
                == ray.get(mesh.workers[1].get_tensor.remote(3))
            )
        )


if __name__ == "__main__":
    unittest.main()
