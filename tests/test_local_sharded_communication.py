# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import jax
import jax.numpy as jnp
import numpy as np
from helper import JaxppUnitTest
from jaxpp.dime import Dime
from jaxpp.ops import CommDesc
from jaxpp.types import CommKeyWithNcclId, UniqueGlobalDeviceIds

TEST_BLOCKING_NON_DEPENDENT = False


class LocalShardedCommunication(JaxppUnitTest):
    def test_sharded_communication(self):
        tot_devs = np.array(jax.devices())
        assert len(tot_devs) % 2 == 0
        n_local_devs = len(tot_devs) // 2

        MF = 2**20  # 1MiF = 4MiB
        size = n_local_devs * 128 * MF

        dime = Dime()
        with dime:
            for sd, rd in zip(
                tot_devs[:n_local_devs], tot_devs[n_local_devs:], strict=True
            ):
                key = UniqueGlobalDeviceIds.strict_create((sd.id, rd.id))
                nccl_id = dime.communicator_nccl_id(key)
                comm_key = CommKeyWithNcclId(key, nccl_id)
                dime.register_gpus(comm_key)

        send_buf = jax.device_put(jnp.ones(size), tot_devs[0])
        recv_buf = jax.device_put(jnp.zeros(size), tot_devs[4])

        if TEST_BLOCKING_NON_DEPENDENT:

            @jax.jit
            def operation(a):
                return (a * a).sum()

            op0 = operation.lower(send_buf).compile()
            op4 = operation.lower(recv_buf).compile()

            tmp0 = op0(send_buf)
            tmp4 = op4(recv_buf)
            jax.block_until_ready(tmp0)
            jax.block_until_ready(tmp4)
            del tmp0
            del tmp4

            jax.profiler.start_trace("./tblogs")

        send_commdesc = CommDesc(None, None, None, from_dev_ids=(0,), to_dev_ids=(4,))
        recv_commdesc = CommDesc(None, None, None, from_dev_ids=(0,), to_dev_ids=(4,))

        with dime:
            dime.send([(send_buf, send_commdesc)])
            (recv_buf,) = dime.recv_in_group([(recv_buf, recv_commdesc)])

        if TEST_BLOCKING_NON_DEPENDENT:
            jax.block_until_ready(op0(send_buf))
            jax.block_until_ready(op4(recv_buf.to_jax_array()))
            jax.profiler.stop_trace()
            return

        mesh = jax.sharding.Mesh(tot_devs[:n_local_devs], ("d",))
        pspec = jax.sharding.PartitionSpec("d")
        send_sharding = jax.sharding.NamedSharding(mesh, pspec)
        recv_sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(tot_devs[n_local_devs:], ("d",)), pspec
        )

        send_buf = jax.device_put(jnp.arange(size, dtype=jnp.float32), send_sharding)
        recv_buf = jax.device_put(jnp.ones(size, dtype=jnp.float32), recv_sharding)

        with dime:
            dime.send(
                [
                    (
                        send_buf,
                        CommDesc(
                            None,
                            None,
                            None,
                            from_dev_ids=(0, 1, 2, 3),
                            to_dev_ids=(4, 5, 6, 7),
                        ),
                    )
                ]
            )
            (recv_buf,) = dime.recv_in_group(
                [
                    (
                        recv_buf,
                        CommDesc(
                            None,
                            None,
                            None,
                            from_dev_ids=(0, 1, 2, 3),
                            to_dev_ids=(4, 5, 6, 7),
                        ),
                    )
                ]
            )

        self.assertEqual(
            jax.device_get(send_buf), jax.device_get(recv_buf.to_jax_array())
        )


if __name__ == "__main__":
    unittest.main()
