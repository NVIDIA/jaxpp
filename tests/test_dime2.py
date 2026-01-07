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

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import PartitionSpec as P
from parameterized import parameterized

import jaxpp.distributed_utils as jppdu
from jaxpp.dime2 import send_or_recv


class SendOrRecvTest(jppdu.JaxDistributedTest):
    @parameterized.expand(
        [
            ("float32", jnp.float32, np.float32),
            ("bfloat16", jnp.bfloat16, ml_dtypes.bfloat16),
            ("float8_e4m3fn", jnp.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            ("float8_e5m2", jnp.float8_e5m2, ml_dtypes.float8_e5m2),
        ]
    )
    def test_send_or_recv(self, name, jax_dtype, np_dtype):
        process_count = jax.process_count()
        process_index = jax.process_index()
        local_device_count = jax.local_device_count()

        # Use first device from each of the first two processes
        devices = np.array(jax.devices()).reshape(process_count, local_device_count)
        sender_device = devices[0:1]
        receiver_device = devices[1:2]

        sender_mesh = jax.sharding.Mesh(sender_device, axis_names=("mpmd", "x"))
        receiver_mesh = jax.sharding.Mesh(receiver_device, axis_names=("mpmd", "x"))

        pspec = P("x")
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, pspec)
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, pspec)

        global_shape = (8,)
        expected_values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np_dtype)

        if process_index == 0:
            array = jax.device_put(
                jnp.array(expected_values, dtype=jax_dtype), sender_sharding
            )

            [wait_send_finish] = send_or_recv(
                [array], [receiver_sharding], is_send=True
            )
            wait_send_finish()
        else:
            buffer = jax.device_put(
                jnp.zeros(global_shape, dtype=jax_dtype), receiver_sharding
            )

            [enqueue_recv] = send_or_recv([buffer], [sender_sharding], is_send=False)
            received_array = enqueue_recv()

            received_values = np.array(received_array)
            np.testing.assert_array_equal(
                received_values,
                expected_values,
                err_msg=f"Received data mismatch for dtype {name}",
            )


if __name__ == "__main__":
    jppdu.distributed_main(unittest.main)
