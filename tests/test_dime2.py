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
from jaxpp.experimental import mpmd
from jaxpp.core import to_local_jaxprs
from jaxpp.dime2 import get_distributed_client, start_transfer
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import (
    CommToken,
    recv_done_p,
    reuse_fence_p,
    transfer_done_impl,
    transfer_start_p,
)
from jaxpp.mesh import MpmdMesh

_ASYNC_READ_DELAY_MATRIX_SIZE = 4096
_ASYNC_READ_DELAY_STEPS = 512


@jax.jit
def _delayed_read(x, m):
    def body(_, acc):
        return acc @ m

    acc = jax.lax.fori_loop(0, _ASYNC_READ_DELAY_STEPS, body, m)
    delay = acc[0, 0].astype(jnp.float32) * jnp.asarray(1e-6, jnp.float32)
    return x + delay


class SendRecvTest(jppdu.JaxDistributedTest):
    @parameterized.expand(
        [
            ("float32", jnp.float32, np.float32),
            ("bfloat16", jnp.bfloat16, ml_dtypes.bfloat16),
            ("float8_e4m3fn", jnp.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            ("float8_e5m2", jnp.float8_e5m2, ml_dtypes.float8_e5m2),
        ]
    )
    def test_send_recv(self, name, jax_dtype, np_dtype):
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

            transfer = start_transfer([array], [receiver_sharding], [], [])
            transfer.done()
        else:
            buffer = jax.device_put(
                jnp.zeros(global_shape, dtype=jax_dtype), receiver_sharding
            )

            transfer = start_transfer([], [], [buffer], [sender_sharding])
            (received_array,) = transfer.done()

            received_values = np.array(received_array)
            np.testing.assert_array_equal(
                received_values,
                expected_values,
                err_msg=f"Received data mismatch for dtype {name}",
            )

    def test_start_transfer_send_only_done_returns_empty(self):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        sender_mesh = jax.sharding.Mesh(devices[0:1], axis_names=("mpmd", "x"))
        receiver_mesh = jax.sharding.Mesh(devices[1:2], axis_names=("mpmd", "x"))
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, P("x"))
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, P("x"))

        if process_index == 0:
            payload = jax.device_put(jnp.arange(8, dtype=jnp.float32), sender_sharding)
            transfer = start_transfer([payload], [receiver_sharding], [], [])
            self.assertEqual(transfer.done(), ())
        else:
            buffer = jax.device_put(
                jnp.zeros((8,), dtype=jnp.float32), receiver_sharding
            )
            start_transfer([], [], [buffer], [sender_sharding]).done()

    def test_start_transfer_groups_bidirectional_send_recv(self):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        local_mesh = jax.sharding.Mesh(
            devices[process_index : process_index + 1], axis_names=("mpmd", "x")
        )
        remote_process_index = 1 - process_index
        remote_mesh = jax.sharding.Mesh(
            devices[remote_process_index : remote_process_index + 1],
            axis_names=("mpmd", "x"),
        )
        local_sharding = jax.sharding.NamedSharding(local_mesh, P("x"))
        remote_sharding = jax.sharding.NamedSharding(remote_mesh, P("x"))

        payload_values = np.arange(8, dtype=np.float32) + process_index * 10
        expected_values = np.arange(8, dtype=np.float32) + remote_process_index * 10
        payload = jax.device_put(jnp.asarray(payload_values), local_sharding)
        buffer = jax.device_put(jnp.zeros((8,), dtype=jnp.float32), local_sharding)

        transfer = start_transfer(
            [payload], [remote_sharding], [buffer], [remote_sharding]
        )
        (received,) = transfer.done()

        np.testing.assert_array_equal(np.asarray(received), expected_values)

    def test_transfer_fanout_eval_local_jaxpr(self):
        if jax.process_count() != 3:
            self.skipTest("Test requires exactly three processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(3, local_device_count)
        mpmd_mesh = MpmdMesh(
            jax.sharding.Mesh(devices[:, :1], axis_names=("mpmd", "x")), "mpmd"
        )
        sharding_0 = jax.sharding.NamedSharding(mpmd_mesh.unstack[0], P())
        sharding_1 = jax.sharding.NamedSharding(mpmd_mesh.unstack[1], P())
        sharding_2 = jax.sharding.NamedSharding(mpmd_mesh.unstack[2], P())

        @mpmd.mpmd(mpmd_mesh, in_shardings=(sharding_0, sharding_0))
        def fanout(a, b):
            return mpmd.transfer((a, b), out_shardings=(sharding_1, sharding_2)).done()

        cjaxpr = jax.make_jaxpr(fanout)(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        )
        local_jaxpr = to_local_jaxprs(cjaxpr, mpmd_mesh)[process_index]

        values = (np.arange(4, dtype=np.float32), np.arange(10, 14, dtype=np.float32))
        if local_jaxpr.global_invar_indices:
            global_args = tuple(jax.device_put(value, sharding_0) for value in values)
            local_args = [global_args[idx] for idx in local_jaxpr.global_invar_indices]
        else:
            local_args = []

        with mpmd_mesh:
            actual = jcore.eval_jaxpr(
                local_jaxpr.closed_jaxpr.jaxpr,
                local_jaxpr.closed_jaxpr.consts,
                *local_args,
            )

        if process_index == 0:
            self.assertEqual(actual, [])
        elif process_index == 1:
            (received,) = actual
            np.testing.assert_array_equal(np.asarray(received), values[0])
        else:
            (received,) = actual
            np.testing.assert_array_equal(np.asarray(received), values[1])

    def test_transfer_done_impl_is_noop(self):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        sender_mesh = jax.sharding.Mesh(devices[0:1], axis_names=("mpmd", "x"))
        receiver_mesh = jax.sharding.Mesh(devices[1:2], axis_names=("mpmd", "x"))
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, P("x"))
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, P("x"))

        if process_index == 0:
            payload = jax.device_put(jnp.arange(8, dtype=jnp.float32), sender_sharding)
            transfer = start_transfer([payload], [receiver_sharding], [], [])

            (returned_payload,) = transfer_done_impl(CommToken(transfer), payload)
            self.assertIs(returned_payload, payload)
        else:
            buffer = jax.device_put(
                jnp.zeros((8,), dtype=jnp.float32), receiver_sharding
            )
            start_transfer([], [], [buffer], [sender_sharding]).done()

    def test_buffered_recv_done_deletes_destination_buffer_after_import(self):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        sender_mesh = jax.sharding.Mesh(devices[0:1], axis_names=("mpmd", "x"))
        receiver_mesh = jax.sharding.Mesh(devices[1:2], axis_names=("mpmd", "x"))
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, P("x"))
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, P("x"))

        global_shape = (8,)
        expected_values = np.ones(global_shape, dtype=np.float32)

        if process_index == 0:
            payload = jax.device_put(
                jnp.ones(global_shape, dtype=jnp.float32), sender_sharding
            )
            start_transfer([payload], [receiver_sharding], [], []).done()
        else:
            buffer = jax.device_put(
                jnp.zeros(global_shape, dtype=jnp.float32), receiver_sharding
            )
            tok, recv_buffer = transfer_start_p.bind(
                buffer,
                send_remote_shardings=(),
                send_local_shardings=(),
                recv_remote_shardings=(sender_sharding,),
                recv_local_shardings=(receiver_sharding,),
            )
            (received,) = recv_done_p.bind(tok, recv_buffer)

            self.assertTrue(recv_buffer.is_deleted())
            np.testing.assert_array_equal(np.asarray(received), expected_values)

    def test_transfer_start_allocates_recv_buffers_when_omitted(self):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        mpmd_mesh = MpmdMesh(
            jax.sharding.Mesh(devices[:, :1], axis_names=("mpmd", "x")), "mpmd"
        )
        sender_mesh, receiver_mesh = mpmd_mesh.unstack
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, P("x"))
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, P("x"))
        expected_values = np.arange(8, dtype=np.float32)

        if process_index == 0:
            payload = jax.device_put(jnp.asarray(expected_values), sender_sharding)
            start_transfer([payload], [receiver_sharding], [], []).done()
        else:
            with mpmd_mesh:
                tok, allocated = transfer_start_p.bind(
                    send_remote_shardings=(),
                    send_local_shardings=(),
                    recv_remote_shardings=(sender_sharding,),
                    recv_local_shardings=(receiver_sharding,),
                    out_avals=(jax.ShapeDtypeStruct((8,), jnp.float32),),
                )
                self.assertEqual(allocated.shape, (8,))
                self.assertEqual(allocated.dtype, jnp.float32)
                self.assertEqual(allocated.sharding, receiver_sharding)
                (received,) = recv_done_p.bind(tok, allocated)

            self.assertTrue(allocated.is_deleted())
            np.testing.assert_array_equal(np.asarray(received), expected_values)

    def test_logical_recv_allocates_private_buffer_for_repeated_recvs(self):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        mpmd_mesh = MpmdMesh(
            jax.sharding.Mesh(devices[:, :1], axis_names=("mpmd", "x")), "mpmd"
        )
        sender_mesh, receiver_mesh = mpmd_mesh.unstack
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, P("x"))
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, P("x"))

        global_shape = (1024 * 1024,)
        aval = jax.ShapeDtypeStruct(global_shape, jnp.float32)
        client = get_distributed_client()

        if process_index == 0:
            first_payload = jax.device_put(
                jnp.full(global_shape, 1.0, dtype=jnp.float32), sender_sharding
            )
            second_payload = jax.device_put(
                jnp.full(global_shape, 2.0, dtype=jnp.float32), sender_sharding
            )

            start_transfer([first_payload], [receiver_sharding], [], []).done()
            client.wait_at_barrier(
                f"{self._testMethodName}_after_first_read_launch", 10000
            )
            start_transfer([second_payload], [receiver_sharding], [], []).done()
            return

        with mpmd_mesh:
            tok, first_recv_buffer = transfer_start_p.bind(
                send_remote_shardings=(),
                send_local_shardings=(),
                recv_remote_shardings=(sender_sharding,),
                recv_local_shardings=(receiver_sharding,),
                out_avals=(aval,),
            )
            (first_received,) = recv_done_p.bind(tok, first_recv_buffer)

            delay_matrix = jax.device_put(
                jnp.eye(_ASYNC_READ_DELAY_MATRIX_SIZE, dtype=jnp.bfloat16),
                receiver_sharding,
            )
            delayed_read = _delayed_read.lower(first_received, delay_matrix).compile()
            delayed_output = delayed_read(first_received, delay_matrix)

            client.wait_at_barrier(
                f"{self._testMethodName}_after_first_read_launch", 10000
            )

            tok, second_recv_buffer = transfer_start_p.bind(
                send_remote_shardings=(),
                send_local_shardings=(),
                recv_remote_shardings=(sender_sharding,),
                recv_local_shardings=(receiver_sharding,),
                out_avals=(aval,),
            )
            (second_received,) = recv_done_p.bind(tok, second_recv_buffer)

        np.testing.assert_array_equal(np.asarray(second_received[:8]), np.full(8, 2.0))

        # Regression check for recv-buffer reuse: if the second recv wrote
        # into first_received's storage, this delayed read would observe 2s.
        delayed_values = np.asarray(delayed_output[:8])
        np.testing.assert_array_less(delayed_values, np.full(8, 1.5))
        np.testing.assert_array_less(np.full(8, 0.5), delayed_values)

    @unittest.expectedFailure
    def test_reusing_recv_buffer_waits_for_prior_async_read(self):
        # Without a reuse fence, reusing a received buffer as the destination for
        # a later DIME recv does not wait for prior PJRT usage events.
        self._run_reusing_recv_buffer_waits_for_prior_async_read(add_fence=False)

    def test_reusing_recv_buffer_with_fence_waits_for_prior_async_read(self):
        self._run_reusing_recv_buffer_waits_for_prior_async_read(add_fence=True)

    def _run_reusing_recv_buffer_waits_for_prior_async_read(self, *, add_fence: bool):
        if jax.process_count() != 2:
            self.skipTest("Test requires exactly two processes.")

        process_index = jax.process_index()
        local_device_count = jax.local_device_count()
        devices = np.array(jax.devices()).reshape(2, local_device_count)
        sender_mesh = jax.sharding.Mesh(devices[0:1], axis_names=("mpmd", "x"))
        receiver_mesh = jax.sharding.Mesh(devices[1:2], axis_names=("mpmd", "x"))
        sender_sharding = jax.sharding.NamedSharding(sender_mesh, P("x"))
        receiver_sharding = jax.sharding.NamedSharding(receiver_mesh, P("x"))

        global_shape = (1024 * 1024,)
        client = get_distributed_client()
        result_key = f"{self._testMethodName}_preserved_first_recv"

        if process_index == 0:
            first_payload = jax.device_put(
                jnp.full(global_shape, 1.0, dtype=jnp.float32), sender_sharding
            )
            second_payload = jax.device_put(
                jnp.full(global_shape, 2.0, dtype=jnp.float32), sender_sharding
            )

            start_transfer([first_payload], [receiver_sharding], [], []).done()
            client.wait_at_barrier(
                f"{self._testMethodName}_after_first_read_launch", 10000
            )
            start_transfer([second_payload], [receiver_sharding], [], []).done()
            preserved = client.blocking_key_value_get_bytes(result_key, 10000)
        else:
            buffer = jax.device_put(
                jnp.zeros(global_shape, dtype=jnp.float32), receiver_sharding
            )
            transfer = start_transfer([], [], [buffer], [sender_sharding])
            (first_received,) = transfer.done()

            delay_matrix = jax.device_put(
                jnp.eye(_ASYNC_READ_DELAY_MATRIX_SIZE, dtype=jnp.bfloat16),
                receiver_sharding,
            )
            delayed_read = _delayed_read.lower(first_received, delay_matrix).compile()
            delayed_output = delayed_read(first_received, delay_matrix)

            if add_fence:
                # DLPack export waits on definition events only, so buffer reuse
                # needs a fresh PJRT value ordered after prior async JAX
                # consumers. Without this fence, the second recv can overwrite
                # memory that delayed_output is still reading.
                first_received = reuse_fence_p.bind(first_received)

            client.wait_at_barrier(
                f"{self._testMethodName}_after_first_read_launch", 10000
            )

            transfer = start_transfer([], [], [first_received], [sender_sharding])
            (second_received,) = transfer.done()
            second_values = np.asarray(second_received[:8])
            delayed_values = np.asarray(delayed_output[:8])
            first_recv_preserved = np.all(delayed_values < 1.5) and np.all(
                delayed_values > 0.5
            )
            second_recv_succeeded = np.array_equal(second_values, np.full(8, 2.0))
            preserved = b"1" if first_recv_preserved and second_recv_succeeded else b"0"
            client.key_value_set_bytes(result_key, preserved)

        self.assertEqual(preserved, b"1")


if __name__ == "__main__":
    jppdu.distributed_main(unittest.main)
