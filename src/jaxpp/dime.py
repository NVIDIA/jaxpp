# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple, Protocol, cast

import cupy
import jax
import jax.core
import jax.numpy as jnp
from cupy.cuda.nccl import NCCL_MAX, NCCL_SUM, NcclCommunicator
from cupy.cuda.nccl import get_unique_id as nccl_get_unique_id
from cupy.cuda.nccl import groupEnd as nccl_group_end
from cupy.cuda.nccl import groupStart as nccl_group_start
from jax._src import array
from jax._src.dlpack import to_dlpack
from jax.lib import xla_client as xc

from jaxpp.dlpack import capsule_name, dlpack_nccl_args
from jaxpp.ops import AllReduceDesc, CommDesc, Operator
from jaxpp.types import (
    CommKeyWithNcclId,
    DLPackCapsule,
    GlobalDeviceId,
    NcclId,
    UniqueGlobalDeviceIds,
)
from jaxpp.utils import RichDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_nccl_operator(op: Operator):
    match op:
        case Operator.SUM:
            return NCCL_SUM
        case Operator.MAX:
            return NCCL_MAX
        case _:
            raise ValueError(f"Unknown operator {op}")


class RawShardedArray(NamedTuple):
    aval: Any
    sharding_spec: jax.sharding.Sharding
    arrays: list[tuple[DLPackCapsule, jax.Device, cupy.cuda.Stream]]
    events: Any = None

    def record_events(self) -> "RawShardedArray":
        events = []
        for _, dev, stream in self.arrays:
            with cupy.cuda.Device(dev.local_hardware_id):
                events.append(stream.record())
        return self._replace(events=events)

    def synchronize(self):
        assert self.events is not None
        for e in self.events:
            e.synchronize()

    def to_jax_array(self):
        jax_single_arrays = []
        for (arr, device, _), event in jax.util.safe_zip(self.arrays, self.events):
            with cupy.cuda.Device(device.local_hardware_id):
                ready_events_stream = device.get_stream_for_external_ready_events()
                cupy.cuda.ExternalStream(ready_events_stream).wait_event(event)
                jax_sda = jnp.asarray(
                    jax._src.lib.xla_client._xla.dlpack_managed_tensor_to_buffer(
                        arr,
                        device,
                        ready_events_stream,
                    )
                )
                jax_single_arrays.append(jax_sda)

        return sharded_array_from_single_arrays(
            self.aval, self.sharding_spec, jax_single_arrays
        )


def sharded_array_from_single_arrays(
    aval: jax.core.ShapedArray, sharding: jax.sharding.Sharding, arrays
):
    res = array.ArrayImpl(aval, sharding, arrays, committed=True)
    return res


def device_buffers(arr: array.ArrayImpl) -> list[array.ArrayImpl | None]:
    return [x.data for x in arr.addressable_shards]


def make_stream(device):
    with cupy.cuda.Device(device.local_hardware_id):
        return cupy.cuda.Stream(non_blocking=True)


class Dime:
    def __init__(self, backend: xc.Client | None = None):
        self.nccl_ids = {}
        self.communicators: dict[
            UniqueGlobalDeviceIds, dict[GlobalDeviceId, NcclCommunicator]
        ] = defaultdict(dict)
        self.backend = backend

        self.send_streams = dict[int, RichDict[int, cupy.cuda.Stream]]()
        self.recv_streams = dict[int, RichDict[int, cupy.cuda.Stream]]()

        self.local_devices_by_id = {
            d.id: d for d in (self.backend or jax).local_devices()
        }
        for d in self.local_devices_by_id.values():
            with cupy.cuda.Device(d.local_hardware_id):
                self.send_streams[d.id] = RichDict[int, cupy.cuda.Stream]()
                self.recv_streams[d.id] = RichDict[int, cupy.cuda.Stream]()

    def communicator_nccl_id(self, key: UniqueGlobalDeviceIds) -> NcclId:
        res = self.nccl_ids.get(key, None)
        if res is None:
            res = nccl_get_unique_id()
            self.nccl_ids[key] = res
        return res

    def register_gpus(self, comm_key: CommKeyWithNcclId):
        if comm_key.device_ids in self.communicators:
            return

        comms = self.communicators[comm_key.device_ids]
        import os

        os.environ["NCCL_COLLNET_ENABLE"] = comm_key.nccl_collnet_enable

        nccl_group_start()
        for gid, rank in comm_key.device_ids.ranks:
            if dev := self.local_devices_by_id.get(gid):
                # NOTE: `setDevice` necessary otherwise call fails
                # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-3-multiple-devices-per-thread
                with cupy.cuda.Device(dev.local_hardware_id):
                    comms[gid] = NcclCommunicator(
                        len(comm_key.device_ids), comm_key.nccl_id, rank
                    )
        nccl_group_end()

    def send(self, arrays: list[tuple[jax.Array, CommDesc]]):
        """
        Asynchronously sends a jax array.
        If the jax array needs to be deleted or updated in place the user must first
        perform `sync_send_streams` on this dime object.
        """

        sends = []

        for arr, comm_desc in arrays:
            dst_dev_ids = dict(
                jax.util.safe_zip(comm_desc.from_dev_ids, comm_desc.to_dev_ids)
            )
            for shard in device_buffers(arr):
                if shard is None:
                    raise ValueError("`shard` is None")

                (src_dev,) = shard.devices()
                dst_dev_id = dst_dev_ids[src_dev.id]
                key = UniqueGlobalDeviceIds.strict_create((src_dev.id, dst_dev_id))
                comm = self.communicators[key][src_dev.id]

                send_stream = self.send_streams[src_dev.id].get_or_else_update(
                    dst_dev_id, partial(make_stream, device=src_dev)
                )
                dlpack = to_dlpack(shard, stream=send_stream.ptr)
                data_ptr, count, dtype = dlpack_nccl_args(dlpack)

                logger.debug(f"key({key}) send {src_dev.id} -({count})> {dst_dev_id}")
                sends.append(
                    (
                        src_dev,
                        comm,
                        data_ptr.value,
                        count,
                        dtype.value,
                        key.rank_of(dst_dev_id),
                        send_stream.ptr,
                    )
                )

        nccl_group_start()
        for e in sends:
            with cupy.cuda.Device(e[0].local_hardware_id):
                e[1].send(*e[2:])
        nccl_group_end()

        # TODO: this is unsafe if `send` is used inside a group scope since these events
        #  might not record the end of the group calls dispatched above.
        #  Add a `send_in_group` similarly to `recv_in_group` below that performs
        #  event recording outside a group scope

        comm_desc = arrays[0][1]
        with cupy.cuda.Device(src_dev.local_hardware_id):
            event = send_stream.record()

        for arr, _ in arrays:
            arr.jaxpp_use_event = event

    def recv_in_group(
        self, arrays: list[tuple[jax.Array, CommDesc]]
    ) -> list[RawShardedArray]:
        """
        Asynchronously receives into the `arrays` passed as argument by consuming
        them and returning the corresponding arrays.
        """
        raw_sharded_arrays = list[RawShardedArray]()

        recvs = []

        for arr, comm_desc in arrays:
            src_dev_ids = dict(
                jax.util.safe_zip(comm_desc.to_dev_ids, comm_desc.from_dev_ids)
            )
            dlpack_capsules_and_devices = list[
                tuple[DLPackCapsule, jax.Device, cupy.cuda.Stream]
            ]()
            for shard in device_buffers(arr):
                if shard is None:
                    raise ValueError("`shard` is None")

                (dst_dev,) = shard.devices()
                src_dev_id = src_dev_ids[dst_dev.id]
                key = UniqueGlobalDeviceIds.strict_create((src_dev_id, dst_dev.id))
                comm = self.communicators[key][dst_dev.id]

                recv_stream = self.recv_streams[dst_dev.id].get_or_else_update(
                    src_dev_id, partial(make_stream, device=dst_dev)
                )
                # NOTE: taking ownership synchronizes host https://github.com/openxla/xla/blob/27c28c8dcfd984ec5ab976d62ac98d0f6afe0ba5/xla/python/dlpack.cc#L290-L295C15
                b = to_dlpack(shard, stream=recv_stream.ptr)
                dlpack_capsules_and_devices.append((b, dst_dev, recv_stream))

                data_ptr, count, dtype = dlpack_nccl_args(b)
                logger.debug(f"key({key}) recv {dst_dev.id} <({count})- {src_dev_id}")
                recvs.append(
                    (
                        dst_dev,
                        comm,
                        data_ptr.value,
                        count,
                        dtype.value,
                        key.rank_of(src_dev_id),
                        recv_stream.ptr,
                    )
                )

            raw_sharded_arrays.append(
                RawShardedArray(arr.aval, arr.sharding, dlpack_capsules_and_devices)
            )

        nccl_group_start()
        for recv in recvs:
            with cupy.cuda.Device(recv[0].local_hardware_id):
                recv[1].recv(*recv[2:])
        nccl_group_end()

        raw_sharded_arrays_with_events = list[RawShardedArray]()
        for csa in raw_sharded_arrays:
            raw_sharded_arrays_with_events.append(csa.record_events())
        return raw_sharded_arrays_with_events

    def all_reduce(self, arrays: Sequence[tuple[jax.Array, AllReduceDesc]]):
        raw_sharded_arrays = list[RawShardedArray]()

        reduces = []

        for arr, desc in arrays:
            dlpack_capsules_and_devices = list[
                tuple[DLPackCapsule, jax.Device, cupy.cuda.Stream]
            ]()
            for shard_to_reduce in device_buffers(arr):
                if shard_to_reduce is None:
                    raise ValueError("`shard` is None")

                (dst_dev,) = shard_to_reduce.devices()

                dst_dev_id = cast(GlobalDeviceId, dst_dev.id)
                comm = self.communicators[desc.within_dev_ids[dst_dev_id]][dst_dev_id]

                recv_stream = self.recv_streams[dst_dev.id].get_or_else_update(
                    -1, partial(make_stream, device=dst_dev)
                )

                b = to_dlpack(shard_to_reduce, stream=recv_stream.ptr)
                dlpack_capsules_and_devices.append((b, dst_dev, recv_stream))

                data_ptr, count, dtype = dlpack_nccl_args(b)

                reduces.append(
                    (
                        dst_dev,
                        comm,
                        data_ptr.value,
                        data_ptr.value,
                        count,
                        dtype.value,
                        to_nccl_operator(desc.operator),
                        recv_stream.ptr,
                    )
                )

            raw_sharded_arrays.append(
                RawShardedArray(arr.aval, arr.sharding, dlpack_capsules_and_devices)
            )

        nccl_group_start()
        for reduce in reduces:
            with cupy.cuda.Device(reduce[0].local_hardware_id):
                reduce[1].allReduce(*reduce[2:])
        nccl_group_end()

        raw_sharded_arrays_with_events = list[RawShardedArray]()
        for csa in raw_sharded_arrays:
            raw_sharded_arrays_with_events.append(csa.record_events())
        return raw_sharded_arrays_with_events

    def recv(
        self,
        arrays: list[tuple[jax.Array, CommDesc]],
    ) -> list[jax.Array]:
        return [a.to_jax_array() for a in self.recv_in_group(arrays)]

    def __enter__(self):
        nccl_group_start()

    def __exit__(self, exc_type, exc_value, traceback):
        nccl_group_end()


class WithDime(Protocol):
    @property
    def comm(self) -> Dime: ...


class DimeMixin:
    def communicator_nccl_id(self: WithDime, key: UniqueGlobalDeviceIds) -> NcclId:
        return self.comm.communicator_nccl_id(key)

    def register_gpus(self: WithDime, comm_key: CommKeyWithNcclId):
        return self.comm.register_gpus(comm_key)

    def send(self: WithDime, arrays: list[tuple[jax.Array, CommDesc]]):
        return self.comm.send(arrays)

    def recv_in_group(
        self: WithDime,
        arrays: list[tuple[jax.Array, CommDesc]],
    ):
        return self.comm.recv_in_group(arrays)

    def recv(
        self: WithDime,
        arrays: list[tuple[jax.Array, CommDesc]],
    ):
        return self.comm.recv(arrays)

    def all_reduce(
        self: WithDime,
        arrays: Sequence[tuple[array.ArrayImpl, AllReduceDesc]],
    ):
        return self.comm.all_reduce(arrays)
