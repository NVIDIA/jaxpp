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

import dataclasses
import importlib
import itertools as it
import logging
import os
import pickle
import subprocess
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import jax
import jax.core
import jax.experimental.topologies as topologies
import jax_cuda12_plugin._versions as cuda_versions
import jaxlib.xla_extension as xe
import numpy as np
import ray
from cupy.cuda.nccl import NcclCommunicator
from cupy.cuda.nccl import get_unique_id as nccl_get_unique_id
from cupy.cuda.nccl import groupEnd as nccl_group_end
from cupy.cuda.nccl import groupStart as nccl_group_start
from jax._src import dtypes, xla_bridge
from jax._src.distributed import initialize as dist_init
from jax._src.distributed import shutdown as dist_shutdown
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
from jax.interpreters import pxla
from jax.lib import xla_client as xc
from ray.air._internal.util import find_free_port
from ray.exceptions import RayError
from ray.runtime_env import RuntimeEnv

from jaxpp.compilation import SerializeableMeshComputation, to_pspec
from jaxpp.dime import Dime, DimeMixin, RawShardedArray
from jaxpp.ops import (
    UID,
    AllReduceOp,
    DeleteOp,
    Op,
    RecvOp,
    RunOp,
    SendOp,
)
from jaxpp.serialization import ArrayRefAsyncCheckpointManager
from jaxpp.types import (
    Bind,
    CommKeyWithNcclId,
    GlobalDeviceId,
    PutArg,
    UniqueGlobalDeviceIds,
    UniqueSortedSequence,
    WorkerId,
)

logger = logging.getLogger(__name__)


def live_arrays_size(backend: xc.Client) -> float:
    bytes_used_list = []
    for dev in backend.local_devices():
        mem_stats = dev.memory_stats()
        if mem_stats is not None:
            bytes_used_list.append(mem_stats["bytes_in_use"])
    return sum(bytes_used_list) / (2**30)


@dataclass
class Ctx:
    coordinator_address: str
    num_processes: int
    gpus_per_process: int
    import_modules: Sequence[str]
    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    event_sync_before_delete: bool
    log_level: int


def initialize_backend(ctx: Ctx) -> jax.sharding.Mesh:
    devices = jax.devices()
    remote_devices = np.array(
        sorted((d for d in devices), key=lambda d: (d.process_index, d.id))
    ).reshape(ctx.mesh_shape)
    return jax.sharding.Mesh(remote_devices, ctx.mesh_axis_names)


def get_pspec(a: jax.Array, lmesh: jax.sharding.Mesh) -> jax.sharding.PartitionSpec:
    assert isinstance(a.sharding, jax.sharding.Sharding)
    hlo_sharding = a.sharding._to_xla_hlo_sharding(a.ndim)
    return to_pspec(hlo_sharding, lmesh)


# Identity function is at the top level so that `GPUWorker.replicate` doesn't
# recompile on every invocation.
def _identity_fn(x):
    return x


class Store:
    def __init__(
        self,
        event_sync_before_delete: bool = False,
        _logger: logging.Logger | None = None,
    ):
        self._executables = dict[UID, Any]()
        self._arrays = dict[UID, jax.Array | RawShardedArray]()
        self._event_sync_before_delete = event_sync_before_delete
        self._logger = _logger or logging.getLogger(__name__)
        if False:
            # Multicontroller debugging
            self._logger.setLevel(logging.DEBUG)

        self._pending_deletes = deque()

    def pop_array(self, uid: UID) -> jax.Array:
        elem = self._arrays.pop(uid)
        if isinstance(elem, RawShardedArray):
            elem = elem.to_jax_array()
        return elem

    def get_array(self, uid: UID) -> jax.Array:
        elem = self._arrays[uid]
        if isinstance(elem, RawShardedArray):
            elem = elem.to_jax_array()
            self._arrays[uid] = elem
        return elem

    def put_array(self, uid: UID, array: jax.Array | RawShardedArray) -> None:
        self._arrays[uid] = array

    def synchronize_use(self, arr: jax.Array) -> None:
        if (event := getattr(arr, "jaxpp_use_event", None)) is not None:
            event.synchronize()

    def collect_pending_deletes(self, wait: bool = False) -> None:
        if not wait:
            remaining = []
            for arr in self._pending_deletes:
                if (event := arr.jaxpp_use_event) and event.done:
                    arr.delete()
                    del arr
                else:
                    remaining.append(arr)
            self._pending_deletes = remaining
        else:
            jax.util.safe_map(self.synchronize_use, self._pending_deletes)
            self._pending_deletes = []

    def maybe_async_delete_array(self, uid: UID) -> None:
        arr: jax.Array = self.pop_array(uid)
        event = getattr(arr, "jaxpp_use_event", None)

        if self._event_sync_before_delete:
            self.synchronize_use(arr)
            arr.delete()
            del arr

        elif event:
            if event.done:
                arr.delete()
                del arr
            else:
                self._pending_deletes.append(arr)
        else:
            arr.delete()
            del arr

    def put_mesh_computation(
        self, uid: UID, mesh_executable: pxla.MeshExecutable, name: str
    ) -> None:
        in_shardings = jax.util.safe_zip(
            mesh_executable.in_avals, mesh_executable.input_shardings()
        )
        self._logger.debug(f"{name} {in_shardings=}")
        out_shardings = jax.util.safe_zip(
            mesh_executable.out_avals, mesh_executable.output_shardings()
        )
        self._logger.debug(f"{name} {out_shardings=}")
        cpp_call = mesh_executable.create_cpp_call(
            no_kwargs=True,
            in_tree=jax.tree_util.tree_structure(
                (tuple(mesh_executable.input_shardings()), {})
            ),
            out_tree=jax.tree_util.tree_structure(mesh_executable.output_shardings()),
        )
        assert cpp_call is not None
        mesh_executable.cpp_call = cpp_call

        # TODO: use mesh_executable.as_text() or directly the hlo module
        # to compute number of collectives as proxy metric for intra-stage
        # communication

        cas = mesh_executable.cost_analysis()

        # TODO
        # mflops = cas[0].get("flops", 0.0) / (10**6)
        # self._logger.info(
        #     f"{serializeable_mesh_computation.name:20} "
        #     f"(Module 1 of {len(cas)}): {mflops:6.2f} MFLOPS"
        # )

        # TODO Memory analysis not very helpful as of 05/25/2023
        # mesh_executable.memory_analysis()

        self._executables[uid] = mesh_executable

    def get_executable(self, uid: UID) -> pxla.MeshExecutable:
        return self._executables[uid]


def zeros(avals):
    return tuple(jax.numpy.zeros(aval.shape, aval.dtype) for aval in avals)


def execute_op(
    comm: Dime, store: Store, op: Op, local_mesh: jax.sharding.Mesh
) -> list[jax.Array] | list[RawShardedArray]:
    args = [store.get_array(uid) for uid in op.in_uids]
    results: list[jax.Array] | list[RawShardedArray] = []
    match op:
        case RunOp():
            executable = store.get_executable(op.exec_uid)
            assert isinstance(
                executable, pxla.MeshExecutable
            ), f"Unexpected executable {op.exec_uid} type {type(executable)}"

            results = executable.cpp_call(*args)

        case SendOp():
            store.collect_pending_deletes()
            # NOTE `send` sets the `jaxpp_use_event` attribute
            comm.send(list(zip(args, op.comm_desc, strict=True)))
            results = args

        case RecvOp():
            avals = tuple(desc.aval for desc in op.comm_desc)
            shardings = tuple(
                desc.sharding.to_named_sharding(local_mesh) for desc in op.comm_desc
            )
            args = jax.jit(zeros, out_shardings=shardings, static_argnums=(0,))(avals)
            results = comm.recv_in_group(list(zip(args, op.comm_desc, strict=True)))
            # NOTE: Recv consumes its arguments
            del args

        case AllReduceOp():
            results = comm.all_reduce(list(zip(args, op.descs, strict=True)))
            # NOTE: AllReduce consumes its arguments
            del args

        case DeleteOp():
            for uid in op.in_uids:
                store.maybe_async_delete_array(uid)

        case _:
            raise TypeError(f"Unknown operation type: {type(op)}")

    if len(op.out_uids) > 0:
        # NOTE: for `SendOp` we
        jax.util.safe_map(store.put_array, op.out_uids, results)

    return results


def execute_instructions(
    comm: Dime,
    store: Store,
    in_binding: list[Bind],
    ops: list[Op],
    out_binding: list[Bind],
    deletions: Sequence[UID],
    local_mesh: jax.sharding.Mesh,
    block_on_output: bool = False,
) -> jax.Array | None:
    store.collect_pending_deletes(wait=True)

    last_run_result = None
    is_debug = store._logger.isEnabledFor(logging.DEBUG)
    block_on_output = block_on_output or is_debug
    start = time.time()

    for binding in in_binding:
        store.put_array(binding.to_, store.get_array(binding.from_))

    for idx, op in enumerate(ops):
        op_start = time.time()
        name = ""
        is_run_op = isinstance(op, RunOp)
        if is_run_op:
            name = store.get_executable(op.exec_uid).unsafe_call.name

        if store._logger.getEffectiveLevel() <= logging.DEBUG:
            store._logger.log(
                logging.DEBUG,
                f"{idx:5}/{len(ops)} {op.__class__.__name__} {name} "
                f"Start - {live_arrays_size(comm.backend):.2f}GiB",
            )

        try:
            results = execute_op(comm, store, op, local_mesh)
            if is_run_op:
                # FIXME: this might be a deleted array
                # Use a local store for executing the instructions and then
                # update global store as needed
                last_run_result: jax.Array = results[-1]
        except Exception as e:
            raise RuntimeError(f"Failed while running {name=} {op}") from e

        if block_on_output:
            for jarray_or_raw in results:
                if isinstance(jarray_or_raw, RawShardedArray):
                    store._logger.debug("Synchronizing on RawShardedArray")
                    jarray_or_raw.synchronize()
                else:
                    store._logger.debug("Synchronizing on block_until_ready")
                    jax.block_until_ready(jarray_or_raw)
                    if (
                        event := getattr(jarray_or_raw, "jaxpp_use_event", None)
                    ) is not None:
                        store._logger.debug("Synchronizing on jaxpp_use_event")
                        event.synchronize()

        if store._logger.getEffectiveLevel() <= logging.DEBUG:
            store._logger.log(
                logging.DEBUG,
                f"{idx:5}/{len(ops)} {op.__class__.__name__} {name} "
                f"Done in {time.time() - op_start:.4f}s - "
                f"{live_arrays_size(comm.backend):.2f}GiB",
            )

    for uid in deletions:
        store.pop_array(uid)

    for binding in out_binding:
        store.put_array(binding.to_, store.get_array(binding.from_))

    store._logger.log(
        logging.DEBUG,
        f"{len(ops)} Instructions done in {time.time() - start:.4f}s",
    )
    return last_run_result


class GPUWorker(DimeMixin):
    def __init__(self, ctx: Ctx, process_id: int, coordinates: tuple[int, ...]):
        self.process_id = process_id

        logging.basicConfig(level=ctx.log_level)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info("Trying to connect to distributed client")

        dist_init(
            coordinator_address=ctx.coordinator_address,
            num_processes=ctx.num_processes,
            process_id=process_id,
            local_device_ids=list(range(ctx.gpus_per_process)),
        )

        self._logger.info(f"{coordinates} Connected")

        self.coordinates = coordinates
        self.gmesh = initialize_backend(ctx)
        self.pp_idx = self.coordinates[0]
        self.lmesh = jax.sharding.Mesh(
            self.gmesh.devices[self.pp_idx], self.gmesh.axis_names[1:]
        )

        for mod in ctx.import_modules:
            importlib.import_module(mod)

        from jax._src.distributed import global_state

        assert isinstance(global_state.client, xe.DistributedRuntimeClient)
        self.distributed_client = global_state.client
        self.backend: xc.Client = jax.local_devices()[0].client
        self.comm = Dime(self.backend)

        # NOTE: maybe useful for self.distributed_client.wait_at_barrier("this_barrier")

        self.store = Store(ctx.event_sync_before_delete, _logger=self._logger)
        self.replicate = jax.jit(
            _identity_fn,
            out_shardings=jax.NamedSharding(self.lmesh, jax.sharding.PartitionSpec()),  # type: ignore
        )

    def start_trace(self, profiler_trace_path):
        self._logger.debug(f"Starting trace {profiler_trace_path}")
        jax.profiler.start_trace(profiler_trace_path)

    def stop_trace(self):
        jax.profiler.stop_trace()

    def put_tensors(self, arrays: list[PutArg]):
        for arr in arrays:
            if self.pp_idx not in arr.mpmd_idxs:
                continue

            t = arr.value
            if dtypes.is_python_scalar(t):
                t = dtypes.coerce_to_array(t)

            sharding = arr.sharding.to_named_sharding(self.lmesh)
            device_ts = [
                jax.device_put(t[idx], d)
                for d, idx in sharding.addressable_devices_indices_map(t.shape).items()
            ]
            ga = jax.make_array_from_single_device_arrays(t.shape, sharding, device_ts)
            self.store.put_array(arr.uid, ga)

    def delete(self, uid: UID) -> None:
        self.store.maybe_async_delete_array(uid)

    def put_mesh_computation(
        self, uid: UID, serializeable_mesh_computation: SerializeableMeshComputation
    ) -> None:
        self._logger.info(
            f"Compiling xla computation {serializeable_mesh_computation.name}"
        )
        xla_dump_to = None
        if serializeable_mesh_computation.compiler_options is not None:
            xla_dump_to = serializeable_mesh_computation.compiler_options.get(
                "xla_dump_to"
            )
        mesh_executable = serializeable_mesh_computation.to_mesh_executable(self.lmesh)

        if xla_dump_to is not None:
            glob_pattern = (
                f"*{serializeable_mesh_computation.name}*buffer-assignment.txt"
            )
            buffer_assignments = list(Path(xla_dump_to).glob(glob_pattern))
            if len(buffer_assignments) == 0:
                self._logger.warning(
                    "Compilation did not result in buffer assignment dump"
                )
            elif len(buffer_assignments) == 1:
                ba_pattern = "Total bytes used: "
                [ba] = buffer_assignments
                with ba.open() as oba:
                    matched = False
                    for line in oba:
                        if line.startswith(ba_pattern):
                            size = line.strip()[len(ba_pattern) :]
                            self._logger.info(
                                "buffer-assignment "
                                f"{serializeable_mesh_computation.name}: {size}"
                            )
                            matched = True
                            break
                    if not matched:
                        self._logger.warning(
                            "Pattern did not match buffer assignment pattern: "
                            f"{ba_pattern} for file {ba}"
                        )
            else:
                paths = "".join(f"\n\t{ba}" for ba in buffer_assignments)
                self._logger.warning(f"Multiple buffer assignments found: {paths}")

        self.store.put_mesh_computation(
            uid, mesh_executable, serializeable_mesh_computation.name
        )

    def mpmd_group_allgather(self, uid, return_value=False):
        res = self.replicate(self.store.get_array(uid))
        if return_value:
            return res
        return None

    def save_arrays(self, uids, path):
        arrays, paths = map(
            list,
            zip(
                *[(self.store.get_array(uid), f"{path}/{uid}") for uid in uids],
                strict=True,
            ),
        )
        tspecs = jax.tree_util.tree_map(get_tensorstore_spec, paths)
        ckpt_manager = ArrayRefAsyncCheckpointManager()
        ckpt_manager.serialize(arrays, tspecs, on_commit_callback=lambda: None)
        ckpt_manager.wait_until_finished()

    def load_arrays(self, shardings, path):
        device_assignment = tuple(self.lmesh.devices.flat)
        uids, shardings, paths = map(
            list,
            zip(
                *[
                    (
                        uid[1],  # new mubatch_uid
                        jax.sharding.GSPMDSharding(device_assignment, sharding),
                        f"{path}/{uid[0]}",  # old mubatch_uid
                    )
                    for sharding, uid in shardings
                ],
                strict=True,
            ),
        )
        tspecs = jax.tree_util.tree_map(get_tensorstore_spec, paths)
        ckpt_manager = ArrayRefAsyncCheckpointManager()
        arrays = ckpt_manager.deserialize(shardings, tspecs)
        for uid, array in zip(uids, arrays, strict=True):
            self.store.put_array(uid, array)

    def execute_instructions(
        self,
        in_binding: list[Bind],
        ops: list[Op],
        out_binding: list[Bind],
        deletions: Sequence[UID],
        block_on_output=False,
    ):
        execute_instructions(
            self.comm,
            self.store,
            in_binding,
            ops,
            out_binding,
            deletions,
            local_mesh=self.lmesh,
            block_on_output=block_on_output,
        )

    def memory_stats(self):
        device_stats = self.backend.local_devices()[0].memory_stats()

        if device_stats is not None:
            pool_size = device_stats["pool_bytes"] / (2**30)
            peak_size = device_stats["peak_bytes_in_use"] / (2**30)
            largest_alloc_size = device_stats["largest_alloc_size"] / (2**30)
            self._logger.info(
                "GPU memory "
                f"peak size: {peak_size:.2f} GiB, "
                f"largest alloc size: {largest_alloc_size:.2f} GiB, "
                f"pool size: {pool_size:.2f} GiB"
            )

    def __repr__(self):
        return f"JaxWorker(process_id={self.process_id}, coords={self.coordinates})"

    def shutdown(self):
        dist_shutdown()


def process_ids(devs: Iterable[jax.Device]) -> UniqueSortedSequence[WorkerId]:
    return UniqueSortedSequence.create(WorkerId(d.process_index) for d in devs)


def mesh_process_ids(pjit_mesh: jax.sharding.Mesh):
    return process_ids(pjit_mesh.devices.flat)


def get_cluster_gpus():
    gpus_in_node = [n["Resources"]["GPU"] for n in ray.nodes() if n["Alive"]]
    return int(max(gpus_in_node)), int(sum(gpus_in_node))


def split_mesh_shape(
    mesh_shape: tuple[int, ...], max_gpus_in_node: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Partitions the mesh_shape into a prefix `pref` and the largest suffix `suff`
    such that `mesh_shape == (*pref, *suff) and np.prod(suff) <= max_gpus_in_node`.
    Raises an `AssertionError` if `mesh_shape[-1] > max_gpus_in_node`.

    >>> split_mesh_shape((5, 2), 3)
    ((5,), (2,))
    >>> split_mesh_shape((4, 2, 2), 4)
    ((4,), (2, 2))
    >>> split_mesh_shape((2, 1, 8), 16)
    ((), (2, 1, 8))
    >>> split_mesh_shape((1, 1, 4), 2)
    Traceback (most recent call last):
        ...
    AssertionError: mesh_shape[-1]=4 > max_gpus_in_node=2

    """
    if mesh_shape[-1] > max_gpus_in_node:
        # TODO: maybe allow the innermost dimension to span multiple
        #  nodes
        raise AssertionError(f"{mesh_shape[-1]=} > {max_gpus_in_node=}")

    process_submesh_start_idx = len(mesh_shape) - 1
    while (
        process_submesh_start_idx - 1 >= 0
        and np.prod(mesh_shape[process_submesh_start_idx - 1 :]) <= max_gpus_in_node
    ):
        process_submesh_start_idx -= 1

    return mesh_shape[:process_submesh_start_idx], mesh_shape[
        process_submesh_start_idx:
    ]


def get_distributed_client() -> xe.DistributedRuntimeClient:
    from jax._src.distributed import global_state

    assert isinstance(global_state.client, xe.DistributedRuntimeClient)
    return global_state.client


local_comms = {}


class UniqueDevices(tuple[jax.Device, ...]):
    def __new__(cls, *args):
        unique = set(args)
        assert len(unique) == len(args)
        return super().__new__(cls, sorted(unique, key=lambda d: d.id))

    @cached_property
    def ranks(self):
        return OrderedDict((d, idx) for idx, d in enumerate(self))

    @property
    def leader(self):
        return self[0]

    @cached_property
    def key(self) -> str:
        return ",".join(str(d.id) for d in self)


def get_nccl_id(devs: UniqueDevices):
    TIMEOUT = 240_000  # FIXME: make it an argument
    if devs.leader.process_index == jax.process_index():
        nccl_id = nccl_get_unique_id()
        get_distributed_client().key_value_set_bytes(devs.key, pickle.dumps(nccl_id))
    else:
        nccl_id = get_distributed_client().blocking_key_value_get_bytes(
            devs.key, TIMEOUT
        )
        nccl_id = pickle.loads(nccl_id)
    return nccl_id


@dataclass(frozen=True)
class MpmdMesh:
    jax_mesh: jax.sharding.Mesh
    mpmd_axis_name: str
    store: Store = dataclasses.field(default_factory=Store)
    dime: Dime = dataclasses.field(
        default_factory=lambda: Dime(jax.local_devices()[0].client)
    )
    strict: bool = True

    def __post_init__(self):
        if self.strict:
            mpmd_idx_by_process = dict[int, int]()
            for d in self.jax_mesh._flat_devices_set:
                if (mpmd_idx := mpmd_idx_by_process.get(d.process_index)) is not None:
                    if self.device_coords[d][self.mpmd_axis] != mpmd_idx:
                        raise AssertionError(
                            f"Process {d.process_index} found in two mpmd indices: {mpmd_idx} {self.device_coords[d]}"
                        )
                else:
                    mpmd_idx_by_process[d.process_index] = self.device_coords[d][
                        self.mpmd_axis
                    ]

    @cached_property
    def device_coords(self) -> Mapping[jax.Device, tuple[int, ...]]:
        return {
            device: coord for coord, device in np.ndenumerate(self.jax_mesh.devices)
        }

    @cached_property
    def mpmd_dim(self):
        return self.jax_mesh.shape[self.mpmd_axis_name]

    @cached_property
    def mpmd_axis(self) -> int:
        return self.jax_mesh.axis_names.index(self.mpmd_axis_name)

    @cached_property
    def my_mpmd_axis_index(self) -> int:
        my_devices_coord = {
            self.device_coords[d][self.mpmd_axis] for d in jax.local_devices()
        }
        (mpmd_axis_index,) = my_devices_coord
        return mpmd_axis_index

    @cached_property
    def my_mpmd_group_mesh(self) -> jax.sharding.Mesh:
        return jax.sharding.Mesh(
            np.expand_dims(
                np.take(self.jax_mesh.devices, self.my_mpmd_axis_index, self.mpmd_axis),
                axis=self.mpmd_axis,
            ),
            self.jax_mesh.axis_names,
        )

    def lowering_mesh(self) -> jax.sharding.Mesh:
        return self.my_mpmd_group_mesh

    @cached_property
    def unstack(self) -> list[jax.sharding.Mesh]:
        axis_names = (
            self.jax_mesh.axis_names[: self.mpmd_axis]
            + self.jax_mesh.axis_names[self.mpmd_axis + 1 :]
        )
        return [
            jax.sharding.Mesh(mpmd_group_devices, axis_names)
            for mpmd_group_devices in np.moveaxis(
                self.jax_mesh.devices, self.mpmd_axis, 0
            )
        ]

    def remote_mesh_at(self, mpmd_index: int) -> jax.sharding.Mesh:
        return self.unstack[mpmd_index]

    def mpmd_submesh(self, mpmd_indices: list[int]) -> "MpmdMesh":
        assert isinstance(mpmd_indices, list)
        jax_mesh = jax.sharding.Mesh(
            np.take(self.jax_mesh.devices, mpmd_indices, self.mpmd_axis),
            self.jax_mesh.axis_names,
        )
        return MpmdMesh(jax_mesh, self.mpmd_axis_name)

    @property
    def as_mpmd_mesh(self) -> "MpmdMesh":
        return self

    def put_mesh_computation(
        self,
        mpmd_idx: int,
        uid: UID,
        serializeable_mesh_computation: SerializeableMeshComputation,
    ) -> None:
        if self.my_mpmd_axis_index == mpmd_idx:
            self.store.put_mesh_computation(
                uid,
                serializeable_mesh_computation.to_mesh_executable(
                    self.my_mpmd_group_mesh
                ),
                serializeable_mesh_computation.name,
            )
        return []

    def establish_nccl_comms(
        self, gidss: Iterable[tuple[UniqueGlobalDeviceIds, str]]
    ) -> None:
        devices_by_id = {d.id: d for d in jax.devices()}
        for global_device_ids, nccl_collnet_enable in gidss:
            devs = UniqueDevices(*(devices_by_id[did] for did in global_device_ids))
            nccl_id = get_nccl_id(devs)

            comm_key = CommKeyWithNcclId(
                global_device_ids, nccl_id, nccl_collnet_enable
            )
            self.dime.register_gpus(comm_key)

    def put_tensors(self, arrays: list[PutArg]):
        for arr in arrays:
            if self.my_mpmd_axis_index not in arr.mpmd_idxs:
                continue
            self.store.put_array(arr.uid, arr.value)
        return []

    @property
    def replicate(self):
        return jax.jit(
            _identity_fn,
            out_shardings=jax.NamedSharding(
                self.my_mpmd_group_mesh, jax.sharding.PartitionSpec()
            ),
        )

    def get_tensor(self, mpmd_idx: int, uid: UID) -> jax.Array:
        self.store._logger.debug("get_tensor")
        if self.my_mpmd_axis_index == mpmd_idx:
            with self.my_mpmd_group_mesh:
                return self.replicate(self.store.get_array(uid))
        # TODO: return

    def delete(self, mpmd_idx: int, uid: UID):
        if self.my_mpmd_axis_index == mpmd_idx:
            self.store.maybe_async_delete_array(uid)

    def execute_instructions(
        self,
        mpmd_idx: int,
        in_binding: list[Bind],
        instructions: list[Op],
        out_binding: list[Bind],
        deletions: Sequence[UID] = (),
    ) -> None:
        if self.my_mpmd_axis_index == mpmd_idx:
            execute_instructions(
                self.dime,
                self.store,
                in_binding=in_binding,
                ops=instructions,
                out_binding=out_binding,
                deletions=deletions,
                local_mesh=self.my_mpmd_group_mesh,
            )
        return []

    def blocking(self, generator: Callable[[], Generator]) -> None:
        generator()

    def blocking_tree(self, tree):
        return tree


# jax.lib.xla_client.get_topology_for_devices(jax.local_devices()).target_config
eos_target_config = """
gpu_device_info {
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: 233472
  threads_per_core_limit: 2048
  core_count: 132
  fpus_per_core: 128
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: 3352320000000
  l2_cache_size: 52428800
  clock_rate_ghz: 1.98
  device_memory_size: 84942979072
  shared_memory_per_block_optin: 232448
  cuda_compute_capability {
    major: 9
  }
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info {
  major: 9
  minor: 6
}
device_description_str: "NVIDIA H100 80GB HBM3"
"""


class RemoteMpmdMesh:
    # Context manager utilities
    _worker_mesh_stack: list["RemoteMpmdMesh"] = []

    _unrecoverable_failure = False

    @property
    def unrecoverable_failure(self):
        return RemoteMpmdMesh._unrecoverable_failure

    @unrecoverable_failure.setter
    def unrecoverable_failure(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError(
                "None boolean value received for `unrecoverable_failure`: "
                f"{type(val)}: {val}"
            )
        RemoteMpmdMesh._unrecoverable_failure = val

    def __enter__(self):
        self._push_current_worker_mesh(self)
        self.lowering_mesh().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lowering_mesh().__exit__(exc_type, exc_value, traceback)
        self._pop_current_worker_mesh()

    @classmethod
    def current_worker_mesh(cls) -> Optional["RemoteMpmdMesh"]:
        if len(cls._worker_mesh_stack) > 0:
            return cls._worker_mesh_stack[-1]
        return None

    @classmethod
    def _push_current_worker_mesh(cls, worker_mesh: "RemoteMpmdMesh"):
        cls._worker_mesh_stack.append(worker_mesh)

    @classmethod
    def _pop_current_worker_mesh(cls) -> "RemoteMpmdMesh":
        return cls._worker_mesh_stack.pop()

    def __init__(
        self,
        pp: int,
        mesh_shape: tuple[int, ...],
        mesh_axis_names: tuple[str, ...],
        *,
        import_modules: Sequence[str] = (),
        event_sync_before_delete: bool = False,
        target_config: str | None = None,  # eos_target_config,
        _env: dict[str, str] | None = None,
    ):
        assert pp > 0
        assert len(mesh_shape) > 0
        assert len(mesh_shape) == len(
            mesh_axis_names
        ), f"{len(mesh_shape) == len(mesh_axis_names)=}"
        for name, sz in zip(mesh_axis_names, mesh_shape, strict=True):
            assert sz > 0, f"{name}={sz}"

        if xla_bridge.backends_are_initialized():
            raise RuntimeError(
                f"{RemoteMpmdMesh.__name__} must be called before any JAX computations are executed."
            )

        # Initialize ray if it hasn't been initialized
        if not ray.is_initialized():
            ray.init("auto")

        max_gpus_in_node, tot_gpus = get_cluster_gpus()
        local_device_count = cuda_versions.cuda_device_count()

        logger.info(f"Detected {local_device_count=}")

        _process_mesh_shape, device_mesh_shape = split_mesh_shape(
            mesh_shape, max_gpus_in_node
        )

        process_mesh_shape = (pp, *_process_mesh_shape)
        num_processes = int(np.prod(process_mesh_shape))
        gpus_per_process = int(np.prod(device_mesh_shape))
        self.full_mesh_shape = (*process_mesh_shape, *device_mesh_shape)

        if target_config is not None:
            topo = topologies.get_topology_desc(
                "topo",
                "cuda",
                target_config=target_config,
                topology=f"{num_processes}x1x{gpus_per_process}",
            )
            self.remote_mesh = jax.sharding.Mesh(
                np.array(topo.devices).reshape((pp, *mesh_shape)),
                ("jaxpp_pp", *mesh_axis_names),
            )
            logger.warning("Running with AOT compilation mesh. No `.workers`.")
            return

        if num_processes * gpus_per_process > tot_gpus:
            raise AssertionError(
                f"Mesh requires {num_processes * gpus_per_process} GPUs but "
                f"only {tot_gpus} found available in the cluster"
            )

        if num_processes > pp:
            logger.warning("Multi-host SPMD detected. This feature is experimental")

        ctx = Ctx(
            coordinator_address=f"{ray.util.get_node_ip_address()}:{find_free_port()}",
            num_processes=num_processes,
            gpus_per_process=gpus_per_process,
            import_modules=import_modules,
            mesh_shape=(pp, *mesh_shape),
            mesh_axis_names=("jaxpp_pp", *mesh_axis_names),
            log_level=logger.getEffectiveLevel(),
            event_sync_before_delete=event_sync_before_delete,
        )

        env_vars = {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": ".94",
            "JAX_COMPILER_DETAILED_LOGGING_MIN_OPS": "0",
        }
        env_vars = {v: os.environ.get(v, env_vars[v]) for v in env_vars}

        if logger.getEffectiveLevel() <= logging.DEBUG:
            env_vars.update(
                {
                    # "NCCL_DEBUG": "INFO",
                    "TF_CPP_MIN_LOG_LEVEL": "0",
                }
            )

        if _env is not None:
            env_vars.update(_env)

        logger.info(f"Environment variables: {env_vars}")

        self.workers = []
        for idx, coords in enumerate(
            it.product(*(range(x) for x in process_mesh_shape))
        ):
            self.workers.append(
                ray.remote(
                    num_cpus=0,
                    num_gpus=ctx.gpus_per_process,
                    runtime_env=RuntimeEnv(env_vars=env_vars),
                )(GPUWorker).remote(ctx, idx, coords)
            )

        jax.config.update("mock_num_gpu_processes", num_processes)

        jax_cuda_visible_devices = jax.config.read("jax_cuda_visible_devices")
        os_environ = os.environ.copy()
        try:
            jax.config.update(
                "jax_cuda_visible_devices",
                ",".join(str(e) for e in range(ctx.gpus_per_process)),
            )
            # The driver might be sharing GPUs with some worker.
            # Here we make sure that it uses the least amount of memory
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

            self.remote_mesh = initialize_backend(ctx)

        finally:
            os.environ.clear()
            os.environ.update(os_environ)
            jax.config.update("jax_cuda_visible_devices", jax_cuda_visible_devices)

    @property
    def devices_by_id(self) -> Mapping[GlobalDeviceId, jax.Device]:
        return {d.id: d for d in self.remote_mesh.devices.flat}

    @property
    def mpmd_dim(self):
        return self.remote_mesh.shape["jaxpp_pp"]

    @property
    def as_mpmd_mesh(self):
        return MpmdMesh(self.remote_mesh, "jaxpp_pp")

    def remote_mesh_at(self, mpmd_idx: int) -> jax.sharding.Mesh:
        return jax.sharding.Mesh(
            self.remote_mesh.devices[mpmd_idx], self.remote_mesh.axis_names[1:]
        )

    def lowering_mesh(self) -> jax.sharding.Mesh:
        return self.remote_mesh_at(0)

    def put_mesh_computation(
        self,
        mpmd_idx: int,
        uid,
        serializeable_mesh_computation: SerializeableMeshComputation,
    ):
        res = []
        remote_mesh = self.remote_mesh_at(mpmd_idx=mpmd_idx)
        for pid in mesh_process_ids(remote_mesh):
            self.workers[pid].put_mesh_computation.remote(
                uid, serializeable_mesh_computation
            )
        return res

    def put_tensors(self, arrays: list[PutArg]):
        arrays = ray.put(arrays)
        return [
            self.workers[pid].put_tensors.remote(arrays)
            for pid in mesh_process_ids(self.remote_mesh)
        ]

    def delete(self, mpmd_idx: int, uid):
        process_ids = mesh_process_ids(self.remote_mesh_at(mpmd_idx))
        return [self.workers[pid].delete.remote(uid) for pid in process_ids]

    def get_tensor(self, mpmd_idx: int, uid):
        pids = mesh_process_ids(self.remote_mesh_at(mpmd_idx))
        p0 = pids[0]
        res = self.workers[p0].mpmd_group_allgather.remote(uid, return_value=True)
        for pid in pids[1:]:
            self.workers[pid].mpmd_group_allgather.remote(uid, return_value=False)
        return res

    def save_arrays(self, mpmd_idx: int, uids, path):
        # FIXME: this will fail for multi-host arrays
        process_ids = mesh_process_ids(self.remote_mesh_at(mpmd_idx))
        return self.workers[process_ids[0]].save_arrays.remote(uids, path)

    def load_arrays(self, mpmd_idx: int, shardings, path):
        # FIXME: this will fail for multi-host arrays
        process_ids = mesh_process_ids(self.remote_mesh_at(mpmd_idx))
        return self.workers[process_ids[0]].load_arrays.remote(shardings, path)

    def execute_instructions(
        self,
        mpmd_idx: int,
        in_binding: list[Bind],
        instructions,
        out_binding: list[Bind],
        deletions: Sequence[UID] = (),
    ):
        process_ids = mesh_process_ids(self.remote_mesh_at(mpmd_idx))
        return [
            self.workers[pid].execute_instructions.remote(
                in_binding, instructions, out_binding, deletions
            )
            for pid in process_ids
        ]

    def establish_nccl_comm(
        self, global_device_ids: UniqueGlobalDeviceIds, nccl_collnet_enable: str
    ):
        nccl_id = self.blocking_tree(
            [
                self.workers[
                    self.devices_by_id[global_device_ids.primary].process_index
                ].communicator_nccl_id.remote(key=global_device_ids)
            ]
        )[0]

        comm_key = CommKeyWithNcclId(global_device_ids, nccl_id, nccl_collnet_enable)

        pids = process_ids(self.devices_by_id[gid] for gid in global_device_ids)
        return [self.workers[pid].register_gpus.remote(comm_key) for pid in pids]

    def establish_nccl_comms(self, gidss: Iterable[tuple[UniqueGlobalDeviceIds, str]]):
        return self.blocking_tree(
            [
                self.establish_nccl_comm(gids, nccl_collnet_enable)
                for gids, nccl_collnet_enable in gidss
            ]
        )

    def start_trace(self, profiler_trace_path):
        futures = []
        for node_id, w in enumerate(self.workers):
            futures.append(
                w.start_trace.remote(f"{profiler_trace_path}/process-{node_id:06}")
            )
        self.blocking_tree(futures)
        self._profiler_trace_path = profiler_trace_path

    def stop_trace(self, merge_multihost_xplanes=False):
        if (
            _profiler_trace_path := getattr(self, "_profiler_trace_path", None)
        ) is not None:
            futures = [w.stop_trace.remote() for w in self.workers]
            self.blocking_tree(futures)

            if merge_multihost_xplanes:
                logger.info(
                    subprocess.run(
                        # Usage: merge_multihost_xplanes(pp_dim, mesh_shape, xplane_files)
                        [
                            f"merge_multihost_xplanes 1 {' '.join(map(str, self.full_mesh_shape))} $(find * -iname '*.xplane.pb')"
                        ],
                        shell=True,
                        cwd=_profiler_trace_path,
                        check=False,
                    )
                )
            del self._profiler_trace_path
        else:
            raise AssertionError("stop_trace called without calling `start_trace`")

    def memory_stats(self):
        futs = [w.memory_stats.remote() for w in self.workers]
        self.blocking_tree(futs)

    def __del__(self):
        try:
            if not RemoteMpmdMesh._unrecoverable_failure:
                self.memory_stats()
                futs = [w.shutdown.remote() for w in self.workers]
                dist_shutdown()
                self.blocking_tree(futs)
        except TypeError:
            # Developer note:
            # More efficient to ask for forgiveness (only during shutdown)
            # Than to perform a check at every single iteration
            pass

    def blocking(self, generator: Callable[[], Generator]) -> None:
        fut = list(generator())
        self.blocking_tree(fut)

    def blocking_tree(self, tree):
        try:
            leaves, tree = jax.tree_util.tree_flatten(tree)
            return jax.tree_util.tree_unflatten(tree, ray.get(leaves))

        except RayError:
            ray.shutdown()
            RemoteMpmdMesh._unrecoverable_failure = True
            raise
