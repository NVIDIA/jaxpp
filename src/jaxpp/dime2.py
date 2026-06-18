import logging
import pickle
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxpp import env_vars
from jaxpp import jax_compat as jc
from jaxpp.dlpack import dlpack_nccl_args


# Lazy imports for cupy to avoid pulling in pytest at module load time
# (cupy unconditionally imports pytest via its testing module)
class LazyDeps:
    @cached_property
    def cupy(self):
        import cupy

        return cupy

    @cached_property
    def nccl(self):
        from cupy.cuda import nccl

        return nccl


lazy_deps = LazyDeps()
DistributedRuntimeClient = jc._jax.DistributedRuntimeClient

logger = logging.getLogger(__name__)
completed_send_capsules_lock = threading.Lock()
completed_send_capsules: list[list[Any]] = []


class UniqueDevices(tuple[jax.Device, ...]):
    def __new__(cls, *args):
        seen = set()
        unique = []
        for d in args:
            if d not in seen:
                unique.append(d)
                seen.add(d)
        return super().__new__(cls, unique)

    @cached_property
    def ranks(self):
        return OrderedDict((d, idx) for idx, d in enumerate(self))

    @property
    def leader(self):
        return self[0]

    @cached_property
    def key(self) -> str:
        return ",".join(str(d.id) for d in self)


class UniqueSortedDevices(UniqueDevices):
    def __new__(cls, *args):
        return super().__new__(cls, *sorted(set(args), key=lambda d: d.id))


def get_distributed_client() -> DistributedRuntimeClient:
    assert isinstance(jc.global_state.client, DistributedRuntimeClient)
    return jc.global_state.client


def get_nccl_id(devs: UniqueDevices):
    TIMEOUT = env_vars.jaxpp_client_timeout.value
    if devs.leader.process_index == jax.process_index():
        nccl_id = lazy_deps.nccl.get_unique_id()
        get_distributed_client().key_value_set_bytes(devs.key, pickle.dumps(nccl_id))
    else:
        nccl_id = get_distributed_client().blocking_key_value_get_bytes(
            devs.key, TIMEOUT
        )
        nccl_id = pickle.loads(nccl_id)
    return nccl_id


local_comms: dict = {}


def get_or_create_comm(devs: UniqueDevices):
    comm = local_comms.get(devs)
    my_process_index = jax.process_index()
    if comm is None:
        logger.info(f"Creating communicator {devs=}")
        nccl_id = get_nccl_id(devs)
        nccl = lazy_deps.nccl
        cupy = lazy_deps.cupy

        nccl.groupStart()
        for d in devs:
            if d.process_index == my_process_index:
                with cupy.cuda.Device(d.local_hardware_id):
                    comm = nccl.NcclCommunicator(len(devs), nccl_id, devs.ranks[d])
        nccl.groupEnd()

        local_comms[devs] = comm
    return comm


local_streams: dict = {}


def get_or_create_stream(
    local_dev: jax.Device, remote_dev: jax.Device, is_send: bool = False
):
    key = (local_dev, remote_dev) if is_send else (remote_dev, local_dev)
    stream = local_streams.get(key)
    if stream is None:
        assert local_dev.process_index == jax.process_index()
        logger.info(f"Creating stream for {key=} {is_send=}")
        cupy = lazy_deps.cupy
        with cupy.cuda.Device(local_dev.local_hardware_id):
            stream = cupy.cuda.Stream(non_blocking=True)
        local_streams[key] = stream
    return stream


def get_shard_ops_and_capsules(
    x: jax.Array, remote_sharding: jax.sharding.Sharding, *, is_send: bool
):
    operations = []
    dlpack_capsules = []

    # TODO: implement reshard for 4 devs -> 2 devs or 2->4 reshards
    # Variant of `Sharding.is_equivalent_to` that skips _internal_device_list.
    assert jc.shardings_are_equivalent(
        x.sharding, remote_sharding, x.ndim, compare_memkind=True
    ), f"incompatible shardings: {x.sharding=} vs {remote_sharding=}"

    shards_by_device: dict[jax.Device, jax.Shard] = {
        shard.device: shard for shard in x.addressable_shards
    }
    for x_device, remote_device in zip(
        x.sharding._device_assignment, remote_sharding._device_assignment, strict=True
    ):
        if x_device.process_index != jax.process_index():
            continue

        shard = shards_by_device[x_device]
        stream = get_or_create_stream(
            local_dev=x_device, remote_dev=remote_device, is_send=is_send
        )

        # `__dlpack__` returns a capsule that owns a PJRT ExternalReference for
        # the underlying buffer. The NCCL call below only receives a raw pointer,
        # so callers must keep this capsule alive until the recorded stream event
        # completes. For recvs, the capsule also backs the buffer later consumed
        # by `dlpack_managed_tensor_to_buffer`.
        dlpack = shard.data.__dlpack__(stream=stream.ptr)
        dlpack_capsules.append((x_device, stream, dlpack))
        data_ptr, count, dtype = dlpack_nccl_args(dlpack)

        key = (
            UniqueSortedDevices(x_device, remote_device)
            if not env_vars.jaxpp_directional_communicators.value
            else (
                UniqueDevices(x_device, remote_device)
                if is_send
                else UniqueDevices(remote_device, x_device)
            )
        )
        comm = get_or_create_comm(key)
        op = comm.send if is_send else comm.recv

        operations.append(
            (
                lazy_deps.cupy.cuda.Device(x_device.local_hardware_id),
                op,
                (
                    data_ptr.value,
                    count,
                    dtype.value,
                    key.ranks[remote_device],
                    stream.ptr,
                ),
            )
        )
    return operations, dlpack_capsules


# A send DLPack capsule pins a PjRtBuffer external reference. Destroying the
# capsule releases that reference (PJRT_Buffer_DecreaseExternalReferenceCount),
# which mutates shared PJRT buffer state without locking. We must not do that
# from the stream callback: CuPy runs launch_host_func on a CUDA-owned thread
# while the main thread runs PJRT with the GIL released, so the release races
# main-thread PJRT work and corrupts the host heap (intermittent "double free or
# corruption", often surfacing later at an unrelated allocation).
#
# So the callback only hands the capsule list off to a queue (cheap and GIL
# safe), and the main thread destroys the capsules in drain_completed_send_capsules
# at the next start_transfer. A list is queued only after its send stream has run
# this callback, so the send has drained and the buffer is safe to release.
def queue_completed_send_capsules(dlpack_capsules: list[Any]) -> None:
    with completed_send_capsules_lock:
        completed_send_capsules.append(dlpack_capsules)


# Runs on the main thread (from start_transfer), so capsule destruction and the
# external-reference release it triggers are serialized with all other PJRT work.
def drain_completed_send_capsules() -> None:
    global completed_send_capsules

    with completed_send_capsules_lock:
        to_release, completed_send_capsules = completed_send_capsules, []

    for dlpack_capsules in to_release:
        dlpack_capsules.clear()


@dataclass(slots=True)
class Transfer:
    """Handle for a grouped transfer that may include sends and receives."""

    future_fns: tuple[Callable[[], jax.Array], ...] | None

    def done(self) -> Sequence[jax.Array]:
        if self.future_fns is None:
            raise RuntimeError("transfer has already completed")
        future_fns = self.future_fns
        self.future_fns = None
        return tuple(future_fn() for future_fn in future_fns)


def make_future_array(
    x: jax.Array, cpy_arrays: list[Any], done_events_by_device: dict[jax.Device, Any]
):
    # Keep only array metadata in the wait closure. Holding onto `x` would keep
    # the caller-owned receive buffer alive past recv_done for no benefit.
    dtype = x.aval.dtype
    shape = x.aval.shape
    sharding = x.sharding

    def enqueue_wait():
        cupy = lazy_deps.cupy
        jax_single_arrays = []
        local_device_assignment = [
            d
            for d in sharding._device_assignment
            if d.process_index == jax.process_index()
        ]
        for x_device, cpy_arr in zip(local_device_assignment, cpy_arrays, strict=True):
            with cupy.cuda.Device(x_device.local_hardware_id):
                ready_events_stream = x_device.get_stream_for_external_ready_events()
                cupy.cuda.ExternalStream(ready_events_stream).wait_event(
                    done_events_by_device[x_device]
                )
                jax_sda = jnp.array(
                    jax._src.lib.xla_client._xla.dlpack_managed_tensor_to_buffer(
                        cpy_arr, x_device, ready_events_stream
                    ),
                    copy=False,  # NOTE: copy is unnecessary
                )
                jax_single_arrays.append(jax_sda)
        return jax.make_array_from_single_device_arrays(
            shape, sharding, jax_single_arrays, dtype=dtype
        )

    return enqueue_wait


def enqueue_nccl_transfer_group(
    send_xs: Sequence[jax.Array],
    send_remote_shardings: Sequence[jax.sharding.Sharding],
    recv_xs: Sequence[jax.Array],
    recv_remote_shardings: Sequence[jax.sharding.Sharding],
) -> tuple[list[dict[jax.Device, Any]], list[list[Any]]]:
    operations: list[list[Any]] = []
    # Capsules grouped by send stream, released together once the stream drains.
    send_capsules_by_stream: dict[int, tuple[jax.Device, Any, list[Any]]] = {}
    # Per recv buffer, the (device, stream, capsule) of each addressable shard.
    recv_by_buffer: list[list[tuple[jax.Device, Any, Any]]] = []

    for x, remote_sharding in zip(send_xs, send_remote_shardings, strict=True):
        ops, device_capsules = get_shard_ops_and_capsules(
            x, remote_sharding, is_send=True
        )
        operations.append(ops)
        for local_device, stream, capsule in device_capsules:
            _, _, capsules = send_capsules_by_stream.setdefault(
                stream.ptr, (local_device, stream, [])
            )
            capsules.append(capsule)

    for x, remote_sharding in zip(recv_xs, recv_remote_shardings, strict=True):
        ops, device_capsules = get_shard_ops_and_capsules(
            x, remote_sharding, is_send=False
        )
        operations.append(ops)
        recv_by_buffer.append(device_capsules)

    nccl = lazy_deps.nccl
    cupy = lazy_deps.cupy
    nccl.groupStart()
    for shard_ops in operations:
        for cpy_dev, op, args in shard_ops:
            with cpy_dev:
                op(*args)
    nccl.groupEnd()

    # NOTE: communicators are blocking, so after groupEnd all sends/recvs have
    #  been enqueued onto their streams. We can therefore release send capsules
    #  after a stream callback marks them complete, and record recv completion
    #  events on the streams.
    for local_device, stream, capsules in send_capsules_by_stream.values():
        with cupy.cuda.Device(local_device.local_hardware_id):
            stream.launch_host_func(queue_completed_send_capsules, capsules)

    done_events_by_buffer: list[dict[jax.Device, Any]] = []
    recv_dlpack_capsules: list[list[Any]] = []
    for device_capsules in recv_by_buffer:
        done_events_by_device = {}
        capsules = []
        for local_device, stream, capsule in device_capsules:
            with cupy.cuda.Device(local_device.local_hardware_id):
                done_events_by_device[local_device] = stream.record()
            capsules.append(capsule)
        done_events_by_buffer.append(done_events_by_device)
        recv_dlpack_capsules.append(capsules)

    return done_events_by_buffer, recv_dlpack_capsules


def start_transfer(
    send_xs: Sequence[jax.Array],
    send_remote_shardings: Sequence[jax.sharding.Sharding],
    recv_xs: Sequence[jax.Array],
    recv_remote_shardings: Sequence[jax.sharding.Sharding],
) -> Transfer:
    drain_completed_send_capsules()
    done_events_by_buffer, dlpack_capsules = enqueue_nccl_transfer_group(
        send_xs, send_remote_shardings, recv_xs, recv_remote_shardings
    )
    future_fns = [
        make_future_array(x, capsules, done_events_by_device)
        for x, capsules, done_events_by_device in zip(
            recv_xs, dlpack_capsules, done_events_by_buffer, strict=True
        )
    ]
    return Transfer(tuple(future_fns))
