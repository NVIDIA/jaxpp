import ctypes
import itertools
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from cuda.bindings import runtime as cuda_runtime
from cuda.core import Device, Event, Stream, StreamOptions
from jax._src.lib import xla_client
from nccl.bindings import nccl as nccl_bindings
from nccl.core import Communicator, UniqueId, get_unique_id, group

from jaxpp import env_vars
from jaxpp import jax_compat as jc
from jaxpp.dlpack import dlpack_nccl_args

logger = logging.getLogger(__name__)

DistributedRuntimeClient = jc._jax.DistributedRuntimeClient

completed_send_capsules_lock = threading.Lock()
completed_send_capsules: list[list[Any]] = []


@contextmanager
def cuda_device(device: Device) -> Iterator[Device]:
    """Select device without changing the caller's current CUDA device."""
    previous_device = Device()
    device.set_current()
    try:
        yield device
    finally:
        previous_device.set_current()


# ---------------------------------------------------------------------------
# Device / communicator / stream bookkeeping
# ---------------------------------------------------------------------------
class UniqueDevices(tuple[jax.Device, ...]):
    def __new__(cls, *args: jax.Device) -> "UniqueDevices":
        seen = set()
        unique = []
        for d in args:
            if d not in seen:
                unique.append(d)
                seen.add(d)
        return super().__new__(cls, unique)

    @cached_property
    def ranks(self) -> OrderedDict[jax.Device, int]:
        return OrderedDict((d, idx) for idx, d in enumerate(self))

    @property
    def leader(self) -> jax.Device:
        return self[0]

    @cached_property
    def key(self) -> str:
        return ",".join(str(d.id) for d in self)


class UniqueSortedDevices(UniqueDevices):
    def __new__(cls, *args: jax.Device) -> "UniqueSortedDevices":
        return super().__new__(cls, *sorted(set(args), key=lambda d: d.id))


def get_distributed_client() -> DistributedRuntimeClient:
    assert isinstance(jc.global_state.client, DistributedRuntimeClient)
    return jc.global_state.client


def get_nccl_id(devs: UniqueDevices) -> UniqueId:
    TIMEOUT = env_vars.jaxpp_client_timeout.value
    if devs.leader.process_index == jax.process_index():
        uid = get_unique_id()
        get_distributed_client().key_value_set_bytes(devs.key, uid.as_bytes)
        return uid
    raw = get_distributed_client().blocking_key_value_get_bytes(devs.key, TIMEOUT)
    return UniqueId.from_bytes(raw)


local_comms: dict[UniqueDevices, Communicator] = {}


def get_or_create_comm(devs: UniqueDevices) -> Communicator:
    comm = local_comms.get(devs)
    my_process_index = jax.process_index()
    if comm is None:
        logger.info(f"Creating communicator {devs=}")
        nccl_id = get_nccl_id(devs)
        for d in devs:
            if d.process_index == my_process_index:
                with cuda_device(Device(d.local_hardware_id)):
                    comm = Communicator.init(len(devs), devs.ranks[d], nccl_id)
        local_comms[devs] = comm
    return comm


local_streams: dict[tuple[jax.Device, jax.Device], Stream] = {}


def get_or_create_stream(
    local_dev: jax.Device, remote_dev: jax.Device, is_send: bool = False
) -> Stream:
    key = (local_dev, remote_dev) if is_send else (remote_dev, local_dev)
    stream = local_streams.get(key)
    if stream is None:
        assert local_dev.process_index == jax.process_index()
        logger.info(f"Creating stream for {key=} {is_send=}")
        with cuda_device(Device(local_dev.local_hardware_id)) as device:
            stream = device.create_stream(options=StreamOptions(nonblocking=True))
        local_streams[key] = stream
    return stream


class ShardOp(NamedTuple):
    """One enqueued point-to-point send/recv op."""

    is_send: bool
    device: Device
    data_ptr: int
    count: int
    nccl_dtype: int
    peer: int
    comm_ptr: int
    stream_ptr: int


class ShardCapsule(NamedTuple):
    device: jax.Device
    stream: Stream
    capsule: Any


def get_shard_ops_and_capsules(
    x: jax.Array, remote_sharding: jax.sharding.Sharding, *, is_send: bool
) -> tuple[list[ShardOp], list[ShardCapsule]]:
    operations: list[ShardOp] = []
    capsules: list[ShardCapsule] = []

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
        capsule = shard.data.__dlpack__(stream=int(stream.handle))
        capsules.append(ShardCapsule(x_device, stream, capsule))
        data_ptr, count, nccl_dtype = dlpack_nccl_args(capsule)

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

        operations.append(
            ShardOp(
                is_send=is_send,
                device=Device(x_device.local_hardware_id),
                data_ptr=data_ptr,
                count=count,
                nccl_dtype=nccl_dtype,
                peer=key.ranks[remote_device],
                comm_ptr=comm.ptr,
                stream_ptr=int(stream.handle),
            )
        )
    return operations, capsules


pending_send_callbacks: dict[int, list[Any]] = {}
next_send_callback_id = itertools.count(1)


# A send DLPack capsule pins a PjRtBuffer external reference. Destroying the
# capsule releases that reference (PJRT_Buffer_DecreaseExternalReferenceCount),
# which mutates shared PJRT buffer state without locking. We must not do that
# from the stream callback: cudaLaunchHostFunc runs on a CUDA-owned thread while
# the main thread runs PJRT with the GIL released, so the release races
# main-thread PJRT work and corrupts the host heap (intermittent "double free or
# corruption", often surfacing later at an unrelated allocation).
#
# So the callback only hands the capsule list off to a queue, and the main
# thread destroys the capsules in drain_completed_send_capsules at the next
# start_transfer. A list is queued only after its send stream has run this
# callback, so the send has drained and the buffer is safe to release.
def queue_completed_send_capsules(callback_id: int) -> None:
    with completed_send_capsules_lock:
        capsules = pending_send_callbacks.pop(callback_id)
        completed_send_capsules.append(capsules)


# Runs on the main thread (from start_transfer), so capsule destruction and the
# external-reference release it triggers are serialized with all other PJRT work.
def drain_completed_send_capsules() -> None:
    global completed_send_capsules

    with completed_send_capsules_lock:
        to_release, completed_send_capsules = completed_send_capsules, []

    for capsules in to_release:
        capsules.clear()


release_send_capsules_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
    queue_completed_send_capsules
)
release_send_capsules_host_fn = cuda_runtime.cudaHostFn_t(
    ctypes.cast(release_send_capsules_callback, ctypes.c_void_p).value
)


def launch_send_capsules_callback(stream: Stream, capsules: list[Any]):
    # cuda.core does not expose cudaLaunchHostFunc, so this is the only direct
    # cuda.bindings call in the stream/event path.
    callback_id = next(next_send_callback_id)
    with completed_send_capsules_lock:
        pending_send_callbacks[callback_id] = capsules
    err = cuda_runtime.cudaLaunchHostFunc(
        cuda_runtime.cudaStream_t(int(stream.handle)),
        release_send_capsules_host_fn,
        callback_id,
    )
    if isinstance(err, tuple):
        err = err[0]
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        with completed_send_capsules_lock:
            pending_send_callbacks.pop(callback_id)
        raise RuntimeError(f"cudaLaunchHostFunc failed with {err!r}")


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
    x: jax.Array, capsules: list[Any], done_events_by_device: dict[jax.Device, Event]
) -> Callable[[], jax.Array]:
    # Keep only array metadata in the wait closure. Holding onto `x` would keep
    # the caller-owned receive buffer alive past recv_done for no benefit.
    dtype = x.aval.dtype
    shape = x.aval.shape
    sharding = x.sharding

    def enqueue_wait() -> jax.Array:
        jax_single_arrays: list[jax.Array] = []
        local_device_assignment = [
            d
            for d in sharding._device_assignment
            if d.process_index == jax.process_index()
        ]
        for x_device, capsule in zip(local_device_assignment, capsules, strict=True):
            with cuda_device(Device(x_device.local_hardware_id)):
                ready_events_stream = x_device.get_stream_for_external_ready_events()
                # Order JAX's "ready events" stream after the NCCL receive completes,
                # then hand the filled buffer back to JAX on that stream.
                Stream.from_handle(int(ready_events_stream)).wait(
                    done_events_by_device[x_device]
                )
                jax_sda = jnp.array(
                    xla_client._xla.dlpack_managed_tensor_to_buffer(
                        capsule, x_device, ready_events_stream
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
) -> tuple[list[dict[jax.Device, Event]], list[list[Any]]]:
    operations: list[list[ShardOp]] = []
    # Capsules grouped by send stream, released together once the stream drains.
    send_capsules_by_stream: dict[int, tuple[jax.Device, Stream, list[Any]]] = {}
    # Per recv buffer, the (device, stream, capsule) of each addressable shard.
    recv_by_buffer: list[list[ShardCapsule]] = []

    for x, remote_sharding in zip(send_xs, send_remote_shardings, strict=True):
        ops, shard_capsules = get_shard_ops_and_capsules(
            x, remote_sharding, is_send=True
        )
        operations.append(ops)
        for local_device, stream, capsule in shard_capsules:
            _, _, capsules = send_capsules_by_stream.setdefault(
                int(stream.handle), (local_device, stream, [])
            )
            capsules.append(capsule)

    for x, remote_sharding in zip(recv_xs, recv_remote_shardings, strict=True):
        ops, shard_capsules = get_shard_ops_and_capsules(
            x, remote_sharding, is_send=False
        )
        operations.append(ops)
        recv_by_buffer.append(shard_capsules)

    with group():
        for shard_ops in operations:
            for shard_op in shard_ops:
                with cuda_device(shard_op.device):
                    nccl_op = (
                        nccl_bindings.send if shard_op.is_send else nccl_bindings.recv
                    )
                    nccl_op(
                        shard_op.data_ptr,
                        shard_op.count,
                        shard_op.nccl_dtype,
                        shard_op.peer,
                        shard_op.comm_ptr,
                        shard_op.stream_ptr,
                    )

    # NOTE: communicators are blocking, so after group end all sends/recvs have
    # been enqueued onto their streams. We can therefore release send capsules
    # after a stream callback marks them complete, and record recv completion
    # events on the streams.
    for local_device, stream, capsules in send_capsules_by_stream.values():
        with cuda_device(Device(local_device.local_hardware_id)):
            launch_send_capsules_callback(stream, capsules)

    done_events_by_buffer: list[dict[jax.Device, Event]] = []
    recv_dlpack_capsules: list[list[Any]] = []
    for shard_capsules in recv_by_buffer:
        done_events_by_device = {}
        capsules = []
        for local_device, stream, capsule in shard_capsules:
            with cuda_device(Device(local_device.local_hardware_id)):
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
