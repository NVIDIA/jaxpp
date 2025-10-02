import logging
import pickle
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from functools import cached_property, partial
from typing import Any, Callable

import cupy
import jax
import jax.numpy as jnp
from cupy.cuda import nccl

if jax.__version_info__ < (0, 7, 2):
    from jax._src.op_shardings import are_op_shardings_equal as are_hlo_shardings_equal
else:
    from jax._src.op_shardings import are_hlo_shardings_equal

from jaxpp import env_vars
from jaxpp.dlpack import capsule_name, dlpack_nccl_args

if jax.__version_info__ > (0, 6, 0):
    from jax._src.lib import _jax

    DistributedRuntimeClient = _jax.DistributedRuntimeClient
else:
    import jaxlib.xla_extension as xe

    DistributedRuntimeClient = xe.DistributedRuntimeClient

logger = logging.getLogger(__name__)


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
    from jax._src.distributed import global_state

    assert isinstance(global_state.client, DistributedRuntimeClient)
    return global_state.client


def get_nccl_id(devs: UniqueDevices):
    TIMEOUT = 240_000  # FIXME: make it an argument
    if devs.leader.process_index == jax.process_index():
        nccl_id = nccl.get_unique_id()
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
        with cupy.cuda.Device(local_dev.local_hardware_id):
            stream = cupy.cuda.Stream(non_blocking=True)
        local_streams[key] = stream
    return stream


def shardings_are_compatible(
    self: jax.sharding.Sharding, other: jax.sharding.Sharding, ndim: int
):
    # NOTE: Variant of `jax.sharding.Sharding.is_equivalent_to` that skips _internal_device_list check
    return (
        are_hlo_shardings_equal(
            self._to_xla_hlo_sharding(ndim), other._to_xla_hlo_sharding(ndim)
        )
        # and self._internal_device_list == other._internal_device_list  # type: ignore
        and self.memory_kind == other.memory_kind
    )


def _get_shard_ops_and_keep_alives(
    x: jax.Array,
    remote_sharding: jax.sharding.Sharding,
    stream_per_local_device: dict[jax.Device, Any],
    *,
    is_send: bool,
):
    operations = []
    cpy_arrays = []

    # TODO: implement reshard for 4 devs -> 2 devs or 2->4 reshards
    assert shardings_are_compatible(
        x.sharding, remote_sharding, x.ndim
    ), f"incompatible shardings: {x.sharding=} vs {remote_sharding=}"

    shards_by_device: dict[jax.Device, jax.Shard] = {
        shard.device: shard for shard in x.addressable_shards
    }
    for x_device, remote_device in zip(
        x.sharding._device_assignment,
        remote_sharding._device_assignment,
        strict=True,
    ):
        if x_device.process_index != jax.process_index():
            continue

        shard = shards_by_device[x_device]
        stream = stream_per_local_device[x_device]

        dlpack = jax.dlpack.to_dlpack(shard.data, stream=stream.ptr)
        cpy_arrays.append(dlpack)
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
                cupy.cuda.Device(x_device.local_hardware_id),
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
    return operations, cpy_arrays


class CachedCall:
    sentinel = object()

    def __init__(self, fn):
        self.fn = fn
        self.value = CachedCall.sentinel

    def __call__(self):
        if self.value is not CachedCall.sentinel:
            return self.value
        self.value = self.fn()
        self.fn = None
        return self.value


def _sync_events(done_events_by_device):
    for e in done_events_by_device.values():
        e.synchronize()


def _make_future_array(
    x: jax.Array, cpy_arrays: list[Any], done_events_by_device: dict[jax.Device, Any]
):
    # NOTE: herre we ensure that `x` is not captured
    # by `enqueue_wait` since the caller uses it as
    # x._enqueue_wait = _make_future_array(x, cpy_arrays=_(x.addressable_shards), ...)
    dtype = x.aval.dtype
    shape = x.aval.shape
    sharding = x.sharding

    def enqueue_wait():
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


def send_or_recv(
    xs: Sequence[jax.Array],
    remote_shardings: Sequence[jax.sharding.Sharding],
    *,
    is_send: bool,
) -> list[Callable[[], Any]]:
    """
    `is_send==True` this function corresponds to a send.
    `is_send==False` this function corresponds to a receive
    and `x` will be consumed, i.e. it's unsafe to use `x` after `send_or_recv(x, ...)`.

    `x` can be a "global" array spanning multiple processes/hosts.
    In that case, this process will send/receive only its corresponding addressable_shards.
    """
    local_device_assignment = xs[0].sharding._device_assignment
    remote_device_assignment = remote_shardings[0]._device_assignment

    assert all(
        local_device_assignment == _.sharding._device_assignment for _ in xs
    ), f"{[_.sharding._device_assignment for _ in xs]}"
    assert all(
        remote_device_assignment == _._device_assignment for _ in remote_shardings
    ), f"Differing remote device assignments {[_._device_assignment for _ in remote_shardings]}"

    my_process_index = jax.process_index()
    stream_per_local_device = {
        local_device: get_or_create_stream(
            local_dev=local_device, remote_dev=remote_device, is_send=is_send
        )
        for local_device, remote_device in zip(
            local_device_assignment, remote_device_assignment, strict=True
        )
        if local_device.process_index == my_process_index
    }

    operations = list[list[Any]]()
    keep_alives = list[list[Any]]()
    for x, remote_sharding in zip(xs, remote_shardings, strict=True):
        ops, kas = _get_shard_ops_and_keep_alives(
            x, remote_sharding, stream_per_local_device, is_send=is_send
        )
        operations.append(ops)
        keep_alives.append(kas)

    nccl.groupStart()
    for shard_ops in operations:
        for cpy_dev, op, args in shard_ops:
            with cpy_dev:
                op(*args)
    nccl.groupEnd()

    # NOTE: since communicators are blocking, after the group_end operation
    #  above, all the send/recvs have been enqueued into the stream. Therefore,
    #  we can record events on the stream

    done_events_by_device = {}
    for local_device in x.sharding._device_assignment:
        if local_device.process_index != my_process_index:
            continue

        with cupy.cuda.Device(local_device.local_hardware_id):
            done_events_by_device[local_device] = stream_per_local_device[
                local_device
            ].record()

    # XXX: I don't like different return types below, however I am not sure
    #  what's a better alternative given we want a "symmetric"
    # `send_or_recv` API
    if is_send:
        res = []
        finalizer = partial(_sync_events, done_events_by_device)
        for x, cpy_arrays in zip(xs, keep_alives, strict=True):
            # NOTE: this `weakref.finalize` call ensures that
            #  we wait for the send to finish before `x` is deleted.
            # In practice all uses of `send_or_recv` today ensure to keep
            # the sent values alive before deleting them.
            # TODO: conservatively do the same for `cpy_arrays`?
            res.append(weakref.finalize(x, finalizer))
        return res
    else:
        res = []
        for x, cpy_arrays in zip(xs, keep_alives, strict=True):
            res.append(
                CachedCall(_make_future_array(x, cpy_arrays, done_events_by_device))
            )
        return res
