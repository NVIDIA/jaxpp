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

import contextlib
import dataclasses
import logging
import operator
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import (
    Any,
    Generator,
    Generic,
    NotRequired,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
)

import jax
import jax.extend.source_info_util as jsiu
from jax._src import effects
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

from jaxpp import array_ops
from jaxpp import jax_compat as jc
from jaxpp.array import MpmdArray
from jaxpp.dime2 import start_transfer
from jaxpp.jax_compat import core as jcore
from jaxpp.mesh import MpmdMesh, _require_mpmd_indices, _resolve_placement
from jaxpp.types import MpmdSharding, TaskType
from jaxpp.utils import (
    filter_axes,
    get_named_sharding,
    print_memstats,
    update_named_sharding,
)

logger = logging.getLogger(__name__)

TOKEN_AVAL = jcore.abstract_token

_CommDoneT = TypeVar("_CommDoneT")
_CommDoneT_co = TypeVar("_CommDoneT_co", covariant=True)


class _CommTransfer(Protocol[_CommDoneT_co]):
    def done(self) -> _CommDoneT_co: ...


class CommToken(Generic[_CommDoneT]):
    """Runtime token for async communication.

    The token wraps a send or recv transfer handle from `dime2`. The handle owns
    any DLPack capsules needed to pin external buffer references, so primitive
    implementations do not duplicate lifetime bookkeeping here.
    """

    __slots__ = ("transfer",)

    def __init__(self, transfer: _CommTransfer[_CommDoneT]):
        self.transfer = transfer

    def done(self) -> _CommDoneT:
        return self.transfer.done()


jcore.pytype_aval_mappings[CommToken] = lambda _t: TOKEN_AVAL
jc.register_canonicalize_value_handler(CommToken, None)


class _NoopTransfer:
    def done(self) -> None:
        return None


ShardingParam = jax.sharding.Sharding | jc.UnspecifiedValue | None
ShardingTuple = tuple[ShardingParam, ...]


def validate_all_reduce_reduction_metadata(
    in_sharding: jax.sharding.NamedSharding,
    out_sharding: jax.sharding.NamedSharding,
    axis_name: str,
) -> None:
    input_spec = filter_axes(in_sharding.spec, {axis_name})
    if (input_spec.unreduced, input_spec.reduced) != (
        out_sharding.spec.unreduced,
        out_sharding.spec.reduced,
    ):
        raise ValueError(
            "cross-MPMD all-reduce input reduction metadata must match "
            f"the output; got {input_spec} and {out_sharding.spec}"
        )


add_multi_p = jcore.Primitive("add_multi")


@add_multi_p.def_abstract_eval
def add_multi_abstract_eval(
    *args,
    in_shardings: ShardingTuple | None = None,
    out_shardings: ShardingTuple | None = None,
    mpmd_idxs=None,
    donate_invars=None,
):
    first = args[0]
    assert all(first.dtype == arg.dtype for arg in args)
    return first


@add_multi_p.def_impl
def add_multi_impl(
    *args,
    in_shardings: ShardingTuple | None = None,
    out_shardings: ShardingTuple | None = None,
    mpmd_idxs=None,
    donate_invars=None,
):
    assert mpmd_idxs is not None
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    assert (
        not mpmd_mesh.jax_mesh.is_multi_process
    ), f"{add_multi_p.name} supported only in single-process runtime"
    assert out_shardings is not None
    (out_sharding,) = out_shardings
    assert isinstance(out_sharding, jax.sharding.NamedSharding)

    prev_shardings: list[jax.NamedSharding] = [a.sharding for a in args]
    comm_mpmd_mesh = mpmd_mesh.mpmd_submesh(list(mpmd_idxs))
    axis_name = mpmd_mesh.mpmd_axis_name
    out_sharding = update_named_sharding(
        out_sharding,
        mesh=comm_mpmd_mesh.jax_mesh,
        spec=filter_axes(out_sharding.spec, {axis_name}),
    )
    for sharding in prev_shardings:
        validate_all_reduce_reduction_metadata(sharding, out_sharding, axis_name)
    logical_in_shardings = tuple(
        update_named_sharding(sharding, spec=out_sharding.spec)
        for sharding in prev_shardings
    )
    if not jc.reduce_sum_accepts_unreduced and out_sharding.spec.unreduced:
        raise NotImplementedError(
            "cross-MPMD all-reduce with unreduced inputs requires JAX >= 0.9.1"
        )

    stacked = stack_p.bind(
        *args,
        **StackParams(
            in_shardings=logical_in_shardings, mpmd_mesh=comm_mpmd_mesh, axis=0
        ),
    )
    with jax.set_mesh(comm_mpmd_mesh.jax_mesh):
        (total,) = jax.jit(
            all_reduce_fn,
            in_shardings=(stacked.sharding,),
            out_shardings=(out_sharding,),
        )((stacked,))
    replicas = slice_p.bind(
        total,
        **SliceParams(
            in_sharding=out_sharding,
            groups=tuple((idx,) for idx in range(comm_mpmd_mesh.mpmd_dim)),
            mpmd_mesh=comm_mpmd_mesh,
        ),
    )
    return MpmdArray(
        list(replicas),
        mpmd_sharding=MpmdSharding(
            mpmd_mesh,
            mpmd_idxs,
            out_sharding.spec,
            memory_kind=out_sharding.memory_kind,
        ),
    )


def add_multi_lower(
    *args,
    in_shardings: ShardingTuple | None = None,
    out_shardings: ShardingTuple | None = None,
    mpmd_idxs=None,
    donate_invars=None,
):
    return reduce(operator.add, args)


mlir.register_lowering(
    add_multi_p, mlir.lower_fun(add_multi_lower, multiple_results=False)
)


def all_reduce_fn(arrs):
    return tuple(a.sum(0, dtype=a.dtype) for a in arrs)


def all_reduce(
    arrs: list[jax.Array],
    mpmd_mesh: MpmdMesh,
    mpmd_idxs: list[int],
    out_specs: list[jax.sharding.PartitionSpec],
    donated: Sequence[int] | None = None,
):
    assert mpmd_mesh.my_mpmd_axis_index in mpmd_idxs
    comm_mpmd_mesh = mpmd_mesh.mpmd_submesh(mpmd_idxs)
    comm_mesh = comm_mpmd_mesh.jax_mesh

    shardings = [get_named_sharding(a) for a in arrs]
    assert len(set(_.mesh for _ in shardings)) == 1

    axis_name = mpmd_mesh.mpmd_axis_name
    out_shardings = tuple(
        update_named_sharding(
            sharding, mesh=comm_mesh, spec=filter_axes(spec, {axis_name})
        )
        for sharding, spec in zip(shardings, out_specs, strict=True)
    )
    for sharding, out_sharding in zip(shardings, out_shardings, strict=True):
        validate_all_reduce_reduction_metadata(sharding, out_sharding, axis_name)
    logical_in_shardings = tuple(
        update_named_sharding(sharding, spec=out_sharding.spec)
        for sharding, out_sharding in zip(shardings, out_shardings, strict=True)
    )
    if not jc.reduce_sum_accepts_unreduced and any(
        sharding.spec.unreduced for sharding in out_shardings
    ):
        raise NotImplementedError(
            "cross-MPMD all-reduce with unreduced inputs requires JAX >= 0.9.1"
        )

    stacked = tuple(
        stack_p.bind(
            a,
            **StackParams(
                in_shardings=(logical_in_sharding,), mpmd_mesh=comm_mpmd_mesh, axis=0
            ),
        )
        for a, logical_in_sharding in zip(arrs, logical_in_shardings, strict=True)
    )
    stacked_shardings = tuple(a.sharding for a in stacked)

    with jax.set_mesh(comm_mesh):
        all_reduced: tuple[jax.Array, ...] = jax.jit(
            all_reduce_fn,
            in_shardings=stacked_shardings,
            out_shardings=out_shardings,
            donate_argnums=donated,
        )(stacked)

    local_group = _require_mpmd_indices(
        comm_mpmd_mesh, mpmd_mesh.lowering_mesh(), name="all_reduce local mesh"
    )
    return [
        slice_p.bind(
            a,
            **SliceParams(
                in_sharding=out_sharding,
                groups=(local_group,),
                mpmd_mesh=comm_mpmd_mesh,
            ),
        )[0]
        for a, out_sharding in zip(all_reduced, out_shardings, strict=True)
    ]


all_reduce_p = jcore.Primitive("all_reduce")


@all_reduce_p.def_abstract_eval
def all_reduce_abstract_eval(
    arg,
    mpmd_idxs: list[int],
    donated: Sequence[int],
    out_spec: jax.sharding.PartitionSpec,
):
    return arg


# TODO: support multi-arity all_reduce
@all_reduce_p.def_impl
def all_reduce_impl(
    arg,
    mpmd_idxs: list[int],
    donated: Sequence[int],
    out_spec: jax.sharding.PartitionSpec,
):
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    return all_reduce(
        [arg],
        mpmd_mesh=mpmd_mesh,
        mpmd_idxs=mpmd_idxs,
        out_specs=[out_spec],
        donated=donated,
    )[0]


gather_multi_p = jcore.Primitive("gather_multi")


@gather_multi_p.def_effectful_abstract_eval
def gather_multi_abstract_eval(
    *args,
    axis: int = 0,
    in_shardings: ShardingTuple | None = None,
    out_shardings: ShardingTuple | None = None,
    mpmd_idxs=None,
    donate_invars=None,
    restore_order_perm=None,
):
    from jax._src.lax import lax as jax_lax

    return jax_lax.concatenate_p.abstract_eval(*args, dimension=axis)


@gather_multi_p.def_impl
def gather_multi_impl(
    *args,
    axis: int = 0,
    in_shardings: ShardingTuple | None = None,
    out_shardings: ShardingTuple | None = None,
    mpmd_idxs=None,
    donate_invars=None,
    restore_order_perm=None,
):
    assert mpmd_idxs is not None
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    assert (
        not mpmd_mesh.jax_mesh.is_multi_process
    ), f"{gather_multi_p.name} supported only in single-process runtime"

    prev_shardings: list[jax.NamedSharding] = [a.sharding for a in args]

    first_rank_arrays = [jax.device_put(a, args[0].sharding) for a in args]
    result = jax.numpy.concatenate(first_rank_arrays, axis=axis)

    # Apply permutation to restore original input order if needed
    if restore_order_perm is not None:
        result = jax.numpy.take(result, jax.numpy.array(restore_order_perm), axis=axis)

    return MpmdArray(
        [jax.device_put(result, s) for s in prev_shardings],
        mpmd_sharding=MpmdSharding(
            mpmd_mesh,
            mpmd_idxs,
            prev_shardings[0].spec,
            memory_kind=prev_shardings[0].memory_kind,
        ),
    )


def gather_multi_lower(
    *arrays,
    axis: int = 0,
    in_shardings: ShardingTuple | None = None,
    out_shardings: ShardingTuple | None = None,
    mpmd_idxs=None,
    donate_invars=None,
    restore_order_perm=None,
):
    result = jax.numpy.concatenate(arrays, axis=axis)
    if restore_order_perm is not None:
        result = jax.numpy.take(result, jax.numpy.array(restore_order_perm), axis=axis)
    return result


mlir.register_lowering(
    gather_multi_p, mlir.lower_fun(gather_multi_lower, multiple_results=False)
)


def _flatten_adjacent_axes(arr, first_axis):
    new_shape = (
        *arr.shape[:first_axis],
        arr.shape[first_axis] * arr.shape[first_axis + 1],
        *arr.shape[first_axis + 2 :],
    )
    return arr.reshape(new_shape)


def _squeezed(arrs, axis: int, perm: Sequence[int]):
    return tuple(
        jax.numpy.take(
            _flatten_adjacent_axes(a, axis), jax.numpy.array(perm), axis=axis
        )
        for a in arrs
    )


def all_gather(
    arrs: list[jax.Array],
    mpmd_mesh: MpmdMesh,
    mpmd_idxs: list[int],
    out_specs: list[jax.sharding.PartitionSpec],
    donated: Sequence[int],
    restore_order_perm: Sequence[int],
    axis: int = 0,
):
    """Gather arrays across MPMD groups, keeping the stacked dimension."""
    assert mpmd_mesh.my_mpmd_axis_index in mpmd_idxs
    comm_mpmd_mesh = mpmd_mesh.mpmd_submesh(mpmd_idxs)
    comm_mesh = comm_mpmd_mesh.jax_mesh
    axis_name = mpmd_mesh.mpmd_axis_name

    shardings = [get_named_sharding(a) for a in arrs]
    assert len(set(_.mesh for _ in shardings)) == 1

    stacked_shardings = tuple(
        array_ops.stack_shape_and_sharding(
            (a.shape,), (sharding,), mpmd_mesh=comm_mpmd_mesh, axis=axis
        )[1]
        for a, sharding in zip(arrs, shardings, strict=True)
    )
    stacked = tuple(
        stack_p.bind(
            a,
            **StackParams(
                in_shardings=(sharding,), mpmd_mesh=comm_mpmd_mesh, axis=axis
            ),
        )
        for a, sharding in zip(arrs, shardings, strict=True)
    )
    out_shardings = tuple(
        update_named_sharding(
            sharding, mesh=comm_mesh, spec=filter_axes(spec, {axis_name})
        )
        for sharding, spec in zip(shardings, out_specs, strict=True)
    )

    with jax.set_mesh(comm_mesh):
        gathered: tuple[jax.Array, ...] = jax.jit(
            _squeezed,
            in_shardings=stacked_shardings,
            out_shardings=out_shardings,
            donate_argnums=donated,
            static_argnums=(1, 2),
        )(stacked, axis, restore_order_perm)

    local_group = _require_mpmd_indices(
        comm_mpmd_mesh, mpmd_mesh.lowering_mesh(), name="all_gather local mesh"
    )
    return [
        slice_p.bind(
            a,
            **SliceParams(
                in_sharding=out_sharding,
                groups=(local_group,),
                mpmd_mesh=comm_mpmd_mesh,
            ),
        )[0]
        for a, out_sharding in zip(gathered, out_shardings, strict=True)
    ]


all_gather_p = jcore.Primitive("all_gather")


@all_gather_p.def_abstract_eval
def all_gather_abstract_eval(
    arg,
    axis: int,
    mpmd_idxs: list[int],
    donated: Sequence[int],
    out_spec: jax.sharding.PartitionSpec,
    restore_order_perm: Sequence[int],
):
    # Output shape: n_mpmd * local_size at axis
    # The permutation only reorders elements, doesn't change shape
    n = len(mpmd_idxs)
    new_shape = arg.shape[:axis] + (n * arg.shape[axis],) + arg.shape[axis + 1 :]
    return jcore.ShapedArray(new_shape, arg.dtype)


@all_gather_p.def_impl
def all_gather_impl(
    arg,
    axis: int,
    mpmd_idxs: list[int],
    donated: Sequence[int],
    out_spec: jax.sharding.PartitionSpec,
    restore_order_perm: Sequence[int],
):
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    return all_gather(
        [arg],
        mpmd_mesh=mpmd_mesh,
        mpmd_idxs=mpmd_idxs,
        axis=axis,
        donated=donated,
        out_specs=[out_spec],
        restore_order_perm=restore_order_perm,
    )[0]


class TransferParams(TypedDict):
    src_shardings: tuple[jax.sharding.Sharding, ...]
    tgt_shardings: tuple[jax.sharding.Sharding, ...]


transfer_p = jcore.Primitive("transfer")
transfer_p.multiple_results = True


def _abstract_sharding_for_aval(sharding):
    if not isinstance(sharding, jax.sharding.NamedSharding):
        return sharding

    mesh = sharding.mesh
    if not isinstance(mesh, jax.sharding.AbstractMesh):
        mesh = mesh.abstract_mesh

    if mesh is not sharding.mesh or sharding.memory_kind is not None:
        return update_named_sharding(sharding, mesh=mesh, memory_kind=None)
    return sharding


@transfer_p.def_effectful_abstract_eval
def transfer_abstract_eval(*args, **params: Unpack[TransferParams]):
    src_shardings = params["src_shardings"]
    del src_shardings
    out_avals = tuple(
        arg.update(sharding=_abstract_sharding_for_aval(tgt_sharding))
        if isinstance(arg, jcore.ShapedArray)
        else arg
        for arg, tgt_sharding in zip(args, params["tgt_shardings"], strict=True)
    )
    return (TOKEN_AVAL, *out_avals), frozenset({communication_effect})


@transfer_p.def_impl
def transfer_impl(*args, **params: Unpack[TransferParams]):
    for a, sh in zip(args, params["src_shardings"], strict=True):
        assert isinstance(a, jax.Array)
        assert a.sharding == sh

    res = jax.device_put(args, params["tgt_shardings"])
    return (CommToken(_NoopTransfer()), *res)


def _aval_with_shape_and_sharding(
    arg, shape: tuple[int, ...], sharding: jax.sharding.Sharding
):
    if isinstance(arg, jcore.ShapedArray):
        return arg.update(shape=shape, sharding=_abstract_sharding_for_aval(sharding))
    return arg


class StackParams(TypedDict):
    in_shardings: tuple[jax.sharding.NamedSharding, ...]
    mpmd_mesh: MpmdMesh
    axis: int


class LocalStackParams(TypedDict):
    in_sharding: jax.sharding.NamedSharding
    out_sharding: jax.sharding.NamedSharding
    out_shape: tuple[int, ...]
    expand: bool
    axis: int


# `mpmd_mesh` on stack/slice primitives is a coordinate scope, not necessarily
# the ambient global MPMD mesh. For stack it is the output scope. For slice it is
# the scope in which `groups` are numbered.
stack_p = jcore.Primitive("stack")
local_stack_p = jcore.Primitive("local_stack")


def stack_abstract_eval(*args, **params: Unpack[StackParams]):
    if len(args) == 0:
        raise ValueError("stack expects at least one argument")
    dtype = args[0].dtype
    if any(arg.dtype != dtype for arg in args[1:]):
        raise ValueError("stack arguments must have the same dtype")
    _, out_sharding, out_shape, _ = array_ops.stack_shape_and_sharding(
        tuple(arg.shape for arg in args),
        params["in_shardings"],
        mpmd_mesh=params["mpmd_mesh"],
        axis=params["axis"],
    )
    return _aval_with_shape_and_sharding(args[0], out_shape, out_sharding)


stack_p.def_abstract_eval(stack_abstract_eval)


def local_stack_abstract_eval(arg, **params: Unpack[LocalStackParams]):
    return _aval_with_shape_and_sharding(
        arg, params["out_shape"], params["out_sharding"]
    )


local_stack_p.def_abstract_eval(local_stack_abstract_eval)


@stack_p.def_impl
def stack_impl(*args, **params: Unpack[StackParams]):
    for arg, sharding in zip(args, params["in_shardings"], strict=True):
        assert isinstance(arg, jax.Array)
        assert jc.shardings_are_equivalent(
            arg.sharding, sharding, arg.ndim, compare_memkind=True
        )
    return array_ops.stack_arrays_with_shardings(
        args, params["in_shardings"], mpmd_mesh=params["mpmd_mesh"], axis=params["axis"]
    )


@local_stack_p.def_impl
def local_stack_impl(arg, **params: Unpack[LocalStackParams]):
    assert isinstance(arg, jax.Array)
    assert jc.shardings_are_equivalent(
        arg.sharding, params["in_sharding"], arg.ndim, compare_memkind=True
    )
    return array_ops.local_stack_array(
        arg,
        out_shape=params["out_shape"],
        out_sharding=params["out_sharding"],
        expand=params["expand"],
        axis=params["axis"],
    )


class SliceParams(TypedDict):
    in_sharding: jax.sharding.NamedSharding
    groups: tuple[tuple[int, ...], ...]
    mpmd_mesh: MpmdMesh


class LocalSliceParams(TypedDict):
    in_sharding: jax.sharding.NamedSharding
    out_shardings: tuple[jax.sharding.NamedSharding, ...]


slice_p = jcore.Primitive("slice")
slice_p.multiple_results = True
local_slice_p = jcore.Primitive("local_slice")
local_slice_p.multiple_results = True


def slice_abstract_eval(arg, **params: Unpack[SliceParams]):
    out_shapes, out_shardings = array_ops.slice_shape_and_shardings(
        arg.shape,
        params["in_sharding"],
        params["groups"],
        mpmd_mesh=params["mpmd_mesh"],
    )
    return tuple(
        _aval_with_shape_and_sharding(arg, out_shape, out_sharding)
        for out_shape, out_sharding in zip(out_shapes, out_shardings, strict=True)
    )


def local_slice_abstract_eval(arg, **params: Unpack[LocalSliceParams]):
    return tuple(
        _aval_with_shape_and_sharding(
            arg,
            array_ops.local_slice_out_shape(
                arg.shape, params["in_sharding"], out_sharding
            ),
            out_sharding,
        )
        for out_sharding in params["out_shardings"]
    )


slice_p.def_abstract_eval(slice_abstract_eval)
local_slice_p.def_abstract_eval(local_slice_abstract_eval)


@slice_p.def_impl
def slice_impl(arg, **params: Unpack[SliceParams]):
    assert isinstance(arg, jax.Array)
    assert arg.sharding == params["in_sharding"]
    return array_ops.slice_arrays(
        arg,
        in_sharding=params["in_sharding"],
        groups=params["groups"],
        mpmd_mesh=params["mpmd_mesh"],
    )


@local_slice_p.def_impl
def local_slice_impl(arg, **params: Unpack[LocalSliceParams]):
    assert isinstance(arg, jax.Array)
    return array_ops.local_slice_arrays(
        arg, in_sharding=params["in_sharding"], out_shardings=params["out_shardings"]
    )


delete_p = jcore.Primitive("delete")
# NOTE: we have delete equations for donated buffers as well
#  which fail if Jax tries to canonicalize them.
#  Hence we skip canonicalization for delete
delete_p.skip_canonicalization = True
delete_p.multiple_results = True


@delete_p.def_abstract_eval
def delete_abstract_eval(*args):
    return args


@delete_p.def_impl
def delete_impl(*args):
    for a in args:
        a.delete()
    return args


class TransferDoneParams(TypedDict):
    pass


transfer_done_p = jcore.Primitive("transfer_done")
transfer_done_p.multiple_results = True


class CommunicationEffect(effects.Effect):
    def __str__(self):
        return "Communication"

    def __hash__(self):
        return hash(CommunicationEffect)

    def __eq__(self, other):
        return isinstance(other, CommunicationEffect)


communication_effect = CommunicationEffect()
effects.lowerable_effects.add_type(CommunicationEffect)
effects.control_flow_allowed_effects.add_type(CommunicationEffect)


@transfer_done_p.def_effectful_abstract_eval
def transfer_done_abstract_eval(tok, *args, **params: Unpack[TransferDoneParams]):
    del tok, params
    return args, frozenset({communication_effect})


@transfer_done_p.def_impl
def transfer_done_impl(tok: CommToken[Any], *args) -> tuple[Any, ...]:
    tok.done()
    return args


def _zeros(shapes_and_dtype):
    return tuple(jax.numpy.zeros(shape, dtype) for shape, dtype in shapes_and_dtype)


def _alloc_zeros(shape_and_dtype, shardings):
    local_shardings = tuple(shardings)
    mesh_context = contextlib.nullcontext()
    for sharding in local_shardings:
        if isinstance(sharding, jax.sharding.NamedSharding):
            mesh_context = jax.set_mesh(sharding.mesh)
            break

    with mesh_context:
        return jax.jit(
            _zeros, static_argnums=(0,), out_shardings=tuple(local_shardings)
        )(tuple(shape_and_dtype))


transfer_start_p = jcore.Primitive("transfer_start")
transfer_start_p.multiple_results = True

# transfer_start groups send starts and recv starts. Send inputs come first;
# recv-buffer inputs follow. For logical recvs, the impl allocates private
# buffers described by out_avals; for bufferized recvs, it aliases the input
# destination buffers. The token and recv buffer aliases are threaded into
# recv_done so finalize_lifetimes does not delete storage while NCCL is still
# writing into it.


@transfer_start_p.def_effectful_abstract_eval
def transfer_start_abstract_eval(
    *args,
    send_remote_shardings,
    send_local_shardings,
    recv_remote_shardings,
    recv_local_shardings,
    out_avals=None,
):
    del send_remote_shardings, recv_remote_shardings, recv_local_shardings
    send_count = len(send_local_shardings)
    if len(args) == send_count and out_avals is not None:
        return (TOKEN_AVAL, *out_avals), frozenset({communication_effect})
    return (TOKEN_AVAL, *args[send_count:]), frozenset({communication_effect})


@transfer_start_p.def_impl
def transfer_start_impl(
    *args: jax.Array,
    send_remote_shardings,
    send_local_shardings,
    recv_remote_shardings,
    recv_local_shardings,
    out_avals=None,
) -> tuple[Any, ...]:
    send_count = len(send_local_shardings)
    send_args = args[:send_count]
    recv_buffers = args[send_count:]
    if len(recv_buffers) == 0 and out_avals is not None and len(out_avals) > 0:
        recv_buffers = _alloc_zeros(
            [(aval.shape, aval.dtype) for aval in out_avals], recv_local_shardings
        )

    transfer = start_transfer(
        send_args, send_remote_shardings, recv_buffers, recv_remote_shardings
    )
    token: CommToken[Sequence[jax.Array]] = CommToken(transfer)
    return (token, *recv_buffers)


zeros_p = jcore.Primitive("zeros")
zeros_p.multiple_results = True


@zeros_p.def_abstract_eval
def zeros_abstract_eval(*, shape_and_dtype, shardings, out_avals):
    return out_avals


@zeros_p.def_impl
def zeros_impl(*, shape_and_dtype, shardings, out_avals):
    return _alloc_zeros(shape_and_dtype, shardings)


reuse_fence_p = jcore.Primitive("reuse_fence")
reuse_fence_p.skip_canonicalization = True


@reuse_fence_p.def_abstract_eval
def reuse_fence_abstract_eval(x):
    return x


@partial(jax.jit, donate_argnums=(0,))
def _reuse_fence(x):
    # This cheap donated identity gives PJRT a fresh output value whose
    # definition event is ordered after earlier JAX consumers of `x`. When the
    # output is later exported with `__dlpack__(stream=recv_stream)`, PJRT makes
    # that recv stream wait for the definition event. Donation normally aliases
    # the output to the same storage, so recv-buffer reuse gets the needed stream
    # dependency without a host block or a device copy.
    return x


@reuse_fence_p.def_impl
def reuse_fence_impl(x: jax.Array):
    return _reuse_fence(x)


recv_done_p = jcore.Primitive("recv_done")
recv_done_p.multiple_results = True

# recv_done returns arrays backed by the same destination storage passed through
# transfer_start. The DLPack view keeps that storage alive after deleting the
# temporary destination handles, so lifetime passes must treat recv_done outvars
# as aliases of the recv buffer inputs.


@recv_done_p.def_effectful_abstract_eval
def recv_done_abstract_eval(tok, *args):
    del tok
    return args, frozenset({communication_effect})


@recv_done_p.def_impl
def recv_done_impl(
    tok: CommToken[Sequence[jax.Array]], *buffers: jax.Array
) -> Sequence[jax.Array]:
    received = tok.done()
    # The DLPack import now owns a view/external reference for the received
    # arrays. The original destination buffers are dead after recv_done.
    for buffer in buffers:
        buffer.delete()
    return received


place_with_p = jcore.Primitive("place_with")
place_with_p.def_impl(lambda val, with_val: val)
place_with_p.def_abstract_eval(lambda val, with_val: val)
batching.defvectorized(place_with_p)
mlir.register_lowering(place_with_p, lambda ctx, val, with_val: [val])

pipeline_yield_p = jcore.Primitive("pipeline_yield")
pipeline_yield_p.multiple_results = True
pipeline_yield_p.def_impl(lambda *args, **kwargs: args)
pipeline_yield_p.def_abstract_eval(lambda *args, **kwargs: args)


def _pipeline_yield_batcher(args, dims, **kwargs):
    return pipeline_yield_p.bind(*args, **kwargs), dims


batching.primitive_batchers[pipeline_yield_p] = _pipeline_yield_batcher
mlir.register_lowering(pipeline_yield_p, lambda ctx, *args, **kwargs: args)


def pipeline_yield_transpose(ts, name: str, task_type: TaskType, stage_id: int):
    assert task_type == TaskType.FWD
    ts = [ad.instantiate_zeros(t) for t in ts]
    return pipeline_yield_p.bind(
        *ts, name=name, task_type=TaskType.BWD, stage_id=stage_id
    )


ad.deflinear(pipeline_yield_p, pipeline_yield_transpose)

mlir.register_lowering(pipeline_yield_p, lambda ctx, *args, **kwargs: args)


def dax_pscan_abstract_eval(
    *args,
    jaxpr,
    n_mubatches,
    n_consts,
    in_shardings,
    out_shardings,
    in_mpmd_refs,
    out_mpmd_defs,
    schedule,
):
    return jaxpr.out_avals


dax_pscan_p = jcore.Primitive("dax_pscan")
dax_pscan_p.multiple_results = True
# TODO: maybe make it a absract_effectful_eval?
dax_pscan_p.def_abstract_eval(dax_pscan_abstract_eval)


def dax_pscan_impl(
    *args,
    jaxpr: jcore.ClosedJaxpr,
    n_mubatches,
    n_consts,
    in_shardings,
    out_shardings,
    in_mpmd_refs,
    out_mpmd_defs,
    schedule,
    eager=False,
):
    # FIXME: acutally implement schedule
    fun = jcore.jaxpr_as_fun(jaxpr)

    if n_mubatches == 1:
        return fun(*args)

    loop_invariant_args, loop_state = args[:n_consts], args[n_consts:]

    if eager:
        for i in range(0, n_mubatches):
            loop_state = fun(*loop_invariant_args, *loop_state)
        return loop_state

    def loop_body(idx, loop_state):
        return fun(*loop_invariant_args, *loop_state)

    return jax.lax.fori_loop(0, n_mubatches, loop_body, list(loop_state))


dax_pscan_p.def_impl(partial(dax_pscan_impl, eager=True))

mlir.register_lowering(
    dax_pscan_p, mlir.lower_fun(dax_pscan_impl, multiple_results=True)
)


def task_lower(
    ctx,
    *args,
    call_jaxpr: jcore.ClosedJaxpr,
    task_name,
    task_info,
    mpmd_idx,
    in_shardings: ShardingTuple,
    out_shardings: ShardingTuple,
    donate_invars,
    latency: float | None = None,
    call_counter=None,
):
    return mlir.core_call_lowering(ctx, *args, name=task_name, call_jaxpr=call_jaxpr)


def dce_jaxpr_dax_pscan(
    used_outputs: list[bool], eqn: jcore.JaxprEqn
) -> tuple[list[bool], jcore.JaxprEqn]:
    jaxpr_ = eqn.params["jaxpr"]
    jaxpr, consts = jaxpr_.jaxpr, jaxpr_.consts

    has_changed = True
    while has_changed:
        has_changed = False
        new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs)
        for o_idx, (i, o) in enumerate(
            jc.safe_zip(used_inputs[eqn.params["n_consts"] :], used_outputs)
        ):
            if i and i != o:
                used_outputs[o_idx] = i
                has_changed = True

    # NOTE: it might happen that some output state is never merged with carried state
    #  (i.e. the `last` component of the LoopState).
    #  Here we make sure that the LoopState part of `used_inputs` agrees
    #  with `used_outputs`.
    for o_idx, (_, o) in enumerate(
        jc.safe_zip(used_inputs[eqn.params["n_consts"] :], used_outputs)
    ):
        used_inputs[eqn.params["n_consts"] + o_idx] = o

    new_jaxpr = new_jaxpr.replace(
        invars=[
            invar for invar, used in jc.safe_zip(jaxpr.invars, used_inputs) if used
        ],
        debug_info=None,  # FIXME
    )

    new_params = dict(
        eqn.params,
        n_consts=sum(used_inputs[: eqn.params["n_consts"]]),
        jaxpr=jcore.ClosedJaxpr(new_jaxpr, consts),
    )
    new_eqn = jcore.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs, strict=True) if used],
        [v for v, used in zip(eqn.outvars, used_outputs, strict=True) if used],
        eqn.primitive,
        new_params,
        new_jaxpr.effects,
        eqn.source_info,
    )
    return used_inputs, new_eqn


@dataclass(frozen=True, kw_only=True)
class PjitKwargs:
    jaxpr: jcore.ClosedJaxpr
    in_shardings: tuple[jax.sharding.NamedSharding, ...]
    out_shardings: tuple[jax.sharding.NamedSharding, ...] | jc.UnspecifiedValue
    in_layouts: tuple
    out_layouts: tuple
    donated_invars: tuple[bool, ...]
    ctx_mesh: jax.sharding.Mesh
    name: str = field(compare=False, hash=False)
    keep_unused: bool = True
    inline: bool = False
    compiler_options_kvs: tuple[tuple[str, Any], ...] | None = None

    def asdict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@jc.cache()
def callable_task(prim: jcore.Primitive, params: PjitKwargs):
    logging.info(f"Compiling {params.name} ({id(params.jaxpr)})")
    p = params.asdict()
    p["compiler_options_kvs"] = ()
    p["inline"] = jc.canonicalize_inline(p["inline"])
    if isinstance(params.out_shardings, jc.UnspecifiedValue):
        p["out_shardings"] = (params.out_shardings,) * len(params.jaxpr.out_avals)

    def prim_fun(*args):
        return tuple(prim.bind(*args, **p))

    prim_fun.__name__ = params.name
    prim_fun.__qualname__ = params.name
    prim_fun._apply_primitive = True
    return jax.jit(
        prim_fun,
        in_shardings=params.in_shardings,
        out_shardings=params.out_shardings,
        donate_argnums=tuple(
            idx for idx, donated in enumerate(params.donated_invars) if donated
        ),
        compiler_options=(
            dict(params.compiler_options_kvs)
            if params.compiler_options_kvs is not None
            else None
        ),
    )


def apply_task(prim: jcore.Primitive, *args, params: PjitKwargs):
    with jax.set_mesh(params.ctx_mesh), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Some donated buffers were not usable.*"
        )
        return callable_task(prim, params)(*args)


check_in_shardings = False

_statistics: dict[Any, list[float]] | None = None


def current_statistics():
    return _statistics


@contextlib.contextmanager
def collect_task_times_ms(
    enabled: bool = True,
) -> Generator[dict[str, list[float]] | None, None, None]:
    """Context manager to collect task execution times in milliseconds.

    Example usage::

        with collect_task_times_ms() as stats:
            # ... run tasks ...

        for task_name, times in stats.items():
            print(f"{task_name}: {times}")

    Example usage with collection disabled::

        with collect_task_times_ms(enabled=False) as stats:
            # ... run tasks ...

        assert stats is None
    """
    if not enabled:
        yield
        return

    global _statistics
    old_statistics = _statistics
    _statistics = defaultdict(list)
    try:
        yield _statistics
    finally:
        _statistics = old_statistics


def task_impl(
    *args,
    call_jaxpr: jcore.ClosedJaxpr,
    task_name,
    task_info,
    mpmd_idx,
    in_shardings: tuple[jax.NamedSharding, ...],
    out_shardings: tuple[jax.NamedSharding, ...],
    donate_invars,
    latency: float | None = None,
    call_counter: int | None = None,
):
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    mpmd_indices, mesh = _resolve_placement(mpmd_mesh, mpmd_idx, name="task mpmd_idx")

    pjit_kwargs = PjitKwargs(
        jaxpr=call_jaxpr,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        in_layouts=(None,) * len(args),
        out_layouts=(None,) * len(out_shardings),
        donated_invars=tuple(donate_invars),
        ctx_mesh=mesh,
        name=task_name,
    )

    # TODO(fixup_multidefs)
    if len(mpmd_indices) != 1 and any(isinstance(a, MpmdArray) for a in args):
        raise NotImplementedError("MpmdArray task inputs require a single MPMD index")
    array_mpmd_idx = mpmd_indices[0]
    maybe_pending_arrays = [
        a._partially_addressable_arrays[array_mpmd_idx]
        if isinstance(a, MpmdArray)
        else a
        for a in args
    ]

    if check_in_shardings:
        for arg_idx, _ in enumerate(maybe_pending_arrays):
            if not _._committed and not mpmd_mesh.jax_mesh.is_multi_process:
                continue

            arg_mpmd_indices = mpmd_mesh.mpmd_indices_for_mesh(_.sharding.mesh)
            if arg_mpmd_indices is None or arg_mpmd_indices != mpmd_indices:
                device_assignment = _.sharding._device_assignment
                raise ValueError(
                    f"Argument {arg_idx} for task {task_name} {call_counter=} "
                    f"@ {mpmd_idx=} found in {arg_mpmd_indices} "
                    f"({device_assignment})"
                )

    arrays = list(maybe_pending_arrays)

    statistics = current_statistics()
    if statistics is not None:
        arrays = jax.block_until_ready(arrays)

    enable_memstats = False
    with (
        jax.profiler.TraceAnnotation(f"{task_name}"),
        print_memstats(f"task_impl {task_name}", enabled=enable_memstats),
    ):
        start = time.perf_counter_ns() / 1_000_000

        res = apply_task(jc.jit_p, *arrays, params=pjit_kwargs)
        if enable_memstats or statistics is not None:
            jax.block_until_ready(res)

        end = time.perf_counter_ns() / 1_000_000

    if statistics is not None:
        statistics[f"{task_name} ({id(call_jaxpr)})"].append(end - start)

    return res


def task_abstract_eval(
    *args,
    call_jaxpr: jcore.ClosedJaxpr,
    name=None,
    task_name,
    task_info,
    mpmd_idx,
    in_shardings: ShardingTuple,
    out_shardings: ShardingTuple,
    donate_invars,
    latency: float | None = None,
    call_counter=None,
):
    return (call_jaxpr.out_avals, call_jaxpr.effects)


task_p = jcore.Primitive("task")
# NOTE: `jcore.canonicalize_value` called on all args of a bind
# calls `jcore.get_aval` which builds an exception that formats
# (blocks and memcopies to host) which is then discarded by
# `jcore.canonicalize_value`. We skip_canonicalization to make dispatch
# fast
task_p.skip_canonicalization = True
task_p.multiple_results = True
# TODO: use `task_impl` above once fixed.
# As of now tasks aren't jitted
task_p.def_impl(task_impl)
task_p.def_effectful_abstract_eval(task_abstract_eval)

mlir.register_lowering(task_p, task_lower)


@jc.register_discharge_rule(task_p)
def _task_state_discharge_rule(
    context_or_in_avals, *args, call_jaxpr: jcore.ClosedJaxpr, **params
):
    if jc.discharge_rule_uses_context:
        context = context_or_in_avals
        in_avals = context.in_avals
        out_avals = context.out_avals
        discharged_call_jaxpr = jc.discharge_state(
            call_jaxpr, strip_memory_space=context.strip_memory_space
        )
    else:
        in_avals = context_or_in_avals
        out_avals, *args = args
        discharged_call_jaxpr = jc.discharge_state(call_jaxpr)

    if len(discharged_call_jaxpr.out_avals) != len(out_avals):
        raise NotImplementedError("JaxPP task Ref inputs are unsupported")

    out = task_p.bind(*args, call_jaxpr=discharged_call_jaxpr, **params)
    return [None] * len(in_avals), out


pe.dce_rules[dax_pscan_p] = dce_jaxpr_dax_pscan


def _dce_shardings(shardings: ShardingTuple, used: Sequence[bool]) -> ShardingTuple:
    return tuple(s for s, keep in zip(shardings, used, strict=True) if keep)


def _task_dce_rule(
    used_outputs: list[bool], eqn: jcore.JaxprEqn
) -> tuple[list[bool], jcore.JaxprEqn | None]:
    if not any(used_outputs) and not jc.has_effects(eqn):
        return [False] * len(eqn.invars), None

    call_jaxpr = eqn.params["call_jaxpr"]
    new_jaxpr, used_inputs = pe.dce_jaxpr(call_jaxpr.jaxpr, used_outputs)
    new_closed_jaxpr = jcore.ClosedJaxpr(new_jaxpr, call_jaxpr.consts)

    new_params = dict(eqn.params)
    new_params["call_jaxpr"] = new_closed_jaxpr
    new_params["in_shardings"] = _dce_shardings(eqn.params["in_shardings"], used_inputs)
    new_params["out_shardings"] = _dce_shardings(
        eqn.params["out_shardings"], used_outputs
    )
    new_params["donate_invars"] = tuple(
        donated
        for donated, used in zip(eqn.params["donate_invars"], used_inputs, strict=True)
        if used
    )

    new_invars = [v for v, used in zip(eqn.invars, used_inputs, strict=True) if used]
    new_outvars = [v for v, used in zip(eqn.outvars, used_outputs, strict=True) if used]
    new_eqn = jcore.new_jaxpr_eqn(
        new_invars,
        new_outvars,
        eqn.primitive,
        new_params,
        jc.eqn_effects(new_closed_jaxpr, new_invars),
        eqn.source_info,
        eqn.ctx,
    )
    return used_inputs, new_eqn


pe.dce_rules[task_p] = _task_dce_rule


# Refined type annotations for key Jaxprs/Eqns we use in the jaxpr
class TaskEqnParams(TypedDict):
    call_jaxpr: jcore.ClosedJaxpr
    task_name: str
    task_info: tuple[int, TaskType] | None
    mpmd_idx: int | jax.sharding.Mesh
    in_shardings: ShardingTuple
    out_shardings: ShardingTuple
    donate_invars: tuple[bool, ...]
    latency: float | None
    call_counter: NotRequired[int | None]


class TaskEqn(jcore.JaxprEqn):
    invars: list[jcore.Var]  # Unique
    params: TaskEqnParams

    def replace(
        self,
        invars: list[jcore.Var] | None = None,
        outvars: list[jcore.Var] | None = None,
        primitive: jcore.Primitive | None = None,
        params: TaskEqnParams | None = None,
        effects: jcore.Effects | None = None,
        source_info: jsiu.SourceInfo | None = None,
    ):
        pass

    @staticmethod
    def make(eqn: jcore.JaxprEqn) -> "TaskEqn":
        assert eqn.primitive is task_p
        for invar in eqn.invars:
            assert isinstance(invar, jcore.Var), "Pipeline stage has literal arguments"
        for outvar in eqn.params["call_jaxpr"].jaxpr.outvars:
            assert isinstance(outvar, jcore.Var), "Pipeline stage has literal results"
        assert len(eqn.invars) == len(set(eqn.invars)), "Duplicate arguments to stage"
        return cast(TaskEqn, eqn)


class PscanJaxpr(jcore.Jaxpr):
    @property
    def eqns(self) -> list[TaskEqn]: ...

    @property
    def outvars(self) -> list[jcore.Var]: ...

    @staticmethod
    def make(jaxpr: jcore.Jaxpr) -> "PscanJaxpr":
        for eqn in jaxpr.eqns:
            TaskEqn.make(eqn)
        # NOTE: also checks that it doesn't have literal outvars
        assert len(set(jaxpr.invars) & set(jaxpr.outvars)) == 0
        return cast(PscanJaxpr, jaxpr)
