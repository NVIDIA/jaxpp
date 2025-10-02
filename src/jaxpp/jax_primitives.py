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
from contextlib import contextmanager
from functools import partial
from typing import Callable, Optional, ParamSpec, Sequence, TypedDict, TypeVar, cast

import jax
import jax._src.core as jcore
import jax._src.util as ju
from jax._src import effects, source_info_util
from jax._src.debugging import debug_effect, inspect_sharding_p

if jax.__version_info__ < (0, 7, 0):
    from jax._src.pjit import pjit_p as jit_p
else:
    from jax._src.pjit import jit_p
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

from jaxpp.array import MpmdArray
from jaxpp.dime2 import send_or_recv
from jaxpp.mesh import MpmdMesh
from jaxpp.types import TaskType
from jaxpp.utils import get_named_sharding, updated_named_sharding_mesh

logger = logging.getLogger(__name__)

add_multi_p = jcore.Primitive("add_multi")


@add_multi_p.def_abstract_eval
def add_multi_abstract_eval(
    *args,
    in_shardings: Optional["ShardingStore"] = None,
    out_shardings: Optional["ShardingStore"] = None,
    mpmd_idxs=None,
    donate_invars=None,
):
    first = args[0]
    assert all(first.dtype == arg.dtype for arg in args)
    return first


@add_multi_p.def_impl
def add_multi_impl(
    *args,
    in_shardings: Optional["ShardingStore"] = None,
    out_shardings: Optional["ShardingStore"] = None,
    mpmd_idxs=None,
    donate_invars=None,
):
    assert mpmd_idxs is not None
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    assert (
        not mpmd_mesh.jax_mesh.is_multi_process
    ), f"{add_multi_p.name} supported only in single-process runtime"
    prev_shardings: list[jax.NamedSharding] = [a.sharding for a in args]
    # TODO: do the stacking similarly to logically_stacked
    _ = sum(jax.device_put(a, args[0].sharding) for a in args)
    return MpmdArray(
        [jax.device_put(_, s) for s in prev_shardings],
        mpmd_mesh=mpmd_mesh,
        mpmd_idxs=mpmd_idxs,
    )


def add_multi_lower(
    *args,
    in_shardings: Optional["ShardingStore"] = None,
    out_shardings: Optional["ShardingStore"] = None,
    mpmd_idxs=None,
    donate_invars=None,
):
    return sum(args)


mlir.register_lowering(
    add_multi_p, mlir.lower_fun(add_multi_lower, multiple_results=False)
)


def all_reduce_fn(arrs):
    return tuple(a.sum(0) for a in arrs)


def logically_stacked(a: jax.Array, comm_mesh: jax.sharding.Mesh, axis_name: str):
    spec = a.sharding.spec
    assert axis_name not in spec
    exp = jax.numpy.expand_dims(a, 0)
    in_sharding = jax.sharding.NamedSharding(
        comm_mesh,
        jax.sharding.PartitionSpec(axis_name, *spec),
    )
    ga = jax.make_array_from_single_device_arrays(
        (comm_mesh.shape[axis_name], *a.shape),
        in_sharding,
        [s.data for s in exp.addressable_shards],
    )
    return ga


def all_reduce(
    arrs: list[jax.Array],
    comm_mesh: jax.sharding.Mesh,
    axis_name: str,
    donated: list[bool] | None = None,
):
    shardings = [get_named_sharding(a) for a in arrs]
    assert len(set(_.mesh for _ in shardings)) == 1

    plogically_stacked = partial(
        logically_stacked, comm_mesh=comm_mesh, axis_name=axis_name
    )
    gas = tuple(plogically_stacked(a) for a in arrs)

    all_reduced: tuple[jax.Array, ...] = jax.jit(
        all_reduce_fn,
        in_shardings=tuple(a.sharding for a in gas),
        out_shardings=tuple(
            jax.sharding.NamedSharding(comm_mesh, sh.spec) for sh in shardings
        ),
        donate_argnums=donated,
    )(gas)

    res = []
    for a, sh in zip(all_reduced, shardings, strict=True):
        res.append(
            jax.make_array_from_single_device_arrays(
                a.shape, sh, [s.data for s in a.addressable_shards]
            )
        )

    return res


all_reduce_p = jcore.Primitive("all_reduce")


@all_reduce_p.def_abstract_eval
def all_reduce_abstract_eval(arg, mpmd_idxs: list[int]):
    return arg


# TODO: support multi-arity all_reduce
@all_reduce_p.def_impl
def all_reduce_impl(arg, mpmd_idxs: list[int]):
    _check_no_attrs(arg)

    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    comm_mesh = mpmd_mesh.mpmd_submesh(mpmd_idxs).jax_mesh
    return all_reduce(
        [arg],
        comm_mesh=comm_mesh,
        axis_name=mpmd_mesh.mpmd_axis_name,
        donated=[False],  # FIXME
    )[0]


transfer_p = jcore.Primitive("transfer")
transfer_p.multiple_results = True


@transfer_p.def_abstract_eval
def transfer_abstract_eval(*args, src_mpmd_idx, tgt_mpmd_idx, src_shardings):
    return args


@transfer_p.def_impl
def transfer_impl(*args, src_mpmd_idx, tgt_mpmd_idx, src_shardings):
    args = list(args)
    mpmd_mesh = MpmdMesh.mesh_stack[-1]

    for a, sh in zip(args, src_shardings, strict=True):
        assert isinstance(a, jax.Array)
        assert a.sharding == sh

    tgt_shardings = updated_named_sharding_mesh(
        [a.sharding for a in args], mpmd_mesh.unstack[tgt_mpmd_idx]
    )
    res = jax.device_put(args, tgt_shardings)
    return res


delete_p = jcore.Primitive("delete")
# NOTE: we have delete equations for donated buffers as well
#  which fail if Jax tries to canonicalize them.
#  Hence we skip canonicalization for delete
delete_p.skip_canonicalization = True
delete_p.multiple_results = True


@delete_p.def_abstract_eval
def delete_abstract_eval(*args, mpmd_idx):
    return args


@delete_p.def_impl
def delete_impl(*args, mpmd_idx):
    for a in args:
        assert not hasattr(a, "_ensure_receive_enqueued")
        # TODO(fixup_multidefs)
        if isinstance(a, MpmdArray):
            a._partially_addressable_arrays[mpmd_idx].delete()
            del a._partially_addressable_arrays[mpmd_idx]
        else:
            a.delete()
    return args


send_done_p = jcore.Primitive("send_done")
send_done_p.multiple_results = True


class _LifetimeEndEffect(effects.Effect):
    def __str__(self):
        return "LifetimeEnd"


LifetimeEndEffect = _LifetimeEndEffect()


@contextmanager
def print_memstats(label: str, enabled: bool = False):
    if not enabled:
        yield
        return
    print(f"\nBefore: {label}:")
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"] / 2**30
        limit = stats["bytes_limit"] / 2**30
        peak_size = stats["peak_bytes_in_use"] / 2**30
        print(
            f"\tUsing (GB) {used:.2f} / {limit:.2f} ({used/limit:%}) ({peak_size=:.2f} GiB) on {d}",
            flush=True,
        )

    yield

    print(f"\nAfter: {label}:")
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"] / 2**30
        limit = stats["bytes_limit"] / 2**30
        peak_size = stats["peak_bytes_in_use"] / 2**30
        print(
            f"\tUsing (GB) {used:.2f} / {limit:.2f} ({used/limit:%}) ({peak_size=:.2f} GiB) on {d}",
            flush=True,
        )


@send_done_p.def_effectful_abstract_eval
def send_done_abstract_eval(*args, mpmd_idx):
    return args, frozenset({LifetimeEndEffect})


@send_done_p.def_impl
def send_done_impl(*args, mpmd_idx):
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    if not mpmd_mesh.jax_mesh.is_multi_process:
        return args

    for a in args:
        if hasattr(a, "_wait_send_finish"):
            a._wait_send_finish()
        # if hasattr(a, "_ensure_receive_enqueued"):
        #     a._ensure_receive_enqueued()
    return args


def _check_no_attrs(a: jax.Array):
    assert not hasattr(a, "_wait_send_finish") and not hasattr(
        a, "_ensure_receive_enqueued"
    )


send_p = jcore.Primitive("send")
send_p.multiple_results = True


@send_p.def_abstract_eval
def send_abstract_eval(*args, id, shardings):
    return args


@send_p.def_impl
def send_impl(*arrs, id, shardings):
    tgt_mpmd_idxs, receiver_shardings = jax._src.util.unzip2(shardings)
    for a, _wait_send_finish in zip(
        arrs,
        send_or_recv(arrs, remote_shardings=receiver_shardings, is_send=True),
        strict=True,
    ):
        if hasattr(a, "_wait_send_finish"):
            # FIXME(multi_send_done): we overwrite the existing
            #  send_finish below. We should "join" them.
            #  This is not an issue now as `send_or_recv` sets a finalizer
            #  for the returned value that ensures waiting for the send to
            #  finish before the arrays is deleted.
            pass
        a._wait_send_finish = _wait_send_finish
    return arrs


recv_p = jcore.Primitive("recv")
recv_p.multiple_results = True


@recv_p.def_abstract_eval
def recv_abstract_eval(*args, id, shardings, shape_and_dtype):
    return (
        args  # [jcore.ShapedArray(shape, dtype) for (shape, dtype) in shape_and_dtype]
    )


def _zeros(shapes_and_dtype):
    return tuple(jax.numpy.zeros(shape, dtype) for shape, dtype in shapes_and_dtype)


@recv_p.def_impl
def recv_impl(*buffers, id, shardings, shape_and_dtype):
    src_mpmd_idxs, sender_shardings = jax._src.util.unzip2(shardings)

    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    my_mesh = mpmd_mesh.unstack[mpmd_mesh.my_mpmd_axis_index]
    local_shardings = updated_named_sharding_mesh(sender_shardings, new_mesh=my_mesh)
    if len(buffers) > 0:
        assert len(buffers) == len(shape_and_dtype), (
            len(buffers),
            len(shape_and_dtype),
        )
    else:
        buffers = jax.jit(
            _zeros, static_argnums=(0,), out_shardings=tuple(local_shardings)
        )(tuple(shape_and_dtype))

    enqueues = send_or_recv(buffers, remote_shardings=sender_shardings, is_send=False)
    for buf, _ensure_receive_enqueued in zip(buffers, enqueues, strict=True):
        buf._ensure_receive_enqueued = _ensure_receive_enqueued

    return buffers


pipeline_yield_p = jcore.Primitive("pipeline_yield")
pipeline_yield_p.multiple_results = True
pipeline_yield_p.def_impl(lambda *args, **kwargs: args)
pipeline_yield_p.def_abstract_eval(lambda *args, **kwargs: args)


def pipeline_yield_batcher(args, dims, **kwargs):
    return pipeline_yield_p.bind(*args, **kwargs), dims


batching.primitive_batchers[pipeline_yield_p] = pipeline_yield_batcher
mlir.register_lowering(pipeline_yield_p, lambda ctx, *args, **kwargs: args)


def pipeline_yield_transpose(ts, **kwargs):
    assert kwargs["task_type"] == TaskType.FWD
    ts = [ad.instantiate_zeros(t) for t in ts]
    return pipeline_yield_p.bind(
        *ts,
        **(
            kwargs
            | {
                "task_type": TaskType.BWD,
                "from_stage_id": kwargs["to_stage_id"],
                "to_stage_id": kwargs["from_stage_id"],
            }
        ),
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
    jaxpr,
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


def _task_transpose_update_params(params, undef_primals, nonzero_cts):
    return dict(params, task_name=f"bwd({params['task_name']})")


def task_lower(
    ctx,
    *args,
    backend=None,
    call_jaxpr: jcore.ClosedJaxpr,
    task_name,
    task_info,
    mpmd_idx,
    in_shardings: "ShardingStore",
    out_shardings: "ShardingStore",
    donate_invars,
    latency: float | None = None,
    call_counter=None,
):
    return mlir.core_call_lowering(
        ctx, *args, name=task_name, backend=backend, call_jaxpr=call_jaxpr
    )


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
            jax._src.util.safe_zip(used_inputs[eqn.params["n_consts"] :], used_outputs)
        ):
            if i and i != o:
                used_outputs[o_idx] = i
                has_changed = True

    # NOTE: it might happen that some output state is never merged with carried state
    #  (i.e. the `last` component of the LoopState).
    #  Here we make sure that the LoopState part of `used_inputs` agrees
    #  with `used_outputs`.
    for o_idx, (_, o) in enumerate(
        jax._src.util.safe_zip(used_inputs[eqn.params["n_consts"] :], used_outputs)
    ):
        used_inputs[eqn.params["n_consts"] + o_idx] = o

    new_jaxpr = new_jaxpr.replace(
        invars=[
            invar
            for invar, used in jax._src.util.safe_zip(jaxpr.invars, used_inputs)
            if used
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


@ju.cache()
def callable_task(prim: jcore.Primitive, **params):
    logging.info(f"Compiling {params['name']}")

    def prim_fun(*args):
        return prim.bind(*args, **params)

    prim_fun.__name__ = params["name"]
    prim_fun.__qualname__ = params["name"]
    prim_fun._apply_primitive = True
    return jax.jit(
        prim_fun,
        in_shardings=params["in_shardings"],
        out_shardings=list(params["out_shardings"]),
        donate_argnums=tuple(
            idx for idx, donated in enumerate(params["donated_invars"]) if donated
        ),
    )


def apply_task(prim: jcore.Primitive, *args, **params):
    with params["ctx_mesh"]:
        if jax.__version_info__ < (0, 5, 3):
            from jax._src.mesh import ResourceEnv

            params["resource_env"] = ResourceEnv(physical_mesh=params["ctx_mesh"])
            del params["ctx_mesh"]
        return callable_task(prim, **params)(*args)


check_in_shardings = False


def task_impl(
    *args,
    call_jaxpr: jcore.ClosedJaxpr,
    backend=None,
    task_name,
    task_info,
    mpmd_idx,
    in_shardings: list[jax.NamedSharding],
    out_shardings: list[jax.NamedSharding],
    donate_invars,
    latency: float | None = None,
    call_counter: int | None = None,
):
    mpmd_mesh = MpmdMesh.mesh_stack[-1]
    mesh = mpmd_mesh.unstack[mpmd_idx]

    pjit_kwargs = dict(
        jaxpr=call_jaxpr,
        in_shardings=tuple(in_shardings),
        out_shardings=tuple(out_shardings),
        in_layouts=(None,) * len(args),
        out_layouts=(None,) * len(out_shardings),
        donated_invars=tuple(donate_invars),
        ctx_mesh=mesh,
        name=task_name,
        # + (
        #     f"_{call_counter}" if call_counter is not None else ""
        # ),  # FIXME: remove call_counter
        keep_unused=True,
        inline=False,
        compiler_options_kvs=tuple(),
    )

    # TODO(fixup_multidefs)
    maybe_pending_arrays = [
        a._partially_addressable_arrays[mpmd_idx] if isinstance(a, MpmdArray) else a
        for a in args
    ]

    if check_in_shardings:
        for arg_idx, _ in enumerate(maybe_pending_arrays):
            if not _._committed and not mpmd_mesh.jax_mesh.is_multi_process:
                continue

            if (
                arg_mpmd_idx := mpmd_mesh.mpmd_idx_for_mesh.get(_.sharding.mesh)
            ) is None or arg_mpmd_idx != mpmd_idx:
                raise ValueError(
                    f"Argument {arg_idx} for task {task_name} {call_counter=} @ {mpmd_idx=} found in {arg_mpmd_idx} ({_.sharding._device_assignment})"
                )

    arrays = []
    for idx, a, is_donated in zip(
        range(len(maybe_pending_arrays)),
        maybe_pending_arrays,
        donate_invars,
        strict=True,
    ):
        if (_wait_send_finish := getattr(a, "_wait_send_finish", None)) is not None:
            # NOTE: it's fine to continue using an array that
            #   has been sent
            pass
        if (
            _ensure_receive_enqueued := getattr(a, "_ensure_receive_enqueued", None)
        ) is not None:
            assert not is_donated, f"{task_name=} {call_counter=} arg_idx={idx}"
            arrays.append(_ensure_receive_enqueued())
        else:
            arrays.append(a)

    with print_memstats(f"task_impl {task_name}"):
        res = apply_task(jit_p, *arrays, **pjit_kwargs)
        # jax.block_until_ready(res)
    return res


def task_abstract_eval(
    *args,
    call_jaxpr: jcore.ClosedJaxpr,
    name=None,
    backend=None,
    task_name,
    task_info,
    mpmd_idx,
    in_shardings: "ShardingStore",
    out_shardings: "ShardingStore",
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
T = TypeVar("T")
P = ParamSpec("P")


def _task(fun, name: str, *args, **kwargs):
    jaxpr, out_shapes = jax.make_jaxpr(partial(fun, **kwargs), return_shape=True)(*args)
    flat_args = jax.tree_util.tree_leaves(args)
    out_tree = jax.tree_util.tree_structure(out_shapes)
    res = task_p.bind(*flat_args, task_type=TaskType.FWD, call_jaxpr=jaxpr)
    return jax.tree_util.tree_unflatten(out_tree, res)


def task(fun: Callable[P, T], *, name: str | None = None) -> Callable[P, T]:
    return partial(_task, fun, name)


mlir.register_lowering(task_p, task_lower)

# FIXME: use closed_call_transpose below
ad.primitive_transposes[task_p] = partial(ad.call_transpose, task_p)
ad.call_transpose_param_updaters[task_p] = _task_transpose_update_params
pe.dce_rules[task_p] = pe.dce_jaxpr_closed_call_rule
pe.dce_rules[dax_pscan_p] = dce_jaxpr_dax_pscan


class ShardingStore:
    def __init__(
        self,
        avals: Sequence[jcore.ShapedArray],
        _provenance_info=None,
        _source_info=None,
        _shardings=None,
    ):
        self.avals = avals
        self._provenance_info = _provenance_info
        self._source_info = _source_info
        if _shardings:
            self._shardings = _shardings
            self._called_at_least_once = True
        else:
            self._shardings = [None] * len(avals)
            self._called_at_least_once = False

    def __len__(self):
        return len(self.shardings)

    def __getitem__(self, index):
        return self.shardings[index]

    def __str__(self):
        if self._called_at_least_once:
            metadata = (
                "["
                + "\n".join(
                    [
                        str(((str(aval.dtype), aval.shape), s.spec))
                        for aval, s in zip(self.avals, self.shardings, strict=True)
                    ]
                )
                + "]"
            )
            return metadata
        else:
            return repr(self)

    @property
    def shardings(self) -> list[jax.NamedSharding]:
        if len(self._shardings) > 0 and not self._called_at_least_once:
            raise AssertionError(
                "Shardings can be inspected only after compiling the jaxpr"
            )
        assert all(s is not None for s in self._shardings)
        return self._shardings

    def _callback_at_index(self, idx: int):
        def cb(s: jax.sharding.NamedSharding):
            self._called_at_least_once = True
            # NOTE: Checks that the inferred sharding is valid
            #  for this shape
            s.shard_shape(self.avals[idx].shape)
            self._shardings[idx] = s

        # This is helpful for debugging when `InspectSharding` fails
        # cb.info = (self, idx)
        return cb

    @classmethod
    def collect(
        cls, values: Sequence[jax.Array], _provenance_info=None, _source_info=None
    ) -> "ShardingStore":
        store = cls(
            [v.aval for v in values],
            _provenance_info=_provenance_info,
            _source_info=_source_info,
        )
        for idx, v in enumerate(values):
            jax.debug.inspect_array_sharding(v, callback=store._callback_at_index(idx))
        return store

    @classmethod
    def collect_jaxpr(
        cls, vars_: Sequence[jcore.Var], _provenance_info=None, _source_info=None
    ) -> tuple["ShardingStore", list[jcore.JaxprEqn]]:
        store = cls(
            [v.aval for v in vars_],
            _provenance_info=_provenance_info,
            _source_info=_source_info,
        )

        res = []
        for idx, v in enumerate(vars_):
            res.append(
                jcore.new_jaxpr_eqn(
                    invars=[v],
                    outvars=[],
                    primitive=inspect_sharding_p,
                    params={"callback": store._callback_at_index(idx)},
                    effects=frozenset({debug_effect}),
                )
            )
        return store, res


# Refined type annotations for key Jaxprs/Eqns we use in the jaxpr
class TaskEqnParams(TypedDict):
    call_jaxpr: jcore.ClosedJaxpr
    task_type: TaskType
    stage_id: int
    out_shardings: ShardingStore


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
        source_info: source_info_util.SourceInfo | None = None,
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
