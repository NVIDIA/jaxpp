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

import dataclasses
from collections.abc import Sequence
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    Unpack,
    runtime_checkable,
)

import jax
import jax.extend.linear_util as lu
from jax import api_util
from jax._src import mesh as mesh_lib
from jax._src import sharding_impls
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxpp import array_ops
from jaxpp import jax_compat as jc
from jaxpp.core import LocalJaxpr, add_deletes, to_local_jaxprs
from jaxpp.core import infer_donation as infer_donation_pass
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import (
    TOKEN_AVAL,
    SliceParams,
    StackParams,
    TaskEqnParams,
    TransferDoneParams,
    TransferParams,
    communication_effect,
    slice_p,
    stack_p,
    task_p,
    transfer_done_p,
    transfer_p,
)
from jaxpp.mesh import MpmdMesh, _require_mpmd_indices
from jaxpp.utils import PyTree

T = TypeVar("T")
P = ParamSpec("P")
_T_co = TypeVar("_T_co", covariant=True)


ShardingPyTree: TypeAlias = PyTree[NamedSharding]
TaskShardingPyTree: TypeAlias = PyTree[NamedSharding | PartitionSpec | None]
ShapeDtypePyTree: TypeAlias = PyTree[jax.ShapeDtypeStruct]


class MpmdTaskParams(TypedDict):
    call_jaxpr: jcore.ClosedJaxpr
    task_name: str
    mesh: Mesh | None
    in_shardings: tuple[object, ...]
    out_shardings: tuple[NamedSharding, ...]
    donate_invars: tuple[bool, ...]


class MpmdTransferParams(TypedDict):
    out_shardings: tuple[NamedSharding, ...]


def _outside_mpmd_impl(name):
    def impl(*args, **params):
        del args, params
        raise ValueError(
            f"jaxpp.experimental.mpmd.{name} must be called inside "
            "jaxpp.experimental.mpmd"
        )

    return impl


mpmd_transfer_p = jcore.Primitive("mpmd_transfer")
mpmd_transfer_p.multiple_results = True


@mpmd_transfer_p.def_effectful_abstract_eval
def _mpmd_transfer_abstract_eval(*args, **params: Unpack[MpmdTransferParams]):
    out_shardings = params["out_shardings"]
    del out_shardings
    return (TOKEN_AVAL, *args), frozenset({communication_effect})


mpmd_transfer_p.def_impl(_outside_mpmd_impl("transfer"))


mpmd_task_p = jcore.Primitive("mpmd_task")
mpmd_task_p.skip_canonicalization = True
mpmd_task_p.multiple_results = True


@mpmd_task_p.def_effectful_abstract_eval
def _mpmd_task_abstract_eval(*args, **params: Unpack[MpmdTaskParams]):
    del args
    call_jaxpr = params["call_jaxpr"]
    return call_jaxpr.out_avals, call_jaxpr.effects


mpmd_task_p.def_impl(_outside_mpmd_impl("task"))


class MpmdTracer(jcore.Tracer):
    def __init__(self, trace, val, sharding: NamedSharding | None):
        jc.init_tracer(self, trace, jc.get_aval(val))
        self.val = val
        self.mpmd_sharding = sharding
        self._sent = False

    @property
    def aval(self):
        return self._aval

    @aval.setter
    def aval(self, aval):
        self._aval = aval


class TransferFuture(Protocol[_T_co]):
    def done(self) -> _T_co:
        """Wait for the transfer and return the received values.

        On the receiver, task dispatches following this operation wait for this
        transfer even if they do not consume its outputs.
        """
        ...


class _TransferFutureImpl:
    """Trace-time future for completing a transfer and exposing its value."""

    __slots__ = ("_done", "_token", "_transferred", "_tree")

    def __init__(self, token, tree, transferred):
        self._token = token
        self._tree = tree
        self._transferred = tuple(transferred)
        self._done = False

    def done(self):
        if self._done:
            raise RuntimeError("transfer future done() may only be called once")
        transferred = transfer_done_p.bind(self._token, *self._transferred)
        self._done = True
        return jax.tree_util.tree_unflatten(self._tree, transferred)


def _validate_named_sharding(
    sharding, *, name: str, mpmd_mesh: MpmdMesh
) -> NamedSharding:
    if not isinstance(sharding, NamedSharding):
        raise TypeError(f"{name} must be a NamedSharding, got {type(sharding)}")
    _require_mpmd_indices(mpmd_mesh, sharding.mesh, name=name)
    return sharding


def _validate_flat_shardings(
    shardings: Sequence[object], *, name: str, mpmd_mesh: MpmdMesh
) -> tuple[NamedSharding, ...]:
    return tuple(
        _validate_named_sharding(sharding, name=f"{name}[{idx}]", mpmd_mesh=mpmd_mesh)
        for idx, sharding in enumerate(shardings)
    )


def _common_mesh(shardings: Sequence[NamedSharding], *, name: str):
    if not shardings:
        return None
    mesh = shardings[0].mesh
    if any(sharding.mesh != mesh for sharding in shardings[1:]):
        raise ValueError(f"{name} must all use the same mesh")
    return mesh


def _reconcile_meshes(
    entries: Sequence[tuple[str, Mesh | None]], *, mesh: Mesh | None = None
) -> Mesh | None:
    """Fold the meshes in `entries` into the single mesh they must all share.

    `mesh` seeds the expected mesh; when it is None the first non-None entry is
    adopted. Entries whose mesh is None are skipped, and each entry's name labels
    its mesh in the error raised on disagreement.
    """
    for name, candidate in entries:
        if candidate is None:
            continue
        if mesh is not None and mesh != candidate:
            raise ValueError(
                f"Context mesh {mesh} must match the mesh passed to "
                f"{name} {candidate}. Task shardings should have the same mesh."
            )
        mesh = candidate
    return mesh


def _infer_task_mesh(args, kwargs, mesh, in_shardings, out_shardings) -> Mesh | None:
    arg_shardings = [
        leaf.mpmd_sharding
        for leaf in jax.tree_util.tree_leaves((args, kwargs))
        if isinstance(leaf, MpmdTracer) and leaf.mpmd_sharding is not None
    ]
    sources = (
        ("task argument shardings", arg_shardings),
        ("task in_shardings", in_shardings),
        ("task out_shardings", out_shardings),
    )
    entries = [
        (name, leaf.mesh)
        for name, shardings in sources
        for leaf in jax.tree_util.tree_leaves(shardings)
        if isinstance(leaf, NamedSharding)
    ]

    task_mesh = _reconcile_meshes(entries, mesh=mesh)
    if task_mesh is None:
        has_partition_spec = any(
            isinstance(leaf, PartitionSpec)
            for leaf in jax.tree_util.tree_leaves((in_shardings, out_shardings))
        )
        if has_partition_spec:
            raise RuntimeError(
                "task requires a non-empty mesh if you are passing "
                "`PartitionSpec`s to in_shardings or out_shardings. Pass "
                "`mesh=` or provide `NamedSharding`s to make the mesh explicit."
            )
        return None

    return task_mesh


@contextmanager
def _task_mesh_context(mesh: Mesh | None):
    if mesh is None:
        yield
        return

    # `jax.set_mesh` rejects an active trace, so install the concrete and
    # abstract mesh contexts directly, the same way pjit does while tracing.
    with (
        sharding_impls._internal_use_concrete_mesh(mesh),
        mesh_lib.use_abstract_mesh(mesh.abstract_mesh),
    ):
        yield


def _tracer_sharding(tracer, *, name: str) -> NamedSharding:
    if not isinstance(tracer, MpmdTracer):
        raise TypeError(f"{name} must be an mpmd traced value, got {type(tracer)}")
    if tracer.mpmd_sharding is None:
        raise TypeError(f"{name} does not have a NamedSharding")
    return tracer.mpmd_sharding


class MpmdTrace(jcore.Trace):
    def __init__(self, parent_trace: jcore.Trace, mpmd_mesh: MpmdMesh):
        super().__init__()
        self.parent_trace = parent_trace
        self.mpmd_mesh = mpmd_mesh

    def call_parent(self, primitive, tracers, params, *, out_shardings):
        parent_tracers = tuple(
            tracer.val if isinstance(tracer, MpmdTracer) else tracer
            for tracer in tracers
        )
        avals = tuple(jc.get_aval(tracer) for tracer in parent_tracers)
        results = jc.bind_with_trace(
            primitive, self.parent_trace, parent_tracers, avals, dict(params)
        )

        if not primitive.multiple_results:
            return MpmdTracer(self, results, out_shardings)

        if len(results) != len(out_shardings):
            raise AssertionError(
                f"{primitive.name} returned {len(results)} values, "
                f"but {len(out_shardings)} shardings were inferred"
            )
        return [
            MpmdTracer(self, result, sharding)
            for result, sharding in zip(results, out_shardings, strict=True)
        ]

    def process_primitive(self, primitive, tracers, params):
        if (rule := mpmd_rules.get(primitive)) is None:
            raise NotImplementedError(
                "jaxpp.experimental.mpmd currently accepts only task, "
                "transfer, transfer_done, stack, and slice "
                f"primitives, got {primitive.name}"
            )
        return rule(self, *tracers, **params)


def task_mpmd(trace: MpmdTrace, *tracers, **params: Unpack[MpmdTaskParams]):
    in_shardings = tuple(
        _tracer_sharding(tracer, name=f"task argument {idx}")
        for idx, tracer in enumerate(tracers)
    )
    in_mesh = _common_mesh(in_shardings, name="task argument shardings")

    out_shardings = _validate_flat_shardings(
        params["out_shardings"], name="task out_shardings", mpmd_mesh=trace.mpmd_mesh
    )
    out_mesh = _common_mesh(out_shardings, name="task out_shardings")
    if out_mesh is None:
        raise ValueError("task out_shardings must not be empty")
    task_mesh = params["mesh"]
    if task_mesh is not None:
        _require_mpmd_indices(trace.mpmd_mesh, task_mesh, name="task mesh")
        _reconcile_meshes(
            (("task argument shardings", in_mesh), ("task out_shardings", out_mesh)),
            mesh=task_mesh,
        )
    if in_mesh is not None and in_mesh != out_mesh:
        raise ValueError("task argument and output shardings must use the same mesh")

    out_mesh_indices = _require_mpmd_indices(
        trace.mpmd_mesh, out_mesh, name="task out_shardings"
    )
    mpmd_idx = out_mesh if len(out_mesh_indices) > 1 else out_mesh_indices[0]
    donate_invars = tuple(params["donate_invars"])
    for idx, (donated, tracer) in enumerate(zip(donate_invars, tracers, strict=True)):
        if donated and tracer._sent:
            raise ValueError(
                "task cannot donate an argument that has been sent; "
                f"argument {idx} is pinned by the transfer source lifetime"
            )
    task_params: TaskEqnParams = {
        "call_jaxpr": params["call_jaxpr"],
        "task_name": params["task_name"],
        "mpmd_idx": mpmd_idx,
        "in_shardings": in_shardings,
        "out_shardings": out_shardings,
        "donate_invars": donate_invars,
        "task_info": None,
        "latency": 1.0,
        "call_counter": None,
    }
    return trace.call_parent(task_p, tracers, task_params, out_shardings=out_shardings)


def transfer_mpmd(trace: MpmdTrace, *tracers, **params: Unpack[MpmdTransferParams]):
    in_shardings = tuple(
        _tracer_sharding(tracer, name=f"transfer argument {idx}")
        for idx, tracer in enumerate(tracers)
    )
    if len(in_shardings) == 0:
        raise ValueError("transfer expects at least one argument")

    out_shardings = _validate_flat_shardings(
        params["out_shardings"],
        name="transfer out_shardings",
        mpmd_mesh=trace.mpmd_mesh,
    )

    if len(out_shardings) == 0:
        raise ValueError("transfer out_shardings must not be empty")
    for idx, (tracer, in_sharding, out_sharding) in enumerate(
        zip(tracers, in_shardings, out_shardings, strict=True)
    ):
        ndim = len(jc.get_aval(tracer).shape)
        if not jc.shardings_are_equivalent(
            in_sharding, out_sharding, ndim, compare_memkind=True
        ):
            raise NotImplementedError(
                "transfer out_shardings must be equivalent to the input "
                f"shardings; mismatch at leaf {idx}"
            )
        src_mpmd_indices = _require_mpmd_indices(
            trace.mpmd_mesh, in_sharding.mesh, name=f"transfer argument {idx}"
        )
        tgt_mpmd_indices = _require_mpmd_indices(
            trace.mpmd_mesh, out_sharding.mesh, name=f"transfer out_shardings[{idx}]"
        )
        if set(src_mpmd_indices) & set(tgt_mpmd_indices):
            raise ValueError(
                "transfer source and target meshes must not overlap for each leaf; "
                f"overlap at leaf {idx}"
            )

    transfer_params: TransferParams = {
        "src_shardings": in_shardings,
        "tgt_shardings": out_shardings,
    }
    out = trace.call_parent(
        transfer_p, tracers, transfer_params, out_shardings=(None, *out_shardings)
    )
    for tracer in tracers:
        tracer._sent = True
    return out


def stack_mpmd(trace: MpmdTrace, *tracers):
    if len(tracers) == 0:
        raise ValueError("stack expects at least one argument")

    in_shardings = tuple(
        _tracer_sharding(tracer, name=f"stack argument {idx}")
        for idx, tracer in enumerate(tracers)
    )
    index_groups = tuple(
        _require_mpmd_indices(
            trace.mpmd_mesh, sharding.mesh, name=f"stack argument {idx}"
        )
        for idx, sharding in enumerate(in_shardings)
    )
    all_indices = array_ops.validate_index_groups(
        index_groups, mpmd_mesh=trace.mpmd_mesh, name="stack argument meshes"
    )
    stack_mpmd_mesh = trace.mpmd_mesh.mpmd_submesh(list(all_indices))
    _, out_sharding, _, _ = array_ops.stack_shape_and_sharding(
        tuple(jc.get_aval(tracer).shape for tracer in tracers),
        in_shardings,
        mpmd_mesh=stack_mpmd_mesh,
    )
    stack_params: StackParams = {
        "in_shardings": in_shardings,
        "mpmd_mesh": stack_mpmd_mesh,
        "axis": 0,
    }
    return trace.call_parent(stack_p, tracers, stack_params, out_shardings=out_sharding)


def slice_mpmd(trace: MpmdTrace, tracer, **params: Unpack[SliceParams]):
    in_sharding = _tracer_sharding(tracer, name="slice argument")
    _, out_shardings = array_ops.slice_shape_and_shardings(
        jc.get_aval(tracer).shape,
        in_sharding,
        params["groups"],
        mpmd_mesh=trace.mpmd_mesh,
    )
    slice_params: SliceParams = {
        "in_sharding": in_sharding,
        "groups": params["groups"],
        "mpmd_mesh": trace.mpmd_mesh,
    }
    return trace.call_parent(
        slice_p, (tracer,), slice_params, out_shardings=out_shardings
    )


def transfer_done_mpmd(trace: MpmdTrace, token, *arrays):
    if not isinstance(token, MpmdTracer) or token.mpmd_sharding is not None:
        raise TypeError("transfer_done token must be an experimental transfer token")
    out_shardings = tuple(
        _tracer_sharding(array, name=f"transfer_done argument {idx}")
        for idx, array in enumerate(arrays)
    )
    tracers = (token, *arrays)
    out = trace.call_parent(
        transfer_done_p, tracers, TransferDoneParams(), out_shardings=out_shardings
    )
    return out


mpmd_rules: dict[jcore.Primitive, Callable[..., object]] = {
    mpmd_task_p: task_mpmd,
    mpmd_transfer_p: transfer_mpmd,
    stack_p: stack_mpmd,
    slice_p: slice_mpmd,
    transfer_done_p: transfer_done_mpmd,
}


def _mpmd_inner(
    fun,
    mpmd_mesh: MpmdMesh,
    flat_in_shardings: tuple[NamedSharding, ...],
    out_shardings_holder,
    *args_flat,
):
    with jcore.take_current_trace() as parent_trace:
        if parent_trace is None:
            raise ValueError("mpmd must be called within a trace")

        trace = MpmdTrace(parent_trace, mpmd_mesh)
        in_tracers = [
            MpmdTracer(trace, arg, sharding)
            for arg, sharding in zip(args_flat, flat_in_shardings, strict=True)
        ]
        with jcore.set_current_trace(trace):
            outs = fun.call_wrapped(*in_tracers)

    if out_shardings_holder is not None:
        out_shardings_holder[:] = [
            out.mpmd_sharding if isinstance(out, MpmdTracer) else None for out in outs
        ]

    return [out.val if isinstance(out, MpmdTracer) else out for out in outs]


def _mpmd_trace(
    mpmd_mesh: MpmdMesh, *, in_shardings: ShardingPyTree, out_shardings_holder=None
):
    def decorator(fun):
        @wraps(fun)
        def wrapped(*args, **kwargs):
            if kwargs:
                raise ValueError(
                    "jaxpp.experimental.mpmd does not support keyword "
                    "arguments when in_shardings is specified"
                )
            args_flat, in_tree = jax.tree_util.tree_flatten(args)
            flat_in_shardings = tuple(
                jc.flatten_axis_resources(
                    "mpmd in_shardings", in_tree, in_shardings, tupled_args=True
                )
            )
            flat_in_shardings = _validate_flat_shardings(
                flat_in_shardings, name="mpmd in_shardings", mpmd_mesh=mpmd_mesh
            )

            flat_fun, out_tree = api_util.flatten_fun_nokwargs(
                lu.wrap_init(
                    fun,
                    debug_info=api_util.debug_info(
                        "jaxpp.experimental.mpmd", fun, args, kwargs
                    ),
                ),
                in_tree,
            )
            outs = _mpmd_inner(
                flat_fun, mpmd_mesh, flat_in_shardings, out_shardings_holder, *args_flat
            )
            return jax.tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return decorator


@dataclasses.dataclass(frozen=True)
class LoweredMpmdFun:
    mpmd_mesh: MpmdMesh
    in_shardings: ShardingPyTree
    local_jaxprs: tuple[LocalJaxpr, ...]
    out_shape: ShapeDtypePyTree
    out_shardings: ShardingPyTree

    @property
    def _local_jaxpr(self) -> LocalJaxpr:
        return self.local_jaxprs[self.mpmd_mesh.my_mpmd_axis_index]

    @property
    def used_inputs(self) -> PyTree[bool]:
        flat_in_shardings, in_tree = jax.tree_util.tree_flatten(self.in_shardings)
        flat_used_inputs = [False] * len(flat_in_shardings)
        for global_invar_idx in self._local_jaxpr.global_invar_indices:
            flat_used_inputs[global_invar_idx] = True
        return jax.tree_util.tree_unflatten(in_tree, flat_used_inputs)

    def eval_local(self, *local_args: Any) -> list[Any]:
        local_jaxpr = self._local_jaxpr
        if len(local_args) != len(local_jaxpr.closed_jaxpr.invars):
            raise ValueError(
                "eval_local expects arguments in local_jaxpr.global_invar_indices "
                f"order: expected {len(local_jaxpr.closed_jaxpr.invars)}, "
                f"got {len(local_args)}"
            )
        with self.mpmd_mesh:
            return jcore.eval_jaxpr(
                local_jaxpr.closed_jaxpr.jaxpr,
                local_jaxpr.closed_jaxpr.consts,
                *local_args,
            )

    def __call__(self, *args: Any) -> Any:
        flat_args, args_tree = jax.tree_util.tree_flatten(args)
        in_tree = jax.tree_util.tree_structure(self.in_shardings)
        if args_tree != in_tree:
            raise ValueError(
                "LoweredMpmdFun.__call__ expects the original input pytree; "
                "use eval_local for rank-local pruned inputs"
            )

        local_jaxpr = self._local_jaxpr
        local_outs = self.eval_local(
            *(flat_args[idx] for idx in local_jaxpr.global_invar_indices)
        )

        local_outs_by_idx = dict(
            zip(local_jaxpr.global_outvar_indices, local_outs, strict=True)
        )
        flat_out_shape, output_tree = jax.tree_util.tree_flatten(self.out_shape)

        flat_outs = []
        for idx, shape_dtype in enumerate(flat_out_shape):
            if idx in local_outs_by_idx:
                flat_outs.append(local_outs_by_idx[idx])
            else:
                flat_outs.append(
                    jax.make_array_from_single_device_arrays(
                        shape=shape_dtype.shape,
                        sharding=shape_dtype.sharding,
                        arrays=[],
                        dtype=shape_dtype.dtype,
                    )
                )
        return jax.tree_util.tree_unflatten(output_tree, flat_outs)


@runtime_checkable
class MpmdFunction(Protocol[P, _T_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> _T_co: ...

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> LoweredMpmdFun: ...


@dataclasses.dataclass(frozen=True)
class MpmdFun:
    fun: Callable
    mpmd_mesh: MpmdMesh
    in_shardings: ShardingPyTree
    donate_argnums: int | Sequence[int] | None = None
    infer_donation: bool = False
    _traced_fun: Callable = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_traced_fun",
            _mpmd_trace(self.mpmd_mesh, in_shardings=self.in_shardings)(self.fun),
        )

    def __call__(self, *args, **kwargs):
        with self.mpmd_mesh:
            return self._traced_fun(*args, **kwargs)

    def lower(self, *args, **kwargs) -> LoweredMpmdFun:
        flat_out_shardings = []
        flat_donated_invars = _flatten_donated_invars(args, self.donate_argnums)
        traced_fun = _mpmd_trace(
            self.mpmd_mesh,
            in_shardings=self.in_shardings,
            out_shardings_holder=flat_out_shardings,
        )(self.fun)
        cjaxpr, out_shape = jax.make_jaxpr(traced_fun, return_shape=True)(
            *args, **kwargs
        )
        out_tree = jax.tree_util.tree_structure(out_shape)
        out_shardings = out_tree.unflatten(flat_out_shardings)
        out_shape = jax.tree.map(
            lambda shape_dtype, sharding: shape_dtype.update(sharding=sharding),
            out_shape,
            out_shardings,
        )
        local_jaxprs = []
        for local_jaxpr in to_local_jaxprs(cjaxpr, self.mpmd_mesh):
            donated_invars = tuple(
                flat_donated_invars[global_invar_idx]
                for global_invar_idx in local_jaxpr.global_invar_indices
            )
            closed_jaxpr = local_jaxpr.closed_jaxpr
            if self.infer_donation:
                closed_jaxpr = closed_jaxpr.map_jaxpr(
                    partial(infer_donation_pass, donated_invars=donated_invars)
                )
            closed_jaxpr = closed_jaxpr.map_jaxpr(
                partial(add_deletes, donated_invars=donated_invars)
            )
            local_jaxprs.append(
                dataclasses.replace(local_jaxpr, closed_jaxpr=closed_jaxpr)
            )

        return LoweredMpmdFun(
            mpmd_mesh=self.mpmd_mesh,
            in_shardings=self.in_shardings,
            local_jaxprs=tuple(local_jaxprs),
            out_shape=out_shape,
            out_shardings=out_shardings,
        )


def _flatten_donated_invars(
    args: tuple[object, ...], donate_argnums: int | Sequence[int] | None
) -> tuple[bool, ...]:
    donated_argnums = (
        () if donate_argnums is None else jc._ensure_index_tuple(donate_argnums)
    )
    donated_argnums = jc._ensure_inbounds(False, len(args), donated_argnums)
    return api_util.donation_vector(
        donated_argnums, (), jax.tree_util.tree_structure(args), kws=False
    )


def mpmd(
    mpmd_mesh: MpmdMesh,
    *,
    in_shardings: ShardingPyTree,
    donate_argnums: int | Sequence[int] | None = None,
    infer_donation: bool = False,
) -> Callable[[Callable[P, T]], MpmdFunction[P, T]]:
    def decorator(fun: Callable[P, T]) -> MpmdFunction[P, T]:
        return MpmdFun(fun, mpmd_mesh, in_shardings, donate_argnums, infer_donation)

    return decorator


def _task(
    fun,
    name: str | None,
    mesh: Mesh | None,
    *,
    args,
    kwargs,
    in_shardings,
    out_shardings,
    donate_argnums,
):
    task_mesh = _infer_task_mesh(args, kwargs, mesh, in_shardings, out_shardings)
    jit_info = jc._parse_jit_arguments(
        fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=None,
        static_argnames=None,
        donate_argnums=donate_argnums,
        donate_argnames=None,
        keep_unused=False,
        device=None,
        backend=None,
        inline=False,
        compiler_options=None,
        use_resource_env=False,
    )
    with _task_mesh_context(task_mesh):
        pjit_params, flat_args = jc._infer_params(fun, jit_info, args, kwargs)
    params = pjit_params.params
    mpmd_task_params: MpmdTaskParams = {
        "call_jaxpr": params["jaxpr"],
        "task_name": name or params["name"],
        "mesh": task_mesh,
        "in_shardings": tuple(params["in_shardings"]),
        "out_shardings": tuple(params["out_shardings"]),
        "donate_invars": tuple(params["donated_invars"]),
    }
    res = mpmd_task_p.bind(*flat_args, **mpmd_task_params)
    return jax.tree_util.tree_unflatten(pjit_params.out_tree, res)


def task(
    fun: Callable[P, T] | None = None,
    /,
    *,
    out_shardings: TaskShardingPyTree,
    name: str | None = None,
    in_shardings: TaskShardingPyTree | jc.UnspecifiedValue = jc.UNSPECIFIED,
    mesh: Mesh | None = None,
    donate_argnums: int | Sequence[int] | None = None,
) -> Callable[P, T]:
    """Wrap a function as one MPMD task.

    All task inputs and outputs must live on the same MPMD mesh slice. Move
    values between slices with ``transfer`` before passing them to another task.
    """
    if fun is None:
        return partial(
            task,
            out_shardings=out_shardings,
            name=name,
            in_shardings=in_shardings,
            mesh=mesh,
            donate_argnums=donate_argnums,
        )

    @wraps(fun)
    def wrapped(*args, **kwargs):
        return _task(
            fun,
            name,
            mesh,
            args=args,
            kwargs=kwargs,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            donate_argnums=donate_argnums,
        )

    return wrapped


def transfer(arrays: T, *, out_shardings: ShardingPyTree) -> TransferFuture[T]:
    """Start transfer sends and receives immediately.

    Returns a future whose ``done()`` method waits for the receiver-side values.
    Source values passed to ``transfer`` may not be donated by later tasks,
    including after ``done()``. The runtime keeps a send-buffer hold that is
    released asynchronously to avoid a host synchronization, so donation cannot
    prove the hold has been dropped.
    """
    flat_arrays, arrays_tree = jax.tree_util.tree_flatten(arrays)
    if len(flat_arrays) == 0:
        raise ValueError("transfer expects at least one array")

    flat_out_shardings = tuple(
        jc.flatten_axis_resources(
            "transfer out_shardings", arrays_tree, out_shardings, tupled_args=True
        )
    )
    transfer_params: MpmdTransferParams = {"out_shardings": flat_out_shardings}
    token, *transferred = mpmd_transfer_p.bind(*flat_arrays, **transfer_params)
    future: TransferFuture[T] = _TransferFutureImpl(token, arrays_tree, transferred)
    return future


def stack(*arrays: T) -> T:
    """Combine values from disjoint MPMD mesh slices into one value."""
    if len(arrays) == 0:
        raise ValueError("stack expects at least one argument")
    if not all(isinstance(array, MpmdTracer) for array in arrays):
        raise ValueError(
            "jaxpp.experimental.mpmd.stack must be called inside "
            "jaxpp.experimental.mpmd"
        )
    return stack_p.bind(*arrays)


def slice(array: T, groups: Sequence[int | Sequence[int]]) -> tuple[T, ...]:  # noqa: A001
    """Split a value across the requested MPMD mesh index groups."""
    if not isinstance(array, MpmdTracer):
        raise ValueError(
            "jaxpp.experimental.mpmd.slice must be called inside "
            "jaxpp.experimental.mpmd"
        )
    if not isinstance(array._trace, MpmdTrace):
        raise ValueError(
            "jaxpp.experimental.mpmd.slice must be called inside "
            "jaxpp.experimental.mpmd"
        )
    in_sharding = _tracer_sharding(array, name="slice argument")
    mpmd_mesh = array._trace.mpmd_mesh
    groups = array_ops.normalize_index_groups(
        groups, mpmd_mesh=mpmd_mesh, name="slice groups"
    )
    in_indices = _require_mpmd_indices(
        mpmd_mesh, in_sharding.mesh, name="slice argument"
    )
    if tuple(idx for group in groups for idx in group) != in_indices:
        raise ValueError("slice groups must partition the input mesh in order")
    slice_params: SliceParams = {
        "in_sharding": in_sharding,
        "groups": groups,
        "mpmd_mesh": mpmd_mesh,
    }
    return tuple(slice_p.bind(array, **slice_params))
