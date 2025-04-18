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

import itertools as it
import logging
import math
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property, partial
from typing import (
    Concatenate,
    Generic,
    NamedTuple,
    NewType,
    ParamSpec,
    TypeVar,
    cast,
)
from weakref import ref

import jax
import jax.core as jcore
import jax.interpreters.partial_eval as pe
import jax.util as ju
import ray
from jax._src.debugging import inspect_sharding_p
from jax._src.util import weakref_lru_cache
from jax.interpreters.ad import add_jaxvals_p as add_any_p

from jaxpp.arrayref import ArrayRef, ArrayRefSharding
from jaxpp.compilation import (
    CompileThunk,
    SerializeableMeshComputation,
    pjit_to_serializeable_mesh_computation,
)
from jaxpp.jax_primitives import (
    PscanJaxpr,
    ShardingStore,
    TaskEqn,
    add_multi_p,
    all_reduce_p,
    dax_pscan_p,
    jit_infer_params,
    pipeline_yield_p,
    recv_p,
    send_p,
    task_p,
)
from jaxpp.jaxpr_utils import (
    check_jaxpr,
    eqns_free_vars,
    jaxpr_from_eqns,
    nonlit,
    schedule_dependencies,
    substitute,
)
from jaxpp.mesh import MpmdMesh, RemoteMpmdMesh
from jaxpp.ops import (
    UID,
    AllReduceDesc,
    AllReduceOp,
    CommDesc,
    Op,
    RecvOp,
    RunOp,
    SendOp,
    add_delete_ops,
    get_comm_keys,
)
from jaxpp.types import (
    Bind,
    DistributedSharding,
    GlobalDeviceId,
    PutArg,
    ScalarUid,
    SerializeableSharding,
    TaskType,
    UniqueGlobalDeviceIds,
    UniqueSortedSequence,
    fresh_scalar_uid,
)
from jaxpp.utils import CtxVar, RichDict, groupby, hbytes, log_elapsed_time

logger = logging.getLogger(__name__)


@contextmanager
def stable_names_ctx(anno: Callable[[jcore.Var], str | None] = lambda v: None):
    prev_repr = jax._src.core.Var.__repr__
    prev_pp_var = jax._src.core.pp_var

    ctx = jcore.JaxprPpContext()

    def __repr__(v):
        if isinstance(v, jcore.Literal):
            return f"{v}"
        if s := anno(v):
            return f"{prev_pp_var(v, ctx)}{s}"

        return f"{prev_pp_var(v, ctx)}"

    jax._src.core.pp_var = lambda v, _: __repr__(v)
    jax._src.core.Var.__repr__ = __repr__

    try:
        yield
    finally:
        jax._src.core.Var.__repr__ = prev_repr
        jax._src.core.pp_var = prev_pp_var


CJaxpr = TypeVar("CJaxpr", jcore.ClosedJaxpr, jcore.Jaxpr)
Res = TypeVar("Res")
P = ParamSpec("P")


def unwrap_closed(
    fun: Callable[Concatenate[jcore.Jaxpr, P], jcore.Jaxpr],
) -> Callable[Concatenate[CJaxpr, P], CJaxpr]:
    def res(closed_jaxpr: CJaxpr, *args: P.args, **kwargs: P.kwargs):
        if isinstance(closed_jaxpr, jcore.ClosedJaxpr):
            assert len(closed_jaxpr.consts) == 0
            return closed_jaxpr.map_jaxpr(lambda jaxpr: fun(jaxpr, *args, **kwargs))
        return fun(closed_jaxpr, *args, **kwargs)

    return res


MaybeEqn = int | None


@partial(weakref_lru_cache, maxsize=16)
def ivar_defs_and_refs(jaxpr: jcore.Jaxpr):
    defs: dict[jcore.Var, MaybeEqn] = {}
    refs: dict[jcore.Var, list[MaybeEqn]] = {}

    def read(a: jcore.Atom, eqn: MaybeEqn):
        if not isinstance(a, jcore.Literal):
            assert a in defs, a
            assert a in refs, a
            refs[a].append(eqn)

    def write(v: jcore.Var, eqn: MaybeEqn):
        assert v not in defs, v
        assert v not in refs, v
        if not isinstance(v, jcore.DropVar):
            defs[v] = eqn
            refs[v] = []

    for v in jaxpr.constvars:
        write(v, None)
    for v in jaxpr.invars:
        write(v, None)

    for i, eqn in enumerate(jaxpr.eqns):
        for a in eqn.invars:
            read(a, i)
        for v in eqn.outvars:
            write(v, i)

    for a in jaxpr.outvars:
        read(a, None)
    return defs, refs


MpmdIdx = NewType("MpmdIdx", int)


def get_mpmd_idx(stage_id: int, mpmd_dim: int) -> MpmdIdx:
    return MpmdIdx(stage_id % mpmd_dim)


def get_task_mpmd_idx(task: TaskEqn) -> MpmdIdx:
    assert task.primitive is task_p
    return task.params["mpmd_idx"]


class AllReduceRewriteTracer(jcore.Tracer):
    def __init__(self, trace, val, placement: set[MpmdIdx] | None = None):
        self._trace = trace
        self.placement = placement
        self.val = val

    @property
    def aval(self):
        return jcore.get_aval(self.val)


SubTrace = TypeVar("SubTrace", bound=jcore.Trace)


class AllReduceRewriteTrace(jcore.Trace, Generic[SubTrace]):
    def __init__(self, parent_trace: SubTrace):
        self.parent_trace = parent_trace

    def call_parent(self, primitive, tracers, params, *, placement=None):
        parent_tracers = [
            t.val if isinstance(t, AllReduceRewriteTracer) else t for t in tracers
        ]
        if primitive is add_multi_p and "mpmd_idxs" not in params:
            multiple_results = False
            with jcore.set_current_trace(self.parent_trace):
                results = sum(parent_tracers)
        else:
            multiple_results = primitive.multiple_results
            results = self.parent_trace.process_primitive(
                primitive, parent_tracers, params
            )

        if not multiple_results:
            results = [results]
        out_tracers = [
            AllReduceRewriteTracer(self, result, placement) for result in results
        ]
        if not multiple_results:
            return out_tracers[0]

        return out_tracers

    def process_primitive(self, primitive, tracers, params):
        known_in_placements = list[set[MpmdIdx]](
            placement
            for tracer in tracers
            if isinstance(tracer, AllReduceRewriteTracer)
            and (placement := tracer.placement) is not None
        )
        if len(known_in_placements) == 0:
            return self.call_parent(
                primitive,
                tracers,
                params,
                placement=None,  # Skip for later
            )

        placement = known_in_placements[0].intersection(*known_in_placements[1:])
        if len(placement) > 0:
            return self.call_parent(primitive, tracers, params, placement=placement)

        # TODO: refine tracing to update "upgraph" when `placement` is found
        if primitive is add_any_p:
            # Rewrite cross_mpmd `add_any` to `add_multi`
            l, r = tracers
            return self.call_parent(
                add_multi_p,
                tracers,
                {"mpmd_idxs": (min(l.placement), min(r.placement))},
                # FIXME below: assumes l, r are tracers, i.e. doesn't handle literals
                placement={min(l.placement), min(r.placement)},
            )
        elif primitive is add_multi_p:
            groups = groupby(
                (min(t.placement), t)
                for t in tracers
                if isinstance(t, AllReduceRewriteTracer)
            )

            placement = set(groups.keys())
            groups[min(placement)].extend(
                t for t in tracers if not isinstance(t, AllReduceRewriteTracer)
            )

            results = []
            with jcore.set_current_trace(self.parent_trace):
                for group in groups.values():
                    if len(group) > 1:
                        e = sum(
                            t.val if isinstance(t, AllReduceRewriteTracer) else t
                            for t in group
                        )
                    else:
                        e = group[0]
                    results.append(e)

            return self.call_parent(
                add_multi_p,
                results,
                {"mpmd_idxs": tuple(groups.keys())},
                placement=placement,
            )
        raise AssertionError("After loop computation is not replicateable")


def propagate_and_rewrite_adds(
    jaxpr: jcore.Jaxpr, invar_mpmd_defs: Iterable[set[MpmdIdx] | None]
) -> tuple[jcore.Jaxpr, list[set[MpmdIdx] | None]]:
    """
    Infers the placement of the outputs and intermediate operations.
    When the placement is ambiguous for `add`-like operations, they are rewritten
    to cross-mpmd reduce operations, otherwise it raises an error.
    """
    mpmd_trace = AllReduceRewriteTrace(pe.DynamicJaxprTrace(jaxpr.debug_info))
    in_tracers = [
        AllReduceRewriteTracer(
            mpmd_trace, mpmd_trace.parent_trace.new_arg(invar.aval), mpmd_defs
        )
        for invar, mpmd_defs in zip(jaxpr.invars, invar_mpmd_defs, strict=True)
    ]

    with jcore.set_current_trace(mpmd_trace):
        res = jcore.eval_jaxpr(jaxpr, (), *in_tracers)

    # TODO: handle literals in `res`
    jaxpr, consts, _ = mpmd_trace.parent_trace.to_jaxpr(
        [v.val for v in res], jaxpr.debug_info
    )
    assert len(consts) == 0
    return jaxpr, [v.placement for v in res]


def mpmd_unzip_forward(
    in_jaxpr: jcore.Jaxpr, invar_mpmd_defs: Sequence[set[MpmdIdx] | None], mpmd_dim: int
) -> tuple[
    jcore.Jaxpr, list[set[MpmdIdx]], list[set[MpmdIdx]], defaultdict[MpmdIdx, int]
]:
    """
    Coarsens the equations of `in_jaxpr` into SPMD tasks depending on the placement
    of their inputs.
    It allows for cross-mpmd all-reduces.

    Returns (
        coarsened_jaxpr, invar_mpmd_refs, outvar_mpmd_defs, number_of_eqns_per_mpmd_idx
    )
    """
    jaxpr, out_placements = propagate_and_rewrite_adds(in_jaxpr, invar_mpmd_defs)
    # NOTE: when the placement is unknown we set it to the widest placement
    out_placements = [v or set(range(mpmd_dim)) for v in out_placements]

    # TODO: schedule equations to reduce materialized buffers
    #  at suspension points
    add_multi_eqn_idxs = [
        idx
        for idx, e in reversed(list(enumerate(jaxpr.eqns)))
        if e.primitive is add_multi_p
    ]
    if len(add_multi_eqn_idxs) == 0 or add_multi_eqn_idxs[-1] != 0:
        add_multi_eqn_idxs.append(0)

    var_placement = dict[jcore.Var, set[int]](
        (outvar, placement) for outvar, placement in zip(jaxpr.outvars, out_placements)
    )

    eqns_in_mpmd_idx = defaultdict(lambda: 0)

    mpmd_idxs = range(mpmd_dim)
    last = len(jaxpr.eqns)
    rev_new_eqns = []
    for eqn_idx in add_multi_eqn_idxs:
        all_reduce_eqn, eqns = None, jaxpr.eqns[eqn_idx:last]
        last = eqn_idx
        if eqns[0].primitive is add_multi_p:
            [all_reduce_eqn, *eqns] = eqns
            eqn_idx += 1

        tmp = jaxpr_from_eqns(eqns, set(var_placement.keys()))

        jaxprs, in_uses = make_replicated_jaxpr(
            tmp,
            tuple(var_placement[outvar] for outvar in tmp.outvars),
            mpmd_idxs,
        )
        for mpmd_idx, j in zip(mpmd_idxs, jaxprs, strict=True):
            eqns_in_mpmd_idx[mpmd_idx] += len(j.eqns)
            rev_new_eqns.append(
                make_task_eqn(
                    invars=j.invars,
                    outvars=j.outvars,
                    eqns=j.eqns,
                    mpmd_idx=mpmd_idx,
                    task_name=f"after_loop_{mpmd_idx}_{fresh_scalar_uid()}",
                )
            )
        for invar, uses in ju.safe_zip(tmp.invars, in_uses):
            if uses is None:
                continue
            if (p := var_placement.get(invar)) is not None:
                uses = uses | p
            var_placement[invar] = uses

        if all_reduce_eqn is None:
            continue

        all_reduce_jaxpr = jaxpr_from_eqns([all_reduce_eqn], set(var_placement.keys()))
        rev_new_eqns.append(
            make_task_eqn(
                invars=all_reduce_jaxpr.invars,
                outvars=all_reduce_jaxpr.outvars,
                eqns=all_reduce_jaxpr.eqns,
                # NOTE(all_reduce_task): almost all the time a task_eqn has
                #  `isinstance(mpmd_idx, int) == True`
                #  except for these all-reduce tasks that have `mpmd_idx : tuple[int, ...]`
                mpmd_idx=all_reduce_eqn.params["mpmd_idxs"],
                task_name=f"all_reduce_{mpmd_idx}",
            )
        )
        for invar, mpmd_idx in ju.safe_zip(
            all_reduce_jaxpr.invars, all_reduce_eqn.params["mpmd_idxs"]
        ):
            uses = {mpmd_idx}
            if (p := var_placement.get(invar)) is not None:
                uses = uses | p
            var_placement[invar] = uses

    sub = dict(ju.safe_zip(jaxpr.invars, in_jaxpr.invars))
    sub.update(ju.safe_zip(jaxpr.outvars, in_jaxpr.outvars))
    new_eqns = substitute(reversed(rev_new_eqns), sub)
    res_jaxpr = jaxpr.replace(
        invars=in_jaxpr.invars, outvars=in_jaxpr.outvars, eqns=new_eqns
    )

    return (
        check_jaxpr(res_jaxpr),
        [var_placement[invar] for invar in jaxpr.invars],
        out_placements,
        eqns_in_mpmd_idx,
    )


def pushout_add_any(loop_body: jcore.Jaxpr) -> jcore.Jaxpr:
    """
    Applies recursively the following commuting rewrite rule.

    ```
    a = add_any b c; d = shard_constraint a
      ~>
    b' = shard_constraint b; c' = shard_constraint c; d = add_any b' c'
    ```

    NOTE that `a` disappears from the equations since there is a single
    use and thus immediately substituted

    # TODO: maybe generalize to multiple uses and instead
    #  add a "dummy equation" instead of performing substitution
    """

    worklist = list[jcore.JaxprEqn | None](reversed(loop_body.eqns))
    res = []
    gensym = jcore.gensym()
    _, mut_refs = ivar_defs_and_refs(loop_body)
    # Iterate over the equations in execution order
    while len(worklist) > 0:
        eqn = worklist.pop()
        if eqn is not None:
            add_any = eqn
            if (
                add_any.primitive is add_any_p
                and (uses := mut_refs[add_any.outvars[0]])
                and len(uses) == 1
                and (use_idx := uses[0])
                and (constraint := loop_body.eqns[use_idx])
                and constraint.primitive is jax.lax.sharding_constraint_p
            ):
                new_add_any_invars = list[jcore.Atom](
                    gensym(invar.aval) for invar in add_any.invars
                )
                [constraint_outvar] = constraint.outvars

                for invar, outvar in zip(
                    add_any.invars, new_add_any_invars, strict=True
                ):
                    res.append(constraint.replace(invars=[invar], outvars=[outvar]))

                worklist.append(
                    add_any.replace(
                        invars=new_add_any_invars, outvars=[constraint_outvar]
                    )
                )

                # Replace references on the fly and erase equation
                mut_refs[add_any.outvars[0]] = mut_refs[constraint_outvar]
                worklist[len(loop_body.eqns) - 1 - use_idx] = None

            else:
                res.append(eqn)

    return loop_body.replace(eqns=res)


def compute_needed(loop_body: jcore.Jaxpr, body_nconsts: int):
    """
    Given the following Jaxpr

    ```
    def loop(c1, c2, c3, ..., c<$body_nconsts> | z, y, prev_x, ...):
                ...
        (128)   x  = add_any x1 x2
                ... # no `x` uses
        (184)   x' = add prev_x x
                return z', y', x', ...
           # position: 0 , 1 , 2
    ```

    returns the edits needed to push the `add_any` outside of the loop

    (
        # Add these two invars at index $body_nconsts + 2
        { $body_nconsts + 2: [prev_x', prev_x''] },
        # At the end of the loop perform add_any between x'' and x''' which are the
        # variables to replace the output at index 2
        {2: (add_any x1 x2, [x'', x'''])},
        # Replace equation at index 184 with the two equations listed
        # Erase equation at index 128
        {
            184: [add prev_x' x, add prev_x'' x],
            128: []
        }
    )
    """
    gensym = jcore.gensym("_licm")
    defs, refs = ivar_defs_and_refs(loop_body)

    replicated_loop_body_invars = defaultdict[int, list[jcore.Var]](list)
    replicated_loop_body_outvars = dict[int, tuple[jcore.JaxprEqn, list[jcore.Var]]]()
    replace_eqns = dict[int, list[jcore.JaxprEqn]]()

    for outvar_idx, outvar in enumerate(loop_body.outvars):
        if isinstance(outvar, jcore.Var) and (add_eqn_idx := defs[outvar]):
            add_eqn = loop_body.eqns[add_eqn_idx]
            if add_eqn.primitive is jax.lax.add_p:
                [linvar, rinvar] = add_eqn.invars
                loop_body_invar, grad = (
                    (linvar, rinvar) if defs[linvar] is None else (rinvar, linvar)
                )
                if (
                    defs[loop_body_invar] is None
                    and (add_any_eqn_idx := defs[grad])
                    and (
                        add_any_eqn := cast(
                            jcore.JaxprEqn, loop_body.eqns[add_any_eqn_idx]
                        )
                    )
                    and add_any_eqn.primitive is add_any_p
                    # Verifying the to-be-deleted reference is not being used elsewhere.
                    and len(refs[add_any_eqn.outvars[0]]) == 1
                    and refs[add_any_eqn.outvars[0]] == [add_eqn_idx]
                    and (
                        (invar_idx := loop_body.invars.index(loop_body_invar))
                        >= body_nconsts
                    )
                    # TODO: maybe add separate workers condition? (not necessary for correctness)
                ):
                    assert outvar_idx == invar_idx - body_nconsts

                    replicated_ga_eqns = []
                    for cross_worker_invar in add_any_eqn.invars:
                        in_replica = gensym(cross_worker_invar.aval)
                        out_replica = gensym(cross_worker_invar.aval)
                        replicated_loop_body_invars[invar_idx].append(in_replica)

                        if outvar_idx not in replicated_loop_body_outvars:
                            replicated_loop_body_outvars[outvar_idx] = (add_any_eqn, [])
                        replicated_loop_body_outvars[outvar_idx][1].append(out_replica)

                        replicated_ga_eqns.append(
                            add_eqn.replace(
                                invars=[in_replica, cross_worker_invar],
                                outvars=[out_replica],
                            )
                        )

                    replace_eqns[add_eqn_idx] = replicated_ga_eqns
                    replace_eqns[add_any_eqn_idx] = []

    return replicated_loop_body_invars, replicated_loop_body_outvars, replace_eqns


# Transformation
def add_jaxpr_parameters(
    loop_body: jcore.Jaxpr,
    replicated_loop_body_invars: Mapping[int, list[jcore.Var]],
    replicated_loop_body_outvars: dict[int, tuple[jcore.JaxprEqn, list[jcore.Var]]],
    replace_eqns: Mapping[int, list[jcore.JaxprEqn]],
) -> jcore.Jaxpr:
    new_loop_body_invars = list[jcore.Var]()
    for idx, invar in enumerate(loop_body.invars):
        new_loop_body_invars.extend(replicated_loop_body_invars.get(idx, [invar]))

    new_loop_body_outvars = list[jcore.Var]()
    for idx, outvar in enumerate(loop_body.outvars):
        outvar: jcore.Var
        new_loop_body_outvars.extend(
            replicated_loop_body_outvars.get(idx, (None, [outvar]))[1]
        )

    new_loop_eqns = list[jcore.JaxprEqn]()
    for idx, eqn in enumerate(loop_body.eqns):
        new_loop_eqns.extend(replace_eqns.get(idx, [eqn]))

    return loop_body.replace(
        invars=new_loop_body_invars,
        outvars=new_loop_body_outvars,
        eqns=new_loop_eqns,
        debug_info=None,  # FIXME
    )


def make_task_eqn(
    invars: Sequence[jcore.Var],
    outvars: Sequence[jcore.Var],
    eqns: list[jcore.JaxprEqn],
    mpmd_idx: int,
    task_name: str,
) -> jcore.JaxprEqn:
    source_infos = [None] * len(outvars)
    outvar_idx = {o: idx for idx, o in enumerate(outvars)}
    for eqn in eqns:
        for o in eqn.outvars:
            if (idx := outvar_idx.get(o)) is not None:
                source_infos[idx] = eqn

    in_sharding_store, inspect_invars = ShardingStore.collect_jaxpr(invars)
    out_sharding_store, inspect_outvars = ShardingStore.collect_jaxpr(
        outvars, _provenance_info=task_name, _source_info=source_infos
    )

    eqns = inspect_invars + eqns + inspect_outvars
    effects = jcore.join_effects(*(eqn.effects for eqn in eqns))
    task_jaxpr = jcore.Jaxpr(
        constvars=(), invars=invars, outvars=outvars, eqns=eqns, effects=effects
    )
    check_jaxpr(task_jaxpr)

    return jcore.new_jaxpr_eqn(
        invars,
        outvars,
        task_p,
        {
            "call_jaxpr": task_jaxpr,
            "task_name": task_name,
            "mpmd_idx": mpmd_idx,
            "in_shardings": in_sharding_store,
            "out_shardings": out_sharding_store,
            "donate_invars": (False,) * len(task_jaxpr.invars),
            "recv_invars": list[tuple[int, list[int]]](),
            "send_outvars": list[tuple[int, list[int]]](),
        },
        effects=task_jaxpr.effects,
    )


class Cluster(NamedTuple):
    """
    A group of equations that will be scheduled to the same `MpmdIdx`
    """

    mpmd_idx: MpmdIdx
    task_type: TaskType
    eqns: list[jcore.JaxprEqn]
    stage_id: int | None = None


class ClusterInfo(NamedTuple):
    var_def_cluster_idx: dict[jcore.Var, int]
    var_ref_cluster_idx: defaultdict[jcore.Var, set[int]]
    last_cluster_idx_for_mpmd_idx: dict[MpmdIdx, int]


def get_cluster_information(clusters: list[Cluster]) -> ClusterInfo:
    var_def_cluster_idx = dict[jcore.Var, int]()
    var_ref_cluster_idx = defaultdict[jcore.Var, set[int]](set)
    last_cluster_idx_for_mpmd_idx = dict[MpmdIdx, int]()

    for cluster_idx, (mpmd_idx, _, eqns, _) in enumerate(clusters):
        last_cluster_idx_for_mpmd_idx[mpmd_idx] = cluster_idx

        refs, defs = eqns_free_vars(eqns)
        for v in refs:
            var_ref_cluster_idx[v].add(mpmd_idx)

        var_def_cluster_idx.update(zip(defs, it.repeat(cluster_idx)))

    return ClusterInfo(
        var_def_cluster_idx, var_ref_cluster_idx, last_cluster_idx_for_mpmd_idx
    )


def first_pipeline_yield_eqn_idx(eqns: Iterable[jcore.JaxprEqn]) -> int | None:
    for idx, eqn in enumerate(eqns):
        if eqn.primitive is pipeline_yield_p:
            return idx


def infer_cluster_idx_for_eqns(
    clusters: list[Cluster],
    eqns: list[jcore.JaxprEqn],
    bias: dict[jcore.Var, set[MpmdIdx]] | None = None,
) -> list[int | None]:
    bias = bias or {}
    cluster_info = get_cluster_information(clusters)
    var_def_cluster_idx = cluster_info.var_def_cluster_idx
    var_ref_cluster_idx = cluster_info.var_ref_cluster_idx
    last_cluster_idx_for_mpmd_idx = cluster_info.last_cluster_idx_for_mpmd_idx

    idefs = dict[jcore.Var, int]()
    for eqn_idx, eqn in enumerate(eqns):
        idefs.update(zip(eqn.outvars, it.repeat(eqn_idx)))

    eqn_cluster_idx: list[int | None] = [None] * len(eqns)

    def update(eqn_idx: int, cluster_idx: int):
        def update_one(eqn_idx: int):
            eqn_cluster_idx[eqn_idx] = cluster_idx
            eqn = eqns[eqn_idx]
            for invar in nonlit(eqn.invars):
                var_ref_cluster_idx[invar].add(clusters[cluster_idx][0])
            for outvar in eqn.outvars:
                var_def_cluster_idx[outvar] = cluster_idx

        worklist = deque(nonlit(eqns[eqn_idx].invars))
        while len(worklist) > 0:
            v = worklist.popleft()
            if (dep_eqn_idx := idefs.get(v)) is not None:
                if eqn_cluster_idx[dep_eqn_idx] is None:
                    update_one(dep_eqn_idx)
                    worklist.extend(nonlit(eqns[dep_eqn_idx].invars))
                else:
                    # NOTE: this is an invariant of the algorithm so this assertion
                    #  is never raised in practice.
                    #  However we leave it here in case of changes
                    assert (
                        p := eqn_cluster_idx[dep_eqn_idx]
                    ) is not None and p <= cluster_idx, f"{p=} {cluster_idx=}"
            update_one(eqn_idx)

    for eqn_idx, eqn in enumerate(eqns):
        cluster_idx_freq = Counter()
        mpmd_idx_freq: dict[MpmdIdx, int] = Counter()
        invars = nonlit(eqn.invars)
        for invar in invars:
            if (cluster_idx := var_def_cluster_idx.get(invar)) is not None:
                cluster_idx_freq[cluster_idx] += 1
            for cluster_idx in var_ref_cluster_idx[invar]:
                mpmd_idx_freq[clusters[cluster_idx][0]] += 1
            for mpmd_idx in bias.get(invar, []):
                mpmd_idx_freq[mpmd_idx] += 1

        if len(invars) == 0:
            # Case like `broadcast_in_dim[shape=...] 0.0`
            # Defer for later: `update` will walk up the definitions
            continue

        # We must schedule this equation later than any equation that
        # defines some of its invars (honor data dependencies)
        earliest_cluster_idx = max(cluster_idx_freq.keys(), default=0)

        if cluster_idx_freq.total() == len(invars):
            # Simple case: all eqn.invars are produced by eqns in `clusters`
            update(eqn_idx, earliest_cluster_idx)

        elif len(mpmd_idx_freq) > 0 or len(cluster_idx_freq) > 0:
            for cluster_idx, freq in cluster_idx_freq.items():
                mpmd_idx_freq[clusters[cluster_idx][0]] += freq

            frequent_mpmd_idx = mpmd_idx_freq.most_common(1)[0][0]

            cluster_idx = None
            if clusters[earliest_cluster_idx][0] == frequent_mpmd_idx:
                # The latest invar definition is on the same `mpmd_idx`
                # as the most common one
                cluster_idx = earliest_cluster_idx
            elif earliest_cluster_idx < (
                p := last_cluster_idx_for_mpmd_idx.get(frequent_mpmd_idx, 0)
            ):
                # The latest invar definition is on a cluster preceding
                # the latest cluster of the most common `mpmd_idx`
                cluster_idx = p
            else:
                # Currently we always only schedule to existing clusters
                raise NotImplementedError(
                    f"{infer_cluster_idx_for_eqns.__name__} does not support "
                    f"opening new {Cluster.__name__}s."
                )
            update(eqn_idx, cluster_idx)

        else:
            # Defer for later: `update` will walk up the definitions
            pass
    return eqn_cluster_idx


def cluster_by_yield_eqns(
    eqns: list[jcore.JaxprEqn], mpmd_dim: int
) -> tuple[list[Cluster], list[jcore.JaxprEqn]]:
    idx_update, *eqns = eqns
    if not (
        idx_update.primitive is jax.lax.add_p
        and isinstance(r := idx_update.invars[1], jcore.Literal)
        and r.val == 1
    ):
        raise AssertionError("First equation in loop body is not an index update")

    pp_eqn_idx = first_pipeline_yield_eqn_idx(eqns)
    if pp_eqn_idx is None:
        # FIXME: is defaulting to MpmdIdx(0) ok?
        return [Cluster(MpmdIdx(0), TaskType.FWD, [idx_update, *eqns], stage_id=0)], []

    stage_0, eqns = schedule_dependencies(eqns, pp_eqn_idx)
    curr_enter_eqn = stage_0.pop()
    stage_0 = [idx_update, *stage_0]
    stages: list[Cluster] = [
        Cluster(
            get_mpmd_idx(curr_enter_eqn.params["from_stage_id"], mpmd_dim),
            TaskType.FWD,
            stage_0,
            stage_id=curr_enter_eqn.params["from_stage_id"],
        )
    ]

    while (pp_eqn_idx := first_pipeline_yield_eqn_idx(eqns)) is not None:
        stage_i, eqns = schedule_dependencies(eqns, pp_eqn_idx)
        next_enter_eqn = stage_i.pop()
        stage_id = next_enter_eqn.params["from_stage_id"]
        mpmd_idx = get_mpmd_idx(stage_id, mpmd_dim)
        stages.append(
            Cluster(
                mpmd_idx,
                curr_enter_eqn.params["task_type"],
                [curr_enter_eqn] + stage_i,
                stage_id=stage_id,
            )
        )
        curr_enter_eqn = next_enter_eqn

    stages.append(
        Cluster(
            get_mpmd_idx(curr_enter_eqn.params["to_stage_id"], mpmd_dim),
            curr_enter_eqn.params["task_type"],
            [curr_enter_eqn],
            stage_id=curr_enter_eqn.params["to_stage_id"],
        )
    )
    return stages, eqns


def cluster_eqns(
    eqns: list[jcore.JaxprEqn],
    mpmd_dim: int,
    bias: dict[jcore.Var, set[MpmdIdx]] | None = None,
) -> tuple[list[Cluster], list[jcore.JaxprEqn]]:
    bias = bias or {}
    clusters, rest = cluster_by_yield_eqns(eqns, mpmd_dim)
    eqns_cluster_idxs = infer_cluster_idx_for_eqns(clusters, rest)
    unclustered_eqns = list[jcore.JaxprEqn]()
    for cluster_idx, eqn in zip(eqns_cluster_idxs, rest, strict=True):
        if cluster_idx is not None:
            clusters[cluster_idx].eqns.append(eqn)
        else:
            unclustered_eqns.append(eqn)
    return clusters, unclustered_eqns


def clusters_to_tasks(
    clusters: list[Cluster], outvars: Iterable[jcore.Var]
) -> list[jcore.JaxprEqn]:
    undef = set[jcore.Var](outvars)
    rev_stage_eqns = []
    for mpmd_idx, ty, stage_eqns, maybe_stage_id in reversed(clusters):
        assert maybe_stage_id is not None
        task_name = f"{ty.name.lower()}_{maybe_stage_id}"
        if len(stage_eqns) == 0:
            logger.warning(f"Empty stage {task_name}")
        free, defs = eqns_free_vars(stage_eqns)
        stage_eqn = make_task_eqn(
            sorted(free), sorted(defs & undef), stage_eqns, mpmd_idx, task_name
        )
        rev_stage_eqns.append(stage_eqn)
        undef.difference_update(defs)
        undef.update(free)

        bytes_str = hbytes(a.aval for a in stage_eqn.outvars)
        logger.info(f"Activation size for {task_name}: {bytes_str}")

    return list(reversed(rev_stage_eqns))


def wrap_into_tasks_inside_loop(
    loop_eqn: jcore.JaxprEqn,
    mpmd_dim: int,
    bias: dict[jcore.Var, set[MpmdIdx]] | None = None,
) -> jcore.JaxprEqn:
    bias = bias or {}
    jaxpr: jcore.Jaxpr = loop_eqn.params["jaxpr"].jaxpr
    # TODO: let bind literals
    assert len(jaxpr.outvars) == len(
        set(jaxpr.outvars)
    ), "Literal outvars (hash error) or duplicate outvars not supported"

    clusters, unclustered_eqns = cluster_eqns(jaxpr.eqns, mpmd_dim, bias)
    assert len(unclustered_eqns) == 0
    del unclustered_eqns

    target_num_stages = loop_eqn.params["schedule"].num_stages
    inferred_num_stages, rem = divmod(len(clusters) + 1, 2)
    if rem != 0:
        raise AssertionError(
            f"Expected even number of stages, {len(clusters) + 1} found"
        )

    if inferred_num_stages != target_num_stages:
        raise AssertionError(
            f"Unexpected number of pipeline markers: found {inferred_num_stages} "
            f"expected {target_num_stages}"
        )

    clustered_jaxpr = jaxpr.replace(eqns=clusters_to_tasks(clusters, jaxpr.outvars))

    # Infer where loop inputs are used (refs) and where loop outputs
    # are defined (defs)
    clustered_inferred_jaxpr, in_mpmd_refs, out_mpmd_defs = compute_loop_placement(
        clustered_jaxpr, loop_eqn.params["n_consts"]
    )

    in_sharding_store, in_inspect = ShardingStore.collect_jaxpr(
        clustered_inferred_jaxpr.invars
    )
    out_sharding_store, out_inspect = ShardingStore.collect_jaxpr(
        clustered_inferred_jaxpr.outvars
    )
    final_eqns = in_inspect + clustered_inferred_jaxpr.eqns + out_inspect
    new_jaxpr = clustered_inferred_jaxpr.replace(
        eqns=final_eqns,
        effects=jcore.join_effects(*(eqn.effects for eqn in final_eqns)),
    )

    check_jaxpr(new_jaxpr)

    return loop_eqn.replace(
        params={
            **loop_eqn.params,
            "jaxpr": loop_eqn.params["jaxpr"].replace(jaxpr=new_jaxpr),
            "in_shardings": in_sharding_store,
            "out_shardings": out_sharding_store,
            "in_mpmd_refs": in_mpmd_refs,
            "out_mpmd_defs": out_mpmd_defs,
        },
        effects=new_jaxpr.effects,
    )


@unwrap_closed
def strip_inspect_sharding_eqns(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive is inspect_sharding_p:
            continue
        if eqn.primitive is task_p or eqn.primitive is dax_pscan_p:
            key = ["jaxpr", "call_jaxpr"][eqn.primitive is task_p]
            new_jaxpr = strip_inspect_sharding_eqns(eqn.params[key])
            new_eqns.append(
                eqn.replace(
                    params={**eqn.params, key: new_jaxpr},
                    effects=new_jaxpr.effects,
                )
            )
        else:
            new_eqns.append(eqn)

    new_effects = jcore.join_effects(*(eqn.effects for eqn in new_eqns))
    return jaxpr.replace(eqns=new_eqns, effects=new_effects)


class DistributedFunction:
    def __init__(
        self,
        consts: Sequence[jax.Array],
        in_tree,
        out_tree,
        in_avals: Sequence[jcore.AbstractValue],
        out_avals: Sequence[jcore.AbstractValue],
        in_shardings: Sequence[jax.sharding.NamedSharding],
        out_shardings: Sequence[jax.sharding.NamedSharding],
        in_mpmd_idxs: Sequence[set[int]],
        out_mpmd_idxs: Sequence[set[int]],
        in_uids: Sequence[UID],
        out_uids: Sequence[UID],
        donated_invars: Sequence[bool],
        used_invars: list[bool],
        worker_mesh: RemoteMpmdMesh,
        instructions_by_worker: list[list[Op]],
        passthrough_outvars: list[int | None],
    ):
        self.consts = tuple(consts)
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.in_avals = in_avals
        self.out_avals = out_avals
        self._in_shardings = in_shardings
        self.out_shardings = out_shardings
        self.in_mpmd_idxs = in_mpmd_idxs
        self.out_mpmd_idxs = out_mpmd_idxs
        self.in_uids = in_uids
        self.out_uids = out_uids

        self.donated_invars = donated_invars
        self.used_invars = used_invars
        self.worker_mesh = worker_mesh
        self.instructions_by_worker = instructions_by_worker
        self.comm_keys = get_comm_keys(self.instructions_by_worker)
        if isinstance(worker_mesh, RemoteMpmdMesh):
            self.instructions_by_worker = [
                ray.put(worker_instructions)
                for worker_instructions in instructions_by_worker
            ]
        self.passthrough_outvars = passthrough_outvars

        self.comm_established = False

    @cached_property
    def in_shardings(self):
        res = jax.tree_util.tree_unflatten(
            self.in_tree,
            [
                DistributedSharding(mpmd_idxs, sharding)
                for mpmd_idxs, sharding in zip(
                    self.in_mpmd_idxs, self._in_shardings, strict=True
                )
            ][len(self.consts) :],
        )
        return res

    def __call__(self, *args):
        if not self.comm_established:
            self.worker_mesh.establish_nccl_comms(self.comm_keys)
            self.comm_established = True

        flat_args_w_path, in_tree = jax.tree_util.tree_flatten_with_path(args)
        assert in_tree == self.in_tree

        local_indices = list[int]()
        # TODO `deletions` below seems unnecessary? deletion computation already covers donations
        deletions = defaultdict[int, list[UID]](list)
        for (idx, (path, a)), used, donated in ju.safe_zip(
            enumerate(it.chain(zip(it.repeat(None), self.consts), flat_args_w_path)),
            self.used_invars,
            self.donated_invars,
        ):
            if used:
                if not isinstance(a, ArrayRef):
                    local_indices.append(idx)
                else:
                    if a.deleted:
                        keystr = jax.tree_util.keystr(path)
                        raise TypeError(
                            f"Consumed ArrayRef passed as argument {keystr}"
                        )
                    if donated:
                        a.deleted = True
                        for mpmd_idx in a.mpmd_idxs:
                            deletions[mpmd_idx].append(a.uid)

        _, flat_args = ju.unzip2(flat_args_w_path)
        flat_args = list(self.consts + flat_args)

        put_args = list[PutArg]()
        for in_idx in local_indices:
            uid = fresh_scalar_uid()
            mpmd_idxs = self.in_mpmd_idxs[in_idx]
            sharding = self._in_shardings[in_idx]
            val = flat_args[in_idx]
            flat_args[in_idx] = ArrayRef(
                ArrayRefSharding(mpmd_idxs, sharding),
                uid,
                self.worker_mesh,
                self.in_avals[in_idx],
            )
            put_args.append(
                PutArg(
                    uid=uid,
                    value=val,
                    sharding=SerializeableSharding(sharding),
                    mpmd_idxs=mpmd_idxs,
                )
            )

        self.worker_mesh.put_tensors(put_args)

        results = [
            flat_args[passthrough_idx]
            if passthrough_idx is not None
            else ArrayRef(
                ArrayRefSharding(
                    self.out_mpmd_idxs[out_idx], self.out_shardings[out_idx]
                ),
                fresh_scalar_uid(),
                self.worker_mesh,
                self.out_avals[out_idx],
            )
            for out_idx, passthrough_idx in enumerate(self.passthrough_outvars)
        ]

        for mpmd_idx in range(self.worker_mesh.mpmd_dim):
            mpmd_in_binding = []
            for arg_array_ref, in_mpmd_idxs, in_uid, used in zip(
                flat_args,
                self.in_mpmd_idxs,
                self.in_uids,
                self.used_invars,
                strict=True,
            ):
                if used and mpmd_idx in in_mpmd_idxs:
                    mpmd_in_binding.append(Bind(from_=arg_array_ref.uid, to_=in_uid))

            mpmd_out_binding = []
            for out_mpmd_idxs, out_uid, res_array_ref in zip(
                self.out_mpmd_idxs, self.out_uids, results, strict=True
            ):
                if mpmd_idx in out_mpmd_idxs:
                    mpmd_out_binding.append(Bind(from_=out_uid, to_=res_array_ref.uid))

            self.worker_mesh.execute_instructions(
                mpmd_idx,
                mpmd_in_binding,
                self.instructions_by_worker[mpmd_idx],
                mpmd_out_binding,
                deletions[mpmd_idx],
            )

        return jax.tree_util.tree_unflatten(self.out_tree, results)


def last_used(jaxpr: jcore.Jaxpr) -> dict[jcore.Var, int | None]:
    """
    Index variant of `jax._src.core.last_used`
    Returns a mapping from every var in jaxpr to what equation index uses it last.
    If a var is returned then its last use is `None`.
    """
    last_used: dict[jcore.Var, int | None] = {
        v: None for v in jaxpr.outvars if not isinstance(v, jcore.Literal)
    }
    for idx, eqn in reversed(list(enumerate(jaxpr.eqns))):
        for v in eqn.invars:
            if not isinstance(v, jcore.Literal) and v not in last_used:
                last_used[v] = idx
    return last_used


def compute_send_and_recv(
    loop_jaxpr: PscanJaxpr,
    n_consts: int,
    mpmd_def: dict[jcore.Var, set[int]],
    mpmd_refs: dict[jcore.Var, set[int]],
) -> PscanJaxpr:
    """
    Computes the send and receive operations for a given `loop_jaxpr`.

    Args:
        loop_jaxpr (PscanJaxpr)
        n_consts (int): Number of constants in the loop.
        mpmd_def (dict[jcore.Var, set[int]]): Mapping of variables to their defined
          MPMD indices.
        mpmd_refs (dict[jcore.Var, set[int]]): Mapping of variables to their
          referenced MPMD indices.
        mpmd_dim (int): Dimension of the MPMD pipeline.
    """

    # The caller of this function must ensure that
    #  the definition mpmd indices of constant variables satisfy
    #  all their uses
    for constvar in loop_jaxpr.invars[:n_consts]:
        assert mpmd_refs[constvar].issubset(mpmd_def[constvar])

    loop_constvars = set(loop_jaxpr.invars[:n_consts])
    loop_jaxpr_outvars_idx = {outvar: i for i, outvar in enumerate(loop_jaxpr.outvars)}
    received_by_mpmd_idx = defaultdict[int, set[jcore.Var]](set)

    _, irefs = ivar_defs_and_refs(loop_jaxpr)
    new_eqns = []
    for eqn_idx, eqn in enumerate(loop_jaxpr.eqns):
        # Compute recvs
        eqn_mpmd_idx = get_task_mpmd_idx(eqn)
        recv_invars = list[tuple[int, int]]()
        for invar_idx, invar in enumerate(eqn.invars):
            # constvars are already placed correctly (no need to receive them)
            #  as checked above
            if invar not in loop_constvars:
                if (
                    eqn_mpmd_idx not in (def_mpmd_idxs := mpmd_def[invar])
                    # The condition below ensures that if an invar is used twice
                    # by an mpmd_idx in different stages, we receive it only once
                    and invar
                    not in (already_received := received_by_mpmd_idx[eqn_mpmd_idx])
                ):
                    # NOTE: this asserts that `def_mpmd_idxs` is a singleton set
                    (def_mpmd_idx,) = def_mpmd_idxs
                    recv_invars.append((invar_idx, def_mpmd_idx))
                    already_received.add(invar)

        # Compute sends
        send_outvars = list[tuple[int, list[int]]]()
        for outvar_idx, outvar in enumerate(eqn.outvars):
            backedge_refs = set()
            # If this is a loop output
            if (loop_outvar_idx := loop_jaxpr_outvars_idx.get(outvar)) is not None:
                # Then it must be also a loop input
                loop_invar = loop_jaxpr.invars[n_consts + loop_outvar_idx]

                eqn_idxs = irefs[loop_invar]
                # If this output is used in an earlier equation in the next iteration
                # that is scheduled in a different mpmd_idx then we raise an error,
                # because it would lead to a slow pipeline.
                if any(
                    idx < eqn_idx
                    and eqn_mpmd_idx != get_task_mpmd_idx(loop_jaxpr.eqns[eqn_idx])
                    for idx in eqn_idxs
                    if idx is not None
                ):
                    raise AssertionError("Loop backedge detected")

                backedge_refs = mpmd_refs[loop_invar] - {eqn_mpmd_idx}

            mpmd_idxs = [
                mpmd_idx
                for mpmd_idx in sorted(mpmd_refs[outvar] | backedge_refs)
                if mpmd_idx != eqn_mpmd_idx
            ]
            if len(mpmd_idxs) > 0:
                send_outvars.append((outvar_idx, mpmd_idxs))

        params = dict(eqn.params)
        params["recv_invars"] = recv_invars
        params["send_outvars"] = send_outvars

        new_eqns.append(eqn.replace(params=params))

    return loop_jaxpr.replace(eqns=new_eqns)


def compute_loop_placement(loop_jaxpr: PscanJaxpr, n_consts: int):
    mpmd_def, mpmd_refs = (
        # For `mpmd_def`, the value is a singleton set for all cases
        #  except when it is a constant invar. Only constants can be replicated.
        dict[jcore.Var, set[int]](),
        defaultdict[jcore.Var, set[int]](set),
    )
    for eqn in loop_jaxpr.eqns:
        eqn_mpmd_idx = get_task_mpmd_idx(eqn)
        for invar in eqn.invars:
            mpmd_refs[invar].add(eqn_mpmd_idx)

        for outvar in eqn.outvars:
            mpmd_def[outvar] = {eqn_mpmd_idx}

    for invar, outvar in ju.safe_zip(
        loop_jaxpr.invars[n_consts:],
        loop_jaxpr.outvars,
    ):
        # State invars are defined where their corresponding
        #  outvars are defined
        mpmd_def[invar] = mpmd_def[outvar]

        # Check that the mpmd_index that produces an outvar
        #  is a subset of the ones that refer to it.
        # Note that, although `mpmd_def[outvar]` is a set, only one
        #  mpmd_idx produces an outvar since we don't allow replicated
        #  computation in the loop
        (mpmd_idx,) = mpmd_def[outvar]
        if len(mpmd_refs[invar]) > 0 and mpmd_idx not in mpmd_refs[invar]:
            raise AssertionError("Loop state is not stable across iterations")

    # Loop constants must be defined where they are referred
    for invar in loop_jaxpr.invars[:n_consts]:
        mpmd_def[invar] = mpmd_refs[invar]

    loop_jaxpr = compute_send_and_recv(loop_jaxpr, n_consts, mpmd_def, mpmd_refs)

    loop_invar_mpmd_refs = [mpmd_refs[invar] for invar in loop_jaxpr.invars]
    loop_outvar_mpmd_def = [mpmd_def[outvar] for outvar in loop_jaxpr.outvars]
    return loop_jaxpr, loop_invar_mpmd_refs, loop_outvar_mpmd_def


def make_replicated_jaxpr(
    jaxpr: jcore.Jaxpr,
    outvar_mpmd_refs: Sequence[set[MpmdIdx]],
    mpmd_indices: Iterable[MpmdIdx],
) -> tuple[list[jcore.Jaxpr], list[set[MpmdIdx] | None]]:
    assert len(jaxpr.outvars) == len(outvar_mpmd_refs)
    invar_mpmd_refs: list[set[MpmdIdx] | None] = [None] * len(jaxpr.invars)
    res = []
    for mpmd_idx in mpmd_indices:
        dced_jaxpr, used_inputs = pe.dce_jaxpr(
            jaxpr,
            used_outputs=[mpmd_idx in place for place in outvar_mpmd_refs],
        )
        res.append(dced_jaxpr)
        for invar_idx, used in enumerate(used_inputs):
            if used:
                p = invar_mpmd_refs[invar_idx]
                if p is None:
                    p = set[MpmdIdx]()
                    invar_mpmd_refs[invar_idx] = p
                p.add(mpmd_idx)

    return res, invar_mpmd_refs


def infer_outvar_placement_rev(
    jaxpr: jcore.Jaxpr, partial_outvar_placement: Iterable[set[MpmdIdx] | None]
) -> tuple[list[set[MpmdIdx]], list[set[MpmdIdx]]]:
    outvars = cast(list[jcore.Var], jaxpr.outvars)
    placement = {
        outvar: maybe_p
        for outvar, maybe_p in ju.safe_zip(outvars, partial_outvar_placement)
        if maybe_p is not None
    }

    # Infer from outvars to invars
    for eqn in reversed(jaxpr.eqns):
        eqn_p = set.union(
            set(), *(placement.get(outvar, set()) for outvar in eqn.outvars)
        )
        if len(eqn_p) > 0:
            for invar in nonlit(eqn.invars):
                placement[invar] = placement.get(invar, set()) | eqn_p

    # Infer from invars to outvars
    for eqn in jaxpr.eqns:
        eqn_p = set.union(
            set(), *(placement.get(invar, set()) for invar in nonlit(eqn.invars))
        )
        if len(eqn_p) > 0:
            for outvar in eqn.outvars:
                placement[outvar] = placement.get(outvar, set()) | eqn_p

    return [placement[invar] for invar in jaxpr.invars], [
        placement[outvar] for outvar in outvars
    ]


def get_one_loop_eqn_idx(
    eqns_or_jaxpr: jcore.ClosedJaxpr | jcore.Jaxpr | Iterable[jcore.JaxprEqn],
) -> int:
    eqns = eqns_or_jaxpr
    if isinstance(eqns_or_jaxpr, (jcore.ClosedJaxpr, jcore.Jaxpr)):
        eqns = eqns_or_jaxpr.eqns

    loop_eqn_idxs = [idx for idx, e in enumerate(eqns) if e.primitive is dax_pscan_p]
    if len(loop_eqn_idxs) != 1:
        raise AssertionError(
            "Expected 1 `accumulate_grads` loop at the top level "
            f"but {len(loop_eqn_idxs)} found."
        )
    return loop_eqn_idxs[0]


def make_ar_device_keys(
    mpmd_mesh: MpmdMesh, mpmd_idxs: list[int]
) -> RichDict[GlobalDeviceId, UniqueGlobalDeviceIds]:
    groups = ju.safe_zip(
        *(
            mpmd_group_mesh.devices.flat
            for mpmd_group_mesh in mpmd_mesh.mpmd_submesh(mpmd_idxs).unstack
        )
    )
    keys = RichDict[GlobalDeviceId, UniqueGlobalDeviceIds]()
    for devices in groups:
        key = UniqueGlobalDeviceIds.strict_create(d.id for d in devices)
        for d in devices:
            keys.set_or_raise_if_present(d.id, key)
    return keys


def log_activation_shardings(closed_jaxpr: jcore.ClosedJaxpr):
    [loop_eqn] = [
        eqn for eqn in closed_jaxpr.jaxpr.eqns if eqn.primitive is dax_pscan_p
    ]
    stage_eqns = [
        eqn for eqn in loop_eqn.params["jaxpr"].eqns if eqn.primitive is task_p
    ]
    logger.info("shardings/activations")
    for eqn in stage_eqns:
        logger.info(f"{eqn.params['name']}")
        for outvar, sharding in ju.safe_zip(
            eqn.outvars, eqn.params["out_shardings"].shardings
        ):
            logger.info(
                f"\t{outvar.aval.shape}, "
                f"{sharding._to_xla_hlo_sharding(outvar.aval.ndim)}"
            )


def wrap_into_tasks_before_loop(
    jaxpr: jcore.Jaxpr,
    partial_mpmd_refs: Mapping[jcore.Var, set[MpmdIdx] | None],
    mpmd_dim: int,
) -> tuple[
    list[jcore.JaxprEqn],
    dict[jcore.Var, set[MpmdIdx] | None],
    dict[jcore.Var, set[MpmdIdx]],
]:
    """
    NOTE: for tasks before and after the loop, the same outvar (object reference)
    can be "defined" by multiple tasks.
    This deviates from "canonical" JAX/Jaxprs, or any ANF-style IR and one should
    take precautions when manipulating or especially using those objects
    to track metadata in a dictionary.
    """
    loop_eqn_idx = get_one_loop_eqn_idx(jaxpr)
    # NOTE: although all the equations preceeding the loop are only the ones
    #  the loop depends on, `before_loop_free_vars` can contain additional variables
    #  that are not input to the loop.
    #  This happens from multiple arity primitives like `a, b = jax.random.split(rng)`
    #  where one of the two results is argument to the loop while the other one
    #  used only later
    before_loop_jaxpr = jaxpr_from_eqns(
        jaxpr.eqns[:loop_eqn_idx], eqns_free_vars(jaxpr.eqns[loop_eqn_idx:])[0]
    )

    # For outputs of the before loop part that are not used in the loop
    # we leave the placement as `None`
    maybe_before_loop_outvar_placement = [
        partial_mpmd_refs.get(outvar) for outvar in before_loop_jaxpr.outvars
    ]

    # NOTE: this inference can widen (replicate) before_loop equations too
    #  much. This is checked below when we check the `before_loop_invar_mpmd_refs`
    #  against the `mpmd_def`.
    #  That check ensures that loop placements have higher priority
    #  and if the before_loop equations replication breaks such placements
    #  an error is raised
    _, before_loop_outvar_mpmd_defs = infer_outvar_placement_rev(
        before_loop_jaxpr,
        partial_outvar_placement=maybe_before_loop_outvar_placement,
    )

    before_loop_mpmd_jaxprs, before_loop_invar_placement = make_replicated_jaxpr(
        before_loop_jaxpr, before_loop_outvar_mpmd_defs, map(MpmdIdx, range(mpmd_dim))
    )

    logger.info(
        f"Before loop: {hbytes(outvar.aval for outvar in before_loop_jaxpr.outvars)}"
    )
    replication_factor = [
        (i, len(j.eqns) / len(before_loop_jaxpr.eqns))
        for i, j in enumerate(before_loop_mpmd_jaxprs)
    ]
    logger.info(f"Before loop replication {replication_factor=}")

    task_eqns = list[jcore.JaxprEqn]()
    for mpmd_idx, j in enumerate(before_loop_mpmd_jaxprs):
        task_eqns.append(
            make_task_eqn(
                invars=j.invars,
                outvars=j.outvars,
                eqns=j.eqns,
                mpmd_idx=mpmd_idx,
                task_name=f"before_loop_{mpmd_idx}",
            )
        )
    return (
        task_eqns,
        dict(zip(before_loop_jaxpr.invars, before_loop_invar_placement, strict=True)),
        dict(
            zip(
                cast(list[jcore.Var], before_loop_jaxpr.outvars),
                before_loop_outvar_mpmd_defs,
                strict=True,
            )
        ),
    )


def check_loop_invars_with_before_loop_invars(
    loop_mpmd_refs: dict[jcore.Var, set[MpmdIdx] | None],
    before_loop_invar_mpmd_refs: Mapping[jcore.Var, set[MpmdIdx] | None],
):
    for invar, mpmd_idxs in loop_mpmd_refs.items():
        if (loop_mpmd_idxs := before_loop_invar_mpmd_refs.get(invar)) is not None:
            if loop_mpmd_idxs is not None:
                if mpmd_idxs != loop_mpmd_idxs:
                    raise AssertionError(
                        f"Loop placement is {loop_mpmd_idxs} while before loop placement is {mpmd_idxs}"
                    )
            else:
                loop_mpmd_refs[invar] = mpmd_idxs


def wrap_into_tasks_after_loop(
    jaxpr: jcore.Jaxpr, used_invars: Sequence[bool], mpmd_dim: int
) -> tuple[jcore.Jaxpr, dict[jcore.Var, set[MpmdIdx]]]:
    """
    NOTE: for tasks before and after the loop, the same outvar (object reference)
    can be "defined" by multiple tasks.
    This deviates from "canonical" JAX/Jaxprs, or any ANF-style IR and one should
    take precautions when manipulating or especially using those objects
    to track metadata in a dictionary.
    """
    loop_eqn_idx = get_one_loop_eqn_idx(jaxpr)

    mpmd_refs = defaultdict[jcore.Var, set[MpmdIdx]](set)
    mpmd_def = defaultdict[jcore.Var, set[MpmdIdx]](set)
    for eqn in jaxpr.eqns[: loop_eqn_idx + 1]:
        if eqn.primitive is task_p:
            task_eqn = TaskEqn.make(eqn)
            mpmd_idx = get_task_mpmd_idx(task_eqn)
            for invar in task_eqn.invars:
                mpmd_refs[invar].add(mpmd_idx)
            for outvar in task_eqn.outvars:
                # NOTE: before loop vars can be defined multiple times
                mpmd_def[outvar].add(mpmd_idx)
        elif eqn.primitive is dax_pscan_p:
            for invar, refs in zip(eqn.invars, eqn.params["in_mpmd_refs"], strict=True):
                assert not isinstance(invar, jcore.Literal), "Unimplemented"
                mpmd_refs[invar].update(refs)
            for outvar, defs in zip(
                eqn.outvars, eqn.params["out_mpmd_defs"], strict=True
            ):
                assert outvar not in mpmd_def
                mpmd_def[outvar] = defs
        else:
            raise AssertionError(f"Unexpected equation {eqn.primitive}")

    new_eqns = list[jcore.JaxprEqn](jaxpr.eqns[: loop_eqn_idx + 1])

    after_loop_jaxpr = jaxpr_from_eqns(
        jaxpr.eqns[loop_eqn_idx + 1 :], set(nonlit(jaxpr.outvars))
    )

    (
        coarsened_after_loop_jaxpr,
        after_loop_invar_mpmd_refs,
        after_loop_outvar_placement,
        eqns_in_mpmd_idx,
    ) = mpmd_unzip_forward(
        after_loop_jaxpr,
        # NOTE: inputs to `after_loop_jaxpr` that might have not been
        #  used so far (such as optimizer state), might not have an mpmd_idx
        #  defined just yet. Hence `.get(invar)` instead of `[invar]`.
        [mpmd_def.get(invar) for invar in after_loop_jaxpr.invars],
        mpmd_dim,
    )

    for invar, after_loop_p in ju.safe_zip(
        cast(list[jcore.Var], after_loop_jaxpr.invars), after_loop_invar_mpmd_refs
    ):
        # This assertion is always true in theory, we leave it here defensively
        #  for potential future changes
        assert after_loop_p is not None
        if (p := mpmd_def.get(invar)) and after_loop_p != p:
            raise NotImplementedError(
                "Loop output used in a MPMD index different from the defining one. "
                f"Defined at {p} and used at {after_loop_p}."
            )

        mpmd_refs[invar].update(after_loop_p)

    replication_factor = [
        (mpmd_idx, n_eqns / len(after_loop_jaxpr.eqns))
        for mpmd_idx, n_eqns in eqns_in_mpmd_idx.items()
    ]
    logger.info(f"After loop replication {replication_factor=}")

    new_eqns.extend(coarsened_after_loop_jaxpr.eqns)

    mpmd_def.update(
        zip(
            cast(list[jcore.Var], after_loop_jaxpr.outvars),
            after_loop_outvar_placement,
            strict=True,
        )
    )

    for invar, is_used in zip(jaxpr.invars, used_invars):
        if is_used:
            refs = mpmd_refs.get(invar)
            assert refs is not None
            mpmd_def[invar] = refs
        else:
            assert invar not in mpmd_def
            mpmd_def[invar] = set()

    return jaxpr.replace(eqns=new_eqns), mpmd_def


def more_sharded_sharding(prev_sharding, alt_sharding, shape):
    prev_sharded_shape = prev_sharding.shard_shape(shape)
    sharded_shape = alt_sharding.shard_shape(shape)
    return (
        prev_sharding
        if math.prod(prev_sharded_shape) <= math.prod(sharded_shape)
        else alt_sharding
    )


def reconcile_sharding_for_replicated_vars(
    cjaxpr: jcore.ClosedJaxpr, in_shardings, out_shardings
):
    class Use(NamedTuple):
        is_invar: bool
        eqn_idx: int
        var_idx: int

    class Update(NamedTuple):
        sharding: jax.sharding.Sharding
        uses: list[Use]

    shardings = defaultdict[jcore.Var, Update](lambda: Update(None, []))
    eqns: list[jcore.JaxprEqn] = cjaxpr.eqns
    for eqn_idx, eqn in enumerate(eqns):
        in_shardings = eqn.params["in_shardings"].shardings
        out_shardings = eqn.params["out_shardings"].shardings
        for var_idx, (is_invar, var_, curr_sharding) in it.chain(
            enumerate(zip(it.repeat(True), eqn.invars, in_shardings)),
            enumerate(zip(it.repeat(False), eqn.outvars, out_shardings)),
        ):
            if (update := shardings.get(var_)) is None:
                update = Update(curr_sharding, [])
                shardings[var_] = update

            update.uses.append(Use(is_invar, eqn_idx, var_idx))
            prev_sharding = update.sharding
            if prev_sharding is not None and curr_sharding != prev_sharding:
                if eqn.primitive is dax_pscan_p:
                    # NOTE(reconcile_sharding): while this is correct in principle
                    #   it might lead to poorer performance than observed before.
                    #   Therefore we decide to handle this in `lower_tasked_jaxpr`
                    shardings[var_] = shardings[var_]._replace(sharding=curr_sharding)
                else:
                    shardings[var_] = shardings[var_]._replace(
                        sharding=more_sharded_sharding(
                            prev_sharding, curr_sharding, var_.aval.shape
                        ),
                    )

    for sharding, uses in shardings.values():
        for is_invar, eqn_idx, var_idx in uses:
            if is_invar:
                shardings = cjaxpr.eqns[eqn_idx].params["in_shardings"].shardings
            else:
                shardings = cjaxpr.eqns[eqn_idx].params["out_shardings"].shardings
            shardings[var_idx] = sharding


def loop_placement_by_clusters(
    loop_eqn: jcore.JaxprEqn, mpmd_dim: int
) -> tuple[list[set[MpmdIdx] | None], list[MpmdIdx | None]]:
    assert loop_eqn.primitive is dax_pscan_p
    n_consts = loop_eqn.params["n_consts"]
    jaxpr: jcore.Jaxpr = loop_eqn.params["jaxpr"].jaxpr

    invar_idx = {invar: idx for idx, invar in enumerate(jaxpr.invars)}
    outvar_idx = {outvar: idx for idx, outvar in enumerate(jaxpr.outvars)}

    clusters, _ = cluster_eqns(jaxpr.eqns, mpmd_dim)
    cluster_info = get_cluster_information(clusters)

    in_mpmd_refs: list[set[MpmdIdx] | None] = [None] * len(invar_idx)
    out_mpmd_defs: list[MpmdIdx | None] = [None] * len(outvar_idx)

    for invar, ref_cluster_idxs in cluster_info.var_ref_cluster_idx.items():
        if (idx := invar_idx.get(invar)) is not None:
            mpmd_refs = in_mpmd_refs[idx]
            if mpmd_refs is None:
                mpmd_refs = set[MpmdIdx]()
                in_mpmd_refs[idx] = mpmd_refs
            for cluster_idx in ref_cluster_idxs:
                mpmd_refs.add(clusters[cluster_idx].mpmd_idx)

    for outvar, def_cluster_idx in cluster_info.var_def_cluster_idx.items():
        if (idx := outvar_idx.get(outvar)) is not None:
            out_mpmd_defs[idx] = clusters[def_cluster_idx].mpmd_idx

    with stable_names_ctx(
        lambda v: {clusters[idx].mpmd_idx for idx in idxs}
        if (idxs := cluster_info.var_ref_cluster_idx.get(v)) is not None
        else {clusters[idx].mpmd_idx}
        if (idx := cluster_info.var_def_cluster_idx.get(v)) is not None
        else None
    ):
        for in_idx, (mpmd_refs, mpmd_def) in enumerate(
            ju.safe_zip(in_mpmd_refs[n_consts:], out_mpmd_defs), start=n_consts
        ):
            # Check that the mpmd_index that produces an outvar
            #  is a subset of the ones that refer to it.
            if mpmd_refs is not None:
                if mpmd_def not in mpmd_refs:
                    raise AssertionError(
                        f"Loop state is not stable across iterations {in_idx=} {in_idx - n_consts=}"
                    )
            elif mpmd_def is not None:
                in_mpmd_refs[in_idx] = {mpmd_def}

    return in_mpmd_refs, out_mpmd_defs


loop_passes = lambda j: j


# TODO: move `preprocess_loop` to the tracing done by `accum_grads`
def wrap_into_tasks(
    cjaxpr: jcore.ClosedJaxpr, used_invars: Sequence[bool], mpmd_dim: int
) -> tuple[jcore.ClosedJaxpr, dict[jcore.Var, set[MpmdIdx]]]:
    """
    After this pass, all the equations in the returned jaxpr are either
    (1) `task` equations, or (2) a `dax_pscan` equation containing `task` equations
    or (3) `inspect_sharding`.
    """
    jaxpr: jcore.Jaxpr = loop_passes(cjaxpr.jaxpr)

    [*before_loop_eqns, loop_eqn], after_loop_eqns = schedule_dependencies(
        jaxpr.eqns, get_one_loop_eqn_idx(jaxpr.eqns)
    )
    jaxpr = jaxpr.replace(eqns=before_loop_eqns + [loop_eqn] + after_loop_eqns)
    loop_eqn_idx = len(before_loop_eqns)
    loop_eqn = jaxpr.eqns[loop_eqn_idx]

    loop_in_mpmd_refs, loop_out_mpmd_defs = loop_placement_by_clusters(
        loop_eqn, mpmd_dim
    )

    loop_in_mpmd_refs_map = defaultdict[jcore.Var, set[int]](set)
    for invar, refs in zip(loop_eqn.invars, loop_in_mpmd_refs, strict=True):
        if refs is not None:
            loop_in_mpmd_refs_map[invar] |= refs

    before_loop_task_eqns, before_loop_invar_placement, before_loop_outvar_mpmd_defs = (
        wrap_into_tasks_before_loop(jaxpr, loop_in_mpmd_refs_map, mpmd_dim)
    )
    check_loop_invars_with_before_loop_invars(
        loop_in_mpmd_refs_map, before_loop_invar_placement
    )

    task_eqns = list[jcore.JaxprEqn](before_loop_task_eqns)

    task_eqns.append(
        wrap_into_tasks_inside_loop(
            loop_eqn,
            mpmd_dim,
            {k: v for k, v in loop_in_mpmd_refs_map.items() if v is not None},
        )
    )
    new_jaxpr = jaxpr.replace(eqns=task_eqns + jaxpr.eqns[loop_eqn_idx + 1 :])

    new_jaxpr, mpmd_def = wrap_into_tasks_after_loop(new_jaxpr, used_invars, mpmd_dim)
    new_jaxpr = new_jaxpr.replace(
        effects=jcore.join_effects(*(eqn.effects for eqn in new_jaxpr.eqns))
    )

    return cjaxpr.replace(jaxpr=new_jaxpr), mpmd_def


def infer_task_donation(
    task_eqn: jcore.JaxprEqn,
    is_last_use_for_invar: Sequence[bool],
    donated_invars_map: dict[jcore.Var, bool],
):
    assert task_eqn.primitive is task_p
    recv_invars_idxs = set(e[0] for e in task_eqn.params["recv_invars"])
    task_donated_invars = [
        is_last_use_for_invar[invar_idx]
        and donated_invars_map.get(invar, True)
        and invar_idx not in recv_invars_idxs
        for invar_idx, invar in enumerate(task_eqn.invars)
    ]

    return task_eqn.replace(
        params={**task_eqn.params, "donate_invars": task_donated_invars}
    )


def infer_loop_donation(
    loop_eqn: jcore.JaxprEqn,
    is_last_use_for_invar: Sequence[bool],
    donated_invars_map: dict[jcore.Var, bool],
):
    assert loop_eqn.primitive is dax_pscan_p
    n_consts = loop_eqn.params["n_consts"]
    loop_donation = [
        is_last_use_for_invar[invar_idx]
        and donated_invars_map.get(invar, True)
        and invar_idx >= n_consts
        for invar_idx, invar in enumerate(loop_eqn.invars)
    ]

    return loop_eqn.replace(
        params={
            **loop_eqn.params,
            "jaxpr": loop_eqn.params["jaxpr"].replace(
                jaxpr=infer_donation(loop_eqn.params["jaxpr"].jaxpr, loop_donation)
            ),
        }
    )


def infer_donation(
    tasked_jaxpr: jcore.Jaxpr, donated_invars: Sequence[bool]
) -> jcore.Jaxpr:
    """
    Returns a new jaxpr identical to the input jaxpr, where every
    `task` equation has `params["donate_invars"]` set properly, according
    to the lifetime of that variable.
    """
    last_use = last_used(tasked_jaxpr)

    invar_is_donated = dict(zip(tasked_jaxpr.invars, donated_invars))
    is_all_reduce_outvar = set[jcore.Var]()

    new_eqns = []
    for task_eqn_idx, task_eqn in enumerate(tasked_jaxpr.eqns):
        if task_eqn.primitive is all_reduce_p:
            is_all_reduce_outvar.update(task_eqn.outvars)
        is_last_use_for_invar = [
            last_use[invar] == task_eqn_idx and invar not in is_all_reduce_outvar
            for invar in task_eqn.invars
        ]

        if task_eqn.primitive is task_p:
            new_eqns.append(
                infer_task_donation(task_eqn, is_last_use_for_invar, invar_is_donated)
            )
        elif task_eqn.primitive is dax_pscan_p:
            new_eqns.append(
                infer_loop_donation(task_eqn, is_last_use_for_invar, invar_is_donated)
            )
        elif task_eqn.primitive in {inspect_sharding_p, send_p, recv_p, all_reduce_p}:
            new_eqns.append(task_eqn)
        else:
            raise AssertionError(
                f"Unexpected equation with primitive {task_eqn.primitive}"
            )
    return tasked_jaxpr.replace(eqns=new_eqns)


def mpmdify_loop(loop_eqn: jcore.JaxprEqn):
    n_consts = loop_eqn.params["n_consts"]
    n_mubatches = loop_eqn.params["n_mubatches"]
    schedule = loop_eqn.params["schedule"]

    loop_jaxpr: jcore.ClosedJaxpr = loop_eqn.params["jaxpr"]
    loop_jaxpr_outvars = {
        outvar: i for i, outvar in enumerate(loop_jaxpr.jaxpr.outvars)
    }

    is_partial_bwd: bool = loop_eqn.params["schedule"].is_partial_bwd
    if not is_partial_bwd:
        # NOTE: there are 2n - 1 stages because the last one is forward
        #  and backward fused
        n_stages, rem = divmod(len(loop_jaxpr.eqns) + 1, 2)
    else:
        # NOTE: there are 3n - 2 stages because the last one is bwd_0 that
        # fuses bwd_i and bwd_w
        n_stages, rem = divmod(len(loop_jaxpr.eqns) + 2, 3)
    assert rem == 0

    tasks = schedule.tasks(n_mubatches)
    n_ticks = len(tasks[0])
    n_workers = len(tasks)
    gensym = jcore.gensym()
    # FIXME: the current single env for all mpmd_idxs is not strict enough
    #  to check for errors in the schedule however later jaxpr checks
    #  can check that
    envs_by_iteration = list[dict[jcore.Var, jcore.Var]](
        [dict(zip(loop_jaxpr.jaxpr.invars, loop_eqn.invars, strict=True))]
    )
    envs_by_iteration.extend(
        dict(zip(loop_jaxpr.jaxpr.invars[:n_consts], loop_eqn.invars[:n_consts]))
        for _ in range(n_mubatches)
    )
    send_recv_id = it.count()
    eqns_by_mpmd_idx = [list[jcore.JaxprEqn]() for _ in range(n_workers)]
    for tick in range(n_ticks):
        end_of_tick_eqns = [list[jcore.JaxprEqn]() for _ in range(n_workers)]
        end_of_tick_defines = list[tuple[int, list[jcore.Var]]]()
        for worker_id in range(n_workers):
            elems = tasks[worker_id][tick]
            if elems is None:
                continue

            stage_id, mubatch_idx, task_type = elems
            if task_type == TaskType.FWD:
                eqn_idx = stage_id
            else:
                if not schedule.is_partial_bwd:
                    # The last stage's bwd is fused in the forward
                    if stage_id == n_stages - 1:
                        continue
                    eqn_idx = (n_stages - 1) + (n_stages - stage_id - 1)
                else:
                    # The last stage's bwd_I is fused in the forward
                    if task_type == TaskType.BWD_I and stage_id == n_stages - 1:
                        continue
                    # The first stage's bwd_w is fused in the backward
                    if task_type == TaskType.BWD_W and stage_id == 0:
                        continue
                    # For example, list of stages [F0,F1,F2,F3,W3,I2,W2,I1,W1,B0]
                    if task_type == TaskType.BWD_I:
                        eqn_idx = n_stages - 1 + (n_stages - 1 - stage_id) * 2
                    elif task_type == TaskType.BWD_W:
                        eqn_idx = n_stages + (n_stages - 1 - stage_id) * 2
                    else:  # TaskType.BWD
                        eqn_idx = n_stages - 1

            eqn: TaskEqn = loop_jaxpr.eqns[eqn_idx]

            assert eqn.params["task_name"] == f"{task_type.name.lower()}_{stage_id}"
            assert (
                stage_id % n_workers == worker_id
                and worker_id == eqn.params["mpmd_idx"]
            )
            orig_outvars = eqn.outvars
            outvars = [gensym(outvar.aval) for outvar in eqn.outvars]
            eqn = eqn.replace(
                invars=[envs_by_iteration[mubatch_idx][invar] for invar in eqn.invars],
                outvars=outvars,
            )
            eqns_by_mpmd_idx[worker_id].append(eqn)
            end_of_tick_defines.append((mubatch_idx, (orig_outvars, outvars)))

            defined_loop_invars = []
            loop_invar_definition = []
            for orig_outvar, outvar in zip(orig_outvars, outvars):
                if (idx := loop_jaxpr_outvars.get(orig_outvar)) is not None:
                    defined_loop_invars.append(loop_jaxpr.jaxpr.invars[n_consts + idx])
                    loop_invar_definition.append(outvar)
            if len(defined_loop_invars) > 0:
                end_of_tick_defines.append(
                    (mubatch_idx + 1, (defined_loop_invars, loop_invar_definition))
                )

            send_to_mpmd_idx: dict[int, list[int]] = groupby(
                (mpmd_idx, outvar_idx)
                for outvar_idx, mpmd_idxs in eqn.params["send_outvars"]
                for mpmd_idx in mpmd_idxs
            )
            for recv_mpmd_idx, send_outvar_idx in send_to_mpmd_idx.items():
                send_shardings = [
                    eqn.params["out_shardings"].shardings[outvar_idx]
                    for outvar_idx in send_outvar_idx
                ]
                invars = [eqn.outvars[idx] for idx in send_outvar_idx]
                if mubatch_idx == 0:
                    logger.info(
                        f"Comm {eqn.params['task_name']} {worker_id} -> {recv_mpmd_idx}: "
                        f"{hbytes(invar.aval for invar in invars)}"
                    )
                op_id = next(send_recv_id)
                params = {
                    "id": op_id,
                    "shardings": tuple(
                        zip((recv_mpmd_idx,) * len(invars), send_shardings, strict=True)
                    ),
                }
                send_eqn = jcore.new_jaxpr_eqn(
                    invars=invars,
                    outvars=[gensym(v.aval) for v in invars],
                    primitive=send_p,
                    params=params,
                    effects=jcore.no_effects,  # FIXME
                )
                recv_eqn = jcore.new_jaxpr_eqn(
                    invars=[],
                    outvars=invars,
                    primitive=recv_p,
                    params={
                        **params,
                        "shape_and_dtype": [
                            (v.aval.shape, v.aval.dtype) for v in invars
                        ],
                        "shardings": tuple(
                            zip((worker_id,) * len(invars), send_shardings, strict=True)
                        ),
                    },
                    effects=jcore.no_effects,  # FIXME
                )
                end_of_tick_eqns[worker_id].append(send_eqn)
                end_of_tick_eqns[recv_mpmd_idx].append(recv_eqn)

        for mpmd_idx, eqns in enumerate(end_of_tick_eqns):
            eqns_by_mpmd_idx[mpmd_idx].extend(eqns)
        for mubatch_idx, (prev_vars, curr_vars) in end_of_tick_defines:
            envs_by_iteration[mubatch_idx].update(zip(prev_vars, curr_vars))

    return [
        envs_by_iteration[n_mubatches - 1][outvar]
        for outvar in loop_jaxpr.jaxpr.outvars
    ], eqns_by_mpmd_idx


def mpmdify(tasked_jaxpr: jcore.ClosedJaxpr, mpmd_dim: int) -> list[jcore.ClosedJaxpr]:
    # TODO: add token args/results to communications
    jaxpr: jcore.Jaxpr = tasked_jaxpr.jaxpr
    eqns_by_mpmd_idx = [list[jcore.JaxprEqn]() for _ in range(mpmd_dim)]
    sub = dict[jcore.Var, jcore.Var]()
    for eqn in jaxpr.eqns:
        eqn = eqn.replace(
            invars=[
                sub.get(invar, invar) if isinstance(invar, jcore.Var) else invar
                for invar in eqn.invars
            ]
        )
        if eqn.primitive is task_p:
            # NOTE(all_reduce_task): almost all the time a task_eqn has
            #  `isinstance(mpmd_idx, int) == True`
            #  except for these all-reduce tasks that have `mpmd_idx : tuple[int, ...]`
            mpmd_idx: int | tuple[int, ...] = eqn.params["mpmd_idx"]
            if isinstance(mpmd_idx, tuple):
                assert len(eqn.params["call_jaxpr"].eqns) == 1
                assert eqn.params["call_jaxpr"].eqns[0].primitive is add_multi_p
                for invar, idx in zip(eqn.invars, mpmd_idx, strict=True):
                    eqns_by_mpmd_idx[idx].append(
                        jcore.new_jaxpr_eqn(
                            [invar],
                            eqn.outvars,
                            all_reduce_p,
                            params={
                                "shardings_by_mpmd_idx": (
                                    mpmd_idx,
                                    eqn.params["out_shardings"].shardings[0],
                                )
                            },
                            effects=jcore.no_effects,
                        )
                    )
                continue
            eqns_by_mpmd_idx[mpmd_idx].append(eqn)
        elif eqn.primitive is dax_pscan_p:
            outvars, loop_eqns_by_mpmd_idx = mpmdify_loop(eqn)
            sub.update(zip(eqn.outvars, outvars))
            for mpmd_idx, eqns in enumerate(loop_eqns_by_mpmd_idx):
                eqns_by_mpmd_idx[mpmd_idx].extend(eqns)
        else:
            raise AssertionError(f"Unkown equation primitive {eqn.primitive}")

    outvar_set = set(jaxpr.outvars)
    jaxprs = list[jcore.ClosedJaxpr]()

    for eqns in eqns_by_mpmd_idx:
        jaxpr = jaxpr_from_eqns(eqns, outvar_set)
        check_jaxpr(jaxpr)
        jaxprs.append(jaxpr)

    return jaxprs


def lower_tasked_jaxpr(
    global_jaxpr: jcore.ClosedJaxpr,
    jaxprs: list[jcore.Jaxpr],
    donated_invars: Sequence[bool],
    shardings_map,
    mpmd_mesh: MpmdMesh | RemoteMpmdMesh,
    compiler_options=None,
    use_pgle: bool = False,
):
    var_ids = defaultdict[jcore.Var, ScalarUid](fresh_scalar_uid)
    in_var_ids = [var_ids[invar] for invar in global_jaxpr.jaxpr.invars]
    out_var_ids = [var_ids[outvar] for outvar in global_jaxpr.jaxpr.outvars]

    donated_invars = dict(zip(global_jaxpr.jaxpr.invars, donated_invars))
    lowering_mesh = mpmd_mesh.lowering_mesh()
    computations = [
        dict[tuple[str, jcore.Jaxpr], SerializeableMeshComputation]()
        for _ in range(mpmd_mesh.mpmd_dim)
    ]

    instructions_by_worker = [[] for _ in range(mpmd_mesh.mpmd_dim)]

    for mpmd_idx, jaxpr in enumerate(jaxprs):
        if isinstance(mpmd_mesh, MpmdMesh) and mpmd_idx != mpmd_mesh.my_mpmd_axis_index:
            continue

        # jaxpr = improve_overlap(jaxpr, shardings_map)
        jaxpr = infer_donation(
            jaxpr, donated_invars=[donated_invars[v] for v in jaxpr.invars]
        )
        for eqn in jaxpr.eqns:
            if eqn.primitive is task_p:
                task_name = eqn.params["task_name"]
                in_shardings = eqn.params["in_shardings"].shardings
                out_shardings = eqn.params["out_shardings"].shardings
                # FIXME: this cache key is too weak, also add in_shardings, out_shardings etc.
                if (
                    mesh_computation := computations[mpmd_idx].get(
                        (task_name, eqn.params["call_jaxpr"])
                    )
                ) is None:
                    mesh_computation = pjit_to_serializeable_mesh_computation(
                        closed_jaxpr=jcore.ClosedJaxpr(eqn.params["call_jaxpr"], ()),
                        in_axis_resources=in_shardings,
                        out_axis_resources=out_shardings,
                        name=task_name,
                        mesh=lowering_mesh,
                        donate_invars=eqn.params["donate_invars"],
                        compiler_options=compiler_options,
                        use_pgle=use_pgle,
                    )
                    computations[mpmd_idx][(task_name, eqn.params["call_jaxpr"])] = (
                        mesh_computation
                    )
                instructions_by_worker[mpmd_idx].append(
                    RunOp(
                        exec_uid=task_name,
                        _in_uids=[var_ids[invar] for invar in eqn.invars],
                        _out_uids=[var_ids[outvar] for outvar in eqn.outvars],
                    )
                )
            elif eqn.primitive is send_p:
                comm_descs = [
                    CommDesc(
                        uid=var_ids[invar],
                        aval=invar.aval,
                        sharding=SerializeableSharding(sharding),
                        from_dev_ids=[
                            d.id
                            for d in mpmd_mesh.remote_mesh_at(
                                mpmd_idx
                            )._flat_devices_tuple
                        ],
                        to_dev_ids=[
                            d.id
                            for d in mpmd_mesh.remote_mesh_at(
                                tgt_mpmd_idx
                            )._flat_devices_tuple
                        ],
                    )
                    for invar, (tgt_mpmd_idx, sharding) in zip(
                        eqn.invars, eqn.params["shardings"], strict=True
                    )
                ]
                instructions_by_worker[mpmd_idx].append(SendOp(comm_descs))
            elif eqn.primitive is recv_p:
                comm_descs = []
                for outvar, (from_mpmd_idx, sharding) in zip(
                    eqn.outvars, eqn.params["shardings"], strict=True
                ):
                    from_dev_ids = [
                        d.id
                        for d in mpmd_mesh.remote_mesh_at(
                            from_mpmd_idx
                        )._flat_devices_tuple
                    ]
                    to_dev_ids = [
                        d.id
                        for d in mpmd_mesh.remote_mesh_at(mpmd_idx)._flat_devices_tuple
                    ]
                    comm_descs.append(
                        CommDesc(
                            uid=var_ids[outvar],
                            aval=outvar.aval,
                            sharding=SerializeableSharding(sharding),
                            from_dev_ids=from_dev_ids,
                            to_dev_ids=to_dev_ids,
                        )
                    )
                instructions_by_worker[mpmd_idx].append(RecvOp(comm_descs))
            elif eqn.primitive is all_reduce_p:
                assert len(eqn.invars) == 1
                mpmd_idxs, sharding = eqn.params["shardings_by_mpmd_idx"]
                keys = make_ar_device_keys(
                    mpmd_mesh.as_mpmd_mesh, list(UniqueSortedSequence.create(mpmd_idxs))
                )
                desc = AllReduceDesc(
                    keys, var_ids[eqn.invars[0]], var_ids[eqn.outvars[0]]
                )
                instructions_by_worker[mpmd_idx].append(AllReduceOp([desc]))
            else:
                raise NotImplementedError(f"Unknown primitive {eqn.primitive}")

    return (
        instructions_by_worker,
        computations,
        (in_var_ids, out_var_ids),
    )


def prepare_pipelined(
    fun: Callable,
    args,
    remote_mesh: RemoteMpmdMesh,
    in_axis_resources=None,
    out_axis_resources=None,
    static_argnums=(),
    donate_argnums=(),
    compiler_options=None,
    use_pgle=False,
):
    lowering_mesh = remote_mesh.lowering_mesh()

    with log_elapsed_time("jaxpr/tracing"), lowering_mesh:
        pjit_params, _ = jit_infer_params(
            fun,
            tuple(args),
            in_axis_resources=in_axis_resources,
            out_axis_resources=out_axis_resources,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
        )

    params = pjit_params.params
    consts = pjit_params.consts
    jaxpr_with_consts = pe.convert_constvars_jaxpr(params["jaxpr"].jaxpr)
    jaxpr, used_invars = pe.dce_jaxpr(
        jaxpr_with_consts,
        used_outputs=[True] * len(jaxpr_with_consts.outvars),
    )

    jaxpr = jaxpr.replace(
        invars=jaxpr_with_consts.invars, debug_info=jaxpr_with_consts.debug_info
    )

    # If a variable is passthrough - an input variable is returned unmodified as output
    # 1. We store its input position
    # 2. We can return the variable by properly identifying it
    invars_idx = {invar: idx for idx, invar in enumerate(jaxpr.invars)}
    passthrough_outvars = [invars_idx.get(outvar) for outvar in jaxpr.outvars]

    closed_jaxpr = pe.close_jaxpr(jaxpr)
    del jaxpr_with_consts

    is_flat_arg_donated: tuple[bool] = params["donated_invars"]
    donated_invars = ((False,) * len(consts)) + is_flat_arg_donated

    with stable_names_ctx():
        closed_jaxpr, mpmd_def = wrap_into_tasks(
            closed_jaxpr, used_invars, remote_mesh.mpmd_dim
        )

    replicated_sharding = jax.sharding.NamedSharding(
        lowering_mesh, jax.sharding.PartitionSpec()
    )
    flat_in_shardings = ((replicated_sharding,) * len(consts)) + params["in_shardings"]
    in_layouts = ((None,) * len(consts)) + params["in_layouts"]
    flat_out_shardings = params["out_shardings"]

    with log_elapsed_time("xla_compilation/driver"):
        # Trigger first compilation on the driver for inferring intermediate shardings
        # NOTE: do not enable PGLE for this compilation otherwise it will lead to hangs
        mesh_executable = pjit_to_serializeable_mesh_computation(
            closed_jaxpr,
            in_axis_resources=flat_in_shardings,
            out_axis_resources=flat_out_shardings,
            in_layouts=in_layouts,
            out_layouts=params["out_layouts"],
            name=fun.__name__,
            mesh=lowering_mesh,
            compiler_options=compiler_options,
        ).to_mesh_executable(lowering_mesh)
        del mesh_executable

    closed_jaxpr = strip_inspect_sharding_eqns(closed_jaxpr)
    with stable_names_ctx(mpmd_def.get):
        # NOTE: mutates sharding stored inside `closed_jaxpr`
        reconcile_sharding_for_replicated_vars(
            closed_jaxpr, flat_in_shardings, flat_out_shardings
        )

        # log_activation_shardings(closed_jaxpr)

        jaxprs = mpmdify(closed_jaxpr, remote_mesh.mpmd_dim)
        for mpmd_idx, jaxpr in enumerate(jaxprs):
            for invar in jaxpr.invars:
                assert mpmd_idx in mpmd_def[invar]
            for outvar in jaxpr.outvars:
                assert mpmd_idx in mpmd_def[outvar]

        instructions, lowerings_by_worker, (in_uids, out_uids) = lower_tasked_jaxpr(
            closed_jaxpr,
            jaxprs,
            donated_invars=donated_invars,
            shardings_map=dict(
                zip(closed_jaxpr.jaxpr.invars, flat_in_shardings, strict=True)
            ),
            mpmd_mesh=remote_mesh,
            compiler_options=compiler_options,
            use_pgle=use_pgle,
        )

    must_live = set(out_uids)
    must_live.update(
        uid for uid, donated in ju.safe_zip(in_uids, donated_invars) if not donated
    )

    mpmd_instructions = [add_delete_ops(instrs, must_live) for instrs in instructions]

    futs = []
    for mpmd_idx, serializeable_computations in enumerate(lowerings_by_worker):
        for (uid, _), comp in serializeable_computations.items():
            futs.append(remote_mesh.put_mesh_computation(mpmd_idx, uid, comp))

    remote_mesh.blocking_tree(futs)

    return DistributedFunction(
        consts=consts,
        # NOTE: in_tree is (args, kwargs) and we do not support kwargs
        # so we drop them from the structure here
        in_tree=pjit_params.in_tree.children()[0],
        out_tree=pjit_params.out_tree,
        in_avals=closed_jaxpr.in_avals,
        out_avals=closed_jaxpr.out_avals,
        in_shardings=flat_in_shardings,
        out_shardings=flat_out_shardings,
        in_mpmd_idxs=[mpmd_def[invar] for invar in closed_jaxpr.jaxpr.invars],
        out_mpmd_idxs=[mpmd_def[outvar] for outvar in closed_jaxpr.jaxpr.outvars],
        in_uids=in_uids,
        out_uids=out_uids,
        donated_invars=donated_invars,
        used_invars=used_invars,
        worker_mesh=remote_mesh,
        instructions_by_worker=mpmd_instructions,
        passthrough_outvars=passthrough_outvars,
    )


def pipelined(
    fun: Callable,
    mpmd_mesh,
    in_axis_resources=None,
    out_axis_resources=None,
    static_argnums=(),
    donate_argnums=(),
    compiler_options: Mapping | None = None,
    use_pgle=False,
) -> CompileThunk:
    def compile_fn(*args, **kwargs):
        assert kwargs == {}, "Unsupported kwargs"
        return prepare_pipelined(
            fun,
            args,
            mpmd_mesh,
            in_axis_resources=in_axis_resources,
            out_axis_resources=out_axis_resources,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
            compiler_options=compiler_options,
            use_pgle=use_pgle,
        )

    return CompileThunk(compile_fn)
