# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import abc
import dataclasses
import itertools as it
import logging
import operator
import weakref
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property, partial
from pathlib import Path
from typing import (
    Any,
    Concatenate,
    Hashable,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

import jax
import jax.extend.source_info_util as jsiu
import jax.interpreters.partial_eval as pe
from jax.interpreters.ad import add_jaxvals_p as add_any_p

from jaxpp import array_ops, env_vars
from jaxpp import jax_compat as jc
from jaxpp.array import MpmdArray
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import (
    LocalSliceParams,
    LocalStackParams,
    PscanJaxpr,
    TaskEqn,
    TransferParams,
    add_multi_p,
    all_gather_p,
    all_reduce_p,
    dax_pscan_p,
    delete_p,
    gather_multi_p,
    local_slice_p,
    local_stack_p,
    pipeline_yield_p,
    recv_done_p,
    reuse_fence_p,
    slice_p,
    stack_p,
    task_p,
    transfer_done_p,
    transfer_p,
    transfer_start_p,
    zeros_p,
)
from jaxpp.jaxpr_utils import (
    DefSite,
    check_jaxpr,
    defs_and_free_uses,
    defs_and_uses,
    eqns_free_vars,
    jaxpr_from_eqns,
    nonlit,
    partition_eqns,
    schedule_dependencies,
    substitute,
)
from jaxpp.jaxpr_utils import gensym as mk_gensym
from jaxpp.licm import (
    hashable_params,
    hoist_and_cse_pscan_invariant_equations,
    inline_eqns,
    outvar_normalization,
)
from jaxpp.mesh import (
    MpmdMesh,
    _require_mpmd_indices,
    _require_single_mpmd_index,
    _resolve_placement,
)
from jaxpp.pipelining import yield_scope
from jaxpp.schedules import FusedTask, Task, mk_task_name, preprocess_schedule_tasks
from jaxpp.sharding_inference import (
    infer_shardings,
    infer_shardings_explicit,
    reconcile_shardings,
)
from jaxpp.types import MpmdIdx, MpmdSharding, TaskType, fresh_scalar_uid
from jaxpp.utils import (
    groupby,
    hbytes,
    log_elapsed_time,
    update_named_sharding,
    updated_named_sharding_mesh,
)

logger = logging.getLogger(__name__)

AnyJaxpr = TypeVar("AnyJaxpr", jcore.ClosedJaxpr, jcore.Jaxpr)
Res = TypeVar("Res")
P = ParamSpec("P")


def unwrap_closed(
    fun: Callable[Concatenate[jcore.Jaxpr, P], jcore.Jaxpr],
) -> Callable[Concatenate[AnyJaxpr, P], AnyJaxpr]:
    def res(closed_jaxpr: AnyJaxpr, *args: P.args, **kwargs: P.kwargs):
        if isinstance(closed_jaxpr, jcore.ClosedJaxpr):
            assert len(closed_jaxpr.consts) == 0
            return closed_jaxpr.map_jaxpr(lambda jaxpr: fun(jaxpr, *args, **kwargs))
        return fun(closed_jaxpr, *args, **kwargs)

    return res


MaybeEqn = int | None


@partial(jc.weakref_lru_cache, maxsize=16)
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
        return jc.get_aval(self.val)


class AllReduceRewriteTrace(jcore.Trace):
    def __init__(self, parent_trace: jc.DynamicJaxprTrace):
        super().__init__()
        self.parent_trace = parent_trace

    def new_arg(self, aval, mpmd_defs):
        val = self.parent_trace.new_arg(aval, jsiu.current())
        return AllReduceRewriteTracer(self, val, mpmd_defs)

    def call_parent(self, primitive, tracers, params, *, placement=None):
        parent_tracers = [
            t.val if isinstance(t, AllReduceRewriteTracer) else t for t in tracers
        ]
        if primitive is add_multi_p and "mpmd_idxs" not in params:
            multiple_results = False
            with jcore.set_current_trace(self.parent_trace):
                results = sum(parent_tracers)
        elif primitive is gather_multi_p and "mpmd_idxs" not in params:
            multiple_results = False
            axis = params.get("axis", 0)
            with jcore.set_current_trace(self.parent_trace):
                expanded = [jax.numpy.expand_dims(t, axis=axis) for t in parent_tracers]
                results = jax.numpy.concatenate(expanded, axis=axis)
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
            lhs, rhs = tracers
            return self.call_parent(
                add_multi_p,
                tracers,
                {"mpmd_idxs": (min(lhs.placement), min(rhs.placement))},
                # FIXME below: assumes lhs/rhs are tracers, i.e. doesn't handle literals
                placement={min(lhs.placement), min(rhs.placement)},
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
        elif primitive is gather_multi_p:
            groups_with_idx = groupby(
                (min(t.placement), (i, t))
                for i, t in enumerate(tracers)
                if isinstance(t, AllReduceRewriteTracer) and t.placement is not None
            )

            placement = set(groups_with_idx.keys())
            axis = params.get("axis", 0)

            natural_order = []
            for mpmd_idx in sorted(groups_with_idx.keys()):
                for orig_idx, _ in groups_with_idx[mpmd_idx]:
                    natural_order.append(orig_idx)

            restore_order_perm = [0] * len(natural_order)
            for pos, orig_idx in enumerate(natural_order):
                restore_order_perm[orig_idx] = pos

            # Locally concatenate arrays within each group
            results = []
            with jcore.set_current_trace(self.parent_trace):
                for mpmd_idx in sorted(groups_with_idx.keys()):
                    group_tracers = [t for _, t in groups_with_idx[mpmd_idx]]
                    parent_group = [
                        t.val if isinstance(t, AllReduceRewriteTracer) else t
                        for t in group_tracers
                    ]
                    results.append(
                        jax.numpy.concatenate(parent_group, axis=axis)
                        if len(parent_group) > 1
                        else parent_group[0]
                    )

            return self.call_parent(
                gather_multi_p,
                results,
                {
                    **params,
                    "mpmd_idxs": tuple(sorted(groups_with_idx.keys())),
                    "restore_order_perm": tuple(restore_order_perm),
                },
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
    mpmd_trace = AllReduceRewriteTrace(jc.DynamicJaxprTrace(jaxpr.debug_info))
    in_tracers = [
        mpmd_trace.new_arg(invar.aval, mpmd_defs)
        for invar, mpmd_defs in zip(jaxpr.invars, invar_mpmd_defs, strict=True)
    ]

    with jcore.set_current_trace(mpmd_trace):
        res = jcore.eval_jaxpr(jaxpr, (), *in_tracers, propagate_source_info=True)

    # TODO: handle literals in `res`
    # Ignore any trailing return values from JAX internals.
    jaxpr, consts, *_ = mpmd_trace.parent_trace.to_jaxpr(
        [v.val for v in res], jaxpr.debug_info, jsiu.current()
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
    # Find both add_multi_p and gather_multi_p equations as cross-MPMD barriers
    cross_mpmd_primitives = {add_multi_p, gather_multi_p}
    cross_mpmd_eqn_idxs = [
        idx
        for idx, e in reversed(list(enumerate(jaxpr.eqns)))
        if e.primitive in cross_mpmd_primitives
    ]
    if len(cross_mpmd_eqn_idxs) == 0 or cross_mpmd_eqn_idxs[-1] != 0:
        cross_mpmd_eqn_idxs.append(0)

    var_placement = dict[jcore.Var, set[int]](
        (outvar, placement) for outvar, placement in zip(jaxpr.outvars, out_placements)
    )

    eqns_in_mpmd_idx = defaultdict(lambda: 0)

    mpmd_idxs = range(mpmd_dim)
    last = len(jaxpr.eqns)
    rev_new_eqns = []
    for eqn_idx in cross_mpmd_eqn_idxs:
        cross_mpmd_eqn, eqns = None, jaxpr.eqns[eqn_idx:last]
        last = eqn_idx
        if eqns[0].primitive in cross_mpmd_primitives:
            [cross_mpmd_eqn, *eqns] = eqns

        tmp = jaxpr_from_eqns(eqns, set(var_placement.keys()))

        jaxprs, in_uses = make_replicated_jaxpr(
            tmp, tuple(var_placement[outvar] for outvar in tmp.outvars), mpmd_idxs
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
        for invar, uses in jc.safe_zip(tmp.invars, in_uses):
            if uses is None:
                continue
            if (p := var_placement.get(invar)) is not None:
                merged_uses = uses | p
            else:
                merged_uses = uses
            var_placement[invar] = merged_uses

        if cross_mpmd_eqn is None:
            continue

        rev_new_eqns.append(
            cross_mpmd_eqn.replace(
                params=cross_mpmd_eqn.params
                | {
                    "in_shardings": (None,) * len(cross_mpmd_eqn.invars),
                    "out_shardings": (None,) * len(cross_mpmd_eqn.outvars),
                }
            )
        )
        for invar, mpmd_idx in jc.safe_zip(
            cross_mpmd_eqn.invars, cross_mpmd_eqn.params["mpmd_idxs"]
        ):
            uses = {mpmd_idx}
            if (p := var_placement.get(invar)) is not None:
                uses = uses | p
            var_placement[invar] = uses

    sub = dict(jc.safe_zip(jaxpr.invars, in_jaxpr.invars))
    sub.update(jc.safe_zip(jaxpr.outvars, in_jaxpr.outvars))
    new_eqns = substitute(reversed(rev_new_eqns), sub)
    res_jaxpr = jaxpr.replace(
        invars=in_jaxpr.invars,
        outvars=in_jaxpr.outvars,
        eqns=new_eqns,
        effects=jcore.join_effects(*(_.effects for _ in new_eqns)),
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
    gensym = mk_gensym()
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


def paranoid_assert(cond: bool, msg: str | None = None):
    if not cond:
        raise AssertionError(msg)


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
    gensym = mk_gensym("_licm")
    defs, refs = ivar_defs_and_refs(loop_body)
    invar_indices = {invar: idx for idx, invar in enumerate(loop_body.invars)}

    replicated_loop_body_invars = defaultdict[int, list[jcore.Var]](list)
    replicated_loop_body_outvars = dict[int, tuple[jcore.JaxprEqn, list[jcore.Var]]]()
    replace_eqns = dict[int, list[jcore.JaxprEqn]]()

    for outvar_idx, outvar in enumerate(loop_body.outvars):
        if not isinstance(outvar, jcore.Var):
            # outvar is jcore.Literal
            continue
        if (add_eqn_idx := defs[outvar]) is None:
            # outvar is not defined in the loop body
            # FIXME: use `pe._jaxpr_forwarding` in an early pass to remove
            # these variables as done in https://github.com/jax-ml/jax/blob/a04b5ecfcdd6a15cf412844d49114c609ae72f50/jax/_src/lax/control_flow/conditionals.py#L154-L158
            raise ValueError("Passthrough output variable found")

        add_eqn = loop_body.eqns[add_eqn_idx]

        if add_eqn.primitive is not jax.lax.add_p:
            continue

        [linvar, rinvar] = add_eqn.invars
        if not isinstance(linvar, jcore.Var) or not isinstance(rinvar, jcore.Var):
            continue

        loop_body_invar, grad = (
            (linvar, rinvar) if defs[linvar] is None else (rinvar, linvar)
        )

        if (
            invar_idx := invar_indices.get(loop_body_invar)
        ) is None or invar_idx < body_nconsts:
            # This is not a loop variable or is a loop constant
            continue

        if (add_any_eqn_idx := defs.get(grad)) is None:
            # Gradient is not produced in the loop body
            # FIXME: maybe raise an error?
            continue

        if (
            add_any_eqn := cast(jcore.JaxprEqn, loop_body.eqns[add_any_eqn_idx])
        ) and add_any_eqn.primitive is not add_any_p:
            continue

        use_eqn_idxs = refs[add_any_eqn.outvars[0]]
        paranoid_assert(
            len(use_eqn_idxs) >= 1,
            "refs is inconsistent with defs. "
            "While walking up the def chain, `add_any_eqn` is not present in refs",
        )

        if len(use_eqn_idxs) > 1:
            # TODO: maybe handle the case when multiple uses of the gradients
            # are present. This should be impossible/uncommon
            # (higher-order gradients (?)).
            continue

        paranoid_assert(use_eqn_idxs[0] == add_eqn_idx)
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
                    invars=[in_replica, cross_worker_invar], outvars=[out_replica]
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


def _bind_task_eqn_to_mesh(
    eqn: jcore.JaxprEqn, new_mesh: jax.sharding.Mesh
) -> jcore.JaxprEqn:
    call_jaxpr = replace_captured_meshes(eqn.params["call_jaxpr"], new_mesh)
    return eqn.replace(
        params=eqn.params
        | {
            "call_jaxpr": call_jaxpr,
            "in_shardings": updated_named_sharding_mesh(
                eqn.params["in_shardings"], new_mesh
            ),
            "out_shardings": updated_named_sharding_mesh(
                eqn.params["out_shardings"], new_mesh
            ),
        },
        effects=call_jaxpr.effects,
    )


@jc.weakref_lru_cache
def replace_captured_meshes(cjaxpr: AnyJaxpr, new_mesh: jax.sharding.Mesh) -> AnyJaxpr:
    jaxpr = cjaxpr.jaxpr if isinstance(cjaxpr, jcore.ClosedJaxpr) else cjaxpr

    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive is task_p:
            new_eqns.append(_bind_task_eqn_to_mesh(eqn, new_mesh))
            continue

        param_update = {}
        if eqn.primitive is jax.lax.sharding_constraint_p:
            param_update = {
                "sharding": updated_named_sharding_mesh(
                    eqn.params["sharding"], new_mesh
                )
            }
        elif eqn.primitive is jc.jit_p:
            param_update = {
                "in_shardings": updated_named_sharding_mesh(
                    eqn.params["in_shardings"], new_mesh
                ),
                "out_shardings": updated_named_sharding_mesh(
                    eqn.params["out_shardings"], new_mesh
                ),
            }
        elif eqn.primitive is jc.shard_map_p:
            mesh = eqn.params["mesh"]
            if isinstance(mesh, jax.sharding.AbstractMesh):
                continue
            param_update = {"mesh": new_mesh}
        elif eqn.primitive is jax.lax.device_put_p:
            param_update = {
                "devices": updated_named_sharding_mesh(eqn.params["devices"], new_mesh)
            }

        for k, v in eqn.params.items():
            if isinstance(v, (jcore.ClosedJaxpr, jcore.Jaxpr)):
                param_update[k] = replace_captured_meshes(v, new_mesh)
        new_eqns.append(eqn.replace(params=eqn.params | param_update))

    res_jaxpr = jaxpr.replace(eqns=new_eqns)
    if isinstance(cjaxpr, jcore.Jaxpr):
        return res_jaxpr
    return cjaxpr.replace(jaxpr=res_jaxpr)


def new_primitive_eqn(
    prim: jcore.Primitive,
    invars: Sequence[jcore.Atom],
    outvars: Sequence[jcore.Var] | None = None,
    **params,
) -> jcore.JaxprEqn:
    out_avals, effects = prim.abstract_eval(*(_.aval for _ in invars), **params)
    if not prim.multiple_results:
        out_avals = (out_avals,)
    if outvars is None:
        outvars = [jcore.Var(aval) for aval in out_avals]
    else:
        assert tuple(outvar.aval for outvar in outvars) == tuple(out_avals)
    return jcore.new_jaxpr_eqn(
        invars=invars, outvars=outvars, primitive=prim, params=params, effects=effects
    )


def _task_eqn(
    invars,
    outvars,
    task_jaxpr: jcore.ClosedJaxpr,
    mpmd_idx: int | jax.sharding.Mesh,
    in_shardings,
    out_shardings,
    donate_invars,
    task_name,
    task_info: tuple[int, TaskType] | None = None,
    latency: float | None = None,
):
    assert len(invars) == len(task_jaxpr.in_avals)
    assert len(donate_invars) == len(invars)
    return new_primitive_eqn(
        task_p,
        invars,
        outvars,
        call_jaxpr=task_jaxpr,
        task_name=task_name,
        task_info=task_info,
        mpmd_idx=mpmd_idx,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        donate_invars=donate_invars,
        latency=latency,
    )


def make_task_eqn(
    invars: Sequence[jcore.Var],
    outvars: Sequence[jcore.Var],
    eqns: list[jcore.JaxprEqn],
    mpmd_idx: int,
    task_name: str,
    task_info: tuple[int, TaskType] | None = None,
    latency: float | None = None,
) -> jcore.JaxprEqn:
    if latency is None:
        if task_info is not None:
            latency = task_info[1].default_latency
        else:
            latency = 1  # FIXME(task_latency)

    in_task_shardings = (None,) * len(invars)
    out_task_shardings = (None,) * len(outvars)
    donate_invars = (False,) * len(invars)

    effects = jcore.join_effects(*(eqn.effects for eqn in eqns))
    task_jaxpr = jcore.Jaxpr(
        constvars=(), invars=invars, outvars=outvars, eqns=eqns, effects=effects
    )
    check_jaxpr(task_jaxpr)

    return _task_eqn(
        invars=invars,
        outvars=outvars,
        task_jaxpr=jcore.ClosedJaxpr(task_jaxpr, ()),
        mpmd_idx=mpmd_idx,
        in_shardings=in_task_shardings,
        out_shardings=out_task_shardings,
        donate_invars=donate_invars,
        task_name=task_name,
        task_info=task_info,
        latency=latency,
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
    var_ref_cluster_idx: defaultdict[jcore.Var, frozenset[int]]
    last_cluster_idx_for_mpmd_idx: dict[MpmdIdx, int]


empty_frozenset = frozenset()


class LitT: ...


Lit = LitT()


def get_cluster_information(clusters: list[Cluster]) -> ClusterInfo:
    var_def_cluster_idx = dict[jcore.Var, int]()
    var_ref_cluster_idx = dict[jcore.Var, frozenset[int]]()
    last_cluster_idx_for_mpmd_idx = dict[MpmdIdx, int]()

    for cluster_idx, (mpmd_idx, _, eqns, _) in enumerate(clusters):
        last_cluster_idx_for_mpmd_idx[mpmd_idx] = cluster_idx

        refs, defs = eqns_free_vars(eqns)
        for v in refs:
            var_ref_cluster_idx[v] = var_ref_cluster_idx.get(v, empty_frozenset) | {
                cluster_idx
            }

        var_def_cluster_idx.update(zip(defs, it.repeat(cluster_idx)))

    return ClusterInfo(
        var_def_cluster_idx, var_ref_cluster_idx, last_cluster_idx_for_mpmd_idx
    )


def first_pipeline_yield_eqn_idx(eqns: Iterable[jcore.JaxprEqn]) -> int | None:
    for idx, eqn in enumerate(eqns):
        if eqn.primitive is pipeline_yield_p:
            return idx


def infer_cluster_idx_for_eqns(
    clusters: list[Cluster], eqns: list[jcore.JaxprEqn]
) -> list[int | None]:
    cluster_info = get_cluster_information(clusters)
    var_def_cluster_idx = cluster_info.var_def_cluster_idx
    var_ref_cluster_idx = cluster_info.var_ref_cluster_idx

    idefs = dict[jcore.Var, int]()
    for eqn_idx, eqn in enumerate(eqns):
        idefs.update(zip(eqn.outvars, it.repeat(eqn_idx)))

    eqn_cluster_idx: list[int | None] = [None] * len(eqns)

    def update_def_use_chain(eqn_idx: int, cluster_idx: int):
        def update_one(eqn_idx: int):
            eqn_cluster_idx[eqn_idx] = cluster_idx
            eqn = eqns[eqn_idx]
            for invar in nonlit(eqn.invars):
                var_ref_cluster_idx[invar] = var_ref_cluster_idx.get(
                    invar, empty_frozenset
                ) | {cluster_idx}
            for outvar in eqn.outvars:
                var_def_cluster_idx[outvar] = cluster_idx

        worklist = deque(nonlit(eqns[eqn_idx].invars))
        while len(worklist) > 0:
            v = worklist.popleft()
            if (dep_eqn_idx := idefs.get(v)) is not None:
                if (p := eqn_cluster_idx[dep_eqn_idx]) is None:
                    update_one(dep_eqn_idx)
                    worklist.extend(nonlit(eqns[dep_eqn_idx].invars))
                else:
                    # NOTE: this is an invariant of the algorithm so this assertion
                    #  is never raised in practice.
                    #  However we leave it here in case of changes
                    assert p <= cluster_idx, f"{p=} {cluster_idx=}"
        update_one(eqn_idx)

    # First propagate only based on loop definitions (not uses)
    for eqn_idx, eqn in enumerate(eqns):
        invar_def_cluster = list[LitT | int | None]()
        for invar in eqn.invars:
            if isinstance(invar, jcore.Literal):
                invar_def_cluster.append(Lit)
            else:
                invar_def_cluster.append(var_def_cluster_idx.get(invar))

        unique_invar_def_clusters = {_ for _ in invar_def_cluster if isinstance(_, int)}
        if len(unique_invar_def_clusters) > 0:
            if len(unique_invar_def_clusters) == 1:
                update_def_use_chain(eqn_idx, next(iter(unique_invar_def_clusters)))
            else:
                # NOTE: conflict resolution
                update_def_use_chain(eqn_idx, max(unique_invar_def_clusters))

    # Then propagate based on both defs and uses
    for eqn_idx, eqn in enumerate(eqns):
        if eqn_cluster_idx[eqn_idx] is not None:
            continue

        invar_ref_clusters = list[LitT | frozenset[int] | None]()
        earliest_invar_def_cluster = None
        for invar in eqn.invars:
            if isinstance(invar, jcore.Literal):
                invar_ref_clusters.append(Lit)
            else:
                if earliest_invar_def_cluster is None:
                    earliest_invar_def_cluster = var_def_cluster_idx.get(invar)
                else:
                    earliest_invar_def_cluster = max(
                        earliest_invar_def_cluster,
                        var_def_cluster_idx.get(invar, earliest_invar_def_cluster),
                    )

                invar_ref_clusters.append(var_ref_cluster_idx.get(invar))

        # The placement is the cluster that uses it the earliest
        known_uses = [_ for _ in invar_ref_clusters if isinstance(_, frozenset)]
        potential_placement = it.chain(*known_uses)
        if earliest_invar_def_cluster is not None:
            potential_placement = (
                _ for _ in potential_placement if _ >= earliest_invar_def_cluster
            )

        potential_placement = min(
            potential_placement, default=earliest_invar_def_cluster
        )
        if potential_placement is not None:
            update_def_use_chain(eqn_idx, potential_placement)

    return eqn_cluster_idx


def cluster_by_yield_eqns(
    eqns: list[jcore.JaxprEqn], get_mpmd_idx: Callable[[int], MpmdIdx]
) -> tuple[list[Cluster], list[jcore.JaxprEqn]]:
    pp_eqn_idx = first_pipeline_yield_eqn_idx(eqns)
    if pp_eqn_idx is None:
        # FIXME: is defaulting to MpmdIdx(0) ok?
        return [Cluster(MpmdIdx(0), TaskType.FWD, eqns, stage_id=0)], []

    stage_0, eqns = schedule_dependencies(eqns, pp_eqn_idx)
    curr_enter_eqn = stage_0.pop()
    stages: list[Cluster] = [
        Cluster(
            get_mpmd_idx(curr_enter_eqn.params["stage_id"]),
            TaskType.FWD,
            stage_0,
            stage_id=curr_enter_eqn.params["stage_id"],
        )
    ]

    passed_backward = False
    while (pp_eqn_idx := first_pipeline_yield_eqn_idx(eqns)) is not None:
        stage_i, eqns = schedule_dependencies(eqns, pp_eqn_idx)
        next_enter_eqn = stage_i.pop()
        if not passed_backward and next_enter_eqn.params["task_type"] is TaskType.BWD:
            stages[-1].eqns.extend([curr_enter_eqn] + stage_i)
            curr_enter_eqn = next_enter_eqn
            passed_backward = True
            continue

        assert not passed_backward or curr_enter_eqn.params["task_type"] is TaskType.BWD
        if curr_enter_eqn.params["task_type"] is TaskType.FWD:
            stage_id = next_enter_eqn.params["stage_id"]
        else:
            stage_id = curr_enter_eqn.params["stage_id"]
        mpmd_idx = get_mpmd_idx(stage_id)
        stages.append(
            Cluster(
                mpmd_idx,
                curr_enter_eqn.params["task_type"],
                [curr_enter_eqn] + stage_i,
                stage_id=stage_id,
            )
        )
        curr_enter_eqn = next_enter_eqn

    if not passed_backward:
        stages[-1].eqns.append(curr_enter_eqn)
    else:
        stage_id = curr_enter_eqn.params["stage_id"]
        stages.append(
            Cluster(
                get_mpmd_idx(stage_id),
                curr_enter_eqn.params["task_type"],
                [curr_enter_eqn],
                stage_id=stage_id,
            )
        )
    return stages, eqns


# TODO: maybe cluster_eqns shouldn't depend on `get_mpmd_idx`
def cluster_eqns(
    eqns: list[jcore.JaxprEqn], get_mpmd_idx: Callable[[int], MpmdIdx]
) -> tuple[list[Cluster], list[jcore.JaxprEqn]]:
    clusters, rest = cluster_by_yield_eqns(eqns, get_mpmd_idx)
    eqns_cluster_idxs = infer_cluster_idx_for_eqns(clusters, rest)
    unclustered_eqns = list[jcore.JaxprEqn]()
    for cluster_idx, eqn in zip(eqns_cluster_idxs, rest, strict=True):
        if cluster_idx is not None:
            clusters[cluster_idx].eqns.append(eqn)
        else:
            unclustered_eqns.append(eqn)
    return clusters, unclustered_eqns


def _replace_pipeline_yields_with_copies(
    eqns: list[jcore.JaxprEqn],
) -> list[jcore.JaxprEqn]:
    """Replace pipeline_yield equations with copy_p equations.

    After clustering, pipeline_yield metadata (stage ids, names) is no longer
    needed -- it's captured in the Cluster/task. Replacing with copy_p removes
    stage-specific params so that structurally identical stages produce
    identical task jaxprs.
    """
    new_eqns = []
    for eqn in eqns:
        if eqn.primitive is pipeline_yield_p:
            assert len(eqn.invars) == len(eqn.outvars)
            for invar, outvar in zip(eqn.invars, eqn.outvars):
                new_eqns.append(new_primitive_eqn(jax.lax.copy_p, [invar], [outvar]))
        else:
            new_eqns.append(eqn)
    return new_eqns


def clusters_to_tasks(
    clusters: list[Cluster], outvars: Iterable[jcore.Var], is_partial_bwd: bool
) -> list[jcore.JaxprEqn]:
    outvars = set(outvars)
    undef = set[jcore.Var](outvars)
    rev_stage_eqns = []
    for mpmd_idx, ty, raw_stage_eqns, maybe_stage_id in reversed(clusters):
        stage_eqns = _replace_pipeline_yields_with_copies(raw_stage_eqns)
        assert maybe_stage_id is not None
        task_info = (maybe_stage_id, ty)
        if len(stage_eqns) == 0:
            logger.warning(f"Empty stage {task_info}")
        if is_partial_bwd and ty is TaskType.BWD:
            task_info = [
                (maybe_stage_id, TaskType.BWD_I),
                (maybe_stage_id, TaskType.BWD_W),
            ]
            dependencies, deferred, _ = partition_eqns(
                stage_eqns,
                undef - outvars,
                is_partial_bwd=is_partial_bwd,
                memory_scarce=True,
            )
            # TODO: revisit filter `len(task) > 0` below
            tasks = list(
                zip(
                    (task for task in [dependencies, deferred] if len(task) > 0),
                    task_info,
                    # dependencies is empty for stage 0 bwd.
                    # Create a single task with BWD_I as the type.
                    strict=False,
                )
            )
        else:
            tasks = [(stage_eqns, task_info)]

        for eqns, task_info in reversed(tasks):
            # TODO(task_name_task_info): remove unnecessary serialization of
            # task_info into task_name
            task_name = mk_task_name(task_info[0], task_info[1])
            free, defs = eqns_free_vars(eqns, ordered=True)
            task_eqn = make_task_eqn(
                list(free),
                [d for d in defs if d in undef],
                eqns,
                mpmd_idx,
                task_name=task_name,
                task_info=task_info,
            )
            rev_stage_eqns.append(task_eqn)
            undef.difference_update(defs)
            undef.update(free)

    return list(reversed(rev_stage_eqns))


def cluster_jaxpr(
    jaxpr: jcore.Jaxpr,
    target_num_stages: int,
    is_partial_bwd: bool,
    get_mpmd_idx: Callable[[int], MpmdIdx],
    is_loop: bool = True,
):
    # TODO: remove is_loop parameter and make the caller perform the checks
    clusters, unclustered_eqns = cluster_eqns(jaxpr.eqns, get_mpmd_idx)
    if (
        is_loop
        and len(unclustered_eqns) != 0
        and env_vars.jaxpp_conservative_loop_clustering.value
    ):
        new_eqns = clusters_to_tasks(
            clusters,
            set(nonlit(jaxpr.outvars))
            | set(defs_and_free_uses(unclustered_eqns)[1].keys()),
            is_partial_bwd,
        )
        error_jaxpr = jaxpr.replace(eqns=new_eqns + unclustered_eqns)
        _loop_msg = ""
        if is_loop:
            _loop_msg = "loop body "
        raise AssertionError(
            f"Failed on {_loop_msg}jaxpr \n{error_jaxpr.pretty_print(source_info=True)}"
        )
    else:
        clusters[-1].eqns.extend(unclustered_eqns)
    del unclustered_eqns

    new_eqns = clusters_to_tasks(clusters, nonlit(jaxpr.outvars), is_partial_bwd)
    clustered_jaxpr = jaxpr.replace(
        eqns=new_eqns, effects=jcore.join_effects(*(eqn.effects for eqn in new_eqns))
    )

    if is_loop:
        # TODO: use `Schedule.get_num_stages`?
        inferred_num_stages, rem = divmod(len(clusters), 2)
        if rem != 0:
            raise AssertionError(
                f"Expected even number of stages, {len(clusters)} found"
                f"\n{clustered_jaxpr.pretty_print(use_color=False)}"
            )
    else:
        inferred_num_stages = len(clusters)

    if is_loop and target_num_stages is not None:
        if inferred_num_stages != target_num_stages:
            raise AssertionError(
                f"Unexpected number of pipeline markers: found {inferred_num_stages} "
                f"expected {target_num_stages}.\n"
                f"Jaxpr: \n{jaxpr.pretty_print(use_color=False)}\n"
            )
    else:
        logger.info(f"Inferred {len(clusters)}")

    return clustered_jaxpr


def wrap_into_tasks_inside_loop(loop_eqn: jcore.JaxprEqn) -> jcore.JaxprEqn:
    jaxpr: jcore.Jaxpr = loop_eqn.params["jaxpr"].jaxpr
    # TODO: let bind literals
    assert len(jaxpr.outvars) == len(
        set(jaxpr.outvars)
    ), "Literal outvars (hash error) or duplicate outvars not supported"

    clustered_jaxpr = cluster_jaxpr(
        jaxpr,
        target_num_stages=loop_eqn.params["schedule"].num_stages,
        is_partial_bwd=loop_eqn.params["schedule"].is_partial_bwd,
        get_mpmd_idx=loop_eqn.params["schedule"].get_mpmd_idx,
    )

    # Infer where loop inputs are used (refs) and where loop outputs
    # are defined (defs)
    clustered_inferred_jaxpr, in_mpmd_refs, out_mpmd_defs = compute_loop_placement(
        clustered_jaxpr, loop_eqn.params["n_consts"]
    )

    new_jaxpr = clustered_inferred_jaxpr

    check_jaxpr(new_jaxpr)

    return loop_eqn.replace(
        params={
            **loop_eqn.params,
            "jaxpr": loop_eqn.params["jaxpr"].replace(jaxpr=new_jaxpr),
            "in_shardings": (None,) * len(new_jaxpr.invars),
            "out_shardings": (None,) * len(new_jaxpr.outvars),
            "in_mpmd_refs": in_mpmd_refs,
            "out_mpmd_defs": out_mpmd_defs,
        },
        effects=new_jaxpr.effects,
    )


JLayout = Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class InInfo:
    in_used: Sequence[bool]
    in_donated: Sequence[bool]
    in_tree: jax.tree_util.PyTreeDef
    out_tree: jax.tree_util.PyTreeDef
    in_avals: Sequence[jcore.AbstractValue]
    out_avals: Sequence[jcore.AbstractValue]
    in_shardings: Sequence[jax.sharding.NamedSharding]
    out_shardings: Sequence[jax.sharding.NamedSharding]
    in_layouts: Sequence[JLayout]
    out_layouts: Sequence[JLayout]
    in_mpmd_defs: Sequence[set[MpmdIdx]]
    out_mpmd_defs: Sequence[set[MpmdIdx]]


def _transfer_channels(
    eqn: jcore.JaxprEqn,
) -> tuple[tuple[jax.sharding.Mesh, jax.sharding.Mesh], ...]:
    return tuple(
        (src_sharding.mesh, tgt_sharding.mesh)
        for src_sharding, tgt_sharding in zip(
            eqn.params["src_shardings"], eqn.params["tgt_shardings"], strict=True
        )
    )


def _unique_transfer_channels(
    eqn: jcore.JaxprEqn,
) -> tuple[tuple[jax.sharding.Mesh, jax.sharding.Mesh], ...]:
    return tuple(dict.fromkeys(_transfer_channels(eqn)))


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


def compute_loop_placement(loop_jaxpr: PscanJaxpr, n_consts: int, is_loop: bool = True):
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

    if is_loop:
        for invar, outvar in jc.safe_zip(
            loop_jaxpr.invars[n_consts:], loop_jaxpr.outvars
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
    else:
        for invar in loop_jaxpr.invars:
            mpmd_def[invar] = mpmd_refs[invar]

    loop_invar_mpmd_refs = tuple(
        frozenset(mpmd_refs[invar]) for invar in loop_jaxpr.invars
    )
    loop_outvar_mpmd_def = tuple(
        frozenset(mpmd_def[outvar]) for outvar in loop_jaxpr.outvars
    )
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
            used_outputs=[
                isinstance(outvar, jcore.Var) and mpmd_idx in place
                for outvar, place in zip(jaxpr.outvars, outvar_mpmd_refs, strict=True)
            ],
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


def get_one_loop_eqn_idx(
    eqns_or_jaxpr: jcore.ClosedJaxpr | jcore.Jaxpr | Iterable[jcore.JaxprEqn],
) -> int:
    eqns = eqns_or_jaxpr
    if isinstance(eqns_or_jaxpr, (jcore.ClosedJaxpr, jcore.Jaxpr)):
        eqns = eqns_or_jaxpr.eqns

    loop_eqn_idxs = [idx for idx, e in enumerate(eqns) if e.primitive is dax_pscan_p]
    if len(loop_eqn_idxs) != 1:
        raise AssertionError(
            f"Expected 1 loop at the top level but {len(loop_eqn_idxs)} found."
        )
    return loop_eqn_idxs[0]


def mpmd_unzip_reverse(
    jaxpr: jcore.Jaxpr, out_refs: Sequence[set[MpmdIdx] | None], name: str
):
    outvar_placement = out_refs
    assert all(p is not None for p in outvar_placement)
    unique_ps = sorted({_ for s in out_refs if s is not None for _ in s})

    jaxprs, invar_placement = make_replicated_jaxpr(
        jaxpr, outvar_placement, map(MpmdIdx, unique_ps)
    )

    logger.info(
        f"{name} output size: {hbytes(outvar.aval for outvar in jaxpr.outvars)}"
    )
    replication_factor = [
        (i, len(j.eqns) / len(jaxpr.eqns)) for i, j in enumerate(jaxprs)
    ]
    logger.info(f"{name} replication {replication_factor=}")

    task_eqns = list[jcore.JaxprEqn]()
    uid = fresh_scalar_uid()
    for mpmd_idx, j in zip(unique_ps, jaxprs):
        task_eqns.append(
            make_task_eqn(
                invars=j.invars,
                outvars=j.outvars,
                eqns=j.eqns,
                mpmd_idx=mpmd_idx,
                task_name=f"{name}_{uid}_{mpmd_idx}",
            )
        )

    return (jaxpr.replace(eqns=task_eqns), invar_placement, outvar_placement)


def _compute_mpmd_def_refs(
    eqns: list[jcore.JaxprEqn],
) -> tuple[dict[jcore.Var, set[MpmdIdx]], dict[jcore.Var, set[MpmdIdx]]]:
    mpmd_refs = defaultdict[jcore.Var, set[MpmdIdx]](set)
    mpmd_def = defaultdict[jcore.Var, set[MpmdIdx]](set)
    for eqn in eqns:
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
            raise ValueError(f"Unexpected equation {eqn.primitive}")

    return mpmd_refs, mpmd_def


def wrap_into_tasks_after_loop(
    after_loop_jaxpr: jcore.Jaxpr,
    in_mpmd_defs: Sequence[set[MpmdIdx] | None],
    mpmd_dim: int,
) -> tuple[jcore.Jaxpr, list[set[MpmdIdx]], list[set[MpmdIdx]]]:
    """
    NOTE: for tasks before and after the loop, the same outvar (object reference)
    can be "defined" by multiple tasks.
    This deviates from "canonical" JAX/Jaxprs, or any ANF-style IR and one should
    take precautions when manipulating or especially using those objects
    to track metadata in a dictionary.
    """

    (
        coarsened_after_loop_jaxpr,
        after_loop_invar_mpmd_refs,
        after_loop_outvar_placement,
        eqns_in_mpmd_idx,
    ) = mpmd_unzip_forward(after_loop_jaxpr, in_mpmd_defs, mpmd_dim)

    for invar, after_loop_use_p, def_p in jc.safe_zip(
        cast(list[jcore.Var], after_loop_jaxpr.invars),
        after_loop_invar_mpmd_refs,
        in_mpmd_defs,
    ):
        # This assertion is always true in theory, we leave it here defensively
        #  for potential future changes
        assert after_loop_use_p is not None
        if def_p is not None and after_loop_use_p != def_p:
            raise NotImplementedError(
                "Loop output used in a MPMD index different from the defining one. "
                f"Defined at {def_p} and used at {after_loop_use_p}."
            )

    replication_factor = [
        (mpmd_idx, n_eqns / len(after_loop_jaxpr.eqns))
        for mpmd_idx, n_eqns in eqns_in_mpmd_idx.items()
    ]
    logger.info(f"After loop replication {replication_factor=}")

    return (
        coarsened_after_loop_jaxpr,
        after_loop_invar_mpmd_refs,
        after_loop_outvar_placement,
    )


@unwrap_closed
def loop_passes(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    if env_vars.jaxpp_enable_licm.value:
        loop_eqn_idxs = [
            idx for idx, e in enumerate(jaxpr.eqns) if e.primitive is dax_pscan_p
        ]
        if len(loop_eqn_idxs) == 0:
            return jaxpr

        logger.info("Running LICM")
        jaxpr = hoist_and_cse_pscan_invariant_equations(jaxpr, cross_remat=True)
    check_jaxpr(jaxpr)
    return jaxpr


def join_argument_refs(
    invars: list[jcore.Var], mpmd_refs: list[set[MpmdIdx] | None]
) -> dict[jcore.Var, set[MpmdIdx]]:
    loop_args_mpmd_refs_map = dict[jcore.Var, set[MpmdIdx]]()
    for invar, refs in zip(invars, mpmd_refs, strict=True):
        if refs is not None:
            loop_args_mpmd_refs_map[invar] = (
                loop_args_mpmd_refs_map.get(invar, set[MpmdIdx]()) | refs
            )
    return loop_args_mpmd_refs_map


@jc.weakref_lru_cache
def _wrap_into_tasks(
    cjaxpr: jcore.ClosedJaxpr, used_invars: Sequence[bool], mpmd_dim: int
) -> tuple[jcore.ClosedJaxpr, tuple[set[MpmdIdx]], tuple[set[MpmdIdx]]]:
    """
    After this pass, all the equations in the returned jaxpr are either
    (1) `task` equations, or (2) a `dax_pscan` equation containing `task` equations.
    """
    jaxpr = cjaxpr.jaxpr
    [*before_loop_eqns, loop_eqn], after_loop_eqns = schedule_dependencies(
        jaxpr.eqns, get_one_loop_eqn_idx(jaxpr.eqns)
    )
    jaxpr = jaxpr.replace(eqns=before_loop_eqns + [loop_eqn] + after_loop_eqns)
    loop_eqn_idx = len(before_loop_eqns)
    loop_eqn = jaxpr.eqns[loop_eqn_idx]

    before_loop_jaxpr = jaxpr_from_eqns(
        jaxpr.eqns[:loop_eqn_idx], eqns_free_vars(jaxpr.eqns[loop_eqn_idx:])[0]
    )
    # Use current placement to taskify loop body
    tasked_loop_eqn = wrap_into_tasks_inside_loop(loop_eqn)

    # Use current placement to taskify before loop
    loop_args_mpmd_refs = join_argument_refs(
        tasked_loop_eqn.invars, tasked_loop_eqn.params["in_mpmd_refs"]
    )

    before_loop_out_refs = tuple(
        loop_args_mpmd_refs.get(outvar)
        for outvar in cast(list[jcore.Var], before_loop_jaxpr.outvars)
    )
    before_loop_tasked_jaxpr, _, _ = mpmd_unzip_reverse(
        before_loop_jaxpr, before_loop_out_refs, name="before_loop"
    )

    task_eqns = list[jcore.JaxprEqn](before_loop_tasked_jaxpr.eqns)
    task_eqns.append(tasked_loop_eqn)

    mpmd_refs, mpmd_def = _compute_mpmd_def_refs(task_eqns)
    if len(jaxpr.eqns[loop_eqn_idx + 1 :]) > 0:
        after_loop_jaxpr = jaxpr_from_eqns(
            jaxpr.eqns[loop_eqn_idx + 1 :], set(nonlit(jaxpr.outvars))
        )
        tasked_after_loop_jaxpr, in_mpmd_refs, after_loop_outvar_placement = (
            wrap_into_tasks_after_loop(
                after_loop_jaxpr,
                # NOTE: inputs to `after_loop_jaxpr` that might have not been
                #  used so far (such as optimizer state), might not have an mpmd_idx
                #  defined just yet. Hence `.get(invar)` instead of `[invar]`.
                [mpmd_def.get(invar) for invar in after_loop_jaxpr.invars],
                mpmd_dim,
            )
        )

        for invar, ref_p in zip(after_loop_jaxpr.invars, in_mpmd_refs, strict=True):
            assert ref_p is not None
            mpmd_refs[invar].update(ref_p)

        mpmd_def.update(
            zip(
                cast(list[jcore.Var], after_loop_jaxpr.outvars),
                after_loop_outvar_placement,
                strict=True,
            )
        )
        new_jaxpr = jaxpr.replace(eqns=task_eqns + tasked_after_loop_jaxpr.eqns)
    else:
        new_jaxpr = jaxpr.replace(eqns=task_eqns)

    for invar, is_used in zip(jaxpr.invars, used_invars):
        if is_used:
            refs = mpmd_refs.get(invar)
            if refs is None:
                raise AssertionError()
            mpmd_def[invar] = refs
        else:
            assert invar not in mpmd_def
            mpmd_def[invar] = set()

    new_jaxpr = new_jaxpr.replace(
        effects=jcore.join_effects(*(eqn.effects for eqn in new_jaxpr.eqns))
    )

    return (
        cjaxpr.replace(jaxpr=new_jaxpr),
        tuple(mpmd_def.get(invar) or set() for invar in new_jaxpr.invars),
        tuple(
            mpmd_def[outvar] if isinstance(outvar, jcore.Var) else set(range(mpmd_dim))
            for outvar in new_jaxpr.outvars
        ),
    )


def wrap_into_tasks(
    cjaxpr: jcore.ClosedJaxpr, used_invars: Sequence[bool], mpmd_dim: int
) -> tuple[jcore.ClosedJaxpr, tuple[set[MpmdIdx]], tuple[set[MpmdIdx]]]:
    return _wrap_into_tasks(cjaxpr, used_invars, mpmd_dim)


@unwrap_closed
def infer_donation(
    tasked_jaxpr: jcore.Jaxpr, donated_invars: Sequence[bool]
) -> jcore.Jaxpr:
    """
    Returns a new jaxpr identical to the input jaxpr, with task and collective
    donation metadata set according to value lifetimes.
    """
    last_use = last_used(tasked_jaxpr)
    invar_is_donated = dict.fromkeys(tasked_jaxpr.constvars, False)
    invar_is_donated.update(zip(tasked_jaxpr.invars, donated_invars, strict=True))
    undonateable_vars = set[jcore.Var]()

    least_donation = dict[jcore.ClosedJaxpr, Sequence[bool]]()
    new_eqns = []

    def can_end_lifetime(invar: jcore.Atom, eqn_idx: int):
        return (
            isinstance(invar, jcore.Var)
            and invar.aval is not jcore.abstract_token
            and last_use.get(invar) == eqn_idx
            and invar_is_donated.get(invar, True)
        )

    def donation_for_eqn(eqn_idx: int, eqn: jcore.JaxprEqn):
        return tuple(
            can_end_lifetime(invar, eqn_idx) and invar not in undonateable_vars
            for invar in eqn.invars
        )

    for eqn_idx, eqn in enumerate(tasked_jaxpr.eqns):
        donation = donation_for_eqn(eqn_idx, eqn)

        if eqn.primitive is task_p:
            eqn = eqn.replace(params=eqn.params | {"donate_invars": donation})
            new_eqns.append(eqn)

            least_donation[eqn.params["call_jaxpr"]] = tuple(
                min(prev, curr)
                for prev, curr in zip(
                    least_donation.get(
                        eqn.params["call_jaxpr"], (True,) * len(donation)
                    ),
                    donation,
                    strict=True,
                )
            )
        elif eqn.primitive is transfer_p:
            undonateable_vars.update(nonlit(eqn.invars))
            # FIXME: it should be ok to donate outvars of transfer equations.
            undonateable_vars.update(nonlit(eqn.outvars))
            new_eqns.append(eqn)
        elif eqn.primitive is transfer_start_p:
            send_count = len(eqn.params["send_local_shardings"])
            undonateable_vars.update(nonlit(eqn.invars[:send_count]))
            new_eqns.append(eqn)
        elif eqn.primitive in {add_multi_p, gather_multi_p}:
            eqn = eqn.replace(params=eqn.params | {"donate_invars": donation})
            new_eqns.append(eqn)
        elif eqn.primitive in {all_reduce_p, all_gather_p}:
            donated_argnums = tuple(
                idx for idx, donated in enumerate(donation) if donated
            )
            eqn = eqn.replace(
                params=eqn.params
                | {"donated": donated_argnums if donated_argnums else None}
            )
            new_eqns.append(eqn)
        elif eqn.primitive is recv_done_p or eqn.primitive is transfer_done_p:
            undonateable_vars.update(outvar for outvar in eqn.outvars)
            new_eqns.append(eqn)
        elif eqn.primitive in {stack_p, slice_p, local_stack_p, local_slice_p}:
            new_eqns.append(eqn)
        elif eqn.primitive in {reuse_fence_p, zeros_p, delete_p}:
            new_eqns.append(eqn)
        else:
            raise ValueError(f"Unexpected equation with primitive {eqn.primitive}")

    # NOTE: the same task function applied to different
    #  microbatches will have the same `call_jaxpr`.
    #  Here, we set the donation to the least donation among all of the task's
    #  instantiations to minimize the compilation misses
    if env_vars.jaxpp_share_donation.value:
        res = []
        for eqn in new_eqns:
            if eqn.primitive is task_p:
                eqn = eqn.replace(
                    params=eqn.params
                    | {"donate_invars": least_donation[eqn.params["call_jaxpr"]]}
                )
            res.append(eqn)
        new_eqns = res
    res = tasked_jaxpr.replace(eqns=new_eqns)
    check_jaxpr(res)
    return res


@unwrap_closed
def add_deletes(
    tasked_jaxpr: jcore.Jaxpr, donated_invars: Sequence[bool]
) -> jcore.Jaxpr:
    """
    Returns a new jaxpr with delete equations inserted after each donated
    value's last use. DLPack-backed recv buffers are not deleted directly.
    """
    last_use = last_used(tasked_jaxpr)
    invar_is_donated = dict.fromkeys(tasked_jaxpr.constvars, False)
    invar_is_donated.update(zip(tasked_jaxpr.invars, donated_invars, strict=True))
    no_delete_vars = set[jcore.Var]()
    for eqn in tasked_jaxpr.eqns:
        if eqn.primitive in {
            transfer_done_p,
            recv_done_p,
            stack_p,
            slice_p,
            local_stack_p,
            local_slice_p,
        }:
            no_delete_vars.update(nonlit(eqn.invars))
        elif eqn.primitive is transfer_start_p:
            send_count = len(eqn.params["send_local_shardings"])
            no_delete_vars.update(nonlit(eqn.invars[send_count:]))

    new_eqns = []

    def can_end_lifetime(invar: jcore.Atom, eqn_idx: int):
        return (
            isinstance(invar, jcore.Var)
            and invar.aval is not jcore.abstract_token
            and last_use.get(invar) == eqn_idx
            and invar_is_donated.get(invar, True)
        )

    def append_deletes(eqn_idx: int, eqn: jcore.JaxprEqn):
        if eqn.primitive is delete_p:
            return
        delete_invars_mask = [
            can_end_lifetime(invar, eqn_idx) and invar not in no_delete_vars
            for invar in eqn.invars
        ]

        if not any(delete_invars_mask):
            return

        # It is fine to emit delete after a donated use. delete_p handles donated
        # buffers.
        delete_invars = [
            invar
            for should_delete, invar in zip(delete_invars_mask, eqn.invars, strict=True)
            if should_delete
        ]
        new_eqns.append(
            new_primitive_eqn(
                delete_p,
                delete_invars,
                [jcore.DropVar(invar.aval) for invar in delete_invars],
            )
        )

    for eqn_idx, eqn in enumerate(tasked_jaxpr.eqns):
        if eqn.primitive not in {
            task_p,
            add_multi_p,
            gather_multi_p,
            all_reduce_p,
            all_gather_p,
            transfer_p,
            transfer_done_p,
            stack_p,
            slice_p,
            local_stack_p,
            local_slice_p,
            recv_done_p,
            transfer_start_p,
            reuse_fence_p,
            zeros_p,
            delete_p,
        }:
            raise ValueError(f"Unexpected equation with primitive {eqn.primitive}")

        new_eqns.append(eqn)
        append_deletes(eqn_idx, eqn)

    res = tasked_jaxpr.replace(eqns=new_eqns)
    check_jaxpr(res)
    return res


@unwrap_closed
def finalize_lifetimes(
    tasked_jaxpr: jcore.Jaxpr, donated_invars: Sequence[bool]
) -> jcore.Jaxpr:
    """
    Returns a new jaxpr with donation metadata and delete equations finalized.
    """
    tasked_jaxpr = infer_donation(tasked_jaxpr, donated_invars)
    return add_deletes(tasked_jaxpr, donated_invars)


def unroll_loop(
    loop_jaxpr: jcore.Jaxpr, n_consts: int, n_mubatches: int
) -> jcore.Jaxpr:
    gensym = mk_gensym()

    consts, carry = loop_jaxpr.invars[:n_consts], loop_jaxpr.invars[n_consts:]
    new_eqns = []
    for mubatch_idx in range(n_mubatches):
        env: dict[jcore.Var, jcore.Atom] = dict(
            zip(loop_jaxpr.invars, it.chain(consts, carry), strict=True)
        )
        for eqn in loop_jaxpr.eqns:
            outvars = [gensym(outvar.aval) for outvar in eqn.outvars]
            new_eqns.append(
                eqn.replace(
                    invars=[
                        env[invar] if isinstance(invar, jcore.Var) else invar
                        for invar in eqn.invars
                    ],
                    outvars=outvars,
                    params=eqn.params
                    | {
                        "call_counter": mubatch_idx,
                        "task_name": eqn.params["task_name"],
                    },
                )
            )
            env.update(zip(eqn.outvars, outvars))

        carry = [
            env[outvar] if isinstance(outvar, jcore.Var) else outvar
            for outvar in loop_jaxpr.outvars
        ]

    return loop_jaxpr.replace(eqns=new_eqns, outvars=carry)


def build_eqn_dependencies(eqns: list[jcore.JaxprEqn]):
    defs = dict[jcore.Var, int]()
    task_dependencies = dict[int, set[int]]()
    task_results_uses = dict[int, set[int]]()
    for eqn_idx, eqn in enumerate(eqns):
        def_eqn_idxs = {
            def_eqn_idx
            for invar in eqn.invars
            if isinstance(invar, jcore.Var)
            and (def_eqn_idx := defs.get(invar)) is not None
        }
        for def_eqn_idx in def_eqn_idxs:
            task_results_uses[def_eqn_idx].add(eqn_idx)
        task_dependencies[eqn_idx] = def_eqn_idxs
        task_results_uses[eqn_idx] = set()
        defs.update(zip(eqn.outvars, it.repeat(eqn_idx)))
    return task_dependencies, task_results_uses


def reorder_nodes_with_schedule(
    nodes: list[Task],
    dependencies_and_uses: tuple[dict[int, set[int]], dict[int, set[int]]],
    schedule_tasks: list[list[Task | FusedTask]],
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    assert len(nodes) == len(set(nodes))

    mpmd_dim = len(schedule_tasks)
    node_unmet_dependencies, task_results_uses = dependencies_and_uses
    node_unmet_dependencies = dict(node_unmet_dependencies)
    node_idx_for_task = {t: i for i, t in enumerate(nodes)}

    node_ready_time = dict[Task, int]()
    for node_idx, deps in node_unmet_dependencies.items():
        if len(deps) == 0:
            node_ready_time[nodes[node_idx]] = 0

    time_by_mpmd_idx = [0] * mpmd_dim
    schedule_idx_by_mpmd_idx = [0] * mpmd_dim

    scheduled_groups = list[tuple[int, int, list[int]]]()
    while len(node_unmet_dependencies) > 0:
        # Select which next mpmd_idx should make progress
        # We choose the mpmd_idx that has the smallest ready_time
        def next_mpmd_idx():
            time_and_mpmd_idx: tuple[int, int, FusedTask] | None = None
            for tentative_mpmd_idx, time in enumerate(time_by_mpmd_idx):
                task_idx_in_schedule = schedule_idx_by_mpmd_idx[tentative_mpmd_idx]

                this_ranks_tasks = schedule_tasks[tentative_mpmd_idx]
                if task_idx_in_schedule >= len(this_ranks_tasks):
                    # This mpmd_idx has no more tasks to schedule
                    continue

                maybe_unfused_task = this_ranks_tasks[task_idx_in_schedule]
                assert isinstance(maybe_unfused_task, (Task, FusedTask))
                fused_task = (
                    FusedTask([maybe_unfused_task])
                    if isinstance(maybe_unfused_task, Task)
                    else maybe_unfused_task
                )

                satisfied_dependencies = set()

                # A fused task is ready only after every member task's external
                # dependencies are ready, so the group waits on its slowest input.
                # Dependencies within the group may be satisfied by earlier members.
                #
                # This is also a runtime boundary: fuse_groups lowers the group to
                # one task_p / XLA call, so externally visible outputs become
                # available only when the fused call finishes. A transfer for an
                # early member's output cannot be inserted inside the fused group.
                #
                # We disallow the pattern below (assuming F1 -> F2 and B2 -> B1)
                # [..., [F1(4) B1(3)], ...]
                # [..., [B2(3) F2(4)], ...]
                not_ready = False
                for t in fused_task:
                    deps = node_unmet_dependencies.get(node_idx_for_task[t], set())
                    if len(deps - satisfied_dependencies) > 0:
                        not_ready = True
                        break
                    # Unmet dependencies are satisfied
                    elif len(deps) > 0:
                        node_ready_time[t] = max(
                            node_ready_time[nodes[_]] for _ in deps
                        )
                    assert t in node_ready_time
                    satisfied_dependencies.add(node_idx_for_task[t])

                if not_ready:
                    continue

                node_scheduled_time = max(
                    time, *(node_ready_time[t] for t in fused_task)
                )
                if (
                    time_and_mpmd_idx is None
                    or node_scheduled_time < time_and_mpmd_idx[0]
                ):
                    time_and_mpmd_idx = (
                        node_scheduled_time,
                        tentative_mpmd_idx,
                        fused_task,
                    )
            return time_and_mpmd_idx

        pos = {
            mpmd_idx: (schedule_idx, tasks[schedule_idx])
            for mpmd_idx, schedule_idx in enumerate(schedule_idx_by_mpmd_idx)
            if schedule_idx < len(tasks := schedule_tasks[mpmd_idx])
        }
        if (_ := next_mpmd_idx()) is None:
            avail_tasks = [
                nodes[node_idx]
                for node_idx, _ in node_unmet_dependencies.items()
                if len(_) == 0
            ]
            msg = (
                f"Tasks available for scheduling {avail_tasks}.\n"
                f"Current position in schedule for ranks {pos}."
            )
            raise AssertionError("Schedule does not honor data dependencies.\n" + msg)

        curr_time, mpmd_idx, fused_tasks = _
        assert schedule_tasks[mpmd_idx][schedule_idx_by_mpmd_idx[mpmd_idx]] == (
            fused_tasks if len(fused_tasks) > 1 else fused_tasks[0]
        )

        start_time = curr_time
        end_time = start_time
        scheduled_node_idxs = []
        for task in fused_tasks:
            node_idx = node_idx_for_task[task]
            node_unmet_dependencies.pop(node_idx, None)

            scheduled_node_idxs.append(node_idx)
            end_time += task.latency

            # Remove dependency for dependent tasks
            for use_eqn_idx in task_results_uses[node_idx]:
                node_unmet_dependencies[use_eqn_idx].remove(node_idx)
                if (
                    len(node_unmet_dependencies[use_eqn_idx]) == 0
                    and nodes[use_eqn_idx] not in node_ready_time
                ):
                    # This scheduling model records the dependency as ready at the
                    # producing member's end_time. After lowering, fused task
                    # outputs are exposed only at the full group's end. Keeping
                    # this optimistic timing preserves the existing scheduler
                    # behavior, but it is not a valid P2P send-start point inside
                    # a fused task_p.
                    node_ready_time[nodes[use_eqn_idx]] = end_time

        scheduled_groups.append((start_time, end_time, scheduled_node_idxs))
        schedule_idx_by_mpmd_idx[mpmd_idx] += 1
        time_by_mpmd_idx[mpmd_idx] = end_time

    for mpmd_idx, time in enumerate(time_by_mpmd_idx):
        if schedule_idx_by_mpmd_idx[mpmd_idx] != len(schedule_tasks[mpmd_idx]):
            raise AssertionError(
                "Loop tasks have ended but schedule contains other tasks. "
                f"{schedule_tasks[mpmd_idx][schedule_idx_by_mpmd_idx[mpmd_idx]:]}"
            )

    scheduled_groups.sort(key=lambda e: (e[0], e[1]))
    return (
        [scheduled_node_idxs for _, _, scheduled_node_idxs in scheduled_groups],
        [(start_time, end_time) for start_time, end_time, _ in scheduled_groups],
    )


class TransferTo(NamedTuple):
    tgt_mpmd_idx: int
    out_idx: int
    first_use_eqn_idx: int
    first_use_invar_idx: int


def compute_transfers(eqns: list[jcore.JaxprEqn]) -> list[list[TransferTo]]:
    class VisitedElem(NamedTuple):
        tgt_mpmd_idx: int
        var: jcore.Var

    mpmd_def, _ = defs_and_uses(eqns)

    transfers = [list[TransferTo]() for _ in range(len(eqns))]
    resolved_transfer = set[VisitedElem]()
    cross_mpmd_primitives = {add_multi_p, gather_multi_p}
    for eqn_idx, eqn in enumerate(eqns):
        if eqn.primitive in cross_mpmd_primitives:
            continue
        assert eqn.primitive is task_p
        eqn_mpmd_idx = eqn.params["mpmd_idx"]
        for invar_idx, invar in enumerate(eqn.invars):
            if not isinstance(invar, jcore.Var):
                continue
            key = (eqn_mpmd_idx, invar)

            if key in resolved_transfer:
                continue

            if (_ := mpmd_def.get(invar)) is None:
                # NOTE: we have to resolve only transfers between
                # equations but not for inputs (i.e. when `_ is None`)
                # because the caller of the loop has ensured that
                # they were placed correctly
                continue

            def_eqn = eqns[_.eqn_idx]
            if def_eqn.primitive in cross_mpmd_primitives:
                continue
            assert def_eqn.primitive is task_p, def_eqn.primitive

            if eqn_mpmd_idx != def_eqn.params["mpmd_idx"]:
                resolved_transfer.add(VisitedElem(eqn_mpmd_idx, invar))
                transfers[_.eqn_idx].append(
                    TransferTo(eqn_mpmd_idx, _.outvar_idx, eqn_idx, invar_idx)
                )

    return transfers


def add_transfers(jaxpr: jcore.Jaxpr, times: list[tuple[int, int]]) -> jcore.Jaxpr:
    assert len(jaxpr.eqns) == len(times), (len(jaxpr.eqns), len(times))
    transfers = compute_transfers(jaxpr.eqns)
    new_eqns = []
    prev_start_time = 0
    sub_by_mpmd_idx = defaultdict(dict[jcore.Var, jcore.Var])
    # TODO: revisit when transfers are scheduled and bufferize receives
    #  earlier than the transfer
    next_time_transfers = list[tuple[int, jcore.JaxprEqn]]()
    for eqn_idx, ((start_time, _end_time), eqn) in enumerate(
        zip(times, jaxpr.eqns, strict=True)
    ):
        if start_time != prev_start_time:
            for _, transfer in sorted(next_time_transfers, key=operator.itemgetter(0)):
                new_eqns.append(transfer)
            prev_start_time = start_time
            next_time_transfers = []

        if eqn.primitive is add_multi_p or eqn.primitive is gather_multi_p:
            new_eqns.append(eqn)
            continue

        assert eqn.primitive is task_p, eqn.primitive

        sub = sub_by_mpmd_idx[eqn.params["mpmd_idx"]]
        eqn = eqn.replace(
            invars=[
                sub.get(invar, invar) if isinstance(invar, jcore.Var) else invar
                for invar in eqn.invars
            ]
        )
        new_eqns.append(eqn)
        for tgt_mpmd_idx, ts in groupby(
            (t.tgt_mpmd_idx, t)
            for t in sorted(
                transfers[eqn_idx], key=lambda _: (_.tgt_mpmd_idx, _.out_idx)
            )
        ).items():
            invars = [eqn.outvars[t.out_idx] for t in ts]
            src_shardings = tuple(eqn.params["out_shardings"][t.out_idx] for t in ts)
            tgt_shardings = tuple(
                jaxpr.eqns[t.first_use_eqn_idx].params["in_shardings"][
                    t.first_use_invar_idx
                ]
                for t in ts
            )
            transfer_params: TransferParams = {
                "src_shardings": src_shardings,
                "tgt_shardings": tgt_shardings,
            }
            transfer_eqn = new_primitive_eqn(transfer_p, invars, **transfer_params)
            token_outvar, *outvars = transfer_eqn.outvars
            next_time_transfers.append(
                (min(_.first_use_eqn_idx for _ in ts), transfer_eqn)
            )
            for i, o in zip(invars, outvars):
                assert i not in sub_by_mpmd_idx[tgt_mpmd_idx]
                sub_by_mpmd_idx[tgt_mpmd_idx][i] = o

    return jaxpr.replace(
        eqns=new_eqns, effects=jcore.join_effects(*(eqn.effects for eqn in new_eqns))
    )


@unwrap_closed
def add_transfer_dones(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    explicit_done_tokens = {
        token
        for eqn in jaxpr.eqns
        if eqn.primitive is transfer_done_p
        for token in eqn.invars[:1]
        if isinstance(token, jcore.Var)
    }
    sub = dict[jcore.Var, jcore.Var]()
    new_eqns = []
    for eqn in jaxpr.eqns:
        eqn = eqn.replace(
            invars=[
                sub.get(invar, invar) if isinstance(invar, jcore.Var) else invar
                for invar in eqn.invars
            ]
        )
        new_eqns.append(eqn)
        if eqn.primitive is transfer_p:
            token, *transfer_outvars = eqn.outvars
            assert isinstance(token, jcore.Var), eqn
            if token in explicit_done_tokens:
                continue
            transfer_done_eqn = new_primitive_eqn(
                transfer_done_p, [token, *transfer_outvars]
            )
            new_eqns.append(transfer_done_eqn)
            sub.update(zip(transfer_outvars, transfer_done_eqn.outvars, strict=True))

    outvars = [
        sub.get(outvar, outvar) if isinstance(outvar, jcore.Var) else outvar
        for outvar in jaxpr.outvars
    ]
    return jaxpr.replace(
        eqns=new_eqns,
        outvars=outvars,
        effects=jcore.join_effects(*(eqn.effects for eqn in new_eqns)),
    )


_fused_open_jaxprs = weakref.WeakValueDictionary[Hashable, jcore.Jaxpr]()
_fused_closed_jaxprs = weakref.WeakValueDictionary[Hashable, jcore.ClosedJaxpr]()
_TASK_METADATA_PARAM_KEYS = {
    "call_counter",
    "latency",
    "mpmd_idx",
    "task_info",
    "task_name",
}


def _fused_jaxpr_cache_key(
    group_eqns: list[jcore.JaxprEqn],
    invars: list[jcore.Var],
    outvars: list[jcore.Atom],
    exclude_param_keys: set[str] | None = None,
    *,
    constvars: Sequence[jcore.Var] = (),
) -> Hashable:
    if exclude_param_keys is None:
        exclude_param_keys = set()

    eqn_keys = []
    counter = it.count()
    ids = defaultdict(lambda: next(counter))

    def _atom_id(atom: jcore.Atom) -> Hashable:
        """Return a hashable id for a Var or Literal atom."""
        if isinstance(atom, jcore.Literal):
            # Literal.__hash__ is None; use Literal.hash property which
            # handles numpy scalars via val.item() (see jax/_src/core.py).
            return ("lit", atom.hash, atom.aval)
        return (ids[atom], atom.aval)

    for eqn in group_eqns:
        invars_ids = tuple(_atom_id(invar) for invar in eqn.invars)
        outvars_ids = tuple(_atom_id(outvar) for outvar in eqn.outvars)
        # FIXME: maybe add effects? Not strictly necessary as usually
        #  are inferred by primitive
        eqn_keys.append(
            (
                invars_ids,
                eqn.primitive,
                outvars_ids,
                hashable_params(eqn.params, exclude_param_keys),
            )
        )

    jaxpr_constvars_key = tuple(_atom_id(v) for v in constvars)
    jaxpr_invars_key = tuple(_atom_id(v) for v in invars)
    jaxpr_outvars_key = tuple(_atom_id(v) for v in outvars)
    return (jaxpr_constvars_key, jaxpr_invars_key, tuple(eqn_keys), jaxpr_outvars_key)


@overload
def _get_fused_jaxpr_cached(
    group_eqns: list[jcore.JaxprEqn],
    invars: list[jcore.Var],
    outvars: list[jcore.Atom],
    exclude_param_keys: set[str] | None = None,
    *,
    want_closed: Literal[False],
    constvars: Sequence[jcore.Var] = (),
    debug_info=None,
) -> jcore.Jaxpr: ...


@overload
def _get_fused_jaxpr_cached(
    group_eqns: list[jcore.JaxprEqn],
    invars: list[jcore.Var],
    outvars: list[jcore.Atom],
    exclude_param_keys: set[str] | None = None,
    *,
    want_closed: Literal[True],
    constvars: Sequence[jcore.Var] = (),
    debug_info=None,
) -> jcore.ClosedJaxpr: ...


def _get_fused_jaxpr_cached(
    group_eqns: list[jcore.JaxprEqn],
    invars: list[jcore.Var],
    outvars: list[jcore.Atom],
    exclude_param_keys: set[str] | None = None,
    *,
    want_closed: bool,
    constvars: Sequence[jcore.Var] = (),
    debug_info=None,
) -> jcore.Jaxpr | jcore.ClosedJaxpr:
    cache_key = _fused_jaxpr_cache_key(
        group_eqns, invars, outvars, exclude_param_keys, constvars=constvars
    )

    open_jaxpr = _fused_open_jaxprs.get(cache_key)
    closed_jaxpr = _fused_closed_jaxprs.get(cache_key)

    if open_jaxpr is None and closed_jaxpr is not None:
        open_jaxpr = closed_jaxpr.jaxpr
        _fused_open_jaxprs[cache_key] = open_jaxpr

    if open_jaxpr is None:
        open_jaxpr = jcore.Jaxpr(
            constvars=constvars,
            invars=invars,
            outvars=outvars,
            eqns=group_eqns,
            effects=jcore.join_effects(*(eqn.effects for eqn in group_eqns)),
            debug_info=debug_info,
        )
        _fused_open_jaxprs[cache_key] = open_jaxpr

    if not want_closed:
        return open_jaxpr

    if closed_jaxpr is None or closed_jaxpr.jaxpr is not open_jaxpr:
        closed_jaxpr = jcore.ClosedJaxpr(open_jaxpr, ())
        _fused_closed_jaxprs[cache_key] = closed_jaxpr
    return closed_jaxpr


def _canonicalize_jaxpr(
    cjaxpr: AnyJaxpr, exclude_param_keys: set[str] | None = None
) -> AnyJaxpr:
    """Bottom-up deduplication of Jaxpr/ClosedJaxpr-typed params.

    For each equation, recursively normalizes any jaxpr-typed params by
    first deduplicating their inner equations, then deduplicating the
    jaxpr itself via _get_fused_jaxpr_cached.  This ensures that
    structurally-equivalent jaxpr objects become the same Python
    object, enabling identity-based cache hits downstream.
    """
    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        if len(cjaxpr.consts) > 0:
            return cjaxpr
        jaxpr = cjaxpr.jaxpr
    else:
        jaxpr = cjaxpr

    new_eqns = []
    for eqn in jaxpr.eqns:
        updated_params = {}
        for k, v in eqn.params.items():
            if isinstance(v, (jcore.ClosedJaxpr, jcore.Jaxpr)):
                deduped = _canonicalize_jaxpr(v, exclude_param_keys)
                if deduped is not v:
                    updated_params[k] = deduped

        if len(updated_params) > 0:
            new_eqns.append(eqn.replace(params=eqn.params | updated_params))
        else:
            new_eqns.append(eqn)

    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        return _get_fused_jaxpr_cached(
            new_eqns,
            jaxpr.invars,
            jaxpr.outvars,
            exclude_param_keys,
            want_closed=True,
            constvars=jaxpr.constvars,
            debug_info=jaxpr.debug_info,
        )

    return _get_fused_jaxpr_cached(
        new_eqns,
        jaxpr.invars,
        jaxpr.outvars,
        exclude_param_keys,
        want_closed=False,
        constvars=jaxpr.constvars,
        debug_info=jaxpr.debug_info,
    )


def deduplicate_task_jaxprs(closed_jaxpr: jcore.ClosedJaxpr) -> jcore.ClosedJaxpr:
    """Deduplicate call_jaxpr of task_p equations.

    Ensures structurally equivalent tasks (e.g. fwd_1 and fwd_2) share
    the same call_jaxpr Python object, so that the identity-based cache
    in fuse_groups / _get_fused_jaxpr_cached produces hits after unrolling.
    """
    changed = False
    new_eqns = []
    for eqn in closed_jaxpr.eqns:
        params_update = {}
        if eqn.primitive is task_p:
            call_jaxpr = eqn.params["call_jaxpr"]
            assert isinstance(call_jaxpr, jcore.ClosedJaxpr)
            deduped = _canonicalize_jaxpr(call_jaxpr, _TASK_METADATA_PARAM_KEYS)
            if deduped is not call_jaxpr:
                params_update["call_jaxpr"] = deduped
        elif eqn.primitive is dax_pscan_p:
            loop_jaxpr = eqn.params["jaxpr"]
            deduped = deduplicate_task_jaxprs(loop_jaxpr)
            if deduped is not loop_jaxpr:
                params_update["jaxpr"] = deduped

        updated_eqn = eqn
        if len(params_update) > 0:
            updated_eqn = eqn.replace(params=eqn.params | params_update)
            changed = True
        new_eqns.append(updated_eqn)

    if not changed:
        return closed_jaxpr
    return closed_jaxpr.replace(jaxpr=closed_jaxpr.jaxpr.replace(eqns=new_eqns))


def fuse_groups(jaxpr: jcore.Jaxpr, groups: list[list[int]]):
    # FIXME: this function does not check that groups are in topological order
    # TODO: assert that fusion groups belong to the same mpmd_idx?
    new_eqns = []
    last_use = last_used(jaxpr)
    for group_eqn_idxs in groups:
        group_eqns = [jaxpr.eqns[eqn_idx] for eqn_idx in group_eqn_idxs]
        if len(group_eqns) == 1:
            new_eqns.append(group_eqns[0])
            continue

        _ = {e.params["mpmd_idx"] for e in group_eqns}
        assert len(_) == 1
        (mpmd_idx,) = _

        # The fused group is lowered to one task_p / XLA call. This is useful
        # for donation and memory reuse inside the fused jaxpr, but it removes
        # any externally visible boundary between member tasks: transfer
        # insertion can only place sends before or after this fused task, and
        # all fused outputs become observable only after the whole call finishes.
        defs, free_uses = defs_and_free_uses(group_eqns)

        invars = []
        in_shardings = []
        donate_invars = []
        for v, uses in free_uses.items():
            first_use = uses[0]
            first_use_eqn = group_eqns[first_use.eqn_idx]
            invars.append(v)
            in_shardings.append(
                first_use_eqn.params["in_shardings"][first_use.invar_idx]
            )
            donate_invars.append(
                first_use_eqn.params["donate_invars"][first_use.invar_idx]
            )

        outvars = []
        out_shardings = []
        group_eqn_idxs_set = set(group_eqn_idxs)
        for v, def_site in defs.items():
            if (eqn_idx := last_use[v]) is None or eqn_idx not in group_eqn_idxs_set:
                outvars.append(v)
                out_shardings.append(
                    group_eqns[def_site.eqn_idx].params["out_shardings"][
                        def_site.outvar_idx
                    ]
                )

        fused_task_jaxpr = _get_fused_jaxpr_cached(
            group_eqns,
            invars,
            outvars,
            exclude_param_keys=_TASK_METADATA_PARAM_KEYS,
            want_closed=True,
        )

        new_eqns.append(
            _task_eqn(
                invars=invars,
                outvars=outvars,
                task_jaxpr=fused_task_jaxpr,
                mpmd_idx=mpmd_idx,
                in_shardings=tuple(in_shardings),
                out_shardings=tuple(out_shardings),
                donate_invars=tuple(donate_invars),
                task_name="fused_"
                + "_".join(e.params["task_name"] for e in group_eqns),
                task_info=None,
                latency=sum(e.params["latency"] for e in group_eqns),
            )
        )

    res = jaxpr.replace(eqns=new_eqns)
    check_jaxpr(res)
    return res


def infer_times(task_eqns: list[jcore.JaxprEqn]):
    defs, _ = defs_and_uses(task_eqns)
    time_by_mpmd_idx = defaultdict(lambda: 0)

    res = []
    for eqn in task_eqns:
        invar_definitions = [
            d for invar in nonlit(eqn.invars) if (d := defs.get(invar)) is not None
        ]
        if eqn.primitive is add_multi_p or eqn.primitive is gather_multi_p:
            start = max(
                *(time_by_mpmd_idx[mpmd_idx] for mpmd_idx in eqn.params["mpmd_idxs"]),
                *(res[d.eqn_idx][1] for d in invar_definitions),
                0,
            )
            end = start + 1  # FIXME(task_latency)
            for mpmd_idx in eqn.params["mpmd_idxs"]:
                time_by_mpmd_idx[mpmd_idx] = end

            res.append((start, end))
            continue

        assert eqn.primitive is task_p, eqn.primitive
        assert eqn.params["latency"] is not None

        start = max(
            time_by_mpmd_idx[eqn.params["mpmd_idx"]],
            max((res[d.eqn_idx][1] for d in invar_definitions), default=0),
        )
        end = start + eqn.params["latency"]
        time_by_mpmd_idx[eqn.params["mpmd_idx"]] = end
        res.append((start, end))
    return res


def unroll_loop_eqn(loop_eqn: jcore.JaxprEqn):
    n_consts = loop_eqn.params["n_consts"]
    n_mubatches = loop_eqn.params["n_mubatches"]
    schedule = loop_eqn.params["schedule"]

    loop_jaxpr: jcore.ClosedJaxpr = loop_eqn.params["jaxpr"]

    # FIXME: make first_stage_id a parameter
    first_stage_id = 0
    schedule_tasks = preprocess_schedule_tasks(
        schedule.tasks(n_mubatches),
        first_stage_id=first_stage_id,
        unpack_fused_tasks=env_vars.jaxpp_disable_schedule_task_fusion.value,
    )

    # NOTE: `unroll_loop.outvars` are fresh
    unrolled_loop_jaxpr = unroll_loop(loop_jaxpr.jaxpr, n_consts, n_mubatches)
    scheduled_node_groups, _scheduled_times = reorder_nodes_with_schedule(
        [
            Task.make(
                stage_id=task_eqn.params["task_info"][0],
                mubatch_idx=task_eqn.params["call_counter"],
                fwd_or_bwd=task_eqn.params["task_info"][1],
            )
            for task_eqn in unrolled_loop_jaxpr.eqns
        ],
        build_eqn_dependencies(unrolled_loop_jaxpr.eqns),
        schedule_tasks=schedule_tasks,
    )

    scheduled_and_fused_jaxpr = fuse_groups(unrolled_loop_jaxpr, scheduled_node_groups)

    inlined_loop_eqns = inline_eqns(
        scheduled_and_fused_jaxpr.eqns,
        dict(zip(scheduled_and_fused_jaxpr.invars, loop_eqn.invars)),
        result_binding=dict(zip(scheduled_and_fused_jaxpr.outvars, loop_eqn.outvars)),
    )
    return inlined_loop_eqns


class MpmdDefs:
    def __init__(self, values, indptr):
        assert len(values) == indptr[-1]
        self.values = values
        self.indptr = indptr

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        return set(self.values[self.indptr[idx] : self.indptr[idx + 1]])


def deduplicate_outvars(
    jaxpr: jcore.Jaxpr, defined_in_mpmd_idx_as: list[dict[int, jcore.Var]]
):
    copy_outvars = []
    outvar_def = []
    replicas = [0]
    new_result_paths = []
    result_paths = jaxpr.debug_info.result_paths

    for i, (outvar, name_in_mpmd_idx) in enumerate(
        zip(jaxpr.outvars, defined_in_mpmd_idx_as, strict=True)
    ):
        if isinstance(outvar, jcore.Literal):
            raise NotImplementedError()

        copies = sorted(name_in_mpmd_idx.items(), key=operator.itemgetter(0))
        mpmd_idxs, vs = jc.unzip2(copies)

        outvar_def.extend(mpmd_idxs)
        copy_outvars.extend(vs)

        if result_paths is not None:
            new_result_paths.extend([result_paths[i]] * len(mpmd_idxs))

        replicas.append(len(copy_outvars))

    return jaxpr.replace(
        outvars=copy_outvars,
        debug_info=jaxpr.debug_info._replace(
            result_paths=tuple(new_result_paths) if result_paths is not None else None
        ),
    ), MpmdDefs(outvar_def, replicas)


def fixup_multidefs(cjaxpr: jcore.ClosedJaxpr) -> tuple[jcore.ClosedJaxpr, MpmdDefs]:
    jaxpr = cjaxpr.jaxpr
    invars = set(jaxpr.invars)
    defined = dict[jcore.Var, int]()
    # Track outputs from cross-MPMD primitives separately. They're defined in
    # multiple mpmd_idxs.
    cross_mpmd_defined = dict[jcore.Var, tuple[int, ...]]()
    sub_by_mpmd_idx = dict[int, dict[jcore.Var, jcore.Var]]()
    new_eqns = []
    gensym = mk_gensym()
    for eqn in jaxpr.eqns:
        if eqn.primitive is gather_multi_p or eqn.primitive is add_multi_p:
            # Cross-MPMD primitives: outputs are defined in all their mpmd_idxs
            new_eqns.append(eqn)
            mpmd_idxs = eqn.params["mpmd_idxs"]
            for outvar in eqn.outvars:
                cross_mpmd_defined[outvar] = mpmd_idxs
            continue
        if eqn.primitive is not task_p:
            # FIXME: maybe apply substitution here
            new_eqns.append(eqn)
            continue

        eqn_mpmd_idx = eqn.params["mpmd_idx"]

        sub = sub_by_mpmd_idx.get(eqn_mpmd_idx, {})
        sub_by_mpmd_idx[eqn_mpmd_idx] = sub

        invars = [
            sub.get(invar, invar) if not isinstance(invar, jcore.Literal) else invar
            for invar in eqn.invars
        ]

        outvars = []
        for outvar in eqn.outvars:
            assert outvar not in invars
            renamed_outvar = outvar
            if (def_mpmd_idx := defined.get(outvar)) is not None:
                if def_mpmd_idx == eqn_mpmd_idx:
                    raise AssertionError(
                        f"Double definition in same mpmd_idx {def_mpmd_idx}"
                    )
                renamed_outvar = gensym(outvar.aval)
                sub[outvar] = renamed_outvar
            defined[renamed_outvar] = eqn_mpmd_idx
            outvars.append(renamed_outvar)

        new_eqns.append(eqn.replace(invars=invars, outvars=outvars))

    defined_in_mpmd_idx_as = dict[jcore.Var, dict[int, jcore.Var]]()
    for d, mpmd_idx in defined.items():
        defined_in_mpmd_idx_as[d] = {mpmd_idx: d}
    for mpmd_idx, sub in sub_by_mpmd_idx.items():
        for orig_var, cpy_var in sub.items():
            _ = defined_in_mpmd_idx_as.get(orig_var, {})
            assert mpmd_idx not in _
            _[mpmd_idx] = cpy_var
    # Register cross-MPMD outputs (defined in all their mpmd_idxs)
    for d, mpmd_idxs in cross_mpmd_defined.items():
        defined_in_mpmd_idx_as[d] = {mpmd_idx: d for mpmd_idx in mpmd_idxs}

    res = []
    for outvar in jaxpr.outvars:
        if isinstance(outvar, jcore.Literal):
            raise NotImplementedError()
        res.append(defined_in_mpmd_idx_as[outvar])

    res_jaxpr, out_mpmd_def = deduplicate_outvars(jaxpr.replace(eqns=new_eqns), res)
    return cjaxpr.replace(jaxpr=res_jaxpr), out_mpmd_def


def maybe_unroll_loop(tasked_jaxpr: jcore.ClosedJaxpr):
    jaxpr: jcore.Jaxpr = tasked_jaxpr.jaxpr
    loop_eqn_idxs = [
        idx for idx, e in enumerate(jaxpr.eqns) if e.primitive is dax_pscan_p
    ]
    if len(loop_eqn_idxs) == 0:
        return tasked_jaxpr
    eqn_idx = get_one_loop_eqn_idx(jaxpr)
    loop_eqns = unroll_loop_eqn(jaxpr.eqns[eqn_idx])
    res = tasked_jaxpr.replace(
        jaxpr=jaxpr.replace(
            eqns=jaxpr.eqns[:eqn_idx] + loop_eqns + jaxpr.eqns[eqn_idx + 1 :]
        )
    )
    return res


def bufferize_recvs(
    jaxpr: jcore.Jaxpr, mpmd_mesh: MpmdMesh, mpmd_idx: int
) -> jcore.Jaxpr:
    """Give every logical (buffer-less) transfer_start recv a destination buffer.

    `to_local_jaxprs` emits recv sides of `transfer_start` in logical form (no
    buffers). This pass rewrites them so each receive gets a destination buffer
    threaded into `recv_done`.
    These buffers are DLPack-backed and `finalize_lifetimes` never deletes a value
    still feeding a transfer_start. Once a received value's last reader has run we
    return it to the free list and reuse it to back a later same-(aval, sharding)
    receive. That reuse bounds how many buffers the single prepended `zeros`
    equation must allocate.
    """
    gensym = mk_gensym()
    last_use = last_used(jaxpr)
    initial_recv_buffers = []
    initial_recv_buffer_shardings = []
    free_recv_buffers: defaultdict[
        tuple[jcore.AbstractValue, jax.sharding.NamedSharding], deque[jcore.Var]
    ] = defaultdict(deque)
    recv_buffer_sharding_by_var = dict[jcore.Var, jax.sharding.NamedSharding]()
    transfer_start_by_token: dict[
        jcore.Var,
        tuple[jcore.Var, list[jcore.Var], tuple[jax.sharding.NamedSharding, ...]],
    ] = {}

    def recv_buffer_needs(eqn: jcore.JaxprEqn):
        local_shardings = tuple(eqn.params["recv_local_shardings"])
        assert all(isinstance(s, jax.sharding.NamedSharding) for s in local_shardings)
        assert all(
            mpmd_idx
            in _require_mpmd_indices(
                mpmd_mesh, sharding.mesh, name="recv local_sharding"
            )
            for sharding in local_shardings
        ), (local_shardings, mpmd_idx)
        out_avals = tuple(eqn.params["out_avals"])
        assert len(out_avals) == len(local_shardings), (out_avals, local_shardings)
        return tuple(zip(out_avals, local_shardings, strict=True))

    def acquire_recv_buffer(
        need: tuple[jcore.AbstractValue, jax.sharding.NamedSharding],
    ):
        if free_recv_buffers[need]:
            return free_recv_buffers[need].popleft()
        aval, sharding = need
        var = gensym(aval)
        recv_buffer_sharding_by_var[var] = sharding
        initial_recv_buffers.append(var)
        initial_recv_buffer_shardings.append(sharding)
        return var

    remaining_recv_needs = Counter(
        need
        for eqn in jaxpr.eqns
        if eqn.primitive is transfer_start_p
        for need in recv_buffer_needs(eqn)
    )

    def release_recv_buffer(var: jcore.Var):
        sharding = recv_buffer_sharding_by_var.get(var)
        if sharding is None:
            return

        need = (var.aval, sharding)
        if remaining_recv_needs[need] <= len(free_recv_buffers[need]):
            return

        fenced_var = gensym(var.aval)
        fence_eqn = new_primitive_eqn(reuse_fence_p, [var], [fenced_var])
        new_eqns.append(fence_eqn)
        recv_buffer_sharding_by_var[fenced_var] = sharding
        free_recv_buffers[need].append(fenced_var)

    def release_last_used_recv_buffers(invars: Sequence[jcore.Atom], eqn_idx: int):
        seen = set[jcore.Var]()
        for invar in nonlit(invars):
            if invar in seen:
                continue
            seen.add(invar)
            if last_use.get(invar) == eqn_idx:
                release_recv_buffer(invar)

    def runtime_recv_params(eqn: jcore.JaxprEqn):
        params = dict(eqn.params)
        params.pop("out_avals", None)
        return params

    new_eqns = list[jcore.JaxprEqn]()
    for eqn_idx, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive is transfer_start_p:
            needs = recv_buffer_needs(eqn)
            remaining_recv_needs.subtract(needs)
            if len(needs) == 0:
                new_eqns.append(eqn)
                release_last_used_recv_buffers(eqn.invars, eqn_idx)
                continue
            _out_avals, buffer_shardings = jc.unzip2(needs)
            recv_buf_vars = [acquire_recv_buffer(need) for need in needs]
            params = runtime_recv_params(eqn)
            send_count = len(eqn.params["send_local_shardings"])
            transfer_start_eqn = new_primitive_eqn(
                eqn.primitive, [*eqn.invars[:send_count], *recv_buf_vars], **params
            )
            recv_tok, *recv_buf2_vars = transfer_start_eqn.outvars
            original_recv_tok, *_ = eqn.outvars
            assert isinstance(original_recv_tok, jcore.Var), eqn
            assert original_recv_tok not in transfer_start_by_token, original_recv_tok
            transfer_start_by_token[original_recv_tok] = (
                recv_tok,
                recv_buf2_vars,
                buffer_shardings,
            )
            new_eqns.append(transfer_start_eqn)
        elif eqn.primitive is recv_done_p:
            original_recv_tok, *_ = eqn.invars
            assert isinstance(original_recv_tok, jcore.Var), eqn
            recv_tok, recv_buf2_vars, buffer_shardings = transfer_start_by_token[
                original_recv_tok
            ]
            params = runtime_recv_params(eqn)
            recv_done_eqn = new_primitive_eqn(
                recv_done_p, [recv_tok, *recv_buf2_vars], eqn.outvars, **params
            )
            new_eqns.append(recv_done_eqn)
            for outvar, buffer_sharding in zip(
                eqn.outvars, buffer_shardings, strict=True
            ):
                recv_buffer_sharding_by_var[outvar] = buffer_sharding
        else:
            new_eqns.append(eqn)

        release_last_used_recv_buffers(eqn.invars, eqn_idx)

    if not initial_recv_buffers:
        return jaxpr

    zeros_params = {
        "shape_and_dtype": [(v.aval.shape, v.aval.dtype) for v in initial_recv_buffers],
        "shardings": tuple(initial_recv_buffer_shardings),
        "out_avals": tuple(v.aval for v in initial_recv_buffers),
    }
    zeros_eqn = new_primitive_eqn(zeros_p, [], initial_recv_buffers, **zeros_params)
    new_eqns = [zeros_eqn, *new_eqns]

    return jaxpr.replace(
        eqns=new_eqns, effects=jcore.join_effects(*(eqn.effects for eqn in new_eqns))
    )


def _transfer_eqn_for_token(
    jaxpr: jcore.Jaxpr, defs: Mapping[jcore.Var, DefSite], token: jcore.Atom
) -> jcore.JaxprEqn:
    if not isinstance(token, jcore.Var):
        raise ValueError(f"transfer_done token is not a transfer token: {token}")
    transfer_def = defs.get(token)
    if transfer_def is None:
        raise ValueError(f"transfer_done token is not a transfer token: {token}")
    transfer_eqn = jaxpr.eqns[transfer_def.eqn_idx]
    if transfer_eqn.primitive is not transfer_p or transfer_def.outvar_idx != 0:
        raise ValueError(f"transfer_done token is not a transfer token: {token}")
    return transfer_eqn


def check_global_jaxpr(jaxpr: jcore.Jaxpr, *, check_transfer_done_fifo: bool = False):
    defs, refs = defs_and_uses(jaxpr.eqns)

    def is_raw_transfer_outvar(var: jcore.Var) -> bool:
        var_def = defs.get(var)
        if var_def is None:
            return False
        def_eqn = jaxpr.eqns[var_def.eqn_idx]
        return def_eqn.primitive is transfer_p and var_def.outvar_idx > 0

    def validate_transfer_done(eqn: jcore.JaxprEqn) -> jcore.JaxprEqn:
        transfer_token, *transfer_outvars = eqn.invars
        transfer_eqn = _transfer_eqn_for_token(jaxpr, defs, transfer_token)
        if list(transfer_outvars) != transfer_eqn.outvars[1:]:
            raise ValueError(f"transfer_done does not match transfer outputs: {eqn}")
        return transfer_eqn

    for var, use_sites in refs.items():
        if is_raw_transfer_outvar(var) and any(
            jaxpr.eqns[use.eqn_idx].primitive is not transfer_done_p
            for use in use_sites
        ):
            raise ValueError("transfer output used without transfer_done")

    for outvar in nonlit(jaxpr.outvars):
        if is_raw_transfer_outvar(outvar):
            raise ValueError("transfer output used without transfer_done")

    transfer_tokens_by_channel = defaultdict[
        tuple[jax.sharding.Mesh, jax.sharding.Mesh], list[jcore.Var]
    ](list)
    done_tokens_by_channel = defaultdict[
        tuple[jax.sharding.Mesh, jax.sharding.Mesh], list[jcore.Var]
    ](list)
    for eqn in jaxpr.eqns:
        if eqn.primitive is transfer_p:
            token, *_ = eqn.outvars
            assert isinstance(token, jcore.Var), eqn
            for channel in _unique_transfer_channels(eqn):
                transfer_tokens_by_channel[channel].append(token)
        elif eqn.primitive is transfer_done_p:
            transfer_eqn = validate_transfer_done(eqn)
            token, *_ = eqn.invars
            assert isinstance(token, jcore.Var), eqn
            for channel in _unique_transfer_channels(transfer_eqn):
                done_tokens_by_channel[channel].append(token)

    for channel, transfer_tokens in transfer_tokens_by_channel.items():
        done_tokens = done_tokens_by_channel[channel]
        if len(done_tokens) < len(transfer_tokens):
            raise ValueError(
                "global jaxpr expects every transfer to have a matching transfer_done"
            )
        if check_transfer_done_fifo and done_tokens != transfer_tokens:
            raise ValueError("transfer_done must be FIFO for each transfer channel")


@dataclasses.dataclass(frozen=True)
class LocalJaxpr:
    closed_jaxpr: jcore.ClosedJaxpr
    global_invar_indices: tuple[int, ...]
    global_outvar_indices: tuple[int, ...]


def to_local_jaxprs(
    tasked_jaxpr: jcore.ClosedJaxpr,
    mpmd_mesh: MpmdMesh,
    *,
    check_transfer_done_fifo: bool = False,
) -> list[LocalJaxpr]:
    mesh_eqns = list[tuple[jax.sharding.Mesh, jcore.JaxprEqn]]()
    # A value can have local handles on multiple meshes. The inner dict is an
    # ordered set.
    local_meshes_by_var = defaultdict[jcore.Var, dict[jax.sharding.Mesh, None]](dict)
    check_global_jaxpr(
        tasked_jaxpr.jaxpr, check_transfer_done_fifo=check_transfer_done_fifo
    )
    defs, _ = defs_and_uses(tasked_jaxpr.eqns)
    global_inputs = set(tasked_jaxpr.jaxpr.constvars) | set(tasked_jaxpr.jaxpr.invars)

    def append_local_eqn(mesh: jax.sharding.Mesh, eqn: jcore.JaxprEqn):
        mesh_eqns.append((mesh, eqn))
        # A global delete must be lowered into every local jaxpr that holds a
        # handle for that value. Eqn outputs create handles locally, while
        # global inputs are handles already present in the local jaxpr.
        for outvar in eqn.outvars:
            if not isinstance(outvar, jcore.DropVar):
                local_meshes_by_var[outvar][mesh] = None
        for invar in nonlit(eqn.invars):
            if invar in global_inputs:
                local_meshes_by_var[invar][mesh] = None

    for eqn in tasked_jaxpr.eqns:
        if eqn.primitive is task_p:
            _, task_mesh = _resolve_placement(
                mpmd_mesh, eqn.params["mpmd_idx"], name="task mpmd_idx"
            )
            append_local_eqn(task_mesh, eqn)
        elif eqn.primitive is transfer_p:
            transfer_token, *transfer_data_outvars = eqn.outvars
            assert isinstance(transfer_token, jcore.Var), eqn
            send_by_mpmd_idx = defaultdict[int, list[int]](list)
            recv_by_mpmd_idx = defaultdict[int, list[int]](list)
            for idx, (src_sharding, tgt_sharding) in enumerate(
                zip(
                    eqn.params["src_shardings"],
                    eqn.params["tgt_shardings"],
                    strict=True,
                )
            ):
                for mpmd_idx in _require_mpmd_indices(
                    mpmd_mesh, src_sharding.mesh, name="src_sharding"
                ):
                    send_by_mpmd_idx[mpmd_idx].append(idx)
                for mpmd_idx in _require_mpmd_indices(
                    mpmd_mesh, tgt_sharding.mesh, name="tgt_sharding"
                ):
                    recv_by_mpmd_idx[mpmd_idx].append(idx)

            for mpmd_idx in sorted(send_by_mpmd_idx.keys() | recv_by_mpmd_idx.keys()):
                send_indices = send_by_mpmd_idx[mpmd_idx]
                recv_indices = recv_by_mpmd_idx[mpmd_idx]
                transfer_start_eqn = new_primitive_eqn(
                    transfer_start_p,
                    [eqn.invars[idx] for idx in send_indices],
                    [
                        transfer_token,
                        *(transfer_data_outvars[idx] for idx in recv_indices),
                    ],
                    send_remote_shardings=tuple(
                        eqn.params["tgt_shardings"][idx] for idx in send_indices
                    ),
                    send_local_shardings=tuple(
                        eqn.params["src_shardings"][idx] for idx in send_indices
                    ),
                    recv_remote_shardings=tuple(
                        eqn.params["src_shardings"][idx] for idx in recv_indices
                    ),
                    recv_local_shardings=tuple(
                        eqn.params["tgt_shardings"][idx] for idx in recv_indices
                    ),
                    out_avals=tuple(
                        transfer_data_outvars[idx].aval for idx in recv_indices
                    ),
                )
                append_local_eqn(mpmd_mesh.unstack[mpmd_idx], transfer_start_eqn)
        elif eqn.primitive is transfer_done_p:
            transfer_eqn = _transfer_eqn_for_token(
                tasked_jaxpr.jaxpr, defs, eqn.invars[0]
            )
            transfer_token, *transfer_outvars = eqn.invars
            done_by_mpmd_idx = defaultdict[int, list[int]](list)
            for idx, tgt_sharding in enumerate(transfer_eqn.params["tgt_shardings"]):
                for mpmd_idx in _require_mpmd_indices(
                    mpmd_mesh, tgt_sharding.mesh, name="transfer_done tgt_sharding"
                ):
                    done_by_mpmd_idx[mpmd_idx].append(idx)

            for mpmd_idx, indices in sorted(done_by_mpmd_idx.items()):
                recv_done_eqn = new_primitive_eqn(
                    recv_done_p,
                    [transfer_token, *(transfer_outvars[idx] for idx in indices)],
                    [eqn.outvars[idx] for idx in indices],
                )
                append_local_eqn(mpmd_mesh.unstack[mpmd_idx], recv_done_eqn)
        elif eqn.primitive is stack_p:
            stack_mpmd_mesh = eqn.params["mpmd_mesh"]
            (in_shardings, out_sharding, out_shape, expand_inputs) = (
                array_ops.stack_shape_and_sharding(
                    tuple(invar.aval.shape for invar in eqn.invars),
                    eqn.params["in_shardings"],
                    mpmd_mesh=stack_mpmd_mesh,
                    axis=eqn.params["axis"],
                )
            )
            _require_mpmd_indices(
                mpmd_mesh, out_sharding.mesh, name="stack out_sharding"
            )
            for invar, in_sharding, expand in zip(
                eqn.invars, in_shardings, expand_inputs, strict=True
            ):
                mesh = in_sharding.mesh
                _require_mpmd_indices(mpmd_mesh, mesh, name="stack in_sharding")
                local_stack_params: LocalStackParams = {
                    "in_sharding": in_sharding,
                    "out_sharding": out_sharding,
                    "out_shape": out_shape,
                    "expand": expand,
                    "axis": eqn.params["axis"],
                }
                local_stack_eqn = new_primitive_eqn(
                    local_stack_p, [invar], eqn.outvars, **local_stack_params
                )
                append_local_eqn(mesh, local_stack_eqn)
        elif eqn.primitive is slice_p:
            (invar,) = eqn.invars
            in_sharding = eqn.params["in_sharding"]
            _require_mpmd_indices(mpmd_mesh, in_sharding.mesh, name="slice in_sharding")
            _, out_shardings = array_ops.slice_shape_and_shardings(
                invar.aval.shape,
                in_sharding,
                eqn.params["groups"],
                mpmd_mesh=eqn.params["mpmd_mesh"],
            )
            for outvar, out_sharding in zip(eqn.outvars, out_shardings, strict=True):
                mesh = out_sharding.mesh
                _require_mpmd_indices(mpmd_mesh, mesh, name="slice out_sharding")
                local_slice_params: LocalSliceParams = {
                    "in_sharding": in_sharding,
                    "out_shardings": (out_sharding,),
                }
                local_slice_eqn = new_primitive_eqn(
                    local_slice_p, [invar], [outvar], **local_slice_params
                )
                append_local_eqn(mesh, local_slice_eqn)
        elif eqn.primitive is add_multi_p:
            # NOTE: all shardings are enforced to be the same
            # by `reconcile_shardings`
            _ = eqn.params["in_shardings"]

            for invar, mpmd_idx in zip(
                eqn.invars, eqn.params["mpmd_idxs"], strict=True
            ):
                mesh = mpmd_mesh.unstack[mpmd_idx]
                all_reduce_eqn = new_primitive_eqn(
                    all_reduce_p,
                    [invar],
                    eqn.outvars,
                    mpmd_idxs=list(eqn.params["mpmd_idxs"]),
                    out_spec=eqn.params["out_shardings"][0].spec,
                    donated=None,
                )
                append_local_eqn(mesh, all_reduce_eqn)
        elif eqn.primitive is gather_multi_p:
            # NOTE: all shardings are enforced to be the same
            # by `reconcile_shardings`
            _ = eqn.params["in_shardings"]

            for invar, mpmd_idx in zip(
                eqn.invars, eqn.params["mpmd_idxs"], strict=True
            ):
                mesh = mpmd_mesh.unstack[mpmd_idx]
                all_gather_eqn = new_primitive_eqn(
                    all_gather_p,
                    [invar],
                    eqn.outvars,
                    axis=eqn.params["axis"],
                    mpmd_idxs=list(eqn.params["mpmd_idxs"]),
                    out_spec=eqn.params["out_shardings"][0].spec,
                    donated=None,
                    restore_order_perm=eqn.params["restore_order_perm"],
                )
                append_local_eqn(mesh, all_gather_eqn)
        elif eqn.primitive is delete_p:
            delete_invars_by_mesh = defaultdict[jax.sharding.Mesh, list[jcore.Var]](
                list
            )
            for invar in nonlit(eqn.invars):
                if invar not in local_meshes_by_var:
                    raise ValueError(
                        "delete input must be present in a local jaxpr before lowering"
                    )
                for mesh in local_meshes_by_var[invar]:
                    delete_invars_by_mesh[mesh].append(invar)

            for mesh, delete_invars in delete_invars_by_mesh.items():
                delete_eqn = new_primitive_eqn(
                    delete_p,
                    delete_invars,
                    [jcore.DropVar(invar.aval) for invar in delete_invars],
                )
                append_local_eqn(mesh, delete_eqn)
        else:
            raise NotImplementedError(f"{eqn.primitive}")

    eqn_by_mpmd_idx = [[] for _ in range(mpmd_mesh.mpmd_dim)]
    for mesh, eqn in mesh_eqns:
        indices = _require_mpmd_indices(mpmd_mesh, mesh, name="local jaxpr mesh")
        for mpmd_idx in indices:
            eqn_by_mpmd_idx[mpmd_idx].append(eqn)

    assert len(tasked_jaxpr.jaxpr.constvars) == len(tasked_jaxpr.consts)
    constvar_idx = {v: idx for idx, v in enumerate(tasked_jaxpr.jaxpr.constvars)}
    invar_idx = {v: idx for idx, v in enumerate(tasked_jaxpr.jaxpr.invars)}
    outvar_idx = {
        v: idx
        for idx, v in enumerate(tasked_jaxpr.jaxpr.outvars)
        if not isinstance(v, jcore.ClosedJaxpr)
    }
    outvar_set = set(nonlit(tasked_jaxpr.jaxpr.outvars))
    local_jaxprs = list[LocalJaxpr]()
    for mpmd_idx, eqns in enumerate(eqn_by_mpmd_idx):
        free, defined = eqns_free_vars(eqns, ordered=True)
        local_constvars = [v for v in tasked_jaxpr.jaxpr.constvars if v in free]
        local_invars = [v for v in free if v not in constvar_idx]
        unknown_invars = [v for v in local_invars if v not in invar_idx]
        assert not unknown_invars, (mpmd_idx, unknown_invars)
        local_invars = sorted(local_invars, key=lambda v: invar_idx[v])
        local_outvars = sorted(
            (v for v in defined if v in outvar_set), key=lambda v: outvar_idx[v]
        )
        jaxpr = jcore.Jaxpr(
            constvars=local_constvars,
            invars=local_invars,
            # TODO: outvars don't have literals now, but they might in the future.
            outvars=local_outvars,
            eqns=eqns,
            effects=jcore.join_effects(*(eqn.effects for eqn in eqns)),
        )
        check_jaxpr(jaxpr)
        if env_vars.jaxpp_reuse_recv_buffers.value:
            jaxpr = bufferize_recvs(jaxpr, mpmd_mesh, mpmd_idx)
            check_jaxpr(jaxpr)
        local_const_indices = tuple(
            constvar_idx[constvar] for constvar in local_constvars
        )
        local_jaxprs.append(
            LocalJaxpr(
                closed_jaxpr=jcore.ClosedJaxpr(
                    jaxpr, tuple(tasked_jaxpr.consts[i] for i in local_const_indices)
                ),
                global_invar_indices=tuple(invar_idx[invar] for invar in jaxpr.invars),
                global_outvar_indices=tuple(
                    outvar_idx[outvar] for outvar in jaxpr.outvars
                ),
            )
        )

    return local_jaxprs


def extract_params(params, n_consts, replicated_sharding):
    donated_invars = ((False,) * n_consts) + params["donated_invars"]
    flat_in_shardings = ((replicated_sharding,) * n_consts) + params["in_shardings"]
    flat_out_shardings = params["out_shardings"]
    flat_in_layouts = ((None,) * n_consts) + params["in_layouts"]
    flat_out_layouts = params["out_layouts"]
    return (
        donated_invars,
        flat_in_shardings,
        flat_out_shardings,
        flat_in_layouts,
        flat_out_layouts,
    )


@jc.weakref_lru_cache
def disable_prevent_cse(cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr):
    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        jaxpr = cjaxpr.jaxpr
    else:
        jaxpr = cjaxpr

    new_eqns = []
    for eqn in jaxpr.eqns:
        params_update = {}
        if eqn.primitive is jc.remat_p:
            params_update["prevent_cse"] = False

        for k, v in eqn.params.items():
            if isinstance(v, (jcore.ClosedJaxpr, jcore.Jaxpr)):
                params_update[k] = disable_prevent_cse(v)

        new_eqns.append(eqn.replace(params=eqn.params | params_update))

    new_jaxpr = jaxpr.replace(eqns=new_eqns)

    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        return cjaxpr.replace(jaxpr=new_jaxpr)

    return new_jaxpr


@jc.weakref_lru_cache
def preprocess_jaxpr(
    cjaxpr: jcore.ClosedJaxpr,
) -> tuple[jcore.ClosedJaxpr, Sequence[bool]]:
    if env_vars.jaxpp_disable_prevent_cse.value:
        cjaxpr = disable_prevent_cse(cjaxpr)

    jaxpr_with_consts = jc.convert_constvars_jaxpr(cjaxpr.jaxpr)
    licm_jaxpr = loop_passes(jaxpr_with_consts)
    dced_jaxpr, used_inputs = pe.dce_jaxpr(
        licm_jaxpr, used_outputs=[True] * len(licm_jaxpr.outvars)
    )
    jaxpr = dced_jaxpr.replace(
        invars=licm_jaxpr.invars, debug_info=licm_jaxpr.debug_info
    )
    return jc.close_jaxpr(jaxpr), tuple(used_inputs)


@dataclasses.dataclass
class Strategy(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        closed_jaxpr: jcore.ClosedJaxpr,
        in_used: Sequence[bool],
        flat_in_shardings,
        out_tree,
        flat_out_shardings,
        mpmd_dim: int,
        name: str,
    ): ...


@dataclasses.dataclass(eq=False, kw_only=True)
class FunctionWithLoop(Strategy):
    def __call__(
        self,
        closed_jaxpr: jcore.ClosedJaxpr,
        in_used: Sequence[bool],
        flat_in_shardings,
        out_tree,
        flat_out_shardings,
        mpmd_dim: int,
        name: str,
    ):
        closed_jaxpr, in_mpmd_defs, out_mpmd_defs = wrap_into_tasks(
            closed_jaxpr, in_used, mpmd_dim
        )
        return closed_jaxpr, in_mpmd_defs, out_mpmd_defs


@dataclasses.dataclass(eq=False, kw_only=True)
class FunctionWithYield(Strategy):
    target_num_stages: int | None = None

    def __call__(
        self,
        closed_jaxpr: jcore.ClosedJaxpr,
        in_used: Sequence[bool],
        flat_in_shardings,
        out_tree,
        flat_out_shardings,
        mpmd_dim: int,
        name: str,
    ):
        def get_mpmd_idx(stage_id: int) -> MpmdIdx:
            return MpmdIdx(stage_id % mpmd_dim)

        jaxpr = cluster_jaxpr(
            closed_jaxpr.jaxpr,
            self.target_num_stages,
            is_partial_bwd=False,
            get_mpmd_idx=get_mpmd_idx,
            is_loop=False,
        )
        jaxpr, in_mpmd_refs, out_mpmd_defs = compute_loop_placement(
            jaxpr, n_consts=0, is_loop=False
        )
        closed_jaxpr = jcore.ClosedJaxpr(jaxpr, closed_jaxpr.consts)
        return closed_jaxpr, in_mpmd_refs, out_mpmd_defs


@dataclasses.dataclass(eq=False, kw_only=True)
class TraceableFunction:
    fun: Callable
    mpmd_mesh: MpmdMesh
    pjit_info: Any
    strategy: Strategy
    _compiled: Callable | None = None

    def compile(self, *args, **kwargs):
        if self._compiled is None:
            self._compiled = (
                self.trace_and_place(*args, **kwargs)
                .infer_intermediate_shardings()
                .mpmdify()
            )
        return self._compiled

    def __call__(self, *args, **kwargs):
        if self._compiled is None:
            self._compiled = self.compile(*args, **kwargs)
        return self._compiled(*args, **kwargs)

    def trace_and_place(self, *args, **kwargs):
        with (
            log_elapsed_time("jaxpr/tracing"),
            self.mpmd_mesh.lowering_mesh(),
            yield_scope(isinstance(self.strategy, FunctionWithYield)),
        ):

            def with_lowering_mesh(a):
                # TODO: does this work for hijax?
                aval = jcore.shaped_abstractify(a)

                if aval.sharding is None:
                    return aval

                if not isinstance(aval.sharding, jax.sharding.NamedSharding):
                    return aval

                return aval.update(
                    sharding=update_named_sharding(
                        aval.sharding, mesh=self.mpmd_mesh.lowering_mesh().abstract_mesh
                    )
                )

            raised_args, raised_kwargs = jc.map_dynamic_args(
                args,
                kwargs,
                self.pjit_info.static_argnums,
                self.pjit_info.static_argnames,
                with_lowering_mesh,
            )

            p, _ = jc._infer_params(
                self.fun, self.pjit_info, raised_args, raised_kwargs
            )

        # JAX may also expose `p.consts`; those are pjit extra args prepended
        # to execution. JaxPP converts this ClosedJaxpr's constvars into
        # ordinary leading invars, so these const payloads are the ones tracked
        # as `GlobalMpmdFunction.consts`.
        jaxpr_consts = p.params["jaxpr"].consts
        closed_jaxpr, in_used = preprocess_jaxpr(p.params["jaxpr"])
        replicated_sharding = jax.sharding.NamedSharding(
            self.mpmd_mesh.lowering_mesh(), jax.sharding.PartitionSpec()
        )

        (
            flat_in_donated,
            flat_in_shardings,
            flat_out_shardings,
            flat_in_layouts,
            flat_out_layouts,
        ) = extract_params(
            p.params, len(jaxpr_consts), replicated_sharding=replicated_sharding
        )

        closed_jaxpr = closed_jaxpr.map_jaxpr(outvar_normalization)

        closed_jaxpr, in_mpmd_defs, out_mpmd_defs = self.strategy(
            closed_jaxpr,
            in_used,
            flat_in_shardings,
            p.out_tree,
            flat_out_shardings,
            mpmd_dim=self.mpmd_mesh.mpmd_dim,
            name=p.params["name"],
        )

        return GlobalMpmdFunction(
            closed_jaxpr=closed_jaxpr,
            consts=jaxpr_consts,
            mpmd_mesh=self.mpmd_mesh,
            pjit_info=self.pjit_info,
            in_info=InInfo(
                in_used=in_used,
                in_donated=flat_in_donated,
                in_tree=p.in_tree,
                out_tree=p.out_tree,
                in_avals=closed_jaxpr.in_avals,
                out_avals=closed_jaxpr.out_avals,
                in_shardings=flat_in_shardings,
                out_shardings=flat_out_shardings,
                in_layouts=flat_in_layouts,
                out_layouts=flat_out_layouts,
                in_mpmd_defs=in_mpmd_defs,
                out_mpmd_defs=out_mpmd_defs,
            ),
            name=p.params["name"],
        )


def bind_meshes(cjaxpr: jcore.ClosedJaxpr, mpmd_mesh: MpmdMesh) -> jcore.ClosedJaxpr:
    def bind_sharding_to_mesh(sharding, *, name: str):
        mpmd_idx = _require_single_mpmd_index(mpmd_mesh, sharding.mesh, name=name)
        (bound,) = updated_named_sharding_mesh((sharding,), mpmd_mesh.unstack[mpmd_idx])
        return bound

    new_eqns = []
    for eqn in cjaxpr.eqns:
        if eqn.primitive is task_p:
            _, new_mesh = _resolve_placement(
                mpmd_mesh, eqn.params["mpmd_idx"], name="task mpmd_idx"
            )
            new_eqns.append(_bind_task_eqn_to_mesh(eqn, new_mesh))
        elif eqn.primitive is dax_pscan_p:
            loop_cjaxpr = bind_meshes(eqn.params["jaxpr"], mpmd_mesh)
            new_eqns.append(
                eqn.replace(
                    params=eqn.params | {"jaxpr": loop_cjaxpr},
                    effects=loop_cjaxpr.effects,
                )
            )
        elif eqn.primitive is transfer_p:
            param_update = {
                "src_shardings": tuple(
                    bind_sharding_to_mesh(s, name="src_sharding")
                    for s in eqn.params["src_shardings"]
                ),
                "tgt_shardings": tuple(
                    bind_sharding_to_mesh(s, name="tgt_sharding")
                    for s in eqn.params["tgt_shardings"]
                ),
            }
            new_eqns.append(eqn.replace(params=eqn.params | param_update))
        # TODO update shardings for add_multi_p too
        #  Note that add_multi_p does not use in/out shardings.
        #  They are inferred and tracked just to make sure that producers
        #  have the same shardings
        else:
            new_eqns.append(eqn)
    return cjaxpr.map_jaxpr(lambda jaxpr: jaxpr.replace(eqns=new_eqns))


def array_has_sharding(a: jax.Array, sharding: jax.sharding.Sharding) -> bool:
    return jc.shardings_are_equivalent(
        sharding, a.sharding, a.ndim, compare_memkind=False
    )


@dataclasses.dataclass(eq=False, frozen=True, kw_only=True)
class ScalarMpmdFunction:
    global_jaxpr: jcore.ClosedJaxpr
    local_jaxpr: jcore.ClosedJaxpr
    consts: Sequence[Any]
    mpmd_mesh: MpmdMesh
    in_info: InInfo
    name: str

    def __post_init__(self):
        # FIXME: self.global_jaxpr.out_avals is the "unpacked" one
        #   (i.e. if an output is replicated over k ranks, there are two out_avals)
        # assert self.in_info.out_avals == self.global_jaxpr.out_avals
        arg_names = self.global_jaxpr.jaxpr.debug_info.safe_arg_names(
            len(self.global_jaxpr.in_avals)
        )
        for name, in_sharding, aval in zip(
            arg_names,
            self.in_info.in_shardings,
            self.global_jaxpr.in_avals,
            strict=True,
        ):
            try:
                in_sharding.shard_shape(aval.shape)
            except Exception:
                logger.warning(
                    f"Failed shard_shape for '{name}': {aval} with {in_sharding=}"
                )

    @cached_property
    def in_shardings(self):
        res = jax.tree_util.tree_unflatten(
            self.in_info.in_tree,
            [
                MpmdSharding(
                    self.mpmd_mesh,
                    mpmd_idxs,
                    sharding.spec,
                    memory_kind=sharding.memory_kind,
                )
                for mpmd_idxs, sharding in zip(
                    self.in_info.in_mpmd_defs, self.in_info.in_shardings, strict=True
                )
            ][len(self.consts) :],
        )
        return res

    def _maybe_shard_inputs(self, flat_args: list[jax.Array]):
        local_args = []
        arg_names = self.global_jaxpr.jaxpr.debug_info.safe_arg_names(
            len(self.in_info.in_mpmd_defs)
        )
        for arg_idx, (arg, mpmd_idxs) in enumerate(
            zip(
                it.chain(self.consts, flat_args), self.in_info.in_mpmd_defs, strict=True
            )
        ):
            # FIXME: why is mpmd_idxs None in some cases
            if mpmd_idxs is None:
                continue

            if self.mpmd_mesh.my_mpmd_axis_index not in mpmd_idxs:
                continue

            arg_name = arg_names[arg_idx]

            # FIXME: in_shardings offset by consts?
            expected_sharding = self.in_info.in_shardings[arg_idx]
            if isinstance(arg, MpmdArray):
                if not arg.is_partially_addressable:
                    raise ValueError(
                        f"{MpmdArray.__name__} passed as argument {arg_name} is not "
                        "partially addressable"
                    )
                local_arg = arg.to_mpmd_local_array
            else:
                local_arg = arg

            if isinstance(local_arg, jax.Array):
                if (
                    not array_has_sharding(local_arg, expected_sharding)
                    and local_arg._committed
                ):
                    logger.warning(
                        f"Resharding '{arg_name}': {local_arg.shape=} "
                        f"tgt_sharding={expected_sharding.spec}"
                    )
                    # FIXME: this fails whenever `arg` is sharded over the full spmd
                    # mesh because device_put does not support resharding a
                    # non-fully-addressable array (although new versions of JAX will).
                    # Should we require users to pass MpmdArrays only altogether?
                    local_arg = jax.device_put(local_arg, expected_sharding)
            else:
                local_arg = jax.device_put(local_arg, expected_sharding)

            local_args.append(local_arg)

        arg_shape_and_dtype = [(a.aval.shape, a.aval.dtype) for a in local_args]
        assert arg_shape_and_dtype == [
            (_.shape, _.dtype) for _ in self.local_jaxpr.in_avals
        ], (
            f"{len(arg_shape_and_dtype)} {arg_shape_and_dtype=}\n"
            f"{len(self.local_jaxpr.in_avals)} {self.local_jaxpr.in_avals=}"
        )
        return local_args

    def __call__(self, *args, **kwargs):
        assert not env_vars.jaxpp_debug_skip_propagation.value, (
            f"Can't run with {env_vars.jaxpp_debug_skip_propagation.env_key}="
            f"{env_vars.jaxpp_debug_skip_propagation.value}"
        )
        flat_args, in_tree = jax.tree.flatten((args, kwargs))
        assert self.in_info.in_tree == in_tree
        local_args = self._maybe_shard_inputs(flat_args)

        with self.mpmd_mesh:
            outs = jcore.eval_jaxpr(
                self.local_jaxpr.jaxpr, self.local_jaxpr.consts, *local_args
            )

        results = self._check_and_build_outputs(outs)
        return jax.tree.unflatten(self.in_info.out_tree, results)

    def _check_and_build_outputs(self, outs: list[jax.Array]):
        results = []
        local_idx = 0
        for global_idx, mpmd_idxs in enumerate(self.in_info.out_mpmd_defs):
            sharding = self.in_info.out_shardings[global_idx]
            mpmd_sharding = MpmdSharding(
                self.mpmd_mesh,
                mpmd_idxs,
                sharding.spec,
                memory_kind=sharding.memory_kind,
            )
            if self.mpmd_mesh.my_mpmd_axis_index in mpmd_idxs:
                out = MpmdArray(
                    partially_addressable_arrays=[outs[local_idx]],
                    mpmd_sharding=mpmd_sharding,
                )
                expected_aval = self.in_info.out_avals[global_idx]
                assert expected_aval.shape == out.aval.shape and (
                    expected_aval.dtype == out.aval.dtype
                ), f"{expected_aval=} != {out.aval=}"
                local_idx += 1
            else:
                aval = self.in_info.out_avals[global_idx]
                out = MpmdArray(
                    partially_addressable_arrays=[],
                    mpmd_sharding=mpmd_sharding,
                    shape=aval.shape,
                    dtype=aval.dtype,
                )
            results.append(out)
        return results


def print_jaxpr(cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr):
    jaxpr = cjaxpr if isinstance(cjaxpr, jcore.Jaxpr) else cjaxpr.jaxpr
    ctx = jcore.JaxprPpContext()
    settings = jcore.JaxprPpSettings()
    res = ""

    for idx, (name, v) in enumerate(
        zip(jaxpr.debug_info.arg_names, jaxpr.invars, strict=True)
    ):
        res += (
            f"({idx}) {jcore.pp_var(v, ctx).format()}: "
            f"{jcore.pp_aval(v.aval, ctx)} # {name}\n"
        )

    res += jcore.pp_jaxpr(jaxpr, ctx, settings).format()
    res += "\n"

    for idx, (name, v) in enumerate(
        zip(jaxpr.debug_info.result_paths, jaxpr.outvars, strict=True)
    ):
        res += (
            f"({idx}) {jcore.pp_var(v, ctx).format()}: "
            f"{jcore.pp_aval(v.aval, ctx)} # {name}\n"
        )
    return res


def dump_jaxpr(
    cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr,
    *,
    name: str,
    ctx: jcore.JaxprPpContext | None = None,
):
    jaxpr = cjaxpr if isinstance(cjaxpr, jcore.Jaxpr) else cjaxpr.jaxpr
    if env_vars.jaxpp_dump_dir.value != "":
        ctx = ctx or jcore.JaxprPpContext()
        output_dir = Path(env_vars.jaxpp_dump_dir.value)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{name}.{jax.process_index()}.jaxpr.txt"
        output_file.write_text(jcore.pp_toplevel_jaxpr(jaxpr).format())


def common_passes(
    jaxpr: jcore.Jaxpr, donated_invars, *, finalize_lifetime_ops: bool = True
):
    assert len(jaxpr.invars) == len(donated_invars), (
        len(jaxpr.invars),
        len(donated_invars),
    )
    times = infer_times(jaxpr.eqns)
    with_transfers = add_transfer_dones(add_transfers(jaxpr, times))

    if finalize_lifetime_ops:
        with_transfers = finalize_lifetimes(
            with_transfers, donated_invars=donated_invars
        )
    return with_transfers


@dataclasses.dataclass(eq=False, frozen=True, kw_only=True)
class GlobalMpmdFunction:
    closed_jaxpr: jcore.ClosedJaxpr
    consts: Sequence[Any]
    mpmd_mesh: MpmdMesh
    pjit_info: Any
    in_info: InInfo
    name: str
    compiler_options: dict[str, Any] | None = None

    def __post_init__(self):
        # assert self.in_info.out_avals == self.closed_jaxpr.out_avals
        pass  # TODO: fix assertion above triggering because of fixup_duplicates

    def infer_intermediate_shardings(self):
        if getattr(self.mpmd_mesh.jax_mesh, "are_all_axes_explicit", False):
            closed_jaxpr = infer_shardings_explicit(
                self.closed_jaxpr,
                self.mpmd_mesh,
                self.in_info.in_shardings,
                self.in_info.out_shardings,
            )
            return dataclasses.replace(self, closed_jaxpr=closed_jaxpr)

        unknown_shardings = reconcile_shardings(
            self.closed_jaxpr, self.in_info.in_shardings, self.in_info.out_shardings
        )
        if unknown_shardings:
            closed_jaxpr = infer_shardings(
                self.mpmd_mesh.lowering_mesh(),
                self.closed_jaxpr,
                in_shardings=self.in_info.in_shardings,
                out_shardings=self.in_info.out_shardings,
                in_layouts=self.in_info.in_layouts,
                out_layouts=self.in_info.out_layouts,
            )
            return dataclasses.replace(self, closed_jaxpr=closed_jaxpr)
        return dataclasses.replace(self, closed_jaxpr=self.closed_jaxpr)

    @cached_property
    def in_shardings(self):
        res = jax.tree_util.tree_unflatten(
            self.in_info.in_tree,
            [
                MpmdSharding(
                    self.mpmd_mesh,
                    mpmd_idxs,
                    sharding.spec,
                    memory_kind=sharding.memory_kind,
                )
                for mpmd_idxs, sharding in zip(
                    self.in_info.in_mpmd_defs, self.in_info.in_shardings, strict=True
                )
            ][len(self.consts) :],
        )
        return res

    def __call__(self, *args, **kwargs):
        assert not env_vars.jaxpp_debug_skip_propagation.value, (
            f"Can't run with {env_vars.jaxpp_debug_skip_propagation.env_key}="
            f"{env_vars.jaxpp_debug_skip_propagation.value}"
        )
        assert not self.mpmd_mesh.jax_mesh.is_multi_process
        flat_args, in_tree = jax.tree.flatten((args, kwargs))
        assert self.in_info.in_tree == in_tree, list(
            jc.equality_errors_pytreedef(self.in_info.in_tree, in_tree)
        )

        n_consts = len(self.consts)
        for i, arg in enumerate(flat_args):
            expected_mpmd_idx = set(self.in_info.in_mpmd_defs[n_consts + i])
            if len(expected_mpmd_idx) == 0:
                continue

            if isinstance(arg, jax.Array):
                mpmd_idx = (
                    self.mpmd_mesh.mpmd_idx_for_mesh.get(arg.sharding.mesh)
                    if isinstance(arg.sharding, jax.sharding.NamedSharding)
                    else None
                )
                if (
                    mpmd_idx is None
                    or mpmd_idx not in expected_mpmd_idx
                    or len(expected_mpmd_idx) > 1
                ):
                    values = {}
                    try:
                        expected_mpmd_idx.remove(mpmd_idx)
                        values[mpmd_idx] = arg
                    except KeyError:
                        pass

                    in_sharding = self.in_info.in_shardings[n_consts + i]
                    for mpmd_idx in expected_mpmd_idx:
                        (sh,) = updated_named_sharding_mesh(
                            (in_sharding,), self.mpmd_mesh.unstack[mpmd_idx]
                        )
                        values[mpmd_idx] = jax.device_put(arg, sh)

                    # TODO(fixup_multidefs): instead of eval_jaxpr on `MpmdArray`s
                    #   "deduplicate" jaxpr's invars in fixup_multidefs
                    flat_args[i] = MpmdArray(
                        values.values(),
                        mpmd_sharding=MpmdSharding(
                            self.mpmd_mesh,
                            expected_mpmd_idx,
                            in_sharding.spec,
                            memory_kind=in_sharding.memory_kind,
                        ),
                    )

            elif isinstance(arg, MpmdArray):
                assert set(arg._mpmd_idxs) == expected_mpmd_idx, (
                    arg._mpmd_idxs,
                    expected_mpmd_idx,
                )
            else:
                pass

        with self.mpmd_mesh:
            outputs = jcore.eval_jaxpr(
                self.closed_jaxpr.jaxpr,
                self.closed_jaxpr.consts,
                *self.consts,
                *flat_args,
            )

        i = 0
        actual_outputs = list[MpmdArray]()
        for out_idx, out_mpmd_def in enumerate(self.in_info.out_mpmd_defs):
            if isinstance(outputs[i], MpmdArray):
                first = outputs[i]
                for output in outputs[i : i + len(out_mpmd_def)]:
                    assert output is first, (output, first)
                actual_outputs.append(first)
            else:
                sharding = self.in_info.out_shardings[out_idx]
                actual_outputs.append(
                    MpmdArray(
                        outputs[i : i + len(out_mpmd_def)],
                        mpmd_sharding=MpmdSharding(
                            self.mpmd_mesh,
                            out_mpmd_def,
                            sharding.spec,
                            memory_kind=sharding.memory_kind,
                        ),
                    )
                )
            i += len(out_mpmd_def)

        return jax.tree.unflatten(self.in_info.out_tree, actual_outputs)

    def mpmdify(self):
        closed_jaxpr = self.closed_jaxpr
        out_mpmd_defs = self.in_info.out_mpmd_defs

        closed_jaxpr = bind_meshes(closed_jaxpr, self.mpmd_mesh)
        if env_vars.jaxpp_enable_task_jaxpr_deduplication.value:
            closed_jaxpr = deduplicate_task_jaxprs(closed_jaxpr)

        pp_ctx = jcore.JaxprPpContext()
        dump_jaxpr(closed_jaxpr, name=f"{self.name}.global", ctx=pp_ctx)

        with_transfers = maybe_unroll_loop(closed_jaxpr)
        # TODO: move fixup_multidefs right after coarsening.
        #  This is slightly challenging as we need to deduplicate_invars
        #  for the loop body too
        with_transfers, out_placement = fixup_multidefs(with_transfers)
        # TODO: check also `in_mpmd_defs` similarly to out_placement
        assert list(out_placement) == list(out_mpmd_defs), (
            list(out_placement),
            out_mpmd_defs,
        )
        should_lower_to_local = (
            env_vars.jaxpp_debug_force_mpmdify.value
            or self.mpmd_mesh.jax_mesh.is_multi_process
        )
        with_transfers = with_transfers.map_jaxpr(
            partial(
                common_passes,
                donated_invars=self.in_info.in_donated,
                finalize_lifetime_ops=not should_lower_to_local,
            )
        )
        dump_jaxpr(with_transfers, name=f"{self.name}.global.unrolled", ctx=pp_ctx)

        if not should_lower_to_local:
            # local_jaxprs = to_local_jaxprs(with_transfers, self.mpmd_mesh)
            return dataclasses.replace(
                self,
                closed_jaxpr=with_transfers,
                in_info=dataclasses.replace(self.in_info, out_mpmd_defs=out_placement),
            )

        local_jaxprs = to_local_jaxprs(with_transfers, self.mpmd_mesh)
        local_closed_jaxprs = [
            local_jaxpr.closed_jaxpr.map_jaxpr(
                partial(
                    finalize_lifetimes,
                    donated_invars=tuple(
                        self.in_info.in_donated[global_invar_idx]
                        for global_invar_idx in local_jaxpr.global_invar_indices
                    ),
                )
            )
            for local_jaxpr in local_jaxprs
        ]

        dump_jaxpr(
            local_closed_jaxprs[self.mpmd_mesh.my_mpmd_axis_index],
            name=f"{self.name}.local",
            ctx=pp_ctx,
        )

        return ScalarMpmdFunction(
            global_jaxpr=with_transfers,
            local_jaxpr=local_closed_jaxprs[self.mpmd_mesh.my_mpmd_axis_index],
            consts=self.consts,
            mpmd_mesh=self.mpmd_mesh,
            in_info=dataclasses.replace(self.in_info, out_mpmd_defs=out_placement),
            name=self.name,
        )


def _mpmd_jit(
    fun: Callable,
    mpmd_mesh: MpmdMesh,
    *,
    strategy,
    in_shardings=None,
    out_shardings=None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> TraceableFunction:
    pjit_info = jc._parse_jit_arguments(
        fun=fun,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        device=None,
        backend=None,
        keep_unused=True,
        inline=False,
        compiler_options=compiler_options,
        use_resource_env=True,  # FIXME
    )
    return TraceableFunction(
        fun=fun, mpmd_mesh=mpmd_mesh, pjit_info=pjit_info, strategy=strategy
    )


def mpmd_jit_with_loop(
    fun: Callable,
    mpmd_mesh: MpmdMesh,
    *,
    in_shardings=None,
    out_shardings=None,
    in_specs=None,
    out_specs=None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> TraceableFunction:
    if in_specs is not None and in_shardings is not None:
        raise ValueError("Can't pass both in_shardings and in_specs")

    if out_specs is not None and out_shardings is not None:
        raise ValueError("Can't pass both out_shardings and out_specs")

    return _mpmd_jit(
        fun=fun,
        mpmd_mesh=mpmd_mesh,
        strategy=FunctionWithLoop(),
        in_shardings=in_shardings or in_specs,
        out_shardings=out_shardings or out_specs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        compiler_options=compiler_options,
    )


def mpmd_jit_by_yield(
    fun: Callable,
    mpmd_mesh: MpmdMesh,
    *,
    target_num_stages: int | None = None,
    in_shardings=None,
    out_shardings=None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    compiler_options: dict[str, Any] | None = None,
):
    return _mpmd_jit(
        fun=fun,
        mpmd_mesh=mpmd_mesh,
        strategy=FunctionWithYield(target_num_stages=target_num_stages),
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        compiler_options=compiler_options,
    )
