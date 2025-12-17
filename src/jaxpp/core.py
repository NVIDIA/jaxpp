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

import abc
import dataclasses
import functools
import itertools as it
import logging
import math
import operator
import weakref
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property, partial
from pathlib import Path
from typing import (
    Any,
    Concatenate,
    Generic,
    NamedTuple,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

import jax
import jax._src.core as jcore
import jax._src.util as ju
import jax.interpreters.partial_eval as pe
from jax._src import source_info_util
from jax._src.ad_checkpoint import remat_p
from jax._src.debugging import inspect_sharding_p

if jax.__version_info__ < (0, 7, 0):
    from jax._src.pjit import pjit_p as jit_p
else:
    from jax._src.pjit import jit_p
if jax.__version_info__ < (0, 7, 2):
    from jax._src.op_shardings import are_op_shardings_equal as are_hlo_shardings_equal
else:
    from jax._src.op_shardings import are_hlo_shardings_equal
from jax._src.pjit import _infer_params, _parse_jit_arguments
from jax._src.shard_map import shard_map_p
from jax._src.tree_util import equality_errors_pytreedef
from jax._src.util import weakref_lru_cache
from jax.interpreters.ad import add_jaxvals_p as add_any_p

from jaxpp import env_vars
from jaxpp.array import MpmdArray
from jaxpp.jax_primitives import (
    PscanJaxpr,
    ShardingStore,
    TaskEqn,
    _check_no_attrs,
    add_multi_p,
    all_reduce_p,
    dax_pscan_p,
    delete_p,
    pipeline_yield_p,
    recv_p,
    send_done_p,
    send_p,
    task_p,
    transfer_p,
)
from jaxpp.jaxpr_utils import (
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
from jaxpp.jaxpr_utils import (
    gensym as mk_gensym,
)
from jaxpp.licm import (
    hashable_params,
    hoist_and_cse_pscan_invariant_equations,
    inline_eqns,
    outvar_normalization,
)
from jaxpp.mesh import MpmdMesh
from jaxpp.pipelining import yield_scope
from jaxpp.schedules import FusedTask, Task, mk_task_name, preprocess_schedule_tasks
from jaxpp.types import (
    DistributedSharding,
    MpmdIdx,
    TaskType,
    fresh_scalar_uid,
)
from jaxpp.utils import (
    groupby,
    hbytes,
    log_elapsed_time,
    updated_named_sharding_mesh,
)

logger = logging.getLogger(__name__)


@contextmanager
def stable_names_ctx(anno: Callable[[jcore.Var], str | None] = lambda v: None):
    prev_repr = jax._src.core.Var.__repr__
    prev_pp_var = jax._src.core.pp_var

    ctx = jcore.JaxprPpContext()

    def __repr__(v):
        if isinstance(v, jcore.Literal):
            return f"{v}"
        if (s := anno(v)) is not None:
            return f"{prev_pp_var(v, ctx)}{{{s}}}"

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
        super().__init__()
        self.parent_trace = parent_trace

    def new_arg(self, aval, mpmd_defs):
        val = self.parent_trace.new_arg(aval, source_info_util.current())
        return AllReduceRewriteTracer(self, val, mpmd_defs)

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
        mpmd_trace.new_arg(invar.aval, mpmd_defs)
        for invar, mpmd_defs in zip(jaxpr.invars, invar_mpmd_defs, strict=True)
    ]

    with jcore.set_current_trace(mpmd_trace):
        res = jcore.eval_jaxpr(jaxpr, (), *in_tracers, propagate_source_info=True)

    additional_args = ()
    if jax.__version_info__ > (0, 6, 1):
        from jax._src import source_info_util

        additional_args = (source_info_util.current(),)

    # TODO: handle literals in `res`
    # *_ is used to ignore the unused return value as to_jaxpr returns three values for
    # jax < 0.7.0, but two values for jax >= 0.7.0
    jaxpr, consts, *_ = mpmd_trace.parent_trace.to_jaxpr(
        [v.val for v in res], jaxpr.debug_info, *additional_args
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
        add_multi_eqn, eqns = None, jaxpr.eqns[eqn_idx:last]
        last = eqn_idx
        if eqns[0].primitive is add_multi_p:
            [add_multi_eqn, *eqns] = eqns
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

        if add_multi_eqn is None:
            continue

        out_store, out_inspect_eqns = ShardingStore.collect_jaxpr(add_multi_eqn.outvars)
        in_store, in_inspect_eqns = ShardingStore.collect_jaxpr(add_multi_eqn.invars)
        rev_new_eqns.extend(out_inspect_eqns)
        rev_new_eqns.append(
            add_multi_eqn.replace(
                params=add_multi_eqn.params
                | {"in_shardings": in_store, "out_shardings": out_store}
            )
        )
        rev_new_eqns.extend(in_inspect_eqns)
        for invar, mpmd_idx in ju.safe_zip(
            add_multi_eqn.invars, add_multi_eqn.params["mpmd_idxs"]
        ):
            uses = {mpmd_idx}
            if (p := var_placement.get(invar)) is not None:
                uses = uses | p
            var_placement[invar] = uses

    sub = dict(ju.safe_zip(jaxpr.invars, in_jaxpr.invars))
    sub.update(ju.safe_zip(jaxpr.outvars, in_jaxpr.outvars))
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
            # are present. This should be impossible/uncommon (higher-order gradients (?)).
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


@weakref_lru_cache
def replace_captured_meshes(
    cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr, new_mesh: jax.sharding.Mesh
) -> jcore.ClosedJaxpr:
    is_closed = isinstance(cjaxpr, jcore.ClosedJaxpr)
    if is_closed:
        jaxpr = cjaxpr.jaxpr
    else:
        jaxpr = cjaxpr

    new_eqns = []
    for eqn in jaxpr.eqns:
        param_update = {}
        if eqn.primitive is jax.lax.sharding_constraint_p:
            param_update = {
                "sharding": updated_named_sharding_mesh(
                    [eqn.params["sharding"]], new_mesh
                )[0]
            }
        elif eqn.primitive is jit_p:
            param_update = {
                "in_shardings": updated_named_sharding_mesh(
                    eqn.params["in_shardings"], new_mesh
                ),
                "out_shardings": updated_named_sharding_mesh(
                    eqn.params["out_shardings"], new_mesh
                ),
            }
        elif eqn.primitive is shard_map_p:
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
    if not is_closed:
        return res_jaxpr
    return cjaxpr.replace(jaxpr=res_jaxpr)


def _task_eqn(
    invars,
    outvars,
    task_jaxpr: jcore.ClosedJaxpr,
    mpmd_idx: int,
    in_shardings,
    out_shardings,
    donate_invars,
    task_name,
    task_info: tuple[int, TaskType] | None = None,
    latency: float | None = None,
):
    assert len(invars) == len(task_jaxpr.in_avals)
    assert len(donate_invars) == len(invars)
    return jcore.new_jaxpr_eqn(
        invars,
        outvars,
        task_p,
        {
            "call_jaxpr": task_jaxpr,
            "task_name": task_name,
            "task_info": task_info,
            "mpmd_idx": mpmd_idx,
            "in_shardings": in_shardings,
            "out_shardings": out_shardings,
            "donate_invars": donate_invars,
            "latency": latency,
        },
        effects=task_jaxpr.effects,
    )


def make_task_eqn(
    invars: Sequence[jcore.Var],
    outvars: Sequence[jcore.Var],
    eqns: list[jcore.JaxprEqn],
    mpmd_idx: int,
    task_name: str,
    task_info: tuple[int, TaskType] | None = None,
    donate_invars=None,
    in_out_shardings=None,
    latency: float | None = None,
) -> jcore.JaxprEqn:
    if latency is None:
        if task_info is not None:
            latency = task_info[1].default_latency
        else:
            latency = 1  # FIXME(task_latency)

    if donate_invars is None:
        donate_invars = (False,) * len(invars)

    if in_out_shardings is None:
        source_infos = [None] * len(outvars)
        outvar_idx = {
            o: idx for idx, o in enumerate(outvars) if isinstance(o, jcore.Var)
        }
        for eqn in eqns:
            for o in eqn.outvars:
                if (idx := outvar_idx.get(o)) is not None:
                    source_infos[idx] = eqn
        in_sharding_store, inspect_invars = ShardingStore.collect_jaxpr(invars)
        out_sharding_store, inspect_outvars = ShardingStore.collect_jaxpr(
            outvars, _provenance_info=task_name, _source_info=source_infos
        )
        eqns = inspect_invars + eqns + inspect_outvars
    else:
        in_sharding_store, out_sharding_store = in_out_shardings

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
        in_shardings=in_sharding_store,
        out_shardings=out_sharding_store,
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
                mpmd_idx
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
    clusters: list[Cluster],
    eqns: list[jcore.JaxprEqn],
    bias: dict[jcore.Var, set[MpmdIdx]] | None = None,
) -> list[int | None]:
    bias = bias or {}
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


# TODO(from,to)
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
            get_mpmd_idx(curr_enter_eqn.params["from_stage_id"]),
            TaskType.FWD,
            stage_0,
            stage_id=curr_enter_eqn.params["from_stage_id"],
        )
    ]

    passed_backward = False
    while (pp_eqn_idx := first_pipeline_yield_eqn_idx(eqns)) is not None:
        stage_i, eqns = schedule_dependencies(eqns, pp_eqn_idx)
        if not passed_backward and stage_i[-1].params["task_type"] is TaskType.BWD:
            next_enter_eqn = stage_i.pop()
            stages[-1].eqns.extend([curr_enter_eqn] + stage_i)
            curr_enter_eqn = next_enter_eqn
            passed_backward = True
            continue

        assert not passed_backward or curr_enter_eqn.params["task_type"] is TaskType.BWD
        next_enter_eqn = stage_i.pop()
        stage_id = next_enter_eqn.params["from_stage_id"]
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
        stages.append(
            Cluster(
                get_mpmd_idx(curr_enter_eqn.params["to_stage_id"]),
                curr_enter_eqn.params["task_type"],
                [curr_enter_eqn],
                stage_id=curr_enter_eqn.params["to_stage_id"],
            )
        )
    return stages, eqns


# TODO: maybe cluster_eqns shouldn't depend on `get_mpmd_idx`
def cluster_eqns(
    eqns: list[jcore.JaxprEqn],
    get_mpmd_idx: Callable[[int], MpmdIdx],
    bias: dict[jcore.Var, set[MpmdIdx]] | None = None,
) -> tuple[list[Cluster], list[jcore.JaxprEqn]]:
    bias = bias or {}
    clusters, rest = cluster_by_yield_eqns(eqns, get_mpmd_idx)
    eqns_cluster_idxs = infer_cluster_idx_for_eqns(clusters, rest, bias)
    unclustered_eqns = list[jcore.JaxprEqn]()
    for cluster_idx, eqn in zip(eqns_cluster_idxs, rest, strict=True):
        if cluster_idx is not None:
            clusters[cluster_idx].eqns.append(eqn)
        else:
            unclustered_eqns.append(eqn)
    return clusters, unclustered_eqns


def clusters_to_tasks(
    clusters: list[Cluster], outvars: Iterable[jcore.Var], is_partial_bwd: bool
) -> list[jcore.JaxprEqn]:
    outvars = set(outvars)
    undef = set[jcore.Var](outvars)
    rev_stage_eqns = []
    for mpmd_idx, ty, stage_eqns, maybe_stage_id in reversed(clusters):
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
            # TODO(task_name_task_info): remove unnecessary serialization of task_info into task_name
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

            bytes_str = hbytes(a.aval for a in task_eqn.outvars)
            logger.info(f"Activation size for {task_name}: {bytes_str}")

    return list(reversed(rev_stage_eqns))


def cluster_jaxpr(
    jaxpr: jcore.Jaxpr,
    target_num_stages: int,
    is_partial_bwd: bool,
    get_mpmd_idx: Callable[[int], MpmdIdx],
    bias: list[set[MpmdIdx] | None] | None = None,
    is_loop: bool = True,
):
    # TODO: remove is_loop parameter and make the caller perform the checks
    bias_map = None
    if bias is not None:
        bias_map = {
            invar: p
            for invar, p in zip(jaxpr.invars, bias, strict=True)
            if p is not None
        }

    clusters, unclustered_eqns = cluster_eqns(jaxpr.eqns, get_mpmd_idx, bias_map)
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
        error_jaxpr = strip_inspect_sharding_eqns(
            jaxpr.replace(eqns=new_eqns + unclustered_eqns)
        )
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


def wrap_into_tasks_inside_loop(
    loop_eqn: jcore.JaxprEqn,
    bias: list[set[MpmdIdx] | None] | None = None,
) -> jcore.JaxprEqn:
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
        bias=bias,
    )

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


@weakref_lru_cache
def _strip_inspect_sharding_eqns(
    cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr,
) -> jcore.Jaxpr:
    is_closed = isinstance(cjaxpr, jcore.ClosedJaxpr)
    if is_closed:
        jaxpr = cjaxpr.jaxpr
    else:
        jaxpr = cjaxpr

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
    new_jaxpr = jaxpr.replace(eqns=new_eqns, effects=new_effects)

    if not is_closed:
        return new_jaxpr
    return cjaxpr.replace(jaxpr=new_jaxpr)


@overload
def strip_inspect_sharding_eqns(cjaxpr: jcore.ClosedJaxpr) -> jcore.ClosedJaxpr: ...


@overload
def strip_inspect_sharding_eqns(cjaxpr: jcore.Jaxpr) -> jcore.Jaxpr: ...


def strip_inspect_sharding_eqns(
    cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr,
) -> jcore.ClosedJaxpr | jcore.Jaxpr:
    return _strip_inspect_sharding_eqns(cjaxpr)


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


def infer_outvar_placement_rev(
    jaxpr: jcore.Jaxpr, partial_outvar_placement: Iterable[set[MpmdIdx] | None]
) -> tuple[list[set[MpmdIdx]], list[set[MpmdIdx] | None]]:
    partial_outvar_placement = tuple(partial_outvar_placement)
    outvars = cast(list[jcore.Var], jaxpr.outvars)
    placement = {
        outvar: maybe_p
        for outvar, maybe_p in ju.safe_zip(outvars, partial_outvar_placement)
        if isinstance(outvar, jcore.Var) and maybe_p is not None
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

    return [placement.get(invar) for invar in jaxpr.invars], [
        placement.get(outvar)
        if isinstance(outvar, jcore.Var)
        else partial_outvar_placement[outvar_idx]
        for outvar_idx, outvar in enumerate(outvars)
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
            f"Expected 1 loop at the top level but {len(loop_eqn_idxs)} found."
        )
    return loop_eqn_idxs[0]


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


def mpmd_unzip_reverse(
    jaxpr: jcore.Jaxpr,
    out_refs: Sequence[set[MpmdIdx] | None],
    name: str,
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

    for invar, after_loop_use_p, def_p in ju.safe_zip(
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


def more_sharded_sharding(prev_sharding, alt_sharding, shape):
    prev_sharded_shape = prev_sharding.shard_shape(shape)
    sharded_shape = alt_sharding.shard_shape(shape)
    return (
        prev_sharding
        if math.prod(prev_sharded_shape) <= math.prod(sharded_shape)
        else alt_sharding
    )


def reconcile_shardings(cjaxpr: jcore.ClosedJaxpr, in_shardings, out_shardings):
    invars_outvars = set(cjaxpr.jaxpr.invars) | set(nonlit(cjaxpr.jaxpr.outvars))
    shardings = dict[jcore.Var, jax.sharding.Sharding]()
    shardings.update(zip(cjaxpr.jaxpr.invars, in_shardings, strict=True))
    for o, s in zip(cjaxpr.jaxpr.outvars, out_shardings, strict=True):
        if isinstance(o, jcore.Var):
            if (prev_sharding := shardings.get(o, None)) is not None:
                if prev_sharding != s:
                    raise NotImplementedError(
                        "Unsupported passthrough arrays with differing sharding"
                    )
            else:
                shardings[o] = s

    eqns: list[jcore.JaxprEqn] = cjaxpr.eqns
    for eqn in eqns:
        if eqn.primitive is inspect_sharding_p:
            continue

        in_shardings = eqn.params["in_shardings"]._shardings
        out_shardings = eqn.params["out_shardings"]._shardings
        # For cross mpmd rank all-reduce force same shardings
        if eqn.primitive is add_multi_p and ():
            _all_shardings = [
                _ for _ in (in_shardings + out_shardings) if _ is not None
            ]
            if len(_all_shardings) > 0:
                sharding = functools.reduce(
                    partial(more_sharded_sharding, shape=eqn.outvars[0].aval.shape),
                    _all_shardings,
                )
            else:
                sharding = None

        for var_, curr_sharding in it.chain(
            zip(eqn.invars, in_shardings), zip(eqn.outvars, out_shardings)
        ):
            if curr_sharding is None:
                continue
            prev_sharding = shardings.get(var_)
            if prev_sharding is None:
                shardings[var_] = curr_sharding
            elif curr_sharding != prev_sharding and var_ not in invars_outvars:
                if eqn.primitive is dax_pscan_p:
                    # NOTE(reconcile_sharding): while this is correct in principle
                    #   it might lead to poorer performance than observed before.
                    #   Therefore we decide to handle this in `lower_tasked_jaxpr`
                    shardings[var_] = curr_sharding
                else:
                    shardings[var_] = more_sharded_sharding(
                        prev_sharding, curr_sharding, var_.aval.shape
                    )

    jaxpr_has_unknown_shardings = False
    for eqn in eqns:
        if eqn.primitive is inspect_sharding_p:
            continue

        unknown_shardings = False
        for invar_idx, invar in enumerate(eqn.invars):
            if shardings.get(invar) is not None:
                eqn.params["in_shardings"]._shardings[invar_idx] = shardings[invar]
            if eqn.params["in_shardings"]._shardings[invar_idx] is None:
                unknown_shardings = True

        if not unknown_shardings:
            eqn.params["in_shardings"]._called_at_least_once = True
        jaxpr_has_unknown_shardings |= unknown_shardings

        unknown_shardings = False
        for outvar_idx, outvar in enumerate(eqn.outvars):
            if shardings.get(outvar) is not None:
                eqn.params["out_shardings"]._shardings[outvar_idx] = shardings[outvar]
            if eqn.params["out_shardings"]._shardings[outvar_idx] is None:
                unknown_shardings = True

        if not unknown_shardings:
            eqn.params["out_shardings"]._called_at_least_once = True
        jaxpr_has_unknown_shardings |= unknown_shardings

    return jaxpr_has_unknown_shardings


def loop_placement_by_clusters(
    loop_eqn: jcore.JaxprEqn, get_mpmd_idx: Callable[[int], MpmdIdx]
) -> tuple[list[set[MpmdIdx] | None], list[MpmdIdx | None]]:
    assert loop_eqn.primitive is dax_pscan_p
    n_consts = loop_eqn.params["n_consts"]
    jaxpr: jcore.Jaxpr = loop_eqn.params["jaxpr"].jaxpr

    invar_idx = {invar: idx for idx, invar in enumerate(jaxpr.invars)}
    outvar_idx = {outvar: idx for idx, outvar in enumerate(jaxpr.outvars)}

    mubatch_idx_outvar = jaxpr.outvars[0]
    mubatch_idx_update_eqn = None
    for mubatch_idx_update_eqn_idx, eqn in enumerate(jaxpr.eqns):
        for outvar in eqn.outvars:
            if outvar == mubatch_idx_outvar:
                mubatch_idx_update_eqn = eqn
                break
        if mubatch_idx_update_eqn is not None:
            break

    if not (
        mubatch_idx_update_eqn is not None
        and mubatch_idx_update_eqn.primitive is jax.lax.add_p
        and isinstance(r := mubatch_idx_update_eqn.invars[1], jcore.Literal)
        and r.val == 1
    ):
        raise AssertionError("Malformed loop body")

    eqns = list(jaxpr.eqns)
    eqns.pop(mubatch_idx_update_eqn_idx)
    clusters, _ = cluster_eqns(eqns, get_mpmd_idx)
    clusters[0].eqns.insert(0, mubatch_idx_update_eqn)
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


def _compute_bias(jaxpr: jcore.Jaxpr, loop_eqn_idx: int):
    loop_eqn = jaxpr.eqns[loop_eqn_idx]

    # Infer partial placement from loop body
    loop_in_mpmd_refs, loop_out_mpmd_defs = loop_placement_by_clusters(
        loop_eqn, loop_eqn.params["schedule"].get_mpmd_idx
    )

    # Use partial placement from loop body to infer
    #  before loop partial placement
    before_loop_jaxpr = jaxpr_from_eqns(
        jaxpr.eqns[:loop_eqn_idx], eqns_free_vars(jaxpr.eqns[loop_eqn_idx:])[0]
    )

    loop_args_mpmd_refs = join_argument_refs(
        cast(list[jcore.Var], loop_eqn.invars), loop_in_mpmd_refs
    )

    before_loop_invar_placement, before_loop_outvar_mpmd_defs = (
        infer_outvar_placement_rev(
            before_loop_jaxpr,
            partial_outvar_placement=tuple(
                loop_args_mpmd_refs.get(outvar)
                # NOTE: before loop outvars are just vars as it comes from
                #  `jaxpr_from_eqns`
                for outvar in cast(list[jcore.Var], before_loop_jaxpr.outvars)
            ),
        )
    )

    # make_replicated_jaxpr(
    #     before_loop_jaxpr,
    #     tuple(
    #         map(
    #             join_argument_refs(loop_eqn.invars, loop_in_mpmd_refs).get,
    #             before_loop_jaxpr.outvars,
    #         )
    #     ),
    #     list(range(mpmd_dim)),
    # )

    # Merge all partial placement known so far
    placement = {}
    for invar, p in zip(
        before_loop_jaxpr.invars, before_loop_invar_placement, strict=True
    ):
        assert invar not in placement
        placement[invar] = p

    for outvar, p in zip(
        before_loop_jaxpr.outvars, before_loop_outvar_mpmd_defs, strict=True
    ):
        assert outvar not in placement
        if p is not None:
            placement[outvar] = p

    bias: list[set[MpmdIdx] | None] = [None] * len(loop_eqn.invars)
    for invar_idx, invar in enumerate(loop_eqn.invars):
        p = placement.get(invar)
        loop_parameter_p = loop_in_mpmd_refs[invar_idx]
        if loop_parameter_p is not None and p is not None:
            if not loop_parameter_p.issubset(p):
                raise AssertionError()

        if loop_parameter_p is None and p is not None:
            bias[invar_idx] = p
    loop_placement_changed = any(b is not None for b in bias)  # noqa: F841

    return before_loop_jaxpr, bias


@weakref_lru_cache
def _wrap_into_tasks(
    cjaxpr: jcore.ClosedJaxpr, used_invars: Sequence[bool], mpmd_dim: int
) -> tuple[jcore.ClosedJaxpr, tuple[set[MpmdIdx]], tuple[set[MpmdIdx]]]:
    """
    After this pass, all the equations in the returned jaxpr are either
    (1) `task` equations, or (2) a `dax_pscan` equation containing `task` equations
    or (3) `inspect_sharding`.
    """
    jaxpr = cjaxpr.jaxpr
    [*before_loop_eqns, loop_eqn], after_loop_eqns = schedule_dependencies(
        jaxpr.eqns, get_one_loop_eqn_idx(jaxpr.eqns)
    )
    jaxpr = jaxpr.replace(eqns=before_loop_eqns + [loop_eqn] + after_loop_eqns)
    loop_eqn_idx = len(before_loop_eqns)
    loop_eqn = jaxpr.eqns[loop_eqn_idx]

    # TODO: use partial placement to obtain partial placement from
    #  after loop part
    before_loop_jaxpr, bias = _compute_bias(jaxpr, loop_eqn_idx)
    # Use current placement to taskify loop body
    tasked_loop_eqn = wrap_into_tasks_inside_loop(loop_eqn, bias)

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


def infer_donation(
    tasked_jaxpr: jcore.Jaxpr, donated_invars: Sequence[bool]
) -> jcore.Jaxpr:
    """
    Returns a new jaxpr identical to the input jaxpr, where every
    `task` equation has `params["donate_invars"]` set properly, according
    to the lifetime of that variable.
    It ensures that dlpack arrays from receive operations and all-reduces
    are never donated as that's not supported in some versions of XLA.
    """
    last_use = last_used(tasked_jaxpr)

    invar_is_donated = dict(zip(tasked_jaxpr.invars, donated_invars))
    received_vars = set[jcore.Var]()

    least_donation = dict[tuple[int, TaskType], Sequence[bool]]()
    new_eqns = []
    for task_eqn_idx, task_eqn in enumerate(tasked_jaxpr.eqns):
        is_last_use_for_invar = [
            last_use[invar] == task_eqn_idx for invar in task_eqn.invars
        ]
        donation = tuple(
            is_last_use_for_invar[invar_idx]
            and invar_is_donated.get(invar, True)
            # NOTE: we avoid donating received invars
            and invar not in received_vars
            for invar_idx, invar in enumerate(task_eqn.invars)
        )

        if task_eqn.primitive is task_p:
            new_eqns.append(
                task_eqn.replace(params=task_eqn.params | {"donate_invars": donation})
            )
            if task_eqn.params[
                "task_info"
            ] is not None and donation <= least_donation.get(
                task_eqn.params["task_info"], (True,) * len(donation)
            ):
                least_donation[task_eqn.params["task_info"]] = donation
        elif task_eqn.primitive is transfer_p:
            # NOTE: we avoid donating received invars.
            # Variables that are sent are not donated because
            #  send_done (below) extends their lifetime to the end of
            # the program
            received_vars.update(task_eqn.outvars)
            new_eqns.append(task_eqn)
        elif task_eqn.primitive is add_multi_p:
            new_eqns.append(
                task_eqn.replace(params=task_eqn.params | {"donate_invars": donation})
            )
        elif task_eqn.primitive is send_done_p:
            new_eqns.append(task_eqn)
        else:
            raise ValueError(f"Unexpected equation with primitive {task_eqn.primitive}")

    # NOTE: after unrolling, the same task function applied to different
    #  microbatches will have the same `task_info`.
    #  Here, we set the donation to the least donation among all of the task's
    #  instantiations to minimize the compilation misses
    minimize_compilation_misses = True
    if minimize_compilation_misses:
        res = []
        for eqn in new_eqns:
            if eqn.primitive is task_p and eqn.params["task_info"] is not None:
                eqn = eqn.replace(
                    params=eqn.params
                    | {"donate_invars": least_donation[eqn.params["task_info"]]}
                )
            res.append(eqn)
        new_eqns = res
    res = tasked_jaxpr.replace(eqns=new_eqns)
    check_jaxpr(res)
    return res


def add_deletes(tasked_jaxpr: jcore.Jaxpr, donated_invars: Sequence[bool]):
    def _use_mpmd_idx(eqn, invar_idx: int | None) -> int:
        if eqn.primitive is task_p:
            mpmd_idx = eqn.params["mpmd_idx"]
        elif eqn.primitive is transfer_p:
            mpmd_idx = eqn.params["src_mpmd_idx"]
        elif eqn.primitive is send_done_p:
            mpmd_idx = eqn.params["mpmd_idx"]
        elif eqn.primitive is add_multi_p:
            assert invar_idx is not None
            mpmd_idx = eqn.params["mpmd_idxs"][invar_idx]
        else:
            raise ValueError(f"Unsupported {eqn.primitive}")
        return mpmd_idx

    def _delete_eqn(invars, mpmd_idx: int):
        return jcore.new_jaxpr_eqn(
            invars,
            [jcore.DropVar(invar.aval) for invar in invars],
            delete_p,
            {"mpmd_idx": mpmd_idx},
            jcore.no_effects,  # FIXME
        )

    _, uses = defs_and_uses(tasked_jaxpr.eqns)
    # TODO(recv_done): add deletions for received invars too
    # after adding recv lifetime extension too

    last_use_by_mpmd_idx = defaultdict(dict)
    for v, us in uses.items():
        for u in us:
            eqn = tasked_jaxpr.eqns[u.eqn_idx]
            mpmd_idx = _use_mpmd_idx(eqn, u.invar_idx)
            last_use_by_mpmd_idx[mpmd_idx][v] = u.eqn_idx

    last_use = last_used(tasked_jaxpr)
    invar_is_donated = dict(zip(tasked_jaxpr.invars, donated_invars))
    # TODO(recv_done): add deletions for received invars too
    # after adding recv lifetime extension too
    received_vars = set().union(
        *(_.outvars for _ in tasked_jaxpr.eqns if _.primitive is transfer_p)
    )

    new_eqns = list[jcore.JaxprEqn]()
    for eqn_idx, eqn in enumerate(tasked_jaxpr.eqns):
        new_eqns.append(eqn)
        delete_invars_mask = [
            (
                last_use[invar] == eqn_idx
                and invar_is_donated.get(invar, True)
                and invar not in received_vars
            )
            for invar in eqn.invars
        ]

        if not any(delete_invars_mask):
            continue

        if eqn.primitive is add_multi_p:
            for should_delete, invar, mpmd_idx in zip(
                delete_invars_mask, eqn.invars, eqn.params["mpmd_idxs"], strict=True
            ):
                if not should_delete:
                    continue
                new_eqns.append(_delete_eqn([invar], mpmd_idx))
            continue

        mpmd_idx = _use_mpmd_idx(eqn, None)
        assert mpmd_idx is not None, eqn
        # NOTE: it's fine to delete a donated buffer
        delete_invars = [
            invar
            for should_delete, invar in zip(delete_invars_mask, eqn.invars, strict=True)
            if should_delete
        ]
        new_eqns.append(_delete_eqn(delete_invars, mpmd_idx))

    return tasked_jaxpr.replace(eqns=new_eqns)


send_recv_id = it.count()


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


T = TypeVar("T")
T2 = TypeVar("T2")


@dataclasses.dataclass(frozen=True)
class GlobalTimeEqn(Generic[T]):
    start_time: int
    end_time: int
    elem: T

    def replace(self, elem: T2) -> "GlobalTimeEqn[T2]":
        return GlobalTimeEqn(self.start_time, self.end_time, elem)


def reorder_nodes_with_schedule(
    nodes: list[Task],
    dependencies_and_uses: tuple[dict[int, set[int]], dict[int, set[int]]],
    schedule_tasks: list[list[Task | FusedTask]],
) -> list[GlobalTimeEqn[list[int]]]:
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

    global_eqns = list[GlobalTimeEqn[list[int]]]()
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

                # A fused task is ready if all its tasks are ready
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
                    # FIXME: end_time should be the one of the full fusion group
                    node_ready_time[nodes[use_eqn_idx]] = end_time

        global_eqns.append(GlobalTimeEqn(start_time, end_time, scheduled_node_idxs))
        schedule_idx_by_mpmd_idx[mpmd_idx] += 1
        time_by_mpmd_idx[mpmd_idx] = end_time

    for mpmd_idx, time in enumerate(time_by_mpmd_idx):
        if schedule_idx_by_mpmd_idx[mpmd_idx] != len(schedule_tasks[mpmd_idx]):
            raise AssertionError(
                "Loop tasks have ended but schedule contains other tasks. "
                f"{schedule_tasks[mpmd_idx][schedule_idx_by_mpmd_idx[mpmd_idx]:]}"
            )

    return sorted(global_eqns, key=lambda e: (e.start_time, e.end_time))


class TransferTo(NamedTuple):
    tgt_mpmd_idx: int
    out_idx: int
    first_use_eqn_idx: int


def compute_transfers(eqns: list[jcore.JaxprEqn]) -> list[list[TransferTo]]:
    class VisitedElem(NamedTuple):
        tgt_mpmd_idx: int
        var: jcore.Var

    mpmd_def, _ = defs_and_uses(eqns)

    transfers = [list[TransferTo]() for _ in range(len(eqns))]
    resolved_transfer = set[VisitedElem]()
    for eqn_idx, eqn in enumerate(eqns):
        if eqn.primitive is add_multi_p:
            continue
        assert eqn.primitive is task_p
        eqn_mpmd_idx = eqn.params["mpmd_idx"]
        for invar in nonlit(eqn.invars):
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
            if def_eqn.primitive is add_multi_p:
                continue
            assert def_eqn.primitive is task_p, def_eqn.primitive

            if eqn_mpmd_idx != def_eqn.params["mpmd_idx"]:
                resolved_transfer.add(VisitedElem(eqn_mpmd_idx, invar))
                transfers[_.eqn_idx].append(
                    TransferTo(eqn_mpmd_idx, _.outvar_idx, eqn_idx)
                )

    return transfers


def add_transfers(eqns: list[GlobalTimeEqn[jcore.JaxprEqn]]) -> list[jcore.JaxprEqn]:
    transfers = compute_transfers([_.elem for _ in eqns])
    gensym = mk_gensym()
    new_eqns = []
    prev_start_time = 0
    sub_by_mpmd_idx = defaultdict(dict[jcore.Var, jcore.Var])
    # TODO: revisit when transfers are scheduled and bufferize receives
    #  earlier than the transfer
    next_time_transfers = list[tuple[int, jcore.JaxprEqn]]()
    for eqn_idx, _ in enumerate(eqns):
        start_time = _.start_time
        eqn = _.elem
        if start_time != prev_start_time:
            for _, transfer in sorted(next_time_transfers, key=operator.itemgetter(0)):
                new_eqns.append(transfer)
            prev_start_time = start_time
            next_time_transfers = []

        # Only task equations can receive
        if eqn.primitive is add_multi_p:
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
            outvars = [gensym(v.aval) for v in invars]
            src_shardings = [eqn.params["out_shardings"][t.out_idx] for t in ts]
            transfer_eqn = jcore.new_jaxpr_eqn(
                invars=invars,
                outvars=outvars,
                primitive=transfer_p,
                params={
                    "src_mpmd_idx": eqn.params["mpmd_idx"],
                    "tgt_mpmd_idx": tgt_mpmd_idx,
                    "src_shardings": src_shardings,
                },
                effects=jcore.no_effects,  # FIXME
            )
            next_time_transfers.append(
                (min(_.first_use_eqn_idx for _ in ts), transfer_eqn)
            )
            for i, o in zip(invars, outvars):
                assert i not in sub_by_mpmd_idx[tgt_mpmd_idx]
                sub_by_mpmd_idx[tgt_mpmd_idx][i] = o

    return new_eqns


def mk_send_done_eqn(invars: list[jcore.Var], mpmd_idx: int):
    return jcore.new_jaxpr_eqn(
        invars=invars,
        outvars=[jcore.DropVar(v.aval) for v in invars],
        primitive=send_done_p,
        params={"mpmd_idx": mpmd_idx},
        effects=send_done_p.abstract_eval(*invars, mpmd_idx=mpmd_idx)[1],
    )


def add_send_dones(
    eqns: list[jcore.JaxprEqn], transfer_done_delay: int
) -> list[jcore.JaxprEqn]:
    # FIXME(multi_send_done): this function will add multiple `send_done`s
    #  for the same variable, one for each send

    # TODO(recv_done): do the same for receives?
    send_channels = defaultdict[tuple[int, int], deque[list[jcore.Var]]](deque)

    new_eqns = []
    for eqn in eqns:
        new_eqns.append(eqn)
        if eqn.primitive is transfer_p:
            k = (eqn.params["src_mpmd_idx"], eqn.params["tgt_mpmd_idx"])
            send_channels[k].append(eqn.invars)

            if len(done_invars_q := send_channels[k]) > transfer_done_delay:
                src_mpmd_idx, tgt_mpmd_idx = k
                done_invars = done_invars_q.popleft()
                new_eqns.append(mk_send_done_eqn(done_invars, mpmd_idx=src_mpmd_idx))

    for src_mpmd_idx, _ in groupby((e[0][0], e) for e in send_channels.items()).items():
        qs = ju.unzip2(_)[1]
        if len(qs) > 0:
            done_invars = [invar for q in qs for invars in q for invar in invars]
            new_eqns.append(mk_send_done_eqn(done_invars, src_mpmd_idx))
    return new_eqns


_fused_jaxprs = weakref.WeakValueDictionary()


def _get_fused_jaxpr_cached(group_eqns: list[jcore.JaxprEqn], invars, outvars):
    eqn_keys = []
    counter = it.count()
    ids = defaultdict(lambda: next(counter))
    exclude_param_keys = {"call_counter"}
    for eqn in group_eqns:
        invars_ids = tuple(ids[invar] for invar in eqn.invars)
        outvars_ids = tuple(ids[outvar] for outvar in eqn.outvars)
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

    if (jaxpr := _fused_jaxprs.get(tuple(eqn_keys))) is not None:
        return jaxpr

    jaxpr = jcore.ClosedJaxpr(
        jcore.Jaxpr(
            constvars=(),
            invars=invars,
            outvars=outvars,
            eqns=group_eqns,
            effects=jcore.join_effects(*(eqn.effects for eqn in group_eqns)),
        ),
        (),
    )
    _fused_jaxprs[tuple(eqn_keys)] = jaxpr
    return jaxpr


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

        fused_task_jaxpr = _get_fused_jaxpr_cached(group_eqns, invars, outvars)

        new_eqns.append(
            _task_eqn(
                invars=invars,
                outvars=outvars,
                task_jaxpr=fused_task_jaxpr,
                mpmd_idx=mpmd_idx,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                donate_invars=donate_invars,
                task_name=f"fused_{'_'.join(e.params['task_name'] for e in group_eqns)}",
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
        if eqn.primitive is add_multi_p:
            start = max(
                max(time_by_mpmd_idx[mpmd_idx] for mpmd_idx in eqn.params["mpmd_idxs"]),
                max((res[d.eqn_idx][1] for d in invar_definitions), default=0),
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
    scheduled_eqns = reorder_nodes_with_schedule(
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

    scheduled_and_fused_jaxpr = fuse_groups(
        unrolled_loop_jaxpr, [_.elem for _ in scheduled_eqns]
    )

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
        mpmd_idxs, vs = ju.unzip2(copies)

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
    sub_by_mpmd_idx = dict[int, dict[jcore.Var, jcore.Var]]()
    new_eqns = []
    gensym = mk_gensym()
    for eqn in jaxpr.eqns:
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
            if (def_mpmd_idx := defined.get(outvar)) is not None:
                if def_mpmd_idx == eqn_mpmd_idx:
                    raise AssertionError(
                        f"Double definition in same mpmd_idx {def_mpmd_idx}"
                    )
                new_outvar = gensym(outvar.aval)
                sub[outvar] = new_outvar
                outvar = new_outvar
            defined[outvar] = eqn_mpmd_idx
            outvars.append(outvar)

        new_eqns.append(eqn.replace(invars=invars, outvars=outvars))

    defined_in_mpmd_idx_as = dict[jcore.Var, dict[int, jcore.Var]]()
    for d, mpmd_idx in defined.items():
        defined_in_mpmd_idx_as[d] = {mpmd_idx: d}
    for mpmd_idx, sub in sub_by_mpmd_idx.items():
        for orig_var, cpy_var in sub.items():
            _ = defined_in_mpmd_idx_as.get(orig_var, {})
            assert mpmd_idx not in _
            _[mpmd_idx] = cpy_var

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


def scalarize(
    tasked_jaxpr: jcore.ClosedJaxpr, mpmd_mesh: MpmdMesh
) -> list[jcore.Jaxpr]:
    # TODO: add token args/results to communications
    eqn_by_mpmd_idx = [[] for _ in range(mpmd_mesh.mpmd_dim)]
    gensym = mk_gensym()
    recvs_buffer_by_mpmd_idx = [list[jcore.Var]() for _ in range(mpmd_mesh.mpmd_dim)]
    for eqn in tasked_jaxpr.eqns:
        if eqn.primitive is task_p:
            eqn_by_mpmd_idx[eqn.params["mpmd_idx"]].append(eqn)
        elif eqn.primitive is transfer_p:
            src_mpmd_idx = eqn.params["src_mpmd_idx"]
            tgt_mpmd_idx = eqn.params["tgt_mpmd_idx"]
            op_id = next(
                send_recv_id
            )  # TODO: make this id dependent on module being compiled

            # TODO: consider what shardings to keep
            sender_shardings = updated_named_sharding_mesh(
                eqn.params["src_shardings"], mpmd_mesh.unstack[src_mpmd_idx]
            )
            receiver_shardings = updated_named_sharding_mesh(
                eqn.params["src_shardings"], mpmd_mesh.unstack[tgt_mpmd_idx]
            )

            send_eqn = jcore.new_jaxpr_eqn(
                invars=eqn.invars,
                outvars=[jcore.DropVar(v.aval) for v in eqn.invars],
                primitive=send_p,
                params={
                    "id": op_id,
                    "shardings": tuple(
                        zip(
                            (tgt_mpmd_idx,) * len(eqn.invars),
                            receiver_shardings,
                            strict=True,
                        )
                    ),
                },
                effects=jcore.no_effects,  # FIXME
            )

            recv_invars = []
            if False:  # FIXME
                recv_buffers = [
                    (gensym(v.aval), sh)
                    for v, sh in zip(eqn.invars, receiver_shardings, strict=True)
                ]
                recvs_buffer_by_mpmd_idx[tgt_mpmd_idx].extend(recv_buffers)
                recv_invars = ju.unzip2(recv_buffers)[0]

            recv_eqn = jcore.new_jaxpr_eqn(
                invars=recv_invars,
                outvars=eqn.outvars,
                primitive=recv_p,
                params={
                    "id": op_id,
                    "shape_and_dtype": [
                        (v.aval.shape, v.aval.dtype) for v in eqn.invars
                    ],
                    "shardings": tuple(
                        zip(
                            (src_mpmd_idx,) * len(eqn.outvars),
                            sender_shardings,
                            strict=True,
                        )
                    ),
                },
                effects=jcore.no_effects,  # FIXME
            )

            eqn_by_mpmd_idx[src_mpmd_idx].append(send_eqn)
            eqn_by_mpmd_idx[tgt_mpmd_idx].append(recv_eqn)
        elif eqn.primitive is add_multi_p:
            # NOTE: all shardings are enforced to be the same
            # by `reconcile_shardings`
            _ = eqn.params["in_shardings"]

            for invar, mpmd_idx, donated in zip(
                eqn.invars,
                eqn.params["mpmd_idxs"],
                eqn.params["donate_invars"],
                strict=True,
            ):
                _ = jcore.new_jaxpr_eqn(
                    [invar],
                    eqn.outvars,
                    all_reduce_p,
                    params={
                        "mpmd_idxs": list(eqn.params["mpmd_idxs"]),
                        # "donated": (0,) if donated else None,
                    },
                    effects=jcore.no_effects,  # FIXME
                )
                eqn_by_mpmd_idx[mpmd_idx].append(_)
        elif eqn.primitive is send_done_p:
            eqn_by_mpmd_idx[eqn.params["mpmd_idx"]].append(eqn)
        elif eqn.primitive is delete_p:
            eqn_by_mpmd_idx[eqn.params["mpmd_idx"]].append(eqn)
        else:
            raise NotImplementedError(f"{eqn.primitive}")

    invar_idx = {v: idx for idx, v in enumerate(tasked_jaxpr.jaxpr.invars)}
    outvar_idx = {
        v: idx
        for idx, v in enumerate(tasked_jaxpr.jaxpr.outvars)
        if not isinstance(v, jcore.ClosedJaxpr)
    }
    outvar_set = set(nonlit(tasked_jaxpr.jaxpr.outvars))
    jaxprs = list[jcore.Jaxpr]()
    for mpmd_idx, eqns in enumerate(eqn_by_mpmd_idx):
        if len(recvs_buffer_by_mpmd_idx[mpmd_idx]) > 0:
            buffer_vars, shardings = ju.unzip2(recvs_buffer_by_mpmd_idx[mpmd_idx])
            zeros_jaxpr = jax.make_jaxpr(
                lambda: [
                    jax.numpy.zeros(v.aval.shape, v.aval.dtype) + 0 for v in buffer_vars
                ]
            )()
            assert all(isinstance(v, jcore.Var) for v in buffer_vars), buffer_vars
            _eqns = make_task_eqn(
                invars=[],
                outvars=buffer_vars,
                eqns=inline_eqns(
                    zeros_jaxpr.eqns,
                    {},
                    dict(zip(zeros_jaxpr.jaxpr.outvars, buffer_vars, strict=True)),
                ),
                mpmd_idx=mpmd_idx,
                task_name="alloc_recv_buffers",
                in_out_shardings=([], shardings),
            )
            eqns = [_eqns] + eqns
        jaxpr = jaxpr_from_eqns(eqns, outvar_set)
        jaxpr = jaxpr.replace(
            invars=sorted(jaxpr.invars, key=lambda v: invar_idx[v]),
            # TODO: outvar doesn't have literals now (because of `jaxpr_from_eqns` above)
            # but it might in the future
            outvars=sorted(jaxpr.outvars, key=lambda v: outvar_idx[v]),
        )
        check_jaxpr(jaxpr)
        jaxprs.append(jaxpr)

    return jaxprs


@contextmanager
def ensuring_pgle_disabled():
    non_ex = object()
    prev_flag = getattr(jax.config, "jax_enable_pgle", non_ex)
    if prev_flag is not non_ex:
        jax.config.update("jax_enable_pgle", False)

    try:
        yield
    finally:
        if prev_flag is not non_ex:
            jax.config.update("jax_enable_pgle", prev_flag)


def infer_shardings2(closed_jaxpr: jcore.ClosedJaxpr, in_shardings, lowering_mesh):
    # TODO: add support for layouts
    env = dict(zip(closed_jaxpr.jaxpr.invars, in_shardings, strict=True))
    for eqn in closed_jaxpr.eqns:
        eqn: jcore.JaxprEqn

        if eqn.primitive is inspect_sharding_p:
            continue

        # TODO: this might fail for literal args
        _in = [env[invar] for invar in eqn.invars]
        if eqn.primitive is task_p:
            if env_vars.jaxpp_debug_skip_propagation.value:
                result_shardings = [
                    jax.NamedSharding(lowering_mesh, jax.sharding.PartitionSpec())
                ] * len(eqn.outvars)
            else:

                def _fn(args):
                    return task_p.bind(*args, **eqn.params)

                with ensuring_pgle_disabled():
                    compiled = (
                        jax.jit(_fn, in_shardings=(_in,))
                        .lower([_.aval for _ in eqn.invars])
                        .compile()
                    )

                result_shardings = compiled.output_shardings

        elif eqn.primitive is dax_pscan_p:
            result_shardings = infer_shardings2(eqn.params["jaxpr"], _in, lowering_mesh)
        elif eqn.primitive is add_multi_p:
            result_shardings = [_in[0]]
        else:
            raise ValueError(f"Unknown primitive {eqn.primitive}")

        for outvar, sh in zip(eqn.outvars, result_shardings, strict=True):
            env[outvar] = sh

        if "in_shardings" in eqn.params:
            eqn.params["in_shardings"] = ShardingStore(
                [_.aval for _ in eqn.invars], _shardings=list(_in)
            )
        if "out_shardings" in eqn.params:
            eqn.params["out_shardings"] = ShardingStore(
                [_.aval for _ in eqn.outvars], _shardings=list(result_shardings)
            )

    return [
        env[outvar] if not isinstance(outvar, jcore.Literal) else outvar
        for outvar in closed_jaxpr.jaxpr.outvars
    ]


def infer_shardings(
    lowering_mesh: jax.sharding.Mesh,
    closed_jaxpr: jcore.ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts,
    out_layouts,
    compiler_options,
    name: str,
) -> jcore.ClosedJaxpr:
    assert all(_ is None for _ in in_layouts)
    assert all(_ is None for _ in out_layouts)

    with log_elapsed_time("xla_compilation/driver"):
        if (
            env_vars.jaxpp_enable_local_propagation.value
            or env_vars.jaxpp_debug_skip_propagation.value
        ):
            _ = infer_shardings2(closed_jaxpr, in_shardings, lowering_mesh)
        else:
            # Trigger first compilation on the driver for inferring intermediate shardings
            with ensuring_pgle_disabled():
                jax.jit(
                    jcore.jaxpr_as_fun(closed_jaxpr),
                    in_shardings=in_shardings,
                    out_shardings=list(out_shardings),
                ).lower(*closed_jaxpr.in_avals).compile()

    closed_jaxpr = strip_inspect_sharding_eqns(closed_jaxpr)
    # NOTE: mutates sharding stored inside `closed_jaxpr`
    reconcile_shardings(closed_jaxpr, in_shardings, out_shardings)

    # log_activation_shardings(closed_jaxpr)
    return closed_jaxpr


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


@weakref_lru_cache
def disable_prevent_cse(cjaxpr: jcore.ClosedJaxpr | jcore.Jaxpr):
    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        jaxpr = cjaxpr.jaxpr
    else:
        jaxpr = cjaxpr

    new_eqns = []
    for eqn in jaxpr.eqns:
        params_update = {}
        if eqn.primitive is remat_p:
            params_update["prevent_cse"] = False

        for k, v in eqn.params.items():
            if isinstance(v, (jcore.ClosedJaxpr, jcore.Jaxpr)):
                params_update[k] = disable_prevent_cse(v)

        new_eqns.append(eqn.replace(params=eqn.params | params_update))

    new_jaxpr = jaxpr.replace(eqns=new_eqns)

    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        return cjaxpr.replace(jaxpr=new_jaxpr)

    return new_jaxpr


@weakref_lru_cache
def preprocess_jaxpr(
    cjaxpr: jcore.ClosedJaxpr,
) -> tuple[jcore.ClosedJaxpr, Sequence[bool]]:
    if env_vars.jaxpp_disable_prevent_cse.value:
        cjaxpr = disable_prevent_cse(cjaxpr)

    jaxpr_with_consts = pe.convert_constvars_jaxpr(cjaxpr.jaxpr)
    licm_jaxpr = loop_passes(jaxpr_with_consts)
    dced_jaxpr, used_inputs = pe.dce_jaxpr(
        licm_jaxpr, used_outputs=[True] * len(licm_jaxpr.outvars)
    )
    jaxpr = dced_jaxpr.replace(
        invars=licm_jaxpr.invars, debug_info=licm_jaxpr.debug_info
    )
    return pe.close_jaxpr(jaxpr), tuple(used_inputs)


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
class FunctionReverse(Strategy):
    out_refs: Any

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
        flat_out_refs, out_refs_tree = jax.tree.flatten(self.out_refs)
        assert out_refs_tree == out_tree
        jaxpr, in_mpmd_defs, out_mpmd_defs = mpmd_unzip_reverse(
            closed_jaxpr.jaxpr, flat_out_refs, name=name
        )
        # TODO: apply `fixup_multidefs` to `jaxpr` here
        # for Mc eval_jaxpr runtime
        closed_jaxpr = jcore.ClosedJaxpr(jaxpr, closed_jaxpr.consts)
        reconcile_shardings(closed_jaxpr, flat_in_shardings, flat_out_shardings)
        return strip_inspect_sharding_eqns(closed_jaxpr), in_mpmd_defs, out_mpmd_defs


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
            p, _ = _infer_params(self.fun, self.pjit_info, args, kwargs)
            consts = p.consts

        # TODO: understand why `true_consts` is inconsistent with `p.consts` above
        true_consts = p.params["jaxpr"].consts
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
            p.params, len(true_consts), replicated_sharding=replicated_sharding
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
            consts=true_consts,
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
    new_eqns = []
    for eqn in cjaxpr.eqns:
        if eqn.primitive is task_p:
            call_jaxpr = eqn.params["call_jaxpr"]
            new_mesh = mpmd_mesh.unstack[eqn.params["mpmd_idx"]]
            param_update = {
                "call_jaxpr": replace_captured_meshes(call_jaxpr, new_mesh=new_mesh),
                # TODO(sharding_store): make {in,out}_shardings Stores to a list in
                # `infer_shardings`
                "in_shardings": updated_named_sharding_mesh(
                    eqn.params["in_shardings"], new_mesh=new_mesh
                ),
                "out_shardings": updated_named_sharding_mesh(
                    list(eqn.params["out_shardings"]), new_mesh=new_mesh
                ),
            }
            new_eqns.append(eqn.replace(params=eqn.params | param_update))
        elif eqn.primitive is transfer_p:
            new_mesh = mpmd_mesh.unstack[eqn.params["src_mpmd_idx"]]
            param_update = {
                "src_shardings": updated_named_sharding_mesh(
                    eqn.params["src_shardings"],
                    new_mesh=mpmd_mesh.unstack[eqn.params["src_mpmd_idx"]],
                )
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
    return are_hlo_shardings_equal(
        sharding._to_xla_hlo_sharding(a.ndim),
        a.sharding._to_xla_hlo_sharding(a.ndim),
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
        arg_names = self.global_jaxpr.jaxpr.debug_info.arg_names
        for name, in_sharding, aval in zip(
            arg_names,
            self.in_info.in_shardings,
            self.global_jaxpr.in_avals,
            strict=True,
        ):
            try:
                in_sharding.shard_shape(aval.shape)
            except:
                logger.warning(
                    f"Failed shard_shape for '{name}': {aval} with {in_sharding=}"
                )

    @cached_property
    def as_fun(self):
        stripped = strip_inspect_sharding_eqns(self.local_jaxpr)
        return jcore.jaxpr_as_fun(stripped)

    def _maybe_shard_inputs(self, flat_args: list[jax.Array]):
        local_args = []
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

            arg_name = self.global_jaxpr.jaxpr.debug_info.arg_names[arg_idx]

            # FIXME: in_shardings offset by consts?
            expected_sharding = self.in_info.in_shardings[arg_idx]
            if isinstance(arg, MpmdArray):
                if not arg.is_partially_addressable:
                    raise ValueError(
                        f"{MpmdArray.__name__} passed as argument {arg_name} is not "
                        "partially addressable"
                    )
                arg = arg.to_mpmd_local_array

            local_arg = arg
            if isinstance(arg, jax.Array):
                if not array_has_sharding(arg, expected_sharding) and arg._committed:
                    logger.warning(
                        f"Resharding '{arg_name}': {local_arg.shape=} "
                        f"tgt_sharding={expected_sharding.spec}"
                    )
                    local_arg = jax.device_put(arg, expected_sharding)
            else:
                local_arg = jax.device_put(arg, expected_sharding)

            local_args.append(local_arg)

        arg_shape_and_dtype = [(a.aval.shape, a.aval.dtype) for a in local_args]
        assert (
            arg_shape_and_dtype
            == [(_.shape, _.dtype) for _ in self.local_jaxpr.in_avals]
        ), f"{len(arg_shape_and_dtype)} {arg_shape_and_dtype=}\n{len(self.local_jaxpr.in_avals)} {self.local_jaxpr.in_avals=}"
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
            outs = self.as_fun(*local_args)

        results = self._check_and_build_outputs(outs)
        return jax.tree.unflatten(self.in_info.out_tree, results)

    def _check_and_build_outputs(self, outs: list[jax.Array]):
        for _ in outs:
            _check_no_attrs(_)

        results = []
        local_idx = 0
        for global_idx, mpmd_idxs in enumerate(self.in_info.out_mpmd_defs):
            if self.mpmd_mesh.my_mpmd_axis_index in mpmd_idxs:
                out = MpmdArray(
                    partially_addressable_arrays=[outs[local_idx]],
                    mpmd_mesh=self.mpmd_mesh,
                    mpmd_idxs=frozenset(mpmd_idxs),
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
                    mpmd_mesh=self.mpmd_mesh,
                    mpmd_idxs=frozenset(mpmd_idxs),
                    shape=aval.shape,
                    spec=self.in_info.out_shardings[global_idx].spec,
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
        res += f"({idx}) {jcore.pp_var(v, ctx).format()}: {jcore.pp_aval(v.aval, ctx)} # {name}\n"

    res += jcore.pp_jaxpr(jaxpr, ctx, settings).format()
    res += "\n"

    for idx, (name, v) in enumerate(
        zip(jaxpr.debug_info.result_paths, jaxpr.outvars, strict=True)
    ):
        res += f"({idx}) {jcore.pp_var(v, ctx).format()}: {jcore.pp_aval(v.aval, ctx)} # {name}\n"
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
        settings = jcore.JaxprPpSettings()
        output_dir = Path(env_vars.jaxpp_dump_dir.value)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{name}.{jax.process_index()}.jaxpr.txt"
        output_file.write_text(jcore.pp_jaxpr(jaxpr, ctx, settings).format())


def common_passes(jaxpr: jcore.Jaxpr, donated_invars):
    assert len(jaxpr.invars) == len(donated_invars), (
        len(jaxpr.invars),
        len(donated_invars),
    )
    times = infer_times(jaxpr.eqns)
    with_transfers = jaxpr.replace(
        eqns=add_send_dones(
            add_transfers(
                [
                    GlobalTimeEqn(t[0], t[1], eqn)
                    for eqn, t in zip(jaxpr.eqns, times, strict=True)
                ]
            ),
            transfer_done_delay=env_vars.jaxpp_transfer_done_delay.value,
        )
    )

    for eqn in with_transfers.eqns:
        if eqn.primitive is task_p:
            bytes_str = hbytes(a.aval for a in eqn.outvars)
            logger.info(
                f"Logical (unsharded) output size for {eqn.params['task_name']}_{eqn.params.get('call_counter', 0)} (mpmd_idx={eqn.params['mpmd_idx']}): {bytes_str}"
            )
        elif eqn.primitive is transfer_p:
            bytes_str = hbytes(a.aval for a in eqn.outvars)
            logger.info(
                f"Transfer {eqn.params['src_mpmd_idx']} -> {eqn.params['tgt_mpmd_idx']}: {bytes_str}"
            )
        else:
            logger.info(f"{eqn.primitive}")

    with_transfers = infer_donation(with_transfers, donated_invars=donated_invars)
    with_transfers = add_deletes(with_transfers, donated_invars=donated_invars)
    return with_transfers


def check_scalar_jaxprs(
    with_transfers: jcore.Jaxpr,
    jaxprs: list[jcore.Jaxpr],
    in_mpmd_defs,
    out_placement,
):
    for mpmd_idx, j in enumerate(jaxprs):
        scalar_invar_idx = 0
        for invar, d in zip(with_transfers.invars, in_mpmd_defs, strict=True):
            # FIXME: d is None here because it's a constvar
            if d is not None and mpmd_idx in d:
                assert j.invars[scalar_invar_idx] is invar, (
                    mpmd_idx,
                    scalar_invar_idx,
                )
                scalar_invar_idx += 1

        scalar_outvar_idx = 0
        for outvar, d in zip(with_transfers.outvars, out_placement.values, strict=True):
            if mpmd_idx == d:
                assert j.outvars[scalar_outvar_idx] is outvar, (
                    mpmd_idx,
                    scalar_outvar_idx,
                )
                scalar_outvar_idx += 1


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

    @cached_property
    def as_fun(self):
        stripped = strip_inspect_sharding_eqns(self.closed_jaxpr)
        return jcore.jaxpr_as_fun(stripped)

    def infer_intermediate_shardings(self):
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
                compiler_options=self.compiler_options,
                name=self.name,
            )
            return dataclasses.replace(self, closed_jaxpr=closed_jaxpr)
        return dataclasses.replace(
            self, closed_jaxpr=strip_inspect_sharding_eqns(self.closed_jaxpr)
        )

    @cached_property
    def in_shardings(self):
        res = jax.tree_util.tree_unflatten(
            self.in_info.in_tree,
            [
                DistributedSharding(mpmd_idxs, sharding)
                for mpmd_idxs, sharding in zip(
                    self.in_info.in_mpmd_defs,
                    self.in_info.in_shardings,
                    strict=True,
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
            equality_errors_pytreedef(self.in_info.in_tree, in_tree)
        )

        for i, arg in enumerate(flat_args):
            expected_mpmd_idx = set(self.in_info.in_mpmd_defs[len(self.consts) + i])
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

                    for mpmd_idx in expected_mpmd_idx:
                        (sh,) = updated_named_sharding_mesh(
                            (self.in_info.in_shardings[i],),
                            self.mpmd_mesh.unstack[mpmd_idx],
                        )
                        values[mpmd_idx] = jax.device_put(arg, sh)

                    # TODO(fixup_multidefs): instead of eval_jaxpr on `MpmdArray`s
                    #   "deduplicate" jaxpr's invars in fixup_multidefs
                    flat_args[i] = MpmdArray(
                        values.values(),
                        mpmd_mesh=self.mpmd_mesh,
                        mpmd_idxs=frozenset(expected_mpmd_idx),
                    )
            elif isinstance(arg, MpmdArray):
                assert set(arg._mpmd_idxs) == expected_mpmd_idx
            else:
                pass

        with self.mpmd_mesh:
            outputs = self.as_fun(*self.consts, *flat_args)

        for output in outputs:
            assert isinstance(output, jax.Array), type(output)

        i = 0
        actual_outputs = list[MpmdArray]()
        for out_mpmd_def in self.in_info.out_mpmd_defs:
            actual_outputs.append(
                MpmdArray(
                    outputs[i : i + len(out_mpmd_def)],
                    mpmd_mesh=self.mpmd_mesh,
                    mpmd_idxs=frozenset(out_mpmd_def),
                )
            )
            i += len(out_mpmd_def)
        return jax.tree.unflatten(self.in_info.out_tree, actual_outputs)

    def mpmdify(self):
        closed_jaxpr = self.closed_jaxpr
        in_mpmd_defs = self.in_info.in_mpmd_defs
        out_mpmd_defs = self.in_info.out_mpmd_defs

        with_transfers = maybe_unroll_loop(closed_jaxpr)
        # TODO: move fixup_multidefs right after coarsening.
        #  This is slightly challenging as we need to deduplicate_invars
        #  for the loop body too
        pp_ctx = jcore.JaxprPpContext()
        with_transfers, out_placement = fixup_multidefs(with_transfers)
        # TODO: check also `in_mpmd_defs` similarly to out_placement
        assert list(out_placement) == list(out_mpmd_defs), (
            list(out_placement),
            out_mpmd_defs,
        )
        with_transfers = with_transfers.map_jaxpr(
            partial(common_passes, donated_invars=self.in_info.in_donated)
        )
        dump_jaxpr(with_transfers, name=f"{self.name}.global", ctx=pp_ctx)

        if (
            not env_vars.jaxpp_debug_force_mpmdify.value
            and not self.mpmd_mesh.jax_mesh.is_multi_process
        ):
            with_transfers = bind_meshes(with_transfers, self.mpmd_mesh)
            # jaxprs = scalarize(with_transfers, self.mpmd_mesh)
            return dataclasses.replace(
                self,
                closed_jaxpr=with_transfers,
                in_info=dataclasses.replace(self.in_info, out_mpmd_defs=out_placement),
            )

        jaxprs = scalarize(with_transfers, self.mpmd_mesh)
        check_scalar_jaxprs(with_transfers.jaxpr, jaxprs, in_mpmd_defs, out_placement)

        dump_jaxpr(
            jaxprs[self.mpmd_mesh.my_mpmd_axis_index],
            name=f"{self.name}.local",
            ctx=pp_ctx,
        )

        return ScalarMpmdFunction(
            global_jaxpr=with_transfers,
            local_jaxpr=jcore.ClosedJaxpr(
                jaxprs[self.mpmd_mesh.my_mpmd_axis_index],
                [
                    c
                    for idx, c in enumerate(with_transfers.jaxpr.constvars)
                    if self.mpmd_mesh.my_mpmd_axis_index in in_mpmd_defs[idx]
                ],
            ),
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
    add_kwargs = {}
    if jax.__version_info__ <= (0, 8, 1):
        add_kwargs = {"abstracted_axes": None}

    pjit_info = _parse_jit_arguments(
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
        **add_kwargs,
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


def mpmd_jit_rev(
    fun: Callable,
    mpmd_mesh: MpmdMesh,
    *,
    out_refs,
    in_shardings=None,
    out_shardings=None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> TraceableFunction:
    return _mpmd_jit(
        fun=fun,
        mpmd_mesh=mpmd_mesh,
        strategy=FunctionReverse(out_refs=out_refs),
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        compiler_options=compiler_options,
    )
