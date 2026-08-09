"""Sharding inference and reconciliation for tasked jaxprs."""

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

import logging
import math
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Annotated, Any, TypeVar, cast

import jax

from jaxpp import env_vars
from jaxpp import jax_compat as jc
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import (
    PjitKwargs,
    add_multi_p,
    callable_task,
    dax_pscan_p,
    gather_multi_p,
    task_p,
)
from jaxpp.mesh import MpmdMesh
from jaxpp.utils import filter_axes, log_elapsed_time, update_named_sharding

logger = logging.getLogger(__name__)

AnyJaxpr = TypeVar("AnyJaxpr", jcore.ClosedJaxpr, jcore.Jaxpr)


class _ShardingStoreCallback:
    def __init__(self, store: "ShardingStore", idx: int):
        self.store = store
        self.idx = idx

    def __call__(self, s: jax.sharding.NamedSharding):
        self.store._called_at_least_once = True
        s.shard_shape(self.store.avals[self.idx].shape)
        self.store._shardings[self.idx] = s


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
        return len(self.avals)

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
        return _ShardingStoreCallback(self, idx)

    @classmethod
    def collect_jaxpr(cls, vars_: Sequence[jcore.Var]):
        store = cls([v.aval for v in vars_])

        res = []
        for idx, v in enumerate(vars_):
            res.append(
                jcore.new_jaxpr_eqn(
                    invars=[v],
                    outvars=[],
                    primitive=jc.inspect_sharding_p,
                    params={"callback": store._callback_at_index(idx)},
                    effects=frozenset({jc.debug_effect}),
                )
            )
        return store, res


def _is_sharding_store_inspect_eqn(eqn: jcore.JaxprEqn) -> bool:
    return eqn.primitive is jc.inspect_sharding_p and isinstance(
        eqn.params.get("callback"), _ShardingStoreCallback
    )


@jc.weakref_lru_cache
def _strip_inspect_sharding_eqns(cjaxpr: AnyJaxpr) -> AnyJaxpr:
    if isinstance(cjaxpr, jcore.ClosedJaxpr):
        jaxpr = cjaxpr.jaxpr
    else:
        jaxpr = cjaxpr

    new_eqns = []
    for eqn in jaxpr.eqns:
        if _is_sharding_store_inspect_eqn(eqn):
            continue
        if eqn.primitive is task_p or eqn.primitive is dax_pscan_p:
            key = ["jaxpr", "call_jaxpr"][eqn.primitive is task_p]
            new_jaxpr = _strip_inspect_sharding_eqns(eqn.params[key])
            new_eqns.append(
                eqn.replace(
                    params={**eqn.params, key: new_jaxpr}, effects=new_jaxpr.effects
                )
            )
        else:
            new_eqns.append(eqn)

    new_effects = jcore.join_effects(*(eqn.effects for eqn in new_eqns))
    res = jaxpr.replace(eqns=new_eqns, effects=new_effects)

    if not isinstance(cjaxpr, jcore.ClosedJaxpr):
        return res
    return cjaxpr.replace(jaxpr=res)


def strip_inspect_sharding_eqns(cjaxpr: AnyJaxpr) -> AnyJaxpr:
    return _strip_inspect_sharding_eqns(cjaxpr)


def more_sharded_sharding(prev_sharding, alt_sharding, shape):
    prev_sharded_shape = prev_sharding.shard_shape(shape)
    sharded_shape = alt_sharding.shard_shape(shape)
    return (
        prev_sharding
        if math.prod(prev_sharded_shape) <= math.prod(sharded_shape)
        else alt_sharding
    )


def _known_sharding(sharding):
    return None if isinstance(sharding, jc.UnspecifiedValue) else sharding


def _add_inspect_sharding_eqns(cjaxpr: AnyJaxpr):
    records = []

    def add(cjaxpr: AnyJaxpr) -> AnyJaxpr:
        is_closed = isinstance(cjaxpr, jcore.ClosedJaxpr)
        jaxpr = cjaxpr.jaxpr if is_closed else cjaxpr
        new_eqns = []
        for eqn in jaxpr.eqns:
            if _is_sharding_store_inspect_eqn(eqn):
                continue

            new_params = None
            new_effects = eqn.effects
            if eqn.primitive is task_p:
                call_jaxpr = add(eqn.params["call_jaxpr"])
                new_params = eqn.params | {"call_jaxpr": call_jaxpr}
                new_effects = call_jaxpr.effects
            elif eqn.primitive is dax_pscan_p:
                loop_jaxpr = add(eqn.params["jaxpr"])
                new_params = eqn.params | {"jaxpr": loop_jaxpr}
                new_effects = loop_jaxpr.effects

            if new_params is not None:
                eqn = eqn.replace(params=new_params, effects=new_effects)

            if "in_shardings" not in eqn.params:
                new_eqns.append(eqn)
                continue

            in_store, in_inspect = ShardingStore.collect_jaxpr(eqn.invars)
            out_store, out_inspect = ShardingStore.collect_jaxpr(eqn.outvars)
            new_eqns.extend(in_inspect)
            new_eqns.append(eqn)
            new_eqns.extend(out_inspect)
            records.append((eqn, in_store, out_store))

        new_jaxpr = jaxpr.replace(
            eqns=new_eqns,
            effects=jcore.join_effects(*(eqn.effects for eqn in new_eqns)),
        )
        if not is_closed:
            return new_jaxpr
        return cjaxpr.replace(jaxpr=new_jaxpr)

    return add(cjaxpr), records


def _write_inspected_shardings(records):
    for eqn, in_store, out_store in records:
        eqn.params["in_shardings"] = tuple(in_store._shardings)
        eqn.params["out_shardings"] = tuple(out_store._shardings)


def reconcile_shardings(
    cjaxpr: jcore.ClosedJaxpr,
    in_shardings,
    out_shardings,
    *,
    _in_labels=None,
    _out_labels=None,
):
    # Resolve every Var to one sharding, then write it into every sharding tuple
    # slot that mentions it. Caller-boundary Vars (the jaxpr in/out contract)
    # are pinned, so they win over inferred proposals; a Var pinned to two
    # different shardings is a conflict. A loop carry-in/out pair must agree.
    # Everything else takes the most-sharded proposal across slots. We warn only
    # when a write overrides a different (non-size-1-equivalent) inferred
    # sharding; that warning's "winner" is derived from the Var's role, so
    # resolution never has to thread it.
    in_shardings = tuple(in_shardings)
    out_shardings = tuple(out_shardings)
    eqns = cjaxpr.eqns
    resolved = dict[jcore.Var, Any]()  # Var -> final sharding
    pinned = dict[jcore.Var, str]()  # boundary Var -> caller label (for warnings)
    carry_vars = set[jcore.Var]()  # Vars that are a loop carry in/out

    if _in_labels is None:
        _in_labels = [f"user-specified input[{i}]" for i in range(len(in_shardings))]
    if _out_labels is None:
        _out_labels = [f"user-specified output[{i}]" for i in range(len(out_shardings))]

    def most_sharded(var, proposal):
        # var's running resolution merged with a new proposal (most-sharded wins).
        cur = resolved.get(var)
        if cur is None or proposal is None:
            return cur if proposal is None else proposal
        return more_sharded_sharding(cur, proposal, var.aval.shape)

    def winner(var):
        return pinned.get(var) or (
            "more-sharded (loop carry)" if var in carry_vars else "more-sharded"
        )

    def equivalent(old, new, ndim):
        return jc.shardings_are_equivalent(old, new, ndim, compare_memkind=False)

    # (1) Pin the caller-boundary Vars.
    for var, s, lbl in (
        *zip(cjaxpr.jaxpr.invars, in_shardings, _in_labels, strict=True),
        *zip(cjaxpr.jaxpr.outvars, out_shardings, _out_labels, strict=True),
    ):
        if not isinstance(var, jcore.Var) or s is None:
            continue
        if var in pinned and resolved[var] != s:
            raise NotImplementedError("conflicting boundary shardings")
        resolved[var], pinned[var] = s, lbl

    # (2) Gather the most-sharded proposal for every non-pinned Var. This must
    # finish across all eqns before coupling carries (3), so a carry can pick up
    # a proposal from a consumer that appears later in `eqns`.
    carry_pairs = []
    for eqn in eqns:
        if eqn.primitive is jc.inspect_sharding_p:
            continue
        for var, s in (
            *zip(eqn.invars, eqn.params["in_shardings"], strict=True),
            *zip(eqn.outvars, eqn.params["out_shardings"], strict=True),
        ):
            s = _known_sharding(s)
            if isinstance(var, jcore.Var) and s is not None and var not in pinned:
                resolved[var] = most_sharded(var, s)
        if eqn.primitive is dax_pscan_p:
            carry_pairs += zip(
                eqn.invars[eqn.params["n_consts"] :], eqn.outvars, strict=True
            )

    # (3) Couple each loop carry: carry-in and carry-out denote one fed-back
    # value, so they share a sharding (a pinned member wins; two pinned and
    # differing is a conflict). Carries never chain (one loop per jaxpr level).
    for invar, outvar in carry_pairs:
        members = [v for v in (invar, outvar) if isinstance(v, jcore.Var)]
        carry_vars.update(members)
        bound = [resolved[v] for v in members if v in pinned]
        if any(s != bound[0] for s in bound[1:]):
            raise NotImplementedError("conflicting boundary shardings")
        shared = bound[0] if bound else None
        if not bound:
            for v in members:
                shared = most_sharded(v, shared)
        for v in members:
            if shared is not None and v not in pinned:
                resolved[v] = shared

    # Write resolutions back into every slot; warn on a non-equivalent override
    # of an inferred sharding; recurse into loop bodies, threading each loop
    # Var's derived winner as the body's boundary label.
    has_unknown = False
    for eqn in eqns:
        if eqn.primitive is jc.inspect_sharding_p:
            continue
        is_loop = eqn.primitive is dax_pscan_p
        target = (
            "loop"
            if is_loop
            else f"task={eqn.params.get('task_name', eqn.primitive.name)}"
        )
        for key, kind, vars_, slots in (
            ("in_shardings", "invar_idx", eqn.invars, eqn.params["in_shardings"]),
            ("out_shardings", "outvar_idx", eqn.outvars, eqn.params["out_shardings"]),
        ):
            updated_slots = []
            for idx, var in enumerate(vars_):
                final = resolved.get(var) if isinstance(var, jcore.Var) else None
                slot = slots[idx]
                if final is not None:
                    old = _known_sharding(slot)
                    if (
                        old is not None
                        and env_vars.jaxpp_warn_reconcile_shardings.value
                        and not equivalent(old, final, var.aval.ndim)
                    ):
                        logger.warning(
                            "reconcile_shardings: %s %s=%d shape=%s "
                            "inferred_sharding=%s reconciled_sharding=%s winner=%s",
                            target,
                            kind,
                            idx,
                            var.aval.shape,
                            old.spec,
                            final.spec,
                            winner(var),
                        )
                    slot = final
                has_unknown |= _known_sharding(slot) is None
                updated_slots.append(slot)
            eqn.params[key] = tuple(updated_slots)
        if is_loop:
            has_unknown |= reconcile_shardings(
                eqn.params["jaxpr"],
                eqn.params["in_shardings"],
                eqn.params["out_shardings"],
                _in_labels=[
                    (
                        winner(v)
                        if isinstance(v, jcore.Var) and v in resolved
                        else f"loop invar_idx={i}"
                    )
                    for i, v in enumerate(eqn.invars)
                ],
                _out_labels=[
                    (
                        winner(v)
                        if isinstance(v, jcore.Var) and v in resolved
                        else f"loop outvar_idx={i}"
                    )
                    for i, v in enumerate(eqn.outvars)
                ],
            )
    return has_unknown


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


@jc.cache()
def _fast_infer_shardings_compiler_options_kvs() -> tuple[tuple[str, Any], ...]:
    compiler_options = {
        "xla_gpu_enable_latency_hiding_scheduler": False,
        "xla_gpu_enable_dynamic_slice_fusion": False,
        "xla_gpu_enable_while_loop_double_buffering": False,
        "xla_llvm_disable_expensive_passes": True,
        "xla_backend_optimization_level": 0,
        "xla_gpu_enable_triton_gemm": False,
        "xla_gpu_autotune_level": 0,
    }
    compiler_options.update(jc.collective_pipelining_off_options_kvs)
    compiler_options["xla_gpu_experimental_enable_fusion_autotuner"] = False
    if jax.__version_info__ < (0, 10):
        compiler_options["xla_gpu_enable_split_k_autotuning"] = False
        compiler_options["xla_gpu_enable_reduction_epilogue_fusion"] = False
    return tuple(compiler_options.items())


def _infer_task_output_shardings(
    call_jaxpr: jcore.ClosedJaxpr,
    in_shardings: tuple[jax.sharding.NamedSharding, ...],
    lowering_mesh: jax.sharding.Mesh,
    compiler_options_kvs: tuple[tuple[str, Any], ...] | None,
):
    params = PjitKwargs(
        jaxpr=call_jaxpr,
        in_shardings=in_shardings,
        out_shardings=jc.UNSPECIFIED,
        in_layouts=(None,) * len(call_jaxpr.in_avals),
        out_layouts=(None,) * len(call_jaxpr.out_avals),
        donated_invars=(False,) * len(in_shardings),
        ctx_mesh=lowering_mesh,
        name=f"infer_shardings2_{id(call_jaxpr)}",
        compiler_options_kvs=compiler_options_kvs,
    )

    with jax.set_mesh(params.ctx_mesh), ensuring_pgle_disabled():
        compiled = callable_task(jc.jit_p, params).lower(*call_jaxpr.in_avals).compile()

    result_shardings = tuple(jax.tree_util.tree_leaves(compiled.output_shardings))
    assert len(result_shardings) == len(call_jaxpr.out_avals), (
        result_shardings,
        call_jaxpr.out_avals,
    )
    return result_shardings


def infer_shardings2(
    closed_jaxpr: jcore.ClosedJaxpr,
    in_shardings,
    lowering_mesh,
    compiler_options_kvs: tuple[tuple[str, Any], ...] | None = None,
):
    # TODO: add support for layouts
    in_shardings = tuple(in_shardings)
    env = dict(zip(closed_jaxpr.jaxpr.invars, in_shardings, strict=True))
    for eqn in closed_jaxpr.eqns:
        eqn: jcore.JaxprEqn

        if eqn.primitive is jc.inspect_sharding_p:
            continue

        # TODO: this might fail for literal args
        _in = tuple(env[invar] for invar in eqn.invars)
        if eqn.primitive is task_p:
            if env_vars.jaxpp_debug_skip_propagation.value:
                result_shardings = (
                    jax.NamedSharding(lowering_mesh, jax.sharding.PartitionSpec()),
                ) * len(eqn.outvars)
            else:
                result_shardings = _infer_task_output_shardings(
                    eqn.params["call_jaxpr"],
                    tuple(_in),
                    lowering_mesh,
                    compiler_options_kvs,
                )

        elif eqn.primitive is dax_pscan_p:
            result_shardings = infer_shardings2(
                eqn.params["jaxpr"], _in, lowering_mesh, compiler_options_kvs
            )
        elif eqn.primitive is add_multi_p:
            result_shardings = (_in[0],)
        elif eqn.primitive is gather_multi_p:
            result_shardings = (_in[0],)
        else:
            raise ValueError(f"Unknown primitive {eqn.primitive}")

        for outvar, sh in zip(eqn.outvars, result_shardings, strict=True):
            env[outvar] = sh

        if "in_shardings" in eqn.params:
            eqn.params["in_shardings"] = tuple(_in)
        if "out_shardings" in eqn.params:
            eqn.params["out_shardings"] = tuple(result_shardings)

    return tuple(
        env[outvar] if not isinstance(outvar, jcore.Literal) else outvar
        for outvar in closed_jaxpr.jaxpr.outvars
    )


def infer_shardings(
    lowering_mesh: jax.sharding.Mesh,
    closed_jaxpr: jcore.ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts,
    out_layouts,
) -> jcore.ClosedJaxpr:
    # deduplicate_task_jaxprs lives in core.py; imported lazily so this module
    # keeps no module-level jaxpp.core import (avoids an import cycle).
    from jaxpp.core import deduplicate_task_jaxprs

    assert all(_ is None for _ in in_layouts)
    assert all(_ is None for _ in out_layouts)
    in_shardings = tuple(in_shardings)
    out_shardings = tuple(out_shardings)

    compiler_options_kvs = (
        _fast_infer_shardings_compiler_options_kvs()
        if env_vars.jaxpp_fast_infer_shardings.value
        else None
    )

    with log_elapsed_time("xla_compilation/infer_shardings"):
        if (
            env_vars.jaxpp_enable_local_propagation.value
            or env_vars.jaxpp_debug_skip_propagation.value
        ):
            closed_jaxpr = deduplicate_task_jaxprs(closed_jaxpr)

            # NOTE: We run per-task sharding inference here, up front, instead of
            # deferring it until after unrolling/scheduling. On the compact
            # pre-unroll task set it is cheap and propagates shardings for the
            # whole program, so at runtime each worker compiles only the tasks it
            # owns (it already knows the boundary shardings produced on other
            # ranks), which spreads the execution compilations across ranks.
            # Deferring until after unrolling, where fusion and the schedule
            # create many more compilation units, would instead make each rank
            # compile more. The tradeoff is that a task is compiled twice: once
            # here for inference and once at runtime for execution.

            _ = infer_shardings2(
                closed_jaxpr, in_shardings, lowering_mesh, compiler_options_kvs
            )
        else:
            closed_jaxpr, inspect_records = _add_inspect_sharding_eqns(closed_jaxpr)
            with ensuring_pgle_disabled():
                jax.jit(
                    jcore.jaxpr_as_fun(closed_jaxpr),
                    in_shardings=in_shardings,
                    out_shardings=list(out_shardings),
                    compiler_options=(
                        dict(compiler_options_kvs)
                        if compiler_options_kvs is not None
                        else None
                    ),
                ).lower(*closed_jaxpr.in_avals).compile()
            _write_inspected_shardings(inspect_records)
            closed_jaxpr = strip_inspect_sharding_eqns(closed_jaxpr)

    # NOTE: mutates sharding stored inside `closed_jaxpr`
    reconcile_shardings(closed_jaxpr, in_shardings, out_shardings)
    return closed_jaxpr


def bind_explicit_shardings(
    closed_jaxpr: Annotated[jcore.ClosedJaxpr, "mutable"],
    mpmd_mesh: MpmdMesh,
    jaxpr_in_shardings: Sequence[jax.sharding.NamedSharding],
    jaxpr_out_shardings: Sequence[jax.sharding.NamedSharding],
):
    """Bind task inputs from dataflow and outputs from annotated avals."""
    jaxpr_in_shardings = tuple(jaxpr_in_shardings)
    jaxpr_out_shardings = tuple(jaxpr_out_shardings)
    # JAX keeps jit in_shardings separate from inner jaxpr input avals. The
    # call boundary therefore initializes the dataflow after task insertion.
    env = dict[jcore.Var, jax.sharding.NamedSharding](
        zip(closed_jaxpr.jaxpr.invars, jaxpr_in_shardings, strict=True)
    )
    lowering_mesh = mpmd_mesh.lowering_mesh()
    requested_out_shardings = {
        outvar: out_sharding
        for outvar, out_sharding in zip(
            closed_jaxpr.jaxpr.outvars, jaxpr_out_shardings, strict=True
        )
        if isinstance(outvar, jcore.Var)
    }

    def maybe_apply_requested_output_memory_kind(
        outvar: jcore.Atom, sharding: jax.sharding.NamedSharding
    ) -> jax.sharding.NamedSharding:
        requested = requested_out_shardings.get(outvar)
        if requested is None:
            return sharding
        return update_named_sharding(sharding, memory_kind=requested.memory_kind)

    for eqn in closed_jaxpr.eqns:
        eqn: jcore.JaxprEqn
        if eqn.primitive is task_p:
            eqn.params["in_shardings"] = tuple(
                update_named_sharding(env[invar], mesh=lowering_mesh)
                for invar in eqn.invars
            )
            eqn_out_shardings = tuple(
                maybe_apply_requested_output_memory_kind(
                    outvar,
                    update_named_sharding(
                        cast(jcore.ShapedArray, outvar.aval).sharding,
                        mesh=lowering_mesh,
                    ),
                )
                for outvar in eqn.outvars
            )
            eqn.params["out_shardings"] = eqn_out_shardings
            for outvar, sh in zip(eqn.outvars, eqn_out_shardings, strict=True):
                env[outvar] = sh

        elif eqn.primitive is dax_pscan_p:
            out_shardings = tuple(
                maybe_apply_requested_output_memory_kind(
                    outvar,
                    update_named_sharding(
                        cast(jcore.ShapedArray, outvar.aval).sharding,
                        mesh=lowering_mesh,
                    ),
                )
                for outvar in eqn.outvars
            )
            bind_explicit_shardings(
                eqn.params["jaxpr"],
                mpmd_mesh,
                tuple(
                    update_named_sharding(env[invar], mesh=lowering_mesh)
                    for invar in eqn.invars
                ),
                out_shardings,
            )
            for outvar, sh in zip(eqn.outvars, out_shardings, strict=True):
                env[outvar] = sh
        elif eqn.primitive in {add_multi_p, gather_multi_p}:
            eqn_in_shardings = tuple(
                update_named_sharding(env[invar], mesh=lowering_mesh)
                for invar in eqn.invars
            )
            assert all(
                jc.shardings_are_equivalent(
                    eqn_in_shardings[0],
                    in_sharding,
                    invar.aval.ndim,
                    compare_memkind=False,
                )
                for invar, in_sharding in zip(eqn.invars, eqn_in_shardings, strict=True)
            ), eqn_in_shardings
            # This axis expands on the collective communication mesh.
            out_source_sharding = maybe_apply_requested_output_memory_kind(
                eqn.outvars[0],
                filter_axes(eqn_in_shardings[0], {mpmd_mesh.mpmd_axis_name}),
            )
            eqn.params["in_shardings"] = eqn_in_shardings
            eqn.params["out_shardings"] = (out_source_sharding,)
            env[eqn.outvars[0]] = out_source_sharding
        else:
            raise NotImplementedError(
                f"Unimplemented equation with primitive {eqn.primitive}"
            )

    for outvar, out_sh in zip(
        closed_jaxpr.jaxpr.outvars, jaxpr_out_shardings, strict=True
    ):
        if isinstance(outvar, jcore.Var):
            sh = env[outvar]
            if not jc.shardings_are_equivalent(
                sh, out_sh, outvar.aval.ndim, compare_memkind=False
            ):
                logger.warning(
                    f"Outvar {outvar} has bound sharding {sh.spec} but requested "
                    f"{out_sh.spec}"
                )
    return closed_jaxpr
