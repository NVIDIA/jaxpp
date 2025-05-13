# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import enum
import itertools as it
from collections.abc import Callable
from typing import Any, Iterable, Sequence, TypeVar

import jax
import jax._src.core as jcore
import jax._src.util as ju
import jax.interpreters.partial_eval as pe
import jax.numpy as jnp
from jax._src.ad_checkpoint import remat_p

from jaxpp.jax_primitives import dax_pscan_p
from jaxpp.jaxpr_utils import eqns_free_vars, nonlit, substitute
from jaxpp.utils import CtxVar, array_bytes

T = TypeVar("T")


def freeze_if_set(v: Any):
    if isinstance(v, set):
        return frozenset(v)
    return v


def hashable_params(params: dict[str, Any]):
    return tuple((k, freeze_if_set(v)) for k, v in params.items())


def schedule(
    vs: Iterable[jcore.Var],
    mut_defns: dict[jcore.Var, jcore.JaxprEqn],
    /,
    *,
    is_defined: Callable[[jcore.Var], bool],
) -> tuple[list[jcore.JaxprEqn], set[jcore.Var]]:
    now_defined = set[jcore.Var]()

    res = list[jcore.JaxprEqn]()
    stack = list(reversed(list(vs)))
    while len(stack) > 0:
        if stack[-1] in now_defined:
            stack.pop()
            continue

        defn_eqn = mut_defns[stack[-1]]
        not_visited = []
        for invar in nonlit(defn_eqn.invars):
            if invar in mut_defns:
                if invar not in now_defined:
                    not_visited.append(invar)
            else:
                assert is_defined(invar)

        if len(not_visited) > 0:
            stack.extend(not_visited)
        else:
            res.append(defn_eqn)
            now_defined.update(defn_eqn.outvars)

    return res, now_defined


class PartialValue(enum.Enum):
    UNKNOWN = 0
    TRIVIALLY_KNOWN = 1
    KNOWN = 2


PartialEvalRuleResult = list[tuple[PartialValue, jcore.JaxprEqn]]
PartialEvalRule = Callable[[jcore.JaxprEqn, list[PartialValue]], PartialEvalRuleResult]


partial_eval_custom_rules = CtxVar(dict[jcore.Primitive, PartialEvalRule]())


def partial_eval_eqns(
    eqns: list[jcore.JaxprEqn], env: dict[jcore.Var, PartialValue]
) -> tuple[list[jcore.JaxprEqn], list[jcore.JaxprEqn]]:
    known_eqns = []
    unknown_eqns = []

    trivially_known_defns = dict[jcore.Var, jcore.JaxprEqn]()

    def maybe_define_triv_known(
        v: jcore.Var, as_: PartialValue, into: list[jcore.JaxprEqn]
    ) -> None:
        if v in trivially_known_defns:
            assert env[v] == PartialValue.TRIVIALLY_KNOWN
            eqns, defined_vars = schedule(
                (v,), trivially_known_defns, is_defined=env.__contains__
            )
            # NOTE: we don't replicate trivial definitions although we could
            ju.safe_map(trivially_known_defns.pop, defined_vars)

            into.extend(eqns)
            for dvar in defined_vars:
                if (ex := env.get(dvar)) is not None:
                    assert ex == PartialValue.TRIVIALLY_KNOWN
                env[dvar] = as_

    custom_rules = partial_eval_custom_rules.value
    for eqn in eqns:
        in_vals = [
            env[invar] if isinstance(invar, jcore.Var) else PartialValue.TRIVIALLY_KNOWN
            for invar in eqn.invars
        ]

        rule = custom_rules.get(eqn.primitive, pe_rule_default)
        results = rule(eqn, in_vals)

        for ty, e in results:
            if ty == PartialValue.TRIVIALLY_KNOWN:
                for outvar in e.outvars:
                    trivially_known_defns[outvar] = e
                    env[outvar] = PartialValue.TRIVIALLY_KNOWN
            else:
                into = {
                    PartialValue.KNOWN: known_eqns,
                    PartialValue.UNKNOWN: unknown_eqns,
                }[ty]

                # FIXME: currently if a TRIVIALLY_KNOWN var is used by both
                #  a KNOWN equation and an UNKOWN equation then that var is
                #  defined as KNOWN or UNKOWN depending on which use comes first.
                #  It would be better that if the first use is UNKOWN we further
                #  delay its definition and if another use is KNOWN then we schedule
                #  this delayed equation as KNOWN.
                for invar in nonlit(e.invars):
                    if env[invar] == PartialValue.TRIVIALLY_KNOWN:
                        maybe_define_triv_known(invar, as_=ty, into=into)

                into.append(e)
                env.update(zip(e.outvars, it.repeat(ty)))

    while len(trivially_known_defns) > 0:
        maybe_define_triv_known(
            next(iter(trivially_known_defns)), as_=PartialValue.KNOWN, into=known_eqns
        )

    return known_eqns, unknown_eqns


def partial_eval_jaxpr(
    jaxpr: jcore.Jaxpr, known_invars: Iterable[bool], memory_scarce: bool = False
) -> tuple[
    jcore.Jaxpr | None,
    jcore.Jaxpr | None,
    list[int],
    list[bool],
    list[jcore.AbstractValue],
]:
    known_eqns, unknown_eqns = partial_eval_eqns(
        jaxpr.eqns,
        {
            invar: PartialValue.KNOWN if known else PartialValue.UNKNOWN
            for invar, known in zip(jaxpr.invars, known_invars)
        },
    )
    if memory_scarce:
        new_known_eqns = list[jcore.JaxprEqn]()
        for eqn in known_eqns:
            if eqn.primitive is remat_p:
                j = eqn.params["jaxpr"]
                sub = dict(zip(j.invars, eqn.invars))
                sub.update(zip(j.outvars, eqn.outvars))
                new_known_eqns.extend(substitute(j.eqns, sub))
            else:
                new_known_eqns.append(eqn)
        known_eqns = new_known_eqns

        unknown_free, _ = eqns_free_vars(unknown_eqns)

        known_to_unknown = []
        true_known = list[jcore.JaxprEqn]()
        known_results = set[jcore.Var](nonlit(jaxpr.outvars))
        for eqn in known_eqns[::-1]:
            used_only_in_unknown = all(
                outvar in unknown_free and outvar not in known_results
                for outvar in eqn.outvars
            )
            is_expansion = array_bytes(
                outvar.aval for outvar in eqn.outvars
            ) >= array_bytes(invar.aval for invar in nonlit(eqn.invars))
            if (
                used_only_in_unknown
                and is_expansion
                and eqn.primitive
                in {
                    jax.lax.broadcast_in_dim_p,
                    jax.lax.convert_element_type_p,
                    jax.lax.add_p,
                }
            ):
                unknown_free.update(nonlit(eqn.invars))
                known_to_unknown.append(eqn)
                continue
            else:
                pass
            known_results.update(nonlit(eqn.invars))
            true_known.append(eqn)

        unknown_eqns = known_to_unknown[::-1] + unknown_eqns
        known_eqns = true_known[::-1]

    unknown_free, unknown_defined = eqns_free_vars(unknown_eqns, ordered=True)
    known_free, known_defined = eqns_free_vars(known_eqns, ordered=True)

    # known_jaxpr uses a subset of _only_ `known_invars`.
    # Some of them might be unused but we leave them there anyways.
    # fmt: off
    def check_known_invars():
        known_vars = {i for i, known in zip(jaxpr.invars, known_invars) if known}
        for v in known_free:
            if v not in known_vars:
                raise AssertionError()
    check_known_invars()
    _, known_jaxpr_invars = ju.partition_list(tuple(known_invars), jaxpr.invars)
    # fmt: on

    # Some of the original invars might be used by both the
    # known and unknown jaxprs.
    # JAX's partial_eval instead threads these variables as residuals
    # of the known_jaxpr, potentially being a "redundant" forwarding
    unknown_in_idx = list[int]()
    for invar_idx, invar in enumerate(jaxpr.invars):
        if invar in unknown_free:
            unknown_in_idx.append(invar_idx)

    # Any invar of unknown that is not in the original invar_set
    # must be a residual coming from the known_jaxpr definitions
    invar_set = set(jaxpr.invars)
    residuals = [invar for invar in unknown_free if invar not in invar_set]
    assert all(r in known_defined for r in residuals)

    known_jaxpr = jcore.Jaxpr(
        (),
        known_jaxpr_invars,
        outvars=[
            outvar
            for outvar in jaxpr.outvars
            if isinstance(outvar, jcore.Literal) or outvar in known_defined
        ]
        + list(residuals),
        eqns=known_eqns,
        effects=jcore.join_effects(*(eqn.effects for eqn in known_eqns)),
    )

    out_is_unknown = [
        isinstance(outvar, jcore.Var) and outvar in unknown_defined
        for outvar in jaxpr.outvars
    ]
    unknown_jaxpr = jcore.Jaxpr(
        (),
        list(residuals) + [jaxpr.invars[idx] for idx in unknown_in_idx],
        outvars=[
            outvar
            for outvar, is_unknown in zip(jaxpr.outvars, out_is_unknown, strict=True)
            if is_unknown
        ],
        eqns=unknown_eqns,
        effects=jcore.join_effects(*(eqn.effects for eqn in unknown_eqns)),
    )

    # defensive sanity check
    assert all(
        invar in known_jaxpr.outvars or invar in jaxpr.invars
        for invar in unknown_jaxpr.invars
    )

    return (
        known_jaxpr if len(known_eqns) > 0 else None,
        unknown_jaxpr if len(unknown_eqns) > 0 else None,
        unknown_in_idx,
        out_is_unknown,
        [r.aval for r in residuals],
    )


def pe_rule_convert(
    eqn: jcore.JaxprEqn, in_vals: list[PartialValue]
) -> PartialEvalRuleResult:
    # NOTE: this is for XLA's pattern for fp8
    # we force the eqn as unknown to ensure that we trigger XLA's gemm_rewriter
    """
    bqi:f8_e4m3fn[16,64] = convert_element_type[
    new_dtype=float8_e4m3fn
    weak_type=False
    ] bqh
    bqj:bf16[16,64] = convert_element_type[new_dtype=bfloat16 weak_type=False] bqi
    bqm:bf16[16,64] = mul bqj bql
    bqn:bf16[2,4,2048,64] = dot_general[
    dimension_numbers=(([3], [0]), ([], []))
    precision=(Precision.DEFAULT, Precision.DEFAULT)
    ] bpu bqm
    """
    if eqn.params["new_dtype"] == jnp.float8_e4m3fn or any(
        v == PartialValue.UNKNOWN for v in in_vals
    ):
        return [(PartialValue.UNKNOWN, eqn)]
    if all(v == PartialValue.TRIVIALLY_KNOWN for v in in_vals):
        return [(PartialValue.TRIVIALLY_KNOWN, eqn)]
    return [(PartialValue.KNOWN, eqn)]


def pe_rule_default(eqn: jcore.JaxprEqn, in_vals: list[PartialValue]):
    if all(v == PartialValue.TRIVIALLY_KNOWN for v in in_vals):
        return [(PartialValue.TRIVIALLY_KNOWN, eqn)]
    if any(v == PartialValue.UNKNOWN for v in in_vals):
        return [(PartialValue.UNKNOWN, eqn)]
    return [(PartialValue.KNOWN, eqn)]


def pe_rule_remat(
    eqn: jcore.JaxprEqn, in_vals: list[PartialValue]
) -> PartialEvalRuleResult:
    jaxpr: jcore.Jaxpr = eqn.params["jaxpr"]
    in_known = [v == PartialValue.KNOWN for v in in_vals]
    known_jaxpr, unknown_jaxpr, unknown_in_idx, out_is_unknown, residual_avals = (
        partial_eval_jaxpr(jaxpr, in_known)
    )

    if known_jaxpr is None:
        return [(PartialValue.UNKNOWN, eqn)]

    if unknown_jaxpr is None:
        return [(PartialValue.KNOWN, eqn)]

    gensym = jcore.gensym()
    residual_outvars = [gensym(aval) for aval in residual_avals]

    _, known_invars = ju.partition_list(in_known, eqn.invars)
    known_outvars, unknown_outvars = ju.partition_list(out_is_unknown, eqn.outvars)

    known_eqn = eqn.replace(
        params={**eqn.params, "jaxpr": known_jaxpr},
        invars=known_invars,
        outvars=known_outvars + residual_outvars,
        effects=known_jaxpr.effects,
    )

    unknown_eqn = eqn.replace(
        params={**eqn.params, "jaxpr": unknown_jaxpr},
        invars=residual_outvars + [eqn.invars[in_idx] for in_idx in unknown_in_idx],
        outvars=unknown_outvars,
        effects=unknown_jaxpr.effects,
    )

    assert all(invar in eqn.invars for invar in known_eqn.invars)
    assert all(
        invar in eqn.invars or invar in known_eqn.outvars
        for invar in unknown_eqn.invars
    )
    return [(PartialValue.KNOWN, known_eqn), (PartialValue.UNKNOWN, unknown_eqn)]


def partial_eval_loop(
    default_process_primitive, primitive, tracers, params, cross_remat: bool = False
):
    assert primitive is dax_pscan_p
    n_consts = params["n_consts"]
    in_known = (True,) * n_consts + (False,) * (len(tracers) - n_consts)

    rules = {jax.lax.convert_element_type_p: pe_rule_convert}
    if cross_remat:
        rules[remat_p] = pe_rule_remat

    with partial_eval_custom_rules.set(to=rules):
        (known_jaxpr, unknown_jaxpr, unknown_in_idx, out_is_unknown, res_avals) = (
            partial_eval_jaxpr(params["jaxpr"].jaxpr, in_known, memory_scarce=True)
        )
        if not all(out_is_unknown):
            raise NotImplementedError()  # FIXME

    known_out_tracers = []
    if known_jaxpr is not None:
        known_out_tracers = jcore.eval_jaxpr(
            known_jaxpr, (), *tracers[:n_consts], propagate_source_info=False
        )

    return default_process_primitive(
        primitive,
        (
            *known_out_tracers[-len(res_avals) :],
            *(tracers[idx] for idx in unknown_in_idx),
        ),
        {
            **params,
            "jaxpr": jcore.ClosedJaxpr(unknown_jaxpr, ()),
            "n_consts": len(res_avals) + sum(idx < n_consts for idx in unknown_in_idx),
        },
    )


class CommonSubexpressionEliminationTrace(pe.DynamicJaxprTrace):
    def __init__(self, debug_info, cross_remat: bool):
        super().__init__(debug_info)
        self.cross_remat = cross_remat
        self.equation_recipe_to_tracers_cache = dict[
            tuple[jcore.Primitive, Sequence[int], Any],
            Sequence[pe.DynamicJaxprTracer] | pe.DynamicJaxprTracer,
        ]()

    def default_process_primitive(self, primitive, tracers, params):
        if primitive is dax_pscan_p:
            with jcore.set_current_trace(self):
                return partial_eval_loop(
                    # NOTE: passing `super()` to avoid infinite recursion when `process_pscan`
                    #  will `dax_pscan_p.bind` the licmed loop
                    super().default_process_primitive,
                    primitive,
                    tracers,
                    params,
                    cross_remat=self.cross_remat,
                )

        avals = [t.aval for t in tracers]
        _, effects = primitive.abstract_eval(*avals, **params)

        has_side_effects = len(effects) > 0
        key = None
        if not has_side_effects:
            key = (primitive, tuple(map(id, tracers)), hashable_params(params))
            try:
                maybe_out_tracers = self.equation_recipe_to_tracers_cache.get(key)
                if maybe_out_tracers is not None:
                    return maybe_out_tracers
            except TypeError:
                key = None

        out_tracers = super().default_process_primitive(primitive, tracers, params)
        if key is not None:
            assert key not in self.equation_recipe_to_tracers_cache
            self.equation_recipe_to_tracers_cache[key] = out_tracers
        return out_tracers


def hoist_and_cse_pscan_invariant_equations(
    jaxpr: jcore.Jaxpr, cross_remat: bool = True
):
    assert len(jaxpr.constvars) == 0
    trace = CommonSubexpressionEliminationTrace(
        jaxpr.debug_info, cross_remat=cross_remat
    )
    new_arg_kwargs = {}
    if jax.__version_info__ > (0, 6, 0):
        from jax._src import source_info_util

        new_arg_kwargs = dict(source_info=source_info_util.current())
    with jcore.set_current_trace(trace):
        out_tracers = jcore.eval_jaxpr(
            jaxpr, (), *(trace.new_arg(a.aval, **new_arg_kwargs) for a in jaxpr.invars)
        )
    new_jaxpr, consts, _ = trace.to_jaxpr(out_tracers, jaxpr.debug_info)
    assert len(consts) == 0
    return new_jaxpr
