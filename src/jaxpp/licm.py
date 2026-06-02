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

import dataclasses
import enum
import functools
import itertools as it
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import jax
import jax.extend.source_info_util as jsiu
import jax.numpy as jnp

from jaxpp import jax_compat as jc
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import dax_pscan_p, place_with_p
from jaxpp.jaxpr_utils import eqns_free_vars, nonlit, substitute, var_is_duplicate
from jaxpp.jaxpr_utils import gensym as mk_gensym
from jaxpp.utils import OverwriteableVar, array_bytes

T = TypeVar("T")


def freeze_if_mut(v: Any):
    if isinstance(v, list):
        return tuple(v)
    if isinstance(v, set):
        return frozenset(v)
    return v


def hashable_params(params: dict[str, Any], exclude: set[str] | None = None):
    if exclude is None:
        exclude = set()

    return tuple((k, freeze_if_mut(v)) for k, v in params.items() if k not in exclude)


class PartialValue(enum.Enum):
    UNKNOWN = 0
    TRIVIALLY_KNOWN = 1
    KNOWN = 2


PartialEvalRuleResult = list[tuple[PartialValue, jcore.JaxprEqn]]
PartialEvalRule = Callable[[jcore.JaxprEqn, list[PartialValue]], PartialEvalRuleResult]


partial_eval_custom_rules = OverwriteableVar(dict[jcore.Primitive, PartialEvalRule]())


@dataclasses.dataclass(frozen=True)
class _SeededCustomEqn:
    partial_value: PartialValue
    eqn: jcore.JaxprEqn


def partial_eval_eqns(
    eqns: Sequence[jcore.JaxprEqn],
    env: dict[jcore.Var, PartialValue],
    demanded_outvars: Sequence[jcore.Atom] | None = None,
) -> tuple[list[jcore.JaxprEqn], list[jcore.JaxprEqn]]:
    """Split equations with seeded partial-eval semantics.

    This is a small local fork of JAX's ``partial_eval_jaxpr_custom`` idea, but
    with a third provenance state:

    * ``UNKNOWN``: depends on loop-carried values and must stay staged.
    * ``KNOWN``: depends on a pscan const seed and is worth hoisting.
    * ``TRIVIALLY_KNOWN``: loop-invariant but unseeded; stage/recompute it only
      when demanded by the loop, and hoist it only when needed by seeded work.

    ``demanded_outvars=None`` preserves the old equation-local API by treating
    every non-trivial equation output as demanded.  ``partial_eval_jaxpr`` passes
    the jaxpr outputs instead, which avoids hoisting dead seeded computations.
    """
    expanded_eqns = []

    custom_rules = partial_eval_custom_rules.value
    # Propagate seeded/unknown provenance through the equations.
    for eqn in eqns:
        in_vals = [
            env[invar] if isinstance(invar, jcore.Var) else PartialValue.TRIVIALLY_KNOWN
            for invar in eqn.invars
        ]

        rule = custom_rules.get(eqn.primitive, pe_rule_default)
        results = rule(eqn, in_vals)

        for ty, e in results:
            # Effects must stay staged even if a custom rule marks outputs known.
            if e.effects:
                ty = PartialValue.UNKNOWN
            expanded_eqns.append(_SeededCustomEqn(ty, e))
            env.update(zip(e.outvars, it.repeat(ty)))

    known_needed = set[jcore.Var]()
    unknown_needed = set[jcore.Var]()

    if demanded_outvars is None:
        for item in expanded_eqns:
            eqn = item.eqn
            if item.partial_value == PartialValue.KNOWN:
                known_needed.update(nonlit(eqn.outvars))
            elif item.partial_value == PartialValue.UNKNOWN:
                unknown_needed.update(nonlit(eqn.outvars))
    else:
        for outvar in demanded_outvars:
            if not isinstance(outvar, jcore.Var):
                continue
            ty = env[outvar]
            if ty == PartialValue.KNOWN:
                known_needed.add(outvar)
            else:
                # Keep unseeded loop-invariant outputs in the staged jaxpr rather
                # than hoisting them merely because they are statically known.
                unknown_needed.add(outvar)

    known_eqns_rev = []
    unknown_eqns_rev = []

    for item in reversed(expanded_eqns):
        eqn = item.eqn
        ty = item.partial_value
        outvars = nonlit(eqn.outvars)
        needed_known = any(outvar in known_needed for outvar in outvars)
        needed_unknown = any(outvar in unknown_needed for outvar in outvars)
        if eqn.effects:
            needed_unknown = True

        if needed_known and ty == PartialValue.UNKNOWN:
            # This should be unreachable for well-formed provenance, but staging
            # is the conservative answer if a custom rule misclassifies demand.
            needed_known = False
            needed_unknown = True

        if needed_known or (needed_unknown and ty == PartialValue.KNOWN):
            known_eqns_rev.append(eqn)
            known_needed.update(nonlit(eqn.invars))
            continue

        if needed_unknown:
            unknown_eqns_rev.append(eqn)
            for invar in nonlit(eqn.invars):
                if env[invar] == PartialValue.KNOWN:
                    known_needed.add(invar)
                else:
                    unknown_needed.add(invar)

    return known_eqns_rev[::-1], unknown_eqns_rev[::-1]


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory:
            dict.__setitem__(self, key, self.default_factory(key))
            return self[key]
        else:
            defaultdict.__missing__(self, key)


def inline_eqns(
    eqns: list[jcore.JaxprEqn],
    env: dict[jcore.Var, jcore.Atom],
    result_binding: Mapping | None = None,
):
    if result_binding is None:
        result_binding = {}

    env = dict(env)
    res_eqns = []
    for eqn in eqns:
        invars = [
            env[invar] if not isinstance(invar, jcore.Literal) else invar
            for invar in eqn.invars
        ]
        outvars = [
            r if (r := result_binding.get(outvar)) is not None else outvar
            for outvar in eqn.outvars
        ]
        res_eqns.append(eqn.replace(invars=invars, outvars=outvars))
        for eqn_outvar, outvar in zip(eqn.outvars, outvars):
            assert eqn_outvar not in env
            env[eqn_outvar] = outvar
    return res_eqns


def cpy_eqn(invar, as_):
    return jcore.new_jaxpr_eqn(
        invars=[invar],
        outvars=[as_],
        primitive=jax.lax.copy_p,
        params={},
        effects=frozenset({}),
    )


def outvar_normalization(jaxpr: jcore.Jaxpr):
    assert len(jaxpr.constvars) == 0

    jaxpr_invars = {v: idx for idx, v in enumerate(jaxpr.invars)}

    outvar_forwards_invar = list[int | None]()
    for outvar in jaxpr.outvars:
        invar_idx = None
        if not isinstance(outvar, jcore.Literal):
            invar_idx = jaxpr_invars.get(outvar)
        outvar_forwards_invar.append(invar_idx)

    duplicate_idx = var_is_duplicate(jaxpr.outvars, mark_first=True)

    gensym = mk_gensym()
    copy_eqns = list[jcore.JaxprEqn]()
    outvars = list[jcore.Atom]()
    for outvar, invar_idx, dup_idx in zip(
        jaxpr.outvars, outvar_forwards_invar, duplicate_idx, strict=True
    ):
        # An output can be a
        # (1) Literal
        if isinstance(outvar, jcore.Literal):
            # Bind it under the name `r`
            new_outvar = gensym(outvar.aval)
            eqn = cpy_eqn(outvar, as_=new_outvar)
            copy_eqns.append(eqn)
        # (2) The input jaxpr.invars[invar_idx]
        elif invar_idx is not None:
            new_outvar = gensym(outvar.aval)
            eqn = cpy_eqn(jaxpr.invars[invar_idx], as_=new_outvar)
            copy_eqns.append(eqn)

        # (3) The same output jaxpr.outvars[dup_idx] returned multiple times
        elif dup_idx is not None:
            # `results[dup_idx]` is returned more than once under different names
            new_outvar = gensym(outvar.aval)
            eqn = cpy_eqn(jaxpr.outvars[dup_idx], as_=new_outvar)
            copy_eqns.append(eqn)
        else:
            new_outvar = outvar

        outvars.append(new_outvar)
    return jaxpr.replace(eqns=jaxpr.eqns + copy_eqns, outvars=outvars)


def inline_jaxpr(
    jaxpr: jcore.Jaxpr,
    consts: list[jcore.Atom],
    args: list[jcore.Atom],
    results: list[jcore.Var],
) -> list[jcore.JaxprEqn]:
    """
    Returns new jaxpr.eqns that can be inlined into other contexts
    where `args` where passed for `jaxpr.invars` and the results
    were bound to `results`.
    It does so by rebinding the variables in the equations to their names in the
    calling context, and freshens the other variables so they don't clash with
    existing ones in such calling context.
    """
    assert len(results) == len(set(results))
    jaxpr = jc.convert_constvars_jaxpr(jaxpr)

    jaxpr_invars = {v: idx for idx, v in enumerate(jaxpr.invars)}

    outvar_forwards_invar = list[int | None]()
    for outvar in jaxpr.outvars:
        invar_idx = None
        if not isinstance(outvar, jcore.Literal):
            invar_idx = jaxpr_invars.get(outvar)
        outvar_forwards_invar.append(invar_idx)

    env = dict(zip(jaxpr.invars, consts + args, strict=True))

    copy_eqns = []
    duplicate_idx = var_is_duplicate(jaxpr.outvars)

    result_binding = dict[jcore.Var, jcore.Var]()
    for outvar, invar_idx, dup_idx, r in zip(
        jaxpr.outvars, outvar_forwards_invar, duplicate_idx, results, strict=True
    ):
        # An output can be a
        # (1) Literal
        if isinstance(outvar, jcore.Literal):
            # Bind it under the name `r` in the calling context.
            eqn = cpy_eqn(outvar, as_=r)
            copy_eqns.append(eqn)
        # (2) The input jaxpr.invars[invar_idx]
        elif invar_idx is not None:
            # In the calling context it means that args[invar_idx]
            # is just renamed into `r`
            eqn = cpy_eqn(args[invar_idx], as_=r)
            copy_eqns.append(eqn)
        # (3) The same output jaxpr.outvars[dup_idx] returned multiple times
        elif dup_idx is not None:
            # In the calling context, `results[dup_idx]` is returned more than
            # once under different names. We must copy it for the new names.
            eqn = cpy_eqn(results[dup_idx], as_=r)
            copy_eqns.append(eqn)
        # (4) Defined by an equation
        else:
            assert outvar not in result_binding
            result_binding[outvar] = r

    gensym = mk_gensym()
    eqns = inline_eqns(
        jaxpr.eqns, env, KeyDefaultDict(lambda v: gensym(v.aval), result_binding)
    )
    return eqns + copy_eqns


def partial_eval_jaxpr(
    jaxpr: jcore.Jaxpr,
    known_invars: Iterable[PartialValue],
    memory_scarce: bool = False,
) -> tuple[
    jcore.Jaxpr | None,
    jcore.Jaxpr | None,
    list[int],
    list[bool],
    list[jcore.AbstractValue],
]:
    jaxpr = outvar_normalization(jaxpr)
    in_vals = list(known_invars)
    known_invars = [v is not PartialValue.UNKNOWN for v in in_vals]
    known_eqns, unknown_eqns = partial_eval_eqns(
        jaxpr.eqns,
        {invar: val for invar, val in zip(jaxpr.invars, in_vals, strict=True)},
        demanded_outvars=jaxpr.outvars,
    )
    if memory_scarce:
        new_known_eqns = list[jcore.JaxprEqn]()
        for eqn in known_eqns:
            if eqn.primitive is jc.remat_p:
                j: jcore.Jaxpr = eqn.params["jaxpr"]
                eqns = inline_jaxpr(j, j.constvars, eqn.invars, results=eqn.outvars)
                new_known_eqns.extend(eqns)
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

    return make_unzipped_jaxprs(jaxpr, known_invars, known_eqns, unknown_eqns)


def make_unzipped_jaxprs(
    jaxpr: jcore.Jaxpr,
    known_invars: Iterable[bool],
    known_eqns: list[jcore.JaxprEqn],
    unknown_eqns: list[jcore.JaxprEqn],
):
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
    _, known_jaxpr_invars = jc.partition_list(tuple(known_invars), jaxpr.invars)
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


def pe_rule_place_with(eqn: jcore.JaxprEqn, in_vals: list[PartialValue]):
    return [(PartialValue.UNKNOWN, eqn)]


def pe_rule_default(eqn: jcore.JaxprEqn, in_vals: list[PartialValue]):
    if len(eqn.effects) > 0:
        # Effectful equations, including zero-output debug/inspect callbacks, must
        # stay in the loop so partial evaluation does not change effect execution.
        return [(PartialValue.UNKNOWN, eqn)]
    if all(v == PartialValue.TRIVIALLY_KNOWN for v in in_vals):
        return [(PartialValue.TRIVIALLY_KNOWN, eqn)]
    if any(v == PartialValue.UNKNOWN for v in in_vals):
        return [(PartialValue.UNKNOWN, eqn)]
    return [(PartialValue.KNOWN, eqn)]


def make_unzipped_application(
    eqn,
    in_known,
    known_jaxpr,
    unknown_jaxpr,
    unknown_in_idx,
    out_is_unknown,
    residual_avals,
):
    gensym = mk_gensym()
    residual_outvars = [gensym(aval) for aval in residual_avals]

    _, known_invars = jc.partition_list(in_known, eqn.invars)
    known_outvars, unknown_outvars = jc.partition_list(out_is_unknown, eqn.outvars)

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
    return known_eqn, unknown_eqn


def pe_rule_remat(
    eqn: jcore.JaxprEqn, in_vals: list[PartialValue]
) -> PartialEvalRuleResult:
    jaxpr: jcore.Jaxpr = eqn.params["jaxpr"]
    in_known = [v is not PartialValue.UNKNOWN for v in in_vals]
    known_jaxpr, unknown_jaxpr, unknown_in_idx, out_is_unknown, residual_avals = (
        partial_eval_jaxpr(jaxpr, in_vals)
    )

    if known_jaxpr is None:
        return [(PartialValue.UNKNOWN, eqn)]

    if unknown_jaxpr is None:
        return [(PartialValue.KNOWN, eqn)]

    known_eqn, unknown_eqn = make_unzipped_application(
        eqn,
        in_known,
        known_jaxpr,
        unknown_jaxpr,
        unknown_in_idx,
        out_is_unknown,
        residual_avals,
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
    in_vals = (PartialValue.KNOWN,) * n_consts + (PartialValue.UNKNOWN,) * (
        len(tracers) - n_consts
    )

    rules = {
        jax.lax.convert_element_type_p: pe_rule_convert,
        place_with_p: pe_rule_place_with,
    }
    if cross_remat:
        rules[jc.remat_p] = pe_rule_remat

    with partial_eval_custom_rules.set(to=rules):
        (known_jaxpr, unknown_jaxpr, unknown_in_idx, out_is_unknown, res_avals) = (
            partial_eval_jaxpr(params["jaxpr"].jaxpr, in_vals, memory_scarce=True)
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


class CommonSubexpressionEliminationTrace(jc.DynamicJaxprTrace):
    def __init__(self, debug_info, cross_remat: bool):
        super().__init__(debug_info)
        self.cross_remat = cross_remat
        self.equation_recipe_to_tracers_cache = dict[
            tuple[jcore.Primitive, Sequence[int], Any],
            Sequence[jc.DynamicJaxprTracer] | jc.DynamicJaxprTracer,
        ]()

    def default_process_primitive(self, primitive, tracers, params, source_info=None):
        super_fn = super().default_process_primitive
        if jax.__version_info__ > (0, 6, 1):
            super_fn = functools.partial(super_fn, source_info=source_info)
        if primitive is dax_pscan_p:
            with jcore.set_current_trace(self):
                return partial_eval_loop(
                    # NOTE: passing `super()` to avoid infinite recursion when `process_pscan`
                    #  will `dax_pscan_p.bind` the licmed loop
                    super_fn,
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

        out_tracers = super_fn(primitive, tracers, params)
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

    with jcore.set_current_trace(trace):
        out_tracers = jcore.eval_jaxpr(
            jaxpr,
            (),
            *(trace.new_arg(a.aval, source_info=jsiu.current()) for a in jaxpr.invars),
        )

    source_info = jsiu.current()
    additional_args = (source_info,)
    if jax.__version_info__ >= (0, 8, 0):
        out_tracers = [trace.to_jaxpr_tracer(t, source_info) for t in out_tracers]

    new_jaxpr, consts, *_ = trace.to_jaxpr(
        out_tracers, jaxpr.debug_info, *additional_args
    )
    assert len(consts) == 0
    return remove_duplicate_consts_invars(new_jaxpr)


def remove_duplicate_consts_invars(jaxpr: jcore.Jaxpr):
    from jaxpp.core import get_one_loop_eqn_idx, unwrap_closed

    loop_eqn_idx = get_one_loop_eqn_idx(jaxpr)
    loop_eqn = jaxpr.eqns[loop_eqn_idx]

    duplicate_idx = var_is_duplicate(loop_eqn.invars)
    assert not any(
        duplicate_idx[loop_eqn.params["n_consts"] :]
    ), "Unexpected duplicate in loop carried state"

    kept_invars, duplicate_invars = jc.partition_list(
        [_ is not None for _ in duplicate_idx], loop_eqn.invars
    )
    new_loop_eqn = loop_eqn.replace(
        invars=kept_invars,
        params=loop_eqn.params
        | {
            "jaxpr": unwrap_closed(
                lambda jaxpr: remove_duplicate_invars(jaxpr, duplicate_idx)
            )(loop_eqn.params["jaxpr"]),
            "n_consts": loop_eqn.params["n_consts"] - len(duplicate_invars),
        },
    )

    return jaxpr.replace(
        eqns=jaxpr.eqns[:loop_eqn_idx] + [new_loop_eqn] + jaxpr.eqns[loop_eqn_idx + 1 :]
    )


def remove_duplicate_invars(jaxpr: jcore.Jaxpr, duplicate_idx: list[int | None]):
    sub = dict[jcore.Var, jcore.Var]()
    kept_invars = []
    for invar, dup_idx in zip(jaxpr.invars, duplicate_idx, strict=True):
        if dup_idx is not None:
            sub[invar] = jaxpr.invars[dup_idx]
        else:
            kept_invars.append(invar)

    return jaxpr.replace(invars=kept_invars, eqns=substitute(jaxpr.eqns, sub))
