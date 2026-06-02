# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from jaxpp import jax_compat as jc
from jaxpp.jax_compat import core as jcore
from jaxpp.licm import PartialValue, partial_eval_custom_rules, partial_eval_eqns


def _input_var():
    return jax.make_jaxpr(lambda x: x)(jnp.array(1.0)).jaxpr.invars[0]


def _zero_output_eqn(invar, effects=frozenset()):
    return jcore.new_jaxpr_eqn(
        invars=[invar],
        outvars=[],
        primitive=jc.inspect_sharding_p,
        params={"callback": lambda _: None},
        effects=effects,
    )


def test_zero_output_effectful_eqn_stays_unknown():
    invar = _input_var()
    eqn = _zero_output_eqn(invar, effects=frozenset({jc.debug_effect}))

    with partial_eval_custom_rules.set(to={}):
        known_eqns, unknown_eqns = partial_eval_eqns(
            [eqn], {invar: PartialValue.KNOWN}
        )

    assert known_eqns == []
    assert unknown_eqns == [eqn]


def test_zero_output_pure_eqn_is_dropped():
    invar = _input_var()
    eqn = _zero_output_eqn(invar)

    with partial_eval_custom_rules.set(to={}):
        known_eqns, unknown_eqns = partial_eval_eqns(
            [eqn], {invar: PartialValue.UNKNOWN}
        )

    assert known_eqns == []
    assert unknown_eqns == []


def test_effectful_eqn_with_custom_partial_eval_rule_is_dispatched():
    jaxpr = jax.make_jaxpr(lambda x: x + 1)(jnp.array(1.0)).jaxpr
    eqn = jaxpr.eqns[0].replace(effects=frozenset({jc.debug_effect}))
    calls = []

    def custom_rule(eqn, in_vals):
        calls.append((eqn, in_vals))
        return [(PartialValue.UNKNOWN, eqn)]

    with partial_eval_custom_rules.set(to={eqn.primitive: custom_rule}):
        known_eqns, unknown_eqns = partial_eval_eqns(
            [eqn], {jaxpr.invars[0]: PartialValue.KNOWN}
        )

    assert calls == [(eqn, [PartialValue.KNOWN, PartialValue.TRIVIALLY_KNOWN])]
    assert known_eqns == []
    assert unknown_eqns == [eqn]
