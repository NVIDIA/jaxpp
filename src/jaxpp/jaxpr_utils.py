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

import os
from collections.abc import Sequence
from typing import Iterable, Mapping

import jax._src.core as jcore


def check_jaxpr(jaxpr: jcore.Jaxpr):
    if os.environ.get("JAXPP_ENABLE_CHECK_JAXPR", "0") == "1":
        jcore.check_jaxpr(jaxpr)
    return jaxpr


def nonlit(atoms: Iterable[jcore.Atom]) -> list[jcore.Var]:
    return [v for v in atoms if isinstance(v, jcore.Var)]


def partition_eqns(
    eqns: Sequence[jcore.JaxprEqn], tgt_vars: Iterable[jcore.Var]
) -> tuple[list[jcore.JaxprEqn], list[jcore.JaxprEqn]]:
    """
    Partition `eqns` into two parts:
    - the first part of equations is scheduled based on target vars
    - the second part of equations is left unchanged
    """
    used_vars = set[jcore.Var](tgt_vars)
    mut_eqns = list[jcore.JaxprEqn | None](eqns)
    rev_scheduled_eqns = []
    for eqn_idx in reversed(range(len(eqns))):
        eqn = eqns[eqn_idx]
        if any(outvar in used_vars for outvar in eqn.outvars):
            mut_eqns[eqn_idx] = None
            rev_scheduled_eqns.append(eqn)
            used_vars.update(nonlit(eqn.invars))
    return list(reversed(rev_scheduled_eqns)), [
        eqn for eqn in mut_eqns if eqn is not None
    ]


def schedule_dependencies(
    eqns: list[jcore.JaxprEqn], tgt_eqn_idx: int
) -> tuple[list[jcore.JaxprEqn], list[jcore.JaxprEqn]]:
    """
    Partition `eqns` into two parts `eqns = dependencies + deferred` where
    `dependencies` are equations that _must_ be scheduled before `eqns[tgt_eqn_idx]`,
    i.e. equations in `dependencies` define `eqns[tgt_eqn_idx].invars`.
    The relative order of the `deferred` equations is left unchanged.
    """
    dependencies, deferred = partition_eqns(
        eqns[: tgt_eqn_idx + 1], eqns[tgt_eqn_idx].outvars
    )
    return dependencies, (deferred + eqns[tgt_eqn_idx + 1 :])


def eqns_free_vars(
    eqns: Iterable[jcore.JaxprEqn],
) -> tuple[set[jcore.Var], set[jcore.Var]]:
    defined = set[jcore.Var]()
    free = set[jcore.Var]()
    for eqn in eqns:
        free.update(invar for invar in nonlit(eqn.invars) if invar not in defined)
        defined.update(eqn.outvars)
    return (free, defined)


def jaxpr_from_eqns(
    eqns: list[jcore.JaxprEqn], outputs_needed: set[jcore.Var]
) -> jcore.Jaxpr:
    free, defined = eqns_free_vars(eqns)
    jaxpr = jcore.Jaxpr(
        constvars=(),
        invars=sorted(free),
        outvars=sorted(defined & outputs_needed),
        eqns=eqns,
        effects=jcore.join_effects(*(eqn.effects for eqn in eqns)),
    )
    check_jaxpr(jaxpr)
    return jaxpr


def substitute(
    eqns: Iterable[jcore.JaxprEqn], map_: Mapping[jcore.Var, jcore.Var]
) -> list[jcore.JaxprEqn]:
    new_eqns = []
    for eqn in eqns:
        invars = [
            map_.get(invar, invar) if isinstance(invar, jcore.Var) else invar
            for invar in eqn.invars
        ]
        outvars = [map_.get(outvar, outvar) for outvar in eqn.outvars]
        new_eqns.append(eqn.replace(invars=invars, outvars=outvars))
    return new_eqns
