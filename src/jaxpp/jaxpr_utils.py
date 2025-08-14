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

import jax
import jax._src.core as jcore
import jax._src.util as ju
from jax._src.ad_checkpoint import remat_p


def gensym(suffix=""):
    if jax.__version_info__ > (0, 6, 1):
        return jcore.Var
    return jcore.gensym(suffix)


def check_jaxpr(jaxpr: jcore.Jaxpr):
    if os.environ.get("JAXPP_ENABLE_CHECK_JAXPR", "0") == "1":
        jcore.check_jaxpr(jaxpr)
    return jaxpr


def nonlit(atoms: Iterable[jcore.Atom]) -> list[jcore.Var]:
    return [v for v in atoms if isinstance(v, jcore.Var)]


def var_is_duplicate(invars: Iterable[jcore.Atom]) -> list[int | None]:
    existing_index = dict[jcore.Var, int]()
    replace_with_orig_idx = []
    for invar_idx, invar in enumerate(invars):
        idx = None
        if isinstance(invar, jcore.Var):
            idx = existing_index.get(invar)
            if idx is None:
                existing_index[invar] = invar_idx
        replace_with_orig_idx.append(idx)
    return replace_with_orig_idx


def partition_remat(
    eqn: jcore.JaxprEqn, outvar_is_used: Sequence[bool]
) -> tuple[jcore.JaxprEqn, jcore.JaxprEqn]:
    from jaxpp.licm import make_unzipped_application, make_unzipped_jaxprs

    dependencies_eqns, deferred_eqns, dependencies_free = partition_eqns(
        eqn.params["jaxpr"].eqns,
        {
            outvar
            for outvar, used in zip(
                eqn.params["jaxpr"].outvars, outvar_is_used, strict=True
            )
            if used
        },
        memory_scarce=True,
    )

    is_dependencies_in = [
        invar in dependencies_free for invar in eqn.params["jaxpr"].invars
    ]

    (
        dependencies_jaxpr,
        deferred_jaxpr,
        deferred_in_idx,
        out_is_deferred,
        residual_avals,
    ) = make_unzipped_jaxprs(
        eqn.params["jaxpr"],
        known_invars=is_dependencies_in,
        known_eqns=dependencies_eqns,
        unknown_eqns=deferred_eqns,
    )

    return make_unzipped_application(
        eqn,
        is_dependencies_in,
        dependencies_jaxpr,
        deferred_jaxpr,
        deferred_in_idx,
        out_is_deferred,
        residual_avals,
    )


def partition_eqns(
    eqns: Sequence[jcore.JaxprEqn],
    tgt_vars: Iterable[jcore.Var],
    is_partial_bwd=False,
    memory_scarce=False,
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
        outvar_is_used = [outvar in used_vars for outvar in eqn.outvars]
        if any(outvar_is_used):
            if eqn.primitive == remat_p and is_partial_bwd:
                dependencies_eqn, deferred_eqn = partition_remat(eqn, outvar_is_used)

                mut_eqns[eqn_idx] = deferred_eqn
                rev_scheduled_eqns.append(dependencies_eqn)
                used_vars.update(nonlit(dependencies_eqn.invars))
            else:
                mut_eqns[eqn_idx] = None
                rev_scheduled_eqns.append(eqn)
                used_vars.update(nonlit(eqn.invars))
    dependencies = list(reversed(rev_scheduled_eqns))

    def get_total_size(vars: jcore.Var):
        return sum([var.aval.dtype.itemsize * var.aval.size for var in vars])

    if memory_scarce and len(dependencies) > 0 and any(eqn for eqn in mut_eqns):
        # `defs_in_deferred` tracks vars that are defined locally in deferred eqns.
        # Any var in `defs_in_deferred` is not from dependencies or from global invars.
        # For example, in the example below, `a` is moved to dependencies and is not
        # added to `defs_in_deferred`. However, `d` is not moved to dependencies because
        # dot_general is blacklisted and `d` is added to `defs_in_deferred`.
        # Any eqn that depends on `d` won't be moved to dependencies.
        #   a = mul b c
        #   d = dot_general a e
        #   ...
        defs_in_deferred = set[jcore.Var]()
        blacklisted_primitives = (jax.lax.dot_general_p, remat_p)
        for eqn_idx in range(len(mut_eqns)):
            eqn = mut_eqns[eqn_idx]
            if eqn is None:
                continue
            if eqn.primitive in blacklisted_primitives or any(
                invar in defs_in_deferred for invar in nonlit(eqn.invars)
            ):
                defs_in_deferred.update(nonlit(eqn.outvars))
                continue
            invars_size = get_total_size(eqn.invars)
            outvars_size = get_total_size(eqn.outvars)
            if outvars_size <= invars_size:
                dependencies.append(eqn)
                used_vars.update(nonlit(eqn.invars))
                used_vars.update(nonlit(eqn.outvars))
                mut_eqns[eqn_idx] = None
            else:
                defs_in_deferred.update(nonlit(eqn.outvars))
    deferred = [eqn for eqn in mut_eqns if eqn is not None]
    return dependencies, deferred, used_vars


def schedule_dependencies(
    eqns: list[jcore.JaxprEqn], tgt_eqn_idx: int
) -> tuple[list[jcore.JaxprEqn], list[jcore.JaxprEqn]]:
    """
    Partition `eqns` into two parts `eqns = dependencies + deferred` where
    `dependencies` are equations that _must_ be scheduled before `eqns[tgt_eqn_idx]`,
    i.e. equations in `dependencies` define `eqns[tgt_eqn_idx].invars`.
    The relative order of the `deferred` equations is left unchanged.
    """
    dependencies, deferred, _ = partition_eqns(
        eqns[: tgt_eqn_idx + 1], eqns[tgt_eqn_idx].outvars
    )
    return dependencies, (deferred + eqns[tgt_eqn_idx + 1 :])


def eqns_free_vars(
    eqns: Iterable[jcore.JaxprEqn], ordered=False
) -> tuple[set[jcore.Var], set[jcore.Var]]:
    set_ctor = (ju.OrderedSet if ordered else set)[jcore.Var]
    defined = set_ctor()
    free = set_ctor()
    for eqn in eqns:
        free.update(invar for invar in nonlit(eqn.invars) if invar not in defined)
        defined.update(eqn.outvars)
    return (free, defined)


def jaxpr_from_eqns(
    eqns: list[jcore.JaxprEqn], outputs_needed: set[jcore.Var]
) -> jcore.Jaxpr:
    free, defined = eqns_free_vars(eqns, ordered=True)
    jaxpr = jcore.Jaxpr(
        constvars=(),
        invars=list(free),
        outvars=[d for d in defined if d in outputs_needed],
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
