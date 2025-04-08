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

from collections.abc import Callable
from functools import partial
from typing import TypeVar

import jax
import jax.api_util as jau
import jax.core as jcore
import jax.extend.linear_util as lu
import jax.numpy as jnp
from jax._src.lax.control_flow.common import _initial_style_jaxpr
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe

from jaxpp.core import add_jaxpr_parameters, compute_needed, pushout_add_any
from jaxpp.jax_primitives import dax_pscan_p
from jaxpp.loop_output import MUBATCH_AXIS, LoopOutput, SubLoopOutput
from jaxpp.pipelining import PipelineStageContext
from jaxpp.schedules import BaseSchedule
from jaxpp.utils import log_elapsed_time

Batch = TypeVar("Batch")


def mapped_aval(path, e, n_mubatches: int):
    aval: jcore.ShapedArray = jcore.get_aval(e)
    if aval.shape[MUBATCH_AXIS] != n_mubatches:
        raise TypeError(
            f"{jax.tree_util.keystr(path)} Axis 1 should be the same for all "
            f"arguments: {aval.shape[MUBATCH_AXIS]=} != {n_mubatches}"
        )
    return aval.update(
        shape=(aval.shape[0:MUBATCH_AXIS] + aval.shape[MUBATCH_AXIS + 1 :])
    )


@lu.transformation
def scan_body_loop_output(batch, mubatch_idx, loop_state: LoopOutput):
    new_mubatch_idx = mubatch_idx + 1
    mubatch = jax.tree.map(lambda v: jnp.take(v, mubatch_idx, axis=MUBATCH_AXIS), batch)
    res: LoopOutput = yield ((mubatch,), {})

    yield (new_mubatch_idx, loop_state.update(mubatch_idx, res))


@lu.transformation
def scan_body(batch, mubatch_idx, carry):
    new_mubatch_idx = mubatch_idx + 1
    mubatch = jax.tree.map(lambda v: jnp.take(v, mubatch_idx, axis=MUBATCH_AXIS), batch)
    res = yield ((carry, mubatch), {})
    yield (new_mubatch_idx, res)


def sharding_with_data(
    spmd_axis_name: str, path: jax.tree_util.KeyPath, s: jax.sharding.NamedSharding
):
    assert isinstance(s, jax.sharding.NamedSharding)
    used = {n for ns in s.spec for n in (ns if isinstance(ns, tuple) else (ns,))}
    if spmd_axis_name in used:
        raise ValueError(
            f"mesh axis name {spmd_axis_name} cannot appear in "
            f"out_shardings. Found out_shardings{jax.tree_util.keystr(path)}={s.spec}"
        )
    parsed_pspec = s._parsed_pspec.insert_axis_partitions(0, (spmd_axis_name,))
    return jax.sharding.NamedSharding._from_parsed_pspec(s.mesh, parsed_pspec)


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def pscan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    schedule: BaseSchedule = None,
):
    pass


def pscan_wrapped(fun: lu.WrappedFun, init, xs, length, schedule):
    # NOTE: + 0 needed so that jax doesn't make it a `Literal` argument
    mubatch_idx = jax.lax.zeros_like_array(0) + 0

    flat_args, in_tree = jax.tree_util.tree_flatten((xs, mubatch_idx, init))
    flat_scan_body, out_tree = jau.flatten_fun_nokwargs(fun, in_tree)

    scan_body_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        flat_scan_body, tuple(jcore.get_aval(e) for e in flat_args)
    )

    n_consts = len(consts) + len(jax.tree_util.tree_leaves(xs))
    flat_args = consts + flat_args
    scan_body_jaxpr = pe.convert_constvars_jaxpr(scan_body_jaxpr)

    n_orig_outvars = len(scan_body_jaxpr.outvars)
    scan_body_jaxpr = pushout_add_any(scan_body_jaxpr)
    # FIXME: ensure that it doesn't produce duplicate outvars
    replicated_loop_body_invars, replicated_loop_body_outvars, replace_eqns = (
        compute_needed(scan_body_jaxpr, n_consts)
    )

    scan_body_jaxpr = add_jaxpr_parameters(
        scan_body_jaxpr,
        replicated_loop_body_invars,
        replicated_loop_body_outvars,
        replace_eqns,
    )
    new_flat_args = []
    for idx, arg in enumerate(flat_args):
        if replicas := replicated_loop_body_invars.get(idx, None):
            new_flat_args.append(arg)
            for _ in replicas[1:]:
                new_flat_args.append(arg)
        else:
            new_flat_args.append(arg)
    flat_args = new_flat_args

    out = dax_pscan_p.bind(
        *flat_args,
        jaxpr=pe.close_jaxpr(scan_body_jaxpr),
        n_mubatches=length,
        n_consts=n_consts,
        in_shardings=None,
        out_shardings=None,
        in_mpmd_refs=None,
        out_mpmd_defs=None,
        schedule=schedule,
    )

    new_out = []
    added_vars = 0
    for idx in range(n_orig_outvars):
        if replicated_loop_body_outvars.get(idx) is not None:
            new_out.append(ad.add_jaxvals(out[idx], out[idx + 1]))
            added_vars += 1
        else:
            new_out.append(out[idx + added_vars])
    # NOTE: drop first output which is the loop index
    return jax.tree_util.tree_unflatten(out_tree(), new_out)[1]


# TODO: generalize to accept sharding instead of size
def accumulate_grads(
    fun: Callable[[Batch], SubLoopOutput],
    batch: Batch,
    out_shardings,
    schedule: BaseSchedule,
    spmd_axis_name: str = "data",
) -> SubLoopOutput:
    """
    fun : batched -> (sum, cat, max, last)
    """
    flat_batch = jax.tree_util.tree_leaves(batch)
    first_batch_shape = flat_batch[0].shape
    if len(first_batch_shape) < 2:
        raise TypeError(
            f"Expected batch of shape (dp_size, n_mubatches, ...), {first_batch_shape} "
            "found"
        )
    dp_size, n_mubatches = first_batch_shape[:2]

    vmapped_fun = jax.vmap(fun, spmd_axis_name=spmd_axis_name)

    flat_args, in_tree = jax.tree_util.tree_flatten_with_path((batch,))

    with PipelineStageContext.tracing_scope():
        with log_elapsed_time("jaxpr/first_loop_tracing"):
            body_args = tuple(
                mapped_aval(path, arg, n_mubatches) for path, arg in flat_args
            )
            dbg_info = jau.debug_info(
                accumulate_grads.__name__, vmapped_fun, (body_args,), {}
            )
            jaxpr, _, loop_out_tree = _initial_style_jaxpr(
                vmapped_fun, in_tree, body_args, dbg_info
            )

        node_data = loop_out_tree.node_data()
        if node_data is None:
            raise ValueError("`loop_out_tree.node_data()` shall not be None")

        if not issubclass(node_data[0], LoopOutput):
            raise TypeError(
                f"{accumulate_grads.__name__} body expects a {LoopOutput.__qualname__}"
                "output"
            )

        loop_output: LoopOutput = jax.tree_util.tree_unflatten(
            loop_out_tree, jaxpr.out_avals
        )
        loop_state = loop_output.state(n_mubatches)

        with log_elapsed_time("jaxpr/second_loop_tracing"):
            PipelineStageContext.reset()  # Reset the stage ID counter to 0
            loop_output = pscan_wrapped(
                scan_body_loop_output(lu.wrap_init(vmapped_fun, debug_info=dbg_info)),
                loop_state,
                batch,
                length=n_mubatches,
                schedule=schedule,
            )

    constrained_loop_output = jax.lax.with_sharding_constraint(
        loop_output,
        jax.tree.map_with_path(
            partial(sharding_with_data, spmd_axis_name), out_shardings
        ),
    )

    loop_output_reduced = constrained_loop_output.finalize()

    # def inspect(path, _a, a1, a2):
    #     def cb(prefix, a, s):
    #         print(prefix)
    #         print(jax.tree_util.keystr(path))
    #         print(a.shape)
    #         print(s.spec._normalized_spec(a.ndim))

    #     jax.debug.inspect_array_sharding(a1, callback=partial(cb, "input", _a))
    #     jax.debug.inspect_array_sharding(a1, callback=partial(cb, "before", a1))
    #     jax.debug.inspect_array_sharding(a2, callback=partial(cb, "after", a2))

    # jax.tree.map_with_path(inspect, loop_state, loop_output, loop_output_reduced)

    return loop_output_reduced
