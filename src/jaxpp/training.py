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

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import jax
import jax.api_util as jau
import jax.extend.linear_util as lu
import numpy as np
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe

from jaxpp import jax_compat as jc
from jaxpp.core import add_jaxpr_parameters, compute_needed, pushout_add_any
from jaxpp.jax_compat import core as jcore
from jaxpp.jax_primitives import dax_pscan_p
from jaxpp.pipelining import yield_scope
from jaxpp.schedules import BaseSchedule
from jaxpp.utils import log_elapsed_time

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def pscan_wrapped(fun: lu.WrappedFun, init, length, schedule):
    # NOTE: + 0 needed so that jax doesn't make it a `Literal` argument
    mubatch_idx = jax.numpy.zeros_like(0) + 0

    flat_args, in_tree = jax.tree_util.tree_flatten((mubatch_idx, init))
    flat_scan_body, out_tree = jau.flatten_fun_nokwargs(fun, in_tree)

    scan_body_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        flat_scan_body, tuple(jcore.get_aval(e) for e in flat_args)
    )

    n_consts = len(consts)
    flat_args = consts + flat_args
    scan_body_jaxpr = jc.convert_constvars_jaxpr(scan_body_jaxpr)

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
                # TODO: maybe create a normalization pass like outvar_normalization in licm.py
                # At different stages of transformations, we have a few assumptions that are checked,
                # but we don't have normalizations for them.
                new_flat_args.append(jax.numpy.array(arg, copy=True))
        else:
            new_flat_args.append(arg)
    flat_args = new_flat_args

    out = dax_pscan_p.bind(
        *flat_args,
        jaxpr=jc.close_jaxpr(scan_body_jaxpr),
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


class Op(Protocol):
    def state(self, n: int, a: jax.ShapeDtypeStruct) -> jax.Array: ...

    def update(self, state: jax.Array, update: jax.Array, index: int) -> jax.Array: ...


@dataclass(frozen=True)
class AddT:
    """Represents an element-wise addition operation."""

    def state(self, _: int, a: jax.ShapeDtypeStruct) -> jax.Array:
        return jax.numpy.zeros(a.shape, dtype=a.dtype)

    def update(self, l: jax.Array, r: jax.Array, _: int) -> jax.Array:
        return jax.lax.add(l, r)


Add = AddT()


@dataclass(frozen=True)
class Concat:
    """Represents a concatenation operation along a specified axis."""

    axis: int = 0

    def state(self, n: int, a: jax.ShapeDtypeStruct) -> jax.Array:
        assert n > 0
        shape = a.shape[: self.axis] + (n,) + a.shape[self.axis :]
        return jax.numpy.zeros(shape, dtype=a.dtype)

    def update(self, state: jax.Array, update: jax.Array, index: int) -> jax.Array:
        return jax.lax.dynamic_update_index_in_dim(state, update, index, axis=self.axis)


@dataclass(frozen=True)
class MaxT:
    """Represents an element-wise maximum operation."""

    def state(self, _: int, a: jax.ShapeDtypeStruct) -> jax.Array:
        return jax.lax.full(
            a.shape,
            (-np.inf if jc.supports_inf(a.dtype) else jc.finfo(a.dtype).min),
            dtype=a.dtype,
        )

    def update(self, l: jax.Array, r: jax.Array, _: int) -> jax.Array:
        return jax.lax.max(l, r)


Max = MaxT()


default_op = (Concat(), Add)


def treduce(
    fun: Callable[[X], Y],
    xs: X,
    schedule: BaseSchedule,
    axis: int = 0,
    operation=default_op,
) -> Y:
    r"""Temporally reduces a sequence of inputs with a pipelined schedule.

    This function behaves like the functional-programming primitive
    ``reduce`` applied along the leading (time / micro-batch) axis of
    ``xs``.  At each timestep ``i`` it applies ``fun`` to the slice
    ``xs[i]`` and combines the resulting values using ``operation``, as shown
    in the following example::

        def treduce(fun, xs, operation=(Concat(), Add())):
          # xs has shape (T, ...)
          state = tree_map(lambda a, op: op.state(len(xs), a),
                           fun(jnp.take(xs[0], 0, axis=axis)), operation)
          for i in range(1, len(xs)):
            state = tree_map(lambda op, s, v: op.update(s, v, i),
                             operation, state, fun(jnp.take(xs[i], i, axis=axis)))
          return state

    Unlike a vanilla reduce, the execution of the loop body may be
    interleaved according to ``schedule`` so that different timesteps can
    overlap on the accelerator.

    Args:
      fun: A function that is applied to a single slice. It
        receives one element ``xs[i]`` (a PyTree slice with the leading axis
        removed) and returns a PyTree.
      xs: A PyTree whose leaf nodes are arrays. All leaf arrays must share
        the same leading dimension size; this leading dimension is the axis
        that is reduced over.
      schedule: A :class:`~jaxpp.schedules.BaseSchedule` specifying how loop
        iterations should overlap.
      axis: Integer specifying which axis of every leaf array in ``xs``
        represents the micro-batch dimension.
        At iteration ``i`` the function gathers the slice at index ``i`` along
        this axis and feeds it to ``fun``.
      operation: A PyTree of :class:`~.Op` objects (default: ``default_op``,
        i.e., ``(Concat(), Add)``) describing how the per-timestep values are
        aggregated. Each leaf :class:`~.Op` object defines ``state``
        and ``update`` methods. Convenient predefined ops include:

        - :data:`~.Add`: Element-wise sum.
        - :data:`~.Max`: Element-wise maximum.
        - :obj:`~.Concat`\ ``(axis=0)``: Stacks results from each timestep.
          The ``axis`` parameter specifies the dimension in the output array
          that corresponds to the reduction timesteps.

    Returns:
      A PyTree containing the aggregated result, with the same structure as ``Y``.
    """
    flat_batch = jax.tree_util.tree_leaves(xs)
    first_batch_shape = flat_batch[0].shape
    if any(a.shape[axis] != first_batch_shape[axis] for a in flat_batch):
        raise AssertionError("Leading dimensions differing among xs")

    @functools.wraps(fun)
    def wrap(i):
        e = jax.tree.map(lambda x: jax.numpy.take(x, i, axis=axis), xs)
        return fun(e)

    return treduce_i(
        wrap, first_batch_shape[axis], schedule=schedule, operation=operation
    )


def copy_if_scalar(x: jax.Array) -> jax.Array:
    if x.ndim == 0:
        return jax.numpy.array(x, copy=True)
    return x


def treduce_i(
    fun: Callable[[int], Y], length: int, schedule: BaseSchedule, operation=default_op
) -> Y:
    r"""Lower-level helper for :func:`~.treduce` that takes an explicit ``length``.

    Instead of slicing from a pre-materialised batch this variant invokes
    ``fun(i)`` directly for each ``0 <= i < length`` and reduces the returned
    values using ``operation``::

        def treduce_i(fun, length, operation):
          state = tree_map(lambda a, op: op.state(length, a),
                           fun(0), operation)
          for i in range(1, length):
            state = tree_map(lambda op, s, v: op.update(s, v, i),
                             operation, state, fun(i))
          return state

    Args:
      fun: Function that receives the micro-batch/timestep index ``i`` (an
        integer) and returns a PyTree to be reduced.
      length: The number of timesteps / micro-batches.
      schedule: A :class:`~jaxpp.schedules.BaseSchedule` determining how
        iterations may overlap.
      operation: A PyTree of :class:`~.Op` objects (default: ``default_op``,
        i.e., ``(Concat(), Add)``) controlling the accumulation. See
        :func:`~.treduce` for details. Convenient predefined ops include:

        - :data:`~.Add`: Element-wise sum.
        - :data:`~.Max`: Element-wise maximum.
        - :obj:`~.Concat`\ ``(axis=0)``: Stacks results from each timestep.
          The ``axis`` parameter specifies the dimension in the output array
          that corresponds to the reduction timesteps.

    Returns:
      A PyTree containing the result of the temporal reduction, with the same
      structure as ``Y``.
    """
    with log_elapsed_time("jaxpr/first_loop_tracing"), yield_scope():
        body_args = jcore.ShapedArray((), dtype=jax.numpy.int32)
        body_jaxpr, loop_out_shapes = jax.make_jaxpr(fun, return_shape=True)(body_args)

    # TODO: maybe use custom definition of `tree_broadcast` and
    #  improve error message if it cannot be broadcasted
    operation = jc.tree_broadcast(
        jax.tree_util.tree_structure(loop_out_shapes), operation
    )

    def state(op: Op, a):
        return copy_if_scalar(op.state(length, a))

    loop_state = jax.tree_util.tree_map(state, operation, loop_out_shapes)

    def _fun(mubatch_idx, loop_state):
        def update(op: Op, state, update):
            return copy_if_scalar(op.update(state, update, mubatch_idx))

        return (
            mubatch_idx + 1,
            jax.tree_util.tree_map(
                update,
                operation,
                loop_state,
                jax.tree.unflatten(
                    jax.tree.structure(loop_out_shapes),
                    jcore.eval_jaxpr(
                        body_jaxpr.jaxpr,
                        body_jaxpr.consts,
                        mubatch_idx,
                        propagate_source_info=False,
                    ),
                ),
            ),
        )

    debug_info = jau.debug_info(treduce_i.__name__, fun, (body_args,), {})
    wrapped_body_fun = lu.wrap_init(_fun, debug_info=debug_info)
    with log_elapsed_time("jaxpr/second_loop_tracing"), yield_scope():
        loop_output = pscan_wrapped(
            wrapped_body_fun, loop_state, length=length, schedule=schedule
        )

    return loop_output
