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
import jax.numpy as jnp
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


def _trace(fun, args, debug_info):
    """Trace ``fun(*args)`` as JAX's control flow traces loop bodies.

    Returns ``(jaxpr, consts, out_tree)``, where the jaxpr takes the closed-over
    ``consts`` as its leading inputs.
    """
    flat_args, in_tree = jax.tree.flatten(args)
    wrapped = lu.wrap_init(fun, debug_info=debug_info)
    flat_fun, out_tree = jau.flatten_fun_nokwargs(wrapped, in_tree)
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        flat_fun, tuple(map(jc.get_aval, flat_args))
    )
    return jc.convert_constvars_jaxpr(jaxpr), consts, out_tree()


def _pscan(step, state, length, schedule: BaseSchedule, debug_info):
    """Bind ``dax_pscan`` to run ``state = step(index, state)`` per microbatch."""
    jaxpr, consts, out_tree = _trace(step, (0, state), debug_info)
    n_consts = len(consts)
    const_invars, [index_invar], carry_invars = jc.split_list(
        jaxpr.invars, [n_consts, 1]
    )

    # As in `jax.lax.scan`, carries returned unchanged are loop invariants. Pass
    # them as constants: placement requires loop outputs defined in the body.
    invariant = [o is v for o, v in zip(jaxpr.outvars, carry_invars)]
    carry_args, invariant_args = jc.partition_list(invariant, jax.tree.leaves(state))
    if invariant_args:
        carry_invars, invariant_invars = jc.partition_list(invariant, carry_invars)
        invars = [*const_invars, *invariant_invars, index_invar, *carry_invars]
        outvars, _ = jc.partition_list(invariant, jaxpr.outvars)
        effects = jc.make_jaxpr_effects([], invars, outvars, jaxpr.eqns)
        jaxpr = jaxpr.replace(
            invars=invars, outvars=outvars, effects=effects, debug_info=None
        )
        consts = [*consts, *invariant_args]
        n_consts = len(consts)

    # Accumulators that several stages add to (e.g. tied embeddings) get one
    # replica per stage, summed after the loop.
    n_outs = len(jaxpr.outvars)
    jaxpr = pushout_add_any(jaxpr)
    # FIXME: ensure that it doesn't produce duplicate outvars
    replicated_invars, replicated_outvars, replace_eqns = compute_needed(
        jaxpr, n_consts + 1
    )
    jaxpr = add_jaxpr_parameters(
        jaxpr, replicated_invars, replicated_outvars, replace_eqns
    )
    # The microbatch index is implicit: dax_pscan supplies it to the body.
    operands = list(consts)
    for invar_idx, arg in enumerate(carry_args, start=n_consts + 1):
        replicas = replicated_invars.get(invar_idx, [arg])
        # TODO: maybe create a normalization pass like outvar_normalization in
        # licm.py. At different stages of transformations, we have a few
        # assumptions that are checked, but we don't have normalizations for them.
        operands += [arg, *(jnp.array(arg, copy=True) for _ in replicas[1:])]

    outs = iter(
        dax_pscan_p.bind(
            *operands,
            jaxpr=jc.close_jaxpr(jaxpr),
            n_mubatches=length,
            n_consts=n_consts,
            in_shardings=(None,) * len(operands),
            out_shardings=(None,) * len(jaxpr.outvars),
            in_mpmd_refs=None,
            out_mpmd_defs=None,
            schedule=schedule,
        )
    )
    carry = [
        ad.add_jaxvals(next(outs), next(outs))
        if i in replicated_outvars
        else next(outs)
        for i in range(n_outs)
    ]
    return out_tree.unflatten(jc.merge_lists(invariant, carry, invariant_args))


class Op(Protocol):
    def state(self, n: int, a: jax.ShapeDtypeStruct) -> jax.Array: ...

    def update(self, state: jax.Array, update: jax.Array, index: int) -> jax.Array: ...


@dataclass(frozen=True)
class AddT:
    """Represents an element-wise addition operation."""

    def state(self, _: int, a: jax.ShapeDtypeStruct) -> jax.Array:
        return jnp.zeros_like(a)

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
        sharding = getattr(a, "sharding", None)
        if sharding is not None:
            assert isinstance(a.sharding, jax.sharding.NamedSharding)
            spec = a.sharding.spec
            new_spec = jax.sharding.PartitionSpec(
                *(tuple(spec[: self.axis]) + (None,) + tuple(spec[self.axis :]))
            )
            sharding = a.sharding.update(spec=new_spec)
            return jnp.zeros_like(a, shape=shape, out_sharding=sharding)
        return jnp.zeros_like(a, shape=shape)

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


def _microbatch(xs, index, axis=0):
    return jax.tree.map(lambda x: jnp.take(x, index, axis=axis), xs)


def _pipelined_loop(fun, init, length, schedule: BaseSchedule, operation, debug_info):
    """Pipeline ``carry, y = fun(index, carry)``, reducing ``y`` with ``operation``.

    As in ``jax.lax.scan``'s discharge rule, Refs closed over by ``fun`` are
    carried through the loop: read before it and written back after it.
    Discharging also removes Refs created in ``fun`` and lowers HiJAX, so
    pipeline placement sees a pure body.
    """
    with log_elapsed_time("jaxpr/first_loop_tracing"), yield_scope():
        jaxpr, consts, out_tree = _trace(fun, (0, init), debug_info)
        in_avals = [v.aval for v in jaxpr.invars[len(consts) :]]
        out_avals = [v.aval for v in jaxpr.outvars]
        if any(isinstance(a, jc.AbstractRef) for a in (*in_avals, *out_avals)):
            raise ValueError(
                "Loop bodies may close over Refs, not carry or return them"
            )
        is_ref = [isinstance(jc.get_aval(c), jc.AbstractRef) for c in consts]
        body = jc.ClosedJaxpr(jaxpr, ())
        stateful = any(
            isinstance(e, (jc.RefEffect, jc.InternalMutableArrayEffect))
            for e in jaxpr.effects
        )
        if any(is_ref) or stateful or getattr(jaxpr, "is_high", False):
            body = jc.discharge_state(
                body, should_discharge=[*is_ref, *(False,) * len(in_avals)]
            )
    pure_consts, refs = jc.partition_list(is_ref, consts)

    carry_shapes, y_shapes = out_tree.unflatten(
        map(jc.aval_to_shape_dtype_struct, out_avals)
    )
    if jax.tree.structure(carry_shapes) != jax.tree.structure(init) or any(
        (a.shape, a.dtype) != (b.shape, b.dtype)
        for a, b in zip(in_avals[1:], jax.tree.leaves(carry_shapes))
    ):
        raise ValueError(
            "The loop body must preserve the carry pytree, shapes and dtypes"
        )
    # TODO: maybe use custom definition of `tree_broadcast` and
    #  improve error message if it cannot be broadcasted
    operation = jc.tree_broadcast(jax.tree.structure(y_shapes), operation)

    def step(index, state):
        carry, ref_values, y_state = state
        consts = jc.merge_lists(is_ref, pure_consts, ref_values)
        outs = jcore.eval_jaxpr(
            body.jaxpr, body.consts, *consts, index, *jax.tree.leaves(carry)
        )
        outs, ref_values = jc.split_list(outs, [len(out_avals)])
        carry, y = out_tree.unflatten(outs)
        y_state = jax.tree.map(
            lambda op, s, v: copy_if_scalar(op.update(s, v, index)),
            operation,
            y_state,
            y,
        )
        return carry, ref_values, y_state

    y_state = jax.tree.map(lambda op, a: op.state(length, a), operation, y_shapes)
    # Keep scalar initial values from becoming literals.
    state = jax.tree.map(copy_if_scalar, (init, [ref[...] for ref in refs], y_state))
    step_info = jau.debug_info(debug_info.traced_for, step, (0, state), {})
    with log_elapsed_time("jaxpr/second_loop_tracing"), yield_scope():
        carry, ref_values, y_state = _pscan(step, state, length, schedule, step_info)
    for ref, value in zip(refs, ref_values, strict=True):
        ref[...] = value
    return carry, y_state


def tscan(fun, init, xs, schedule: BaseSchedule):
    """Pipeline ``fun(carry, microbatch) -> (carry, output)`` over ``xs``.

    Like ``jax.lax.scan``, outputs are stacked on a new leading microbatch axis,
    and ``fun`` may close over Refs, such as gradient accumulators passed to
    ``jax.vjp(...).with_refs``: their updates are threaded through the loop and
    written back after it. Refs created in the body are discharged, while kernel
    input/output aliases are retained. Refs must not be carried or escape the
    body.

    The carry and closed-over Refs become dependencies between consecutive
    microbatches, which the schedule must honor. Gradient accumulators read and
    written only by one stage's backward pass do; dependencies a schedule cannot
    meet fail placement with "Schedule does not honor data dependencies".
    """
    leaves = jax.tree.leaves(xs)
    if not leaves or any(not x.ndim for x in leaves):
        raise ValueError("tscan requires nonempty arrays with a microbatch dimension")
    length = leaves[0].shape[0]
    if length < 1 or any(x.shape[0] != length for x in leaves):
        raise ValueError("tscan requires matching, positive microbatch dimensions")

    def body(index, carry):
        return fun(carry, _microbatch(xs, index))

    debug_info = jau.debug_info("tscan", body, (0, init), {})
    return _pipelined_loop(body, init, length, schedule, Concat(), debug_info)


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
    overlap on the accelerator. Like :func:`~.tscan`, ``fun`` may close over
    Refs, such as gradient accumulators passed to ``jax.vjp(...).with_refs``.

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
    leaves = jax.tree.leaves(xs)
    length = leaves[0].shape[axis]
    if any(x.shape[axis] != length for x in leaves):
        raise ValueError("treduce requires matching microbatch dimensions")

    @functools.wraps(fun)
    def body(i):
        return fun(_microbatch(xs, i, axis))

    return treduce_i(body, length, schedule=schedule, operation=operation)


def copy_if_scalar(x: jax.Array) -> jax.Array:
    if jnp.ndim(x) == 0:
        return jnp.array(x, copy=True)
    return x


def treduce_i(
    fun: Callable[[int], Y], length: int, schedule: BaseSchedule, operation=default_op
) -> Y:
    r"""Lower-level helper for :func:`~.treduce` that takes an explicit ``length``.

    Instead of slicing from a pre-materialised batch this variant invokes
    ``fun(i)`` directly for each ``0 <= i < length`` and reduces the returned
    values using ``operation``. Like :func:`~.tscan`, ``fun`` may close over
    Refs::

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

    def body(index, carry):
        return carry, fun(index)

    debug_info = jau.debug_info(treduce_i.__name__, fun, (jc.get_aval(0),), {})
    _, loop_output = _pipelined_loop(body, (), length, schedule, operation, debug_info)
    return loop_output
