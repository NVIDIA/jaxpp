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

from functools import partial
from pprint import pformat
from typing import Callable, ParamSpec, Sequence, TypedDict, TypeVar, cast

import jax
import jax.core as jcore
from jax._src import source_info_util
from jax._src.debugging import debug_effect, inspect_sharding_p
from jax._src.pjit import _infer_params, _parse_jit_arguments
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

from jaxpp.types import TaskType


def jit_infer_params(
    fun,
    args,
    in_axis_resources,
    out_axis_resources,
    static_argnums=(),
    donate_argnums=(),
):
    jit_info = _parse_jit_arguments(
        fun=fun,
        in_shardings=in_axis_resources,
        out_shardings=out_axis_resources,
        donate_argnums=donate_argnums,
        donate_argnames=None,
        static_argnums=static_argnums,
        static_argnames=None,
        device=None,
        backend=None,
        abstracted_axes=None,
        keep_unused=True,
        inline=False,
        compiler_options=None,
        use_resource_env=True,
    )
    return _infer_params(fun, jit_info, args, {})


add_multi_p = jcore.Primitive("add_multi")


@add_multi_p.def_abstract_eval
def add_multi_abstract_eval(*args, mpmd_idxs=None):
    first = args[0]
    assert all(first.dtype == arg.dtype for arg in args)
    return first


def add_multi_lower(*args, mpmd_idxs=None):
    return sum(args)


mlir.register_lowering(
    add_multi_p, mlir.lower_fun(add_multi_lower, multiple_results=False)
)

all_reduce_p = jcore.Primitive("all_reduce")


@all_reduce_p.def_abstract_eval
def all_reduce_abstract_eval(arg, shardings_by_mpmd_idx=None):
    return arg


send_p = jcore.Primitive("send")
send_p.multiple_results = True


def send_abstract_eval(*args, id, shardings):
    return args


send_p.def_abstract_eval(send_abstract_eval)

recv_p = jcore.Primitive("recv")
recv_p.multiple_results = True


def recv_abstract_eval(id, shardings, shape_and_dtype):
    return [jcore.ShapedArray(shape, dtype) for (shape, dtype) in shape_and_dtype]


recv_p.def_abstract_eval(recv_abstract_eval)

pipeline_yield_p = jcore.Primitive("pipeline_yield")
pipeline_yield_p.multiple_results = True
pipeline_yield_p.def_impl(lambda *args, **kwargs: args)
pipeline_yield_p.def_abstract_eval(lambda *args, **kwargs: args)


def pipeline_yield_batcher(args, dims, **kwargs):
    return pipeline_yield_p.bind(*args, **kwargs), dims


batching.primitive_batchers[pipeline_yield_p] = pipeline_yield_batcher
mlir.register_lowering(pipeline_yield_p, lambda ctx, *args, **kwargs: args)


def pipeline_yield_jvp(primals, tangents, **kwargs):
    return pipeline_yield_p.bind(*primals, **kwargs), pipeline_yield_p.bind(
        *tangents, **kwargs
    )


def pipeline_yield_transpose(ts, *prims, **kwargs):
    assert kwargs["task_type"] == TaskType.FWD
    return pipeline_yield_p.bind(
        *ts,
        **{
            **kwargs,
            "task_type": TaskType.BWD,
            "from_stage_id": kwargs["to_stage_id"],
            "to_stage_id": kwargs["from_stage_id"],
        },
    )


ad.primitive_jvps[pipeline_yield_p] = pipeline_yield_jvp
ad.primitive_transposes[pipeline_yield_p] = pipeline_yield_transpose
mlir.register_lowering(pipeline_yield_p, lambda ctx, *args, **kwargs: args)


def dax_pscan_abstract_eval(
    *args,
    jaxpr,
    n_mubatches,
    n_consts,
    in_shardings,
    out_shardings,
    in_mpmd_refs,
    out_mpmd_defs,
    schedule,
):
    return jaxpr.out_avals


dax_pscan_p = jcore.Primitive("dax_pscan")
dax_pscan_p.multiple_results = True
# TODO: maybe make it a absract_effectful_eval?
dax_pscan_p.def_abstract_eval(dax_pscan_abstract_eval)


def dax_pscan_lower(
    *args,
    jaxpr,
    n_mubatches,
    n_consts,
    in_shardings,
    out_shardings,
    in_mpmd_refs,
    out_mpmd_defs,
    schedule,
):
    fun = jcore.jaxpr_as_fun(jaxpr)

    if n_mubatches == 1:
        return fun(*args)

    loop_invariant_args = args[:n_consts]

    def loop_body(idx, loop_state):
        return fun(*(*loop_invariant_args, *loop_state))

    return jax.lax.fori_loop(0, n_mubatches, loop_body, list(args[n_consts:]))


mlir.register_lowering(
    dax_pscan_p, mlir.lower_fun(dax_pscan_lower, multiple_results=True)
)


def _task_transpose_update_params(params, undef_primals, nonzero_cts):
    return dict(params, task_name=f"bwd({params['task_name']})")


def task_lower(
    ctx,
    *args,
    name=None,
    backend=None,
    call_jaxpr,
    task_name,
    mpmd_idx,
    in_shardings,
    out_shardings,
    recv_invars,
    send_outvars,
    donate_invars,
):
    return mlir.core_call_lowering(
        ctx,
        *args,
        name=name,
        backend=backend,
        call_jaxpr=call_jaxpr,
    )


def dce_jaxpr_dax_pscan(
    used_outputs: list[bool], eqn: jcore.JaxprEqn
) -> tuple[list[bool], jcore.JaxprEqn]:
    jaxpr_ = eqn.params["jaxpr"]
    jaxpr, consts = jaxpr_.jaxpr, jaxpr_.consts

    has_changed = True
    while has_changed:
        has_changed = False
        new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs)
        for o_idx, (i, o) in enumerate(
            jax.util.safe_zip(used_inputs[eqn.params["n_consts"] :], used_outputs)
        ):
            if i and i != o:
                used_outputs[o_idx] = i
                has_changed = True

    # NOTE: it might happen that some output state is never merged with carried state
    #  (i.e. the `last` component of the LoopState).
    #  Here we make sure that the LoopState part of `used_inputs` agrees
    #  with `used_outputs`.
    for o_idx, (_, o) in enumerate(
        jax.util.safe_zip(used_inputs[eqn.params["n_consts"] :], used_outputs)
    ):
        used_inputs[eqn.params["n_consts"] + o_idx] = o

    new_jaxpr = new_jaxpr.replace(
        invars=[
            invar
            for invar, used in jax.util.safe_zip(jaxpr.invars, used_inputs)
            if used
        ],
        debug_info=None,  # FIXME
    )

    new_params = dict(
        eqn.params,
        n_consts=sum(used_inputs[: eqn.params["n_consts"]]),
        jaxpr=jcore.ClosedJaxpr(new_jaxpr, consts),
    )
    new_eqn = jcore.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs, strict=True) if used],
        [v for v, used in zip(eqn.outvars, used_outputs, strict=True) if used],
        eqn.primitive,
        new_params,
        new_jaxpr.effects,
        eqn.source_info,
    )
    return used_inputs, new_eqn


task_p = jcore.CallPrimitive("task")
task_p.def_impl(jcore.call_impl)

T = TypeVar("T")
P = ParamSpec("P")


def _task(fun, name: str, *args, **kwargs):
    jaxpr, out_shapes = jax.make_jaxpr(partial(fun, **kwargs), return_shape=True)(*args)
    flat_args = jax.tree_util.tree_leaves(args)
    out_tree = jax.tree_util.tree_structure(out_shapes)
    res = task_p.bind(*flat_args, task_type=TaskType.FWD, call_jaxpr=jaxpr)
    return jax.tree_util.tree_unflatten(out_tree, res)


def task(fun: Callable[P, T], *, name: str | None = None) -> Callable[P, T]:
    return partial(_task, fun, name)


lowering_rule: mlir.LoweringRule = partial(task_lower, name="task_call")

mlir.register_lowering(task_p, lowering_rule)
mlir.register_lowering(task_p, lowering_rule, platform="cpu")

ad.primitive_transposes[task_p] = partial(ad.call_transpose, task_p)
ad.call_transpose_param_updaters[task_p] = _task_transpose_update_params
pe.dce_rules[task_p] = pe.dce_jaxpr_call_rule
pe.dce_rules[dax_pscan_p] = dce_jaxpr_dax_pscan


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

    def __str__(self):
        if self._called_at_least_once:
            return f"{pformat(self.shardings)}"
        else:
            return repr(self)

    @property
    def shardings(self) -> list:
        if len(self._shardings) > 0 and not self._called_at_least_once:
            raise AssertionError(
                "Shardings can be inspected only after compiling the jaxpr"
            )
        assert all(s is not None for s in self._shardings)
        return self._shardings

    def callback_at_index(self, idx: int):
        def cb(s: jax.sharding.NamedSharding):
            self._called_at_least_once = True
            s.shard_shape(self.avals[idx].shape)
            self._shardings[idx] = s

        # This is helpful for debugging when `InspectSharding` fails
        cb.info = (self, idx)
        return cb

    @classmethod
    def collect(
        cls, values: Sequence[jax.Array], _provenance_info=None, _source_info=None
    ) -> "ShardingStore":
        store = cls(
            [v.aval for v in values],
            _provenance_info=_provenance_info,
            _source_info=_source_info,
        )
        for idx, v in enumerate(values):
            jax.debug.inspect_array_sharding(v, callback=store.callback_at_index(idx))
        return store

    @classmethod
    def collect_jaxpr(
        cls, vars_: Sequence[jcore.Var], _provenance_info=None, _source_info=None
    ) -> tuple["ShardingStore", list[jcore.JaxprEqn]]:
        store = cls(
            [v.aval for v in vars_],
            _provenance_info=_provenance_info,
            _source_info=_source_info,
        )

        res = []
        for idx, v in enumerate(vars_):
            res.append(
                jcore.new_jaxpr_eqn(
                    invars=[v],
                    outvars=[],
                    primitive=inspect_sharding_p,
                    params={"callback": store.callback_at_index(idx)},
                    effects=frozenset({debug_effect}),
                )
            )
        return store, res


# Refined type annotations for key Jaxprs/Eqns we use in the jaxpr
class TaskEqnParams(TypedDict):
    call_jaxpr: jcore.Jaxpr
    task_type: TaskType
    stage_id: int
    out_shardings: ShardingStore


class TaskEqn(jcore.JaxprEqn):
    invars: list[jcore.Var]  # Unique
    params: TaskEqnParams

    def replace(
        self,
        invars: list[jcore.Var] | None = None,
        outvars: list[jcore.Var] | None = None,
        primitive: jcore.Primitive | None = None,
        params: TaskEqnParams | None = None,
        effects: jcore.Effects | None = None,
        source_info: source_info_util.SourceInfo | None = None,
    ):
        pass

    @staticmethod
    def make(eqn: jcore.JaxprEqn) -> "TaskEqn":
        assert eqn.primitive is task_p
        for invar in eqn.invars:
            assert isinstance(invar, jcore.Var), "Pipeline stage has literal arguments"
        for outvar in eqn.params["call_jaxpr"].outvars:
            assert isinstance(outvar, jcore.Var), "Pipeline stage has literal results"
        assert len(eqn.invars) == len(set(eqn.invars)), "Duplicate arguments to stage"
        return cast(TaskEqn, eqn)


class PscanJaxpr(jcore.Jaxpr):
    @property
    def eqns(self) -> list[TaskEqn]: ...

    @property
    def outvars(self) -> list[jcore.Var]: ...

    @staticmethod
    def make(jaxpr: jcore.Jaxpr) -> "PscanJaxpr":
        for eqn in jaxpr.eqns:
            TaskEqn.make(eqn)
        # NOTE: also checks that it doesn't have literal outvars
        assert len(set(jaxpr.invars) & set(jaxpr.outvars)) == 0
        return cast(PscanJaxpr, jaxpr)
