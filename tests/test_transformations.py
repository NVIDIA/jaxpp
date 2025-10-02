import dataclasses
from contextlib import contextmanager
from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jcore
from jax.scipy.special import logsumexp

from jaxpp import env_vars
from jaxpp.api import BaseSchedule, pipeline_enter_stage, treduce
from jaxpp.core import (
    cluster_jaxpr,
    infer_shardings2,
    maybe_unroll_loop,
    strip_inspect_sharding_eqns,
    wrap_into_tasks,
)
from jaxpp.jax_primitives import dax_pscan_p
from jaxpp.mesh import MpmdMesh
from jaxpp.pipelining import yield_scope
from jaxpp.schedules import DualPipeV, Eager1F1B, Interleaved1F1B, Std1F1B, ZeroBubble


def named_computation(fun, name):
    def _fn(*args, **kwargs):
        return pipeline_enter_stage(fun(*args, **kwargs), name)

    return _fn


def relu(x):
    return jnp.maximum(0, x)


def predict(params, X):
    # TODO: add a constant here

    # per-example predictions
    activations = X
    layer_1_inputs = None
    for layer, (w, b) in enumerate(params):

        def stage(activations):
            outputs = jnp.dot(activations, w) + b
            activations = relu(outputs)
            return activations

        activations = named_computation(stage, f"stage_{layer}")(activations)
        if layer == 0:
            layer_1_inputs = activations

    logits = activations
    return logits - logsumexp(logits), layer_1_inputs


@jax.value_and_grad
def grads(params, data):
    X, Y = data
    preds, layer_1_inputs = predict(params, X)
    return jnp.mean((preds - Y) ** 2)


step_size = 0.01


def accumulate_grads(params, X, Y, schedule, skip_apply_grads: bool):
    losses, total_grads = treduce(partial(grads, params), (X, Y), schedule=schedule)
    if skip_apply_grads:
        return ((losses, total_grads), 5, X)
    new_params = [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, total_grads)
    ]
    return ((losses, new_params), 5, X)


def get_context(num_stages, n_mubatches):
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (n_mubatches, 4, 10))
    Y = jax.random.uniform(key, (n_mubatches, 4, 10))
    keys = jax.random.split(key, num_stages)
    params = [
        (jax.random.uniform(k, (10, 10)), jax.random.uniform(k, (10,))) for k in keys
    ]

    stage_mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("devs",))
    replicated_sharding = jax.NamedSharding(stage_mesh, jax.sharding.PartitionSpec())
    return (params, X, Y), stage_mesh, replicated_sharding


def setup(num_stages, n_mubatches, schedule, skip_apply_grads: bool = False):
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages, n_mubatches
    )

    total_grads_fn = jax.jit(
        partial(accumulate_grads, schedule=schedule, skip_apply_grads=skip_apply_grads)
    )
    return total_grads_fn, (params, X, Y), stage_mesh, replicated_sharding


def get_scheduled_jaxpr(
    cjaxpr, mpmd_dim, stage_mesh, replicated_sharding, skip_propagation: bool
):
    wrapped_cjaxpr, in_mpmd_refs, out_mpmd_defs = wrap_into_tasks(
        cjaxpr, used_invars=(True,) * len(cjaxpr.in_avals), mpmd_dim=mpmd_dim
    )
    with env_vars.jaxpp_debug_skip_propagation.set(skip_propagation):
        infer_shardings2(
            wrapped_cjaxpr, [replicated_sharding] * len(cjaxpr.in_avals), stage_mesh
        )
        wrapped_cjaxpr = strip_inspect_sharding_eqns(wrapped_cjaxpr)

    return maybe_unroll_loop(wrapped_cjaxpr)


@dataclasses.dataclass(frozen=True)
class MpmdMeshLike:
    jax_mesh: jax.sharding.Mesh
    mpmd_dim: int

    @property
    def unstack(self) -> Mapping[int, jax.sharding.Mesh]:
        return {mpmd_idx: self.jax_mesh for mpmd_idx in range(self.mpmd_dim)}


@contextmanager
def cleanup(fn):
    try:
        yield
    finally:
        fn()


@pytest.mark.parametrize(
    "mpmd_dim,num_stages,n_mubatches,schedule",
    [
        (1, 1, 4, Std1F1B(num_stages=1)),
        (2, 2, 4, Std1F1B(num_stages=2)),
        (3, 3, 5, Eager1F1B(num_stages=3)),
        (1, 1, 1, Interleaved1F1B(num_stages=1, mpmd_dim=1)),
        (4, 8, 1, Interleaved1F1B(num_stages=8, mpmd_dim=4)),
        (2, 4, 12, Interleaved1F1B(num_stages=4, mpmd_dim=2)),
        (4, 4, 4, ZeroBubble(num_stages=4)),
        (4, 4, 8, ZeroBubble(num_stages=4)),
        (3, 6, 6, DualPipeV(num_stages=6, mpmd_dim=3)),
        (3, 6, 10, DualPipeV(num_stages=6, mpmd_dim=3)),
    ],
)
def test_equivalence_scheduled(
    mpmd_dim: int,
    num_stages: int,
    n_mubatches: int,
    schedule: BaseSchedule,
    skip_propagation: bool = True,
    skip_apply_grads: bool = False,
):
    total_grads_fn, (params, X, Y), stage_mesh, replicated_sharding = setup(
        num_stages, n_mubatches, schedule, skip_apply_grads=skip_apply_grads
    )

    cjaxpr = total_grads_fn.trace(params, X, Y).jaxpr
    loop_eqn_idx = None
    for eqn_idx, eqn in enumerate(cjaxpr.eqns):
        if eqn.primitive is dax_pscan_p:
            loop_eqn_idx = eqn_idx
            break

    assert loop_eqn_idx is not None

    scheduled_jaxpr = get_scheduled_jaxpr(
        cjaxpr,
        mpmd_dim,
        stage_mesh,
        replicated_sharding,
        skip_propagation=skip_propagation,
    )

    MpmdMesh.mesh_stack.append(MpmdMeshLike(stage_mesh, mpmd_dim))
    with cleanup(lambda: MpmdMesh.mesh_stack.pop()):
        flat_args = jax.tree_util.tree_leaves((params, X, Y))
        transformed_res = jcore.jaxpr_as_fun(scheduled_jaxpr)(*flat_args)
        plain_res = jax.tree_util.tree_leaves(total_grads_fn(*(params, X, Y)))
        for l, r in zip(plain_res, transformed_res):
            assert jnp.all(l == r)


@pytest.mark.parametrize(
    "mpmd_dim,num_stages,n_mubatches,schedule",
    [
        (8, 8 * 6, 64, Interleaved1F1B(num_stages=8 * 6, mpmd_dim=8)),
        (8, 16, 18, DualPipeV(num_stages=16, mpmd_dim=8)),
    ],
)
def test_transformations_dont_fail(
    mpmd_dim: int, num_stages: int, n_mubatches: int, schedule: BaseSchedule
):
    total_grads_fn, (params, X, Y), stage_mesh, replicated_sharding = setup(
        num_stages, n_mubatches, schedule
    )

    cjaxpr = total_grads_fn.trace(params, X, Y).jaxpr
    scheduled_jaxpr = get_scheduled_jaxpr(
        cjaxpr,
        mpmd_dim,
        stage_mesh,
        replicated_sharding,
        skip_propagation=True,
    )


def test_skip_propagation_false():
    test_equivalence_scheduled(
        *(3, 3, 5, Eager1F1B(num_stages=3)), skip_propagation=False
    )


@pytest.mark.parametrize("num_stages", [4])
def test_inference(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        return predict(params, X)

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        get_mpmd_idx=lambda _: _,
        is_loop=False,
    )


@pytest.mark.parametrize("num_stages", [4])
def test_inference_opening_not_triggered(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        preds, layer_1_inputs = predict(params, X)
        return preds + params[1][1]

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        get_mpmd_idx=lambda _: _,
        is_loop=False,
    )


# NOTE(#41)
def test_transformation_succeeds_without_after_loop():
    test_equivalence_scheduled(
        *(3, 3, 5, Eager1F1B(num_stages=3)), skip_apply_grads=True
    )


# NOTE(#42)
@pytest.mark.parametrize("num_stages", [4])
def test_inference(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        preds, layer_1_inputs = predict(params, X)
        return layer_1_inputs + params[1][1]

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        get_mpmd_idx=lambda _: _,
        is_loop=False,
    )


@pytest.mark.parametrize("num_stages", [4])
def test_inference_complex(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        preds, layer_1_inputs = predict(params, X)
        neg = preds < 0
        zeros = jnp.zeros_like(layer_1_inputs)
        twice = layer_1_inputs * 2
        _ = jax.lax.cond(neg.any(), lambda: layer_1_inputs, lambda: twice + zeros)
        return _ + params[0][1]

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        get_mpmd_idx=lambda _: _,
        is_loop=False,
    )
