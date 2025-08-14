from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from jaxpp.api import BaseSchedule, pipeline_enter_stage, treduce
from jaxpp.core import (
    cluster_jaxpr,
    infer_shardings2,
    jaxpp_debug_skip_propagation,
    mpmdify,
    strip_inspect_sharding_eqns,
    wrap_into_tasks,
)
from jaxpp.jax_primitives import dax_pscan_p
from jaxpp.schedules import Eager1F1B, Interleaved1F1B, Std1F1B
from jaxpp.pipelining import yield_scope


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
    layer_0_activations = None
    for layer, (w, b) in enumerate(params):

        def stage(activations):
            outputs = jnp.dot(activations, w) + b
            activations = relu(outputs)
            return activations

        if layer == 0:
            layer_0_activations = activations

        if layer != len(params) - 1:
            activations = named_computation(stage, f"stage_{layer}")(activations)
        else:
            activations = stage(activations)

    logits = activations
    return logits - logsumexp(logits), layer_0_activations


@jax.value_and_grad
def grads(params, data):
    X, Y = data
    preds, layer_0_activations = predict(params, X)
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
    with jaxpp_debug_skip_propagation.set(skip_propagation):
        infer_shardings2(
            wrapped_cjaxpr, [replicated_sharding] * len(cjaxpr.in_avals), stage_mesh
        )
        wrapped_cjaxpr = strip_inspect_sharding_eqns(wrapped_cjaxpr)

    return mpmdify(wrapped_cjaxpr, mpmd_dim)


@pytest.mark.parametrize(
    "mpmd_dim,num_stages,n_mubatches,schedule",
    [
        (1, 1, 4, Std1F1B(num_stages=1)),
        (2, 2, 4, Std1F1B(num_stages=2)),
        (3, 3, 5, Eager1F1B(num_stages=3)),
        (1, 1, 1, Interleaved1F1B(num_stages=1, mpmd_dim=1)),
        (2, 4, 12, Interleaved1F1B(num_stages=4, mpmd_dim=2)),
    ],
)
def test_transformations_succeed(
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


def test_skip_propagation_false():
    test_transformations_succeed(
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
        mpmd_dim=num_stages,
        is_loop=False,
    )


@pytest.mark.parametrize("num_stages", [4])
def test_inference_opening_not_triggered(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        preds, layer_0_activations = predict(params, X)
        return preds + params[1][1]

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        mpmd_dim=num_stages,
        is_loop=False,
    )


# FIXME(#41)
def test_bug_transformation_fails_without_after_loop():
    with pytest.raises(AssertionError) as exc:
        test_transformations_succeed(
            *(3, 3, 5, Eager1F1B(num_stages=3)), skip_apply_grads=True
        )
    assert "Unsupported empty after_loop" in str(exc.value)


# NOTE(#42)
@pytest.mark.parametrize("num_stages", [4])
def test_inference(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        preds, layer_0_activations = predict(params, X)
        return layer_0_activations + params[1][1]

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        mpmd_dim=num_stages,
        is_loop=False,
    )


@pytest.mark.parametrize("num_stages", [4])
def test_inference_complex(num_stages: int):
    n_mubatches = 4
    (params, X, Y), stage_mesh, replicated_sharding = get_context(
        num_stages=num_stages, n_mubatches=n_mubatches
    )

    def _fun(params, X):
        preds, layer_1_activations = predict(params, X)
        neg = preds < 0
        zeros = jnp.zeros_like(layer_1_activations)
        twice = layer_1_activations * 2
        _ = jax.lax.cond(neg.any(), lambda: layer_1_activations, lambda: twice + zeros)
        return _ + params[0][1]

    jitted = jax.jit(_fun)
    with yield_scope():
        cjaxpr = jitted.trace(params, X).jaxpr

    _ = cluster_jaxpr(
        cjaxpr.jaxpr,
        num_stages,
        is_partial_bwd=False,
        mpmd_dim=num_stages,
        is_loop=False,
    )
    assert len(_.eqns) == num_stages + 1, (len(_.eqns), num_stages + 1)

