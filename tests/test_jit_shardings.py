import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import jaxpp.api
from jaxpp.mesh import MpmdMesh


def step(w, xs, unused):
    del unused

    def grads(x):
        return jax.grad(lambda w: jnp.sum(jnp.tanh(jaxpp.api.mark_stage_end(x @ w))))(w)

    summed = jaxpp.api.treduce(
        grads, xs, schedule=jaxpp.api.Std1F1B(num_stages=1), operation=jaxpp.api.Add
    )
    return w - 0.1 * summed


def test_unspecified_shardings_follow_jax_jit():
    # One device is enough: specs are kept on size-1 axes.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("stage", "x"))
    w = jax.device_put(jnp.ones((4, 4)), NamedSharding(mesh, P(None, "x")))
    xs = jnp.ones((2, 3, 4))
    unused = jnp.zeros(5)

    fn = jaxpp.api.mpmd_jit_with_loop(step, mpmd_mesh=MpmdMesh(mesh, "stage"))
    in_info = fn.compile(w, xs, unused).in_info
    w_sharding, _, unused_sharding = in_info.in_shardings
    (out_sharding,) = in_info.out_shardings

    # A committed argument keeps its sharding; an unconstrained one is
    # replicated, as for an unused `jax.jit` argument.
    assert w_sharding.spec == w.sharding.spec
    assert unused_sharding.spec == P()
    # An unspecified output takes the sharding inference chose for it.
    assert isinstance(out_sharding, NamedSharding)

    out = fn(w, xs, unused)
    np.testing.assert_allclose(
        np.asarray(out.to_mpmd_local_array), np.asarray(jax.jit(step)(w, xs, unused))
    )


def test_committed_sharding_must_be_named():
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("stage", "x"))
    w = jax.device_put(jnp.ones((4, 4)), jax.devices()[0])

    fn = jaxpp.api.mpmd_jit_with_loop(step, mpmd_mesh=MpmdMesh(mesh, "stage"))
    with pytest.raises(ValueError, match="NamedSharding"):
        fn.compile(w, jnp.ones((2, 3, 4)), jnp.zeros(5))
