# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpp.core import preprocess_jaxpr
from jaxpp.jax_compat import core
from jaxpp.schedules import GPipe
from jaxpp.training import Add, Concat, treduce, treduce_i


@pytest.mark.parametrize("indexed", [False, True])
def test_treduce_checks_with_nested_loop_state(indexed):
    xs = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

    def body(x):
        return x * 2, {"sum": x.sum()}

    def reduce(values):
        kwargs = dict(schedule=GPipe(1), operation=(Concat(), {"sum": Add}))
        if indexed:
            return treduce_i(lambda i: body(values[i]), len(values), **kwargs)
        return treduce(body, values, **kwargs)

    # The second trace receives both the index and every loop-state leaf.
    with jax.enable_checks(True):
        actual = jax.jit(reduce)(xs)

    np.testing.assert_array_equal(actual[0], xs * 2)
    np.testing.assert_array_equal(actual[1]["sum"], xs.sum())


def test_treduce_preserves_local_ref_effects_through_optimization():
    xs = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

    def body(x):
        ref = jax.new_ref(x)
        ref[...] += 1
        return ref[...]

    def reduce(values):
        return treduce(body, values, schedule=GPipe(1), operation=Add)

    with jax.enable_checks(True):
        traced = jax.make_jaxpr(reduce)(xs)
        optimized, _ = preprocess_jaxpr(traced)
        (actual,) = core.jaxpr_as_fun(optimized)(xs)

    np.testing.assert_array_equal(actual, (xs + 1).sum(axis=0))
