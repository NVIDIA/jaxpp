# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import jax
import jax.lib
import jax.random
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from jax._src.lib import cuda_versions
from jax.lib import xla_bridge

import jaxpp
import jaxpp.api
import jaxpp.schedules
from jaxpp.mesh import MpmdMesh

TENSOR_SHAPE = (10, 100)
XLA_GPU_MEM_FRACTION = 0.4


class ModelWithPassthrough(nn.Module):
    n_layers: int = 5
    dense_dim: int = TENSOR_SHAPE[-1]

    @nn.compact
    def __call__(
        self, x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        out = x

        for _ in range(self.n_layers):
            out = nn.Dense(self.dense_dim)(out)

        return out + y, x, y

    @staticmethod
    def train_step(
        state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray
    ) -> tuple[train_state.TrainState, jax.Array]:
        def loss_fn(params, data):
            (x, y), _targets = data

            # logits, x, y = state.apply_fn(
            logits, x_out, y_out = state.apply_fn({"params": params}, x=x, y=y)

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, _targets)

            return loss.mean(), (logits, x_out, y_out)

        def compute_grads(params, data):
            grad_fn = jax.value_and_grad(
                loss_fn,
                # * has_aux (bool): Optional, bool. Indicates whether fun returns a
                #                   pair where the first element is considered the
                #                   output of the mathematical function to be
                #                   differentiated and the second element is
                #                   auxiliary data.
                #                   Default: False.
                has_aux=True,
                # argnums (int | Sequence[int]): Optional, integer or sequence of
                #                                integers. Specifies which
                #                                positional argument(s) to
                #                                differentiate with respect to
                #                                Default: 0.
                argnums=0,
            )
            (loss, (logits, x, y)), grads = grad_fn(state.params, data)

            return jaxpp.api.LoopOutput(grads, (loss, (logits, x, y)))

        grads, (loss, (logits, x, y)), _, _ = jaxpp.api.accumulate_grads(
            lambda data: compute_grads(params=state.params, data=data),
            batch=(inputs, targets),
            out_shardings=None,
            schedule=jaxpp.schedules.Std1F1B(num_stages=1),
        )

        state = state.apply_gradients(grads=grads)
        return state, loss.mean()


class FlaxModelWithPassthroughExecutionTest(unittest.TestCase):
    def setUp(self):
        """Set up a basic environment before each test with 8 GPU devices."""
        xla_bridge.get_backend.cache_clear()

        if cuda_versions.cuda_device_count() < 1:
            # Skip the test if no GPU are available
            pytest.skip()

        self._mesh = MpmdMesh(
            jax.sharding.Mesh(np.array(jax.devices())[:1].reshape(1,1,1), ("stage", "data", "model")),
            "stage"
        )

        self._root_key = jax.random.key(seed=0)

    def test_flax_model_with_passthrough(self):
        data_key, params_key = jax.random.split(key=self._root_key, num=2)

        model = ModelWithPassthrough()

        x_init = jax.random.uniform(jax.random.fold_in(data_key, 1), shape=TENSOR_SHAPE)
        y_init = jax.random.uniform(jax.random.fold_in(data_key, 2), shape=TENSOR_SHAPE)

        params = model.init(params_key, x=x_init, y=y_init)["params"]

        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
        )

        # 10 Steps of Training
        for step_id in range(10):
            x_rng = jax.random.fold_in(key=data_key, data=3 * step_id)
            x = jax.random.uniform(key=x_rng, shape=[1, *TENSOR_SHAPE])

            y_rng = jax.random.fold_in(key=data_key, data=3 * step_id + 1)
            y = jax.random.uniform(key=y_rng, shape=[1, *TENSOR_SHAPE])

            targets_rng = jax.random.fold_in(key=data_key, data=3 * step_id + 2)
            targets = jax.random.randint(
                key=targets_rng, shape=(1, TENSOR_SHAPE[0]), minval=1, maxval=10
            )

            state, loss = jaxpp.api.pipelined(model.train_step, mpmd_mesh=self._mesh)(
                state, (x, y), targets
            )

            # Test API is functional - No timing test
            loss.block_until_ready()  # Forced Resync to allow accurate timing

            print(f"[Step {step_id+1:02d}] Train Loss: {loss:.3f}")

        # Verifying the passthrough is correct
        x_infer = jax.random.uniform(jax.random.fold_in(data_key, 1), TENSOR_SHAPE)
        y_infer = jax.random.uniform(jax.random.fold_in(data_key, 2), TENSOR_SHAPE)

        logits, x_out, y_out = state.apply_fn(
            {"params": state.params}, x=x_infer, y=y_infer
        )

        assert logits.shape == (10, 100)
        np.testing.assert_array_equal(x_out, x_infer)
        np.testing.assert_array_equal(y_out, y_infer)


if __name__ == "__main__":
    unittest.main()
