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
from functools import partial

import jax
import jax.lib
import jax.random
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax._src.lib import cuda_versions
from jax.lib import xla_bridge

import jaxpp
import jaxpp.api
import jaxpp.schedules
from jaxpp.mesh import MpmdMesh

TENSOR_SHAPE = (10, 100)


class DropoutEnabledTrainState(TrainState):
    key: jax.Array


class DropoutEnabledModel(nn.Module):
    dense_dim: int
    n_layers: int
    use_dropout: bool

    @nn.compact
    def __call__(self, x, training: bool):
        for _ in range(self.n_layers):
            x = nn.Dense(self.dense_dim)(x)

            if self.use_dropout:
                # Set the dropout layer with a `rate` of 50%.
                # When the `deterministic` flag is `True`, dropout is turned off.
                x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = jaxpp.api.pipeline_enter_stage(x)
        return x

    @staticmethod
    def train_step(
        state: DropoutEnabledTrainState | TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ):
        try:
            dropout_rng = jax.random.fold_in(key=state.key, data=state.step)
        except AttributeError:
            dropout_rng = None

        def loss_fn(params, data):
            _inputs, _targets = data

            kwargs = {} if dropout_rng is None else {"rngs": {"dropout": dropout_rng}}

            logits = state.apply_fn(
                {"params": params}, x=_inputs, training=True, **kwargs
            )

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, _targets)

            return loss.mean(), logits

        (loss, aux), grads = jaxpp.api.treduce(
            partial(jax.value_and_grad(loss_fn, has_aux=True, argnums=0), state.params),
            (inputs, targets),
            schedule=jaxpp.schedules.Eager1F1B(num_stages=1),
        )

        state = state.apply_gradients(grads=grads)
        return state, loss.mean()


class FlaxModelExecutionTest(unittest.TestCase):
    def setUp(self):
        """Set up a basic environment before each test with 8 GPU devices."""
        xla_bridge.get_backend.cache_clear()

        if cuda_versions.cuda_device_count() < 1:
            # Skip the test if no GPU are available
            pytest.skip()

        self._mesh = MpmdMesh(
            jax.sharding.Mesh(
                np.array(jax.devices())[:1].reshape(1, 1, 1), ("stage", "data", "model")
            ),
            "stage",
        )

        self._root_key = jax.random.key(seed=0)

    def _exec_training_loop(self, model: nn.Module, data_key, model_rngs: dict):
        params = model.init(
            model_rngs,
            jax.random.uniform(key=data_key, shape=TENSOR_SHAPE),
            training=False,
        )["params"]

        if "dropout" in model_rngs:
            model_state = DropoutEnabledTrainState.create(
                apply_fn=model.apply,
                params=params,
                key=model_rngs["dropout"],
                tx=optax.adam(1e-3),
            )

        else:
            model_state = TrainState.create(
                apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
            )

        train_step = jaxpp.api.mpmd_jit_with_loop(
            model.train_step, mpmd_mesh=self._mesh
        )
        for step_id in range(10):
            data_rng = jax.random.fold_in(key=data_key, data=step_id + 1)
            x = jax.random.uniform(key=data_rng, shape=(1, *TENSOR_SHAPE))
            y = jax.random.randint(
                key=data_rng, shape=(1, TENSOR_SHAPE[0]), minval=1, maxval=10
            )

            model_state, loss = train_step(model_state, x, y)

            # Test API is functional - No timing test
            loss.to_mpmd_local_array.block_until_ready()  # Forced Resync to allow accurate timing

            print(f"[Step {step_id+1:02d}] Train Loss: {loss:.3f}")

    def test_flax_model_with_dropout(self):
        data_key, params_key, dropout_key = jax.random.split(key=self._root_key, num=3)
        model = DropoutEnabledModel(dense_dim=10, n_layers=5, use_dropout=True)
        model_rngs = {"params": params_key, "dropout": dropout_key}
        self._exec_training_loop(model=model, data_key=data_key, model_rngs=model_rngs)

    def test_flax_model_without_dropout(self):
        data_key, params_key = jax.random.split(key=self._root_key, num=2)
        model = DropoutEnabledModel(dense_dim=10, n_layers=5, use_dropout=False)
        model_rngs = {"params": params_key}
        self._exec_training_loop(model=model, data_key=data_key, model_rngs=model_rngs)


if __name__ == "__main__":
    unittest.main()
