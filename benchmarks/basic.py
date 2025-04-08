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

import argparse
import os
from contextlib import contextmanager
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertLayer

import jaxpp.api as jaxpp


class JaxBasicBertModel(nn.Module):
    config: BertConfig
    args: argparse.Namespace
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBertLayer(self.config, name=f"flax_bert_layer_{i}", dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states):
        for layer in self.layers:
            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]
        return hidden_states


class JaxPPBasicBertModel(JaxBasicBertModel):
    def __call__(self, hidden_states):
        num_layers_per_stage = self.config.num_hidden_layers // self.args.num_workers
        stage_id = 0

        for i, layer in enumerate(self.layers):
            # Mark that we are entering a new stage at every layer
            if (
                self.args.num_workers != 1
                and i % num_layers_per_stage == 0
                and stage_id < self.args.num_workers
            ):
                hidden_states = jaxpp.pipeline_enter_stage(hidden_states)
                stage_id += 1

            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]
        return hidden_states


def jax_train_step(loss_fn, optimizer, remote_mesh=None):
    use_jaxpp = remote_mesh is not None
    jax_decorator = (
        partial(jaxpp.pipelined, mpmd_mesh=remote_mesh) if use_jaxpp else jax.jit
    )

    @jax_decorator
    def train_step(opt_state, params, batch):
        def µbatch_grad(µbatch):
            (loss, (preds, _)), grads = jax.value_and_grad(
                loss_fn,
                # * has_aux (bool): Optional, bool. Indicates whether fun returns a pair
                # where the first element is considered the output of the mathematical
                # function to be differentiated and the second element is auxiliary data
                # Default: False.
                has_aux=True,
            )(params, µbatch)
            return jaxpp.LoopOutput(grads, (loss, preds))

        if use_jaxpp:
            grad, (losses, preds), _, _ = jaxpp.accumulate_grads(
                µbatch_grad,
                batch=batch,
                out_shardings=None,
                schedule=jaxpp.Eager1F1B(1),
            )

            # divide the grads by the number of micro batches - preserve relative grad norm
            num_mubatch = int(batch.shape[1])
            grad = jax.tree_util.tree_map(lambda x: jnp.divide(x, num_mubatch), grad)

            # Apply the optimizer as usual
            (updates, opt_state) = optimizer.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            losses = jnp.array(losses)
            preds = jnp.array(preds)

        else:
            acc_grads = None
            losses = []
            preds = []

            for ubatch_idx in range(args.num_ubatches):
                grads, (loss, _preds), _, _ = µbatch_grad(batch[0, ubatch_idx])
                losses.append(loss)
                preds.append(_preds)
                acc_grads = (
                    jax.tree_util.tree_map(jnp.add, acc_grads, grads)
                    if acc_grads is not None
                    else grads
                )

            # Divide by the number of microbatches to normalize
            acc_grads = jax.tree_util.tree_map(
                lambda x: jnp.divide(x, args.num_ubatches), acc_grads
            )

            (updates, opt_state) = optimizer.update(acc_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

        return opt_state, new_params, jnp.asarray(losses), jnp.asarray(preds)

    return train_step


@contextmanager
def assert_context(tensor_name):
    print(
        f"[*] `{tensor_name}` Validation {'.' * (80 - 19 - 6 - len(tensor_name))} ",
        end="",
    )
    try:
        yield
        print("PASS !")
    except AssertionError as e:
        print("FAIL !")
        raise AssertionError(f"`{tensor_name}` validation failure") from e


def main(args):
    """
    Simple program for testing purposes using one single-node
    with `args.num_workers=1` and `num_stages=1` where each stage has two layers.
    The worker stage uses a (1, 1) mesh, for a total of 1 device.
    """
    xla_python_client_mem_fraction = ".4"

    jaxpp_mesh = jaxpp.RemoteMpmdMesh(
        args.num_workers,
        (1, 1),
        ("data", "model"),
        _env={"XLA_PYTHON_CLIENT_MEM_FRACTION": xla_python_client_mem_fraction},
    )

    # Normally, when using JaxPP, the driver process should set the default device
    # to the cpu as shown in the line below.
    # jax.config.update("jax_default_device", jax.local_devices(backend="cpu")[0])
    # For this test _only_, we instead decide to set `XLA_PYTHON_CLIENT_MEM_FRACTION`
    # to a sensible fraction so that the driver can use the same GPU as the worker
    # to speedup CI.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = xla_python_client_mem_fraction

    args.dtype = jax.numpy.dtype(args.dtype)

    rng = jax.random.PRNGKey(0)
    config = BertConfig(
        num_hidden_layers=args.num_workers * 2,
        hidden_size=12 * 2,
        intermediate_size=12 * 2 * 3,
    )

    model = JaxBasicBertModel(config, args, dtype=args.dtype)
    model_jaxpp = JaxPPBasicBertModel(config, args, dtype=args.dtype)

    optimizer = optax.adam(learning_rate=0.005)

    shape = (args.num_workers, args.num_ubatches, 32, 128, config.hidden_size)

    hidden_states = jax.random.uniform(rng, shape, dtype=args.dtype)

    # Model Initialization
    params = model.init(rng, hidden_states[0, 0])
    params_jaxpp = model_jaxpp.init(rng, hidden_states[0, 0])

    opt_state = optimizer.init(params)
    opt_state_jaxpp = optimizer.init(params_jaxpp)

    def get_loss_fn(_model):
        def loss_fn(params, batch):
            res = _model.apply(params, batch)
            return (jnp.mean((res - batch) ** 2), (res, 4))

        return loss_fn

    jitted_train_step_fn = jax_train_step(get_loss_fn(model), optimizer)
    jaxpp_train_step_fn = jax_train_step(
        get_loss_fn(model_jaxpp), optimizer, remote_mesh=jaxpp_mesh
    )

    # =========================== 1st step of inference =========================== #

    jax_opt_state, jax_params, jax_loss, jax_preds = jitted_train_step_fn(
        opt_state, params, hidden_states
    )
    print(f"Done first step JIT, loss: {np.array(jax_loss)}")

    with jaxpp_mesh:
        jaxpp_opt_state, jaxpp_params, jaxpp_loss, jaxpp_preds = jaxpp_train_step_fn(
            opt_state_jaxpp, params_jaxpp, hidden_states
        )
        print(f"Done first step JAXPP, loss: {np.array(jaxpp_loss)}")

    # ============================== VALIDATION ============================== #

    print(f"\n{'=' * 34} VALIDATION {'=' * 34}\n")

    rtol = atol = 1e-3 if args.dtype == jnp.float32 else 1e-2

    with assert_context("OPT State"):
        opt_state_allclose = jax.tree_util.tree_map(
            lambda state_a, state_b: np.testing.assert_allclose(
                state_a, state_b, rtol=rtol, atol=atol
            ),
            jax_opt_state,
            jaxpp_opt_state,
        )

        success = True
        for k, v in jax.tree_util.tree_flatten_with_path(opt_state_allclose)[0]:
            if not v:
                if success:
                    print("")  # return to the next line
                success = False
                print(f"\t [*] {jax.tree_util.keystr(k)}: FAIL !")

    if not success:
        raise AssertionError("Opt State Validation Error")

    with assert_context("Params"):
        new_params_allclose = jax.tree_util.tree_map(
            lambda params_a, params_b: np.testing.assert_allclose(
                params_a, params_b, rtol=rtol, atol=atol
            ),
            jax_params,
            jaxpp_params,
        )

        success = True
        for k, v in jax.tree_util.tree_flatten_with_path(new_params_allclose)[0]:
            if not v:
                if success:
                    print("")  # return to the next line
                success = False
                print(f"\t [*] {jax.tree_util.keystr(k)}: FAIL !")

        if not success:
            raise AssertionError("Params Validation Error")

    with assert_context("Loss"):
        np.testing.assert_allclose(
            jax_loss, np.array(jaxpp_loss).squeeze(0), rtol=rtol, atol=atol
        )

    with assert_context("Prediction"):
        np.testing.assert_allclose(
            jax_preds, np.array(jaxpp_preds).squeeze(0), rtol=rtol, atol=atol
        )

    # =============================== TRAINING =============================== #

    print(f"\n{'=' * 29} TRAINING: {args.train_steps:04d} Steps {'=' * 29}")

    rtol = atol = 1e-4 if args.dtype == jnp.float32 else 5e-4

    for step in range(args.train_steps):
        jax_opt_state, jax_params, jax_loss, jax_preds = jitted_train_step_fn(
            jax_opt_state, jax_params, hidden_states
        )

        with jaxpp_mesh:
            jaxpp_opt_state, jaxpp_params, jaxpp_loss, jaxpp_preds = (
                jaxpp_train_step_fn(jaxpp_opt_state, jaxpp_params, hidden_states)
            )

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"\n[{step + 1:04}/{args.train_steps:04}]:"
                f"\n\t- JAX Loss:   {np.array(jax_loss).sum()}"
                f"\n\t- JAXPP Loss: {np.array(jaxpp_loss).sum()}"
            )

        # Adapting the tolerance is necessary due to small differences
        # building up over time and leading to a progressive drift.
        rtol = atol = 1e-4 if args.dtype == jnp.float32 else 5e-4

        np.testing.assert_allclose(
            jax_loss, np.array(jaxpp_loss).squeeze(0), rtol=rtol, atol=atol
        )

    print("\nSUCCESS !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="JAXPP")

    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of Ray workers."
    )

    parser.add_argument(
        "--num_ubatches", type=int, default=4, help="Number of micro batches."
    )

    parser.add_argument(
        "--train_steps", type=int, default=500, help="Number of training steps."
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        required=True,
        help="Compute Precision of the model.",
    )

    args = parser.parse_args()

    assert args.num_workers > 0, "Expected at least one worker."
    assert args.num_workers == 1, "This script has only been tested with 1 worker."
    assert args.num_ubatches > 0, "Expected at least one microbatch."
    assert args.train_steps <= 500, "Training Steps over 500 has not been tested."

    main(args)
