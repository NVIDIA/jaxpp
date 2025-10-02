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
import multiprocessing as mp
import sys
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
        num_layers_per_stage = self.config.num_hidden_layers // self.args.pp
        stage_id = 0

        for i, layer in enumerate(self.layers):
            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]
            # Mark the end of a stage
            if i > 0 and i % num_layers_per_stage == 0:
                hidden_states = jaxpp.pipeline_enter_stage(hidden_states)
                stage_id += 1
        hidden_states = jaxpp.pipeline_enter_stage(hidden_states)
        return hidden_states


def jax_train_step(loss_fn, optimizer, remote_mesh=None):
    use_jaxpp = remote_mesh is not None
    jax_decorator = (
        partial(jaxpp.mpmd_jit_with_loop, mpmd_mesh=remote_mesh)
        if use_jaxpp
        else jax.jit
    )

    @jax_decorator
    def train_step(opt_state, params, batch):
        µbatch_grad = partial(jax.value_and_grad(loss_fn, has_aux=True), params)

        if use_jaxpp:
            (losses, (preds, _)), grad = jaxpp.treduce(
                µbatch_grad, batch, schedule=jaxpp.Std1F1B(remote_mesh.mpmd_dim)
            )

            # Apply the optimizer as usual
            (updates, opt_state) = optimizer.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            losses = jnp.array(losses)
            preds = jnp.array(preds)

        else:
            grad = None
            losses = []
            preds = []

            for ubatch_idx in range(args.num_ubatches):
                (loss, (_preds, _)), pgrad = µbatch_grad(batch[ubatch_idx])
                losses.append(loss)
                preds.append(_preds)
                grad = (
                    jax.tree_util.tree_map(jnp.add, grad, pgrad)
                    if grad is not None
                    else pgrad
                )

            (updates, opt_state) = optimizer.update(grad, opt_state, params)
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


def main(args, process_id=None):
    """
    Simple example using JaxPP with Std1F1B schedule where each stage has two
    layers.
    Each pipeline-parallel rank uses a (1, 1) mesh, for a total of 1 device per rank.
    """

    assert process_id is not None
    jax.distributed.initialize(
        "localhost:1234",
        num_processes=args.pp,
        process_id=process_id,
        local_device_ids=(process_id,),
    )
    jaxpp_mesh = jaxpp.MpmdMesh(
        jax.sharding.Mesh(
            np.array(jax.devices()).reshape((len(jax.devices()), 1, 1)),
            ("stages", "data", "model"),
        ),
        "stages",
    )

    args.dtype = jax.numpy.dtype(args.dtype)

    rng = jax.random.PRNGKey(0)
    config = BertConfig(
        num_hidden_layers=args.pp * 2,
        hidden_size=12 * 2,
        intermediate_size=12 * 2 * 3,
    )

    model = JaxBasicBertModel(config, args, dtype=args.dtype)
    model_jaxpp = JaxPPBasicBertModel(config, args, dtype=args.dtype)

    optimizer = optax.adam(learning_rate=0.005)

    shape = (args.num_ubatches, 16, 128, config.hidden_size)

    hidden_states = jax.random.uniform(rng, shape, dtype=args.dtype)

    # Model Initialization
    params = model.init(rng, hidden_states[0])
    params_jaxpp = model_jaxpp.init(rng, hidden_states[0])

    opt_state = optimizer.init(params)
    opt_state_jaxpp = optimizer.init(params_jaxpp)

    def get_loss_fn(_model):
        def loss_fn(params, batch):
            res = _model.apply(params, batch)
            return (jnp.mean((res - batch) ** 2) / args.num_ubatches, (res, 4))

        return loss_fn

    jitted_train_step_fn = jax_train_step(get_loss_fn(model), optimizer)
    jaxpp_train_step_fn = jax_train_step(
        get_loss_fn(model_jaxpp), optimizer, remote_mesh=jaxpp_mesh
    )

    # =========================== 1st step of inference =========================== #

    jaxpp_opt_state, jaxpp_params, jaxpp_loss, jaxpp_preds = jaxpp_train_step_fn(
        opt_state_jaxpp, params_jaxpp, batch=hidden_states
    )
    print(f"Done first step JAXPP, loss: {np.array(jaxpp_loss)}")

    jax_opt_state, jax_params, jax_loss, jax_preds = jitted_train_step_fn(
        opt_state, params, hidden_states
    )
    print(f"Done first step JIT, loss: {np.array(jax_loss)}")

    # ============================== VALIDATION ============================== #

    print(f"\n{'=' * 34} VALIDATION {'=' * 34}\n")

    rtol = atol = 1e-3 if args.dtype == jnp.float32 else 1e-2

    def is_close(a, b):
        return np.isclose(a, b, rtol=rtol, atol=atol).all()

    with assert_context("OPT State"):
        opt_state_allclose = jax.tree_util.tree_map(
            lambda state_a, state_b: is_close(state_a, state_b.to_mpmd_local_array),
            jax_opt_state,
            jaxpp_opt_state,
        )

        success = True
        for k, v in jax.tree_util.tree_leaves_with_path(opt_state_allclose):
            if not v:
                if success:
                    print("")  # return to the next line
                success = False
                print(f"\t [*] {jax.tree_util.keystr(k)}: FAIL !")

    if not success:
        raise AssertionError("Opt State Validation Error")

    with assert_context("Params"):
        new_params_allclose = jax.tree_util.tree_map(
            lambda params_a, params_b: is_close(params_a, params_b.to_mpmd_local_array),
            jax_params,
            jaxpp_params,
        )

        success = True
        for k, v in jax.tree_util.tree_leaves_with_path(new_params_allclose):
            if not v:
                if success:
                    print("")  # return to the next line
                success = False
                print(f"\t [*] {jax.tree_util.keystr(k)}: FAIL !")

        if not success:
            raise AssertionError("Params Validation Error")

    with assert_context("Loss"):
        is_close(jax_loss, jaxpp_loss.to_mpmd_local_array)

    with assert_context("Prediction"):
        is_close(jax_preds, jaxpp_preds.to_mpmd_local_array)

    # =============================== TRAINING =============================== #

    print(f"\n{'=' * 29} TRAINING: {args.train_steps:04d} Steps {'=' * 29}")

    rtol = atol = 1e-4 if args.dtype == jnp.float32 else 5e-4

    for step in range(args.train_steps):
        jax_opt_state, jax_params, jax_loss, jax_preds = jitted_train_step_fn(
            jax_opt_state, jax_params, hidden_states
        )

        jaxpp_opt_state, jaxpp_params, jaxpp_loss, jaxpp_preds = jaxpp_train_step_fn(
            jaxpp_opt_state, jaxpp_params, batch=hidden_states
        )

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"\n[{step + 1:04}/{args.train_steps:04}]:"
                f"\n\t- JAX Loss:   {np.array(jax_loss).sum()}"
                f"\n\t- JAXPP Loss: {jaxpp_loss.to_mpmd_local_array.sum()}"
            )

        # Adapting the tolerance is necessary due to small differences
        # building up over time and leading to a progressive drift.
        rtol = atol = 1e-4 if args.dtype == jnp.float32 else 5e-4

        is_close(jax_loss, jaxpp_loss.to_mpmd_local_array)

    print("\nSUCCESS !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="JAXPP",
        description="Example Usage: \n"
        "  python examples/basic.py --pp=2"
        "  python examples/basic.py --pp=2 --multicontroller",
    )

    parser.add_argument(
        "--pp", type=int, default=1, help="Number of pipeline parallel ranks."
    )

    parser.add_argument(
        "--num_ubatches", type=int, default=8, help="Number of micro batches."
    )

    parser.add_argument(
        "--train_steps", type=int, default=500, help="Number of training steps."
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Compute Precision of the model.",
    )

    def parse_bool(s):
        if s.lower() in ["true", "1", "t", "y", "yes"]:
            return True
        elif s.lower() in ["false", "0", "f", "n", "no"]:
            return False
        raise ValueError("Not a valid boolean string")

    args = parser.parse_args()

    assert args.pp > 0, "Expected at least one worker."
    assert args.num_ubatches > 0, "Expected at least one microbatch."
    assert args.train_steps <= 500, "Training Steps over 500 has not been tested."
    if args.pp == 1:
        main(args, 0)
        exit(0)

    processes = []
    exitcode = 0
    for i in range(args.pp):
        p = mp.Process(target=main, args=(args, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            exitcode = p.exitcode
    sys.exit(exitcode)
