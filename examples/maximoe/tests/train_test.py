# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Trainer configuration, optimizer precision, and JaxPP task sharing."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jaxpp.api as jaxpp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import AxisType

from maximoe import model, test_util, train, train_test_util

jax.config.parse_flags_with_absl()


class ConfigTest(test_util.MaximoeTestCase):
    @parameterized.parameters(("None", None), ("profiles/run", "profiles/run"))
    def test_nullable_str(self, value, expected):
        self.assertEqual(train.nullable_str(value), expected)

    def test_ep_as_context_moves_expert_axis_from_batch_to_sequence(self):
        config = dataclasses.replace(
            train.Config(),
            pp=16,
            data=2,
            ep=8,
            schedule="kimik2",
            num_pipeline_repeats=2,
            per_device_batch_size=0.125,
            num_microbatches=16,
            seq_length=16_384,
            mtp_num_layers=0,
            ep_as_context=True,
        )

        self.assertEqual(config.data_parallel_size, 2)
        self.assertEqual(config.microbatch_size, 2)
        self.assertEqual(config.num_microbatches, 16)
        mesh = jax.sharding.AbstractMesh(
            (16, 2, 1, 8),
            ("stage", "data", "fsdp", "expert"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        batch = train.batch_shapes(config, mesh)
        self.assertEqual(batch.inputs.sharding.spec, jax.P(None, "data", "expert"))

    @parameterized.parameters({"seq_length": 130}, {"num_query_heads": 10})
    def test_ep_as_context_requires_ep_divisible_sequence_and_heads(self, **override):
        with self.assertRaisesRegex(ValueError, "ep-as-context"):
            dataclasses.replace(
                train.tiny_config(), ep=4, ep_as_context=True, **override
            )

    def test_ep_as_context_allows_finite_capacity_for_learned_routing(self):
        config = dataclasses.replace(
            train.tiny_config(), ep=4, ep_as_context=True, capacity_factor=1.2
        )
        self.assertEqual(config.capacity_factor, 1.2)

    @parameterized.parameters({"param_dtype": jnp.bfloat16}, {"dtype": jnp.float32})
    def test_compressed_expert_masters_reject_incompatible_configs(self, **override):
        with self.assertRaisesRegex(ValueError, "expert master compression"):
            dataclasses.replace(
                train.tiny_config(), compress_expert_master_params=True, **override
            )


class RouteTest(test_util.MaximoeTestCase):
    @parameterized.named_parameters(("replicated", False), ("sharded", True))
    def test_context_aux_loss_matches_whole_sequences(self, sharded):
        if sharded and jax.device_count() < 2:
            self.skipTest("requires two devices")
        config = dataclasses.replace(
            train.tiny_config(),
            num_experts=2,
            num_experts_per_tok=1,
            moe_aux_loss_weight=1,
        )
        context_config = dataclasses.replace(config, ep=2, ep_as_context=True)
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[: 2 if sharded else 1]),
            ("expert",),
            axis_types=(AxisType.Explicit,),
        )

        def aux_loss(config, gate, inputs):
            return train.route(config, gate, bias, inputs)[2]

        with jax.set_mesh(mesh):
            inputs = jnp.asarray(np.random.default_rng(42).normal(size=(2, 8, 2)))
            gate = jnp.asarray([[1.0, -0.3], [0.2, 0.8]])
            bias = jnp.zeros(2, jnp.float32)
            expected_loss, expected_grad = jax.jit(
                jax.value_and_grad(functools.partial(aux_loss, config, inputs=inputs))
            )(gate)
            chunks = model.context_to_expert_batch(inputs, sequence_shards=2)
            if sharded:
                chunks = jax.device_put(chunks, jax.P("expert", None, None))
            actual_loss, actual_grad = jax.jit(
                jax.value_and_grad(
                    functools.partial(aux_loss, context_config, inputs=chunks)
                )
            )(gate)

        self.assertAllClose(actual_loss, expected_loss, rtol=1e-6, atol=0)
        self.assertAllClose(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)
        self.assertGreater(np.max(np.abs(np.asarray(expected_grad))), 0)


class OptimizerTest(test_util.MaximoeTestCase):
    @parameterized.named_parameters(("cast", False), ("flushing_cast", True))
    def test_compressed_expert_master_round_trips_all_high_words(self, flush):
        # Every high word, with low words exactly on, below, at, and above the
        # BF16 rounding midpoint. Low words are otherwise stored verbatim.
        low = np.asarray([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF], np.uint32)
        bits = (np.arange(65536, dtype=np.uint32)[:, None] << 16) | low
        bits = jnp.asarray(bits.ravel())

        def pack(bits):
            master = jax.lax.bitcast_convert_type(bits, jnp.float32)
            compute = master.astype(jnp.bfloat16)
            if flush:
                # Cover conversions that discard NaN sign/payload or flush
                # subnormals and negative zero to positive zero.
                high = jax.lax.bitcast_convert_type(compute, jnp.uint16)
                high = jnp.where((high & 0x7FFF) > 0x7F80, 0x7FC0, high)
                high = jnp.where((high & 0x7F80) == 0, 0, high).astype(jnp.uint16)
                compute = jax.lax.bitcast_convert_type(high, jnp.bfloat16)
            return compute, train.pack_expert_master(master, compute)

        # Separate programs, as the pipeline loop separates packing from unpacking.
        compute, residual = jax.jit(pack)(bits)
        restored = jax.jit(train.unpack_expert_master)(compute, residual)
        self.assertEqual(sum(leaf.dtype.itemsize for leaf in residual), 3)
        self.assertArraysEqual(np.asarray(restored).view(np.uint32), bits)

    def test_compressed_expert_masters_reject_host_offload(self):
        config = dataclasses.replace(
            train.tiny_config(),
            compress_expert_master_params=True,
            host_offload_regex=r"^\.params\.layers\[\d+\]\.ffn\.wo$",
        )
        mesh = jax.sharding.AbstractMesh(
            (1,), ("expert",), axis_types=(AxisType.Explicit,)
        )
        excluded_axes = {"data", "fsdp"}
        state = train.shape_tree(
            train.train_state_specs(config),
            train.filter_axes(train.train_state_shardings(config), excluded_axes),
            mesh,
            host_offload_regex=config.host_offload_regex,
        )
        step = functools.partial(
            train.train_step,
            config,
            train.filter_axes(train.parameter_shardings(config), excluded_axes),
        )
        with jax.sharding.use_abstract_mesh(mesh):
            with self.assertRaisesRegex(ValueError, "cannot be host-offloaded"):
                jax.eval_shape(step, state, train.batch_shapes(config, mesh))

    def test_default_state_uses_fp32_masters_and_bf16_moments(self):
        config = train.tiny_config()
        state = train.train_state_specs(config)

        def is_spec(value):
            return isinstance(value, train.ArraySpec)

        self.assertEqual(config.dtype, jnp.bfloat16)
        for spec in jax.tree.leaves(state.params, is_leaf=is_spec):
            self.assertEqual(spec.dtype, jnp.float32)
        for spec in state.router.biases:
            self.assertEqual(spec.dtype, jnp.float32)
        for moments in (state.opt.mu, state.opt.nu):
            for spec in jax.tree.leaves(moments, is_leaf=is_spec):
                self.assertEqual(spec.dtype, jnp.bfloat16)

    def test_small_updates_accumulate_in_master_weights(self):
        config = dataclasses.replace(train.tiny_config(), weight_decay=0.0)
        update = jax.jit(functools.partial(train.adam_update, config))
        param = jnp.ones((1,), jnp.float32)
        mu = nu = jnp.zeros((1,), jnp.bfloat16)
        for count in range(1, 101):
            param, mu, nu = update(
                param,
                jnp.ones((1,), jnp.bfloat16),
                mu,
                nu,
                count=jnp.asarray(count),
                learning_rate=jnp.asarray(1e-4),
            )

        self.assertEqual(param.dtype, jnp.float32)
        self.assertEqual(mu.dtype, jnp.bfloat16)
        self.assertEqual(nu.dtype, jnp.bfloat16)
        # Every individual update is below BF16 resolution near one, but the
        # accumulated master-weight change must eventually reach compute weights.
        self.assertBetween(float(param[0]), 0.989, 0.991)
        self.assertLess(float(param.astype(jnp.bfloat16)[0]), 1.0)

    def test_router_bias_stays_fp32_with_bf16_parameters(self):
        config = dataclasses.replace(train.tiny_config(), param_dtype=jnp.bfloat16)
        specs = train.train_state_specs(config)
        for spec in specs.router.biases:
            self.assertEqual(spec.dtype, jnp.float32)

        state = train.RouterState((jnp.ones((2,), specs.router.biases[0].dtype),))
        updated = train.update_router_biases(config, state, (jnp.array([0, 2]),))

        # This increment would disappear when stored in BF16 near one.
        self.assertAllClose(
            updated.biases[0], np.array([1.001, 0.999], np.float32), rtol=0, atol=1e-7
        )

    def test_weight_update_uses_unrounded_new_moments(self):
        config = dataclasses.replace(train.tiny_config(), weight_decay=0.0)
        grad = jnp.asarray([0.003, -0.017, 0.123], jnp.bfloat16)
        zero = jnp.zeros_like(grad)

        result = jax.jit(functools.partial(train.adam_update, config))(
            jnp.ones(grad.shape, jnp.float32),
            grad,
            zero,
            zero,
            count=jnp.asarray(1),
            learning_rate=jnp.asarray(0.01),
        )

        # At the first step, bias correction cancels both beta factors. Computing
        # the update from rounded BF16 moments breaks this identity.
        g = np.asarray(grad, dtype=np.float64)
        expected = 1.0 - 0.01 * g / (np.abs(g) + config.adam_epsilon)
        self.assertAllClose(
            result.param, expected, rtol=0, atol=2e-7, check_dtypes=False
        )
        self.assertEqual(result.mu.dtype, jnp.bfloat16)
        self.assertEqual(result.nu.dtype, jnp.bfloat16)


# Trace under the trainer's config (see ``train.main``).
@jtu.with_config(jax_enable_checks=True, jax_remove_size_one_mesh_axis_from_type=True)
class TrainStepTest(test_util.MaximoeTestCase):
    # ZeRO-1 gathers expert compute weights over FSDP once per step, so
    # unpacking reshards them back. ZeRO-3 keeps them sharded and runs the
    # recipes' interleaved KimiK2 schedule, with two stages per rank.
    @parameterized.named_parameters(
        dict(testcase_name="zero1_1f1b", zero=1),
        dict(
            testcase_name="zero3_kimik2",
            zero=3,
            schedule="kimik2",
            num_pipeline_repeats=2,
            num_layers=4,
        ),
    )
    def test_compressed_expert_masters_train_bitwise_identically(self, **override):
        self.skip_unless_gpus(4)
        config = dataclasses.replace(
            train.tiny_config(),
            pp=2,
            fsdp=2,
            num_microbatches=2,
            # Expert layers on every stage catch residuals placed with another
            # layer's counts. The MTP layer's residuals and counts come last.
            num_dense_layers=0,
            mtp_num_layers=1,
            # Pin the gradient-Ref path that compression must preserve.
            expert_gradient_accumulation=True,
            **override,
        )
        runs = []
        for compressed in (False, True):
            step, state, batch = train_test_util.prepare_step(
                dataclasses.replace(config, compress_expert_master_params=compressed),
                jax.devices()[:4],
            )
            if compressed:
                # Only a before-loop task reads each FP32 expert master, so
                # JaxPP can free it before the pipeline loop.
                jaxpr = step.closed_jaxpr.jaxpr
                state_vars = jax.tree.unflatten(
                    jax.tree.structure(state),
                    jaxpr.invars[len(step.consts) :][: len(jax.tree.leaves(state))],
                )
                params = state_vars.params
                for ffn in [
                    *(layer.ffn for layer in params.layers),
                    params.mtp.layer.ffn,
                ]:
                    for master in (ffn.wi, ffn.wo):
                        readers = [
                            eqn.params["task_name"]
                            for eqn in jaxpr.eqns
                            if eqn.primitive.name == "task"
                            and any(var is master for var in eqn.invars)
                        ]
                        self.assertLen(readers, 1, readers)
                        self.assertStartsWith(readers[0], "before_loop")
            for _ in range(2):
                state, metrics = jax.block_until_ready(step(state, batch))
            runs.append((state, metrics))
        # Exact masters make every optimizer input identical, so compare the
        # bits of all state and metrics, including router biases updated from
        # the counts that place the residuals.
        for (path, actual), expected in zip(
            jax.tree.leaves_with_path(runs[1]), jax.tree.leaves(runs[0]), strict=True
        ):
            for got, want in zip(
                map(np.asarray, train_test_util.local_arrays(actual)),
                map(np.asarray, train_test_util.local_arrays(expected)),
                strict=True,
            ):
                self.assertEqual(got.dtype, want.dtype)
                bits = np.dtype(f"u{got.dtype.itemsize}")
                self.assertArraysEqual(
                    got.view(bits), want.view(bits), err_msg=jax.tree_util.keystr(path)
                )

    def test_repeated_ragged_dot_combine_tasks_share_jaxpr(self):
        self.skip_unless_gpus(2)
        devices = jax.devices()

        config = dataclasses.replace(
            train.tiny_config(),
            pp=1,
            ep=2,
            schedule="kimik2",
            num_pipeline_repeats=4,
            num_layers=4,
            num_dense_layers=0,
            mtp_num_layers=0,
            per_device_batch_size=2,
            num_microbatches=2,
            fuse_steady_state=True,
        )
        mesh = train.make_mesh(config, devices[:2])
        mpmd_mesh = jaxpp.MpmdMesh(mesh, "stage")
        local_mesh = mpmd_mesh.lowering_mesh()
        excluded_axes = {
            axis
            for axis in ("data", "fsdp", "expert")
            if local_mesh.shape.get(axis, 1) == 1
        }
        param_shardings = train.filter_axes(
            train.parameter_shardings(config), excluded_axes
        )
        state_shardings = train.filter_axes(
            train.train_state_shardings(config), excluded_axes | {"data"}
        )
        state = train.shape_tree(
            train.train_state_specs(config), state_shardings, local_mesh
        )
        batch = train.batch_shapes(config, local_mesh)
        state_shardings = jax.tree.map(lambda value: value.sharding, state)
        batch_shardings = jax.tree.map(lambda value: value.sharding, batch)
        metric_shardings = train.Metrics(
            *(jax.NamedSharding(local_mesh, jax.P()) for _ in train.Metrics._fields)
        )
        traceable = jaxpp.mpmd_jit_with_loop(
            functools.partial(train.train_step, config, param_shardings),
            mpmd_mesh=mpmd_mesh,
            in_shardings=(state_shardings, batch_shardings),
            out_shardings=(state_shardings, metric_shardings),
            donate_argnums=(0,),
        )

        prepared = traceable.trace_and_place(state, batch).mpmdify()
        tasks = {
            eqn.params["task_name"]: eqn
            for eqn in prepared.closed_jaxpr.eqns
            if eqn.primitive.name == "task"
        }

        self.assertIs(
            tasks["fwd_1"].params["call_jaxpr"], tasks["fwd_2"].params["call_jaxpr"]
        )
        self.assertIs(
            tasks["fused_fwd_1_bwd_2"].params["call_jaxpr"],
            tasks["fused_fwd_2_bwd_1"].params["call_jaxpr"],
        )

    def test_context_parallel_training_matches_whole_sequences(self):
        """EP-as-context, with head-sharded MLA up-projections, trains the same."""
        self.skip_unless_gpus(4)

        config = dataclasses.replace(
            train.tiny_config(),
            pp=2,
            ep=2,
            per_device_batch_size=1,
            num_microbatches=2,
            warmup_steps=1,
            decay_steps=1,
            min_learning_rate=1e-3,
        )
        runs = []
        for ep_as_context in (False, True):
            step, state, batch = train_test_util.prepare_step(
                dataclasses.replace(config, ep_as_context=ep_as_context),
                jax.devices()[:4],
            )
            metrics = []
            for _ in range(3):
                state, step_metrics = jax.block_until_ready(step(state, batch))
                metrics.append(
                    [
                        float(np.asarray(train_test_util.local_arrays(value)[0]))
                        for value in (step_metrics.loss, step_metrics.grad_norm)
                    ]
                )
            runs.append(metrics)
        # Same sequences and dropless routing; only BF16 reduction order differs.
        self.assertAllClose(runs[1], runs[0], rtol=2e-2, atol=0)

    def test_ring_of_experts_trains_like_dedup(self):
        """Ring of experts trains the same as the dedup exchange."""
        self.skip_unless_gpus(4)

        config = dataclasses.replace(
            train.tiny_config(),
            pp=2,
            ep=2,
            per_device_batch_size=1,
            num_microbatches=2,
            warmup_steps=1,
            decay_steps=1,
            min_learning_rate=1e-3,
        )
        runs = []
        for ring in (False, True):
            step, state, batch = train_test_util.prepare_step(
                dataclasses.replace(config, use_ring_of_experts=ring), jax.devices()[:4]
            )
            metrics = []
            for _ in range(3):
                state, step_metrics = jax.block_until_ready(step(state, batch))
                metrics.append(
                    [
                        float(np.asarray(train_test_util.local_arrays(value)[0]))
                        for value in (step_metrics.loss, step_metrics.grad_norm)
                    ]
                )
            runs.append(metrics)
        # Dropless routing either way; ring reduces expert outputs in BF16.
        self.assertAllClose(runs[1], runs[0], rtol=2e-2, atol=0)

    def test_zero_stages_train_the_same(self):
        """ZeRO-1/2/3 move FSDP collectives but not the optimizer update."""
        self.skip_unless_gpus(8)

        config = dataclasses.replace(
            train.tiny_config(),
            pp=2,
            data=2,
            fsdp=2,
            per_device_batch_size=1,
            num_microbatches=2,
            warmup_steps=1,
            decay_steps=1,
            min_learning_rate=1e-3,
        )
        runs = []
        for zero in (1, 2, 3):
            step, state, batch = train_test_util.prepare_step(
                dataclasses.replace(config, zero=zero), jax.devices()[:8]
            )
            # DATA replicates the training state; FSDP shards it.
            mu = train_test_util.local_arrays(state.opt.mu.layers[1].ffn.wi)[0]
            self.assertEqual(mu.sharding.spec, jax.P(None, "fsdp", None))
            metrics = []
            for _ in range(3):
                state, step_metrics = jax.block_until_ready(step(state, batch))
                metrics.append(
                    [
                        float(np.asarray(train_test_util.local_arrays(value)[0]))
                        for value in (step_metrics.loss, step_metrics.grad_norm)
                    ]
                )
            runs.append(metrics)
        # Only BF16 gradient reduction order differs between the stages.
        self.assertAllClose(runs[1], runs[0], rtol=2e-2, atol=0)
        self.assertAllClose(runs[2], runs[0], rtol=2e-2, atol=0)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
