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

"""Folded single-layer profiling harness."""

import functools
from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from maximoe import model, profile, test_util, train

jax.config.parse_flags_with_absl()


class ProfileTest(test_util.MaximoeTestCase):
    @parameterized.parameters((8, 4_096, 32, 32_768), (16, 2_048, 16, 16_384))
    def test_folded_profile_geometry(
        self, ep, sequence_length, local_experts, packed_size
    ):
        config = profile.make_config(ep, "profiles")
        ffn = train.parameter_specs(config).layers[0].ffn
        shardings = train.filter_axes(
            train.parameter_shardings(config).layers[0].ffn, {"data", "fsdp"}
        )

        self.assertEqual(config.global_batch_size, ep)
        self.assertEqual(config.seq_length, sequence_length)
        self.assertEqual(
            config.global_batch_size * config.seq_length, profile.GLOBAL_TOKENS
        )
        self.assertEqual(config.microbatch_size, ep)
        self.assertEqual(config.num_microbatches, 1)
        self.assertEqual(config.num_layers, 1)
        self.assertIsNone(config.kernel_implementation)
        self.assertEqual(ffn.gate.shape, (7_168, 256))
        self.assertEqual(ffn.wi.shape, (256, 7_168, 4_096))
        self.assertEqual(ffn.wo.shape, (256, 2_048, 7_168))
        self.assertEqual(ffn.gate.dtype, jnp.bfloat16)
        self.assertEqual(ffn.wi.dtype, jnp.bfloat16)
        self.assertEqual(ffn.wo.dtype, jnp.bfloat16)
        self.assertEqual(shardings.gate, jax.P(None, None, reduced={"expert"}))
        self.assertEqual(shardings.wi, jax.P("expert", None, None))
        self.assertEqual(shardings.wo, jax.P("expert", None, None))

        actual_packed_size = model.storage_packed_size_per_batch(
            sequence_length=sequence_length,
            num_experts_per_tok=config.num_experts_per_tok,
            num_target_experts=local_experts,
            drop_capacity=None,
        )
        self.assertEqual(actual_packed_size, packed_size)
        self.assertEqual(
            sequence_length * min(config.num_experts_per_tok, ep),
            32_768 if ep == 8 else 16_384,
        )
        self.assertEqual(ep * sequence_length, 32_768)

    def test_folded_profile_work_numerator(self):
        ep8 = profile.make_config(8, "profiles")
        ep16 = profile.make_config(16, "profiles")
        flops = profile.calculate_routed_training_flops_per_device

        self.assertEqual(flops(ep8), 8_703_751_225_344)
        self.assertEqual(flops(ep16), 4_351_875_612_672)
        self.assertEqual(flops(ep8), 2 * flops(ep16))
        self.assertEqual(
            flops(profile.make_config(8, "profiles", num_layers=8)), 8 * flops(ep8)
        )

    def test_random_keys_are_stable_across_stack_depths(self):
        ep8 = profile.make_config(8, "profiles", num_layers=8)
        ep16 = profile.make_config(16, "profiles", num_layers=15)
        ep8_layers, ep8_input, ep8_cotangent = profile.profile_random_keys(
            ep8.param_seed, ep8.num_layers
        )
        ep16_layers, ep16_input, ep16_cotangent = profile.profile_random_keys(
            ep16.param_seed, ep16.num_layers
        )

        def sample(key):
            return jax.random.normal(key, (2, 3))

        for ep8_keys, ep16_keys in zip(
            ep8_layers, ep16_layers[: len(ep8_layers)], strict=True
        ):
            for ep8_key, ep16_key in zip(ep8_keys, ep16_keys, strict=True):
                self.assertArraysEqual(sample(ep8_key), sample(ep16_key))
        self.assertArraysEqual(sample(ep8_input), sample(ep16_input))
        self.assertArraysEqual(sample(ep8_cotangent), sample(ep16_cotangent))
        self.assertFalse(jnp.array_equal(sample(ep8_input), sample(ep8_cotangent)))

    @parameterized.parameters(1, 2)
    def test_result_and_global_diagnostic_shardings(self, num_layers):
        config = profile.make_config(8, "profiles", num_layers=num_layers)
        mesh = jax.sharding.AbstractMesh(
            (8,), ("expert",), axis_types=(jax.sharding.AxisType.Explicit,)
        )
        specs = tuple(layer.ffn for layer in train.parameter_specs(config).layers)
        shardings = train.filter_axes(
            tuple(layer.ffn for layer in train.parameter_shardings(config).layers),
            {"data", "fsdp"},
        )

        args = (
            tuple(
                jax.ShapeDtypeStruct(
                    spec.gate.shape,
                    spec.gate.dtype,
                    sharding=jax.NamedSharding(mesh, sharding.gate),
                )
                for spec, sharding in zip(specs, shardings, strict=True)
            ),
            tuple(
                jax.ShapeDtypeStruct(
                    spec.wi.shape,
                    spec.wi.dtype,
                    sharding=jax.NamedSharding(mesh, sharding.wi),
                )
                for spec, sharding in zip(specs, shardings, strict=True)
            ),
            tuple(
                jax.ShapeDtypeStruct(
                    spec.wo.shape,
                    spec.wo.dtype,
                    sharding=jax.NamedSharding(mesh, sharding.wo),
                )
                for spec, sharding in zip(specs, shardings, strict=True)
            ),
            jax.ShapeDtypeStruct(
                (8, 4_096, 7_168),
                jnp.bfloat16,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None, None)),
            ),
            jax.ShapeDtypeStruct(
                (8, 4_096, 7_168),
                jnp.bfloat16,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None, None)),
            ),
            tuple(
                jax.ShapeDtypeStruct(
                    (256,), jnp.bfloat16, sharding=jax.NamedSharding(mesh, jax.P())
                )
                for _ in range(num_layers)
            ),
        )
        with jax.sharding.use_abstract_mesh(mesh):
            result = jax.eval_shape(
                functools.partial(profile.profile_step, config), *args
            )
            signatures = jax.eval_shape(profile.numerical_signatures, result)

        self.assertEqual(
            tuple(value.sharding.spec for value in jax.tree.leaves(result)),
            (
                jax.P(),
                jax.P("expert", None, None),
                jax.P(None),
                *((jax.P(None, None, unreduced={"expert"}),) * num_layers),
                *((jax.P("expert", None, None),) * num_layers),
                *((jax.P("expert", None, None),) * num_layers),
                jax.P("expert", None, None),
            ),
        )
        for value in jax.tree.leaves(signatures):
            self.assertEqual(value.sharding.spec, jax.P())

    def test_profile_step_chains_routed_layers_through_residual(self):
        config = profile.make_config(8, "profiles", num_layers=2)

        def routed(config_arg, gate, wi, wo, router_bias, inputs):
            self.assertIs(config_arg, config)
            output = inputs * gate + 0 * (wi + wo + router_bias.sum())
            return (
                output,
                jnp.zeros((), dtype=jnp.float32),
                jnp.ones((config.num_experts,), dtype=jnp.int32),
            )

        self.enterContext(
            mock.patch.object(profile, "checkpointed_routed_moe_apply", routed)
        )
        result = profile.profile_step(
            config,
            (jnp.asarray(1.0), jnp.asarray(2.0)),
            (jnp.asarray(0.0), jnp.asarray(0.0)),
            (jnp.asarray(0.0), jnp.asarray(0.0)),
            jnp.asarray([2.0]),
            jnp.asarray([1.0]),
            (jnp.zeros((config.num_experts,)),) * 2,
        )

        (loss, (output, counts)), (gate_grads, _, _, input_grad) = result
        self.assertArraysEqual(output, jnp.asarray([12.0]))
        self.assertEqual(loss, 12.0 / profile.GLOBAL_TOKENS)
        self.assertArraysEqual(counts, jnp.full((config.num_experts,), 2))
        self.assertEqual(gate_grads[0], 6.0 / profile.GLOBAL_TOKENS)
        self.assertEqual(gate_grads[1], 4.0 / profile.GLOBAL_TOKENS)
        self.assertArraysEqual(input_grad, jnp.asarray([6.0 / profile.GLOBAL_TOKENS]))

    def test_routed_moe_apply_preserves_trainer_contract(self):
        config = profile.make_config(8, "profiles", implementation="xla")
        inputs = jnp.zeros((2, 3, 4), dtype=jnp.float32)
        gate = jnp.zeros((4, config.num_experts), dtype=jnp.bfloat16)
        wi = jnp.zeros((config.num_experts, 4, 8), dtype=jnp.bfloat16)
        wo = jnp.zeros((config.num_experts, 4, 4), dtype=jnp.bfloat16)
        router_bias = jnp.zeros((config.num_experts,), dtype=jnp.bfloat16)
        selection = jnp.zeros((2, 3, config.num_experts), dtype=jnp.float32)
        unbiased = jnp.ones_like(selection)
        aux_loss = jnp.asarray(3.0, dtype=jnp.float32)
        requested = jnp.arange(2 * config.num_experts, dtype=jnp.int32).reshape(
            2, config.num_experts
        )
        stats = model.RoutingStats(requested, jnp.asarray(-1, dtype=jnp.int32))
        route = self.enterContext(
            mock.patch.object(
                train, "route", return_value=(selection, unbiased, aux_loss)
            )
        )
        optimized = self.enterContext(
            mock.patch.object(
                train, "optimized_routed_ep", return_value=(inputs, stats)
            )
        )

        routed, actual_aux_loss, counts = train.routed_moe_apply(
            config, gate, wi, wo, router_bias, inputs
        )

        self.assertIs(routed, inputs)
        self.assertIs(actual_aux_loss, aux_loss)
        self.assertArraysEqual(counts, requested.sum(axis=0))
        for actual, expected in zip(
            route.call_args.args, (config, gate, router_bias, inputs), strict=True
        ):
            self.assertIs(actual, expected)
        self.assertEqual(route.call_args.kwargs, {})
        for actual, expected in zip(
            optimized.call_args.args, (inputs, selection, unbiased, wi, wo), strict=True
        ):
            self.assertIs(actual, expected)
        self.assertEqual(
            optimized.call_args.kwargs,
            {
                "capacity_factor": -1.0,
                "sequence_shards": 1,
                "implementation": "xla",
                "use_ring_of_experts": False,
                "dtype": wi.dtype,
            },
        )

    def test_parse_config(self):
        config = profile.parse_config(
            [
                "--ep",
                "16",
                "--profile-dir",
                "profiles/ep16",
                "--num-layers",
                "15",
                "--kernel-implementation",
                "xla",
            ]
        )

        self.assertEqual(config.ep, 16)
        self.assertEqual(config.profile_dir, "profiles/ep16")
        self.assertEqual(config.num_layers, 15)
        self.assertEqual(config.kernel_implementation, "xla")
        self.assertFalse(config.use_ring_of_experts)

    def test_parse_config_defaults_to_one_layer(self):
        config = profile.parse_config(["--ep", "8", "--profile-dir", "profiles/ep8"])

        self.assertEqual(config.num_layers, 1)
        self.assertIsNone(config.kernel_implementation)

    def test_parse_config_selects_ring_of_experts(self):
        config = profile.parse_config(
            ["--ep", "8", "--profile-dir", "profiles/ep8", "--use-ring-of-experts"]
        )

        self.assertTrue(config.use_ring_of_experts)

    @parameterized.parameters(([],), (["--ep", "4", "--profile-dir", "profiles"],))
    def test_parse_config_rejects_incomplete_or_unsupported_config(self, argv):
        with self.assertRaises(SystemExit):
            profile.parse_config(argv)

    def test_make_config_rejects_unsupported_ep(self):
        with self.assertRaisesRegex(ValueError, "only EP8 and EP16"):
            profile.make_config(4, "profiles")

    @parameterized.parameters((8, 0, 8), (8, 9, 8), (16, 16, 15))
    def test_make_config_rejects_unsupported_stack_depth(self, ep, num_layers, maximum):
        with self.assertRaisesRegex(
            ValueError, f"EP{ep} supports between 1 and {maximum} layers"
        ):
            profile.make_config(ep, "profiles", num_layers=num_layers)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
