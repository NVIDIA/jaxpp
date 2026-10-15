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

"""Expert-parallel MoE dispatch, MLP, and combine paths of ``maximoe.model``."""

import dataclasses
import functools
import math
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import jaxpr_util
from jax._src import test_util as jtu

from maximoe import model, test_util
from maximoe.kernels.moe_routing import xla as routing_spec
from maximoe.kernels.ragged_dot import xla as ragged_dot_spec

jax.config.parse_flags_with_absl()

EXPLICIT = jax.sharding.AxisType.Explicit


def routed_reference(
    inputs,
    top_k_weights,
    top_k_indices,
    wi,
    wo,
    *,
    capacity_factor,
    activations_in_float32,
):
    """Stock-JAX routed expert oracle with per-example capacity."""
    num_experts = wi.shape[0]
    num_experts_per_tok = top_k_indices.shape[-1]
    flat_indices = top_k_indices.reshape(top_k_indices.shape[0], -1)
    route_counts = jax.nn.one_hot(flat_indices, num_experts, dtype=jnp.int32)
    route_counts = jnp.cumsum(route_counts, axis=1, dtype=jnp.int32) - 1
    route_slots = jnp.take_along_axis(
        route_counts, flat_indices[..., None], axis=-1
    ).squeeze(-1)
    route_slots = route_slots.reshape(top_k_indices.shape)
    if capacity_factor < 0:
        route_keep = jnp.ones(top_k_indices.shape, dtype=jnp.bool_)
    else:
        routes_per_example = inputs.shape[1] * num_experts_per_tok
        capacity = int(
            max(
                math.ceil(routes_per_example / num_experts) * capacity_factor,
                capacity_factor,
            )
        )
        route_keep = route_slots < capacity

    wi_gate, wi_up = ragged_dot_spec.deinterleave_gate_up(wi)

    def expert_ffn(gate_weight, up_weight, down_weight):
        gate = jnp.einsum(
            "bsm,mh->bsh",
            jnp.asarray(inputs, jnp.bfloat16),
            jnp.asarray(gate_weight, jnp.bfloat16),
        )
        up = jnp.einsum(
            "bsm,mh->bsh",
            jnp.asarray(inputs, jnp.bfloat16),
            jnp.asarray(up_weight, jnp.bfloat16),
        )
        if activations_in_float32:
            gate = gate.astype(jnp.float32)
            up = up.astype(jnp.float32)
        intermediate = jax.nn.silu(gate) * up
        return jnp.einsum(
            "bsh,hm->bsm",
            intermediate.astype(jnp.bfloat16),
            jnp.asarray(down_weight, jnp.bfloat16),
        )

    expert_outputs = jax.vmap(expert_ffn)(wi_gate, wi_up, wo)
    expert_outputs = jnp.transpose(expert_outputs, (1, 2, 0, 3))
    selected_outputs = jnp.take_along_axis(
        expert_outputs, top_k_indices[..., None], axis=2
    )
    output_dtype = jnp.float32 if activations_in_float32 else jnp.bfloat16
    weighted = (
        selected_outputs.astype(jnp.float32)
        * top_k_weights.astype(jnp.float32)[..., None]
    )
    return (
        jnp.where(route_keep[..., None], weighted, 0).sum(axis=2).astype(output_dtype)
    )


def descriptor_counts(jaxpr):
    """Send descriptors of every ragged all-to-all in ``jaxpr``, in program order.

    The primitive takes ``operand, output, input_offsets, send_sizes,
    output_offsets, recv_sizes``.
    """
    return [
        eqn.invars[3].aval.size
        for _, eqn in jaxpr_util.all_eqns(jaxpr)
        if eqn.primitive.name == "ragged_all_to_all"
    ]


def sequential_admission(top_k_indices, num_experts):
    """Slot of every route among its sequence's routes to the same expert.

    Returns the slots in the shape of ``top_k_indices`` and the per-sequence
    expert counts.
    """
    flat = np.asarray(top_k_indices).reshape(top_k_indices.shape[0], -1)
    one_hot = np.eye(num_experts, dtype=np.int32)[flat]
    slots = np.take_along_axis(np.cumsum(one_hot, axis=1) - 1, flat[..., None], axis=-1)
    return slots.reshape(top_k_indices.shape), one_hot.sum(axis=1)


class OptimizedRoutedEpTest(test_util.MaximoeTestCase):
    def test_expert_batch_axes_includes_data_replicas(self):
        mesh = jax.sharding.AbstractMesh(
            (4, 8), ("data", "expert"), axis_types=(EXPLICIT,) * 2
        )

        self.assertEqual(model.expert_batch_axes(mesh), ("data", "expert"))

    def test_requires_expert_only_weight_sharding(self):
        mesh = jax.sharding.AbstractMesh(
            (2, 2), ("fsdp", "expert"), axis_types=(EXPLICIT,) * 2
        )
        batch_sharding = jax.NamedSharding(mesh, jax.P(("fsdp", "expert"), None, None))
        args = (
            jax.ShapeDtypeStruct((4, 128, 256), jnp.bfloat16, sharding=batch_sharding),
            jax.ShapeDtypeStruct((4, 128, 2), jnp.float32, sharding=batch_sharding),
            jax.ShapeDtypeStruct((4, 128, 2), jnp.int32, sharding=batch_sharding),
            jax.ShapeDtypeStruct(
                (2, 256, 512),
                jnp.bfloat16,
                sharding=jax.NamedSharding(mesh, jax.P("expert", "fsdp", None)),
            ),
            jax.ShapeDtypeStruct(
                (2, 256, 256),
                jnp.bfloat16,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None, "fsdp")),
            ),
        )

        def routed(inputs, weights, indices, wi, wo):
            return model.optimized_routed_ep(
                inputs,
                weights,
                indices,
                wi,
                wo,
                capacity_factor=-1.0,
                dtype=jnp.bfloat16,
                mesh=mesh,
            )

        with jax.sharding.use_abstract_mesh(mesh):
            with self.assertRaisesRegex(ValueError, "partitioned only"):
                jax.make_jaxpr(routed)(*args)

    def test_expert_metadata_exchange_implementations_match(self):
        self.skip_unless_gpus(2)

        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:2]), ("expert",), axis_types=(EXPLICIT,)
        )
        source_values = jnp.arange(2 * 4 * 3, dtype=jnp.int32).reshape(2, 4, 3)
        source_values = (
            source_values + jnp.arange(2, dtype=jnp.int32)[:, None, None] * 100
        )

        def run(mode):
            with jtu.set_env(**{model.ALL_TO_ALL_MODE_ENV: mode}):
                return jax.shard_map(
                    lambda values: model.all_to_all(
                        values[0], "expert", 0, 0, tiled=True
                    ),
                    in_specs=jax.P("expert", None, None),
                    out_specs=jax.P("expert", None),
                    mesh=mesh,
                )(source_values)

        with jax.set_mesh(mesh):
            source_values = jax.device_put(source_values, jax.P("expert", None, None))
            all_to_all_output = run("all_to_all")
            all_gather_output = run("all_gather")
            with self.assertRaisesRegex(
                ValueError, f"Unsupported {model.ALL_TO_ALL_MODE_ENV}"
            ):
                run("invalid")

        source_values = np.asarray(source_values)
        expected = np.concatenate(
            (
                source_values[0, :2],
                source_values[1, :2],
                source_values[0, 2:],
                source_values[1, 2:],
            )
        )
        self.assertArraysEqual(all_to_all_output, expected)
        self.assertArraysEqual(all_gather_output, expected)

    @parameterized.named_parameters(
        ("ep1_dropped_bf16", 1, 2, 2, 0.5, False),
        ("ep1_dropless_nonuniform_bf16", 1, 4, 2, -1.0, False),
        ("ep2_dropless_fp32", 2, 2, 2, -1.0, True),
        ("ep2_dropless_nonuniform_bf16", 2, 4, 2, -1.0, False),
        ("ep4_asymmetric_bf16", 4, 4, 2, -1.0, False),
    )
    def test_value_and_grad_match_reference(
        self, expert_axis_size, experts, top_k, capacity_factor, float32
    ):
        # No ``implementation`` is passed, so this runs the local GPU's kernels.
        self.skip_unless_gpus(expert_axis_size)
        selected_devices = jax.devices()[:expert_axis_size]

        mesh = jax.sharding.Mesh(
            np.asarray(selected_devices), ("expert",), axis_types=(EXPLICIT,)
        )
        reference_mesh = jax.sharding.Mesh(
            np.asarray(selected_devices[:1]), ("expert",), axis_types=(EXPLICIT,)
        )
        batch, sequence, model_dim, hidden = max(2, expert_axis_size), 128, 256, 256
        input_key, gate_key, up_key, down_key, cotangent_key = jax.random.split(
            jax.random.key(7 + expert_axis_size), 5
        )
        inputs = jax.random.normal(
            input_key, (batch, sequence, model_dim), dtype=jnp.bfloat16
        )
        wi_gate = jax.random.normal(
            gate_key, (experts, model_dim, hidden), dtype=jnp.bfloat16
        ) / jnp.sqrt(model_dim)
        wi_up = jax.random.normal(
            up_key, (experts, model_dim, hidden), dtype=jnp.bfloat16
        ) / jnp.sqrt(model_dim)
        wi = ragged_dot_spec.interleave_gate_up(wi_gate, wi_up, jnp.bfloat16)
        wo = jax.random.normal(
            down_key, (experts, hidden, model_dim), dtype=jnp.bfloat16
        ) / jnp.sqrt(hidden)
        cotangent = jax.random.normal(
            cotangent_key, inputs.shape, dtype=jnp.bfloat16
        ) / jnp.sqrt(sequence)

        token = jnp.arange(batch * sequence, dtype=jnp.int32).reshape(batch, sequence)
        preferred = (token // sequence) % experts
        logits = 0.01 + jax.nn.one_hot(preferred, experts)
        logits += 0.05 * jax.random.normal(jax.random.key(5), logits.shape)
        _, top_k_indices = jax.lax.top_k(logits, top_k)
        top_k_weights = jnp.take_along_axis(logits, top_k_indices, axis=-1)
        top_k_weights = 1.3 * top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)

        def loss(x, fused_weight, down_weight, route_weights, route_indices, *, ring):
            output, stats = model.optimized_routed_ep(
                x,
                route_weights,
                route_indices,
                fused_weight,
                down_weight,
                capacity_factor=capacity_factor,
                activations_in_float32=float32,
                dtype=jnp.bfloat16,
                use_ring_of_experts=ring,
                mesh=mesh,
            )
            scalar = jnp.vdot(output.astype(jnp.float32), cotangent.astype(jnp.float32))
            return scalar, (output, stats)

        def reference_loss(x, fused_weight, down_weight, route_weights, route_indices):
            output = routed_reference(
                x,
                route_weights,
                route_indices,
                fused_weight,
                down_weight,
                capacity_factor=capacity_factor,
                activations_in_float32=float32,
            )
            scalar = jnp.vdot(output.astype(jnp.float32), cotangent.astype(jnp.float32))
            return scalar, output

        compiled = jax.jit(
            jax.value_and_grad(loss, (0, 1, 2, 3), has_aux=True),
            static_argnames=("ring",),
        )
        reference_compiled = jax.jit(
            jax.value_and_grad(reference_loss, (0, 1, 2, 3), has_aux=True)
        )
        expert_partition = "expert" if expert_axis_size > 1 else None
        sharding = jax.sharding.NamedSharding(mesh, jax.P(expert_partition, None, None))
        candidate_args = tuple(
            jax.device_put(value, sharding)
            for value in (inputs, wi, wo, top_k_weights, top_k_indices)
        )
        replicated = jax.sharding.NamedSharding(reference_mesh, jax.P())
        reference_args = tuple(
            jax.device_put(value, replicated)
            for value in (inputs, wi, wo, top_k_weights, top_k_indices)
        )

        with jax.set_mesh(reference_mesh):
            reference = jax.block_until_ready(reference_compiled(*reference_args))
        variants = (False,) if expert_axis_size == 1 else (False, True)
        with jax.set_mesh(mesh):
            results = {
                ring: jax.block_until_ready(compiled(*candidate_args, ring=ring))
                for ring in variants
            }

        (_, reference_output), reference_grads = reference
        expected_requested = np.asarray(
            jax.nn.one_hot(top_k_indices, experts, dtype=jnp.int32).sum(axis=(1, 2))
        )
        expected_capacity = model.compute_drop_capacity(
            routes_per_sequence=sequence * top_k,
            num_experts=experts,
            capacity_factor=capacity_factor,
        )
        if expected_capacity is None:
            expected_capacity = -1
        for ring, result in results.items():
            (_, (output, stats)), grads = result
            self.assertArraysEqual(stats.requested_per_seq_expert, expected_requested)
            self.assertEqual(int(stats.capacity), expected_capacity)
            for name, actual, expected in zip(
                (
                    "output",
                    "input gradient",
                    "wi gradient",
                    "wo gradient",
                    "weight gradient",
                ),
                (output, *grads),
                (reference_output, *reference_grads),
                strict=True,
            ):
                self.assertAllClose(
                    actual,
                    expected,
                    rtol=5e-2,
                    atol=4e-2,
                    check_dtypes=False,
                    err_msg=f"ring={ring}: {name}",
                )

    def test_dropless_nonuniform_dispatch_is_compact_and_matches_dense_grad(self):
        indices_host = np.array(
            [
                [[0, 1], [0, 1], [0, 2], [0, 1], [0, 2]],
                [[0, 1], [0, 2], [0, 2], [0, 2], [1, 2]],
            ],
            dtype=np.int32,
        )
        sizes = np.array([9, 5, 6, 0], dtype=np.int32)
        next_row = np.cumsum(sizes) - sizes
        expected_positions = np.empty_like(indices_host)
        for index, expert in np.ndenumerate(indices_host):
            expected_positions[index] = next_row[expert]
            next_row[expert] += 1
        indices = jnp.asarray(indices_host)
        slots, _ = jax.vmap(
            functools.partial(model.calculate_expert_buffer_indices, num_experts=4)
        )(indices.reshape(2, -1))
        slots = slots.reshape(indices.shape)
        live_rows = int(sizes.sum())
        original_permute = model.moe_routing.permute

        def poison_tail(inputs, positions, packed_size, *, implementation=None):
            return (
                original_permute(
                    inputs, positions, packed_size, implementation=implementation
                )
                .at[live_rows:]
                .set(jnp.nan)
            )

        self.enterContext(mock.patch.object(model.moe_routing, "permute", poison_tail))
        rng = np.random.default_rng(327)
        inputs = jnp.asarray(rng.normal(0, 0.2, (2, 5, 4)), jnp.float32)
        wi = ragged_dot_spec.interleave_gate_up(
            jnp.asarray(rng.normal(0, 0.2, (4, 4, 32)), jnp.float32),
            jnp.asarray(rng.normal(0, 0.2, (4, 4, 32)), jnp.float32),
            jnp.float32,
        )
        wo = jnp.asarray(rng.normal(0, 0.2, (4, 32, 4)), jnp.float32)
        weights = jnp.asarray(rng.normal(0, 0.2, indices.shape), jnp.float32)
        cotangent = jnp.asarray(rng.normal(size=inputs.shape), jnp.float32)
        packed_size = model.storage_packed_size_per_batch(
            sequence_length=5,
            num_experts_per_tok=2,
            num_target_experts=4,
            drop_capacity=None,
        )
        packed, positions = model.local_ep_exchange_packed(
            inputs,
            indices,
            slots,
            jnp.ones(indices.shape, dtype=jnp.bool_),
            num_experts=4,
            packed_size=packed_size,
        )
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(packed)),
            ("values", "sizes"),
        )
        self.assertEqual(packed.values.shape, (512, 4))
        self.assertArraysEqual(packed.sizes, sizes)
        self.assertArraysEqual(positions, expected_positions)
        self.assertArraysEqual(
            packed.values[:live_rows],
            np.repeat(np.asarray(inputs).reshape(-1, 4), 2, axis=0)[
                np.argsort(expected_positions.reshape(-1))
            ],
        )
        self.assertTrue(np.isnan(np.asarray(packed.values[live_rows:])).all())
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:1]), ("expert",), axis_types=(EXPLICIT,)
        )

        def actual(x, fused_wi, down, route_weights):
            output, stats = model.optimized_routed_ep(
                x,
                route_weights,
                indices,
                fused_wi,
                down,
                capacity_factor=-1.0,
                activations_in_float32=True,
                dtype=jnp.float32,
                mesh=mesh,
                implementation="xla",
            )
            return jnp.vdot(output, cotangent), (output, stats)

        def reference(x, fused_wi, down, route_weights):
            gate, up = ragged_dot_spec.deinterleave_gate_up(fused_wi)
            gate_out = jnp.einsum(
                "bsm,bskmh->bskh", x, gate[indices], precision=jax.lax.Precision.HIGHEST
            )
            up_out = jnp.einsum(
                "bsm,bskmh->bskh", x, up[indices], precision=jax.lax.Precision.HIGHEST
            )
            routed = jnp.einsum(
                "bskh,bskhm->bskm",
                jax.nn.silu(gate_out) * up_out,
                down[indices],
                precision=jax.lax.Precision.HIGHEST,
            )
            output = jnp.sum(routed * route_weights[..., None], axis=2)
            return jnp.vdot(output, cotangent), output

        with jax.set_mesh(mesh):
            (_, (output, stats)), grads = jax.jit(
                jax.value_and_grad(actual, (0, 1, 2, 3), has_aux=True)
            )(inputs, wi, wo, weights)
            (_, expected_output), expected_grads = jax.jit(
                jax.value_and_grad(reference, (0, 1, 2, 3), has_aux=True)
            )(inputs, wi, wo, weights)

        self.assertEqual(int(stats.capacity), -1)
        self.assertArraysEqual(
            stats.requested_per_seq_expert,
            np.array([[5, 3, 2, 0], [4, 2, 4, 0]], np.int32),
        )
        for got, want in zip(
            (output, *grads), (expected_output, *expected_grads), strict=True
        ):
            self.assertTrue(np.isfinite(np.asarray(got)).all())
            self.assertAllClose(got, want, rtol=1e-5, atol=1e-6)
        self.assertArraysEqual(grads[1][3], 0, check_dtypes=False)
        self.assertArraysEqual(grads[2][3], 0, check_dtypes=False)

    def test_nonring_remat_reuses_forward_work(self):
        self.skip_unless_gpus(2)

        def marked_down_projection(
            lhs, rhs, group_sizes, cost_estimate_num_rows=None, *, output_dtype=None
        ):
            del group_sizes, cost_estimate_num_rows, output_dtype
            return jnp.sin(jnp.zeros((lhs.shape[0], rhs.shape[-1]), lhs.dtype))

        self.enterContext(
            mock.patch.object(
                model.ragged_dot_ops.get_implementation(),
                "ragged_dot",
                marked_down_projection,
            )
        )
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:2]), ("expert",), axis_types=(EXPLICIT,)
        )
        spec = jax.P("expert", None, None)
        args = (
            jax.ShapeDtypeStruct((2, 128, 256), jnp.bfloat16, sharding=spec),
            jax.ShapeDtypeStruct((2, 256, 256), jnp.bfloat16, sharding=spec),
            jax.ShapeDtypeStruct((2, 256, 256), jnp.bfloat16, sharding=spec),
            jax.ShapeDtypeStruct((2, 256, 256), jnp.bfloat16, sharding=spec),
            jax.ShapeDtypeStruct((2, 128, 2), jnp.float32, sharding=spec),
        )

        def loss(inputs, wi_0, wi_1, wo, logits):
            def body(inputs, wi_0, wi_1, wo, logits):
                _, top_k_indices = jax.lax.top_k(logits, 2)
                top_k_weights = jnp.take_along_axis(logits, top_k_indices, axis=-1)
                top_k_weights = (
                    1.3 * top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
                )
                output, _ = model.optimized_routed_ep(
                    inputs,
                    top_k_weights,
                    top_k_indices,
                    ragged_dot_spec.interleave_gate_up(wi_0, wi_1, jnp.bfloat16),
                    wo,
                    capacity_factor=-1.0,
                    dtype=jnp.bfloat16,
                    use_ring_of_experts=False,
                    mesh=mesh,
                )
                return output

            return (
                jax.checkpoint(body)(inputs, wi_0, wi_1, wo, logits)
                .astype(jnp.float32)
                .sum()
            )

        with jax.set_mesh(mesh):
            jaxpr = jax.make_jaxpr(jax.value_and_grad(loss, (0, 1, 2, 3)))(*args)

        eqns = [eqn for _, eqn in jaxpr_util.all_eqns(jaxpr)]
        self.assertEqual(sum(eqn.primitive.name == "sin" for eqn in eqns), 1)
        offset_exchanges = [
            eqn
            for eqn in eqns
            if eqn.primitive.name == "all_to_all"
            and eqn.invars[0].aval.dtype == jnp.int32
            and eqn.invars[0].aval.ndim == 1
        ]
        self.assertEmpty(offset_exchanges)


class ContextParallelAttentionTest(test_util.MaximoeTestCase):
    """MLA attention with latent gathers matches unsharded attention."""

    @parameterized.named_parameters(("cp2", 2), ("cp4", 4))
    def test_value_and_grad_match_unsharded_attention(self, cp):
        self.skip_unless_gpus(cp)
        sequence, emb, heads, q_rank, kv_rank = 512, 128, 8, 64, 32
        nope, rope_dim, v_dim = 32, 16, 32
        rng = np.random.default_rng(3)

        def normal(*shape, scale=1.0):
            return jnp.asarray(rng.normal(scale=scale, size=shape), jnp.bfloat16)

        params = model.AttentionLayerParams(
            pre_attn_norm_scale=jnp.ones((emb,), jnp.bfloat16),
            wq_a=normal(emb, q_rank, scale=emb**-0.5),
            q_norm_scale=normal(q_rank, scale=0.1) + 1,
            wq_b=normal(q_rank, heads, nope + rope_dim, scale=q_rank**-0.5),
            wkv_a=normal(emb, kv_rank + rope_dim, scale=emb**-0.5),
            kv_norm_scale=normal(kv_rank, scale=0.1) + 1,
            wkv_b=normal(kv_rank, heads, nope + v_dim, scale=kv_rank**-0.5),
            out_kernel=normal(heads, v_dim, emb, scale=(heads * v_dim) ** -0.5),
            post_attn_norm_scale=jnp.ones((emb,), jnp.bfloat16),
        )
        inputs = normal(1, sequence, emb)
        positions = jnp.arange(sequence, dtype=jnp.int32)[None]
        # Padding exercises the token count summed over context shards.
        segments = (positions < sequence - 40).astype(jnp.int32)
        cotangent = normal(1, sequence, emb)
        rope = model.YarnRope.mk(
            embedding_dims=rope_dim,
            rope_theta=10000.0,
            original_max_position_embeddings=128,
            beta_fast=32.0,
            beta_slow=1.0,
            rope_factor=4.0,
        )

        def attention(x, pos, seg, weights, yarn, axis):
            return model.self_attention(
                x,
                pos,
                seg,
                weights,
                epsilon=1e-6,
                kv_lora_rank=kv_rank,
                qk_nope_head_dim=nope,
                softmax_scale=(nope + rope_dim) ** -0.5,
                dtype=jnp.bfloat16,
                yarn_rope=yarn,
                context_parallel_axis=axis,
            )

        def loss(apply, x, weights):
            out = apply(x, weights)
            return jnp.sum(out.astype(jnp.float32) * cotangent), out

        reference = jax.jit(
            jax.value_and_grad(
                functools.partial(
                    loss, lambda x, w: attention(x, positions, segments, w, rope, None)
                ),
                (0, 1),
                has_aux=True,
            )
        )
        (_, expected_out), expected_grads = reference(inputs, params)

        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:cp]), ("expert",), axis_types=(EXPLICIT,)
        )
        sequence_sharded = jax.sharding.NamedSharding(mesh, jax.P(None, "expert"))
        activation = jax.sharding.NamedSharding(mesh, jax.P(None, "expert", None))
        heads_sharded = jax.sharding.NamedSharding(mesh, jax.P(None, "expert", None))
        replicated = jax.sharding.NamedSharding(mesh, jax.P())
        weight_shardings = jax.tree.map(lambda _: replicated, params)._replace(
            wq_b=heads_sharded, wkv_b=heads_sharded
        )

        def sharded(x, weights):
            return jax.shard_map(
                functools.partial(attention, axis="expert"),
                out_specs=jax.P(None, "expert", None),
            )(x, positions, segments, weights, rope)

        with jax.set_mesh(mesh):
            positions = jax.device_put(positions, sequence_sharded)
            segments = jax.device_put(segments, sequence_sharded)
            cotangent = jax.device_put(cotangent, activation)
            rope = jax.device_put(rope, replicated)
            (_, out), grads = jax.jit(
                jax.value_and_grad(
                    functools.partial(loss, sharded), (0, 1), has_aux=True
                )
            )(
                jax.device_put(inputs, activation),
                jax.device_put(params, weight_shardings),
            )
            hlo = (
                jax.jit(jax.grad(lambda x, w: loss(sharded, x, w)[0], (0, 1)))
                .lower(
                    jax.device_put(inputs, activation),
                    jax.device_put(params, weight_shardings),
                )
                .as_text()
            )

        self.assertClose(out, expected_out)
        for name, got, want in zip(
            ("inputs", *model.AttentionLayerParams._fields),
            jax.tree.leaves(grads),
            jax.tree.leaves(expected_grads),
            strict=True,
        ):
            with self.subTest(name):
                self.assertClose(got, want, atol_scale=3e-2)
        # Only the attention output crosses the context axis as a dense
        # sequence/head exchange, once forward and once for its gradient.
        self.assertEqual(hlo.count("stablehlo.all_to_all"), 2)


class ContextShardingTest(test_util.MaximoeTestCase):
    """EP-as-context: chunk layouts and capacity pooled over a sequence's shards."""

    @parameterized.parameters(1, 2, 4)
    def test_admission_follows_original_token_order(self, sequence_shards):
        # Sequences are independent: overflowing one cannot consume another's slots.
        indices = np.random.default_rng(3).integers(0, 8, (4, 32, 2), dtype=np.int32)
        expected_slots, expected_counts = sequential_admission(indices, 8)
        chunks = model.context_to_expert_batch(
            jnp.asarray(indices), sequence_shards=sequence_shards
        )

        slots, counts = jax.jit(
            functools.partial(
                model.sequence_expert_buffer_indices,
                num_experts=8,
                sequence_shards=sequence_shards,
            )
        )(chunks)

        self.assertArraysEqual(
            model.expert_batch_to_context(slots, sequence_shards=sequence_shards),
            expected_slots,
        )
        self.assertArraysEqual(counts, expected_counts)

    def test_pooled_capacity_preserves_chunk_receive_budget(self):
        # 16k tokens, EP8, top-8, 256 experts: 1024 assignments per expert. Eight
        # chunks at capacity 128 allocate as much as one sequence at 1024.
        per_chunk = model.storage_packed_size_per_batch(
            sequence_length=2048,
            num_experts_per_tok=8,
            num_target_experts=32,
            drop_capacity=128,
        )
        pooled = model.storage_packed_size_per_batch(
            sequence_length=2048 * 8,
            num_experts_per_tok=8,
            num_target_experts=32,
            drop_capacity=1024,
        )

        self.assertEqual(pooled, 8 * per_chunk)
        self.assertEqual(pooled, 32768)

    @parameterized.named_parameters(("ep4", 1, 4, 2), ("data2_ep2", 2, 2, 4))
    def test_layouts_keep_local_tokens_without_collectives(self, data, ep, batch):
        if jax.device_count() < data * ep:
            self.skipTest(f"requires {data * ep} devices")
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[: data * ep]).reshape(data, ep),
            ("data", "expert"),
            axis_types=(EXPLICIT,) * 2,
        )
        values = np.arange(batch * 128 * 4, dtype=np.float32).reshape(batch, 128, 4)
        collective = (
            r"\b(all-gather|all-to-all|collective-permute|reduce-scatter|all-reduce)"
            r"(-start|-done)?\("
        )

        with jax.set_mesh(mesh):
            tokens = jax.device_put(
                values, jax.P("data" if data > 1 else None, "expert", None)
            )
            split = (
                jax.jit(
                    functools.partial(model.context_to_expert_batch, sequence_shards=ep)
                )
                .lower(tokens)
                .compile()
            )
            chunks = split(tokens)
            join = (
                jax.jit(
                    functools.partial(model.expert_batch_to_context, sequence_shards=ep)
                )
                .lower(chunks)
                .compile()
            )
            restored = join(chunks)

        for executable in (split, join):
            self.assertNotRegex(executable.as_text(), collective)
        tokens_by_device = {
            shard.device: shard.data for shard in tokens.addressable_shards
        }
        for shard in chunks.addressable_shards:
            self.assertArraysEqual(shard.data, tokens_by_device[shard.device])
        self.assertArraysEqual(restored, values)

    @parameterized.named_parameters(
        ("ep2", 2, 1, 1, 1.0, False, 1),
        ("ep2_ring", 2, 1, 1, 1.0, True, 1),
        ("data2_ep2_half_capacity", 2, 2, 4, 0.5, False, 1),
        ("ep4_overflow", 4, 1, 2, 1.2, False, 1),
        ("ep2_zero_capacity", 2, 1, 2, 0.0, False, 1),
        ("ep2_ring_dropless", 2, 1, 2, -1.0, True, 1),
        ("data2_ep2_top2", 2, 2, 4, 1.0, False, 2),
        ("data2_ep2_ring_half_capacity_top2", 2, 2, 4, 0.5, True, 2),
    )
    def test_value_and_grad_match_whole_sequence_reference(
        self, ep, data, batch, capacity_factor, ring, top_k
    ):
        self.skip_unless_gpus(ep * data)
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[: ep * data]).reshape(data, ep),
            ("data", "expert"),
            axis_types=(EXPLICIT,) * 2,
        )
        reference_mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:1]), ("expert",), axis_types=(EXPLICIT,)
        )
        sequence, width, hidden, experts = 256, 256, 256, ep * top_k
        rng = np.random.default_rng(19)
        inputs = jnp.asarray(rng.normal(size=(batch, sequence, width)), jnp.bfloat16)
        wi = jnp.asarray(
            rng.normal(size=(experts, width, 2 * hidden)) / math.sqrt(width),
            jnp.bfloat16,
        )
        wo = jnp.asarray(
            rng.normal(size=(experts, hidden, width)) / math.sqrt(hidden), jnp.bfloat16
        )
        weights = jnp.asarray(
            rng.uniform(0.5, 1.5, (batch, sequence, top_k)), jnp.float32
        )
        # Even sequences route each context chunk to its own experts, which fits
        # a whole-sequence budget. Odd sequences overload the first experts past
        # a chunk boundary, so admission spans shards yet stays per sequence.
        chunk = np.arange(sequence) // (sequence // ep)
        indices = np.broadcast_to(
            chunk[None, :, None] * top_k + np.arange(top_k), (batch, sequence, top_k)
        ).astype(np.int32)
        indices[1::2, : 3 * sequence // 4] = np.arange(top_k)
        cotangent = np.asarray(
            rng.normal(size=inputs.shape) / math.sqrt(sequence), np.float32
        )

        def reference_loss(x, wi, wo, weights, indices):
            output = routed_reference(
                x,
                weights,
                indices,
                wi,
                wo,
                capacity_factor=capacity_factor,
                activations_in_float32=False,
            )
            return jnp.sum(output.astype(jnp.float32) * cotangent), output

        def loss(x, wi, wo, weights, indices):
            split = functools.partial(model.context_to_expert_batch, sequence_shards=ep)
            output, stats = model.optimized_routed_ep(
                split(x),
                split(weights),
                split(indices),
                wi,
                wo,
                capacity_factor=capacity_factor,
                sequence_shards=ep,
                dtype=jnp.bfloat16,
                use_ring_of_experts=ring,
            )
            output = model.expert_batch_to_context(output, sequence_shards=ep)
            return jnp.sum(output.astype(jnp.float32) * cotangent), (output, stats)

        args = (inputs, wi, wo, weights, indices)
        with jax.set_mesh(reference_mesh):
            replicated = jax.sharding.NamedSharding(reference_mesh, jax.P())
            (_, expected_output), expected_grads = jax.block_until_ready(
                jax.jit(jax.value_and_grad(reference_loss, (0, 1, 2, 3), has_aux=True))(
                    *(jax.device_put(value, replicated) for value in args)
                )
            )
        activation_sharding = jax.sharding.NamedSharding(
            mesh, jax.P("data" if data > 1 else None, "expert", None)
        )
        weight_sharding = jax.sharding.NamedSharding(
            mesh, jax.P("expert", None, None, reduced={"data"} if data > 1 else set())
        )
        shardings = (
            activation_sharding,
            weight_sharding,
            weight_sharding,
            activation_sharding,
            activation_sharding,
        )
        with jax.set_mesh(mesh):
            (_, (output, stats)), grads = jax.block_until_ready(
                jax.jit(jax.value_and_grad(loss, (0, 1, 2, 3), has_aux=True))(
                    *map(jax.device_put, args, shardings)
                )
            )
            # The trainer reduces data-local weight gradients once after
            # accumulation; materialize that reduction before comparing.
            grads = jax.tree.map(
                lambda grad: jax.reshard(
                    grad,
                    grad.sharding.update(
                        spec=grad.sharding.spec.update(
                            unreduced=frozenset(), unreduced_kind=None
                        )
                    ),
                ),
                grads,
            )

        _, expected_counts = sequential_admission(indices, experts)
        self.assertArraysEqual(stats.requested_per_seq_expert, expected_counts)
        expected_capacity = model.compute_drop_capacity(
            routes_per_sequence=sequence * top_k,
            num_experts=experts,
            capacity_factor=capacity_factor,
        )
        if expected_capacity is None:
            expected_capacity = -1
        self.assertEqual(int(stats.capacity), expected_capacity)
        for name, actual, expected in zip(
            (
                "output",
                "input gradient",
                "wi gradient",
                "wo gradient",
                "weight gradient",
            ),
            (output, *grads),
            (expected_output, *expected_grads),
            strict=True,
        ):
            self.assertAllClose(
                actual, expected, rtol=5e-2, atol=4e-2, check_dtypes=False, err_msg=name
            )


class RaggedDotCombineTest(test_util.MaximoeTestCase):
    """``ragged_dot_combine`` on the ``jax.numpy`` specification of every kernel."""

    @parameterized.product(remat=(False, True), argnums=((0,), (1,), (0, 1)))
    def test_eager_weight_gradient_with_fixed_weights(self, remat, argnums):
        group_sizes = jnp.array([2, 3], dtype=jnp.int32)
        positions = jnp.array([[[0, 2], [1, 3], [8, 4], [8, 8]]], dtype=jnp.int32)
        intermediate = jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3) / 11
        wo = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 13
        weights = jnp.array(
            [[[0.25, 0.5], [0.0, -0.75], [7.0, 1e-8], [-2.0, 3.0]]], dtype=jnp.float32
        )

        def actual(h, route_weights):
            return model.ragged_dot_ops.ragged_dot_combine(
                h,
                wo,
                group_sizes,
                route_weights,
                positions,
                None,
                False,
                5,
                implementation="xla",
                eager_weight_gradient=True,
            ).sum()

        def reference(h, route_weights):
            expert_output = ragged_dot_spec.ragged_dot(h, wo, group_sizes, 5)
            return routing_spec.combine(
                expert_output, route_weights, positions, False
            ).sum()

        if remat:
            actual = jax.checkpoint(actual)
        actual_value, actual_grads = jax.jit(jax.value_and_grad(actual, argnums))(
            intermediate, weights
        )
        reference_value, reference_grads = jax.jit(
            jax.value_and_grad(reference, argnums)
        )(intermediate, weights)
        self.assertAllClose(actual_value, reference_value, rtol=1e-6, atol=1e-6)
        for actual_grad, reference_grad in zip(
            actual_grads, reference_grads, strict=True
        ):
            self.assertAllClose(actual_grad, reference_grad, rtol=1e-6, atol=1e-6)

    def mark_down_projection(self):
        """Wrap the down projection in ``sin`` so recomputations are countable."""

        original = ragged_dot_spec.ragged_dot

        def marked(
            lhs, rhs, group_sizes, cost_estimate_num_rows=None, *, output_dtype=None
        ):
            output = original(
                lhs, rhs, group_sizes, cost_estimate_num_rows, output_dtype=output_dtype
            )
            return jnp.sin(output)

        self.enterContext(mock.patch.object(ragged_dot_spec, "ragged_dot", marked))

    def test_compact_matches_composed_vjp(self):
        group_sizes = jnp.array([2, 3], dtype=jnp.int32)
        positions = jnp.array([[[0, 2], [1, 3], [8, 4], [8, 8]]], dtype=jnp.int32)
        weights = jnp.array(
            [[[0.25, 0.5], [0.0, -0.75], [jnp.nan, 1e-8], [jnp.nan, jnp.nan]]],
            dtype=jnp.float32,
        )
        intermediate = jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3) / 11
        intermediate = intermediate.at[jnp.array([5, 6, 7])].set(jnp.nan)
        wo = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 13
        grad_out = jnp.arange(4 * 4, dtype=jnp.float32).reshape(1, 4, 4) / 17
        grad_out = grad_out.at[0, 3].set(jnp.nan)

        def actual(h, down, route_weights):
            return model.ragged_dot_ops.ragged_dot_combine(
                h,
                down,
                group_sizes,
                route_weights,
                positions,
                jnp.array([3], dtype=jnp.int32),
                False,
                5,
                implementation="xla",
            )

        def reference(h, down, route_weights):
            expert_output = ragged_dot_spec.ragged_dot(h, down, group_sizes, 5)
            dense = routing_spec.combine(expert_output, route_weights, positions, False)
            return dense.at[:, 3:].set(0)

        actual_output, actual_vjp = jax.vjp(actual, intermediate, wo, weights)
        reference_output, reference_vjp = jax.vjp(reference, intermediate, wo, weights)
        actual_grads = actual_vjp(grad_out)
        reference_grads = reference_vjp(grad_out)

        self.assertAllClose(actual_output, reference_output, rtol=1e-6, atol=1e-6)
        for actual_grad, reference_grad in zip(
            actual_grads, reference_grads, strict=True
        ):
            self.assertTrue(np.isfinite(np.asarray(actual_grad)).all())
            self.assertAllClose(actual_grad, reference_grad, rtol=1e-6, atol=1e-6)
        self.assertArraysEqual(
            actual_grads[0][jnp.array([5, 6, 7])], 0, check_dtypes=False
        )
        self.assertArraysEqual(actual_grads[2][positions >= 8], 0, check_dtypes=False)

    def test_compact_empty_routes_are_zero(self):
        group_sizes = jnp.zeros((2,), dtype=jnp.int32)
        positions = jnp.full((1, 2, 2), 4, dtype=jnp.int32)
        intermediate = jnp.full((4, 2), jnp.nan, dtype=jnp.float32)
        wo = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
        weights = jnp.full(positions.shape, jnp.nan, dtype=jnp.float32)

        def routed(h, down, route_weights):
            return model.ragged_dot_ops.ragged_dot_combine(
                h,
                down,
                group_sizes,
                route_weights,
                positions,
                jnp.array([0], dtype=jnp.int32),
                False,
                0,
                implementation="xla",
            )

        output, vjp = jax.vjp(routed, intermediate, wo, weights)
        grads = vjp(jnp.ones_like(output))

        self.assertArraysEqual(output, 0, check_dtypes=False)
        for grad in grads:
            self.assertArraysEqual(grad, 0, check_dtypes=False)

    def test_compact_outer_remat_keeps_one_down_projection(self):
        self.mark_down_projection()
        group_sizes = jnp.array([2, 1], dtype=jnp.int32)
        positions = jnp.array([[[0, 2], [1, 4]]], dtype=jnp.int32)
        intermediate = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 5
        wo = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
        weights = jnp.array([[[0.25, 0.5], [0.75, jnp.nan]]], dtype=jnp.float32)

        def loss(h, down, route_weights):
            output = jax.checkpoint(
                lambda x, w, rw: model.ragged_dot_ops.ragged_dot_combine(
                    x,
                    w,
                    group_sizes,
                    rw,
                    positions,
                    jnp.array([2], dtype=jnp.int32),
                    False,
                    3,
                    implementation="xla",
                )
            )(h, down, route_weights)
            return output.sum()

        jaxpr = jax.make_jaxpr(jax.value_and_grad(loss, (0, 1, 2)))(
            intermediate, wo, weights
        )

        self.assertEqual(
            sum(
                eqn.primitive is jax.lax.sin_p for _, eqn in jaxpr_util.all_eqns(jaxpr)
            ),
            1,
        )

    def test_matches_composed_vjp(self):
        group_sizes = jnp.array([2, 3], dtype=jnp.int32)
        positions = jnp.array([[[0, 2], [1, 3], [8, 4], [8, 8]]], dtype=jnp.int32)
        weights = jnp.array(
            [[[0.25, 0.5], [0.0, -0.75], [7.0, 1e-8], [-2.0, 3.0]]], dtype=jnp.float32
        )
        intermediate = jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3) / 11
        wo = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 13
        grad_out = jnp.arange(4 * 4, dtype=jnp.float32).reshape(1, 4, 4) / 17

        def actual(h, down, route_weights):
            return model.ragged_dot_ops.ragged_dot_combine(
                h,
                down,
                group_sizes,
                route_weights,
                positions,
                None,
                False,
                5,
                implementation="xla",
            )

        def reference(h, down, route_weights):
            expert_output = ragged_dot_spec.ragged_dot(h, down, group_sizes, 5)
            return routing_spec.combine(expert_output, route_weights, positions, False)

        actual_output, actual_vjp = jax.vjp(actual, intermediate, wo, weights)
        reference_output, reference_vjp = jax.vjp(reference, intermediate, wo, weights)
        actual_grads = actual_vjp(grad_out)
        reference_grads = reference_vjp(grad_out)

        self.assertAllClose(actual_output, reference_output, rtol=1e-6, atol=1e-6)
        for actual_grad, reference_grad in zip(
            actual_grads, reference_grads, strict=True
        ):
            self.assertAllClose(actual_grad, reference_grad, rtol=1e-6, atol=1e-6)

        self.assertEqual(actual_grads[0][1].tolist(), [0.0, 0.0, 0.0])
        self.assertNotEqual(actual_grads[2][0, 1, 0], 0)
        self.assertArraysEqual(actual_grads[2][positions >= 8], 0, check_dtypes=False)

    def test_remat_does_not_repeat_down_projection(self):
        self.mark_down_projection()
        group_sizes = jnp.array([2, 1], dtype=jnp.int32)
        positions = jnp.array([[[0, 2], [1, 4]]], dtype=jnp.int32)
        intermediate = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 5
        wo = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
        weights = jnp.array([[[0.25, 0.5], [0.75, 3.0]]], dtype=jnp.float32)

        def loss(h, down, route_weights):
            output = jax.checkpoint(
                lambda x, w, rw: model.ragged_dot_ops.ragged_dot_combine(
                    x,
                    w,
                    group_sizes,
                    rw,
                    positions,
                    None,
                    False,
                    3,
                    implementation="xla",
                )
            )(h, down, route_weights)
            return output.sum()

        jaxpr = jax.make_jaxpr(jax.value_and_grad(loss, (0, 1, 2)))(
            intermediate, wo, weights
        )

        self.assertEqual(
            sum(
                eqn.primitive is jax.lax.sin_p for _, eqn in jaxpr_util.all_eqns(jaxpr)
            ),
            1,
        )


class RaggedDotCombineMosaicGpuTest(test_util.MaximoeTestCase):
    """``ragged_dot_combine`` composed from the Hopper or Blackwell kernels."""

    def setUp(self):
        super().setUp()
        self.skip_unless_gpus()

    def test_compact_forward_has_no_separate_permute(self):
        def down_projection(
            lhs, rhs, group_sizes, cost_estimate_num_rows=None, *, output_dtype=None
        ):
            del group_sizes, cost_estimate_num_rows, output_dtype
            return jnp.zeros((lhs.shape[0], rhs.shape[-1]), lhs.dtype)

        self.enterContext(
            mock.patch.object(
                model.ragged_dot_ops.get_implementation(), "ragged_dot", down_projection
            )
        )
        traced = jax.jit(
            lambda intermediate, wo, sizes, weights, positions: (
                model.ragged_dot_ops.ragged_dot_combine(
                    intermediate,
                    wo,
                    sizes,
                    weights,
                    positions,
                    jnp.array([4], dtype=jnp.int32),
                    False,
                    4,
                )
            )
        ).trace(
            jnp.zeros((8, 16), jnp.bfloat16),
            jnp.zeros((2, 16, 32), jnp.bfloat16),
            jnp.array([4, 4], jnp.int32),
            jnp.ones((1, 4, 2), jnp.float32),
            jnp.zeros((1, 4, 2), jnp.int32),
        )

        kernel_names = test_util.kernel_names(traced.lojax.jaxpr)
        self.assertEqual(kernel_names.count("moe_combine_mosaic_gpu_fwd"), 1)
        self.assertNotIn("permute_buffer_fwd", kernel_names)

    @parameterized.parameters(False, True)
    def test_ragged_mlp_padding_is_unobservable(self, activations_in_float32):
        group_sizes = jnp.array([9, 7, 0], dtype=jnp.int32)
        live_positions = jnp.arange(16, dtype=jnp.int32)
        positions = live_positions.reshape(8, 2)
        padding_rows = jnp.arange(16, 256, dtype=jnp.int32)

        keys = jax.random.split(jax.random.key(29), 6)
        tokens = jax.random.normal(keys[0], (8, 256), dtype=jnp.bfloat16) / 8
        gate = jax.random.normal(keys[1], (3, 256, 256), dtype=jnp.bfloat16) / 16
        up = jax.random.normal(keys[2], (3, 256, 256), dtype=jnp.bfloat16) / 16
        wi = ragged_dot_spec.interleave_gate_up(gate, up, jnp.bfloat16)
        wo = jax.random.normal(keys[3], (3, 256, 256), dtype=jnp.bfloat16) / 16
        weights = jax.random.normal(keys[4], (1, 8, 2), dtype=jnp.float32) / 4
        cotangent = jax.random.normal(keys[5], (1, 8, 256), dtype=jnp.float32) / 8

        def loss(token_values, fused_wi, down_wo, route_weights, padding_value):
            packed = model.moe_routing.permute(token_values, positions, 256)
            packed = packed.at[padding_rows].set(padding_value)
            intermediate = model.ffn_ragged_intermediate(
                fused_wi,
                model.RaggedArray(packed, group_sizes),
                dtype=jnp.bfloat16,
                activations_in_float32=activations_in_float32,
                cost_estimate_num_rows=16,
            )
            output = model.ragged_dot_ops.ragged_dot_combine(
                intermediate.astype(jnp.bfloat16),
                down_wo,
                group_sizes,
                route_weights,
                positions[None],
                None,
                activations_in_float32,
                16,
            )
            return jnp.vdot(output.astype(jnp.float32), cotangent), output

        run = jax.jit(jax.value_and_grad(loss, (0, 1, 2, 3), has_aux=True))
        zero = run(tokens, wi, wo, weights, jnp.asarray(0, dtype=jnp.bfloat16))
        poisoned = run(
            tokens, wi, wo, weights, jnp.asarray(jnp.nan, dtype=jnp.bfloat16)
        )

        for expected, actual in zip(
            jax.tree.leaves(zero), jax.tree.leaves(poisoned), strict=True
        ):
            self.assertTrue(np.isfinite(np.asarray(actual)).all())
            self.assertArraysEqual(actual, expected)

    @parameterized.product(activations_in_float32=[False, True])
    def test_matches_composed_vjp(self, activations_in_float32):
        group_sizes = jnp.array([65, 95, 0], dtype=jnp.int32)
        live_positions = jnp.arange(160, dtype=jnp.int32)
        positions = jnp.pad(
            live_positions, (0, 256 - live_positions.size), constant_values=256
        )
        positions = positions.reshape(1, 128, 2)

        h_key, wo_key, weight_key, cotangent_key = jax.random.split(
            jax.random.key(13), 4
        )
        intermediate = jax.random.normal(h_key, (256, 256), dtype=jnp.bfloat16) / 8
        padding_rows = jnp.arange(160, 256, dtype=jnp.int32)
        intermediate = intermediate.at[padding_rows].set(jnp.nan)
        wo = jax.random.normal(wo_key, (3, 256, 256), dtype=jnp.bfloat16) / 8
        weights = jax.random.normal(weight_key, positions.shape, dtype=jnp.float32)
        weights = weights.reshape(-1).at[17].set(0).at[160:].set(jnp.nan)
        weights = weights.reshape(positions.shape)
        cotangent = jax.random.normal(cotangent_key, (1, 128, 256), dtype=jnp.bfloat16)

        def loss(h, down, route_weights, *, fused):
            if fused:
                output = model.ragged_dot_ops.ragged_dot_combine(
                    h,
                    down,
                    group_sizes,
                    route_weights,
                    positions,
                    None,
                    activations_in_float32,
                    160,
                )
            else:
                # The Mosaic ``combine`` has no VJP, so differentiate the XLA spec of
                # the combine around the same Mosaic projection.
                expert_output = model.ragged_dot_ops.ragged_dot(
                    h, down, group_sizes, 160
                )
                output = model.moe_routing.combine(
                    expert_output,
                    route_weights,
                    positions,
                    activations_in_float32,
                    implementation="xla",
                )
            loss = jnp.vdot(output.astype(jnp.float32), cotangent.astype(jnp.float32))
            return loss, output

        run = jax.jit(
            jax.value_and_grad(loss, (0, 1, 2), has_aux=True),
            static_argnames=("fused",),
        )
        (actual_loss, actual_output), actual_grads = jax.block_until_ready(
            run(intermediate, wo, weights, fused=True)
        )
        (_, expected_output), expected_grads = jax.block_until_ready(
            run(intermediate, wo, weights, fused=False)
        )
        self.assertTrue(np.isfinite(np.asarray(actual_loss)))

        def assert_relative_l2(actual_value, expected_value, name):
            actual_value = np.asarray(actual_value, dtype=np.float32)
            expected_value = np.asarray(expected_value, dtype=np.float32)
            relative_l2 = np.linalg.norm(
                actual_value - expected_value
            ) / np.linalg.norm(expected_value)
            self.assertLess(relative_l2, 1e-2, f"{name} relative L2 error")

        # Compare tensors rather than the vdot: XLA evaluates that float32 reduction
        # with different precision in the two programs.
        assert_relative_l2(actual_output, expected_output, "output")
        live_rows = np.arange(160)
        assert_relative_l2(
            actual_grads[0][live_rows], expected_grads[0][live_rows], "dH"
        )
        # The reference weights the cotangent in the XLA spec, so gradients agree
        # only to rounding.
        assert_relative_l2(actual_grads[1], expected_grads[1], "dW")
        assert_relative_l2(actual_grads[2], expected_grads[2], "dweights")

        self.assertArraysEqual(
            np.asarray(actual_grads[2]).reshape(-1)[160:], 0, check_dtypes=False
        )
        self.assertArraysEqual(actual_grads[0][17], 0, check_dtypes=False)

    def vma_trace_args(self, mesh):
        return (
            jax.ShapeDtypeStruct(
                (512, 256),
                jnp.bfloat16,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None)),
            ),
            jax.ShapeDtypeStruct(
                (6, 256, 256),
                jnp.bfloat16,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None, None)),
            ),
            jax.ShapeDtypeStruct(
                (6,), jnp.int32, sharding=jax.NamedSharding(mesh, jax.P("expert"))
            ),
            jax.ShapeDtypeStruct(
                (2, 2, 128, 2),
                jnp.float32,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None, None, None)),
            ),
            jax.ShapeDtypeStruct(
                (2, 2, 128, 2),
                jnp.int32,
                sharding=jax.NamedSharding(mesh, jax.P("expert", None, None, None)),
            ),
        )

    def test_reverse_trace_preserves_expert_vma(self):
        mesh = jax.sharding.AbstractMesh((2,), ("expert",), axis_types=(EXPLICIT,))

        def loss(intermediate, wo, group_sizes, weights, positions):
            def local(intermediate, wo, group_sizes, weights, positions):
                local_routed = model.ragged_dot_ops.ragged_dot_combine(
                    intermediate,
                    wo,
                    group_sizes,
                    weights[0],
                    positions[0],
                    None,
                    False,
                    160,
                )
                return jax.lax.psum_scatter(
                    local_routed, "expert", scatter_dimension=0, tiled=True
                )

            output = jax.shard_map(
                local,
                in_specs=(
                    jax.P("expert", None),
                    jax.P("expert", None, None),
                    jax.P("expert"),
                    jax.P("expert", None, None, None),
                    jax.P("expert", None, None, None),
                ),
                out_specs=jax.P("expert", None, None),
                mesh=mesh,
            )(intermediate, wo, group_sizes, weights, positions)
            return output.astype(jnp.float32).sum()

        with jax.sharding.use_abstract_mesh(mesh):
            jaxpr = jax.make_jaxpr(jax.grad(loss, (0, 1, 3)))(
                *self.vma_trace_args(mesh)
            )

        all_gather = next(
            eqn
            for _, eqn in jaxpr_util.all_eqns(jaxpr)
            if eqn.primitive.name == "all_gather"
        )
        transpose = next(
            eqn
            for eqn in test_util.kernel_launches(jaxpr)
            if eqn.params["name"] == "moe_combine_mosaic_gpu_transpose"
        )
        self.assertEqual(
            all_gather.outvars[0].aval.manual_axis_type.varying, {"expert"}
        )
        self.assertEqual(transpose.outvars[0].aval.manual_axis_type.varying, {"expert"})

    def test_compact_trace_preserves_expert_vma(self):
        mesh = jax.sharding.AbstractMesh((2,), ("expert",), axis_types=(EXPLICIT,))

        def loss(intermediate, wo, group_sizes, weights, positions):
            def local(intermediate, wo, group_sizes, weights, positions):
                return model.ragged_dot_ops.ragged_dot_combine(
                    intermediate,
                    wo,
                    group_sizes,
                    weights[0],
                    positions[0],
                    jnp.full(
                        (positions.shape[1],), positions.shape[2], dtype=jnp.int32
                    ),
                    False,
                    160,
                )

            output = jax.shard_map(
                local,
                in_specs=(
                    jax.P("expert", None),
                    jax.P("expert", None, None),
                    jax.P("expert"),
                    jax.P("expert", None, None, None),
                    jax.P("expert", None, None, None),
                ),
                out_specs=jax.P("expert", None, None),
                mesh=mesh,
            )(intermediate, wo, group_sizes, weights, positions)
            return output.astype(jnp.float32).sum()

        with jax.sharding.use_abstract_mesh(mesh):
            jaxpr = jax.make_jaxpr(jax.grad(loss, (0, 1, 3)))(
                *self.vma_trace_args(mesh)
            )

        kernels = {eqn.params["name"]: eqn for eqn in test_util.kernel_launches(jaxpr)}
        for name in ("moe_combine_mosaic_gpu_fwd", "moe_combine_mosaic_gpu_transpose"):
            self.assertEqual(
                kernels[name].outvars[0].aval.manual_axis_type.varying, {"expert"}
            )
        self.assertEmpty(
            [
                name
                for name in kernels
                if name is not None and name.startswith("gather_sum")
            ]
        )


class DedupExchangeTest(test_util.MaximoeTestCase):
    """Deduplicated all-to-all dispatch and combine of the non-ring path."""

    @parameterized.named_parameters(("sentinels", False), ("empty", True))
    def test_nonring_compact_tail_matches_composed_vjp(self, empty):

        def ragged_all_to_all(
            operand,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            output_shape,
            axis_name,
        ):
            del recv_sizes, axis_name
            group_count = send_sizes.size
            capacity = operand.shape[0] // group_count
            slots = jnp.arange(capacity, dtype=jnp.int32)
            live = slots < send_sizes[:, None]
            input_positions = input_offsets[:, None] + slots
            output_positions = jnp.where(
                live, output_offsets[:, None] + slots, output_shape[0]
            )
            return (
                jnp.zeros(output_shape, operand.dtype)
                .at[output_positions]
                .set(
                    jnp.where(live[..., None], operand[input_positions], 0), mode="drop"
                )
            )

        def dedup_direct(
            inputs, source_token_indices, send_sizes, recv_sizes, axis_name
        ):
            del recv_sizes, axis_name
            local_batch, destinations, capacity = source_token_indices.shape
            sizes = jnp.sum(send_sizes, axis=1, dtype=jnp.int32)
            token_offsets = jnp.transpose(source_token_indices, (1, 0, 2))
            token_offsets = token_offsets + (
                jnp.arange(local_batch, dtype=jnp.int32)[None, :, None]
                * inputs.shape[1]
            )
            flat_inputs = jnp.reshape(inputs, (-1, inputs.shape[-1]))
            live = jnp.arange(capacity) < sizes[..., None]
            positions = (
                jnp.arange(destinations, dtype=jnp.int32)[:, None] * local_batch
                + jnp.arange(local_batch, dtype=jnp.int32)[None, :]
            )[..., None] * capacity + jnp.arange(capacity, dtype=jnp.int32)
            output_shape = (destinations * local_batch * capacity, inputs.shape[-1])
            positions = jnp.where(live, positions, output_shape[0])
            return (
                jnp.zeros(output_shape, inputs.dtype)
                .at[positions]
                .set(
                    jnp.where(live[..., None], flat_inputs[token_offsets], 0),
                    mode="drop",
                )
            )

        self.enterContext(
            mock.patch.object(model, "ragged_all_to_all_mt", ragged_all_to_all)
        )
        self.enterContext(
            mock.patch.object(model, "dedup_ragged_all_to_all_direct", dedup_direct)
        )

        src, local_batch, sequence, source_token_capacity, top_k, model_dim = (
            2,
            1,
            3,
            2,
            2,
            2,
        )
        expert_buffer_size = 8
        dedup_packed_size = sequence * min(top_k, src)
        if empty:
            expert_positions = jnp.full(
                (1, src, local_batch, source_token_capacity, top_k),
                expert_buffer_size,
                dtype=jnp.int32,
            )
            source_positions = jnp.full(
                (local_batch, sequence, top_k), dedup_packed_size, dtype=jnp.int32
            )
            source_token_indices = jnp.full(
                (local_batch, src, source_token_capacity), sequence, dtype=jnp.int32
            )
            offsets = jnp.zeros((src,), dtype=jnp.int32)
            sizes = jnp.zeros((1, src, 1, local_batch), dtype=jnp.int32)
        else:
            expert_positions = jnp.array(
                [[[[[0, 8], [1, 8]]], [[[2, 8], [8, 8]]]]], dtype=jnp.int32
            )
            source_positions = jnp.array([[[0, 2], [1, 6], [6, 6]]], dtype=jnp.int32)
            source_token_indices = jnp.array([[[0, 1], [0, 3]]], dtype=jnp.int32)
            offsets = jnp.array([0, 2], dtype=jnp.int32)
            sizes = jnp.array([[[[2]], [[1]]]], dtype=jnp.int32)

        plan = model.DedupRoutePlan(
            expert_routes=model.RouteMetadata(
                weights=jnp.zeros(expert_positions.shape, dtype=jnp.float32),
                positions=expert_positions,
            ),
            expert_sizes=jnp.zeros((1, 1), dtype=jnp.int32),
            source_positions=source_positions,
            source_token_indices=source_token_indices,
            send_sizes=sizes,
            recv_sizes=sizes,
            return_offsets=offsets.reshape(1, src, 1, local_batch),
        )
        local_compact = (
            jnp.arange(
                src * local_batch * source_token_capacity * model_dim, dtype=jnp.float32
            ).reshape(1, src, local_batch, source_token_capacity, model_dim)
            / 7
        )
        grad_out = jnp.arange(
            local_batch * sequence * model_dim, dtype=jnp.float32
        ).reshape(local_batch, sequence, model_dim)

        def actual(values):
            return model.combine_all_to_all_dedup(
                values, plan, "expert", implementation="xla"
            )

        def expected(values):
            return model.combine_all_to_all_dedup_impl(
                values, plan, axis_name="expert", implementation="xla"
            )

        actual_output, actual_vjp = jax.vjp(actual, local_compact)
        expected_output, expected_vjp = jax.vjp(expected, local_compact)
        (actual_grad,) = actual_vjp(grad_out)
        (expected_grad,) = expected_vjp(grad_out)

        self.assertArraysEqual(actual_output, expected_output)
        self.assertArraysEqual(actual_grad, expected_grad)
        if empty:
            self.assertArraysEqual(actual_output, 0, check_dtypes=False)
            self.assertArraysEqual(actual_grad, 0, check_dtypes=False)

    def test_prepare_builds_fixed_route_plan(self):
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:1]), ("expert",), axis_types=(EXPLICIT,)
        )
        expert_route_spec = jax.P("expert", None, None, None, None)
        source_spec = jax.P(None, None, None)
        exchange_spec = jax.P("expert", None, None, None)
        plan_spec = model.DedupRoutePlan(
            expert_routes=model.RouteMetadata(
                weights=expert_route_spec, positions=expert_route_spec
            ),
            expert_sizes=jax.P("expert", None),
            source_positions=source_spec,
            source_token_indices=source_spec,
            send_sizes=exchange_spec,
            recv_sizes=exchange_spec,
            return_offsets=exchange_spec,
        )
        top_k_weights = jnp.array(
            [[[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.9, 0.1]]], dtype=jnp.float32
        )
        top_k_indices = jnp.array([[[0, 1], [0, 1], [0, 1], [0, 1]]], dtype=jnp.int32)
        token_idx_for_expert = jnp.array(
            [[[0, 0], [1, 1], [2, 2], [3, 3]]], dtype=jnp.int32
        )

        def prepare(weights, indices, token_indices):
            return model.prepare_all_to_all_dedup_ragged(
                weights, indices, token_indices, num_experts=2, drop_capacity=1
            )

        with jax.set_mesh(mesh):
            plan = jax.shard_map(
                prepare, in_specs=jax.P(), out_specs=plan_spec, mesh=mesh
            )(top_k_weights, top_k_indices, token_idx_for_expert)

        self.assertEqual(
            jax.tree.map(lambda value: value.shape, plan),
            model.DedupRoutePlan(
                expert_routes=model.RouteMetadata(
                    weights=(1, 1, 1, 2, 2), positions=(1, 1, 1, 2, 2)
                ),
                expert_sizes=(1, 2),
                source_positions=(1, 4, 2),
                source_token_indices=(1, 1, 2),
                send_sizes=(1, 1, 2, 1),
                recv_sizes=(1, 1, 2, 1),
                return_offsets=(1, 1, 2, 1),
            ),
        )
        self.assertArraysEqual(
            plan.source_positions,
            np.array([[[0, 4], [4, 4], [4, 4], [4, 4]]], np.int32),
        )
        self.assertArraysEqual(
            plan.source_token_indices, np.array([[[0, 4]]], np.int32)
        )
        self.assertArraysEqual(plan.send_sizes, np.array([[[[1], [0]]]], np.int32))
        self.assertArraysEqual(plan.recv_sizes, np.array([[[[1], [0]]]], np.int32))
        self.assertArraysEqual(plan.return_offsets, np.array([[[[0], [1]]]], np.int32))
        self.assertArraysEqual(plan.expert_sizes, np.array([[1, 1]], np.int32))
        self.assertArraysEqual(
            plan.expert_routes.positions, np.array([[[[[0, 1], [256, 256]]]]], np.int32)
        )
        self.assertArraysEqual(
            plan.expert_routes.weights,
            np.array([[[[[0.7, 0.3], [0.0, 0.0]]]]], dtype=np.float32),
        )

    def test_token_routing_builds_capacity_bounded_source_token_indices(self):
        top_k_indices = jnp.array(
            [[0, 2, 3], [1, 4, 5], [2, 4, 0], [5, 3, 1]], dtype=jnp.int32
        )
        route_keep = jnp.array(
            [
                [True, True, True],
                [True, True, True],
                [True, False, False],
                [False, False, False],
            ]
        )

        send_sizes, packed_positions, source_token_indices = model.dedup_token_routing(
            top_k_indices,
            route_keep,
            ep=3,
            lE=2,
            packed_size=12,
            source_token_capacity=2,
        )

        self.assertArraysEqual(send_sizes, np.array([2, 2, 1], np.int32))
        self.assertArraysEqual(
            packed_positions,
            np.array([[0, 2, 12], [1, 4, 12], [3, 12, 12], [12, 12, 12]], np.int32),
        )
        self.assertArraysEqual(
            source_token_indices, np.array([[0, 1], [0, 2], [1, 4]], np.int32)
        )

    def test_return_metadata_uses_fixed_receive_slots(self):
        self.skip_unless_gpus(2)

        self.enterContext(jtu.set_env(**{model.ALL_TO_ALL_MODE_ENV: "all_to_all"}))
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:2]), ("expert",), axis_types=(EXPLICIT,)
        )
        exchange_sizes = jnp.array(
            [[[3, 0], [1, 5]], [[2, 4], [6, 1]]], dtype=jnp.int32
        )
        packed_size = 16
        chunk_count = 3
        chunk_size = 2

        def local(local_send_sizes):
            local_send_sizes = local_send_sizes[0]
            recv_sizes, return_offsets = model.compute_dedup_return_metadata(
                local_send_sizes,
                axis_name="expert",
                packed_size=packed_size,
                chunk_count=chunk_count,
                chunk_size=chunk_size,
            )
            return recv_sizes[None], return_offsets[None]

        with jax.set_mesh(mesh):
            exchange_sizes = jax.device_put(exchange_sizes, jax.P("expert", None, None))
            actual_recv_sizes, actual_return_offsets = jax.shard_map(
                local,
                in_specs=jax.P("expert", None, None),
                out_specs=(
                    jax.P("expert", None, None, None),
                    jax.P("expert", None, None, None),
                ),
                mesh=mesh,
            )(exchange_sizes)

        expected_recv_sizes = np.empty((2, 2, chunk_count, 2), dtype=np.int32)
        expected_return_offsets = np.empty_like(expected_recv_sizes)
        exchange_sizes_np = np.asarray(exchange_sizes)
        for rank in range(2):
            recv_group_sizes = exchange_sizes_np[:, rank]
            source_offsets = exchange_sizes_np[:, :rank].sum(axis=1)
            source_offsets += np.arange(2, dtype=np.int32)[None] * packed_size
            for chunk in range(chunk_count):
                expected_recv_sizes[rank, :, chunk] = np.clip(
                    recv_group_sizes - chunk * chunk_size, 0, chunk_size
                )
                expected_return_offsets[rank, :, chunk] = source_offsets + np.minimum(
                    recv_group_sizes, chunk * chunk_size
                )

        self.assertArraysEqual(actual_recv_sizes, expected_recv_sizes)
        self.assertArraysEqual(actual_return_offsets, expected_return_offsets)

    def test_direct_exchange_fills_fixed_receive_slots(self):
        ranks, local_batch, sequence, capacity, hidden = 3, 2, 4, 2, 2
        self.skip_unless_gpus(ranks)
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:ranks]), ("expert",), axis_types=(EXPLICIT,)
        )
        rng = self.rng()
        # Rows every source sends to every destination, per local batch element.
        counts = rng.randint(0, capacity + 1, size=(ranks, ranks, local_batch))
        inputs = jnp.arange(
            ranks * local_batch * sequence * hidden, dtype=jnp.float32
        ).reshape(ranks, local_batch, sequence, hidden)
        source_token_indices = np.full(
            (ranks, local_batch, ranks, capacity), sequence, np.int32
        )
        for source, destination, batch in np.ndindex(counts.shape):
            count = counts[source, destination, batch]
            source_token_indices[source, batch, destination, :count] = rng.choice(
                sequence, size=count, replace=False
            )
        chunk = functools.partial(
            model.dedup_group_chunk_sizes, chunk_count=2, chunk_size=1
        )
        send_sizes = jnp.stack(
            [chunk(jnp.asarray(counts[rank])) for rank in range(ranks)]
        )
        recv_sizes = jnp.stack(
            [chunk(jnp.asarray(counts[:, rank])) for rank in range(ranks)]
        )

        with jax.set_mesh(mesh):
            args = jax.device_put(
                (inputs, source_token_indices, send_sizes, recv_sizes), jax.P("expert")
            )
            exchanged = jax.shard_map(
                lambda x, indices, send, recv: model.dedup_ragged_all_to_all_direct(
                    x[0], indices[0], send[0], recv[0], "expert"
                )[None],
                mesh=mesh,
                in_specs=jax.P("expert"),
                out_specs=jax.P("expert"),
            )(*args)

        self.assertEqual(
            exchanged.shape, (ranks, ranks * local_batch * capacity, hidden)
        )
        received = np.asarray(exchanged).reshape(
            ranks, ranks, local_batch, capacity, hidden
        )
        flat_inputs = np.asarray(inputs).reshape(ranks, local_batch * sequence, hidden)
        for source, destination, batch in np.ndindex(counts.shape):
            count = counts[source, destination, batch]
            tokens = source_token_indices[source, batch, destination, :count]
            # Slots past ``count`` are never written and stay unspecified.
            self.assertArraysEqual(
                received[destination, source, batch, :count],
                flat_inputs[source, tokens + batch * sequence],
            )

    def test_dispatch_custom_vjp_handles_capacity_sentinels(self):
        # The ragged all-to-all collective has no CPU implementation.
        self.skip_unless_gpus()
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:1]), ("expert",), axis_types=(EXPLICIT,)
        )
        inputs = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
        source_positions = jnp.array([[[1], [4], [0], [4]]], dtype=jnp.int32)
        source_token_indices = jnp.array([[[2, 0, 4, 4]]], dtype=jnp.int32)
        send_sizes = jnp.array([[[2], [0]]], dtype=jnp.int32)
        recv_sizes = send_sizes
        return_offsets = jnp.array([[[0], [2]]], dtype=jnp.int32)
        cotangent = jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 1, 4, 2)
        # Two of the four fixed receive slots are live; the exchange leaves the
        # rest uninitialized, so only the live prefix is observed.
        live = (slice(None), slice(None), slice(2))

        def single_rank(function):
            return jax.shard_map(
                function, mesh=mesh, in_specs=jax.P(), out_specs=jax.P()
            )

        def loss(value):
            dispatched = model.dedup_ragged_all_to_all_mt(
                value,
                source_positions,
                source_token_indices,
                send_sizes,
                recv_sizes,
                return_offsets,
                4,
                "expert",
                implementation="xla",
            )
            return jnp.sum(dispatched[live] * cotangent[live]), dispatched

        value_and_grad = single_rank(jax.value_and_grad(loss, has_aux=True))
        (actual_loss, actual_output), actual_grad = value_and_grad(inputs)

        expected_output = inputs[:, [2, 0]].reshape(1, 1, 2, 2)
        expected_grad = jnp.array([[[3, 4], [0, 0], [1, 2], [0, 0]]], dtype=jnp.float32)
        self.assertEqual(actual_output.shape, (1, 1, 4, 2))
        self.assertArraysEqual(actual_output[live], expected_output)
        self.assertArraysEqual(actual_loss, jnp.sum(expected_output * cotangent[live]))
        self.assertArraysEqual(actual_grad, expected_grad)
        # The forward exchanges one descriptor per fixed slot, the backward one
        # per chunk.
        self.assertEqual(
            descriptor_counts(jax.make_jaxpr(value_and_grad)(inputs)), [4, 2]
        )

        combine_bwd = single_rank(
            lambda grad_out: model.combine_all_to_all_dedup_bwd(
                "expert",
                "xla",
                (source_token_indices, send_sizes[None], recv_sizes[None]),
                grad_out,
            )
        )
        combine_grad, plan_grad = combine_bwd(inputs)
        self.assertArraysEqual(
            combine_grad[0][live], expected_output.reshape(1, 1, 2, 2)
        )
        self.assertIsNone(plan_grad)
        self.assertEqual(descriptor_counts(jax.make_jaxpr(combine_bwd)(inputs)), [4])


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
