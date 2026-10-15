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

"""DeepSeek V3 model operations with an optimized expert-parallel MoE path.

Token-drop capacity is computed per batch element by vmapping
``calculate_expert_buffer_indices``. Folding batch into sequence would make
drops depend on batch order. With capacity C, route counts C+5 and C+3 would
fit under a global 2C limit until the second element, dropping all eight rows
from that element. Per-element capacity drops five and three rows respectively.
Context shards of the same sequence share one capacity budget: only their
per-expert counts are gathered to compute sequence-wide admission indices.

Axis glossary (MoE / dispatch):
    B    global batch                   lB   per-shard local batch
    S    sequence length                T    flattened tokens or expert rows
    M    model (hidden) dim             H    expert MLP intermediate dim
    H2   interleaved gate/up width 2H   K    routes per token (top-k)
    E    global expert count            lE   experts per expert-axis shard
    C    per-expert capacity            N    flattened routes of one sequence
    P    packed expert-buffer rows      U    deduplicated tokens per destination
    src  sending expert-axis shard      dst  receiving expert-axis shard
    peer expert-axis shard where send/receive direction does not matter
    chunk  part of one fixed exchange slot sent as a separate collective range

    Use src/dst in collective-adjacent code (send/receive sizes and offsets for
    ``ragged_all_to_all``) and peer before the direction is meaningful. ``ep``
    names the expert-axis size. Lowercase einsum and einops labels, such as the
    attention code's ``b s d n``, are local to the function that uses them.
"""

import dataclasses
import functools
import math
import os
from typing import Generic, NamedTuple, Self, TypeVar

import einops
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from jax import ad_checkpoint as adc
from jax._src.cudnn.fused_attention_stablehlo import MaskType as CudnnMaskType
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention as cudnn_dot_product_attention,
)
from jax._src.pallas.utils import align_to
from jax.experimental import xla_metadata

from maximoe.kernels import backend, moe_routing
from maximoe.kernels import ragged_dot as ragged_dot_ops

# Only the allocation is tile-aligned; expert groups are packed without gaps.
RAGGED_BUFFER_ALIGNMENT = ragged_dot_ops.RAGGED_BUFFER_ALIGNMENT
ALL_TO_ALL_MODE_ENV = "MAXIMOE_ALL_TO_ALL_MODE"


T = TypeVar("T")
V = TypeVar("V")


def all_to_all(
    operand: jax.Array,
    axis_name: str,
    split_axis: int,
    concat_axis: int,
    *,
    tiled: bool = False,
) -> jax.Array:
    """Run all-to-all, optionally using all-gather plus a local slice.

    The fallback trades a larger metadata payload for avoiding NCCL all-to-all
    peer setup, which can exhaust transport memory at large expert-axis sizes.
    The environment value is captured while JAX traces this function, so it
    must be set before compilation.
    """
    mode = os.environ.get(ALL_TO_ALL_MODE_ENV, "all_to_all").strip().lower()
    if mode in ("", "all_to_all", "a2a", "nccl", "false", "0", "off", "no"):
        return jax.lax.all_to_all(
            operand,
            axis_name=axis_name,
            split_axis=split_axis,
            concat_axis=concat_axis,
            tiled=tiled,
        )
    if mode not in ("all_gather", "gather", "true", "1", "on", "yes"):
        raise ValueError(
            f"Unsupported {ALL_TO_ALL_MODE_ENV}={mode!r}; expected all_to_all or "
            "all_gather"
        )

    ndim = operand.ndim
    split_axis %= ndim
    concat_axis %= ndim
    if not tiled or split_axis != concat_axis:
        raise NotImplementedError(
            f"{ALL_TO_ALL_MODE_ENV}=all_gather only supports tiled all_to_all "
            "with split_axis == concat_axis"
        )

    peer_count = jax.lax.axis_size(axis_name)
    moved = jnp.moveaxis(operand, split_axis, 0)
    by_destination = moved.reshape(
        peer_count, moved.shape[0] // peer_count, *moved.shape[1:]
    )
    gathered = jax.lax.all_gather(by_destination, axis_name=axis_name)
    by_source = gathered[:, jax.lax.axis_index(axis_name)]
    merged = by_source.reshape(moved.shape)
    return jnp.moveaxis(merged, 0, split_axis)


class AttentionLayerParams(NamedTuple, Generic[T]):
    """Attention layer weights for MLA with pre/post layer norms.

    Weight shapes (for DeepSeek v3 defaults):
      pre_attn_norm_scale:  [emb_dim]
      post_attn_norm_scale: [emb_dim]
      wq_a:                 [emb_dim, q_lora_rank]
      q_norm_scale:         [q_lora_rank]
      wq_b:                 [q_lora_rank, num_heads, qk_head_dim]
      wkv_a:                [emb_dim, kv_lora_rank + qk_rope_head_dim]
      kv_norm_scale:        [kv_lora_rank]
      wkv_b:                [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
      out_kernel:           [num_heads, v_head_dim, emb_dim]
    """

    pre_attn_norm_scale: T
    wq_a: T
    q_norm_scale: T
    wq_b: T
    wkv_a: T
    kv_norm_scale: T
    wkv_b: T
    out_kernel: T
    post_attn_norm_scale: T


class RoutingStats(NamedTuple):
    """Pre-drop routing metadata for one MoE invocation."""

    requested_per_seq_expert: jaxtyping.Int[jax.Array, "B E"]
    capacity: jaxtyping.Int[jax.Array, ""]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RaggedArray(Generic[V]):
    """Contiguous expert groups with true row counts and an unused buffer tail."""

    values: V
    sizes: V


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RouteMetadata(Generic[V]):
    """Token-route metadata needed to gather expert outputs back to tokens."""

    weights: V
    positions: V


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DedupRoutePlan(Generic[V]):
    """Routing and exchange state shared by dispatch, experts, and combine.

    Shapes describe values local to one expert-axis shard. ``expert_routes`` is
    ``[1, src, lB, U, K]``, while ``expert_sizes[1, lE]`` counts compact expert
    rows. ``source_positions[lB, S, K]`` maps each original route into
    the compact source return buffer, and ``source_token_indices[lB, dst, U]``
    maps each fixed destination slot back to its original source token. Keeping
    both inverse views avoids materializing a source activation buffer in either
    transpose. ``send_sizes[1, dst, chunk, lB]``,
    ``recv_sizes[1, src, chunk, lB]``, and equally shaped ``return_offsets``
    describe the live collective prefixes and their compact return locations.
    """

    expert_routes: RouteMetadata[V]
    expert_sizes: V
    source_positions: V
    source_token_indices: V
    send_sizes: V
    recv_sizes: V
    return_offsets: V


def ffn(
    wi_0: jaxtyping.Float[jax.Array, "M H"],
    wi_1: jaxtyping.Float[jax.Array, "M H"],
    wo: jaxtyping.Float[jax.Array, "H M"],
    inputs: jaxtyping.Float[jax.Array, "*batch M"],
    *,
    dtype: jnp.dtype,
    out_sharding: jax.sharding.NamedSharding | jax.P | None,
) -> jaxtyping.Float[jax.Array, "*batch M"]:
    """Dense SwiGLU MLP ``(silu(x @ wi_0) * (x @ wi_1)) @ wo`` in ``dtype``."""
    x = inputs.astype(dtype)
    gate = jnp.einsum("...M,MH->...H", x, wi_0.astype(dtype))
    gate = adc.checkpoint_name(gate, "mlpwi_0")
    up = jnp.einsum("...M,MH->...H", x, wi_1.astype(dtype))
    up = adc.checkpoint_name(up, "mlpwi_1")
    hidden = (jax.nn.silu(gate) * up).astype(dtype)
    output = jnp.einsum(
        "...H,HM->...M", hidden, wo.astype(dtype), out_sharding=out_sharding
    )
    return adc.checkpoint_name(output, "mlpwo")


def ffn_ragged_intermediate(
    wi: jaxtyping.Float[jax.Array, "E M H2"],
    inputs: RaggedArray[jaxtyping.Float[jax.Array, "T M"]],
    *,
    dtype: jnp.dtype,
    activations_in_float32: bool = False,
    cost_estimate_num_rows: int | None = None,
    implementation: str | None = None,
) -> jaxtyping.Float[jax.Array, "T H"]:
    casted_inputs = jnp.asarray(inputs.values, dtype)
    group_sizes = inputs.sizes

    out_dtype = jnp.float32 if activations_in_float32 else dtype
    intermediate = ragged_dot_ops.ragged_dot_swiglu(
        casted_inputs,
        jnp.asarray(wi, dtype),
        group_sizes,
        out_dtype,
        cost_estimate_num_rows,
        implementation=implementation,
    )
    intermediate = adc.checkpoint_name(intermediate, "mlpwi_0")
    return intermediate


def storage_packed_size_per_batch(
    *,
    sequence_length: int,
    num_experts_per_tok: int,
    num_target_experts: int,
    drop_capacity: int | None,
) -> int:
    """Tile-aligned allocation for compact routes; ``None`` means dropless."""
    live_bound = sequence_length * min(num_experts_per_tok, num_target_experts)
    if drop_capacity is not None:
        live_bound = min(live_bound, num_target_experts * drop_capacity)
    return align_to(live_bound, RAGGED_BUFFER_ALIGNMENT)


def compute_drop_capacity(
    *, routes_per_sequence: int, num_experts: int, capacity_factor: float
) -> int | None:
    """Routes each sequence may keep per expert; ``None`` means dropless."""
    if capacity_factor < 0:
        return None
    return int(
        max(
            math.ceil(routes_per_sequence / num_experts) * capacity_factor,
            capacity_factor,
        )
    )


def dedup_expanded_buffer_size(
    ep: int,
    lB: int,
    S: int,
    K: int,
    lE: int,
    *,
    drop_capacity: int | None,
    sequence_shards: int,
) -> int:
    """Expert-buffer allocation for the routes received from ``ep * lB`` chunks.

    The chunks form ``ep * lB // sequence_shards`` whole sequences, each with its
    own capacity budget.
    """
    return (ep * lB // sequence_shards) * storage_packed_size_per_batch(
        sequence_length=S * sequence_shards,
        num_experts_per_tok=K,
        num_target_experts=lE,
        drop_capacity=drop_capacity,
    )


def first_route_per_destination(
    dst_idx: jaxtyping.Int[jax.Array, "S K"],
    route_keep: jaxtyping.Bool[jax.Array, "S K"],
) -> jaxtyping.Bool[jax.Array, "S K"]:
    """Keep only the first live top-k route for each token/destination rank."""
    K = dst_idx.shape[-1]
    first = []
    for k in range(K):
        prev_same = jnp.zeros(dst_idx.shape[:-1], dtype=jnp.bool_)
        for prev in range(k):
            prev_same = prev_same | (
                route_keep[..., prev] & (dst_idx[..., prev] == dst_idx[..., k])
            )
        first.append(route_keep[..., k] & ~prev_same)
    return jnp.stack(first, axis=-1)


def dedup_token_routing(
    top_k_indices: jaxtyping.Int[jax.Array, "S K"],
    route_keep: jaxtyping.Bool[jax.Array, "S K"],
    *,
    ep: int,
    lE: int,
    packed_size: int,
    source_token_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-batch compact routing that sends one token row per destination rank.

    ``packed_pos`` uses ``packed_size`` as the sentinel for duplicate and
    dropped routes, so it fully describes the compact activation exchange.
    """
    dst_idx = top_k_indices.astype(jnp.int32) // lE

    s_idx = jnp.arange(top_k_indices.shape[0], dtype=jnp.int32)[:, None]
    token_dst_counts = (
        jnp.zeros((top_k_indices.shape[0], ep), dtype=jnp.int32)
        .at[s_idx, dst_idx]
        .add(route_keep.astype(jnp.int32), mode="promise_in_bounds")
    )
    token_dst_mask = token_dst_counts > 0
    send_sizes = jnp.sum(token_dst_mask, axis=0, dtype=jnp.int32)
    offsets = jnp.cumsum(send_sizes, axis=0, dtype=jnp.int32) - send_sizes
    token_dst_slots = jnp.cumsum(token_dst_mask, axis=0, dtype=jnp.int32) - 1
    route_token_slots = jnp.take_along_axis(token_dst_slots, dst_idx, axis=-1)
    route_packed_pos = offsets[dst_idx] + route_token_slots
    first_route_mask = first_route_per_destination(dst_idx, route_keep)
    packed_pos = jnp.where(first_route_mask, route_packed_pos, packed_size)
    source_token_indices = jnp.full(
        (ep, source_token_capacity), top_k_indices.shape[0], dtype=jnp.int32
    )
    source_token_indices = source_token_indices.at[
        jnp.where(first_route_mask, dst_idx, ep),
        jnp.where(first_route_mask, route_token_slots, source_token_capacity),
    ].set(jnp.broadcast_to(s_idx, top_k_indices.shape), mode="drop")
    return send_sizes, packed_pos, source_token_indices


def take_along_last_dim_no_scatter_add(
    values: jaxtyping.Shaped[jax.Array, "*batch E"],
    indices: jaxtyping.Int[jax.Array, "*batch K"],
) -> jaxtyping.Shaped[jax.Array, "*batch K"]:
    axis = values.ndim - 1
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(axis,),
        start_index_map=(axis,),
        operand_batching_dims=tuple(range(axis)),
        start_indices_batching_dims=tuple(range(axis)),
    )
    # top_k indices are in-bounds and unique per token slice.
    return jax.lax.gather(
        values,
        indices[..., None],
        dnums,
        slice_sizes=tuple(1 for _ in range(values.ndim)),
        unique_indices=True,
        indices_are_sorted=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def calculate_expert_buffer_indices(
    selected_expert_indices: jaxtyping.Int[jax.Array, "N"], *, num_experts: int
) -> tuple[jaxtyping.Int[jax.Array, "N"], jaxtyping.Int[jax.Array, "E"]]:
    """Return each route's slot within its expert and the routes per expert.

    ``selected_expert_indices[n]`` in ``[0, num_experts)`` is the expert of
    route ``n``. A route's slot counts the earlier routes to the same expert, so
    slots follow route order and lie in ``[0, N)``.
    """
    # [N, E]
    expert_one_hot = jax.nn.one_hot(
        selected_expert_indices, num_experts, axis=-1, dtype=jnp.int32
    )

    # [N, E]: routes to expert e among routes 0..n inclusive.
    cumulative_expert_counts = jnp.cumsum(expert_one_hot, axis=0)
    expert_counts = cumulative_expert_counts[-1]

    # Each route is counted in its own expert's column, so one less is its slot.
    buffer_indices = jnp.take_along_axis(
        cumulative_expert_counts - 1, selected_expert_indices[..., None], axis=-1
    ).squeeze(-1)

    return buffer_indices, expert_counts


def sequence_expert_buffer_indices(
    top_k_indices: jax.Array,
    *,
    num_experts: int,
    sequence_shards: int = 1,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Admission slots in original token order, pooling context shards.

    Activations and token-level routing metadata stay sharded. Only the small
    [context chunk, expert] counts are replicated across the expert axis; each
    chunk offsets its slots by the counts of the chunks preceding it in the
    sequence. Returns the slots and the per-sequence expert counts.
    """
    batch, sequence, top_k = top_k_indices.shape
    slots, counts = jax.vmap(
        functools.partial(calculate_expert_buffer_indices, num_experts=num_experts)
    )(top_k_indices.reshape(batch, sequence * top_k))
    slots = slots.reshape(top_k_indices.shape)
    if sequence_shards == 1:
        return slots, counts
    mesh = mesh if mesh is not None else jax.sharding.get_abstract_mesh()
    if not mesh.empty:
        replica_spec = jax.P(data_replica_axes(mesh) or None, None)
        counts = jax.reshard(counts, jax.sharding.NamedSharding(mesh, replica_spec))
    replicas = data_replicas(mesh)
    # Pseudo-batch order is [data replica, context rank, local sequence].
    grouped = einops.rearrange(
        counts, "(r c b) e -> r c b e", r=replicas, c=sequence_shards
    )
    prefix = einops.rearrange(
        jnp.cumsum(grouped, axis=1) - grouped, "r c b e -> (r c b) e"
    )
    if not mesh.empty:
        batch_spec = jax.P(expert_batch_axes(mesh), None)
        prefix = jax.reshard(prefix, jax.sharding.NamedSharding(mesh, batch_spec))
    slots = slots + jnp.take_along_axis(prefix[:, None, :], top_k_indices, axis=-1)
    return slots, sum_context_shards(
        counts, sequence_shards=sequence_shards, replicas=replicas
    )


def dedup_group_chunk_sizes(
    group_sizes: jaxtyping.Int[jax.Array, "peer lB"],
    *,
    chunk_count: int,
    chunk_size: int,
) -> jaxtyping.Int[jax.Array, "peer chunk lB"]:
    """Split dedup token group sizes into fixed collective chunks."""
    chunk_starts = jnp.arange(chunk_count, dtype=jnp.int32).reshape(1, chunk_count, 1)
    chunk_starts = chunk_starts * jnp.asarray(chunk_size, dtype=jnp.int32)
    sizes = group_sizes.astype(jnp.int32)[:, None, :]
    return jnp.clip(sizes - chunk_starts, 0, chunk_size)


def dedup_fixed_chunk_offsets(
    peer_count: int, local_batch: int, capacity: int, chunk_count: int
) -> jaxtyping.Int[jax.Array, "peer chunk lB"]:
    """Offsets for chunks within fixed ``[peer, local_batch, capacity]`` slots."""
    chunk_size = math.ceil(capacity / chunk_count)
    return (
        jnp.arange(peer_count, dtype=jnp.int32)[:, None, None] * local_batch * capacity
        + jnp.arange(chunk_count, dtype=jnp.int32)[None, :, None] * chunk_size
        + jnp.arange(local_batch, dtype=jnp.int32)[None, None, :] * capacity
    )


def compute_dedup_return_metadata(
    local_send_sizes: jaxtyping.Int[jax.Array, "dst lB"],
    *,
    axis_name: str,
    packed_size: int,
    chunk_count: int,
    chunk_size: int,
) -> tuple[
    jaxtyping.Int[jax.Array, "src chunk lB"], jaxtyping.Int[jax.Array, "src chunk lB"]
]:
    """Receive sizes and offsets into each source's compact return buffer."""
    _, lB = local_send_sizes.shape
    axis_index = jax.lax.axis_index(axis_name)
    exchange_sizes: jax.Array = jax.lax.all_gather(
        local_send_sizes, axis_name=axis_name
    )  # [src, dst, lB]
    recv_group_sizes = exchange_sizes[:, axis_index]
    source_offsets = (
        jnp.cumsum(exchange_sizes, axis=1, dtype=jnp.int32)[:, axis_index]
        - recv_group_sizes
        + jnp.arange(lB, dtype=jnp.int32)[None, :] * packed_size
    )
    recv_sizes = dedup_group_chunk_sizes(
        recv_group_sizes, chunk_count=chunk_count, chunk_size=chunk_size
    )
    return_offsets = (
        source_offsets[:, None, :]
        + jnp.cumsum(recv_sizes, axis=1, dtype=jnp.int32)
        - recv_sizes
    )
    return recv_sizes, return_offsets


def local_ep_exchange_packed(
    inputs: jaxtyping.Float[jax.Array, "B S M"],
    top_k_indices: jaxtyping.Int[jax.Array, "B S K"],
    token_idx_for_expert: jaxtyping.Int[jax.Array, "B S K"],
    route_keep: jaxtyping.Bool[jax.Array, "B S K"],
    *,
    num_experts: int,
    packed_size: int,
    implementation: str | None = None,
) -> tuple[
    RaggedArray[jaxtyping.Float[jax.Array, "P M"]], jaxtyping.Int[jax.Array, "B S K"]
]:
    """Pack expert groups without gaps.

    Kept routes must form a prefix of each sequence's per-expert slot indices,
    as they do after applying the model's per-sequence capacity limit.
    """
    B, S, _ = inputs.shape
    expert_idx = top_k_indices.astype(jnp.int32)
    slot = token_idx_for_expert.astype(jnp.int32)

    def count_routes(expert_idx_b, keep_b):
        sizes_b = (
            jnp.zeros((num_experts,), dtype=jnp.int32)
            .at[expert_idx_b]
            .add(keep_b.astype(jnp.int32), mode="promise_in_bounds")
        )
        return sizes_b

    sizes = jax.vmap(count_routes)(expert_idx, route_keep)

    expert_sizes = einops.einsum(sizes, "B E -> E").astype(jnp.int32)
    expert_offsets = jnp.cumsum(expert_sizes, axis=0, dtype=jnp.int32) - expert_sizes
    batch_prefix = jnp.cumsum(sizes, axis=0, dtype=jnp.int32) - sizes
    batch_prefix_for_route = jnp.take_along_axis(
        jnp.broadcast_to(batch_prefix[:, None, :], (B, S, num_experts)),
        top_k_indices,
        axis=-1,
    )

    safe_slot = jnp.where(route_keep, slot, 0)
    expert_pos = expert_offsets[top_k_indices] + batch_prefix_for_route + safe_slot

    packed_total = B * packed_size
    expert_pos = jnp.where(route_keep, expert_pos, packed_total)
    flat_inputs = einops.rearrange(inputs, "B S M -> (B S) M")
    flat_expert_pos = einops.rearrange(expert_pos, "B S K -> (B S) K")

    expert_values = moe_routing.permute(
        flat_inputs, flat_expert_pos, packed_total, implementation=implementation
    )
    expert_inputs = RaggedArray(values=expert_values, sizes=expert_sizes)
    return expert_inputs, expert_pos


@jax.named_call
def ring_of_experts(
    inputs: jaxtyping.Float[jax.Array, "lB S M"],
    top_k_weights: jaxtyping.Float[jax.Array, "lB S K"],
    top_k_indices: jaxtyping.Int[jax.Array, "lB S K"],
    token_idx_for_expert: jaxtyping.Int[jax.Array, "lB S K"],
    wi: jaxtyping.Float[jax.Array, "lE M H2"],
    wo: jaxtyping.Float[jax.Array, "lE H M"],
    *,
    num_experts: int,
    drop_capacity: int | None,
    sequence_shards: int,
    dtype: jnp.dtype,
    activations_in_float32: bool,
    cost_estimate_num_rows: int,
    axis_name: str = "expert",
    implementation: str | None = None,
) -> jaxtyping.Float[jax.Array, "lB S M"]:
    """Gather every token, apply the local experts, and reduce-scatter.

    With one expert shard this is also the EP=1 path. It then skips the
    collectives, which would be invalid: the trainer strips size-one mesh axes
    from types.
    """
    ep = jax.lax.axis_size(axis_name)
    lE = num_experts // ep
    packed_size = storage_packed_size_per_batch(
        sequence_length=inputs.shape[1] * sequence_shards,
        num_experts_per_tok=top_k_indices.shape[-1],
        num_target_experts=lE,
        drop_capacity=drop_capacity,
    )
    if ep > 1:
        inputs = jax.lax.all_gather(inputs, axis_name=axis_name, tiled=True)
        top_k_weights, top_k_indices, token_idx_for_expert = [
            xla_metadata.set_xla_metadata(
                jax.lax.all_gather(z, axis_name=axis_name, tiled=True),
                combiner_key="moe_route_metadata",
            )
            for z in (top_k_weights, top_k_indices, token_idx_for_expert)
        ]
        top_k_indices = top_k_indices - jax.lax.axis_index(axis_name) * lE
    if sequence_shards > 1:
        # The gathered pseudo-batch is [context rank, local sequence]; rejoin
        # whole sequences so admission and capacity span the context shards.
        inputs, top_k_indices, token_idx_for_expert = (
            expert_batch_to_context(x, sequence_shards=sequence_shards, replicas=1)
            for x in (inputs, top_k_indices, token_idx_for_expert)
        )
    is_local = (top_k_indices >= 0) & (top_k_indices < lE)
    route_keep = is_local
    if drop_capacity is not None:
        route_keep = route_keep & (token_idx_for_expert < drop_capacity)
    expert_inputs, positions = local_ep_exchange_packed(
        inputs,
        jnp.where(is_local, top_k_indices, 0).astype(jnp.int32),
        token_idx_for_expert,
        route_keep,
        num_experts=lE,
        packed_size=packed_size,
        implementation=implementation,
    )
    if sequence_shards > 1:
        positions = context_to_expert_batch(
            positions, sequence_shards=sequence_shards, replicas=1
        )
    intermediate = ffn_ragged_intermediate(
        wi,
        expert_inputs,
        dtype=dtype,
        activations_in_float32=activations_in_float32,
        cost_estimate_num_rows=cost_estimate_num_rows,
        implementation=implementation,
    )
    # Otherwise XLA defers every layer's expert dW to the end of the backward
    # pass, holding each layer's expert cotangent until then.
    routed = ragged_dot_ops.ragged_dot_combine(
        jnp.asarray(intermediate, dtype),
        jnp.asarray(wo, dtype),
        expert_inputs.sizes,
        top_k_weights,
        positions,
        None,
        activations_in_float32,
        cost_estimate_num_rows,
        implementation=implementation,
        eager_weight_gradient=True,
    )
    if ep > 1:
        routed = jax.lax.psum_scatter(
            routed, axis_name=axis_name, scatter_dimension=0, tiled=True
        )
    return routed


def ragged_all_to_all_mt(
    operand,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    output_shape,
    axis_name,
):
    """Exchange live row ranges into an uninitialized destination.

    Callers must not observe unwritten rows: the routing kernels read only live
    route positions, and the expert kernels read only rows within their group
    sizes. Do not differentiate this function directly: JAX's transpose would
    zero-fill the operand gradient with a full broadcast. The dispatch and
    combine custom VJPs instead route their cotangents through this exchange,
    so neither direction initializes a buffer. The destination preserves every
    varying axis of the operand so the exchange does not reintroduce
    ``psum_invariant`` inside the shard map.
    """
    varying_axes = tuple(sorted(jax.typeof(operand).manual_axis_type.varying))
    out = jax.lax.pcast(
        jax.lax.empty(output_shape, dtype=operand.dtype), varying_axes, to="varying"
    )
    return jax.lax.ragged_all_to_all(
        operand,
        output=out,
        input_offsets=input_offsets,
        send_sizes=send_sizes,
        output_offsets=output_offsets,
        recv_sizes=recv_sizes,
        axis_name=axis_name,
    )


def dedup_ragged_all_to_all_direct(
    inputs: jaxtyping.Float[jax.Array, "lB S M"],
    source_token_indices: jaxtyping.Int[jax.Array, "lB dst U"],
    send_sizes: jaxtyping.Int[jax.Array, "dst chunk lB"],
    recv_sizes: jaxtyping.Int[jax.Array, "src chunk lB"],
    axis_name: str,
):
    """Send token rows directly into fixed-capacity per-source receive slots."""
    ep = jax.lax.axis_size(axis_name)
    lB, S, M = inputs.shape
    source_capacity = source_token_indices.shape[-1]
    group_sizes = jnp.sum(send_sizes, axis=1, dtype=jnp.int32)
    recv_group_sizes = jnp.sum(recv_sizes, axis=1, dtype=jnp.int32)
    row = jnp.arange(source_capacity, dtype=jnp.int32)
    row_send_sizes = (row < group_sizes[..., None]).astype(jnp.int32)
    row_recv_sizes = (row < recv_group_sizes[..., None]).astype(jnp.int32)
    input_offsets = einops.rearrange(
        source_token_indices + jnp.arange(lB, dtype=jnp.int32)[:, None, None] * S,
        "lB dst U -> dst lB U",
    )
    output_offsets = (
        jax.lax.axis_index(axis_name) * lB + jnp.arange(lB, dtype=jnp.int32)
    )[:, None] * source_capacity + row
    output_offsets = jnp.broadcast_to(output_offsets, (ep, lB, source_capacity))
    flat_inputs = jnp.reshape(inputs, (lB * S, M))
    return ragged_all_to_all_mt(
        flat_inputs,
        einops.rearrange(input_offsets, "dst lB U -> (dst lB U)"),
        einops.rearrange(row_send_sizes, "dst lB U -> (dst lB U)"),
        einops.rearrange(output_offsets, "dst lB U -> (dst lB U)"),
        einops.rearrange(row_recv_sizes, "src lB U -> (src lB U)"),
        (ep * lB * source_capacity, M),
        axis_name,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def dedup_ragged_all_to_all_mt(
    inputs: jaxtyping.Float[jax.Array, "lB S M"],
    source_positions: jaxtyping.Int[jax.Array, "lB S K"],
    source_token_indices: jaxtyping.Int[jax.Array, "lB dst U"],
    send_sizes: jaxtyping.Int[jax.Array, "dst chunk lB"],
    recv_sizes: jaxtyping.Int[jax.Array, "src chunk lB"],
    return_offsets: jaxtyping.Int[jax.Array, "src chunk lB"],
    packed_size: int,
    axis_name: str,
    implementation: str | None = None,
) -> jaxtyping.Float[jax.Array, "src lB U M"]:
    """Exchange deduplicated rows into fixed per-source batch slots."""
    out, _ = dedup_ragged_all_to_all_fwd(
        inputs,
        source_positions,
        source_token_indices,
        send_sizes,
        recv_sizes,
        return_offsets,
        packed_size,
        axis_name,
        implementation=implementation,
    )
    return out


def dedup_ragged_all_to_all_fwd(
    inputs,
    source_positions,
    source_token_indices,
    send_sizes,
    recv_sizes,
    return_offsets,
    packed_size,
    axis_name,
    implementation: str | None = None,
):
    del packed_size, implementation
    flat_exchanged = dedup_ragged_all_to_all_direct(
        inputs, source_token_indices, send_sizes, recv_sizes, axis_name
    )
    src = recv_sizes.shape[0]
    lB, _, M = inputs.shape
    U = source_token_indices.shape[-1]
    exchanged = jnp.reshape(flat_exchanged, (src, lB, U, M))
    return exchanged, (source_positions, send_sizes, recv_sizes, return_offsets)


def dedup_ragged_all_to_all_bwd(
    packed_size, axis_name, implementation, residuals, grad_out
):
    (source_positions, send_sizes, recv_sizes, return_offsets) = residuals
    src, lB, U, M = grad_out.shape
    input_offsets = dedup_fixed_chunk_offsets(src, lB, U, recv_sizes.shape[1])
    grad_packed = ragged_all_to_all_mt(
        jnp.reshape(grad_out, (src * lB * U, M)),
        einops.rearrange(input_offsets, "src chunk lB -> (src chunk lB)"),
        einops.rearrange(recv_sizes, "src chunk lB -> (src chunk lB)"),
        einops.rearrange(return_offsets, "src chunk lB -> (src chunk lB)"),
        einops.rearrange(send_sizes, "dst chunk lB -> (dst chunk lB)"),
        (lB * packed_size, M),
        axis_name,
    )
    grad_inputs = moe_routing.gather_sum(
        jnp.reshape(grad_packed, (lB, packed_size, M)),
        source_positions,
        implementation=implementation,
    )
    return grad_inputs, None, None, None, None, None


dedup_ragged_all_to_all_mt.defvjp(
    dedup_ragged_all_to_all_fwd, dedup_ragged_all_to_all_bwd
)


@jax.named_call
def prepare_all_to_all_dedup_ragged(
    top_k_weights: jaxtyping.Float[jax.Array, "lB S K"],
    top_k_indices: jaxtyping.Int[jax.Array, "lB S K"],
    token_idx_for_expert: jaxtyping.Int[jax.Array, "lB S K"],
    *,
    num_experts: int,
    drop_capacity: int | None,
    sequence_shards: int = 1,
    axis_name: str = "expert",
):
    ep = jax.lax.axis_size(axis_name)
    axis_index = jax.lax.axis_index(axis_name)
    lB, S, K = top_k_indices.shape
    lE = num_experts // ep
    dedup_packed_size = S * min(K, ep)
    # Keep one physical slot because ragged all-to-all requires a static
    # descriptor extent even when the configured route capacity is zero.
    source_token_capacity = (
        S if drop_capacity is None else max(1, min(S, lE * drop_capacity))
    )
    dedup_chunk_count = min(lE, source_token_capacity)
    dedup_chunk_size = math.ceil(source_token_capacity / dedup_chunk_count)

    if drop_capacity is None:
        route_keep = jnp.ones(token_idx_for_expert.shape, dtype=jnp.bool_)
    else:
        route_keep = token_idx_for_expert < drop_capacity
    send_group_sizes, source_positions, source_token_indices = jax.vmap(
        functools.partial(
            dedup_token_routing,
            ep=ep,
            lE=lE,
            packed_size=dedup_packed_size,
            source_token_capacity=source_token_capacity,
        )
    )(top_k_indices, route_keep)
    local_send_sizes = einops.rearrange(send_group_sizes, "lB dst -> dst lB")
    send_sizes = dedup_group_chunk_sizes(
        local_send_sizes, chunk_count=dedup_chunk_count, chunk_size=dedup_chunk_size
    )
    recv_sizes, return_offsets = compute_dedup_return_metadata(
        local_send_sizes,
        axis_name=axis_name,
        packed_size=dedup_packed_size,
        chunk_count=dedup_chunk_count,
        chunk_size=dedup_chunk_size,
    )

    source_token_live = source_token_indices < S
    safe_source_token_indices = jnp.minimum(source_token_indices, S - 1)
    batch_idx = jnp.arange(lB, dtype=jnp.int32)[:, None, None]
    send_top_k_indices = top_k_indices[batch_idx, safe_source_token_indices]
    send_top_k_weights = top_k_weights[batch_idx, safe_source_token_indices]
    send_route_keep = route_keep[batch_idx, safe_source_token_indices]
    send_route_keep = send_route_keep & source_token_live[..., None]
    send_top_k_indices = jnp.where(
        send_route_keep, send_top_k_indices, num_experts
    ).astype(jnp.int32)
    send_top_k_weights = jnp.where(send_route_keep, send_top_k_weights, 0)
    recv_top_k_indices = all_to_all(
        einops.rearrange(send_top_k_indices, "lB dst U K -> dst lB U K"),
        axis_name,
        0,
        0,
        tiled=True,
    )
    recv_top_k_weights = all_to_all(
        einops.rearrange(send_top_k_weights, "lB dst U K -> dst lB U K"),
        axis_name,
        0,
        0,
        tiled=True,
    )

    local_expert_idx = recv_top_k_indices.astype(jnp.int32) - axis_index * lE
    local_route_keep = (local_expert_idx >= 0) & (local_expert_idx < lE)
    safe_local_expert = jnp.where(local_route_keep, local_expert_idx, 0)
    flat_local_expert = einops.rearrange(safe_local_expert, "... -> (...)")
    flat_keep = einops.rearrange(local_route_keep, "... -> (...)")
    expert_one_hot = jax.nn.one_hot(flat_local_expert, lE, dtype=jnp.int32) * flat_keep[
        :, None
    ].astype(jnp.int32)
    cumulative_expert_counts = jnp.cumsum(expert_one_hot, axis=0, dtype=jnp.int32)
    expert_sizes = jnp.sum(expert_one_hot, axis=0, dtype=jnp.int32)
    slot = jnp.take_along_axis(
        cumulative_expert_counts - 1, flat_local_expert[:, None], axis=-1
    ).squeeze(-1)
    expert_offsets = jnp.cumsum(expert_sizes, axis=0, dtype=jnp.int32) - expert_sizes
    expanded_buffer_size = dedup_expanded_buffer_size(
        ep, lB, S, K, lE, drop_capacity=drop_capacity, sequence_shards=sequence_shards
    )
    flat_expert_positions = expert_offsets[flat_local_expert] + slot
    expert_positions = jnp.where(
        local_route_keep,
        flat_expert_positions.reshape(local_route_keep.shape),
        expanded_buffer_size,
    )

    return DedupRoutePlan(
        expert_routes=RouteMetadata(
            weights=recv_top_k_weights[None, ...], positions=expert_positions[None, ...]
        ),
        expert_sizes=expert_sizes[None, ...],
        source_positions=source_positions,
        source_token_indices=source_token_indices,
        send_sizes=send_sizes[None, ...],
        recv_sizes=recv_sizes[None, ...],
        return_offsets=return_offsets[None, ...],
    )


@jax.named_call
def dispatch_all_to_all_dedup_ragged(
    inputs: jaxtyping.Float[jax.Array, "lB S M"],
    plan: DedupRoutePlan[jax.Array],
    *,
    num_experts: int,
    drop_capacity: int | None,
    sequence_shards: int = 1,
    axis_name: str = "expert",
    implementation: str | None = None,
):
    ep = jax.lax.axis_size(axis_name)
    lB, S, _ = inputs.shape
    K = plan.source_positions.shape[-1]
    lE = num_experts // ep
    dedup_packed_size = S * min(K, ep)
    expanded_buffer_size = dedup_expanded_buffer_size(
        ep, lB, S, K, lE, drop_capacity=drop_capacity, sequence_shards=sequence_shards
    )
    send_sizes = jnp.squeeze(plan.send_sizes, axis=0)
    recv_sizes = jnp.squeeze(plan.recv_sizes, axis=0)
    return_offsets = jnp.squeeze(plan.return_offsets, axis=0)

    exchanged = dedup_ragged_all_to_all_mt(
        inputs,
        plan.source_positions,
        plan.source_token_indices,
        send_sizes,
        recv_sizes,
        return_offsets,
        dedup_packed_size,
        axis_name,
        implementation=implementation,
    )
    exchanged = adc.checkpoint_name(exchanged, "dispatch_exchanged_dedup")

    expert_sizes = jnp.squeeze(plan.expert_sizes, axis=0)
    expert_values = moe_routing.permute(
        einops.rearrange(exchanged, "src lB U M -> (src lB U) M"),
        einops.rearrange(
            jnp.squeeze(plan.expert_routes.positions, axis=0),
            "src lB U K -> (src lB U) K",
        ),
        expanded_buffer_size,
        implementation=implementation,
    )
    return RaggedArray(
        values=adc.checkpoint_name(expert_values, "dispatch_expanded"),
        sizes=expert_sizes,
    )


def combine_all_to_all_dedup_impl(
    local_compact: jaxtyping.Float[jax.Array, "1 src lB U M"],
    plan: DedupRoutePlan[jax.Array],
    *,
    axis_name: str = "expert",
    implementation: str | None = None,
):
    _, src, lB, U, _ = local_compact.shape
    _, source_sequence, source_top_k = plan.source_positions.shape
    dedup_packed_size = source_sequence * min(source_top_k, src)
    reduced_compact = jnp.reshape(
        local_compact, (src * lB * U, local_compact.shape[-1])
    )

    send_sizes = jnp.squeeze(plan.send_sizes, axis=0)
    recv_sizes = jnp.squeeze(plan.recv_sizes, axis=0)
    return_offsets = jnp.squeeze(plan.return_offsets, axis=0)
    input_offsets = dedup_fixed_chunk_offsets(src, lB, U, recv_sizes.shape[1])
    back_exchanged = ragged_all_to_all_mt(
        reduced_compact,
        einops.rearrange(input_offsets, "src chunk lB -> (src chunk lB)"),
        einops.rearrange(recv_sizes, "src chunk lB -> (src chunk lB)"),
        einops.rearrange(return_offsets, "src chunk lB -> (src chunk lB)"),
        einops.rearrange(send_sizes, "dst chunk lB -> (dst chunk lB)"),
        (lB * dedup_packed_size, reduced_compact.shape[-1]),
        axis_name,
    )
    back_exchanged = adc.checkpoint_name(back_exchanged, "combine_exchanged_dedup")
    packed_back = einops.rearrange(
        back_exchanged, "(lB P) M -> lB P M", lB=lB, P=dedup_packed_size
    )

    return moe_routing.gather_sum(
        packed_back, plan.source_positions, implementation=implementation
    )


def combine_all_to_all_dedup(
    local_compact, plan, axis_name="expert", *, implementation: str | None = None
):
    # Normalize nondifferentiable arguments before optimize_remat sees the call.
    return _combine_all_to_all_dedup(
        local_compact, plan, axis_name, backend.resolve(implementation)
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _combine_all_to_all_dedup(
    local_compact: jaxtyping.Float[jax.Array, "1 src lB U M"],
    plan: DedupRoutePlan[jax.Array],
    axis_name: str = "expert",
    implementation: str | None = None,
):
    return combine_all_to_all_dedup_impl(
        local_compact, plan, axis_name=axis_name, implementation=implementation
    )


def combine_all_to_all_dedup_fwd(
    local_compact, plan, axis_name, implementation: str | None = None
):
    output = combine_all_to_all_dedup_impl(
        local_compact, plan, axis_name=axis_name, implementation=implementation
    )
    return output, (plan.source_token_indices, plan.send_sizes, plan.recv_sizes)


def combine_all_to_all_dedup_bwd(axis_name, implementation, residuals, grad_out):
    del implementation
    source_token_indices, send_sizes, recv_sizes = residuals
    lB, src, U = source_token_indices.shape
    grad_reduced_compact = dedup_ragged_all_to_all_direct(
        grad_out,
        source_token_indices,
        jnp.squeeze(send_sizes, axis=0),
        jnp.squeeze(recv_sizes, axis=0),
        axis_name,
    )
    return (
        jnp.reshape(grad_reduced_compact, (1, src, lB, U, grad_out.shape[-1])),
        None,
    )


_combine_all_to_all_dedup.defvjp(
    combine_all_to_all_dedup_fwd, combine_all_to_all_dedup_bwd, optimize_remat=True
)

# ==============================================================================
# MLA ATTENTION
# ==============================================================================
# DeepSeek Multi-Head Latent Attention (MLA) with pre/post RMSNorm, YaRN RoPE,
# LoRA-based Q/KV projections, and cuDNN fused attention.
# ==============================================================================


@jax.named_call
def rms_norm(
    x: jax.Array, scale: jax.Array, *, epsilon: float, dtype: jnp.dtype
) -> jax.Array:
    """Divide ``x`` by its root mean square over the last axis, then scale.

    The statistic is computed in float32. The normalized activations are
    rounded to ``dtype`` before the learned ``scale`` is applied in ``dtype``.
    """
    x32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.square(x32).mean(axis=-1, keepdims=True) + epsilon)
    normalized = (x32 * inv_rms).astype(dtype)
    # As an einsum, the scale gradient is a contraction over tokens that
    # accumulates in float32; a broadcast multiply would reduce in ``dtype``.
    return jnp.einsum("...k,k->...k", normalized, scale.astype(dtype))


def mk_yarn_inv_freq(
    *,
    embedding_dims: int,
    rope_theta: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
    rope_factor: float,
) -> jax.Array:
    """Return the YaRN inverse frequency of every rotary feature pair.

    Pair ``i`` turns ``rope_theta ** (-2i / embedding_dims)`` radians per
    position, completing ``L * inv_freq / (2 pi)`` turns over the original
    context ``L``. YaRN (Peng et al., 2023) keeps the frequency of pairs that
    turn more than ``beta_fast`` times, divides it by ``rope_factor`` for pairs
    that turn fewer than ``beta_slow`` times, and ramps linearly between the
    two pair indices. The frequencies are static, so they are computed once in
    float64.
    """
    if embedding_dims % 2:
        raise ValueError(
            "Embedding dim for rotary position embedding must be a multiple of 2."
        )

    def pair_with_turns(turns: float) -> float:
        """Fractional pair index completing ``turns`` turns over ``L``."""
        context = original_max_position_embeddings
        return (
            embedding_dims
            * math.log(context / (2 * math.pi * turns))
            / (2 * math.log(rope_theta))
        )

    ramp_start = max(math.floor(pair_with_turns(beta_fast)), 0)
    ramp_end = min(math.ceil(pair_with_turns(beta_slow)), embedding_dims - 1)
    pairs = np.arange(embedding_dims // 2)
    base = rope_theta ** (-2.0 * pairs / embedding_dims)
    # 0 keeps the base frequency; 1 divides it by rope_factor.
    interpolation = np.clip(
        (pairs - ramp_start) / max(ramp_end - ramp_start, 1e-3), 0.0, 1.0
    )
    scaled = base * (1 - interpolation + interpolation / rope_factor)
    return jnp.asarray(scaled, jnp.float32)


@functools.partial(jax.named_call, name="yarn_rope")
def yarn(inv_freq: jax.Array, inputs: jax.Array, positions: jax.Array) -> jax.Array:
    """Rotate the interleaved feature pairs of rank-3 ``[B, S, D]`` inputs.

    Pair ``(inputs[..., 2i], inputs[..., 2i + 1])`` turns by
    ``positions * inv_freq[i]`` radians. The output is de-interleaved: its first
    ``D / 2`` features are the rotated first elements of every pair, followed by
    the rotated second elements. Queries and keys share this layout, so their
    dot products equal those of the interleaved rotation. Real cosines and sines
    keep complex residuals out of ``jax.checkpoint``.
    """
    if inputs.ndim != 3:
        raise ValueError(
            "Input is assumed to be a rank 3 tensor of shape [batch, sequence, dims]."
        )
    batch, sequence, dims = inputs.shape
    if dims != 2 * inv_freq.shape[0]:
        raise ValueError(
            "The rotary embedding dimensions must match the input hidden dimension."
        )

    angles = positions.astype(jnp.float32)[..., None] * inv_freq[None, None, :]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    pairs = inputs.reshape(batch, sequence, dims // 2, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    return jnp.concatenate(
        [first * cos - second * sin, first * sin + second * cos], axis=-1
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class YarnRope:
    """YaRN rotary embedding with precomputed corrected inverse frequencies."""

    inv_freq: jaxtyping.Float[jax.Array, "half_dim"]
    embedding_dims: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def mk(
        cls,
        *,
        embedding_dims: int,
        rope_theta: float,
        original_max_position_embeddings: int,
        beta_fast: float,
        beta_slow: float,
        rope_factor: float,
    ) -> Self:
        inv_freq = mk_yarn_inv_freq(
            embedding_dims=embedding_dims,
            rope_theta=rope_theta,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            rope_factor=rope_factor,
        )
        return cls(inv_freq=inv_freq, embedding_dims=embedding_dims)

    def __call__(self, inputs: jax.Array, positions: jax.Array) -> jax.Array:
        return yarn(self.inv_freq, inputs, positions)


@jax.named_call
def proj_q(
    inputs_q: jaxtyping.Float[jax.Array, "b s d"],
    inputs_positions: jaxtyping.Shaped[jax.Array, "b t"],
    wq_a: jaxtyping.Float[jax.Array, "d r"],
    q_norm_scale: jax.Array,
    wq_b: jaxtyping.Float[jax.Array, "r n dq"],
    *,
    epsilon: float,
    qk_nope_head_dim: int,
    dtype: jnp.dtype,
    yarn_rope: YarnRope,
    context_parallel_axis: str | None = None,
) -> jaxtyping.Float[jax.Array, "b t n dq"]:
    """Project tokens to MLA queries through a normalized low-rank latent.

    ``wq_a`` compresses each token, the latent is RMS-normalized, and ``wq_b``
    expands it per head. Features past ``qk_nope_head_dim`` in every head are
    rotated by ``yarn_rope``. With ``context_parallel_axis``, the latent is
    gathered over the sequence before the expansion, so ``wq_b`` and
    ``inputs_positions`` hold this rank's heads and the whole sequence.
    """
    latent = jnp.einsum("bsd,dr->bsr", inputs_q, wq_a)
    latent = adc.checkpoint_name(latent, "query_wa_proj")
    latent = rms_norm(latent, q_norm_scale, epsilon=epsilon, dtype=dtype)
    latent = adc.checkpoint_name(latent, "mla_q")
    if context_parallel_axis is not None:
        latent = jax.lax.all_gather(latent, context_parallel_axis, axis=1, tiled=True)

    query = jnp.einsum("bsr,rnd->bsnd", latent, wq_b)
    rotate_heads = jax.vmap(yarn_rope, in_axes=(2, None), out_axes=2)
    rotated = rotate_heads(query[..., qk_nope_head_dim:], inputs_positions)
    return jnp.concatenate(
        [query[..., :qk_nope_head_dim], rotated.astype(dtype)], axis=-1
    )


@jax.named_call
def proj_kv(
    inputs: jaxtyping.Float[jax.Array, "b s d"],
    inputs_positions: jaxtyping.Int[jax.Array, "b t"],
    wkv_a: jaxtyping.Float[jax.Array, "d k"],
    kv_norm_scale: jax.Array,
    wkv_b: jaxtyping.Float[jax.Array, "r n kv"],
    *,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    epsilon: float,
    dtype: jnp.dtype,
    yarn_rope: YarnRope,
    context_parallel_axis: str | None = None,
) -> tuple[
    jaxtyping.Float[jax.Array, "b t n dq"], jaxtyping.Float[jax.Array, "b t n dv"]
]:
    """Project tokens to MLA keys and values through a shared latent.

    ``wkv_a`` maps each token to a ``kv_lora_rank`` latent followed by one
    rotary key shared by all heads. ``wkv_b`` expands the RMS-normalized latent
    per head into the non-rotary key features and the value. Context
    parallelism gathers the latent and the shared rotary key
    (``kv_lora_rank + qk_rope_head_dim`` values per token), as in ``proj_q``.
    """
    compressed = jnp.einsum("bsd,dk->bsk", inputs, wkv_a)
    latent, shared_key = jnp.split(compressed, [kv_lora_rank], axis=-1)
    latent = rms_norm(latent, kv_norm_scale, epsilon=epsilon, dtype=dtype)
    latent = adc.checkpoint_name(latent, "mla_kv")
    if context_parallel_axis is not None:
        latent, shared_key = (
            jax.lax.all_gather(x, context_parallel_axis, axis=1, tiled=True)
            for x in (latent, shared_key)
        )

    expanded = jnp.einsum("bsr,rnd->bsnd", latent, wkv_b)
    content_key, value = jnp.split(expanded, [qk_nope_head_dim], axis=-1)
    shared_key = yarn_rope(shared_key, inputs_positions).astype(dtype)
    rotary_key = einops.repeat(shared_key, "b s d -> b s n d", n=wkv_b.shape[1])
    return jnp.concatenate([content_key, rotary_key], axis=-1), value


def self_attention(
    inputs: jaxtyping.Float[jax.Array, "b s d"],
    decoder_positions: jaxtyping.Int[jax.Array, "b s"],
    decoder_segment_ids: jaxtyping.Int[jax.Array, "b s"],
    attn_params: AttentionLayerParams[jax.Array],
    *,
    epsilon: float,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    softmax_scale: float,
    dtype: jnp.dtype,
    yarn_rope: YarnRope,
    context_parallel_axis: str | None = None,
) -> jaxtyping.Float[jax.Array, "b s d"]:
    """MLA self-attention, optionally context parallel over a mesh axis.

    Under context parallelism, activations enter with the sequence sharded over
    ``context_parallel_axis`` and ``wq_b``/``wkv_b`` hold a disjoint head shard.
    Each rank gathers the normalized low-rank latents for the whole sequence and
    up-projects only its own heads, so ordinary causal cuDNN attention remains
    balanced and exact. The latents are 1536 + 512 + 64 values per token for
    DeepSeek V3, versus 128 x (192 + 192 + 128) for exchanging up-projected
    Q/K/V; the backward pass reduce-scatters their gradients. An all-to-all
    restores sequence sharding before the output projection and MoE.
    """
    positions = decoder_positions
    if context_parallel_axis is not None:
        heads = attn_params.wq_b.shape[1] * jax.lax.axis_size(context_parallel_axis)
        if heads != attn_params.out_kernel.shape[0]:
            raise ValueError(
                "context-parallel attention expects wq_b/wkv_b sharded by head over "
                f"{context_parallel_axis!r}: got {attn_params.wq_b.shape[1]} local "
                f"heads for {attn_params.out_kernel.shape[0]} output heads"
            )
        positions = jax.lax.all_gather(
            positions, context_parallel_axis, axis=1, tiled=True
        )

    query = proj_q(
        inputs,
        positions,
        attn_params.wq_a,
        attn_params.q_norm_scale,
        attn_params.wq_b,
        epsilon=epsilon,
        qk_nope_head_dim=qk_nope_head_dim,
        dtype=dtype,
        yarn_rope=yarn_rope,
        context_parallel_axis=context_parallel_axis,
    )
    query = query * softmax_scale

    key, value = proj_kv(
        inputs,
        positions,
        attn_params.wkv_a,
        attn_params.kv_norm_scale,
        attn_params.wkv_b,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        epsilon=epsilon,
        dtype=dtype,
        yarn_rope=yarn_rope,
        context_parallel_axis=context_parallel_axis,
    )

    query = adc.checkpoint_name(query, "query_proj")
    key = adc.checkpoint_name(key, "key_proj")
    value = adc.checkpoint_name(value, "value_proj")

    if context_parallel_axis is not None:
        decoder_segment_ids = jax.lax.all_gather(
            decoder_segment_ids, axis_name=context_parallel_axis, axis=1, tiled=True
        )

    valid_tokens = decoder_segment_ids > 0
    token_lengths = valid_tokens.sum(axis=-1, dtype=jnp.int32)
    attn_out = cudnn_dot_product_attention(
        query,
        key,
        value,
        q_seqlen=token_lengths,
        kv_seqlen=token_lengths,
        mask_type=CudnnMaskType.PADDING_CAUSAL,
        scale=1.0,  # query is already scaled by softmax_scale
        qkv_layout="BTNH",
    )

    if context_parallel_axis is not None:
        attn_out = jax.lax.all_to_all(
            attn_out,
            axis_name=context_parallel_axis,
            split_axis=1,
            concat_axis=2,
            tiled=True,
        )

    out = jnp.einsum("bsnh,nhd->bsd", attn_out, attn_params.out_kernel)
    out = adc.checkpoint_name(out, "out_proj")
    return out


def data_replica_axes(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> tuple[str, ...]:
    """Sharded batch axes other than ``expert``; each one replicates the experts."""
    return tuple(axis for axis in ("data", "fsdp") if mesh.shape.get(axis, 1) > 1)


def data_replicas(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> int:
    """Number of independent expert groups in ``mesh``."""
    return math.prod(mesh.shape[axis] for axis in data_replica_axes(mesh))


def expert_batch_axes(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> str | tuple[str, ...] | None:
    """Batch axes for independent replicas of expert-axis collectives."""
    axes = data_replica_axes(mesh)
    if mesh.shape.get("expert", 1) > 1:
        axes += ("expert",)
    return axes[0] if len(axes) == 1 else axes or None


def context_to_expert_batch(
    inputs: jax.Array, *, sequence_shards: int, replicas: int | None = None
) -> jax.Array:
    """View context-sharded tokens as an EP-sharded batch without redistribution.

    ``[B, S, ...]`` becomes ``[B * sequence_shards, S // sequence_shards, ...]``
    in pseudo-batch order [data replica, context rank, local sequence], so every
    device keeps its own tokens. ``replicas`` defaults to the data replicas of
    the current mesh; pass ``replicas=1`` for per-device values in ``shard_map``.
    """
    if replicas is None:
        replicas = data_replicas(jax.sharding.get_abstract_mesh())
    # Reshape one dimension at a time so JAX can infer the output shardings.
    inputs = einops.rearrange(inputs, "(r b) s ... -> r b s ...", r=replicas)
    return einops.rearrange(inputs, "r b (c s) ... -> (r c b) s ...", c=sequence_shards)


def expert_batch_to_context(
    inputs: jax.Array, *, sequence_shards: int, replicas: int | None = None
) -> jax.Array:
    """Inverse of :func:`context_to_expert_batch`."""
    if replicas is None:
        replicas = data_replicas(jax.sharding.get_abstract_mesh())
    # Reshape one dimension at a time so JAX can infer the output shardings.
    inputs = einops.rearrange(
        inputs, "(r c b) s ... -> r b (c s) ...", r=replicas, c=sequence_shards
    )
    return einops.rearrange(inputs, "r b s ... -> (r b) s ...")


def sum_context_shards(
    values: jax.Array, *, sequence_shards: int, replicas: int | None = None
) -> jax.Array:
    """Sum per-chunk summaries in :func:`context_to_expert_batch` order per sequence."""
    if replicas is None:
        replicas = data_replicas(jax.sharding.get_abstract_mesh())
    return einops.reduce(
        values, "(r c b) ... -> (r b) ...", "sum", r=replicas, c=sequence_shards
    )


def optimized_routed_ep(
    inputs: jaxtyping.Float[jax.Array, "B S M"],
    top_k_weights: jaxtyping.Float[jax.Array, "B S K"],
    top_k_indices: jaxtyping.Int[jax.Array, "B S K"],
    wi: jaxtyping.Float[jax.Array, "E M H2"],
    wo: jaxtyping.Float[jax.Array, "E H M"],
    *,
    capacity_factor: float,
    sequence_shards: int = 1,
    activations_in_float32: bool = False,
    use_ring_of_experts: bool = False,
    dtype: jnp.dtype,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    implementation: str | None = None,
) -> tuple[jaxtyping.Float[jax.Array, "B S M"], RoutingStats]:
    """Routed expert dispatch, FFN computation, and combine.

    Routes caller-selected experts, applies their FFNs, and combines the results.
    This is the core routed-expert path used by MaxiMoE. ``wi`` and ``wo`` must be
    partitioned only on their leading expert dimension; callers gather any other
    weight axes.

    Args:
        inputs: [B, S, M] input activations
        top_k_weights: [B, S, K] selected route weights
        top_k_indices: [B, S, K] selected expert indices
        wi: [E, M, 2H] expert input weights in fused SwiGLU epilogue layout
        wo: [E, H, M] expert down projection weights
        capacity_factor: capacity factor for expert buffers; a negative value
            enables dropless routing and reports ``routing_stats.capacity == -1``
        sequence_shards: 1, or the expert axis size for EP-as-context. Context
            inputs use context_to_expert_batch order. Shards of each original
            sequence share a capacity budget in original token order.
        activations_in_float32: whether to compute activations in float32
        use_ring_of_experts: gather every token to each expert shard and
            reduce-scatter the outputs, instead of the deduplicated ragged
            all-to-all. A single expert shard always takes this path.
        dtype: computation dtype
        mesh: must have an ``expert`` axis, which may have size one; all axes
            must use ``AxisType.Explicit``
        implementation: ``xla``, ``sm90``, ``sm100``, or None for the platform default

    Returns:
        routed_output: [B, S, M] weighted output from routed experts
        routing_stats: pre-drop per-sequence routing demand and capacity
    """
    implementation = backend.resolve(implementation)
    mesh = mesh if mesh is not None else jax.sharding.get_abstract_mesh()
    if "expert" not in mesh.shape:
        raise ValueError(
            f"optimized_routed_ep requires an 'expert' mesh axis; got {mesh.shape}"
        )
    nonexplicit_axes = tuple(
        axis for axis in mesh.shape if axis not in mesh.explicit_axes
    )
    if nonexplicit_axes:
        raise ValueError(
            "optimized_routed_ep requires explicit sharding; "
            f"non-explicit axes: {nonexplicit_axes}"
        )
    num_experts = wo.shape[0]
    num_experts_per_tok = top_k_indices.shape[-1]
    expert_axis_size = mesh.shape["expert"]
    if sequence_shards not in (1, expert_axis_size):
        raise ValueError("sequence_shards must be 1 or the expert axis size")
    batch_axes = expert_batch_axes(mesh)
    replica_axes = data_replica_axes(mesh) or None
    activation_spec = jax.P(batch_axes, None, None)
    if any(size > 1 for size in mesh.shape.values()):
        wi_sharding = jax.typeof(wi).sharding
        wo_sharding = jax.typeof(wo).sharding
        if not isinstance(wi_sharding, jax.sharding.NamedSharding) or not isinstance(
            wo_sharding, jax.sharding.NamedSharding
        ):
            raise ValueError(
                "optimized_routed_ep requires wi and wo to have named "
                "expert-only shardings; "
                f"got wi={wi_sharding}, wo={wo_sharding}"
            )
        wi_spec = wi_sharding.spec
        wo_spec = wo_sharding.spec
        expected_weight_partitions = (
            "expert" if expert_axis_size > 1 else None,
            None,
            None,
        )
        wi_partitions = wi_spec.partitions + (None,) * (
            wi.ndim - len(wi_spec.partitions)
        )
        wo_partitions = wo_spec.partitions + (None,) * (
            wo.ndim - len(wo_spec.partitions)
        )
        if (
            wi_partitions != expected_weight_partitions
            or wo_partitions != expected_weight_partitions
        ):
            raise ValueError(
                "optimized_routed_ep requires wi and wo partitioned only on "
                "their expert dimension; "
                f"got wi={wi_spec}, wo={wo_spec}"
            )
    if batch_axes is not None:
        activation_sharding = jax.sharding.NamedSharding(mesh, activation_spec)
        inputs = jax.reshard(inputs, activation_sharding)
        top_k_weights = jax.reshard(top_k_weights, activation_sharding)
        top_k_indices = jax.reshard(top_k_indices, activation_sharding)

    routes_per_sequence = inputs.shape[1] * sequence_shards * num_experts_per_tok
    drop_capacity = compute_drop_capacity(
        routes_per_sequence=routes_per_sequence,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    with jax.named_scope("calculate_expert_buffer_indices"):
        token_idx_for_expert, requested_per_seq_expert = sequence_expert_buffer_indices(
            top_k_indices,
            num_experts=num_experts,
            sequence_shards=sequence_shards,
            mesh=mesh,
        )
    routing_stats = RoutingStats(
        requested_per_seq_expert=requested_per_seq_expert,
        capacity=jnp.asarray(
            -1 if drop_capacity is None else drop_capacity, dtype=jnp.int32
        ),
    )
    if drop_capacity == 0:
        # No route survives; avoid zero-row ragged kernels (unsupported on SM90).
        return jnp.zeros_like(
            inputs, dtype=jnp.float32 if activations_in_float32 else dtype
        ), routing_stats

    # Keep the cost model static and independent of runtime routing metadata.
    # It assumes uniform expert assignment with no storage padding.
    replica_sequences = inputs.shape[0] // (sequence_shards * data_replicas(mesh))
    uniform_routes_per_sequence = (
        routes_per_sequence
        if drop_capacity is None
        else min(routes_per_sequence, num_experts * drop_capacity)
    )
    cost_estimate_num_rows = (
        replica_sequences * uniform_routes_per_sequence // expert_axis_size
    )

    casted_inputs = jnp.asarray(inputs, dtype)
    # Explicit-axis shard_map calls infer specs, including reduced axes, from
    # argument types.
    output_spec = jax.P(batch_axes, None, None)
    if use_ring_of_experts or expert_axis_size == 1:
        routed_output = jax.shard_map(
            functools.partial(
                ring_of_experts,
                num_experts=num_experts,
                drop_capacity=drop_capacity,
                sequence_shards=sequence_shards,
                dtype=dtype,
                activations_in_float32=activations_in_float32,
                cost_estimate_num_rows=cost_estimate_num_rows,
                implementation=implementation,
            ),
            out_specs=output_spec,
            mesh=mesh,
        )(casted_inputs, top_k_weights, top_k_indices, token_idx_for_expert, wi, wo)
        return routed_output, routing_stats

    expert_inputs_spec = RaggedArray(
        values=jax.P(batch_axes, None), sizes=jax.P(batch_axes)
    )
    source_metadata_spec = jax.P(batch_axes, None, None)
    expert_route_metadata_spec = jax.P("expert", None, replica_axes, None, None)
    expert_routes_spec = RouteMetadata(
        weights=expert_route_metadata_spec, positions=expert_route_metadata_spec
    )
    dedup_chunk_size_spec = jax.P("expert", None, None, replica_axes)
    route_plan_spec = DedupRoutePlan(
        expert_routes=expert_routes_spec,
        expert_sizes=jax.P(batch_axes, None),
        source_positions=source_metadata_spec,
        source_token_indices=source_metadata_spec,
        send_sizes=dedup_chunk_size_spec,
        recv_sizes=dedup_chunk_size_spec,
        return_offsets=dedup_chunk_size_spec,
    )
    mlp_output_spec = jax.P("expert", None, replica_axes, None, None)

    def dedup_mlp(wi, wo, expert_inputs, routes, recv_sizes):
        intermediate = ffn_ragged_intermediate(
            wi,
            expert_inputs,
            dtype=dtype,
            activations_in_float32=activations_in_float32,
            cost_estimate_num_rows=cost_estimate_num_rows,
            implementation=implementation,
        )
        route_weights = jnp.squeeze(routes.weights, axis=0)
        route_positions = jnp.squeeze(routes.positions, axis=0)
        src, lB, U, _ = route_positions.shape
        recv_group_sizes = jnp.sum(
            jnp.squeeze(recv_sizes, axis=0), axis=1, dtype=jnp.int32
        )
        local_routed = ragged_dot_ops.ragged_dot_combine(
            jnp.asarray(intermediate, dtype),
            jnp.asarray(wo, dtype),
            expert_inputs.sizes,
            einops.rearrange(route_weights, "src lB U K -> (src lB) U K"),
            einops.rearrange(route_positions, "src lB U K -> (src lB) U K"),
            einops.rearrange(recv_group_sizes, "src lB -> (src lB)"),
            activations_in_float32,
            cost_estimate_num_rows,
            implementation=implementation,
        )
        return jnp.reshape(local_routed, (1, src, lB, U, local_routed.shape[-1]))

    route_plan = jax.shard_map(
        functools.partial(
            prepare_all_to_all_dedup_ragged,
            num_experts=num_experts,
            drop_capacity=drop_capacity,
            sequence_shards=sequence_shards,
        ),
        out_specs=route_plan_spec,
        mesh=mesh,
    )(top_k_weights, top_k_indices, token_idx_for_expert)
    route_plan = adc.checkpoint_name(route_plan, "dedup_dispatch_metadata")
    expert_inputs = jax.shard_map(
        functools.partial(
            dispatch_all_to_all_dedup_ragged,
            num_experts=num_experts,
            drop_capacity=drop_capacity,
            sequence_shards=sequence_shards,
            implementation=implementation,
        ),
        out_specs=expert_inputs_spec,
        mesh=mesh,
    )(casted_inputs, route_plan)
    mlp_output = jax.shard_map(dedup_mlp, out_specs=mlp_output_spec, mesh=mesh)(
        wi, wo, expert_inputs, route_plan.expert_routes, route_plan.recv_sizes
    )
    routed_output = jax.shard_map(
        functools.partial(combine_all_to_all_dedup, implementation=implementation),
        out_specs=output_spec,
        mesh=mesh,
    )(mlp_output, route_plan)
    return routed_output, routing_stats
