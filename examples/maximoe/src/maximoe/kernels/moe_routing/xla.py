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

"""XLA specification of the MaxiMoE routing operations.

This module defines what the ``mosaic_gpu`` kernels compute. It is plain
``jax.numpy``, so it runs on any backend and JAX differentiates it without
custom VJPs: the expert-major gradient of :func:`combine` is the reference for the
composition of :func:`combine_transpose` with :func:`weight_expert_rows` that
``maximoe.kernels.ragged_dot.ops.RaggedDotCombine`` uses as its backward; the
gradient of :func:`permute` is the
reference for :func:`gather_sum`. Every public function
accepts the same arguments as its kernel counterpart, including arguments that
only tune kernel launches.

Route conventions:

* ``positions[b, s, k]`` is the buffer row holding route ``k`` of token
  ``(b, s)``. A position equal to the buffer extent marks a dead route. Dead
  routes contribute nothing and their weights may hold any value, including
  NaN, so weights are masked with ``where`` rather than multiplied by zero.
* Live positions are unique within an expert-major buffer and within each
  batch of a batched buffer, so scattering rows never accumulates. Scatters
  below use ``add`` because under uniqueness it equals assignment and stays
  linear for transposition.
* Rows that no live route targets are unspecified in the kernels. This module
  leaves them zero in freshly created buffers and unchanged in buffers it
  rewrites. Tests compare targeted rows only.
* Products and sums use float32. ``activations_in_float32`` selects a float32
  combined result; otherwise the result uses the buffer dtype. Route weights
  retain their input precision until multiplication.
"""

import jax
import jax.numpy as jnp
import jaxtyping


def gather_rows(
    buffer: jaxtyping.Shaped[jax.Array, "P M"] | jaxtyping.Shaped[jax.Array, "B P M"],
    positions: jaxtyping.Int[jax.Array, "*T K"],
) -> jaxtyping.Shaped[jax.Array, "*T K M"]:
    """Return ``buffer[position]`` per route, or zeros for dead routes.

    A rank-3 buffer is indexed per batch: ``buffer[b, positions[b, ...]]``.
    """
    if buffer.ndim == 3:
        batch = jnp.arange(positions.shape[0], dtype=positions.dtype)
        index = (batch.reshape((-1,) + (1,) * (positions.ndim - 1)), positions)
    else:
        index = (positions,)
    return buffer.at[index].get(mode="fill", fill_value=0)


def scatter_rows(
    values: jaxtyping.Shaped[jax.Array, "*T K M"],
    positions: jaxtyping.Int[jax.Array, "*T K"],
    extent: int,
) -> jaxtyping.Shaped[jax.Array, "P M"]:
    """Write each live route's row into a zero buffer of ``extent`` rows."""
    shape = (extent, values.shape[-1])
    return jnp.zeros(shape, values.dtype).at[positions].add(values, mode="drop")


def combine(
    buffer: jaxtyping.Float[jax.Array, "P M"],
    top_k_weights: jaxtyping.Float[jax.Array, "B S K"] | None,
    positions: jaxtyping.Int[jax.Array, "B S K"],
    activations_in_float32: bool,
    group_sizes: jaxtyping.Int[jax.Array, "B"] | None = None,
) -> jaxtyping.Float[jax.Array, "B S M"]:
    """``out[b, s] = sum_k weight[b, s, k] * buffer[positions[b, s, k]]``.

    Dead routes are skipped. Without ``top_k_weights`` rows are summed as is.
    Weights and products stay in float32 until the output is cast. With
    ``group_sizes`` only the first ``group_sizes[b]`` tokens of each batch are
    combined; later tokens are zero here and unspecified in the kernels.
    """
    if buffer.ndim != 2:
        raise ValueError(f"combine expects [P, M], got {buffer.shape}")
    output_dtype = jnp.float32 if activations_in_float32 else buffer.dtype
    rows = gather_rows(buffer, positions).astype(jnp.float32)
    if top_k_weights is not None:
        live = positions < buffer.shape[-2]
        weights = jnp.where(live, top_k_weights, 0)
        rows = rows * weights[..., None]
    output = jnp.sum(rows, axis=-2).astype(output_dtype)
    if group_sizes is not None:
        sequence = positions.shape[1]
        token_live = (
            jnp.arange(sequence, dtype=group_sizes.dtype) < group_sizes[:, None]
        )
        output = jnp.where(token_live[..., None], output, 0)
    return output


def combine_transpose(
    grad_out: jaxtyping.Float[jax.Array, "B S M"],
    positions: jaxtyping.Int[jax.Array, "B S K"],
    extent: int,
    cost_estimate_num_rows: int | None = None,
    group_sizes: jaxtyping.Int[jax.Array, "B"] | None = None,
) -> jaxtyping.Float[jax.Array, "P M"]:
    """``out[positions[b, s, k]] = grad_out[b, s]`` for every live route.

    With ``group_sizes`` only the first ``group_sizes[b]`` tokens of each batch
    are scattered. Rows no live route targets are zero.
    """
    del cost_estimate_num_rows
    if grad_out.ndim != 3:
        raise ValueError(f"combine_transpose expects [B, S, M], got {grad_out.shape}")
    if grad_out.shape[:2] != positions.shape[:2]:
        raise ValueError(
            "combine_transpose expects matching batch and sequence dims, got "
            f"{grad_out.shape[:2]} and {positions.shape[:2]}"
        )
    batch_size, sequence, routes = positions.shape
    hidden = grad_out.shape[-1]
    live = positions < extent
    if group_sizes is not None:
        token_live = (
            jnp.arange(sequence, dtype=group_sizes.dtype) < group_sizes[:, None]
        )
        live = live & token_live[..., None]
    positions = jnp.where(live, positions, extent)
    values = jnp.broadcast_to(
        grad_out[:, :, None, :], (batch_size, sequence, routes, hidden)
    )
    return scatter_rows(values, positions, extent)


def weight_expert_rows(
    buffer: jaxtyping.Float[jax.Array, "P M"],
    top_k_weights: jaxtyping.Float[jax.Array, "B S K"],
    positions: jaxtyping.Int[jax.Array, "B S K"],
    route_dot_values: jaxtyping.Float[jax.Array, "P M"] | None = None,
    cost_estimate_num_rows: int | None = None,
) -> (
    jaxtyping.Float[jax.Array, "P M"]
    | tuple[jaxtyping.Float[jax.Array, "P M"], jaxtyping.Float[jax.Array, "B S K"]]
):
    """``buffer[p] *= weight`` for every live route ``p``; other rows unchanged.

    Weights and products stay in float32 until the buffer is written. With
    ``route_dot_values`` the function also returns
    ``route_dot[b, s, k] = <buffer[p], route_dot_values[p]>`` over the
    unweighted row, cast to the weight dtype and zero for dead routes.
    """
    del cost_estimate_num_rows
    if buffer.ndim != 2:
        raise ValueError(f"weight_expert_rows expects [P, M], got {buffer.shape}")
    if route_dot_values is not None and route_dot_values.shape != buffer.shape:
        raise ValueError(
            f"route_dot_values shape {route_dot_values.shape} must match buffer "
            f"shape {buffer.shape}"
        )
    extent = buffer.shape[0]
    live = positions < extent
    weights = jnp.where(live, top_k_weights, 0)
    rows = gather_rows(buffer, positions).astype(jnp.float32)
    weighted_rows = rows * weights[..., None]
    weighted = buffer.at[positions].set(weighted_rows.astype(buffer.dtype), mode="drop")
    if route_dot_values is None:
        return weighted
    values = gather_rows(route_dot_values, positions).astype(jnp.float32)
    route_dot = jnp.sum(rows * values, axis=-1).astype(top_k_weights.dtype)
    return weighted, route_dot


def permute(
    inputs: jaxtyping.Float[jax.Array, "T M"],
    positions: jaxtyping.Int[jax.Array, "T K"],
    packed_size: int,
) -> jaxtyping.Float[jax.Array, "P M"]:
    """``out[positions[t, k]] = inputs[t]`` for every live route.

    Rows no live route targets are zero.
    """
    tokens, hidden = inputs.shape
    position_tokens, routes = positions.shape
    if position_tokens != tokens:
        raise ValueError(
            f"positions.shape[0]={position_tokens} must equal inputs.shape[0]={tokens}"
        )
    values = jnp.broadcast_to(inputs[:, None, :], (tokens, routes, hidden))
    return scatter_rows(values, positions, packed_size)


def gather_sum(
    buffer: jaxtyping.Float[jax.Array, "P M"] | jaxtyping.Float[jax.Array, "B P M"],
    positions: jaxtyping.Int[jax.Array, "T K"] | jaxtyping.Int[jax.Array, "B T K"],
) -> jaxtyping.Float[jax.Array, "T M"] | jaxtyping.Float[jax.Array, "B T M"]:
    """``out[t] = sum_k buffer[positions[t, k]]`` over live routes.

    Equivalent to gathering rows with zero fill for dead routes, summing over
    the route dimension in float32, and casting to the buffer dtype. A rank-3
    buffer is gathered per batch. This also serves as the transpose of
    :func:`permute`.
    """
    if buffer.ndim not in (2, 3) or positions.ndim != buffer.ndim:
        raise ValueError(
            "gather_sum expects [P, M] with [T, K] or [B, P, M] with [B, T, K], "
            f"got {buffer.shape} and {positions.shape}"
        )
    if buffer.ndim == 3 and positions.shape[0] != buffer.shape[0]:
        raise ValueError(
            f"positions.shape[0]={positions.shape[0]} must equal "
            f"buffer.shape[0]={buffer.shape[0]}"
        )
    rows = gather_rows(buffer, positions).astype(jnp.float32)
    return jnp.sum(rows, axis=-2).astype(buffer.dtype)
