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

"""XLA specification of the MaxiMoE ragged-dot operations.

This module defines what the ``sm90`` and ``sm100`` Mosaic GPU kernels compute.
It uses :func:`jax.lax.ragged_dot_general` for the grouped products and
``jax.numpy`` for everything else, so it runs on any backend and JAX
differentiates it without custom VJPs. Every public function accepts the same
arguments as its kernel counterpart, including arguments that only tune kernel
launches, so callers switch implementations without changes.

Row conventions:

* Groups are packed contiguously from row zero. ``group_sizes[g]`` is the
  number of rows in group ``g``; groups have no alignment padding.
* Only rows below ``sum(group_sizes)`` take part in a product. The unused buffer
  tail is zeroed before the product and in its output, so tail values cannot
  enter a value or gradient. The GPU kernels leave output rows in that tail
  unwritten, so tests compare live rows only.
* Products accumulate in float32. Ordinary ragged-dot results are cast to
  ``output_dtype``, which defaults to the operand dtype; fused SwiGLU
  preactivations and results are cast to its selected output dtype.
"""

import jax
import jax.numpy as jnp
from jax import lax

from maximoe.kernels.ragged_dot import common

SWIGLU_EPILOGUE_BLOCK = common.SWIGLU_EPILOGUE_BLOCK

# out[m, n] = sum_k lhs[m, k] * rhs[g(m), k, n]
LHS_RAGGED_DIMENSION_NUMBERS = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [1]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)
# dx[m, k] = sum_n dout[m, n] * rhs[g(m), k, n]
TRANSPOSED_RHS_DIMENSION_NUMBERS = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)
# dw[g, k, n] = sum_{m in group g} lhs[m, k] * dout[m, n]
CONTRACTING_ROWS_DIMENSION_NUMBERS = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def grouped_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dimension_numbers: lax.RaggedDotDimensionNumbers,
) -> jax.Array:
    """Grouped product with float32 accumulation and exact float32 inputs."""
    precision = lax.Precision.HIGHEST if lhs.dtype == jnp.float32 else None
    return lax.ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        dimension_numbers,
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def live_rows_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dimension_numbers: lax.RaggedDotDimensionNumbers,
) -> jax.Array:
    """Row-ragged product whose unused tail is zero and gradient-free."""
    live = jnp.arange(lhs.shape[0], dtype=group_sizes.dtype) < jnp.sum(group_sizes)
    lhs = jnp.where(live[:, None], lhs, 0)
    out = grouped_dot(lhs, rhs, group_sizes, dimension_numbers)
    return jnp.where(live[:, None], out, 0)


def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    cost_estimate_num_rows: int | None = None,
    *,
    output_dtype=None,
) -> jax.Array:
    """``out[m] = lhs[m] @ rhs[g(m)]`` for live rows ``m``; zero elsewhere.

    Shapes: ``lhs[M, K]``, ``rhs[G, K, N]``, ``group_sizes[G]`` -> ``out[M, N]``
    in ``output_dtype`` (defaulting to ``lhs.dtype``).
    """
    del cost_estimate_num_rows
    out = live_rows_dot(lhs, rhs, group_sizes, LHS_RAGGED_DIMENSION_NUMBERS)
    return out.astype(lhs.dtype if output_dtype is None else output_dtype)


def ragged_dot_dx(
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    cost_estimate_num_rows: int | None = None,
) -> jax.Array:
    """``dx[m] = dout[m] @ rhs[g(m)].T`` for rows ``m < sum(group_sizes)``.

    Shapes: ``dout[M, N]``, ``rhs[G, K, N]`` -> ``dx[M, K]`` in ``rhs.dtype``.
    Rows at or beyond ``sum(group_sizes)`` are zero.
    """
    del cost_estimate_num_rows
    dx = live_rows_dot(
        dout.astype(rhs.dtype), rhs, group_sizes, TRANSPOSED_RHS_DIMENSION_NUMBERS
    )
    return dx.astype(rhs.dtype)


def ragged_dot_dw(
    lhs: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    *,
    output_manual_axis_type=None,
    accumulator: jax.Array | jax.Ref | None = None,
    cost_estimate_num_rows: int | None = None,
) -> jax.Array | None:
    """``dw[g] = sum over live rows m of group g of outer(lhs[m], dout[m])``.

    Shapes: ``lhs[M, K]``, ``dout[M, N]``, ``group_sizes[G]`` -> ``dw[G, K, N]``
    in ``lhs.dtype``. With ``accumulator[G, K, N]`` the result is
    ``accumulator + dw`` after rounding ``dw`` to the accumulator dtype, which
    is what the in-place kernels produce. A Ref accumulator is updated in place
    and returns ``None``.
    """
    del cost_estimate_num_rows
    if lhs.dtype != dout.dtype:
        raise ValueError(
            f"dW lhs and dout have incompatible dtypes {lhs.dtype} vs {dout.dtype}"
        )
    live = jnp.arange(lhs.shape[0], dtype=group_sizes.dtype) < jnp.sum(group_sizes)
    lhs = jnp.where(live[:, None], lhs, 0)
    dout = jnp.where(live[:, None], dout, 0)
    dw = grouped_dot(lhs, dout, group_sizes, CONTRACTING_ROWS_DIMENSION_NUMBERS)
    dw = dw.astype(lhs.dtype)
    if output_manual_axis_type is not None:
        # Replica-local weight gradients contribute to an unreduced cotangent.
        # Keep this cast inside the shared primitive's differentiation boundary.
        axes = (
            output_manual_axis_type.unreduced
            - jax.typeof(dw).manual_axis_type.unreduced
        )
        dw = jax.lax.pcast(dw, tuple(axes), to="unreduced")
    if accumulator is None:
        return dw
    if accumulator.shape != dw.shape:
        raise ValueError(f"dW accumulator shape {accumulator.shape} must be {dw.shape}")
    if accumulator.dtype != dw.dtype:
        raise ValueError(f"dW accumulator dtype {accumulator.dtype} must be {dw.dtype}")
    if isinstance(accumulator, jax.Ref):
        accumulator[...] += dw
        return None
    return accumulator + dw


def interleave_gate_up(gate: jax.Array, up: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Fuse gate and up ``[..., H]`` into ``[..., 2H]`` epilogue-blocked columns.

    Inverse of :func:`deinterleave_gate_up`; the result keeps ``gate``'s sharding.
    """
    gate = jnp.asarray(gate, dtype)
    up = jnp.asarray(up, dtype)
    hidden = gate.shape[-1]
    if hidden % SWIGLU_EPILOGUE_BLOCK:
        raise ValueError(
            f"SwiGLU hidden dim {hidden} must be divisible by {SWIGLU_EPILOGUE_BLOCK}."
        )
    leading = gate.shape[:-1]
    blocks = hidden // SWIGLU_EPILOGUE_BLOCK
    gate_blocks = gate.reshape(*leading, blocks, SWIGLU_EPILOGUE_BLOCK)
    up_blocks = up.reshape(*leading, blocks, SWIGLU_EPILOGUE_BLOCK)
    return jnp.reshape(
        jnp.concatenate((gate_blocks, up_blocks), axis=-1),
        (*leading, 2 * hidden),
        out_sharding=jax.typeof(gate).sharding,
    )


def deinterleave_gate_up(interleaved: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split ``[..., 2H]`` epilogue-blocked columns into gate and up ``[..., H]``.

    Column layout, with ``B = SWIGLU_EPILOGUE_BLOCK``:
    ``[gate_0:gate_B, up_0:up_B, gate_B:gate_2B, up_B:up_2B, ...]``. Prefused
    weights and preactivations share this layout across every implementation.
    """
    two_hidden = interleaved.shape[-1]
    if two_hidden % 2:
        raise ValueError(f"SwiGLU projection width must be even, got {two_hidden}.")
    hidden = two_hidden // 2
    if hidden % SWIGLU_EPILOGUE_BLOCK:
        raise ValueError(
            f"SwiGLU hidden dim {hidden} must be divisible by {SWIGLU_EPILOGUE_BLOCK}."
        )
    leading = interleaved.shape[:-1]
    blocks = interleaved.reshape(
        *leading, hidden // SWIGLU_EPILOGUE_BLOCK, 2, SWIGLU_EPILOGUE_BLOCK
    )
    gate = blocks[..., 0, :].reshape(*leading, hidden)
    up = blocks[..., 1, :].reshape(*leading, hidden)
    return gate, up


def swiglu_interleaved(preactivations: jax.Array, out_dtype) -> jax.Array:
    """``silu(gate) * up`` in float32 from interleaved ``[..., 2H]`` columns."""
    gate, up = deinterleave_gate_up(preactivations)
    output = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    return output.astype(out_dtype)


def ragged_dot_swiglu(
    lhs: jax.Array,
    rhs_interleaved: jax.Array,
    group_sizes: jax.Array,
    out_dtype=None,
    cost_estimate_num_rows: int | None = None,
    *,
    save_preact: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """``silu(gate) * up`` with ``[gate | up]`` from a ragged first projection.

    Shapes: ``lhs[M, K]``, ``rhs_interleaved[G, K, 2H]`` -> ``out[M, H]`` in
    ``out_dtype`` (default ``lhs.dtype``). ``rhs_interleaved`` follows
    :func:`deinterleave_gate_up`. Products accumulate in float32 and are rounded
    once to ``out_dtype`` before the activation. SwiGLU itself is evaluated in
    float32 from those rounded values. The unused output tail is zero. With
    ``save_preact`` the rounded ``[M, 2H]`` preactivations are returned as well,
    for :func:`swiglu_backward`.
    """
    del cost_estimate_num_rows
    output_dtype = jnp.dtype(lhs.dtype if out_dtype is None else out_dtype)
    preactivations = live_rows_dot(
        lhs, rhs_interleaved, group_sizes, LHS_RAGGED_DIMENSION_NUMBERS
    ).astype(output_dtype)
    output = swiglu_interleaved(preactivations, output_dtype)
    return (output, preactivations) if save_preact else output


def swiglu_backward(
    preact: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    cost_estimate_num_rows: int | None = None,
) -> jax.Array:
    """Cotangent of :func:`swiglu_interleaved` with respect to ``preact``.

    Shapes: ``preact[M, 2H]``, ``dout[M, H]`` -> ``dpreact[M, 2H]`` in
    ``preact.dtype``.
    """
    del group_sizes, cost_estimate_num_rows
    _, pullback = jax.vjp(lambda x: swiglu_interleaved(x, x.dtype), preact)
    return pullback(dout.astype(preact.dtype))[0]
