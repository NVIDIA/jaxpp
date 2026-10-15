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

"""Mosaic GPU SwiGLU over the live prefix of an interleaved capacity buffer.

The allocated shape stays static. Only tiles intersecting ``[0, num_rows)``
are visited; the unused output and gradient tails are unspecified. Callers must
mask downstream contractions with the same live group sizes. Differentiation is
defined by the shared operations in :mod:`maximoe.kernels.ragged_dot.ops`.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from maximoe.kernels.ragged_dot import common


@functools.partial(jax.jit, static_argnums=(3, 4))
def _kernel(preact, num_rows, dy, out_dtype, cost_estimate_num_rows):
    backward = dy is not None
    rows, two_hidden = preact.shape
    block = common.SWIGLU_EPILOGUE_BLOCK
    if two_hidden % (2 * block):
        raise ValueError("SwiGLU requires interleaved blocks of 32 gate/up elements.")
    hidden = two_hidden // 2
    groups = rows * hidden // block
    tile_groups = math.gcd(groups, 32)
    tile_elements = tile_groups * block
    if tile_elements < 128:
        raise ValueError(
            "SwiGLU buffer must contain a multiple of 128 hidden elements."
        )
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = rows
    if not 0 <= cost_estimate_num_rows <= rows:
        raise ValueError(f"Cost-estimate rows must be in [0, {rows}].")
    inputs = (preact, num_rows) if not backward else (preact, num_rows, dy)
    manual_axis_type = common.output_manual_axis_type(*inputs)
    preact_blocks = preact.reshape(groups, 2, block)
    dy_blocks = None if dy is None else dy.reshape(groups, block)
    output_shape = preact_blocks.shape if backward else (groups, block)
    output_dtype = preact.dtype if backward else out_dtype
    layout = plgpu.Layout.WG_STRIDED(
        (tile_groups, block), vec_size=min(4, tile_elements // 128)
    )

    def kernel(preact_ref, num_rows_ref, dy_ref, out_ref):
        num_tiles = pl.cdiv(num_rows_ref[...] * hidden, tile_elements)

        @plgpu.nd_loop((num_tiles,), collective_axes="program")
        def tiles(loop_info):
            span = pl.ds(loop_info.index[0] * tile_groups, tile_groups)
            gate = plgpu.load(preact_ref.at[span, 0, :], layout=layout).astype(
                jnp.float32
            )
            up = plgpu.load(preact_ref.at[span, 1, :], layout=layout).astype(
                jnp.float32
            )
            if backward:
                dout = plgpu.load(dy_ref.at[span, :], layout=layout).astype(jnp.float32)
                sigmoid = jax.nn.sigmoid(gate)
                dsilu = dout * up
                dgate = dsilu * sigmoid + (dsilu * gate) * sigmoid * (1.0 - sigmoid)
                dup = dout * (gate * sigmoid)
                out_ref[span, 0, :] = dgate.astype(output_dtype)
                out_ref[span, 1, :] = dup.astype(output_dtype)
            else:
                out_ref[span, :] = (jax.nn.silu(gate) * up).astype(output_dtype)

    elements = cost_estimate_num_rows * hidden
    name = f"swiglu_live_{'bwd' if backward else 'fwd'}"
    result = pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(
            output_shape, output_dtype, manual_axis_type=manual_axis_type
        ),
        mesh=plgpu.Mesh(
            grid=(min(1024, groups // tile_groups),), grid_names=("program",)
        ),
        compiler_params=plgpu.CompilerParams(),
        name=name,
        cost_estimate=pl.CostEstimate(
            flops=(10 if backward else 3) * elements,
            transcendentals=elements,
            bytes_accessed=elements
            * (
                2 * preact.dtype.itemsize
                + (dy.dtype.itemsize if backward else 0)
                + (2 if backward else 1) * jnp.dtype(output_dtype).itemsize
            ),
        ),
        metadata={"cost_estimate_num_rows": str(cost_estimate_num_rows)},
    )(preact_blocks, num_rows, dy_blocks)
    return result.reshape(preact.shape if backward else (rows, hidden))


def forward(preact, num_rows, out_dtype, cost_estimate_num_rows=None):
    """``silu(gate) * up`` for rows below ``num_rows`` of an interleaved buffer."""
    return _kernel(preact, num_rows, None, out_dtype, cost_estimate_num_rows)


def backward(preact, num_rows, dout, cost_estimate_num_rows=None):
    """Preactivation cotangent for rows below ``num_rows``; the tail is unspecified."""
    return _kernel(preact, num_rows, dout, preact.dtype, cost_estimate_num_rows)
