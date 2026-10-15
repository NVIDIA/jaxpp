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

"""Architecture-neutral Mosaic-GPU permutation and combination for MoE routing.

The ``xla`` module in this package is the specification these kernels
implement. Both modules expose the same public operation names.

The kernels use scalar route indices and contiguous hidden-dimension vector
loads. This avoids Mosaic GPU's lack of register-array gather indices while
keeping the memory path coalesced on Hopper and Blackwell.

Live route positions are unique across an expert-major buffer and within each
batch of a packed buffer. Backward kernels can therefore write buffer gradients
directly without atomics. The buffer extent is the invalid position sentinel.
Permutation initializes only live route positions; callers either address those
positions directly or mask contractions with live group sizes.
"""

import functools

import jax
import jax.numpy as jnp
import jaxtyping
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from maximoe.kernels import gpu

HIDDEN_BLOCK_SIZES = (1024, 512, 256, 128)
WIDE_HIDDEN_BLOCK_SIZE = 3584
# Mosaic maps the leading Pallas grid axis to CUDA grid-y, whose maximum is
# 65,535 on Hopper and Blackwell. Hopper favors a wider route block, while
# Blackwell's measurements favor eight routes per block.
WEIGHT_ROUTES_PER_BLOCK_SM90 = 32
WEIGHT_ROUTES_PER_BLOCK_SM100 = 8
COMBINE_FWD_TOKENS_PER_TASK = 2
COMBINE_FWD_CTAS_PER_SM = 24
TRANSPOSE_CTAS_PER_SM = 8
COMBINE_COMPILER_PARAMS = plgpu.CompilerParams()
DISJOINT_IO_COMPILER_PARAMS = plgpu.CompilerParams(unsafe_no_auto_barriers=True)


def weight_routes_per_block() -> int:
    """Return the fixed route-weighting policy for the selected GPU."""
    architecture = gpu.architecture()
    if architecture is gpu.GpuArchitecture.SM90:
        return WEIGHT_ROUTES_PER_BLOCK_SM90
    if architecture is gpu.GpuArchitecture.SM100:
        return WEIGHT_ROUTES_PER_BLOCK_SM100
    raise AssertionError(f"Unhandled GPU architecture: {architecture}")


def hidden_block_size(hidden_size: int) -> int:
    """Choose a vector tile that avoids padding for standard model widths."""
    for block_size in HIDDEN_BLOCK_SIZES:
        if hidden_size >= block_size and hidden_size % block_size == 0:
            return block_size
    return HIDDEN_BLOCK_SIZES[-1]


def pad_hidden(values: jax.Array, padded_hidden_size: int) -> jax.Array:
    """Pad only the compatibility path for non-vector-aligned hidden sizes."""
    padding = padded_hidden_size - values.shape[-1]
    if padding == 0:
        return values
    return jnp.pad(values, ((0, 0),) * (values.ndim - 1) + ((0, padding),))


def union_varying_axes(*values: jax.Array | None) -> frozenset[str]:
    """Return every manual axis on which a present operand varies."""
    return frozenset().union(
        *(
            jax.typeof(value).manual_axis_type.varying
            for value in values
            if value is not None
        )
    )


def pcast_to_varying(value: jax.Array, axes: frozenset[str]) -> jax.Array:
    """Promote ``value`` to varying on all of ``axes``."""
    for axis in sorted(axes - jax.typeof(value).manual_axis_type.varying):
        value = jax.lax.pcast(value, axis, to="varying")
    return value


def strided_layout(block_m: int, dtype: jnp.dtype) -> plgpu.Layout:
    """Use one naturally aligned eight-byte vector per lane where possible."""
    vec_size = min(8 // jnp.dtype(dtype).itemsize, block_m // 128)
    return plgpu.Layout.WG_STRIDED((block_m,), vec_size=vec_size)


def persistent_grid(ctas_per_sm: int) -> tuple[int]:
    """Size a persistent grid from the configured usable SM count."""
    total_sms = jax.local_devices()[0].core_count
    sm_margin = gpu.MOSAIC_SM_MARGIN
    if not 0 <= sm_margin < total_sms:
        raise ValueError(
            f"RAGGED_DOT_SM_MARGIN must be in [0, {total_sms - 1}], got {sm_margin}"
        )
    return ((total_sms - sm_margin) * ctas_per_sm,)


def persistent_group_loop(
    group_sizes_ref, num_groups: int, block_s: int, num_m_tiles: int, body
) -> None:
    """Stride concatenated live-prefix tiles over a fixed program grid."""
    program_id = lax.axis_index("program")
    num_programs = lax.axis_size("program")

    def group_loop(group_id, task_base_mod):
        group_size = group_sizes_ref[group_id]
        group_tasks = pl.cdiv(group_size, block_s) * num_m_tiles
        first = (program_id - task_base_mod + num_programs) % num_programs
        local_steps = pl.cdiv(jnp.maximum(group_tasks - first, 0), num_programs)

        def task_loop(local_step, unused):
            local_task = first + local_step * num_programs
            token = (local_task // num_m_tiles) * block_s
            body(
                group_id,
                token,
                jnp.minimum(block_s, group_size - token),
                local_task % num_m_tiles,
            )
            return unused

        lax.fori_loop(0, local_steps, task_loop, None)
        return (task_base_mod + group_tasks) % num_programs

    lax.fori_loop(0, num_groups, group_loop, jnp.int32(0))


def combine_row(
    buffer_ref,
    positions_ref,
    weights_ref,
    batch,
    token,
    m_tile,
    *,
    K: int,
    extent: int,
    block_m: int,
):
    """Gather and reduce one token's routes for one hidden tile."""
    columns = pl.ds(m_tile * block_m, block_m)
    layout = strided_layout(block_m, buffer_ref.dtype)
    acc = plgpu.layout_cast(jnp.zeros((block_m,), jnp.float32), layout)

    for route in range(K):
        position = positions_ref[batch, token, route]
        live = position < extent

        def add_live_route(route_acc):
            row = plgpu.load(buffer_ref.at[position, columns], (), layout=layout)
            row = row.astype(jnp.float32)
            if weights_ref is None:
                return route_acc + row
            weight = weights_ref[batch, token, route].astype(jnp.float32)
            return route_acc + row * weight

        acc = lax.cond(live, add_live_route, lambda route_acc: route_acc, acc)
    return acc


def combine_fwd_kernel(
    buffer_ref,
    positions_ref,
    *refs,
    K: int,
    extent: int,
    block_m: int,
    weighted: bool,
    persistent: bool,
):
    if weighted:
        weights_ref, *refs = refs
    else:
        weights_ref = None
    if persistent:
        group_sizes_ref, *refs = refs
    (out_ref,) = refs
    num_m_tiles = out_ref.shape[-1] // block_m

    def write_token(batch, token, m_tile):
        result = combine_row(
            buffer_ref,
            positions_ref,
            weights_ref,
            batch,
            token,
            m_tile,
            K=K,
            extent=extent,
            block_m=block_m,
        )
        out_ref.at[batch, token, pl.ds(m_tile * block_m, block_m)][...] = result.astype(
            out_ref.dtype
        )

    if persistent:

        def run_tile(batch, token_start, row_count, m_tile):
            for token_offset in range(COMBINE_FWD_TOKENS_PER_TASK):

                @pl.when(token_offset < row_count)
                def run_token():
                    write_token(batch, token_start + token_offset, m_tile)

        persistent_group_loop(
            group_sizes_ref,
            num_groups=group_sizes_ref.shape[0],
            block_s=COMBINE_FWD_TOKENS_PER_TASK,
            num_m_tiles=num_m_tiles,
            body=run_tile,
        )
    else:
        batch, token = divmod(lax.axis_index("token"), positions_ref.shape[1])

        def tile_loop(m_tile, unused):
            write_token(batch, token, m_tile)
            return unused

        lax.fori_loop(0, num_m_tiles, tile_loop, None)


def combine_transpose_kernel(
    grad_out_ref,
    positions_ref,
    *refs,
    S: int,
    K: int,
    extent: int,
    block_m: int,
    persistent: bool,
):
    if persistent:
        group_sizes_ref, *refs = refs
    (out_ref,) = refs

    layout = strided_layout(block_m, grad_out_ref.dtype)

    def run_tile(batch, token, row_count, m_tile):
        del row_count
        columns = pl.ds(m_tile * block_m, block_m)
        grad = plgpu.load(grad_out_ref.at[batch, token, columns], (), layout=layout)

        for route in range(K):
            position = positions_ref[batch, token, route]

            @pl.when(position < extent)
            def write_row():
                out_ref.at[position, columns][...] = grad

    if persistent:
        persistent_group_loop(
            group_sizes_ref,
            num_groups=group_sizes_ref.shape[0],
            block_s=1,
            num_m_tiles=grad_out_ref.shape[-1] // block_m,
            body=run_tile,
        )
    else:
        linear_token = lax.axis_index("token")
        batch, token = divmod(linear_token, S)
        run_tile(batch, token, 1, lax.axis_index("m"))


def weight_expert_rows_kernel(
    buffer_ref,
    weights_ref,
    positions_ref,
    *refs,
    S: int,
    K: int,
    extent: int,
    block_m: int,
    num_routes: int,
    routes_per_block: int,
    compute_route_dot: bool,
):
    if compute_route_dot:
        route_dot_values_ref, route_dot_partial_ref = refs
    else:
        del refs

    route_block = lax.axis_index("route")
    m_tile = lax.axis_index("m")
    columns = pl.ds(m_tile * block_m, block_m)
    layout = strided_layout(block_m, buffer_ref.dtype)

    for route_offset in range(routes_per_block):
        linear_route = route_block * routes_per_block + route_offset

        @pl.when(linear_route < num_routes)
        def run_route():
            batch = linear_route // (S * K)
            token_route = linear_route % (S * K)
            token, route = divmod(token_route, K)
            position = positions_ref[batch, token, route]
            if compute_route_dot:
                route_dot_partial_ref[m_tile, batch, token, route] = 0.0

            @pl.when(position < extent)
            def run_live_route():
                buffer_row = buffer_ref.at[position, columns]
                row_f32 = plgpu.load(buffer_row, (), layout=layout).astype(jnp.float32)

                if compute_route_dot:
                    route_dot_values = plgpu.load(
                        route_dot_values_ref.at[position, columns], (), layout=layout
                    ).astype(jnp.float32)
                    route_dot_partial_ref[m_tile, batch, token, route] = jnp.sum(
                        row_f32 * route_dot_values, dtype=jnp.float32
                    )

                weight = weights_ref[batch, token, route].astype(jnp.float32)
                buffer_row[...] = (row_f32 * weight).astype(buffer_ref.dtype)


def weight_expert_rows_mosaic_gpu(
    buffer: jaxtyping.Float[jax.Array, "P M"],
    top_k_weights: jaxtyping.Float[jax.Array, "B S K"],
    positions: jaxtyping.Int[jax.Array, "B S K"],
    route_dot_values: jaxtyping.Float[jax.Array, "P M"] | None = None,
    cost_estimate_num_rows: int | None = None,
) -> (
    jaxtyping.Float[jax.Array, "P M"]
    | tuple[jaxtyping.Float[jax.Array, "P M"], jaxtyping.Float[jax.Array, "B S K"]]
):
    """Multiply each live expert row by its route weight in place."""
    B, S, K = positions.shape
    if buffer.ndim != 2:
        raise ValueError(f"weight_expert_rows expects [P, M], got {buffer.shape}")
    extent, M = buffer.shape
    if route_dot_values is not None and route_dot_values.shape != buffer.shape:
        raise ValueError(
            f"route_dot_values shape {route_dot_values.shape} must match buffer "
            f"shape {buffer.shape}"
        )
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = B * S * K
    top_k_weights_dtype = top_k_weights.dtype
    top_k_weights = jnp.where(positions < extent, top_k_weights, 0)
    inputs = (buffer, top_k_weights, positions)
    if route_dot_values is not None:
        inputs = (*inputs, route_dot_values)
    buffer = pcast_to_varying(buffer, union_varying_axes(*inputs))

    block_m = (
        WIDE_HIDDEN_BLOCK_SIZE
        if M % WIDE_HIDDEN_BLOCK_SIZE == 0
        else hidden_block_size(M)
    )
    padded_m = pl.cdiv(M, block_m) * block_m
    buffer = pad_hidden(buffer, padded_m)
    if route_dot_values is not None:
        route_dot_values = pad_hidden(route_dot_values, padded_m)
    manual_axis_type = jax.typeof(buffer).manual_axis_type
    num_m_tiles = padded_m // block_m
    if route_dot_values is None:
        out_type = ()
    else:
        out_type = jax.ShapeDtypeStruct(
            (num_m_tiles, B, S, K), jnp.float32, manual_axis_type=manual_axis_type
        )

    bytes_accessed = (
        2 * cost_estimate_num_rows * M * buffer.dtype.itemsize
        + top_k_weights.size * top_k_weights.dtype.itemsize
        + positions.size * positions.dtype.itemsize
    )
    if route_dot_values is not None:
        bytes_accessed += (
            cost_estimate_num_rows * M * route_dot_values.dtype.itemsize
            + num_m_tiles * B * S * K * jnp.dtype(jnp.float32).itemsize
        )
    compute_route_dot = route_dot_values is not None
    routes_per_block = weight_routes_per_block()
    name = "weight_expert_rows_mosaic_gpu"
    if compute_route_dot:
        name += "_route_dot"
    buffer_ref = jax.new_ref(buffer)
    args = (buffer_ref, top_k_weights, positions)
    if route_dot_values is not None:
        args = (*args, route_dot_values)
    launch = pl.kernel(
        functools.partial(
            weight_expert_rows_kernel,
            S=S,
            K=K,
            extent=extent,
            block_m=block_m,
            num_routes=B * S * K,
            routes_per_block=routes_per_block,
            compute_route_dot=compute_route_dot,
        ),
        out_type=out_type,
        mesh=plgpu.Mesh(
            grid=(num_m_tiles, pl.cdiv(B * S * K, routes_per_block)),
            grid_names=("m", "route"),
            kernel_name=name,
        ),
        compiler_params=COMBINE_COMPILER_PARAMS,
        cost_estimate=pl.CostEstimate(
            flops=(3 if compute_route_dot else 1) * cost_estimate_num_rows * M,
            transcendentals=0,
            bytes_accessed=bytes_accessed,
        ),
        name=name,
    )
    result = launch(*args)
    if route_dot_values is None:
        return buffer_ref[...][..., :M]
    route_dot_partial = result
    route_dot = route_dot_partial.sum(axis=0).astype(top_k_weights_dtype)
    return (buffer_ref[...][..., :M], jnp.where(positions < extent, route_dot, 0))


def combine_transpose_mosaic_gpu(
    grad_out: jaxtyping.Float[jax.Array, "B S M"],
    positions: jaxtyping.Int[jax.Array, "B S K"],
    extent: int,
    cost_estimate_num_rows: int | None = None,
    group_sizes: jaxtyping.Int[jax.Array, "B"] | None = None,
) -> jaxtyping.Float[jax.Array, "P M"]:
    """Scatter token cotangents into expert-row order.

    With ``group_sizes`` a persistent grid strides over the live token prefixes;
    otherwise one program handles each token and hidden tile.
    """
    if grad_out.ndim != 3:
        raise ValueError(f"combine_transpose expects [B, S, M], got {grad_out.shape}")
    if grad_out.shape[:2] != positions.shape[:2]:
        raise ValueError(
            "combine_transpose expects matching batch and sequence dims, got "
            f"{grad_out.shape[:2]} and {positions.shape[:2]}"
        )
    B, S, K = positions.shape
    M = grad_out.shape[-1]
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = B * S * K
    positions = jnp.where(positions < extent, positions, extent).astype(jnp.int32)

    block_m = (
        WIDE_HIDDEN_BLOCK_SIZE
        if M % WIDE_HIDDEN_BLOCK_SIZE == 0
        else hidden_block_size(M)
    )
    padded_m = pl.cdiv(M, block_m) * block_m
    grad_out = pad_hidden(grad_out, padded_m)
    persistent = group_sizes is not None
    inputs = (grad_out, positions)
    if persistent:
        inputs = (*inputs, jnp.clip(group_sizes, 0, S))
    varying_axes = union_varying_axes(*inputs)
    grad_axis_type = jax.typeof(grad_out).manual_axis_type
    manual_axis_type = grad_axis_type.update(
        varying=varying_axes,
        unreduced=grad_axis_type.unreduced - varying_axes,
        reduced=grad_axis_type.reduced - varying_axes,
    )
    bytes_accessed = (
        B * S + cost_estimate_num_rows
    ) * M * grad_out.dtype.itemsize + positions.size * positions.dtype.itemsize
    name = "moe_combine_mosaic_gpu_transpose"
    launch = pl.kernel(
        functools.partial(
            combine_transpose_kernel,
            S=S,
            K=K,
            extent=extent,
            block_m=block_m,
            persistent=persistent,
        ),
        out_type=jax.ShapeDtypeStruct(
            (extent, padded_m), grad_out.dtype, manual_axis_type=manual_axis_type
        ),
        mesh=plgpu.Mesh(
            grid=(
                persistent_grid(TRANSPOSE_CTAS_PER_SM)
                if persistent
                else (padded_m // block_m, B * S)
            ),
            grid_names=("program",) if persistent else ("m", "token"),
            kernel_name=name,
        ),
        compiler_params=DISJOINT_IO_COMPILER_PARAMS,
        cost_estimate=pl.CostEstimate(
            flops=0, transcendentals=0, bytes_accessed=bytes_accessed
        ),
        name=name,
    )
    return launch(*inputs)[..., :M]


def combine_mosaic_gpu(
    buffer: jaxtyping.Float[jax.Array, "P M"],
    top_k_weights: jaxtyping.Float[jax.Array, "B S K"] | None,
    positions: jaxtyping.Int[jax.Array, "B S K"],
    activations_in_float32: bool,
    group_sizes: jaxtyping.Int[jax.Array, "B"] | None = None,
) -> jaxtyping.Float[jax.Array, "B S M"]:
    """Sum each token's weighted expert rows; see ``xla.combine``.

    Without ``group_sizes`` one program handles each token; the hardware
    scheduler balances tokens whose routes are mostly dead. With ``group_sizes``
    a persistent grid strides over the live token prefixes only and leaves later
    rows unwritten, so callers must not read them.

    This forward has no VJP: ``maximoe.kernels.ragged_dot.RaggedDotCombine`` composes
    the backward from :func:`combine_transpose_mosaic_gpu` and
    :func:`weight_expert_rows_mosaic_gpu`. The output varies on every manual
    axis any operand varies on, which keeps variation introduced by route
    metadata visible to the surrounding program.
    """
    if buffer.ndim != 2:
        raise ValueError(f"combine expects [P, M], got {buffer.shape}")
    extent, M = buffer.shape
    B, S, K = positions.shape
    persistent = group_sizes is not None

    varying_axes = union_varying_axes(buffer, top_k_weights, positions, group_sizes)
    buffer = pcast_to_varying(buffer, varying_axes)
    positions = pcast_to_varying(positions, varying_axes)
    output_dtype = jnp.float32 if activations_in_float32 else buffer.dtype
    live = positions < extent
    positions = jnp.where(live, positions, extent).astype(jnp.int32)
    args = (buffer, positions)
    if top_k_weights is not None:
        top_k_weights = pcast_to_varying(top_k_weights, varying_axes)
        top_k_weights = jnp.where(live, top_k_weights, 0)
        args = (*args, top_k_weights)
    if persistent:
        args = (*args, pcast_to_varying(jnp.clip(group_sizes, 0, S), varying_axes))

    block_m = (
        WIDE_HIDDEN_BLOCK_SIZE
        if M % WIDE_HIDDEN_BLOCK_SIZE == 0
        else hidden_block_size(M)
    )
    padded_m = pl.cdiv(M, block_m) * block_m
    args = (pad_hidden(buffer, padded_m), *args[1:])
    name = "moe_combine_mosaic_gpu_fwd"
    bytes_accessed = (
        B * S * K * M * buffer.dtype.itemsize
        + B * S * M * jnp.dtype(output_dtype).itemsize
    )
    if top_k_weights is not None:
        bytes_accessed += top_k_weights.size * top_k_weights.dtype.itemsize
    launch = pl.kernel(
        functools.partial(
            combine_fwd_kernel,
            K=K,
            extent=extent,
            block_m=block_m,
            weighted=top_k_weights is not None,
            persistent=persistent,
        ),
        out_type=jax.ShapeDtypeStruct(
            (B, S, padded_m),
            output_dtype,
            manual_axis_type=jax.typeof(args[0]).manual_axis_type,
        ),
        mesh=plgpu.Mesh(
            grid=persistent_grid(COMBINE_FWD_CTAS_PER_SM) if persistent else (B * S,),
            grid_names=("program",) if persistent else ("token",),
            kernel_name=name,
        ),
        compiler_params=DISJOINT_IO_COMPILER_PARAMS,
        cost_estimate=pl.CostEstimate(
            flops=B * S * K * M * (2 if top_k_weights is not None else 1),
            transcendentals=0,
            bytes_accessed=bytes_accessed,
        ),
        name=name,
    )
    return launch(*args)[..., :M]


# JaxPP task dedup does not inspect the jaxprs stored by ``mpmd_map``. These JIT
# boundaries let pipeline stages reuse one traced and compiled kernel.
# See docs/compilation.md ("Cache misses") in the JaxPP repository.
combine_mosaic_gpu = jax.jit(combine_mosaic_gpu, static_argnums=(3,))  # type: ignore
combine_transpose_mosaic_gpu = jax.jit(
    combine_transpose_mosaic_gpu, static_argnums=(2, 3)
)  # type: ignore
weight_expert_rows_mosaic_gpu = jax.jit(
    weight_expert_rows_mosaic_gpu, donate_argnums=(0,), static_argnums=(4,)
)  # type: ignore


# Permutation kernel inputs are read-only and outputs are disjoint, satisfying
# the no read/write overlap contract required by this compiler option.
POSTPROCESS_COMPILER_PARAMS = plgpu.CompilerParams(unsafe_no_auto_barriers=True)


def permute_scatter_kernel(
    inputs_ref,
    positions_ref,
    out_ref,
    *,
    routes: int,
    extent: int,
    block_m: int,
    num_m_tiles: int,
):
    token = lax.axis_index("token")
    positions = tuple(positions_ref[token, route] for route in range(routes))
    live = tuple(position < extent for position in positions)
    any_live = live[0]
    for route_live in live[1:]:
        any_live = jnp.logical_or(any_live, route_live)

    @pl.when(any_live)
    def scatter_live_token():
        def tile_loop(m_tile, unused):
            columns = pl.ds(m_tile * block_m, block_m)
            layout = strided_layout(block_m, inputs_ref.dtype)
            values = plgpu.load(inputs_ref.at[token, columns], (), layout=layout)
            for route in range(routes):

                @pl.when(live[route])
                def store_route():
                    out_ref.at[positions[route], columns][...] = values

            return unused

        lax.fori_loop(0, num_m_tiles, tile_loop, None)


def gather_sum_kernel(
    buffer_ref,
    positions_ref,
    out_ref,
    *,
    tokens: int,
    routes: int,
    extent: int,
    block_m: int,
    num_m_tiles: int,
    batched: bool,
    tiled: bool,
):
    linear_token = lax.axis_index("token")
    batch, token = divmod(linear_token, tokens)
    positions = tuple(
        (positions_ref[batch, token, route] if batched else positions_ref[token, route])
        for route in range(routes)
    )
    live = tuple(position < extent for position in positions)

    def tile_loop(m_tile, unused):
        columns = pl.ds(m_tile * block_m, block_m)
        layout = strided_layout(block_m, buffer_ref.dtype)
        acc = plgpu.layout_cast(jnp.zeros((block_m,), dtype=jnp.float32), layout)

        for route in range(routes):

            def add_route(route_acc):
                if batched:
                    values = plgpu.load(
                        buffer_ref.at[batch, positions[route], columns],
                        (),
                        layout=layout,
                    )
                else:
                    values = plgpu.load(
                        buffer_ref.at[positions[route], columns], (), layout=layout
                    )
                return route_acc + values.astype(jnp.float32)

            acc = lax.cond(live[route], add_route, lambda route_acc: route_acc, acc)

        if batched:
            out_ref.at[batch, token, columns][...] = acc.astype(out_ref.dtype)
        else:
            out_ref.at[token, columns][...] = acc.astype(out_ref.dtype)
        return unused

    if tiled:
        tile_loop(lax.axis_index("m"), None)
    else:
        lax.fori_loop(0, num_m_tiles, tile_loop, None)


def permute_buffer_token_id_fwd_mosaic_gpu(
    inputs: jaxtyping.Float[jax.Array, "T M"],
    positions: jaxtyping.Int[jax.Array, "T K"],
    packed_size: int,
) -> jaxtyping.Float[jax.Array, "P M"]:
    """Scatter token rows into their packed positions."""
    tokens, model_size = inputs.shape
    position_tokens, routes = positions.shape
    if position_tokens != tokens:
        raise ValueError(
            f"positions.shape[0]={position_tokens} must equal inputs.shape[0]={tokens}"
        )

    varying_axes = union_varying_axes(inputs, positions)
    inputs = pcast_to_varying(inputs, varying_axes)
    positions = pcast_to_varying(positions, varying_axes)

    block_m = (
        model_size
        if model_size == 2 * WIDE_HIDDEN_BLOCK_SIZE
        else hidden_block_size(model_size)
    )
    padded_model_size = pl.cdiv(model_size, block_m) * block_m
    inputs = pad_hidden(inputs, padded_model_size)
    manual_axis_type = jax.typeof(inputs).manual_axis_type
    out_type = jax.ShapeDtypeStruct(
        (packed_size, padded_model_size),
        inputs.dtype,
        manual_axis_type=manual_axis_type,
    )

    name = "permute_buffer_fwd_mosaic_gpu"
    launch = pl.kernel(
        functools.partial(
            permute_scatter_kernel,
            routes=routes,
            extent=packed_size,
            block_m=block_m,
            num_m_tiles=padded_model_size // block_m,
        ),
        out_type=out_type,
        mesh=plgpu.Mesh(grid=(tokens,), grid_names=("token",), kernel_name=name),
        compiler_params=POSTPROCESS_COMPILER_PARAMS,
        cost_estimate=pl.CostEstimate(
            flops=0,
            transcendentals=0,
            bytes_accessed=(tokens * model_size + tokens * routes * model_size)
            * inputs.dtype.itemsize,
        ),
        name=name,
    )
    return launch(inputs, positions)[..., :model_size]


def gather_sum_mosaic_gpu(
    buffer: jaxtyping.Float[jax.Array, "P M"] | jaxtyping.Float[jax.Array, "B P M"],
    positions: jaxtyping.Int[jax.Array, "T K"] | jaxtyping.Int[jax.Array, "B T K"],
) -> jaxtyping.Float[jax.Array, "T M"] | jaxtyping.Float[jax.Array, "B T M"]:
    """Gather buffer rows and sum over routes; see ``xla.gather_sum``."""
    if buffer.ndim == 3:
        batch_size, extent, model_size = buffer.shape
        position_batch, tokens, routes = positions.shape
        if position_batch != batch_size:
            raise ValueError(
                f"positions.shape[0]={position_batch} must equal "
                f"buffer.shape[0]={batch_size}"
            )
        batched = True
        output_shape = (batch_size, tokens, model_size)
    elif buffer.ndim == 2:
        extent, model_size = buffer.shape
        tokens, routes = positions.shape
        batch_size = 1
        batched = False
        output_shape = (tokens, model_size)
    else:
        raise ValueError(f"gather_sum buffer must have rank 2 or 3, got {buffer.ndim}")

    varying_axes = union_varying_axes(buffer, positions)
    buffer = pcast_to_varying(buffer, varying_axes)
    positions = pcast_to_varying(positions, varying_axes)

    tiled = batch_size * tokens <= 4096
    block_m = (
        model_size
        if tiled and model_size == 2 * WIDE_HIDDEN_BLOCK_SIZE
        else (
            WIDE_HIDDEN_BLOCK_SIZE
            if model_size % WIDE_HIDDEN_BLOCK_SIZE == 0
            else hidden_block_size(model_size)
        )
    )
    padded_model_size = pl.cdiv(model_size, block_m) * block_m
    buffer = pad_hidden(buffer, padded_model_size)
    output_shape = (*output_shape[:-1], padded_model_size)
    name = "gather_sum_batched_mosaic_gpu" if batched else "gather_sum_mosaic_gpu"
    if tiled:
        name += "_tiled"
    launch = pl.kernel(
        functools.partial(
            gather_sum_kernel,
            tokens=tokens,
            routes=routes,
            extent=extent,
            block_m=block_m,
            num_m_tiles=padded_model_size // block_m,
            batched=batched,
            tiled=tiled,
        ),
        out_type=jax.ShapeDtypeStruct(
            output_shape,
            buffer.dtype,
            manual_axis_type=jax.typeof(buffer).manual_axis_type,
        ),
        mesh=plgpu.Mesh(
            grid=(
                (batch_size * tokens, padded_model_size // block_m)
                if tiled
                else (batch_size * tokens,)
            ),
            grid_names=("token", "m") if tiled else ("token",),
            kernel_name=name,
        ),
        compiler_params=POSTPROCESS_COMPILER_PARAMS,
        cost_estimate=pl.CostEstimate(
            flops=batch_size * tokens * routes * model_size,
            transcendentals=0,
            bytes_accessed=(
                batch_size
                * (tokens * routes * model_size + tokens * model_size)
                * buffer.dtype.itemsize
            ),
        ),
        name=name,
    )
    return launch(buffer, positions)[..., :model_size]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def permute_buffer_token_id_mosaic_gpu(
    inputs: jaxtyping.Float[jax.Array, "T M"],
    positions: jaxtyping.Int[jax.Array, "T K"],
    packed_size: int,
) -> jaxtyping.Float[jax.Array, "P M"]:
    """Scatter token rows into their packed positions."""
    return permute_buffer_token_id_fwd_mosaic_gpu(inputs, positions, packed_size)


@jax.named_call
def permute_buffer_token_id_fwd_rule(inputs, positions, packed_size):
    buffer = permute_buffer_token_id_fwd_mosaic_gpu(inputs, positions, packed_size)
    return buffer, positions


@jax.named_call
def permute_buffer_token_id_bwd_rule(packed_size, positions, buffer_grad):
    del packed_size
    grad_inputs = gather_sum_mosaic_gpu(buffer_grad, positions)
    return grad_inputs, None


permute_buffer_token_id_mosaic_gpu.defvjp(
    permute_buffer_token_id_fwd_rule, permute_buffer_token_id_bwd_rule
)

# JaxPP task dedup does not inspect the jaxprs stored by ``mpmd_map``. This JIT
# boundary lets pipeline stages reuse one traced and compiled kernel.
# See docs/compilation.md ("Cache misses") in the JaxPP repository.
permute_buffer_token_id_mosaic_gpu = jax.jit(
    permute_buffer_token_id_mosaic_gpu, static_argnums=(2,)
)  # type: ignore


# Public operation names shared with the ``xla`` module. They are assigned only
# after custom VJP registration and JIT wrapping, so the package dispatch calls
# the exact same callable objects rather than forwarding wrappers.
combine = combine_mosaic_gpu
combine_transpose = combine_transpose_mosaic_gpu
weight_expert_rows = weight_expert_rows_mosaic_gpu
permute = permute_buffer_token_id_mosaic_gpu
gather_sum = gather_sum_mosaic_gpu
