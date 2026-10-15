# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright 2025 The JAX Authors.
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

"""Shared helpers for MaxiMoE ragged-dot kernels."""

import dataclasses
import enum
import math
import os
from typing import Self

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm

# Prefused weights keep one checkpoint-stable layout across backends. SM100's
# fused epilogue fixes the block width; SM90 consumes the same layout in software.
SWIGLU_EPILOGUE_BLOCK = 32


class GridMinorDimension(enum.IntEnum):
    """Fast-changing dimension in the persistent planar-snake schedule."""

    M = 0
    N = 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Aligned group metadata; only the first ``num_tasks`` entries are live."""

    group_id: jax.Array
    block_start: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array
    num_tasks: jax.Array
    num_rows: jax.Array

    @classmethod
    def create_aligned(
        cls, group_sizes: jax.Array, tile: int, tid_size: int, extent: int
    ) -> Self:
        """Adapted from Tokamax's ``GroupInfo.create_aligned``.

        Adds TMA bounds clamping and live task/row counts. Assumes nonnegative
        group sizes and fixes alignment at eight rows.
        """
        group_offsets = jnp.cumulative_sum(group_sizes, include_initial=True)
        group_starts = group_offsets[:-1]
        group_ends = group_starts + group_sizes
        group_starts_aligned = group_starts // 8 * 8
        group_num_blocks = jnp.where(
            group_sizes == 0, 0, pl.cdiv(group_ends - group_starts_aligned, tile)
        )
        group_start_tasks = jnp.cumulative_sum(group_num_blocks, include_initial=True)
        # Keep one authoritative prefix sum for both lookup and the runtime bound.
        group_start_tasks = jax.lax.optimization_barrier(group_start_tasks)
        group_end_tasks = group_start_tasks[:-1] + group_num_blocks
        task_idx = jnp.arange(tid_size, dtype=jnp.int32)
        task_group_idx = jnp.searchsorted(
            group_end_tasks[:-1], task_idx, side="right", method="compare_all"
        )
        task_block_idx = task_idx - group_start_tasks[task_group_idx]
        task_starts = group_starts_aligned[task_group_idx] + tile * task_block_idx
        # Keep full TMA tiles in bounds without assigning shifted rows twice.
        block_start = jnp.minimum(task_starts, extent - tile)
        task_valid_starts = jnp.maximum(task_starts, group_starts[task_group_idx])
        task_valid_ends = jnp.minimum(block_start + tile, group_ends[task_group_idx])
        task_valid_size = jnp.maximum(0, task_valid_ends - task_valid_starts)
        return cls(
            group_id=task_group_idx,
            block_start=block_start,
            start_within_block=task_valid_starts - block_start,
            actual_size=task_valid_size,
            num_tasks=group_start_tasks[-1],
            num_rows=group_offsets[-1],
        )


def tile_swizzle_transforms(
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    *,
    tiling_prefix: tuple[int, ...] = (8,),
) -> tuple[plgpu.TilingTransform, plgpu.SwizzleTransform]:
    element_bits = jnp.dtype(dtype).itemsize * 8
    swizzle = plgpu.find_swizzle(shape[-1] * element_bits)
    tiling = (*tiling_prefix, 8 * swizzle // element_bits)
    return plgpu.TilingTransform(tiling), plgpu.SwizzleTransform(swizzle)


def compiler_params_from_env() -> plgpu.CompilerParams | None:
    """Read Mosaic in-kernel profiler settings from the environment."""
    profile_dir = os.environ.get("RAGGED_DOT_PROFILE_DIR")
    if not profile_dir:
        return None
    os.makedirs(profile_dir, exist_ok=True)
    return plgpu.CompilerParams(
        profile_space=int(os.environ.get("RAGGED_DOT_PROFILE_SPACE", "1024")),
        profile_dir=profile_dir,
    )


def output_manual_axis_type(*inputs):
    """Keep the first input's axis type and union all varying axes."""
    manual_axis_type = jax.typeof(inputs[0]).manual_axis_type
    input_varying = frozenset().union(
        *(jax.typeof(x).manual_axis_type.varying for x in inputs)
    )
    return manual_axis_type.update(varying=input_varying)


def store_rows(smem_ref, gmem_ref, row, num_rows):
    """Adapt JAX's power-of-two TMA row stores to use one commit group.

    The final wait permits SMEM reuse without waiting for global stores.
    """
    plgpu.commit_smem()
    remaining_rows = smem_ref.shape[0]
    while remaining_rows > 0:
        const_rows = 1 << int(math.log2(remaining_rows))
        remaining_rows //= 2

        @pl.when(num_rows & const_rows != 0)
        def _():
            rows = pl.ds(row, const_rows)
            plgpu.copy_smem_to_gmem(
                smem_ref.at[rows], gmem_ref.at[rows], commit_group=False
            )

        row += num_rows & const_rows
    plgpu.commit_smem_to_gmem_group()
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def trap_if_rows_exceed(num_rows, max_rows: int):
    """Fail before an invalid row extent can address beyond static buffers."""

    @pl.when(num_rows > jnp.int32(max_rows))
    def _():
        @plgpu.inline_mgpu(arg_types=())
        def trap(ctx):
            del ctx
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"), [], "trap;", "", has_side_effects=True
            )

        trap()
