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

"""SM90 Mosaic-GPU ragged-dot kernels.

This adapts Tokamax's SM90 ragged-dot kernel to MaxiMoE. Compared with JAX's
original Hopper example it uses an eight-row-aligned compact task
table, handles empty groups without issuing duplicate GEMMs, derives the SM
count from the target device, and attaches an XLA cost estimate.

The kernel uses native ``rhs[G, K, N]`` weights for both forward and dX. The
dX operand transpose is a shared-memory descriptor transform; it does not
materialize an HBM transpose.

The warp-specialized pipeline and tiled epilogue adapt JAX's
``hopper_matmul_mgpu`` example. The dW kernel's grouped persistent loop adapts
JAX's ``transposed_ragged_dot_mgpu`` example.

The cooperative dW path adopts DeepGEMM's broad Hopper structure: one TMA
producer warpgroup feeds two math warpgroups, with a wide-N WGMMA tile and a
double-buffered TMA epilogue.
"""

import dataclasses
import functools
import math
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend

from maximoe.kernels import gpu
from maximoe.kernels.ragged_dot import common, swiglu

SWIGLU_EPILOGUE_BLOCK = common.SWIGLU_EPILOGUE_BLOCK
GridMinorDimension = common.GridMinorDimension
tile_swizzle_transforms = common.tile_swizzle_transforms


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    """SM90 WGMMA tile and persistent scheduler configuration."""

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    grid_minor_dim: GridMinorDimension = GridMinorDimension.N
    grid_tile_width: int = 8
    num_compute_wgs: int = 1
    wg_dimension: GridMinorDimension = GridMinorDimension.N
    epilogue_m: int = 64
    epilogue_n: int = 64
    memory_registers: int = 40
    delay_release: int = 0
    sm_margin: int | None = None
    cluster_n: int = 1

    def __str__(self) -> str:
        return (
            f"m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_s{self.num_stages}"
            f"_r{self.grid_minor_dim.name.lower()}{self.grid_tile_width}"
            f"_wg{self.num_compute_wgs}{self.wg_dimension.name.lower()}"
            f"_e{self.epilogue_m}x{self.epilogue_n}"
            f"_mr{self.memory_registers}_d{self.delay_release}"
            f"_sm{'env' if self.sm_margin is None else self.sm_margin}"
            + (f"_cn{self.cluster_n}" if self.cluster_n != 1 else "")
        )


def device_sm_count() -> int:
    abstract_device = jax.sharding.get_abstract_mesh().abstract_device
    if abstract_device:
        return abstract_device.num_cores
    return backend.get_default_device().core_count


def configured_sm_margin(config: TuningConfig) -> int:
    if env_margin := os.environ.get("RAGGED_DOT_SM_MARGIN"):
        return int(env_margin)
    if config.sm_margin is not None:
        return config.sm_margin
    return gpu.MOSAIC_SM_MARGIN


def mosaic_compiler_params(
    *,
    lowering_semantics: plgpu.LoweringSemantics,
    unsafe_no_auto_barriers: bool = False,
) -> plgpu.CompilerParams:
    """Build compiler parameters, optionally enabling device-side tracing."""
    profile_dir = os.environ.get("RAGGED_DOT_PROFILE_DIR", "")
    profile_space = int(os.environ.get("RAGGED_DOT_PROFILE_SPACE", "0"))
    if (not profile_dir and profile_space != 0) or (profile_dir and profile_space <= 0):
        raise ValueError(
            "RAGGED_DOT_PROFILE_DIR and a positive RAGGED_DOT_PROFILE_SPACE "
            "must either both be set or both be unset."
        )
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    return plgpu.CompilerParams(
        approx_math=True,
        unsafe_no_auto_barriers=unsafe_no_auto_barriers,
        lowering_semantics=lowering_semantics,
        profile_space=profile_space,
        profile_dir=profile_dir,
        profile_trace_scope=plgpu.TraceScope.WARPGROUP,
    )


def block_spec(
    shape,
    dtype,
    index_map,
    *,
    delay_release: int = 0,
    collective_axes: tuple[str, ...] = (),
) -> plgpu.BlockSpec:
    return plgpu.BlockSpec(
        shape,
        index_map,
        transforms=tile_swizzle_transforms(tuple(shape), dtype),
        delay_release=delay_release,
        collective_axes=collective_axes,
    )


def validate(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: TuningConfig,
    transpose_rhs: bool,
) -> tuple[int, int, int, int]:
    if lhs.ndim != 2 or rhs.ndim != 3 or group_sizes.ndim != 1:
        raise ValueError(
            "Expected lhs[M,K], rhs[G,K,N], and group_sizes[G], got "
            f"{lhs.shape=}, {rhs.shape=}, {group_sizes.shape=}."
        )
    if lhs.dtype != rhs.dtype:
        raise ValueError(
            f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}."
        )
    if lhs.dtype not in (jnp.bfloat16, jnp.float16):
        raise NotImplementedError(
            f"SM90 ragged_dot supports bfloat16/float16, got {lhs.dtype}."
        )

    m, k_contract = lhs.shape
    num_groups = rhs.shape[0]
    if num_groups == 0:
        raise ValueError("ragged_dot requires at least one group.")
    if transpose_rhs:
        n_out, rhs_k = rhs.shape[1:]
    else:
        rhs_k, n_out = rhs.shape[1:]
    if group_sizes.shape != (num_groups,):
        raise ValueError(
            f"group_sizes must have shape {(num_groups,)}, got {group_sizes.shape}."
        )
    if rhs_k != k_contract:
        raise ValueError(
            f"Contracting dimensions differ: lhs {k_contract}, rhs {rhs_k}."
        )
    if m < config.block_m or m % config.block_m:
        raise ValueError(f"{m=} must be a positive multiple of {config.block_m=}.")
    if n_out % config.block_n:
        raise ValueError(f"{n_out=} must be divisible by {config.block_n=}.")
    if k_contract % config.block_k:
        raise ValueError(f"{k_contract=} must be divisible by {config.block_k=}.")
    if config.block_m % 8:
        raise ValueError(f"{config.block_m=} must be divisible by 8.")
    if config.num_stages < 1:
        raise ValueError(f"{config.num_stages=} must be positive.")
    if config.cluster_n not in (1, 2):
        raise ValueError(f"cluster_n must be 1 or 2, got {config.cluster_n}.")
    if n_out // config.block_n % config.cluster_n:
        raise ValueError(
            f"The N grid must be divisible by cluster_n, got "
            f"{n_out // config.block_n} and {config.cluster_n}."
        )
    if config.cluster_n == 2 and config.num_compute_wgs != 2:
        raise ValueError("A two-CTA N cluster requires two compute warpgroups.")
    if config.grid_tile_width < 1:
        raise ValueError(f"{config.grid_tile_width=} must be positive.")
    if config.num_compute_wgs not in (1, 2):
        raise ValueError(
            f"num_compute_wgs must be 1 or 2, got {config.num_compute_wgs}."
        )
    if config.num_compute_wgs == 2:
        split_tile = (
            config.block_m
            if config.wg_dimension == GridMinorDimension.M
            else config.block_n
        )
        if split_tile % 2:
            raise ValueError(
                "A two-WG split requires an even tile in the selected "
                f"dimension, got {split_tile}."
            )
        wg_m = config.block_m // (
            2 if config.wg_dimension == GridMinorDimension.M else 1
        )
        wg_n = config.block_n // (
            2 if config.wg_dimension == GridMinorDimension.N else 1
        )
        if wg_m % config.epilogue_m or wg_n % config.epilogue_n:
            raise ValueError(
                "Warp-specialized tiles must be divisible by the epilogue tile: "
                f"wg_tile=({wg_m}, {wg_n}), "
                f"epilogue=({config.epilogue_m}, {config.epilogue_n})."
            )
        if config.memory_registers < 16:
            raise ValueError(
                f"memory_registers must be at least 16, got {config.memory_registers}."
            )
        if not 0 <= config.delay_release < config.num_stages:
            raise ValueError(
                "delay_release must be in [0, num_stages), got "
                f"{config.delay_release} for {config.num_stages} stages."
            )
    return m, k_contract, n_out, num_groups


@functools.partial(
    jax.jit,
    static_argnums=(3, 4),
    static_argnames=("output_dtype", "cost_estimate_num_rows"),
)
def ragged_dot_kernel(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: TuningConfig,
    transpose_rhs: bool = False,
    *,
    output_dtype=None,
    cost_estimate_num_rows: int | None = None,
) -> jax.Array:
    """Compute a mode-1 ragged GEMM on SM90.

    With ``transpose_rhs=False`` this computes ``[M,K] @ [G,K,N]``.  With
    ``transpose_rhs=True`` it computes ``[M,K] @ [G,N,K]^T`` without changing
    the physical RHS layout. Group sizes are trusted route metadata: they must
    be nonnegative and sum to at most M. Rows beyond that sum are not written
    and must be ignored by capacity-buffer callers. The WGMMA accumulator is
    rounded once to ``output_dtype`` (default ``lhs.dtype``) when it is stored.
    """
    m, k_contract, n_out, num_groups = validate(
        lhs, rhs, group_sizes, config, transpose_rhs
    )
    num_sms = device_sm_count()
    sm_margin = configured_sm_margin(config)
    if not 0 <= sm_margin < num_sms:
        raise ValueError(
            f"RAGGED_DOT_SM_MARGIN must be in [0, {num_sms - 1}], got {sm_margin}."
        )
    active_sms = num_sms - sm_margin
    if active_sms % config.cluster_n:
        # Preserve the requested margin when the grid cannot pair every CTA.
        config = dataclasses.replace(config, cluster_n=1)
    output_dtype = jnp.dtype(lhs.dtype if output_dtype is None else output_dtype)
    if output_dtype not in (jnp.bfloat16, jnp.float16, jnp.float32):
        raise NotImplementedError(
            f"SM90 ragged_dot output dtype must be bf16, f16, or f32, got "
            f"{output_dtype}."
        )
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = m
    if not 0 <= cost_estimate_num_rows <= m:
        raise ValueError(
            f"Cost-estimate rows must be in [0, {m}], got {cost_estimate_num_rows}."
        )

    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    n_iters = n_out // block_n
    k_iters = k_contract // block_k
    max_tasks = m // block_m + num_groups - 1
    group_info = common.GroupInfo.create_aligned(
        group_sizes.astype(jnp.int32), block_m, max_tasks, m
    )

    def kernel(
        lhs_gmem,
        rhs_gmem,
        task_group_ids_gmem,
        task_block_starts_gmem,
        task_row_offsets_gmem,
        task_row_counts_gmem,
        num_tasks_gmem,
        num_rows_gmem,
        out_gmem,
    ):
        num_tasks = num_tasks_gmem[()]
        num_rows = num_rows_gmem[()]
        common.trap_if_rows_exceed(num_rows, m)
        linear_grid = num_tasks * jnp.int32(n_iters)
        axis_index = lax.axis_index("sm")
        axis_size = lax.axis_size("sm")

        @plgpu.nd_loop((linear_grid,), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            # Constructing the step directly avoids a dynamic-grid remainder in
            # NDLoopInfo when the number of live tasks changes at runtime.
            step = loop_info.local_index * axis_size + axis_index
            task_index, n_index = plgpu.planar_snake(
                step,
                (num_tasks, n_iters),
                config.grid_minor_dim,
                config.grid_tile_width,
            )
            group_id = task_group_ids_gmem[task_index]
            block_start = pl.multiple_of(task_block_starts_gmem[task_index], 8)
            row_offset = task_row_offsets_gmem[task_index]
            row_count = task_row_counts_gmem[task_index]

            def compute_acc(acc):
                def pipeline_body(_, lhs_smem, rhs_smem):
                    rhs_operand = rhs_smem
                    if transpose_rhs:
                        rhs_operand = plgpu.transpose_ref(rhs_smem, (1, 0))
                    plgpu.wgmma(acc, lhs_smem, rhs_operand)
                    plgpu.wgmma_wait(1 if config.num_stages > 1 else 0)

                lhs_spec = block_spec(
                    (pl.Element(block_m), block_k),
                    lhs_gmem.dtype,
                    lambda ki: (block_start, ki),
                    delay_release=1 if config.num_stages > 1 else 0,
                )
                if transpose_rhs:
                    rhs_shape = (block_n, block_k)

                    def rhs_index_map(ki):
                        return n_index, ki

                else:
                    rhs_shape = (block_k, block_n)

                    def rhs_index_map(ki):
                        return ki, n_index

                rhs_spec = block_spec(
                    rhs_shape,
                    rhs_gmem.dtype,
                    rhs_index_map,
                    delay_release=1 if config.num_stages > 1 else 0,
                )
                plgpu.emit_pipeline(
                    pipeline_body,
                    grid=(k_iters,),
                    in_specs=(lhs_spec, rhs_spec),
                    max_concurrent_steps=config.num_stages,
                )(lhs_gmem, rhs_gmem.at[group_id])
                return acc[...]

            acc = pl.run_scoped(compute_acc, plgpu.ACC((block_m, block_n)))
            out_smem_type = plgpu.SMEM(
                (block_m, block_n),
                out_gmem.dtype,
                transforms=tile_swizzle_transforms(
                    (block_m, block_n), out_gmem.dtype, tiling_prefix=(1,)
                ),
            )

            @functools.partial(pl.run_scoped, out_smem=out_smem_type)
            def epilogue(out_smem):
                out_smem[...] = acc.astype(out_smem.dtype)
                plgpu.commit_smem()

                smem_start = row_offset
                remaining_rows = block_m
                while remaining_rows > 0:
                    const_rows = 1 << int(math.log2(remaining_rows))
                    remaining_rows //= 2

                    @pl.when(row_count & const_rows != 0)
                    def copy_rows():
                        plgpu.copy_smem_to_gmem(
                            out_smem.at[pl.ds(smem_start, const_rows)],
                            out_gmem.at[
                                pl.ds(block_start + smem_start, const_rows),
                                pl.ds(n_index * block_n, block_n),
                            ],
                            commit_group=False,
                        )

                    smem_start += row_count & const_rows
                plgpu.commit_smem_to_gmem_group()
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    def kernel_warp_specialized(
        lhs_gmem,
        rhs_gmem,
        task_group_ids_gmem,
        task_block_starts_gmem,
        task_row_offsets_gmem,
        task_row_counts_gmem,
        num_tasks_gmem,
        num_rows_gmem,
        out_gmem,
    ):
        """One TMA producer WG feeding two compute WGs."""
        num_tasks = num_tasks_gmem[()]
        num_rows = num_rows_gmem[()]
        common.trap_if_rows_exceed(num_rows, m)
        n_cluster_iters = n_iters // config.cluster_n
        linear_grid = num_tasks * jnp.int32(n_cluster_iters)
        axis_index = lax.axis_index("sm")
        axis_size = lax.axis_size("sm")
        wg_tile_m = block_m // (
            config.num_compute_wgs if config.wg_dimension == GridMinorDimension.M else 1
        )
        wg_tile_n = block_n // (
            config.num_compute_wgs if config.wg_dimension == GridMinorDimension.N else 1
        )
        epilogue_m = config.epilogue_m
        epilogue_n = config.epilogue_n
        num_epilogue_tiles = (wg_tile_m // epilogue_m) * (wg_tile_n // epilogue_n)
        num_out_slots = min(2, num_epilogue_tiles)

        def get_pipeline(pipeline_body, compute_context):
            if transpose_rhs:
                rhs_shape = (block_n, block_k)

                def rhs_index_map(ki):
                    return 0, ki

            else:
                rhs_shape = (block_k, block_n)

                def rhs_index_map(ki):
                    return ki, 0

            return plgpu.emit_pipeline_warp_specialized(
                pipeline_body,
                grid=(k_iters,),
                memory_registers=config.memory_registers,
                in_specs=(
                    block_spec(
                        (block_m, block_k),
                        lhs_gmem.dtype,
                        lambda ki: (0, ki),
                        delay_release=config.delay_release,
                        collective_axes=("cluster",) if config.cluster_n == 2 else (),
                    ),
                    block_spec(
                        rhs_shape,
                        rhs_gmem.dtype,
                        rhs_index_map,
                        delay_release=config.delay_release,
                    ),
                ),
                wg_axis="wg",
                num_compute_wgs=config.num_compute_wgs,
                max_concurrent_steps=config.num_stages,
                compute_context=compute_context,
            )

        def ignore(*_, **__):
            return None

        pipeline_allocations = get_pipeline(ignore, ignore).get_allocations(
            lhs_gmem, rhs_gmem.at[0]
        )
        out_transforms = tile_swizzle_transforms(
            (epilogue_m, epilogue_n),
            out_gmem.dtype,
            # A unit row tile is required by the binary partial-row TMA
            # ladder, whose smallest descriptor stores one row.
            tiling_prefix=(1,),
        )

        @functools.partial(
            pl.run_scoped,
            pipeline_allocations=pipeline_allocations,
            out_smem=plgpu.SMEM(
                (config.num_compute_wgs, num_out_slots, epilogue_m, epilogue_n),
                out_gmem.dtype,
                transforms=out_transforms,
            ),
            collective_axes="wg",
        )
        def scoped(pipeline_allocations, out_smem):
            wg_index = lax.axis_index("wg")

            @plgpu.nd_loop((linear_grid,), collective_axes="sm")
            def mn_loop(loop_info: plgpu.NDLoopInfo):
                with jax.named_scope("scheduler"):
                    step = loop_info.local_index * axis_size + axis_index
                    task_index, n_cluster_index = plgpu.planar_snake(
                        step,
                        (num_tasks, n_cluster_iters),
                        config.grid_minor_dim,
                        config.grid_tile_width,
                    )
                    n_index = (
                        n_cluster_index * config.cluster_n + lax.axis_index("cluster")
                        if config.cluster_n == 2
                        else n_cluster_index
                    )
                    group_id = task_group_ids_gmem[task_index]
                    block_start = pl.multiple_of(task_block_starts_gmem[task_index], 8)
                    row_offset = task_row_offsets_gmem[task_index]
                    row_count = task_row_counts_gmem[task_index]
                if config.wg_dimension == GridMinorDimension.M:
                    wg_m_offset = wg_index * wg_tile_m
                    wg_m_slice = pl.ds(wg_index * wg_tile_m, wg_tile_m)
                    wg_n_slice = slice(None)
                else:
                    wg_m_offset = jnp.int32(0)
                    wg_m_slice = slice(None)
                    wg_n_slice = pl.ds(wg_index * wg_tile_n, wg_tile_n)

                def compute_context(eval_pipeline):
                    @functools.partial(
                        pl.run_scoped,
                        acc_ref=plgpu.ACC((wg_tile_m, wg_tile_n), jnp.float32),
                    )
                    def acc_scope(acc_ref):
                        with jax.named_scope("compute"):
                            eval_pipeline(acc_ref)
                        with jax.named_scope("epilogue"):
                            acc = acc_ref[...].astype(out_gmem.dtype)
                            # The persistent CTA reuses the two epilogue buffers
                            # on every task, so drain stores from the prior task.
                            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                            for epi_mi in range(wg_tile_m // epilogue_m):
                                for epi_ni in range(wg_tile_n // epilogue_n):
                                    epi_index = (
                                        epi_mi * (wg_tile_n // epilogue_n) + epi_ni
                                    )
                                    slot = epi_index % num_out_slots
                                    epi_m_start = epi_mi * epilogue_m
                                    epi_n_start = epi_ni * epilogue_n
                                    epi_m_slice = slice(
                                        epi_m_start, epi_m_start + epilogue_m
                                    )
                                    epi_n_slice = slice(
                                        epi_n_start, epi_n_start + epilogue_n
                                    )
                                    plgpu.wait_smem_to_gmem(
                                        num_out_slots - 1, wait_read_only=True
                                    )
                                    out_smem[wg_index, slot] = acc[
                                        epi_m_slice, epi_n_slice
                                    ]
                                    plgpu.commit_smem()

                                    epi_cta_m_start = wg_m_offset + epi_m_start
                                    copy_start = lax.max(row_offset, epi_cta_m_start)
                                    copy_end = lax.min(
                                        row_offset + row_count,
                                        epi_cta_m_start + epilogue_m,
                                    )
                                    copy_rows = lax.max(
                                        jnp.int32(0), copy_end - copy_start
                                    )
                                    smem_start = copy_start - epi_cta_m_start
                                    gmem_n_start = (
                                        n_index * block_n
                                        + (
                                            wg_index * wg_tile_n
                                            if config.wg_dimension
                                            == GridMinorDimension.N
                                            else 0
                                        )
                                        + epi_n_start
                                    )
                                    remaining_rows = epilogue_m
                                    while remaining_rows > 0:
                                        const_rows = 1 << int(math.log2(remaining_rows))
                                        remaining_rows //= 2

                                        @pl.when(copy_rows & const_rows != 0)
                                        def _copy_rows():
                                            plgpu.copy_smem_to_gmem(
                                                out_smem.at[
                                                    wg_index,
                                                    slot,
                                                    pl.ds(smem_start, const_rows),
                                                    :,
                                                ],
                                                out_gmem.at[
                                                    pl.ds(
                                                        block_start + copy_start,
                                                        const_rows,
                                                    ),
                                                    pl.ds(gmem_n_start, epilogue_n),
                                                ],
                                                commit_group=False,
                                            )

                                        smem_start += copy_rows & const_rows
                                        copy_start += copy_rows & const_rows
                                    plgpu.commit_smem_to_gmem_group()

                def mma_body(_, lhs_smem, rhs_smem, acc_ref):
                    if transpose_rhs:
                        # Slice the physical N rows before applying the
                        # descriptor transpose. Mosaic 0.10 cannot form a
                        # subview of an already-transposed memref.
                        rhs_operand = plgpu.transpose_ref(
                            rhs_smem.at[wg_n_slice, :], (1, 0)
                        )
                    else:
                        rhs_operand = rhs_smem.at[:, wg_n_slice]
                    plgpu.wgmma(acc_ref, lhs_smem.at[wg_m_slice], rhs_operand)
                    plgpu.wgmma_wait(config.delay_release)
                    return acc_ref

                lhs_tile = lhs_gmem.at[pl.ds(block_start, block_m), :]
                if transpose_rhs:
                    rhs_tile = rhs_gmem.at[
                        group_id, pl.ds(n_index * block_n, block_n), :
                    ]
                else:
                    rhs_tile = rhs_gmem.at[
                        group_id, :, pl.ds(n_index * block_n, block_n)
                    ]
                with jax.named_scope("pipeline"):
                    get_pipeline(mma_body, compute_context)(
                        lhs_tile, rhs_tile, allocations=pipeline_allocations
                    )

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    manual_axis_type = common.output_manual_axis_type(lhs, rhs, group_sizes)
    out_type = jax.ShapeDtypeStruct(
        (m, n_out), output_dtype, manual_axis_type=manual_axis_type
    )
    output_suffix = "" if output_dtype == lhs.dtype else f"_out{output_dtype.name}"
    launch_name = (
        f"ragged_dot_sm90_{'dx' if transpose_rhs else 'fwd'}{output_suffix}_{config}"
        f"sm_margin={sm_margin}"
    )
    warp_specialized = config.num_compute_wgs > 1
    compiler_params = mosaic_compiler_params(
        unsafe_no_auto_barriers=not warp_specialized,
        lowering_semantics=(
            plgpu.LoweringSemantics.Warpgroup
            if warp_specialized
            else plgpu.LoweringSemantics.Lane
        ),
    )
    launch = pl.kernel(
        kernel_warp_specialized if warp_specialized else kernel,
        out_type=out_type,
        mesh=plgpu.Mesh(
            grid=(active_sms // config.cluster_n,),
            grid_names=("sm",),
            cluster=(config.cluster_n,) if config.cluster_n == 2 else (),
            cluster_names=("cluster",) if config.cluster_n == 2 else (),
            num_threads=(config.num_compute_wgs + 1) if warp_specialized else None,
            thread_name="wg" if warp_specialized else None,
            kernel_name=launch_name,
        ),
        compiler_params=compiler_params,
        cost_estimate=pl.CostEstimate(
            flops=2 * cost_estimate_num_rows * k_contract * n_out,
            transcendentals=0,
            bytes_accessed=(
                cost_estimate_num_rows * k_contract * lhs.dtype.itemsize
                + rhs.size * rhs.dtype.itemsize
                + cost_estimate_num_rows * n_out * output_dtype.itemsize
            ),
        ),
        name=launch_name,
        metadata={
            "cost_model": "uniform_unpadded",
            "cost_estimate_num_rows": str(cost_estimate_num_rows),
        },
    )
    return launch(
        lhs,
        rhs,
        group_info.group_id,
        group_info.block_start,
        group_info.start_within_block,
        group_info.actual_size,
        group_info.num_tasks,
        group_info.num_rows,
    )


def ragged_dot_dx_kernel(
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: TuningConfig,
    *,
    cost_estimate_num_rows: int | None = None,
) -> jax.Array:
    """Compute dX from native forward weights ``rhs[G, K, N]``."""
    return ragged_dot_kernel(
        dout,
        rhs,
        group_sizes,
        config,
        True,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


@functools.partial(
    jax.jit, static_argnums=(3, 4), static_argnames=("cost_estimate_num_rows",)
)
def ragged_dot_dw_kernel(
    lhs: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    config: TuningConfig,
    output_manual_axis_type=None,
    *,
    cost_estimate_num_rows: int | None = None,
    accumulator: jax.Array | jax.Ref | None = None,
) -> jax.Array | None:
    """Compute ``dW[g] = lhs[g].T @ dout[g]`` with ragged contraction spans.

    Groups are compact, with offsets given by the prefix sum of ``group_sizes``.
    Boundary tiles mask adjacent groups and the unused token-buffer tail.
    Callers must provide nonnegative sizes whose sum fits in the token buffer.
    An array accumulator returns its value plus the rounded dW. A caller-owned
    Ref is updated in place and returns None. Empty groups leave it untouched.
    """
    if lhs.ndim != 2 or dout.ndim != 2 or group_sizes.ndim != 1:
        raise ValueError(
            "Expected lhs[T,K], dout[T,N], group_sizes[G], got "
            f"{lhs.shape=}, {dout.shape=}, {group_sizes.shape=}."
        )
    if lhs.shape[0] != dout.shape[0]:
        raise ValueError(
            f"lhs and dout row counts differ: {lhs.shape[0]} vs {dout.shape[0]}."
        )
    if lhs.dtype != dout.dtype:
        raise ValueError(
            f"lhs and dout must have the same dtype, got {lhs.dtype}, {dout.dtype}."
        )
    if lhs.dtype not in (jnp.bfloat16, jnp.float16):
        raise NotImplementedError(
            f"SM90 ragged_dot dW supports bfloat16/float16, got {lhs.dtype}."
        )
    if config.num_compute_wgs not in (1, 2):
        raise ValueError("SM90 dW supports one or two compute warpgroups.")
    if config.num_compute_wgs == 2:
        if config.wg_dimension != GridMinorDimension.M:
            raise ValueError("Cooperative SM90 dW currently requires an M split.")
        if config.block_m % 2:
            raise ValueError("Cooperative SM90 dW requires an even block_m.")
        wg_m = config.block_m // 2
        if wg_m % config.epilogue_m or config.block_n % config.epilogue_n:
            raise ValueError(
                "Cooperative dW tiles must be divisible by the epilogue tile: "
                f"wg_tile=({wg_m}, {config.block_n}), "
                f"epilogue=({config.epilogue_m}, {config.epilogue_n})."
            )
    if group_sizes.shape[0] == 0:
        raise ValueError("ragged_dot dW requires at least one group.")

    accumulating = accumulator is not None
    if accumulating and config.num_compute_wgs != 2:
        raise NotImplementedError(
            "SM90 dW accumulation requires two compute warpgroups."
        )
    if accumulating:
        output_shape = (group_sizes.shape[0], lhs.shape[1], dout.shape[1])
        if accumulator.shape != output_shape:
            raise ValueError(
                f"dW accumulator shape {accumulator.shape} must be {output_shape}"
            )
        if accumulator.dtype != lhs.dtype:
            raise ValueError(
                f"dW accumulator dtype {accumulator.dtype} must be {lhs.dtype}"
            )
    group_sizes_i32 = group_sizes.astype(jnp.int32)

    tokens, k_out = lhs.shape
    _, n_out = dout.shape
    num_groups = group_sizes.shape[0]
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    if tokens % block_k:
        raise ValueError(f"{tokens=} must be divisible by dW {block_k=}.")
    if k_out % block_m:
        raise ValueError(f"{k_out=} must be divisible by dW {block_m=}.")
    if n_out % block_n:
        raise ValueError(f"{n_out=} must be divisible by dW {block_n=}.")
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = tokens
    if not 0 <= cost_estimate_num_rows <= tokens:
        raise ValueError(
            f"Cost-estimate rows must be in [0, {tokens}], "
            f"got {cost_estimate_num_rows}."
        )

    group_starts = jnp.cumulative_sum(group_sizes_i32, include_initial=True)[:-1]
    num_rows = group_starts[-1] + group_sizes_i32[-1]
    group_block_starts = group_starts // block_k * block_k
    prefixes = group_starts - group_block_starts
    group_num_blocks = jnp.where(
        group_sizes_i32 == 0, jnp.int32(0), pl.cdiv(prefixes + group_sizes_i32, block_k)
    )
    grid_m = k_out // block_m
    grid_n = n_out // block_n
    lhs_transforms = tile_swizzle_transforms((block_k, block_m), lhs.dtype)
    dout_transforms = tile_swizzle_transforms((block_k, block_n), dout.dtype)

    def kernel(
        lhs_gmem,
        dout_gmem,
        group_starts_gmem,
        group_block_starts_gmem,
        group_num_blocks_gmem,
        group_sizes_gmem,
        num_rows_gmem,
        dw_gmem,
    ):
        common.trap_if_rows_exceed(num_rows_gmem[()], tokens)

        out_transforms = tile_swizzle_transforms(
            (block_m, block_n), dw_gmem.dtype, tiling_prefix=(1,)
        )

        @functools.partial(
            pl.run_scoped,
            out_smem=plgpu.SMEM(
                (2, block_m, block_n), dw_gmem.dtype, transforms=out_transforms
            ),
        )
        def scoped(out_smem):
            @plgpu.nd_loop((num_groups, grid_m * grid_n), collective_axes="sm")
            def group_mn_loop(loop_info: plgpu.NDLoopInfo):
                with jax.named_scope("scheduler"):
                    group_index = loop_info.index[0]
                    m_index, n_index = plgpu.planar_snake(
                        loop_info.index[1],
                        (grid_m, grid_n),
                        config.grid_minor_dim,
                        config.grid_tile_width,
                    )
                    group_start = group_starts_gmem[group_index]
                    group_size = group_sizes_gmem[group_index]
                    group_end = group_start + group_size
                    group_block_start = pl.multiple_of(
                        group_block_starts_gmem[group_index], block_k
                    )
                    num_blocks = group_num_blocks_gmem[group_index]
                    token_slice = pl.ds(group_block_start, tokens)

                def accumulate(acc):
                    def pipeline_body(indices, lhs_smem, dout_smem):
                        (block_index,) = indices
                        block_row_start = group_block_start + block_index * block_k

                        # Most routed groups are block-k aligned. Avoid a full
                        # SMEM read/modify/write and commit on their interior
                        # blocks; only first/last partial blocks need masking.
                        needs_mask = (block_row_start < group_start) | (
                            block_row_start + block_k > group_end
                        )

                        @pl.when(needs_mask)
                        def mask_partial_block():
                            lhs_rows = plgpu.broadcasted_iota(
                                jnp.int32, lhs_smem.shape, 0, layout=plgpu.Layout.WGMMA
                            )
                            dout_rows = plgpu.broadcasted_iota(
                                jnp.int32, dout_smem.shape, 0, layout=plgpu.Layout.WGMMA
                            )
                            lhs_valid = (block_row_start + lhs_rows >= group_start) & (
                                block_row_start + lhs_rows < group_end
                            )
                            dout_valid = (
                                block_row_start + dout_rows >= group_start
                            ) & (block_row_start + dout_rows < group_end)
                            # where(), rather than multiplication, makes
                            # poisoned NaN padding unable to leak into dW.
                            lhs_values = lhs_smem[...]
                            lhs_smem[...] = jnp.where(
                                lhs_valid, lhs_values, jnp.zeros_like(lhs_values)
                            )
                            dout_values = dout_smem[...]
                            dout_smem[...] = jnp.where(
                                dout_valid, dout_values, jnp.zeros_like(dout_values)
                            )
                            plgpu.commit_smem()

                        plgpu.wgmma(
                            acc, plgpu.transpose_ref(lhs_smem, (1, 0)), dout_smem
                        )
                        plgpu.wgmma_wait(1 if config.num_stages > 1 else 0)

                    @pl.when(group_size > 0)
                    def run_pipeline():
                        plgpu.emit_pipeline(
                            pipeline_body,
                            grid=(num_blocks,),
                            in_specs=(
                                plgpu.BlockSpec(
                                    (block_k, block_m),
                                    lambda block_index: (block_index, m_index),
                                    transforms=lhs_transforms,
                                    delay_release=(1 if config.num_stages > 1 else 0),
                                ),
                                plgpu.BlockSpec(
                                    (block_k, block_n),
                                    lambda block_index: (block_index, n_index),
                                    transforms=dout_transforms,
                                    delay_release=(1 if config.num_stages > 1 else 0),
                                ),
                            ),
                            max_concurrent_steps=config.num_stages,
                        )(lhs_gmem.at[token_slice, :], dout_gmem.at[token_slice, :])

                    return acc[...]

                @pl.when((group_size > 0) | jnp.bool_(not accumulating))
                def run_nonempty_group():
                    acc = pl.run_scoped(accumulate, plgpu.ACC((block_m, block_n)))
                    slot = lax.rem(loop_info.local_index, 2)
                    # Carry the TMA-store pipeline across persistent work tiles.
                    # Waiting to one protects the alternating slot while allowing
                    # the previous tile's store to overlap the next tile's MMA.
                    plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                    out_smem[slot] = acc.astype(out_smem.dtype)
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(
                        out_smem.at[slot],
                        dw_gmem.at[
                            group_index,
                            pl.ds(m_index * block_m, block_m),
                            pl.ds(n_index * block_n, block_n),
                        ],
                        reduction_op="add" if accumulating else None,
                    )

            plgpu.wait_smem_to_gmem(0, wait_read_only=not accumulating)

    def kernel_warp_specialized(
        lhs_gmem,
        dout_gmem,
        group_starts_gmem,
        group_block_starts_gmem,
        group_num_blocks_gmem,
        group_sizes_gmem,
        num_rows_gmem,
        dw_gmem,
    ):
        """One TMA producer WG feeding two M-partitioned dW compute WGs."""
        common.trap_if_rows_exceed(num_rows_gmem[()], tokens)
        wg_tile_m = block_m // config.num_compute_wgs
        epilogue_m = config.epilogue_m
        epilogue_n = config.epilogue_n
        num_epilogue_tiles = (wg_tile_m // epilogue_m) * (block_n // epilogue_n)
        num_out_slots = min(2, num_epilogue_tiles)

        def ignore(*_, **__):
            return None

        # The dynamic reduction length does not affect pipeline allocations.
        # Use a static full-stage pipeline to obtain the shared-memory types;
        # a shorter grid would shrink the allocation below num_stages.
        def allocation_pipeline(pipeline_body, compute_context):
            return plgpu.emit_pipeline_warp_specialized(
                pipeline_body,
                grid=(config.num_stages,),
                memory_registers=config.memory_registers,
                in_specs=(
                    plgpu.BlockSpec(
                        (block_k, block_m),
                        lambda block_index: (block_index, 0),
                        transforms=lhs_transforms,
                        delay_release=config.delay_release,
                    ),
                    plgpu.BlockSpec(
                        (block_k, block_n),
                        lambda block_index: (block_index, 0),
                        transforms=dout_transforms,
                        delay_release=config.delay_release,
                    ),
                ),
                wg_axis="wg",
                num_compute_wgs=config.num_compute_wgs,
                max_concurrent_steps=config.num_stages,
                compute_context=compute_context,
            )

        pipeline_allocations = allocation_pipeline(ignore, ignore).get_allocations(
            lhs_gmem, dout_gmem
        )
        out_transforms = tile_swizzle_transforms(
            (epilogue_m, epilogue_n), dw_gmem.dtype, tiling_prefix=(1,)
        )

        @functools.partial(
            pl.run_scoped,
            pipeline_allocations=pipeline_allocations,
            out_smem=plgpu.SMEM(
                (config.num_compute_wgs, num_out_slots, epilogue_m, epilogue_n),
                dw_gmem.dtype,
                transforms=out_transforms,
            ),
            mask_barriers=plgpu.Barrier(
                num_arrivals=config.num_compute_wgs, num_barriers=config.num_stages
            ),
            collective_axes="wg",
        )
        def scoped(pipeline_allocations, out_smem, mask_barriers):
            wg_index = lax.axis_index("wg")
            wg_m_offset = wg_index * wg_tile_m
            wg_m_slice = pl.ds(wg_m_offset, wg_tile_m)

            @plgpu.nd_loop((num_groups, grid_m * grid_n), collective_axes="sm")
            def group_mn_loop(loop_info: plgpu.NDLoopInfo):
                with jax.named_scope("scheduler"):
                    group_index = loop_info.index[0]
                    m_index, n_index = plgpu.planar_snake(
                        loop_info.index[1],
                        (grid_m, grid_n),
                        config.grid_minor_dim,
                        config.grid_tile_width,
                    )
                    group_start = group_starts_gmem[group_index]
                    group_size = group_sizes_gmem[group_index]
                    group_end = group_start + group_size
                    group_block_start = pl.multiple_of(
                        group_block_starts_gmem[group_index], block_k
                    )
                    num_blocks = group_num_blocks_gmem[group_index]
                    token_slice = pl.ds(group_block_start, tokens)

                def compute_context(eval_pipeline):
                    @functools.partial(
                        pl.run_scoped,
                        acc_ref=plgpu.ACC((wg_tile_m, block_n), jnp.float32),
                    )
                    def acc_scope(acc_ref):
                        with jax.named_scope("compute"):
                            eval_pipeline(acc_ref)
                        with jax.named_scope("epilogue"):
                            acc = acc_ref[...].astype(dw_gmem.dtype)
                            for epi_mi in range(wg_tile_m // epilogue_m):
                                for epi_ni in range(block_n // epilogue_n):
                                    epi_index = (
                                        epi_mi * (block_n // epilogue_n) + epi_ni
                                    )
                                    slot = epi_index % num_out_slots
                                    epi_m_start = epi_mi * epilogue_m
                                    epi_n_start = epi_ni * epilogue_n
                                    epi_m_slice = slice(
                                        epi_m_start, epi_m_start + epilogue_m
                                    )
                                    epi_n_slice = slice(
                                        epi_n_start, epi_n_start + epilogue_n
                                    )
                                    plgpu.wait_smem_to_gmem(
                                        num_out_slots - 1, wait_read_only=True
                                    )
                                    out_smem[wg_index, slot] = acc[
                                        epi_m_slice, epi_n_slice
                                    ]
                                    plgpu.commit_smem()
                                    plgpu.copy_smem_to_gmem(
                                        out_smem.at[wg_index, slot],
                                        dw_gmem.at[
                                            group_index,
                                            pl.ds(
                                                m_index * block_m
                                                + wg_m_offset
                                                + epi_m_start,
                                                epilogue_m,
                                            ),
                                            pl.ds(
                                                n_index * block_n + epi_n_start,
                                                epilogue_n,
                                            ),
                                        ],
                                        reduction_op="add" if accumulating else None,
                                    )

                def mma_body(indices, lhs_smem, dout_smem, acc_ref):
                    (block_index,) = indices
                    block_row_start = group_block_start + block_index * block_k
                    needs_mask = (block_row_start < group_start) | (
                        block_row_start + block_k > group_end
                    )

                    @pl.when(needs_mask)
                    def mask_partial_block():
                        lhs_slice = lhs_smem.at[:, wg_m_slice]
                        lhs_rows = plgpu.broadcasted_iota(
                            jnp.int32, lhs_slice.shape, 0, layout=plgpu.Layout.WGMMA
                        )
                        lhs_valid = (block_row_start + lhs_rows >= group_start) & (
                            block_row_start + lhs_rows < group_end
                        )
                        lhs_values = lhs_slice[...]
                        lhs_slice[...] = jnp.where(
                            lhs_valid, lhs_values, jnp.zeros_like(lhs_values)
                        )

                        @pl.when(wg_index == 0)
                        def mask_dout():
                            dout_rows = plgpu.broadcasted_iota(
                                jnp.int32, dout_smem.shape, 0, layout=plgpu.Layout.WGMMA
                            )
                            dout_valid = (
                                block_row_start + dout_rows >= group_start
                            ) & (block_row_start + dout_rows < group_end)
                            dout_values = dout_smem[...]
                            dout_smem[...] = jnp.where(
                                dout_valid, dout_values, jnp.zeros_like(dout_values)
                            )

                        plgpu.commit_smem()
                        mask_slot = lax.rem(block_index, config.num_stages)
                        plgpu.barrier_arrive(mask_barriers.at[mask_slot])
                        plgpu.barrier_wait(mask_barriers.at[mask_slot])

                    plgpu.wgmma(
                        acc_ref,
                        plgpu.transpose_ref(lhs_smem.at[:, wg_m_slice], (1, 0)),
                        dout_smem,
                    )
                    plgpu.wgmma_wait(config.delay_release)
                    return acc_ref

                # Rebuild the pipeline with this group's dynamic K extent. A
                # zero-step pipeline still enters compute_context, whose fresh
                # accumulator gives empty groups a zero epilogue without a
                # second hot-path output loop.
                def dynamic_pipeline(pipeline_body, compute_ctx):
                    return plgpu.emit_pipeline_warp_specialized(
                        pipeline_body,
                        grid=(num_blocks,),
                        memory_registers=config.memory_registers,
                        in_specs=(
                            plgpu.BlockSpec(
                                (block_k, block_m),
                                lambda block_index: (block_index, 0),
                                transforms=lhs_transforms,
                                delay_release=config.delay_release,
                            ),
                            plgpu.BlockSpec(
                                (block_k, block_n),
                                lambda block_index: (block_index, 0),
                                transforms=dout_transforms,
                                delay_release=config.delay_release,
                            ),
                        ),
                        wg_axis="wg",
                        num_compute_wgs=config.num_compute_wgs,
                        max_concurrent_steps=config.num_stages,
                        compute_context=compute_ctx,
                    )

                @pl.when((group_size > 0) | jnp.bool_(not accumulating))
                def run_nonempty_group():
                    with jax.named_scope("pipeline"):
                        dynamic_pipeline(mma_body, compute_context)(
                            lhs_gmem.at[token_slice, pl.ds(m_index * block_m, block_m)],
                            dout_gmem.at[
                                token_slice, pl.ds(n_index * block_n, block_n)
                            ],
                            allocations=pipeline_allocations,
                        )

            plgpu.wait_smem_to_gmem(0, wait_read_only=not accumulating)

    if output_manual_axis_type is None:
        output_manual_axis_type = common.output_manual_axis_type(lhs, dout, group_sizes)
    if (
        accumulating
        and jax.typeof(accumulator).manual_axis_type != output_manual_axis_type
    ):
        raise ValueError(
            "dW accumulator and output have incompatible manual axis types"
        )
    out_type = jax.ShapeDtypeStruct(
        (num_groups, k_out, n_out), lhs.dtype, manual_axis_type=output_manual_axis_type
    )
    num_sms = device_sm_count()
    sm_margin = configured_sm_margin(config)
    if not 0 <= sm_margin < num_sms:
        raise ValueError(
            f"RAGGED_DOT_SM_MARGIN must be in [0, {num_sms - 1}], got {sm_margin}."
        )
    launch_name = (
        f"ragged_dot_sm90_dw_{'accumulate_' if accumulating else ''}"
        f"{config}sm_margin={sm_margin}"
    )
    warp_specialized = config.num_compute_wgs > 1
    launch = pl.kernel(
        kernel_warp_specialized if warp_specialized else kernel,
        out_type=() if accumulating else out_type,
        mesh=plgpu.Mesh(
            grid=(num_sms - sm_margin,),
            grid_names=("sm",),
            num_threads=(config.num_compute_wgs + 1) if warp_specialized else None,
            thread_name="wg" if warp_specialized else None,
            kernel_name=launch_name,
        ),
        compiler_params=mosaic_compiler_params(
            lowering_semantics=plgpu.LoweringSemantics.Warpgroup
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                2 * cost_estimate_num_rows * k_out * n_out
                + (num_groups * k_out * n_out if accumulating else 0)
            ),
            transcendentals=0,
            bytes_accessed=(
                cost_estimate_num_rows * (k_out + n_out) * lhs.dtype.itemsize
                + (2 if accumulating else 1)
                * num_groups
                * k_out
                * n_out
                * lhs.dtype.itemsize
            ),
        ),
        name=launch_name,
        metadata={
            "cost_model": "uniform_unpadded",
            "cost_estimate_num_rows": str(cost_estimate_num_rows),
        },
    )
    args = (
        lhs,
        dout,
        group_starts,
        group_block_starts,
        group_num_blocks,
        group_sizes_i32,
        num_rows,
    )
    if accumulating:
        if isinstance(jax.typeof(accumulator), jax.ref.AbstractRef):
            launch(*args, accumulator)
            return None
        accumulator_ref = jax.new_ref(accumulator)
        launch(*args, accumulator_ref)
        return jax.freeze(accumulator_ref)
    return launch(*args)


# DeepSeek-v3 local expert shapes have M divisible by 256 and hidden dimensions
# divisible by 128. Two compute warpgroups split M, while a producer warpgroup
# pipelines TMA and pairs of N-adjacent CTAs share the LHS tile. CUPTI tuning on
# H100 favors one 256-row task per typical 256-row expert, an M-minor width of
# one, and a 128-SM grid whose work waves divide exactly for the projection
# shapes.
DEFAULT_CONFIG = TuningConfig(
    block_m=256,
    block_n=128,
    block_k=64,
    num_stages=4,
    grid_minor_dim=GridMinorDimension.M,
    grid_tile_width=1,
    num_compute_wgs=2,
    wg_dimension=GridMinorDimension.M,
    sm_margin=4,
    cluster_n=2,
)

# cuDNN's H100 grouped GEMM pairs two N-adjacent CTAs, multicasts their shared
# LHS tile, and splits each CTA's 256 rows between the compute warpgroups. The
# matching Mosaic topology lowers L2 traffic and halves the number of GMMA
# issue instructions while retaining the M-minor traversal's stable behavior
# for both uniform and uneven expert loads.

# FP32 preactivations double the forward epilogue scratch. Three pipeline stages
# keep the complete allocation under Hopper's per-CTA shared-memory limit.
FP32_SWIGLU_CONFIG = dataclasses.replace(DEFAULT_CONFIG, num_stages=3)


def fit_cluster_to_output(config: TuningConfig, n_out: int) -> TuningConfig:
    """Fall back to one CTA when the output has an unpaired final N tile."""
    if n_out % config.block_n == 0:
        n_tiles = n_out // config.block_n
        if n_tiles % config.cluster_n:
            return dataclasses.replace(config, cluster_n=1)
    return config


# dW maps activation K and output N to the WGMMA M/N dimensions. A producer
# feeds two M-split compute warpgroups over a 128x256 CTA tile. Unlike forward,
# dW benefits from retaining all 132 SMs despite the less even tail wave.
DW_DEFAULT_CONFIG = TuningConfig(
    block_m=128,
    block_n=256,
    block_k=64,
    num_stages=4,
    grid_minor_dim=GridMinorDimension.N,
    grid_tile_width=4,
    num_compute_wgs=2,
    wg_dimension=GridMinorDimension.M,
    # Each expert reuses the pipeline allocations. Delayed release can reuse
    # barriers before the preceding short group has finished consuming them.
    delay_release=0,
    sm_margin=0,
)

# The gate/up weight gradient has Kout > Nout. H100 measurements favor a
# shorter N-minor snake for that aspect ratio (width 2 and 16 were tied, while
# 4 and 8 were slower); width 2 also remains valid for smaller output grids.
DW_WIDE_K_CONFIG = dataclasses.replace(DW_DEFAULT_CONFIG, grid_tile_width=2)


def ragged_dot_dw_config(k_out: int, n_out: int):
    """Select the cooperative dW tile, retaining support for small test dims."""
    config = DW_WIDE_K_CONFIG if k_out > n_out else DW_DEFAULT_CONFIG
    if n_out % config.block_n:
        if n_out % 128:
            raise ValueError(f"SM90 dW output width {n_out} must be divisible by 128.")
        config = dataclasses.replace(config, block_n=128)
    if k_out % config.block_m:
        raise ValueError(f"SM90 dW hidden width {k_out} must be divisible by 128.")
    return config


def projection_config(output_dtype, n_out: int) -> TuningConfig:
    """Forward tuning for an output dtype and width; float32 stores need more SMEM."""
    config = FP32_SWIGLU_CONFIG if output_dtype == jnp.float32 else DEFAULT_CONFIG
    return fit_cluster_to_output(config, n_out)


def ragged_dot(
    lhs, rhs, group_sizes, cost_estimate_num_rows=None, *, output_dtype=None
):
    """Raw grouped product; differentiation is defined by the shared operation."""
    output_dtype = jnp.dtype(lhs.dtype if output_dtype is None else output_dtype)
    return ragged_dot_kernel(
        lhs,
        rhs,
        group_sizes,
        projection_config(output_dtype, rhs.shape[-1]),
        output_dtype=output_dtype,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


def ragged_dot_swiglu(
    lhs,
    rhs_interleaved,
    group_sizes,
    out_dtype=None,
    cost_estimate_num_rows=None,
    *,
    save_preact=False,
):
    """Project at output precision, then apply SwiGLU over the live row prefix."""
    output_dtype = jnp.dtype(lhs.dtype if out_dtype is None else out_dtype)
    preact = ragged_dot_kernel(
        lhs,
        rhs_interleaved,
        group_sizes,
        projection_config(output_dtype, rhs_interleaved.shape[-1]),
        output_dtype=output_dtype,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )
    num_rows = jnp.sum(group_sizes, dtype=jnp.int32)
    output = swiglu.forward(preact, num_rows, output_dtype, cost_estimate_num_rows)
    return (output, preact) if save_preact else output


def swiglu_backward(preact, dout, group_sizes, cost_estimate_num_rows=None):
    """Preactivation cotangent over the live row prefix; the tail is unspecified."""
    num_rows = jnp.sum(group_sizes, dtype=jnp.int32)
    return swiglu.backward(preact, num_rows, dout, cost_estimate_num_rows)


def ragged_dot_dw(
    lhs: jax.Array,
    dout: jax.Array,
    group_sizes: jax.Array,
    *,
    output_manual_axis_type=None,
    accumulator: jax.Array | jax.Ref | None = None,
    cost_estimate_num_rows=None,
) -> jax.Array | None:
    """Compute dW, add it to an array, or update a caller-owned Ref in place."""
    config = ragged_dot_dw_config(lhs.shape[-1], dout.shape[-1])
    return ragged_dot_dw_kernel(
        lhs,
        dout,
        group_sizes,
        config,
        output_manual_axis_type,
        cost_estimate_num_rows=cost_estimate_num_rows,
        accumulator=accumulator,
    )


def ragged_dot_dx(
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    cost_estimate_num_rows=None,
) -> jax.Array:
    """Compute the grouped input gradient using the SM90 ragged kernel."""
    config = fit_cluster_to_output(DEFAULT_CONFIG, rhs.shape[1])
    return ragged_dot_dx_kernel(
        dout.astype(rhs.dtype),
        rhs,
        group_sizes,
        config,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )
