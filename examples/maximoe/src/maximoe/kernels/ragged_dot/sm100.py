# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""SM100 Mosaic-GPU ragged-dot kernels.

This variant of
``jax.experimental.pallas.ops.gpu.blackwell_ragged_dot_mgpu`` computes
an aligned task table from ``group_sizes``. The output allocation keeps its
static height, but padded rows and empty groups do not consume tensor-core work.

The dynamic schedule uses ``plgpu.nd_loop`` with an M-minor planar snake. The
partial final snake tile avoids scheduling padded M blocks.

Every scheduled iteration owns live rows and runs the full TMA, MMA, and store
sequence. Rows at or above ``sum(group_sizes)`` remain unwritten.
"""

import dataclasses
import enum
import functools

import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, nvvm
from jaxlib.mlir.dialects import memref as memref_dialect

from maximoe.kernels import gpu
from maximoe.kernels.ragged_dot import common

SWIGLU_EPILOGUE_BLOCK = common.SWIGLU_EPILOGUE_BLOCK


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    """Logical ``lhs[M, K] @ rhs[K, N]`` tile and scheduler configuration."""

    tile_m: int
    tile_n: int
    tile_k: int
    max_concurrent_steps: int
    collective: bool
    grid_tile_width: int
    epilogue_tile_n: int = 2 * SWIGLU_EPILOGUE_BLOCK

    def __str__(self):
        return "_".join(
            f"{key}={value}" for key, value in dataclasses.asdict(self).items()
        )


def persistent_grid(config: TuningConfig, total_sms: int, sm_margin: int) -> tuple[int]:
    """Keep the requested margin exact instead of flooring clustered grids."""
    cluster_size = 2 if config.collective else 1
    active_sms = total_sms - sm_margin
    if not 0 <= sm_margin <= total_sms - cluster_size:
        raise ValueError(
            f"RAGGED_DOT_SM_MARGIN must be in [0, {total_sms - cluster_size}], "
            f"got {sm_margin}"
        )
    if active_sms % cluster_size:
        raise ValueError(
            f"{active_sms=} must be divisible by {cluster_size=} "
            "for the persistent grid"
        )
    return (active_sms // cluster_size,)


class RhsLayout(enum.Enum):
    # rhs[g] is laid out as [contract, out], as in forward rhs[G, K, N].
    CONTRACT_OUT = enum.auto()
    # rhs[g] is laid out as [out, contract], as in dX rhs[G, K, N].
    OUT_CONTRACT = enum.auto()


TMA_WARP = 0
MMA_WARP = 1
COMPUTE_WG = 0
STORE_WG = 1
# Double-buffered TMEM accumulators and their handshake barriers.
NUM_ACC_SLOTS = 2


def store_identity(
    acc_tmem_slot, block_slice_m, n_index, row, num_rows, *, out_gmem, acc_smem
):
    tile_n = acc_tmem_slot.shape[1]
    epilogue_tile_n = acc_smem.shape[1]
    step_out = out_gmem.at[block_slice_m, pl.ds(n_index * tile_n, tile_n)]
    for ni in range(tile_n // epilogue_tile_n):
        columns = pl.ds(ni * epilogue_tile_n, epilogue_tile_n)
        acc_smem[...] = plgpu.async_load_tmem(
            acc_tmem_slot.at[:, columns], layout=plgpu.Layout.TCGEN05
        ).astype(out_gmem.dtype)
        common.store_rows(acc_smem, step_out.at[:, columns], row, num_rows)


def store_swiglu(
    acc_tmem_slot,
    block_slice_m,
    n_index,
    row,
    num_rows,
    *,
    out_gmem,
    preact_gmem,
    acc_smem,
    swiglu_smem,
):
    tile_n = acc_tmem_slot.shape[1]
    epilogue_tile_n = acc_smem.shape[1]
    half_epilogue = epilogue_tile_n // 2
    store_n = swiglu_smem.shape[1]
    step_out = out_gmem.at[block_slice_m, pl.ds(n_index * (tile_n // 2), tile_n // 2)]
    if preact_gmem is not None:
        step_preact = preact_gmem.at[block_slice_m, pl.ds(n_index * tile_n, tile_n)]

    def swiglu_and_save_preact(column):
        columns = pl.ds(column, epilogue_tile_n)
        # Preserve the rounded, interleaved preactivation consumed by SwiGLU
        # and its backward replay, even when two chunks share an output store.
        preact = plgpu.async_load_tmem(
            acc_tmem_slot.at[:, columns], layout=plgpu.Layout.TCGEN05
        ).astype(out_gmem.dtype)
        if preact_gmem is not None:
            acc_smem[...] = preact
            common.store_rows(acc_smem, step_preact.at[:, columns], row, num_rows)
        gate = preact[:, :half_epilogue].astype(jnp.float32)
        up = preact[:, half_epilogue:].astype(jnp.float32)
        return (gate * jax.nn.sigmoid(gate) * up).astype(out_gmem.dtype)

    def store_group(ni, _):
        def store_chunk(chunk, _):
            swiglu_smem[:, pl.ds(chunk * half_epilogue, half_epilogue)] = (
                swiglu_and_save_preact(2 * ni * store_n + chunk * epilogue_tile_n)
            )

        lax.fori_loop(0, store_n // half_epilogue, store_chunk, None)
        common.store_rows(
            swiglu_smem, step_out.at[:, pl.ds(ni * store_n, store_n)], row, num_rows
        )

    num_full_stores, remaining_cols = divmod(tile_n // 2, store_n)
    # A full-width store cannot even be traced for a narrower output tile.
    if num_full_stores:
        lax.fori_loop(0, num_full_stores, store_group, None)
    if remaining_cols:
        # A custom tile's unpaired chunk cannot fill a 128-byte TMA row.
        @plgpu.inline_mgpu(
            arg_types=(
                plgpu.RefType(()),
                plgpu.Layout.TCGEN05,
                plgpu.Layout.WG_SPLAT,
                plgpu.Layout.WG_SPLAT,
            )
        )
        def store_partial(ctx, destination, values, start, count):
            del ctx
            index = ir.IndexType.get()
            start = arith.index_cast(index, start.registers.item())
            end = arith.addi(start, arith.index_cast(index, count.registers.item()))

            def store_element(element, indices):
                row_idx, _ = indices
                active = arith.andi(
                    arith.cmpi(arith.CmpIPredicate.sge, row_idx, start),
                    arith.cmpi(arith.CmpIPredicate.slt, row_idx, end),
                )
                with mgpu.when(active):
                    memref_dialect.store(element, destination, indices)

            values.foreach(store_element)

        def store_chunk(chunk, _):
            column = num_full_stores * store_n + chunk * half_epilogue
            value = swiglu_and_save_preact(2 * column)
            store_partial(
                step_out.at[:, pl.ds(column, half_epilogue)], value, row, num_rows
            )

        lax.fori_loop(0, remaining_cols // half_epilogue, store_chunk, None)


def do_ragged_gemm_tile(
    lhs_gmem,
    rhs_gmem,
    block_start,
    n_index,
    cluster_idx,
    wg_axis: str,
    collective_axis: str | None,
    local_index: jax.Array | int,
    tiling: TuningConfig,
    rhs_layout: RhsLayout,
    row_offset,
    row_count,
    a_smem,
    b_smem,
    acc_tmem,
    a_tma_barrier,
    b_tma_barrier,
    acc_consumed_barrier,
    mma_done_barrier,
    consumed_barrier,
    store,
):
    """Compute one logical ragged GEMM output tile."""
    _, k_contract = lhs_gmem.shape
    collective = tiling.collective
    tile_m, tile_n, tile_k = (tiling.tile_m, tiling.tile_n, tiling.tile_k)
    max_concurrent_steps = tiling.max_concurrent_steps
    block_tile_m = tile_m
    if collective:
        tile_m *= 2
        tile_n *= 2
    k_iters = k_contract // tile_k

    block_start = pl.multiple_of(block_start, 8)
    local_block_start = block_start + cluster_idx * block_tile_m
    is_lead_block = cluster_idx == 0
    wg_idx = lax.axis_index(wg_axis)

    block_slice_m = pl.ds(local_block_start, block_tile_m)
    slice_m = pl.ds(block_start, tile_m)
    slice_n = pl.ds(n_index * tile_n, tile_n)
    acc_slot = lax.rem(local_index, jnp.int32(NUM_ACC_SLOTS))
    rhs_partition_dim = 1 if rhs_layout is RhsLayout.CONTRACT_OUT else 0

    @pl.when(wg_idx == COMPUTE_WG)
    @jax.named_scope("compute_wg_ragged_gemm")
    def _():
        @plgpu.warp_map
        def per_warp(warp_id):
            @pl.when(warp_id == TMA_WARP)
            def memory():
                def loop_body(ki, _):
                    slice_k = pl.ds(ki * tile_k, tile_k)
                    step = local_index * jnp.int32(k_iters) + ki
                    slot = lax.rem(step, max_concurrent_steps)

                    @pl.when(step >= max_concurrent_steps)
                    def _():
                        plgpu.barrier_wait(consumed_barrier.at[slot])

                    plgpu.copy_gmem_to_smem(
                        lhs_gmem.at[slice_m, slice_k],
                        a_smem.at[slot],
                        a_tma_barrier.at[slot],
                        leader_tracked=plgpu.CopyPartition.PARTITIONED(0)
                        if collective
                        else None,
                        collective_axes=collective_axis,
                    )
                    if rhs_layout is RhsLayout.CONTRACT_OUT:
                        rhs_tile = rhs_gmem.at[slice_k, slice_n]
                    else:
                        rhs_tile = rhs_gmem.at[slice_n, slice_k]
                    plgpu.copy_gmem_to_smem(
                        rhs_tile,
                        b_smem.at[slot],
                        b_tma_barrier.at[slot],
                        leader_tracked=plgpu.CopyPartition.PARTITIONED(
                            rhs_partition_dim
                        )
                        if collective
                        else None,
                        collective_axes=collective_axis,
                    )

                lax.fori_loop(0, k_iters, loop_body, None)

            @pl.when(jnp.logical_and(warp_id == MMA_WARP, local_index >= NUM_ACC_SLOTS))
            def wait_acc_consumed():
                plgpu.barrier_wait(acc_consumed_barrier.at[acc_slot])

            @pl.when(jnp.logical_and(warp_id == MMA_WARP, is_lead_block))
            def compute():
                def loop_body(ki, _):
                    step = local_index * jnp.int32(k_iters) + ki
                    slot = lax.rem(step, max_concurrent_steps)
                    plgpu.barrier_wait(a_tma_barrier.at[slot])
                    plgpu.barrier_wait(b_tma_barrier.at[slot])

                    is_last_iter = ki >= k_iters - 1
                    acc_tmem_slice = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
                    b_operand = b_smem.at[slot]
                    if rhs_layout is RhsLayout.OUT_CONTRACT:
                        # The lowering turns this metadata-only SMEM transpose into the
                        # UMMA RHS descriptor transform without moving data.
                        b_operand = b_operand.transpose((1, 0))
                    plgpu.tcgen05_mma(
                        acc_tmem_slice,
                        a_smem.at[slot],
                        b_operand,
                        consumed_barrier.at[slot],
                        accumulate=(ki > 0),
                        collective_axis=collective_axis,
                    )

                    @pl.when(is_last_iter)
                    def _():
                        plgpu.tcgen05_commit_arrive(
                            mma_done_barrier.at[acc_slot],
                            collective_axis=collective_axis,
                        )

                lax.fori_loop(0, k_iters, loop_body, None)

    @pl.when(wg_idx == STORE_WG)
    @jax.named_scope("store_wg_ragged_gemm")
    def _():
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        acc_tmem_slot = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
        smem_start = row_offset - cluster_idx * block_tile_m
        smem_start = lax.max(smem_start, jnp.int32(0))
        block0_copy_size = lax.max(
            lax.min(block_tile_m - row_offset, row_count), jnp.int32(0)
        )
        block_local_size = lax.select(
            is_lead_block, block0_copy_size, row_count - block0_copy_size
        )
        store(acc_tmem_slot, block_slice_m, n_index, smem_start, block_local_size)
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(acc_consumed_barrier.at[acc_slot])


def do_ragged_dw_tile(
    lhs_gmem,
    dout_gmem,
    dw_gmem,
    group_starts_gmem,
    group_sizes_gmem,
    group_id,
    m_index,
    n_index,
    wg_axis: str,
    local_index: jax.Array | int,
    step_base: jax.Array | int,
    tiling: TuningConfig,
    accumulate: bool,
    a_smem,
    b_smem,
    acc_tmem,
    acc_smem,
    ab_tma_barrier,
    store_done_barrier,
    mma_done_barrier,
    consumed_barrier,
):
    """Compute one K-grouped dW tile: ``lhs[token, k].T @ dout[token, n]``.

    Pipelining mirrors the forward ``do_ragged_gemm_tile``: TMA and MMA warps
    each run an independent ``lax.fori_loop`` over ``k_iters``, synchronizing
    through the per-slot ``ab_tma_barrier`` and ``consumed_barrier``. That gives
    ``max_concurrent_steps``-deep producer/consumer overlap identical to forward,
    with no WG-level sync per ``ki``.

    ``group_starts_gmem`` contains the prefix sums of ``group_sizes_gmem``.
    Boundary contraction tiles are select-zeroed in SMEM, so the unused buffer
    tail or adjacent groups cannot enter dW. TMA loads start at the preceding
    8-row boundary; the logical group start itself need not be aligned.
    """
    dtype = dw_gmem.dtype

    tile_m, tile_n, tile_k = (tiling.tile_m, tiling.tile_n, tiling.tile_k)
    epilogue_tile_n = tiling.epilogue_tile_n
    max_concurrent_steps = tiling.max_concurrent_steps

    group_start = group_starts_gmem[group_id]
    aligned_start = pl.multiple_of(group_start // 8 * 8, 8)
    prefix = group_start - aligned_start
    group_size = group_sizes_gmem[group_id]
    # An empty group may have a nonzero prefix. It must use only the empty-tile
    # handshake, without also issuing an MMA completion arrival.
    k_iters = jnp.where(group_size == 0, 0, pl.cdiv(prefix + group_size, tile_k))

    wg_idx = lax.axis_index(wg_axis)

    slice_m = pl.ds(m_index * tile_m, tile_m)
    slice_n = pl.ds(n_index * tile_n, tile_n)
    acc_slot = lax.rem(local_index, jnp.int32(NUM_ACC_SLOTS))
    regs_layout = plgpu.Layout.TCGEN05
    transforms = common.tile_swizzle_transforms((tile_k,), dtype)
    swizzle_elems = transforms[0].tiling[-1]

    def zero_smem_row(smem_ref, row, cols: int):
        # Zeroing a whole row is swizzle-invariant; only the MMA warp participates.
        @plgpu.inline_mgpu(
            arg_types=(
                plgpu.RefType(transforms),
                plgpu.Layout.WG_SPLAT,
                plgpu.Layout.WG_SPLAT,
            )
        )
        def store(ctx, smem_ref, row, zero):
            del ctx
            i32 = ir.IntegerType.get_signless(32)
            index = ir.IndexType.get()
            lane = arith.remui(mgpu.thread_idx(), mgpu.c(32, i32))
            row_idx = arith.index_cast(index, row.registers.item())
            row_outer = arith.divui(row_idx, mgpu.c(8, index))
            row_inner = arith.remui(row_idx, mgpu.c(8, index))
            zero_value = zero.registers.item()
            for col_start in range(0, cols, 32):
                col_i32 = arith.addi(lane, mgpu.c(col_start, i32))
                in_bounds = arith.cmpi(
                    arith.CmpIPredicate.ult, col_i32, mgpu.c(cols, i32)
                )
                with mgpu.when(in_bounds):
                    col_idx = arith.index_cast(index, col_i32)
                    col_outer = arith.divui(col_idx, mgpu.c(swizzle_elems, index))
                    col_inner = arith.remui(col_idx, mgpu.c(swizzle_elems, index))
                    memref_dialect.store(
                        zero_value,
                        smem_ref,
                        [row_outer, col_outer, row_inner, col_inner],
                    )

        store(smem_ref, row, jnp.zeros((), dtype))

    def mask_smem_tile(smem_ref, valid_start, valid_end, cols: int):
        def loop_body(row, _):
            zero_smem_row(smem_ref, row, cols)

        lax.fori_loop(jnp.int32(0), valid_start, loop_body, None)
        lax.fori_loop(valid_end, jnp.int32(tile_k), loop_body, None)

    def commit_warp_smem_to_mma():
        @plgpu.inline_mgpu(arg_types=())
        def commit(ctx):
            del ctx
            nvvm.fence_proxy(
                nvvm.ProxyKind.async_shared, space=nvvm.SharedSpace.shared_cta
            )
            mgpu.utils.warp_barrier()

        commit()

    @pl.when(wg_idx == COMPUTE_WG)
    @jax.named_scope("compute_wg_ragged_dw")
    def _():
        @plgpu.warp_map
        def per_warp(warp_id):
            # TMA warp issues all aligned contraction tiles.
            @pl.when(warp_id == TMA_WARP)
            def memory():
                def loop_body(ki, _):
                    step = step_base + ki
                    slot = lax.rem(step, max_concurrent_steps)
                    token_base = aligned_start + ki * jnp.int32(tile_k)
                    token_slice = pl.ds(token_base, tile_k)

                    # Empty groups advance no steps and initialize no pipeline slots.
                    @pl.when(step >= max_concurrent_steps)
                    def _():
                        plgpu.barrier_wait(consumed_barrier.at[slot])

                    # Logical A is lhs.T (contract on M). Load the physical
                    # [tokens, K_out] tile; the MMA uses transpose_ref to view it.
                    plgpu.copy_gmem_to_smem(
                        lhs_gmem.at[token_slice, slice_m],
                        a_smem.at[slot],
                        ab_tma_barrier.at[slot],
                    )
                    plgpu.copy_gmem_to_smem(
                        dout_gmem.at[token_slice, slice_n],
                        b_smem.at[slot],
                        ab_tma_barrier.at[slot],
                    )

                lax.fori_loop(0, k_iters, loop_body, None)

            @pl.when(jnp.logical_and(warp_id == MMA_WARP, local_index >= NUM_ACC_SLOTS))
            def wait_store():
                plgpu.barrier_wait(store_done_barrier.at[acc_slot])

            # MMA warp processes the aligned contraction tiles.
            @pl.when(warp_id == MMA_WARP)
            def compute():
                def loop_body(ki, _):
                    slot = lax.rem(step_base + ki, max_concurrent_steps)
                    plgpu.barrier_wait(ab_tma_barrier.at[slot])

                    tile_start = ki * jnp.int32(tile_k)
                    valid_start = lax.max(prefix - tile_start, jnp.int32(0))
                    valid_end = lax.min(
                        prefix + group_size - tile_start, jnp.int32(tile_k)
                    )

                    @pl.when((valid_start > 0) | (valid_end < jnp.int32(tile_k)))
                    def mask_boundaries():
                        mask_smem_tile(a_smem.at[slot], valid_start, valid_end, tile_m)
                        mask_smem_tile(b_smem.at[slot], valid_start, valid_end, tile_n)
                        commit_warp_smem_to_mma()

                    acc_tmem_slice = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
                    plgpu.tcgen05_mma(
                        acc_tmem_slice,
                        plgpu.transpose_ref(a_smem.at[slot], (1, 0)),
                        b_smem.at[slot],
                        consumed_barrier.at[slot],
                        accumulate=(ki > 0),
                    )

                    @pl.when(ki >= k_iters - 1)
                    def _():
                        plgpu.tcgen05_commit_arrive(mma_done_barrier.at[acc_slot])

                lax.fori_loop(0, k_iters, loop_body, None)

                # Keep the cross-warpgroup handshake when there is no MMA. Otherwise
                # StoreWG can reuse the one-bit store_done phase before its prior wait.
                @pl.when(group_size == 0)
                def signal_empty_tile():
                    plgpu.barrier_arrive(mma_done_barrier.at[acc_slot])

    @pl.when(wg_idx == STORE_WG)
    @jax.named_scope("store_wg_ragged_dw")
    def _():
        step_out_gmem = dw_gmem.at[group_id, slice_m, slice_n]
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])

        @pl.when(group_size == 0)
        def zero_store():
            if not accumulate:
                for ni in range(tile_n // epilogue_tile_n):
                    acc_smem[...] = jnp.zeros((tile_m, epilogue_tile_n), dtype=dtype)
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(
                        acc_smem,
                        step_out_gmem.at[
                            :, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)
                        ],
                    )
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

        @pl.when(group_size != 0)
        def normal_store():
            acc_tmem_slot = acc_tmem.at[:, pl.ds(acc_slot * tile_n, tile_n)]
            for ni in range(tile_n // epilogue_tile_n):
                acc_smem[...] = plgpu.async_load_tmem(
                    acc_tmem_slot.at[:, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)],
                    layout=regs_layout,
                ).astype(dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    acc_smem,
                    step_out_gmem.at[:, pl.ds(ni * epilogue_tile_n, epilogue_tile_n)],
                    reduction_op="add" if accumulate else None,
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            plgpu.wait_load_tmem()
            plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

    return step_base + k_iters


def ragged_gemm_kernel(
    lhs,
    rhs,
    group_sizes,
    tiling: TuningConfig,
    rhs_layout: RhsLayout,
    kernel_name: str,
    *,
    swiglu: bool = False,
    output_dtype=None,
    save_preact: bool = False,
    cost_estimate_num_rows=None,
):
    """Canonical ragged GEMM launcher: ``out[M, N] = lhs[M, K] @ rhs[g]``.

    ``rhs[g]`` is stored as ``[K, N]`` for ``CONTRACT_OUT`` or ``[N, K]`` for
    ``OUT_CONTRACT``.
    """
    dtype = lhs.dtype
    output_dtype = dtype if output_dtype is None else jnp.dtype(output_dtype)
    if lhs.dtype != rhs.dtype:
        raise ValueError(
            f"Matmul LHS and RHS have incompatible dtypes {lhs.dtype} vs {rhs.dtype}"
        )
    m, k_contract = lhs.shape
    if rhs_layout is RhsLayout.CONTRACT_OUT:
        num_groups, k2, n_out = rhs.shape
    else:
        num_groups, n_out, k2 = rhs.shape
    if num_groups != group_sizes.shape[0]:
        raise ValueError("RHS and group_sizes have incompatible shapes.")
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = m
    if not 0 <= cost_estimate_num_rows <= m:
        raise ValueError(
            f"Cost estimate rows {cost_estimate_num_rows} must be in [0, {m}]."
        )
    if k_contract != k2:
        raise ValueError(
            "Matmul LHS and RHS have incompatible contraction dim "
            f"{lhs.shape} vs {rhs.shape[1:]}"
        )
    if save_preact and not swiglu:
        raise ValueError("save_preact requires the SwiGLU store function.")
    if swiglu:
        if rhs_layout is not RhsLayout.CONTRACT_OUT:
            raise ValueError("SwiGLU epilogue requires CONTRACT_OUT rhs layout.")
        if n_out % 2 != 0:
            raise ValueError(f"SwiGLU epilogue requires even output dim, got {n_out}.")

    collective = tiling.collective
    tile_m, tile_n, tile_k = (tiling.tile_m, tiling.tile_n, tiling.tile_k)
    block_tile_m = tile_m
    block_tile_n = tile_n
    if collective:
        tile_m *= 2
        tile_n *= 2
    n_iters = n_out // tile_n

    max_concurrent_steps = tiling.max_concurrent_steps
    epilogue_tile_n = tiling.epilogue_tile_n
    if swiglu and epilogue_tile_n != 2 * SWIGLU_EPILOGUE_BLOCK:
        raise ValueError(
            f"SwiGLU requires epilogue_tile_n={2 * SWIGLU_EPILOGUE_BLOCK} for the "
            f"fixed interleaved gate/up layout, got {epilogue_tile_n}."
        )
    if tile_n % epilogue_tile_n != 0:
        raise ValueError(f"{tile_n=} must be divisible by {epilogue_tile_n=}")

    if m % tile_m != 0:
        raise ValueError(f"{m=} must be divisible by {tile_m=}")
    if n_out % tile_n != 0:
        raise ValueError(f"{n_out=} must be divisible by {tile_n=}")
    if k_contract % tile_k != 0:
        raise ValueError(f"{k_contract=} must be divisible by {tile_k=}")
    transforms = common.tile_swizzle_transforms((tile_k,), dtype)
    if rhs_layout is RhsLayout.CONTRACT_OUT:
        b_smem_shape = (max_concurrent_steps, tile_k, block_tile_n)
    else:
        b_smem_shape = (max_concurrent_steps, block_tile_n, tile_k)
    acc_smem_shape = (block_tile_m, epilogue_tile_n)
    # TMA row fragments can start at any row, so each SMEM row needs 128 bytes.
    swiglu_store_n = max(epilogue_tile_n // 2, 128 // output_dtype.itemsize)
    swiglu_smem_shape = (block_tile_m, swiglu_store_n)
    logical_out_n = n_out // 2 if swiglu else n_out
    max_tasks = m // tile_m + num_groups - 1
    group_info = common.GroupInfo.create_aligned(group_sizes, tile_m, max_tasks, m)

    def kernel(
        lhs_gmem,
        rhs_gmem,
        task_group_ids_gmem,
        task_block_starts_gmem,
        task_row_offsets_gmem,
        task_row_counts_gmem,
        num_tasks_gmem,
        num_rows_gmem,
        *refs,
    ):
        if save_preact:
            out_gmem, preact_gmem = refs
        else:
            (out_gmem,) = refs
            preact_gmem = None

        num_tasks = num_tasks_gmem[()]
        num_rows = num_rows_gmem[()]
        common.trap_if_rows_exceed(num_rows, m)
        grid_tile_width = tiling.grid_tile_width
        linear_grid_dyn = num_tasks * jnp.int32(n_iters)

        cluster_idx = lax.axis_index("x") if collective else jnp.int32(0)
        # Collective TMEM cannot be reused until both CTA epilogues finish reading.
        if collective:
            acc_consumed_barrier = plgpu.ClusterBarrier(
                collective_axes=("x",),
                num_arrivals=1,
                num_barriers=NUM_ACC_SLOTS,
                orders_tensor_core=True,
            )
        else:
            acc_consumed_barrier = plgpu.Barrier(
                num_arrivals=1, num_barriers=NUM_ACC_SLOTS, orders_tensor_core=True
            )

        scoped_scratch = dict(
            a_smem=plgpu.SMEM(
                (max_concurrent_steps, block_tile_m, tile_k),
                dtype,
                transforms=transforms,
            ),
            b_smem=plgpu.SMEM(b_smem_shape, dtype, transforms=transforms),
            acc_smem=plgpu.SMEM(acc_smem_shape, output_dtype),
            a_tma_barrier=plgpu.Barrier(
                num_arrivals=1, num_barriers=max_concurrent_steps
            ),
            b_tma_barrier=plgpu.Barrier(
                num_arrivals=1, num_barriers=max_concurrent_steps
            ),
            acc_consumed_barrier=acc_consumed_barrier,
            mma_done_barrier=plgpu.Barrier(
                num_arrivals=1, num_barriers=NUM_ACC_SLOTS, orders_tensor_core=True
            ),
            consumed_barrier=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=max_concurrent_steps,
                orders_tensor_core=True,
            ),
            acc_tmem=plgpu.TMEM(
                (block_tile_m, tile_n * NUM_ACC_SLOTS),
                jnp.float32,
                collective=collective,
            ),
            collective_axes=("wg",),
        )
        if swiglu:
            scoped_scratch["swiglu_smem"] = plgpu.SMEM(swiglu_smem_shape, output_dtype)

        @functools.partial(pl.run_scoped, **scoped_scratch)
        def scoped(**ref_kwargs):
            axis_index = lax.axis_index("sm")
            axis_size = lax.axis_size("sm")
            store_kwargs = dict(out_gmem=out_gmem, acc_smem=ref_kwargs["acc_smem"])
            if swiglu:
                store_kwargs.update(
                    preact_gmem=preact_gmem, swiglu_smem=ref_kwargs["swiglu_smem"]
                )
            store_tile = functools.partial(
                store_swiglu if swiglu else store_identity, **store_kwargs
            )

            @plgpu.nd_loop((linear_grid_dyn,), collective_axes="sm")
            def wrapper(loop_info: plgpu.NDLoopInfo):
                # Avoid the dynamic-grid modulo used to form ``loop_info.index[0]``.
                step = loop_info.local_index * axis_size + axis_index
                task_index, n_index = plgpu.planar_snake(
                    step,
                    (num_tasks, n_iters),
                    common.GridMinorDimension.M,
                    grid_tile_width,
                )
                group_id = task_group_ids_gmem[task_index]
                block_start = task_block_starts_gmem[task_index]
                row_offset = task_row_offsets_gmem[task_index]
                row_count = task_row_counts_gmem[task_index]
                do_ragged_gemm_tile(
                    lhs_gmem,
                    rhs_gmem.at[group_id],
                    block_start,
                    n_index,
                    cluster_idx,
                    wg_axis="wg",
                    collective_axis="x" if collective else None,
                    local_index=loop_info.local_index,
                    tiling=tiling,
                    rhs_layout=rhs_layout,
                    row_offset=row_offset,
                    row_count=row_count,
                    a_smem=ref_kwargs["a_smem"],
                    b_smem=ref_kwargs["b_smem"],
                    acc_tmem=ref_kwargs["acc_tmem"],
                    a_tma_barrier=ref_kwargs["a_tma_barrier"],
                    b_tma_barrier=ref_kwargs["b_tma_barrier"],
                    acc_consumed_barrier=ref_kwargs["acc_consumed_barrier"],
                    mma_done_barrier=ref_kwargs["mma_done_barrier"],
                    consumed_barrier=ref_kwargs["consumed_barrier"],
                    store=store_tile,
                )

    output_manual_axis_type = common.output_manual_axis_type(lhs, rhs, group_sizes)
    out_shape = jax.ShapeDtypeStruct(
        (m, logical_out_n), output_dtype, manual_axis_type=output_manual_axis_type
    )
    if save_preact:
        out_shapes = (
            out_shape,
            jax.ShapeDtypeStruct(
                (m, n_out), output_dtype, manual_axis_type=output_manual_axis_type
            ),
        )
    else:
        out_shapes = out_shape

    launch_name = f"{kernel_name}_validm_{tiling}_sm_margin={gpu.MOSAIC_SM_MARGIN}"
    flops = 2 * cost_estimate_num_rows * k_contract * n_out
    transcendentals = 0
    if swiglu:
        flops += cost_estimate_num_rows * n_out
        transcendentals = cost_estimate_num_rows * n_out // 2
    bytes_accessed = (
        cost_estimate_num_rows * k_contract * lhs.dtype.itemsize
        + rhs.size * rhs.dtype.itemsize
        + cost_estimate_num_rows * logical_out_n * output_dtype.itemsize
    )
    if save_preact:
        bytes_accessed += cost_estimate_num_rows * n_out * output_dtype.itemsize

    launch = pl.kernel(
        kernel,
        out_type=out_shapes,
        mesh=plgpu.Mesh(
            grid=persistent_grid(
                tiling, jax.local_devices()[0].core_count, gpu.MOSAIC_SM_MARGIN
            ),
            grid_names=("sm",),
            cluster=(2,) if collective else (),
            cluster_names=("x",) if collective else (),
            num_threads=2,
            thread_name="wg",
            kernel_name=launch_name,
        ),
        compiler_params=common.compiler_params_from_env(),
        cost_estimate=pl.CostEstimate(
            flops=flops, transcendentals=transcendentals, bytes_accessed=bytes_accessed
        ),
        name=launch_name,
        metadata={
            "cost_model": "uniform_unpadded",
            "cost_estimate_num_rows": str(cost_estimate_num_rows),
        },
    )
    launch_args = (
        lhs,
        rhs,
        group_info.group_id,
        group_info.block_start,
        group_info.start_within_block,
        group_info.actual_size,
        group_info.num_tasks,
        group_info.num_rows,
    )
    return launch(*launch_args)


# JaxPP task dedup does not inspect the jaxprs stored by ``mpmd_map``. These JIT
# boundaries let decoder layers reuse one traced and compiled kernel.
# See docs/compilation.md ("Cache misses") in the JaxPP repository.
@functools.partial(
    jax.jit, static_argnums=(3, 4), static_argnames=("cost_estimate_num_rows",)
)
def ragged_dot_dw_kernel(
    lhs,
    dout,
    group_sizes,
    config: TuningConfig,
    output_manual_axis_type=None,
    *,
    accumulator: jax.Array | jax.Ref | None = None,
    cost_estimate_num_rows=None,
):
    """dW kernel: ``dw[g, k, n] = sum_m lhs[m, k] * dout[m, n]``.

    Shapes: ``lhs[M, K], dout[M, N], group_sizes[G] -> dw[G, K, N]``.
    The token dimension is treated like DeepGEMM's K-grouped contiguous GEMM:
    output ``K``/``N`` are fixed per group, while the contraction span is variable
    and described by ``group_sizes``.

    Groups are compact, with offsets given by the prefix sum of ``group_sizes``.
    Boundary contraction tiles are select-zeroed outside each group before MMA,
    including any prefix loaded to satisfy TMA alignment. Groups may start at
    any row. This kernel supports only noncollective M-minor
    scheduling. An array accumulator is wrapped in a Ref and returned after
    adding dW; a caller-owned Ref is updated directly and returns ``None``.
    """
    tiling = config
    dtype = lhs.dtype
    if lhs.dtype != dout.dtype:
        raise ValueError(
            f"dW lhs and dout have incompatible dtypes {lhs.dtype} vs {dout.dtype}"
        )
    m_tokens, k_out = lhs.shape
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = m_tokens
    if not 0 <= cost_estimate_num_rows <= m_tokens:
        raise ValueError(
            f"Cost estimate rows {cost_estimate_num_rows} must be in [0, {m_tokens}]."
        )
    m2, n_out = dout.shape
    if m_tokens != m2:
        raise ValueError(
            f"dW lhs and dout have incompatible token dims {lhs.shape} vs {dout.shape}"
        )
    num_groups = group_sizes.shape[0]
    output_shape = (num_groups, k_out, n_out)
    if tiling.collective:
        raise ValueError("dW requires a non-collective tuning configuration.")
    if accumulator is not None:
        if accumulator.shape != output_shape:
            raise ValueError(
                f"dW accumulator shape {accumulator.shape} must be {output_shape}"
            )
        if accumulator.dtype != dtype:
            raise ValueError(
                f"dW accumulator dtype {accumulator.dtype} must be {dtype}"
            )

    tile_m, tile_n, tile_k = (tiling.tile_m, tiling.tile_n, tiling.tile_k)

    epilogue_tile_n = tiling.epilogue_tile_n
    if tile_n % epilogue_tile_n != 0:
        raise ValueError(f"{tile_n=} must be divisible by {epilogue_tile_n=}")
    if k_out % tile_m != 0:
        raise ValueError(f"{k_out=} must be divisible by {tile_m=}")
    if n_out % tile_n != 0:
        raise ValueError(f"{n_out=} must be divisible by {tile_n=}")
    if m_tokens >= 8 and m_tokens % 8 != 0:
        raise ValueError(
            f"{m_tokens=} must be smaller than 8 or divisible by 8 for dW TMA"
            " row tiling"
        )
    m_iters = k_out // tile_m
    n_iters = n_out // tile_n
    max_concurrent_steps = tiling.max_concurrent_steps

    transforms = common.tile_swizzle_transforms((tile_k,), dtype)

    def kernel(lhs_gmem, dout_gmem, group_starts_gmem, group_sizes_gmem, dw_gmem):
        @functools.partial(
            pl.run_scoped,
            a_smem=plgpu.SMEM(
                (max_concurrent_steps, tile_k, tile_m), dtype, transforms=transforms
            ),
            b_smem=plgpu.SMEM(
                (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
            ),
            acc_smem=plgpu.SMEM((tile_m, epilogue_tile_n), dtype),
            ab_tma_barrier=plgpu.Barrier(
                num_arrivals=2, num_barriers=max_concurrent_steps
            ),
            store_done_barrier=plgpu.Barrier(
                num_arrivals=1, num_barriers=NUM_ACC_SLOTS, orders_tensor_core=True
            ),
            mma_done_barrier=plgpu.Barrier(
                num_arrivals=1, num_barriers=NUM_ACC_SLOTS, orders_tensor_core=True
            ),
            consumed_barrier=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=max_concurrent_steps,
                orders_tensor_core=True,
            ),
            acc_tmem=plgpu.TMEM(
                (tile_m, tile_n * NUM_ACC_SLOTS), jnp.float32, collective=False
            ),
            collective_axes=("wg",),
        )
        def scoped(**ref_kwargs):
            @plgpu.nd_loop(
                (num_groups * m_iters * n_iters,),
                collective_axes="sm",
                init_carry=jnp.int32(0),
            )
            def dw_loop(loop_info: plgpu.NDLoopInfo, carry):
                (lin_idx,) = loop_info.index
                group_id = lin_idx // (m_iters * n_iters)
                mn_idx = lin_idx - group_id * (m_iters * n_iters)
                m_index, n_index = plgpu.planar_snake(
                    mn_idx,
                    (m_iters, n_iters),
                    common.GridMinorDimension.M,
                    tiling.grid_tile_width,
                )
                return do_ragged_dw_tile(
                    lhs_gmem,
                    dout_gmem,
                    dw_gmem,
                    group_starts_gmem,
                    group_sizes_gmem,
                    group_id,
                    m_index,
                    n_index,
                    wg_axis="wg",
                    local_index=loop_info.local_index,
                    step_base=carry,
                    tiling=tiling,
                    accumulate=accumulator is not None,
                    **ref_kwargs,
                )

            if accumulator is not None:

                @pl.when(lax.axis_index("wg") == 1)
                def wait_for_reduction_stores():
                    plgpu.wait_smem_to_gmem(0)

    if output_manual_axis_type is None:
        output_manual_axis_type = common.output_manual_axis_type(lhs, dout, group_sizes)
    if (
        accumulator is not None
        and jax.typeof(accumulator).manual_axis_type != output_manual_axis_type
    ):
        raise ValueError(
            "dW accumulator and output have incompatible manual axis types "
            f"{jax.typeof(accumulator).manual_axis_type} vs "
            f"{output_manual_axis_type}"
        )
    launch_name = (
        f"ragged_dot_dw_kernel_validk_"
        f"{'accumulate_' if accumulator is not None else ''}"
        f"{config}_sm_margin={gpu.MOSAIC_SM_MARGIN}"
    )
    compiler_params = common.compiler_params_from_env()
    grid = persistent_grid(
        config, jax.local_devices()[0].core_count, gpu.MOSAIC_SM_MARGIN
    )
    group_starts = jnp.cumulative_sum(group_sizes, include_initial=True)[:-1]
    output_elements = num_groups * k_out * n_out
    bytes_accessed = (
        cost_estimate_num_rows * k_out * lhs.dtype.itemsize
        + cost_estimate_num_rows * n_out * dout.dtype.itemsize
        + output_elements * dtype.itemsize
    )
    if accumulator is not None:
        bytes_accessed += output_elements * dtype.itemsize
    cost_estimate = pl.CostEstimate(
        flops=(
            2 * cost_estimate_num_rows * k_out * n_out
            + (output_elements if accumulator is not None else 0)
        ),
        transcendentals=0,
        bytes_accessed=bytes_accessed,
    )
    out_type = (
        ()
        if accumulator is not None
        else jax.ShapeDtypeStruct(
            output_shape, dtype, manual_axis_type=output_manual_axis_type
        )
    )
    launch = pl.kernel(
        kernel,
        out_type=out_type,
        mesh=plgpu.Mesh(
            grid=grid,
            grid_names=("sm",),
            num_threads=2,
            thread_name="wg",
            kernel_name=launch_name,
        ),
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
        name=launch_name,
        metadata={
            "cost_model": "uniform_unpadded",
            "cost_estimate_num_rows": str(cost_estimate_num_rows),
        },
    )
    if isinstance(accumulator, jax.Ref):
        launch(lhs, dout, group_starts, group_sizes, accumulator)
        return None
    if accumulator is not None:
        accumulator_ref = jax.new_ref(accumulator)
        launch(lhs, dout, group_starts, group_sizes, accumulator_ref)
        return accumulator_ref[...]
    return launch(lhs, dout, group_starts, group_sizes)


# JaxPP task dedup does not inspect the jaxprs stored by ``mpmd_map``. These JIT
# boundaries let decoder layers reuse one traced and compiled kernel.
# See docs/compilation.md ("Cache misses") in the JaxPP repository.
@functools.partial(jax.jit, static_argnums=(3, 4), static_argnames=("output_dtype",))
def ragged_dot_kernel(
    a,
    b,
    group_sizes,
    config: TuningConfig,
    cost_estimate_num_rows=None,
    *,
    output_dtype=None,
):
    """Same surface as upstream's ``ragged_dot_kernel``.

    The persistent scheduler consumes an 8-row-aligned task table derived from
    ``group_sizes`` and runs only tasks that own live rows.
    """
    return ragged_gemm_kernel(
        a,
        b,
        group_sizes,
        config,
        RhsLayout.CONTRACT_OUT,
        "ragged_dot_kernel",
        output_dtype=output_dtype,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def ragged_dot_swiglu_kernel(
    a,
    b_interleaved,
    group_sizes,
    config: TuningConfig,
    output_dtype=None,
    save_preact: bool = False,
    cost_estimate_num_rows=None,
):
    """Fused first FFN projection + SwiGLU.

    ``b_interleaved`` uses the fixed gate/up layout shared with packing and backward:
    ``[gate_0:gate_B, up_0:up_B, gate_B:gate_2B, up_B:up_2B, ...]`` where
    ``B = SWIGLU_EPILOGUE_BLOCK``. ``config.epilogue_tile_n`` must equal ``2 * B``.
    The MMA computes that interleaved ``[M, 2H]`` tile, and the store epilogue
    writes ``silu(gate_h) * up_h`` directly to ``[M, H]``.
    """
    return ragged_gemm_kernel(
        a,
        b_interleaved,
        group_sizes,
        config,
        RhsLayout.CONTRACT_OUT,
        "ragged_dot_swiglu_kernel",
        swiglu=True,
        output_dtype=output_dtype,
        save_preact=save_preact,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


@functools.partial(jax.jit, static_argnums=(3, 4))
def ragged_dot_dx_kernel(
    dy, rhs, group_sizes, config: TuningConfig, cost_estimate_num_rows=None
):
    """dX kernel: ``dx[m, k] = sum_n dy[m, n] * rhs[g(m), k, n]``.

    Shapes: ``dy[M, N], rhs[G, K, N], group_sizes[G] -> dx[M, K]``.
    ``config.tile_n`` tiles the output ``K`` axis and ``config.tile_k`` tiles the
    contraction ``N`` axis.
    The RHS SMEM tile is presented to UMMA with a metadata-only transpose, so
    no HBM or SMEM transpose is materialized.
    """
    return ragged_gemm_kernel(
        dy,
        rhs,
        group_sizes,
        config,
        RhsLayout.OUT_CONTRACT,
        "ragged_dot_dx_kernel",
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


# Tile shape rationale (DeepSeek-v3 671B @ EP=8, B=1, S=4096):
#   M_per_shard = 8192, K/N in {2048, 4096, 7168}, E_local = 32.
#   All N are divisible by the 256-wide collective cluster tile, and K by 64.
#
# Since JAX 0.10.0, Pallas core-map manual axes are scoped during
# `check_vma=True`, so the Mosaic-GPU cluster axis used by the collective
# kernel no longer leaks into the surrounding shard_map.
DEFAULT_CONFIG = TuningConfig(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    max_concurrent_steps=6,
    collective=True,
    grid_tile_width=8,
)

# FP32 SwiGLU output doubles epilogue scratch. Five pipeline stages keep the
# complete scratch tree under the SM100 per-CTA shared-memory limit.
FP32_SWIGLU_CONFIG = dataclasses.replace(DEFAULT_CONFIG, max_concurrent_steps=5)

# dW-relative config: logical output rows are lhs hidden K, output columns are
# dout hidden N, and the contraction tile is tokens within a group.
# Keep four pipeline stages because a fifth exceeds the dynamic shared-memory
# limit with this tile shape.
DW_DEFAULT_CONFIG = TuningConfig(
    tile_m=128,
    tile_n=256,
    tile_k=64,
    max_concurrent_steps=4,
    collective=False,
    grid_tile_width=2,
    epilogue_tile_n=128,
)

DW_WIDE_K_CONFIG = dataclasses.replace(DW_DEFAULT_CONFIG, grid_tile_width=8)
DW_TILE_N128_CONFIG = dataclasses.replace(DW_DEFAULT_CONFIG, tile_n=128)


def ragged_dot_swiglu_config(
    input_dtype: jnp.dtype, output_dtype: jnp.dtype | None
) -> TuningConfig:
    output_dtype = input_dtype if output_dtype is None else output_dtype
    return (
        FP32_SWIGLU_CONFIG
        if jnp.dtype(output_dtype) == jnp.dtype(jnp.float32)
        else DEFAULT_CONFIG
    )


def ragged_dot_dw_config(k_out: int, n_out: int) -> TuningConfig:
    """Choose the measured dW raster policy for an output ``[G, K, N]``."""
    if n_out % DW_DEFAULT_CONFIG.tile_n:
        return DW_TILE_N128_CONFIG
    if k_out > n_out:
        return DW_WIDE_K_CONFIG
    return DW_DEFAULT_CONFIG


@functools.partial(jax.jit, static_argnums=(3,), static_argnames=("output_dtype",))
def ragged_dot(
    lhs, rhs, group_sizes, cost_estimate_num_rows=None, *, output_dtype=None
):
    """Raw grouped product; differentiation is defined by the shared operation."""
    return ragged_dot_kernel(
        lhs,
        rhs,
        group_sizes,
        DEFAULT_CONFIG,
        cost_estimate_num_rows,
        output_dtype=output_dtype,
    )


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
        accumulator=accumulator,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


def ragged_dot_dx(
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    cost_estimate_num_rows=None,
) -> jax.Array:
    """Compute the grouped input gradient using the SM100 ragged kernel."""
    return ragged_dot_dx_kernel(
        dout.astype(rhs.dtype), rhs, group_sizes, DEFAULT_CONFIG, cost_estimate_num_rows
    )


def swiglu_interleaved_bwd_kernel(
    preact_ref,
    dy_ref,
    group_sizes_ref,
    out_ref,
    preact_smem,
    dy_smem,
    dg_smem,
    du_smem,
    preact_barrier,
    dy_barrier,
    *,
    BLOCK_T: int,
    BLOCK_H: int,
):
    t_block = lax.axis_index("tokens")
    h_block = lax.axis_index("hidden")
    t_off = t_block * BLOCK_T
    h_off = h_block * BLOCK_H
    preact_h_off = h_block * (2 * BLOCK_H)

    valid_t = jnp.int32(0)
    for i in range(group_sizes_ref.shape[0]):
        valid_t += group_sizes_ref[i]

    @pl.when(t_off < valid_t)
    def _():
        plgpu.copy_gmem_to_smem(
            preact_ref.at[pl.ds(t_off, BLOCK_T), pl.ds(preact_h_off, 2 * BLOCK_H)],
            preact_smem,
            preact_barrier,
        )
        plgpu.copy_gmem_to_smem(
            dy_ref.at[pl.ds(t_off, BLOCK_T), pl.ds(h_off, BLOCK_H)], dy_smem, dy_barrier
        )
        plgpu.barrier_wait(preact_barrier)
        plgpu.barrier_wait(dy_barrier)

        preact_tile = preact_smem[...].astype(jnp.float32)
        dy_tile = dy_smem[...].astype(jnp.float32)
        g = preact_tile[:, :BLOCK_H]
        u = preact_tile[:, BLOCK_H:]

        sig = jax.nn.sigmoid(g)
        silu_g = g * sig
        dsilu_dg = sig * (1.0 + g * (1.0 - sig))

        dg = dy_tile * u * dsilu_dg
        du = dy_tile * silu_g

        dg_smem[...] = dg.astype(dg_smem.dtype)
        du_smem[...] = du.astype(du_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            dg_smem, out_ref.at[pl.ds(t_off, BLOCK_T), pl.ds(preact_h_off, BLOCK_H)]
        )
        plgpu.copy_smem_to_gmem(
            du_smem,
            out_ref.at[pl.ds(t_off, BLOCK_T), pl.ds(preact_h_off + BLOCK_H, BLOCK_H)],
        )
        plgpu.wait_smem_to_gmem(0)


# The JIT boundary reuses one traced grid across decoder layers. Donation keeps
# the mutable preactivation Ref in place.
@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(3,))
def swiglu_backward_in_place(
    preact_interleaved, dy, group_sizes, cost_estimate_num_rows=None
):
    T, twoH = preact_interleaved.shape
    H = twoH // 2
    if cost_estimate_num_rows is None:
        cost_estimate_num_rows = T
    if not 0 <= cost_estimate_num_rows <= T:
        raise ValueError(
            f"Cost estimate rows {cost_estimate_num_rows} must be in [0, {T}]."
        )
    block_t = 128
    block_h = SWIGLU_EPILOGUE_BLOCK
    kernel = functools.partial(
        swiglu_interleaved_bwd_kernel, BLOCK_T=block_t, BLOCK_H=block_h
    )
    full_transforms = common.tile_swizzle_transforms(
        (2 * block_h,), preact_interleaved.dtype
    )
    half_transforms = common.tile_swizzle_transforms(
        (block_h,), preact_interleaved.dtype
    )

    def launch_body(preact_ref, dy_ref, group_sizes_ref):
        # A single mutable Ref is both input and output. Each grid tile loads
        # its own preactivation before overwriting it with the cotangent.
        pl.run_scoped(
            functools.partial(kernel, preact_ref, dy_ref, group_sizes_ref, preact_ref),
            plgpu.SMEM(
                (block_t, 2 * block_h),
                preact_interleaved.dtype,
                transforms=full_transforms,
            ),
            plgpu.SMEM(
                (block_t, block_h), preact_interleaved.dtype, transforms=half_transforms
            ),
            plgpu.SMEM(
                (block_t, block_h), preact_interleaved.dtype, transforms=half_transforms
            ),
            plgpu.SMEM(
                (block_t, block_h), preact_interleaved.dtype, transforms=half_transforms
            ),
            plgpu.Barrier(),
            plgpu.Barrier(),
        )

    preact_ref = jax.new_ref(preact_interleaved)
    pl.kernel(
        launch_body,
        out_type=(),
        mesh=plgpu.Mesh(
            grid=(T // block_t, H // block_h), grid_names=("tokens", "hidden")
        ),
        compiler_params=plgpu.CompilerParams(),
        cost_estimate=pl.CostEstimate(
            flops=8 * cost_estimate_num_rows * H,
            transcendentals=cost_estimate_num_rows * H,
            bytes_accessed=(
                5 * cost_estimate_num_rows * H * preact_interleaved.dtype.itemsize
            ),
        ),
        name="swiglu_interleaved_bwd_in_place",
        metadata={
            "cost_model": "uniform_unpadded",
            "cost_estimate_num_rows": str(cost_estimate_num_rows),
        },
    )(preact_ref, dy.astype(preact_interleaved.dtype), group_sizes)
    return preact_ref[...]


def ragged_dot_swiglu(
    lhs,
    rhs_interleaved,
    group_sizes,
    out_dtype=None,
    cost_estimate_num_rows=None,
    *,
    save_preact=False,
):
    """Fused projection and SwiGLU epilogue; ``save_preact`` also returns preact."""
    return ragged_dot_swiglu_kernel(
        lhs,
        rhs_interleaved,
        group_sizes,
        ragged_dot_swiglu_config(lhs.dtype, out_dtype),
        out_dtype,
        save_preact,
        cost_estimate_num_rows,
    )


def swiglu_backward(preact, dout, group_sizes, cost_estimate_num_rows=None):
    """Preactivation cotangent, written in place over the donated ``preact``."""
    return swiglu_backward_in_place(preact, dout, group_sizes, cost_estimate_num_rows)
