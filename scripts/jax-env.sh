#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Sourceable JAX environment shared by the local and Slurm launchers. The
# caller provides process rank, coordinator, device, and output paths.
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_IP}:${JAX_COORDINATOR_PORT}"

jax_env_output_root="${OUTPUT_ROOT:-/tmp/jaxpp-logs}"
export JAX_DUMP_IR_TO="${jax_env_output_root}/dumps/${NODE_RANK}/jaxprs"
export JAX_DUMP_IR_MODES="jaxpr,stablehlo"
export JAXPP_DUMP_DIR="${jax_env_output_root}/dumps/${NODE_RANK}/jaxprs"

export JAX_LOG_CHECKPOINT_RESIDUALS=True
export JAX_TRACEBACK_FILTERING=off

export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=${jax_env_output_root}/dumps/${NODE_RANK}/hlos"

# MOSAIC_GPU_DUMP_TO is the only way to retain Mosaic MLIR, PTX, SASS, and
# assembler diagnostics as files.
export MOSAIC_GPU_DUMP_TO="${jax_env_output_root}/dumps/${NODE_RANK}/mosaic"
mkdir -p "${MOSAIC_GPU_DUMP_TO}"

export TF_CPP_MIN_LOG_LEVEL=0
export TF_BFC_MEMORY_DUMP="${jax_env_output_root}/bfc_memdumps/${NODE_RANK}/memdump"
mkdir -p "$(dirname "${TF_BFC_MEMORY_DUMP}")"

if [[ "${JAXPP_DEBUG_HANG:-0}" == "1" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_execution_progress_tracking=20"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_execution_terminate_timeout=4m"
  export TF_CPP_VMODULE=thunk_executor=1,collective_thunk=1,nccl_collectives=5,collective_cliques=4,gpu_executable=1
  export CUDA_LAUNCH_BLOCKING=1
  if [[ "${JAXPP_DEBUG_HANG_NCCL:-0}" == "1" ]]; then
    mkdir -p "${jax_env_output_root}/nccl"
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,COLL,NET,GRAPH,TUNING,ENV
  fi
fi

unset jax_env_output_root
