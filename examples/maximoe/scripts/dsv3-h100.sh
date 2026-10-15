#!/usr/bin/env bash

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

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

export SLURM_LOG_TAG="${SLURM_LOG_TAG:-maximoe-deepseek3-671b-h100}"
export JAXPP_CLIENT_TIMEOUT=600000
export JAXPP_DISABLE_PREVENT_CSE=0
# Leave device memory outside the XLA pool for NCCL communicator allocation.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.85}"
export NCCL_NVLS_ENABLE=0
export RAGGED_DOT_SM_MARGIN=4
# FUSION command buffers fail graph construction for this EP=8 layout, so
# command buffers are disabled.
XLA_FLAGS="
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_command_buffer=
--xla_gpu_enable_triton_gemm=false
--xla_gpu_all_gather_combine_threshold_bytes=268435456
--xla_gpu_reduce_scatter_combine_threshold_bytes=67108864
--xla_disable_hlo_passes=rematerialization
"

# Each MPMD module is compiled by only some processes, so XLA cannot collect
# autotuning shards from the full process set.
XLA_FLAGS+=" --xla_gpu_shard_autotuning=false"

# The after_loop otherwise OOMs. A slop factor of 50 keeps the PP=16 optimizer
# temporary below the largest H100 BFC hole while retaining the established
# DeepSeek performance schedule.
XLA_FLAGS+=" --xla_gpu_memory_limit_slop_factor=50"

# TODO: Necessary for JAX 0.11.x.
XLA_FLAGS+=" --xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"

export XLA_FLAGS

launcher_args=(--num-nodes 32)

exec ../../scripts/slurm/launch \
  "${launcher_args[@]}" \
  -- \
  python -u -m maximoe.train \
  --param-dtype fp32 \
  --pp 16 \
  --data 1 \
  --fsdp 2 \
  --ep 8 \
  --zero 1 \
  --schedule kimik2 \
  --num-pipeline-repeats 2 \
  --fuse-steady-state \
  --no-use-ring-of-experts \
  --capacity-factor 2 \
  --host-offload-regex '^\.(opt\.(mu|nu)\.|params\.layers\[\d+\]\.ffn\.wo$)' \
  --per-device-batch-size 1 \
  --num-microbatches 16 \
  --seq-length 4096 \
  --num-layers 61 \
  --num-dense-layers 3 \
  --num-train-steps 20 \
  --profile-dir /output/profiles/h100 \
  --num-routing-groups 1 \
  --num-selected-groups 1 \
  --moe-aux-loss-weight 0.0001 \
  --router-bias-update-rate 0.001 \
  --mtp-num-layers 0 \
  "$@"
