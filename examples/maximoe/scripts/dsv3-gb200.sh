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

export SLURM_LOG_TAG=maximoe-deepseek3-671b-gb200
export JAXPP_CLIENT_TIMEOUT=600000
export JAXPP_DIRECTIONAL_COMMUNICATORS=False
export JAXPP_DISABLE_PREVENT_CSE=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export NCCL_NVLS_ENABLE=0
export RAGGED_DOT_SM_MARGIN=4
# FUSION command buffers are valid for this PP=4, EP=16 GB200 layout.
XLA_FLAGS="
--xla_gpu_first_collective_call_terminate_timeout_seconds=300
--xla_gpu_executable_terminate_timeout=300
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_command_buffer=FUSION
--xla_gpu_enable_triton_gemm=false
--xla_gpu_all_gather_combine_threshold_bytes=268435456
--xla_gpu_reduce_scatter_combine_threshold_bytes=67108864
--xla_disable_hlo_passes=rematerialization
"

# Each MPMD module is compiled by only some processes, so XLA cannot collect
# autotuning shards from the full process set.
XLA_FLAGS+=" --xla_gpu_shard_autotuning=false"

# XLA in JAX 0.11.x shares its preallocated BFC arena with collective
# allocations by default. NCCL cannot register that 147 GiB backing range as a
# symmetric window, so keep collective buffers in their separate growable
# allocator.
XLA_FLAGS+=" --xla_gpu_enable_allocator_spatial_partitioning=false"

export XLA_FLAGS

launcher_args=(--num-nodes 16)

exec ../../scripts/slurm/launch \
  "${launcher_args[@]}" \
  -- \
  python -u -m maximoe.train \
  --param-dtype bf16 \
  --pp 4 \
  --data 1 \
  --fsdp 1 \
  --ep 16 \
  --schedule kimik2 \
  --num-pipeline-repeats 8 \
  --fuse-steady-state \
  --no-use-ring-of-experts \
  --capacity-factor 1.2 \
  --per-device-batch-size 4 \
  --num-microbatches 16 \
  --seq-length 4096 \
  --num-layers 61 \
  --num-dense-layers 3 \
  --num-train-steps 20 \
  --profile-dir /output/profiles/gb200 \
  --num-routing-groups 1 \
  --num-selected-groups 1 \
  --moe-aux-loss-weight 0.0001 \
  --router-bias-update-rate 0.001 \
  --mtp-num-layers 0 \
  "$@"
