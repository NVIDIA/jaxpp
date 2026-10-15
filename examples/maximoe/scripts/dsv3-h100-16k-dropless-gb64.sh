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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SLURM_LOG_TAG="${SLURM_LOG_TAG:-maximoe-deepseek3-671b-h100-16k-dropless-gb64}"

# Global batch 64: 32 one-sequence microbatches per DATA replica halve the
# pipeline bubble relative to dsv3-h100-16k-dropless.sh. The extra in-flight
# activations leave too little of an 85% pool with all optimizer state on device.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.90}"

exec "${script_dir}/dsv3-h100-16k-dropless.sh" \
  --per-device-batch-size 0.25 \
  --num-microbatches 32 \
  "$@"
