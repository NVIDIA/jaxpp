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

export SLURM_LOG_TAG="${SLURM_LOG_TAG:-maximoe-deepseek3-671b-h100-16k-dropless}"

# Unfused steady-state tasks need much smaller temporaries than fused
# forward/backward pairs (about 10.5 GiB, which the fragmented pool cannot place
# on stages 8-12). That leaves room to keep all optimizer state on device,
# removing the host reload and write-back from the step tail.

exec "${script_dir}/dsv3-h100-16k.sh" \
  --capacity-factor -1 \
  --no-fuse-steady-state \
  "$@"
