#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

N_PROCS="${N_PROCS:-2}"
N_GPUS="${N_GPUS:-2}"

python -m pytest --ignore tests/test_mpmd_array.py
python -m pytest tests/test_mpmd_array.py
JAXPP_FAST_INFER_SHARDINGS=1 \
    JAXPP_DEBUG_FORCE_MPMDIFY=True \
    JAXPP_ENABLE_LICM=True \
    python examples/basic.py --train_steps=10

if command -v nvidia-smi >/dev/null 2>&1; then
    available_gpus=$(nvidia-smi -L | wc -l)
else
    available_gpus=0
fi

required_gpus=$((N_PROCS * N_GPUS))
if [ "$available_gpus" -ge "$required_gpus" ]; then
    N_PROCS="$N_PROCS" N_GPUS="$N_GPUS" COMMAND="python -u tests/test_reshard_utils.py" ./scripts/local_mc.sh
    N_PROCS="$N_PROCS" N_GPUS="$N_GPUS" COMMAND="python -u examples/mpmd_reshard.py" ./scripts/local_mc.sh
    N_PROCS="$N_PROCS" N_GPUS="$N_GPUS" COMMAND="python -u tests/test_dime2.py" ./scripts/local_mc.sh
else
    echo "Skipping multi-process tests: need ${required_gpus} GPUs, found ${available_gpus}."
fi
