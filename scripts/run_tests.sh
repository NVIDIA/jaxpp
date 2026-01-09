#!/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

pytest --ignore tests/test_mpmd_array.py
pytest tests/test_mpmd_array.py
python examples/basic.py


if [ $(nvidia-smi -L | wc -l) -ge 8 ]; then
    N_PROCS=2 N_GPUS=4 COMMAND="python -u tests/test_reshard_utils.py" ./scripts/local_mc.sh
    N_PROCS=2 N_GPUS=4 COMMAND="python -u examples/mpmd_reshard.py" ./scripts/local_mc.sh
    N_PROCS=2 N_GPUS=4 COMMAND="python -u tests/test_dime2.py" ./scripts/local_mc.sh
    N_PROCS=2 N_GPUS=4 COMMAND="python -u examples/internal/issue_7.py" ./scripts/local_mc.sh
fi
