#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e
test_path=$(dirname $(realpath -s $0))

set -x
python3 "$test_path/test_local_sharded_communication.py" -vv

# non-interleaved
RAY_ADDRESS=local python3 benchmarks/t5x/jaxpp_decoder_only_test_no_sharded_init_hf_loss.py \
    --model=small --dtype=float16 --num_steps=10 --spmd_mesh_shape='(1,2)' \
    --num_microbatches=32 --microbatch_size=2 --jaxpp_mesh_shape='(1,4)' --distributed_initialization \
    --vocab_size=51200 --decoder_seq_len=2048 --use_jaxpp --disable_te_mha --log_dir=debug

# multi-host stages
RAY_ADDRESS=local python3 benchmarks/t5x/jaxpp_decoder_only_test_no_sharded_init_hf_loss.py \
    --model=small --vocab_size=51200 --dtype=float16 \
    --num_steps=10 --spmd_mesh_shape='(2,2)' \
    --use_jaxpp --jaxpp_mesh_shape='(1,2)' --distributed_initialization \
    --num_microbatches=2 --microbatch_size=2 --decoder_seq_len=2048 --disable_te_mha --log_dir=debug

# interleaved
RAY_ADDRESS=local python3 benchmarks/t5x/jaxpp_decoder_only_test_no_sharded_init_hf_loss.py \
    --model=small --dtype=float16 --num_steps=10 --spmd_mesh_shape='(1,2)' --num_microbatches=32 \
    --microbatch_size=2 --jaxpp_mesh_shape='(1,2)' --distributed_initialization --vocab_size=51200 \
    --decoder_seq_len=2048 --use_jaxpp --pp_interleaving=4 --schedule=interleaved_1f1b --disable_te_mha \
    --log_dir=debug

RAY_ADDRESS=local python benchmarks/basic.py
# python3 test_sharded_communication.py -vv

echo "DONE"
