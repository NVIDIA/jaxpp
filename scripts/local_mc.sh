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

set -u
set -o pipefail

if [ -z "${N_PROCS:-}" ] || [ -z "${N_GPUS:-}" ] || [ -z "${COMMAND:-}" ]; then
  echo "N_PROCS, N_GPUS, and COMMAND must be set"
  exit 1
fi



# Default coordinator setup
export JAX_COORDINATOR_IP="${JAX_COORDINATOR_IP:-localhost}"
export JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-1234}"
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_IP}:${JAX_COORDINATOR_PORT}"
export NNODES=$N_PROCS

PIDS=()

# Cleanup function to kill all child processes
cleanup() {
  echo "Cleaning up..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
    fi
  done
}

trap cleanup EXIT SIGINT SIGTERM

echo "Starting $N_PROCS processes with $N_GPUS GPUs each..."
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"

for ((i=0; i<N_PROCS; i++)); do
  start=$((i * N_GPUS))
  end=$((start + N_GPUS - 1))

  # Calculate CUDA_VISIBLE_DEVICES
  if [ "$N_GPUS" -gt 0 ]; then
    DEVICES=$(seq -s, $start $end)
  else
    DEVICES=""
  fi

  # Launch process in background
  (
    export NODE_RANK=$i
    export CUDA_VISIBLE_DEVICES=$DEVICES

    # Redirect output to file and also pipe to stdout with prefix for visibility
    echo "Process $i: CUDA_VISIBLE_DEVICES=$DEVICES"
    $COMMAND 2>&1 | tee "output_${i}.log" | sed -u "s/^/$i: /"
  ) &

  pid=$!
  PIDS+=("$pid")
  echo "Launched process $i with PID $pid"
done

# Monitor processes
# We expect N_PROCS completions.
# fail-fast: if any process returns non-zero, we exit immediately (triggering cleanup).
for ((i=0; i<N_PROCS; i++)); do
  wait -n
  CODE=$?
  if [ "$CODE" -ne 0 ]; then
    echo "A process failed with exit code $CODE"
    exit "$CODE"
  fi
done

echo "All processes completed successfully."
