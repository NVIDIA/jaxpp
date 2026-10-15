#!/usr/bin/env bash

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

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: job.sh --container-image <image> --workspace <host-path> --workdir <host-path> --output <host-path> --process-mode <per_gpu|per_node> --num-nodes <N> --ntasks-per-node <N> [--setup-script <project-path>] -- <command> [args...]
EOF
}

container_image=""
workspace=""
workdir=""
output=""
coordinator_port="${JAX_COORDINATOR_PORT:-6789}"
process_mode=""
num_nodes=""
ntasks_per_node=""
setup_script_path=""

while (($#)); do
  case "$1" in
    --container-image)
      if (($# < 2)); then
        echo "Missing value for --container-image" >&2
        exit 2
      fi
      container_image="$2"
      shift 2
      ;;
    --workspace)
      if (($# < 2)); then
        echo "Missing value for --workspace" >&2
        exit 2
      fi
      workspace="$2"
      shift 2
      ;;
    --workdir)
      if (($# < 2)); then
        echo "Missing value for --workdir" >&2
        exit 2
      fi
      workdir="$2"
      shift 2
      ;;
    --output)
      if (($# < 2)); then
        echo "Missing value for --output" >&2
        exit 2
      fi
      output="$2"
      shift 2
      ;;
    --process-mode)
      if (($# < 2)); then
        echo "Missing value for --process-mode" >&2
        exit 2
      fi
      process_mode="$2"
      shift 2
      ;;
    --num-nodes)
      if (($# < 2)); then
        echo "Missing value for --num-nodes" >&2
        exit 2
      fi
      num_nodes="$2"
      shift 2
      ;;
    --ntasks-per-node)
      if (($# < 2)); then
        echo "Missing value for --ntasks-per-node" >&2
        exit 2
      fi
      ntasks_per_node="$2"
      shift 2
      ;;
    --setup-script)
      if (($# < 2)); then
        echo "Missing value for --setup-script" >&2
        exit 2
      fi
      setup_script_path="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${container_image}" || -z "${workspace}" || -z "${workdir}" || -z "${output}" || -z "${num_nodes}" || -z "${ntasks_per_node}" || -z "${process_mode}" ]]; then
  usage >&2
  exit 2
fi
if (($# == 0)); then
  echo "Missing command after --" >&2
  usage >&2
  exit 2
fi
if ! [[ "${coordinator_port}" =~ ^[0-9]+$ ]]; then
  echo "--coordinator-port must be numeric: ${coordinator_port}" >&2
  exit 2
fi

mapfile -t nodes_array < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#nodes_array[@]} == 0)); then
  echo "Failed to resolve SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}" >&2
  exit 1
fi

node_srun_flags=()
if ((num_nodes > ${#nodes_array[@]})); then
  echo "--num-nodes (${num_nodes}) exceeds available nodes (${#nodes_array[@]})" >&2
  exit 1
fi
nodes_array=("${nodes_array[@]:0:${num_nodes}}")
nodelist=$(IFS=,; echo "${nodes_array[*]}")
node_srun_flags=(-N "${num_nodes}" -w "${nodelist}")

export JAX_COORDINATOR_IP="${JAX_COORDINATOR_IP:-${nodes_array[0]}}"
export JAX_COORDINATOR_PORT="${coordinator_port}"
export OUTPUT_ROOT=/output

# Reusing a container carries setup state, including failed CUDA compatibility
# probes, between launches. The output directory is unique for each launch.
container_name="jaxpp-${SLURM_JOB_ID}-${output##*/}"
container_srun_flags=(
  "--container-image=${container_image}"
  "--container-mounts=${workspace}:/workdir/jaxpp,${output}:/output"
  "--container-workdir=/workdir/jaxpp${workdir#"${workspace}"}"
  --no-container-mount-home
  --no-container-remap-root
  "--container-name=${container_name}"
)

if [[ -n "${setup_script_path}" ]]; then
  echo "Running setup: ${setup_script_path} (1 task/node, ${num_nodes} nodes, container: ${container_name})" >&2
  srun --ntasks="${num_nodes}" --ntasks-per-node=1 \
    "${node_srun_flags[@]}" \
    "${container_srun_flags[@]}" \
    -o setup-%t-%j-%N.out -e setup-%t-%j-%N.out \
    bash "${setup_script_path}"
fi

main_step_srun_overrides=(
  --ntasks="$((num_nodes * ntasks_per_node))"
  --ntasks-per-node="${ntasks_per_node}"
)

exec srun \
  "${main_step_srun_overrides[@]}" \
  "${node_srun_flags[@]}" \
  "${container_srun_flags[@]}" \
  -o output-%t-%j-%N.out -e output-%t-%j-%N.out \
  bash -c '
    process_mode="$1"
    shift
    export NNODES="${SLURM_NTASKS}"
    export NODE_RANK="${SLURM_PROCID}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=, read -ra visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    else
      mapfile -t visible_devices < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
    if ((${#visible_devices[@]} == 0)); then
      echo "No GPUs found for rank ${NODE_RANK}." >&2
      exit 1
    fi
    num_visible_devices="${#visible_devices[@]}"
    if [[ "${process_mode}" == "per_node" ]]; then
      export N_GPUS="${num_visible_devices}"
      export JAX_LOCAL_DEVICE_IDS="$(seq -s, 0 $((N_GPUS - 1)))"
    else
      export N_GPUS=1
      if ((num_visible_devices == 1)); then
        export JAX_LOCAL_DEVICE_IDS=0
      elif ((SLURM_LOCALID < num_visible_devices)); then
        export JAX_LOCAL_DEVICE_IDS="${SLURM_LOCALID}"
      else
        echo "SLURM_LOCALID=${SLURM_LOCALID} exceeds ${num_visible_devices} visible GPUs." >&2
        exit 1
      fi
    fi
    echo "JAX rank ${NODE_RANK}: process_mode=${process_mode} SLURM_LOCALID=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} JAX_LOCAL_DEVICE_IDS=${JAX_LOCAL_DEVICE_IDS}" >&2
    source /workdir/jaxpp/scripts/jax-env.sh && exec "$@"
  ' jaxpp-launch "${process_mode}" "$@"
