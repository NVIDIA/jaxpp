#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail

# Keep the version and installer digest in sync. The verified installer also
# checks the SHA-256 of the platform-specific uv archive before installing it.
readonly UV_VERSION="0.13.4"
readonly UV_INSTALLER_SHA256="1f5390d273b0158bd5eaa3527ae049f3667baa881abb1e7c883dd637d84432f9"
readonly UV_INSTALLER_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[failed] VIRTUAL_ENV variable is not set."
    exit 1
fi

if [ "$#" -ne 0 ]; then
    echo "[failed] setup-env.sh does not accept arguments; edit pyproject.toml and run uv lock instead."
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "${script_dir}/..")"
echo "Creating env ${VIRTUAL_ENV} from ${repo_root}/uv.lock"

uv_installer="$(mktemp)"
trap 'rm -f "${uv_installer}"' EXIT

curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
    --location --silent --show-error --fail \
    "${UV_INSTALLER_URL}" --output "${uv_installer}"
printf '%s  %s\n' "${UV_INSTALLER_SHA256}" "${uv_installer}" | sha256sum --check -
UV_UNMANAGED_INSTALL="${HOME}/.local/bin" sh "${uv_installer}"

rm -f "${uv_installer}"
trap - EXIT

# The installer omits its env helper when the install directory is already on PATH.
export PATH="${HOME}/.local/bin:${PATH}"

uv venv --python 3.12 "${VIRTUAL_ENV}"
# Install the extension's dependencies before preparing NCCL for its build.
uv --directory "${repo_root}" sync --active --locked --all-packages --group container \
    --no-install-package transformer-engine-jax --no-cache

# nvidia-nccl-cu13 ships only libnccl.so.2; create the unversioned symlink so
# transformer-engine's sdist build can resolve `-lnccl` via LIBRARY_PATH.
nccl_lib="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nccl/lib"
if [ ! -e "${nccl_lib}/libnccl.so.2" ]; then
    echo "[failed] ${nccl_lib}/libnccl.so.2 not found; nvidia-nccl-cu13 was not installed from ${repo_root}/uv.lock"
    exit 1
fi
ln -sf libnccl.so.2 "${nccl_lib}/libnccl.so"

uv --directory "${repo_root}" sync --active --locked --all-packages --group container --no-cache
"${VIRTUAL_ENV}/bin/pip" check
# MaxText imports NNX; package metadata alone does not catch its JAX API dependencies.
"${VIRTUAL_ENV}/bin/python" -c 'from flax import linen, nnx'
