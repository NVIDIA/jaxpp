#!/bin/bash
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

# Keep test tools locked while allowing the compatibility matrix to replace JAX and Flax.
uv sync --active --locked --group test --inexact

if [ -n "${JAXPP_JAX_VERSIONS:-}" ]; then
  read -r -a JAX_VERSIONS <<< "$JAXPP_JAX_VERSIONS"
else
  # Mirror released jax-cuda13-plugin versions supported by JaxPP.
  JAX_VERSIONS=(
    "0.9.1"
    "0.9.2"
    "0.10.0"
    "0.10.1"
    "0.10.2"
    "0.11.0"
    "0.11.1"
    "0.11.2"
  )
fi

for version in "${JAX_VERSIONS[@]}"
do
  echo "Installing JAX version: ${version}"
  # Resolve Flax with JAX so older JAX releases get a compatible Flax version.
  pip install --upgrade "jax[cuda13]==${version}" flax
  echo "Running tests with JAX version: ${version}"
  ./scripts/run_tests.sh
done

# Test nightly
pip install -U --pre jax jaxlib "jax-cuda13-plugin[with-cuda]" jax-cuda13-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
pip install --upgrade flax "jax==$(python -c 'from importlib.metadata import version; print(version("jax"))')"
./scripts/run_tests.sh
