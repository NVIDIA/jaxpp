#!/usr/bin/env bash

set -eo pipefail

# Keep the version and installer digest in sync. The verified installer also
# checks the SHA-256 of the platform-specific uv archive before installing it.
readonly UV_VERSION="0.12.1"
readonly UV_INSTALLER_SHA256="d3f5412d38c99f9d024901843bf98206f0d2c6dbe64df40d0b740e2751ca62c1"
readonly UV_INSTALLER_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"

if [ -z "${VIRTUAL_ENV}" ]; then
    echo "[failed] VIRTUAL_ENV variable is not set."
    exit 1
fi

jaxpp_pip_path="${1}"

if [ -z "${jaxpp_pip_path}" ]; then
    echo "[failed] pass jaxpp path such as ./setup-env.sh '/path/to/jaxpp[dev]'"
    exit 1
fi

echo "Creating env ${VIRTUAL_ENV} and installing ${jaxpp_pip_path}"

uv_installer="$(mktemp)"
trap 'rm -f "${uv_installer}"' EXIT

curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
    --location --silent --show-error --fail \
    "${UV_INSTALLER_URL}" --output "${uv_installer}"
printf '%s  %s\n' "${UV_INSTALLER_SHA256}" "${uv_installer}" | sha256sum --check -
sh "${uv_installer}"

rm -f "${uv_installer}"
trap - EXIT

# The verified uv installer generates this environment file.
# shellcheck disable=SC1091
source "${HOME}/.local/bin/env"

installed_uv_output="$(uv --version)"
installed_uv_version="${installed_uv_output#uv }"
installed_uv_version="${installed_uv_version%% *}"
if [ "${installed_uv_version}" != "${UV_VERSION}" ]; then
    echo "[failed] expected uv ${UV_VERSION}, got ${installed_uv_output}"
    exit 1
fi

uv venv --python 3.12 "${VIRTUAL_ENV}"
uv pip install pip wheel setuptools

uv pip install --no-cache-dir -e "${jaxpp_pip_path}"
uv pip install --no-cache-dir pybind11

# nvidia-nccl-cu13 ships only libnccl.so.2; create the unversioned symlink so
# transformer-engine's sdist build can resolve `-lnccl` via LIBRARY_PATH.
nccl_lib="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nccl/lib"
if [ ! -e "${nccl_lib}/libnccl.so.2" ]; then
    echo "[failed] ${nccl_lib}/libnccl.so.2 not found; nvidia-nccl-cu13 was not pulled in by '${jaxpp_pip_path}'"
    exit 1
fi
ln -sf libnccl.so.2 "${nccl_lib}/libnccl.so"

# Don't use uv for TE as uv installs transformer-engine-cu12 even though we
# need transformer-engine-cu13.
"${VIRTUAL_ENV}/bin/pip" install --no-build-isolation transformer-engine[jax]==2.16.0
if "${VIRTUAL_ENV}/bin/pip" show transformer-engine-cu12 >/dev/null 2>&1; then
    echo "[failed] transformer-engine-cu12 was installed, expected cu13."
    exit 1
fi
if ! "${VIRTUAL_ENV}/bin/pip" show transformer-engine-cu13 >/dev/null 2>&1; then
    echo "[failed] transformer-engine-cu13 is not installed."
    exit 1
fi
