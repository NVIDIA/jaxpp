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

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv --python 3.12 "${VIRTUAL_ENV}"
uv pip install pip wheel setuptools

uv pip install --no-cache-dir -e "${jaxpp_pip_path}"
uv pip install --no-cache-dir pybind11

# nvidia-nccl-cu12 ships only libnccl.so.2; create the unversioned symlink so
# transformer-engine's sdist build can resolve `-lnccl` via LIBRARY_PATH.
nccl_lib="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nccl/lib"
if [ ! -e "${nccl_lib}/libnccl.so.2" ]; then
    echo "[failed] ${nccl_lib}/libnccl.so.2 not found; nvidia-nccl-cu12 was not pulled in by '${jaxpp_pip_path}'"
    exit 1
fi
ln -sf libnccl.so.2 "${nccl_lib}/libnccl.so"

uv pip install --no-build-isolation transformer-engine[jax]==2.8.0
