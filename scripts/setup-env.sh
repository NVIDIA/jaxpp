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
uv pip install --no-build-isolation transformer-engine[jax]==2.1.0
