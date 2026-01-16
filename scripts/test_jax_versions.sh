#!/bin/bash
set -e

JAX_VERSIONS=("0.6.2" "0.7.0" "0.7.1" "0.7.2" "0.8.0" "0.8.1" "0.8.2")

for version in "${JAX_VERSIONS[@]}"
do
  echo "Installing JAX version: $version"
  pip install jax[cuda12]==$version
  echo "Running command with JAX version: $version"
  JAXPP_DEBUG_FORCE_MPMDIFY=True JAXPP_ENABLE_LICM=True python examples/basic.py --train_steps=10
done

# Test nightly
pip install -U --pre jax jaxlib "jax-cuda12-plugin[with-cuda]" jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
JAXPP_DEBUG_FORCE_MPMDIFY=True JAXPP_ENABLE_LICM=True python examples/basic.py --train_steps=10

exit 0
