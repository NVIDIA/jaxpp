#!/bin/bash
set -e

JAX_VERSIONS=("0.5.1" "0.5.2" "0.5.3" "0.6.0")

for version in "${JAX_VERSIONS[@]}"
do
  echo "Installing JAX version: $version"
  pip install jax[cuda12]==$version
  echo "Running command with JAX version: $version"
  python examples/basic.py --train_steps=10
  python examples/basic.py --train_steps=10 --enable_multicontroller=False
done

# Test nightly
pip install -U --pre jax jaxlib "jax-cuda12-plugin[with-cuda]" jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
python examples/basic.py --train_steps=10
python examples/basic.py --train_steps=10 --enable_multicontroller=False

exit 0
