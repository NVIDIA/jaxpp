# JaxPP

JaxPP is a JAX library enabling Multiple-Program Multiple-Data (MPMD)
pipeline parallelism through simple user annotations `pipeline_enter_stage(layer)`
and decorators `@mpmd_jit_with_loop`.

JaxPP automatically splits JAX computations into multiple SPMD modules that
are independently jitted and dispatched to different devices.


# Status
JaxPP is under active development, and its APIs are currently unstable and subject to change.

## Changelog

* [Aug 19, 2025] Users must now add a final `pipeline_enter_stage` to mark the last
  stage as well.

# Contacts

As project development is ongoing, we are not accepting Pull Requests to the GitHub repository.
Please contact the maintainers for any questions or concerns.

Issues and feature requests are welcome.

# Installation instructions
JaxPP dependencies and supported JAX versions are listed in [`pyproject.toml`](https://github.com/NVIDIA/jaxpp/-/blob/main/pyproject.toml).

```bash
git clone https://github.com/NVIDIA/jaxpp.git
cd jaxpp
pip install -e .
```

You can verify the setup with [`examples/basic.py`](examples/basic.py) on a single-node.

```bash
python examples/basic.py
```

# Example

The example here shows the typical pattern used in a `flax` module to enable JaxPP.

```python
class ManualStagesModel(nn.Module):
    config: BertConfig
    pipeline_parallelism: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBertLayer(
                self.config, name=f"flax_bert_layer_{i}", dtype=self.dtype
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states):
        num_layers_per_stage = self.config.num_hidden_layers // self.pipeline_parallelism
        stage_id = 0
        for i, layer in enumerate(self.layers):
            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]

            # Mark that we are entering a new stage
            if (i + 1) % num_layers_per_stage == 0:
                hidden_states = jaxpp.pipeline_enter_stage(hidden_states)
                stage_id += 1

        return hidden_states
```

And the code snippet below shows a typical train step function with JaxPP.
```python
def loss(pars, batch):
    res = model.apply(pars, batch)
    return jnp.mean((res - batch) ** 2) / num_mubatches, (res, 4)

# The `mpmd_jit_with_loop` transformation, with `treduce`,
# will make this function execute in mpmd_jit_with_loop fashion over 2 devices
# using the `Eager1F1B` schedule
@partial(jaxpp.mpmd_jit_with_loop, mpmd_mesh=mpmd_mesh)
def pp_train_step(opt_state, pars, batch):
    mubatch_grad = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
    # Compute loss and gradients
    (losses, (pred, _)), grad = jaxpp.treduce(
        mubatch_grad, batch, schedule=jaxpp.Std1F1B(mpmd_mesh.mpmd_dim)
    )
    # Apply the optimizer as usual
    (updates, opt_state) = optimizer.update(grad, opt_state, pars)
    new_pars = optax.apply_updates(pars, updates)
    return opt_state, new_pars, losses, pred
```

To run the train step, we need to create a `MpmdMesh` object, which
is a wrapper of a standard Jax `Mesh` describing which dimension is the
mpmd one.

```python
devices = np.array(jax.devices()[0]).reshape(2, 1, 4)
jax_mesh = jax.sharding.Mesh(devices, ("mpmd", "data", "model"))
mpmd_mesh = jaxpp.MpmdMesh(jax_mesh, "mpmd")
print(mpmd_mesh.lowering_mesh().shape) # OrderedDict([('mpmd', 1), ('data', 1), ('model', 4)])
```

[examples/basic.py](examples/basic.py) provides a complete example.

# Building and Testing Docker Container

JaxPP provides Docker containers for development and testing. The build process consists of two stages: building a base image and then building the main image.

## Prerequisites
- Docker installed and configured
- NVIDIA Container Toolkit installed

## Building the Base Image

The base image contains all the core dependencies and is built using CUDA 12.8:

```bash
docker build --force-rm=true \
  -f scripts/docker/Dockerfile.base \
  --build-arg CUDA_BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -t jaxpp-base .
```

## Building the Main Image

After building the base image, you can build the main image:

```bash
docker build --force-rm=true \
  -f scripts/docker/Dockerfile \
  --build-arg BASE_IMAGE=jaxpp-base \
  -t jaxpp .
```

## Running Tests

The container includes several test suites that can be run:

1. **Unit Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  --rm --workdir=/workdir/jaxpp jaxpp \
  "python /workdir/jaxpp/examples/basic.py --dtype=float32 && \
   python /workdir/jaxpp/examples/basic.py --dtype=float16"
```

2. **PyTest Suite**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --rm --workdir=/workdir/jaxpp jaxpp "nvidia-smi && make install && pytest"
```

Note: The tests require GPU access and sufficient GPU memory.


# Multi-node setup
JaxPP needs to be installed on all nodes that are participating in the parallel
execution and the [installation instruction](#installation-instructions) needs
to be repeated on each node.
In addition, all packages that are needed for the execution of the workload
needs to be installed on all nodes.

# Benchmarks

JaxPP has been tested with several models from MaxText.
We have integrated JaxPP into a [fork of MaxText](https://github.com/NVIDIA/maxtext-jaxpp/blob/jaxpp/main/jaxpp.README.md) with minimal changes.


# Citing JaxPP

```
@misc{jaxpp,
      title={Scaling Deep Learning Training with MPMD Pipeline Parallelism}, 
      author={Anxhelo Xhebraj and Sean Lee and Hanfeng Chen and Vinod Grover},
      year={2024},
      eprint={2412.14374},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2412.14374}, 
}
```
