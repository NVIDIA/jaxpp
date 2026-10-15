# JaxPP

JaxPP is a JAX library enabling Multiple-Program Multiple-Data (MPMD)
pipeline parallelism through simple user annotations `mark_stage_end(layer)`
and decorators `@mpmd_jit_with_loop`.

JaxPP automatically splits JAX computations into multiple SPMD modules that
are independently jitted and dispatched to different devices.

[**Change log**](https://github.com/NVIDIA/jaxpp/blob/main/CHANGELOG.md)
| [**Reference docs**](https://nvidia.github.io/jaxpp/)


# Status
JaxPP is under active development, and its APIs are currently unstable and subject to change.

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

For multiple local JAX processes, use `scripts/jax-run` from the repository
root. It assigns process ranks and GPUs and retains per-rank logs and compiler
artifacts under `logs/`.

```bash
./scripts/jax-run --n-procs 2 --n-gpus-per-proc 2 -- \
  python -u examples/mpmd_reshard.py
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
                hidden_states = jaxpp.mark_stage_end(hidden_states)
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

# Development environment

Install uv 0.12.1, then run these commands from the repository root:

```bash
uv sync --python 3.12 --locked --all-packages --group test --group examples
source .venv/bin/activate
pre-commit install
```

The shared `uv.lock` targets Python 3.12 on Linux AMD64 and ARM64. The two
packages retain their own published dependency requirements. For formatting
alone, use `--only-group dev` in place of `--all-packages --group test --group examples`
in the initial sync to skip GPU dependencies.

With the environment activated, run hooks directly. The development container
already puts these tools on `PATH`:

```bash
pre-commit run --all-files
```

Pre-commit and CI run the same locked Ruff version. Ruff uses each package's
formatting configuration. Run tests with `python -m pytest`;
see `docs/README.md` in the checkout for building documentation.

After editing dependency groups or package requirements, run `uv lock` and
commit `uv.lock` with the input changes. Use `uv lock --upgrade-package NAME`
for an intentional package upgrade, or `uv lock --upgrade` to refresh all pins.
CI checks freshness with `uv lock --check` and validates installation plans for
both container architectures.

The container installs the `container` group, which includes development,
test, documentation, and shared example dependencies. Transformer Engine's
CUDA 13 core and JAX extension are both locked. Container setup installs the
build dependencies and prepares NCCL before syncing the extension, which builds
against the locked environment without build isolation.
The JAX compatibility matrix deliberately replaces JAX and resolves a compatible
Flax version after the initial sync.

# Building and Testing Docker Container

JaxPP provides Docker containers for development and testing. The build process consists of two stages: building a base image and then building the main image.

## Prerequisites
- Docker installed and configured
- NVIDIA Container Toolkit installed

## Building the Base Image

The base image starts from NVIDIA's CUDA 13.2 image. CUDA compilation tools,
headers, and GPU libraries come from the NVIDIA wheels pinned in `uv.lock`;
their CUDA version can differ from the parent image's version. The host C++
toolchain builds TE's JAX extension against wheel headers, including NVTX,
while TE's CUDA core comes from its prebuilt wheel. cuRAND is installed explicitly
for TE's runtime loader because JAX does not depend on it. The image build checks
that TE's JAX bindings import successfully without a GPU.

For multi-node deployments, the parent image and runtime environment must
provide the networking libraries required by the target fabric.


```bash
docker build --force-rm=true \
  -f scripts/docker/Dockerfile.base \
  --build-arg CUDA_BASE_IMAGE=nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
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
  --rm --workdir=/workdir/jaxpp jaxpp \
  "nvidia-smi && python -m pytest"
```

Note: The tests require GPU access and sufficient GPU memory.


# Multi-node setup
JaxPP needs to be installed on all nodes that are participating in the parallel
execution, and the [installation instructions](#installation-instructions) need
to be followed on each node.
In addition, all packages that are needed for the execution of the workload
need to be installed on all nodes.

JaxPP does not require a particular cluster launcher or distributed runtime
integration. For our benchmarks and development workflows, we launch one
process per participant and initialize JAX's multi-controller distributed
runtime with `jax.distributed.initialize()` before creating the `MpmdMesh`.
Other orchestration environments, such as Ray, can also be used.


# Benchmarks

JaxPP has been tested with several models from MaxText.
We have integrated JaxPP into a [fork of MaxText](https://github.com/NVIDIA/maxtext-jaxpp/blob/jaxpp/main/jaxpp.README.md) with minimal changes.


# Known issues

## NCCL 2.30.7 with device-initiated symmetric communication

JaxPP's DIME transfer path uses two-sided NCCL send/receive rather than
one-sided RMA. This limitation applies to computations that XLA lowers through
its device-initiated symmetric communication path, such as a
`ragged_all_to_all` lowered through symmetric memory.

With NCCL 2.30.7, these workloads may encounter a hang consistent with a
[known NCCL issue](https://github.com/NVIDIA/nccl/releases/tag/v2.30.7-1#known-issues)
in which one-sided host RMA initialization can block when only a subset of
communicator ranks reaches the first RMA operation. The
[corresponding fix](https://github.com/NVIDIA/nccl/commit/e12963e1c7994e054c29df0994e8675ff4749875)
is included in NCCL 2.31.2. Install `nvidia-nccl-cu13>=2.31.2` when using
this path.

[`NCCL_RMA_EAGER_INIT`](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/env.html#nccl-rma-eager-init)
is not implemented in NCCL 2.30.7, so setting or unsetting it has no effect in
that version. In NCCL 2.31.2, setting it to `1` performs RMA initialization
during communicator creation; when it is unset, initialization normally occurs
at the first collective `ncclCommWindowRegister`.

# Citing JaxPP

```
@inproceedings{MLSYS2025_9f73d65a,
      author={Xhebraj, Anxhelo and Lee, Sean and Chen, Hanfeng and Grover, Vinod},
      booktitle={Proceedings of Machine Learning and Systems},
      editor={M. Zaharia and G. Joshi and Y. Lin},
      pages={},
      publisher={MLSys},
      title={Scaling Deep Learning Training with MPMD Pipeline Parallelism},
      url={https://proceedings.mlsys.org/paper_files/paper/2025/file/9f73d65a4186198152357be871345771-Paper-Conference.pdf},
      volume={7},
      year={2025}
}
```
