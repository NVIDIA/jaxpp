# JaxPP

JaxPP is a library built using Python, Ray, and JAX in order to allow an easy and usable pipeline parallelism in JAX through simple user annotations `jaxpp.api.pipeline_enter_stage(layer)` and decorators `@jaxpp.pipelined`.

## Related presentations

- [JaxPP Final Internship Presentation](https://docs.google.com/presentation/d/1Yos_ji5SSqOsV9OScWkhDwCbrVJBUzwnNMU2xUvb8kU/edit#slide=id.p1)
- [JaxPP Piepline Parallelism In AutoPjit Mid-Internship Presentation](https://docs.google.com/presentation/d/10WWcLKdQXRqbGM19bjLV4HNvtmw3k1Rp8fK9WgpmiIA/edit#slide=id.p1)
- [Pipeline Parallelism Survey](https://docs.google.com/presentation/d/1qB921uz9JfeY9X4wpN5ikoNqADb_7n1SHi8Cuqm5Pqo/edit#slide=id.g1519599a83e_0_180)
- [All Presentation Recordings Here](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DL&title=Machine+Learning+Compiler+Technology)

# Installation instructions

JaxPP currently supports JAX 0.5.1 and it requires CUDA 12 and cuDNN 9, similarly to `jax[cuda_12]`'s default dependencies.

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/CML/jaxpp_dev/jaxpp.git
cd jaxpp
pip install -e .
```

You can verify the setup with [`benchmarks/basic.py`](benchmarks/basic.py) on a single-node.

```bash
RAY_ADDRESS=local python benchmarks/basic.py
```

`RAY_ADDRESS=local` allows a new Ray session to be created locally for this test.
However, other `RAY_ADDRESS` values are supported (see [Multi-node setup](#multi-node-setup)).


# Example

The example here shows the typical pattern used in a `flax` module to enable JaxPP.

```python
class ManualStagesModel(nn.Module):
    config: BertConfig
    num_workers: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBertLayer(
                self.config, name=f"flax_bert_layer_{i}", dtype=self.dtype
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states):
        num_layers_per_stage = self.config.num_hidden_layers // self.num_workers
        stage_id = 0
        for i, layer in enumerate(self.layers):
            # Mark that we are entering a new stage at every layer
            if (
                self.num_workers != 1
                and i % num_layers_per_stage == 0
                and stage_id < self.num_workers
            ):
                hidden_states = jaxpp.pipeline_enter_stage(
                    hidden_states, f"stage_{stage_id}", stage_id
                )
                stage_id += 1
            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]
        return hidden_states
```

And the code snippet below shows a typical train step function with JaxPP.
```python
    # The `wrap_forward` transformation wraps every stage in a different jaxpr
    # to intercept the respective backward equations for each stage
    def loss(pars, batch):
        res = model.apply(pars, batch)
        return jnp.mean((res - batch) ** 2) / num_mubatches, (res, 4)

    # The `pipelined` transformation will make this function execute
    # in pipelined fashion over ray actors
    @jaxpp.pipelined
    def pp_train_step(opt_state, pars, batch):
        def mubatch_grad(mubatch):
            ((l, (pred, _)), grad) = jax.value_and_grad(loss, has_aux=True)(pars, mubatch)
            return grad, (l, pred)

        # Compute loss and gradients
        grad, (l, pred) = jaxpp.accumulate_grads(
            mubatch_grad,
            batch=batch,
            out_shardings=None,
            schedule=jaxpp.Eager1F1B(1),
        )
        # Apply the optimizer as usual
        (updates, opt_state) = optimizer.update(grad, opt_state, pars)
        new_pars = optax.apply_updates(pars, updates)
        return opt_state, new_pars, l, pred
```

To run the train step, we need to create a `RemoteMpmdMesh` object, and set CPU as the
default device in the driver process to dedicate all the devices to the worker processes.

```python
    mesh = RemoteMpmdMesh((num_workers, 1, 1), ("mpmd", "data", "model"))
    jax.config.update("jax_default_device", jax.local_devices(backend="cpu")[0])
```

Then the `RemoteMpmdMesh` object is used for the execution

```python
    with mesh:
        for _ in range(num_steps):
            # NOTE(semantics/consume_array_ref): all the arguments of `pp_train_step` that are `ArrayRef`s are consumed
            #  by the function
            (pp_opt_state, pp_new_pars, pp_loss, _) = pp_train_step(
                pp_opt_state, pp_new_pars, hidden_states
            )
```

[benchmarks/basic.py](benchmarks/basic.py) provides a complete example.

# Multi-node setup
JaxPP needs to be installed on all nodes that are participating in the parallel
execution and the [installation instruction](#installation-instructions) needs
to be repeated on each node.
In addition, all packages that are needed for the execution of the workload
needs to be installed on all nodes.
To simplify the multi-node setup, we provide [JaxPP docker images](#available-docker-images),
which has all the dependencies installed, for the benchmarks we track.

JaxPP relies on Ray to manage nodes in the cluster and to distribute work.
Once all the packages have been installed on each node, the following command needs to be
executed to start the driver process on the head node.

```bash
ray start --head --port=<port>
```

Then on each worker node, run the following to start the worker process and
register it with the driver process on the head node.

```bash
ray start --address=<head node address>:<port>
```

When the first `RemoteMpmdMesh` object is created, `ray.init("auto")` is run to start the ray
session with the registered nodes, unless it has already been started.
Then the worker mesh is created with the devices from the registered nodes.

Note that it is possible to create multiple worker processes on a single machine to create
an environment similar to a multi-node environment by running `ray start
--address=<head node address>:<port>` on the head node.

# Benchmarks

JaxPP has been tested with several models from MaxText.
We have integrated JaxPP into a [fork of MaxText](https://gitlab-master.nvidia.com/CML/jaxpp_dev/maxtext/-/blob/jaxpp_main/jaxpp.README.md) with minimal changes.

# Profiling

JAX profiling can be enabled even when JaxPP is used.  `RemoteMpmdMesh` provides two methods `start_trace` and `stop_trace` to start and stop profiling trace on all workers.
`start_trace` takes a path string as a parameter and each worker produces its own profile data in the path suffixed with its ID.
For example, if `/tmp/profile/dump` is given to `start_trace`, the first two workers produce the profile data in `/tmp/profile/dump-0` and `/tmp/profile/dump-1` respectively.

We also provide a tool to merge profile data from multiple workers into a single file.
This tool is in [script/profiling-tools](script/profiling-tools) and requires [bazel](https://bazel.build/install).
After installing Bazel, you can build the tool wiht the commands below.

```bash
cd script/profiling-tools
bazel build --compilation_mode=opt //main:main
cp bazel-bin/main/main <install path>
```

In the docker images we provide, it is already installed as `merge_multihost_xplanes`, and you can simply run it by supplying the pb files from all workers.
For example, if the profile data is dumped under `/tmp/profile/dump-*`, you can run the command below and it produces the merged profile data in the directory `out.json.gz` in the current directory.

```bash
merge_multihost_xplanes $(find /tmp/profile/dump-* -name *.pb)
```

The produced `out.json.gz` directory can be passed to tensorboard for visualization.

```bash
tensorboard --logdir out.json.gz/
```

# Saving and restoring train state

JaxPP provides functions to save train state and restore train state:
- `save_state` takes the train state in the form of [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html), serializes it, and saves it to the file system, and
- `load_state` deserializes the state, and returns the restored state in the same form of PyTree.

These functions are defined in [src/jaxpp/api.py](src/jaxpp/api.py).

# Available Docker Images

| Image                                                | Description |
| ---                                                  | ---         |
| gitlab-master.nvidia.com:5005/cml/jaxpp_dev/jaxpp:main         | Image that contains JaxPP and its dependencies. |

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