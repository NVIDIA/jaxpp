# MaxiMoE

MaxiMoE is a standalone DeepSeek V3 mixture-of-experts training example built
on JaxPP. It contains the physical-axis trainer, expert routing, SM90 and SM100
Mosaic GPU kernels, and routed-layer profiler used for large-scale JaxPP MoE
training. It has no dependency on a larger model framework.

The default model shape follows the MLPerf DeepSeek V3 benchmark. Training uses
synthetic token data so the example stays focused on JaxPP pipelining, sharding,
MoE communication, and kernels.

Training defaults to BF16 compute and gradient accumulation, FP32 master
weights and router-bias state, and BF16 Adam first and second moments. Adam
computes each update in FP32 and rounds the moments only when storing them for
the next step. Master weights retain updates that are too small to change the
BF16 compute copy.
`--param-dtype bf16` selects BF16 parameter storage for memory comparisons.
Router biases remain FP32 in either mode, and routing probabilities stay FP32
through expert weighting before the result is rounded to the activation dtype.

`--host-offload-regex` stores selected training-state leaves in pinned host
memory. It uses Python `re.search` on JAX pytree key paths, including the leading
dot and bracketed layer indices. For example:

```bash
# All master parameters and Adam state (including the step counter).
CUDA_VISIBLE_DEVICES=0 maximoe-train --preset tiny \
  --host-offload-regex '^\.(params|opt)\.'

# Adam moments only.
--host-offload-regex '^\.opt\.(mu|nu)\.'

# Master parameters and moments for decoder layers 0 and 1.
--host-offload-regex '^\.(params|opt\.(mu|nu))\.layers\[[01]\]\.'
```

Offloading is disabled by default. Invalid expressions and expressions matching
no leaves fail before compilation. The trainer logs the selected leaf count and
global storage size. Partitioning and storage precision are preserved, including
MTP parameters. Compute copies and Adam updates run on GPU; selected masters and
moments are reloaded when needed and updated values return to host memory.
Optimizer reloads are issued before the cross-stage gradient-norm reduction,
allowing transfers to overlap that reduction.
This trades device memory for pinned host memory and transfer bandwidth. Router
biases remain on GPU unless explicitly matched (for example, `^\.router\.`).

`--capacity-factor` bounds the padded route-buffer allocation to reduce memory
usage. Each original sequence/expert keeps at most
`int(max(ceil(seq_length * top_k / num_experts) * capacity_factor,
capacity_factor))` assignments. Excess assignments are dropped, so this changes
the model computation if a sequence exceeds capacity; it does not make the
router's requested assignments more balanced. A negative factor disables
dropping. With `--ep-as-context`, all context chunks share the original
`seq_length` capacity budget, admitted in original token order. Different
sequences retain independent budgets.

Under `--ep-as-context`, the sequence auxiliary loss sums per-expert counts and
scores across context chunks, so it matches the whole-sequence loss.

## Installation

From the JaxPP repository root, install JaxPP and MaxiMoE into the same
environment:

```bash
uv sync --python 3.12 --locked --all-packages --group test
source .venv/bin/activate
```

This uses the shared Python 3.12 workspace lock. Development tools and test
dependencies are managed in the root `pyproject.toml`. Run the root pre-commit
hooks to format both packages with the locked Ruff version.

For an existing environment with custom JAX and CUDA packages, install the
sources with `python -m pip install --no-deps -e . -e examples/maximoe`.
The launchers below use the installed packages and do not modify `PYTHONPATH`.

## Smoke test

Run commands in this section from `examples/maximoe`. The tiny preset runs the
complete trainer on one GPU with reduced model shapes:

```bash
../../scripts/jax-run \
  --n-procs 1 \
  --n-gpus-per-proc 1 \
  -- \
  python -m maximoe.train --preset tiny
```

JaxPP's `scripts/jax-run` selects GPUs, assigns process ranks, configures
distributed JAX, and writes per-rank logs and compiler artifacts under `logs/`.
A direct invocation is also supported:

```bash
CUDA_VISIBLE_DEVICES=0 maximoe-train --preset tiny
```

## Context-parallel attention

With `--ep-as-context`, MLA attention is context parallel over the expert axis.
Each EP rank holds a disjoint head shard of `wq_b` and `wkv_b`. It gathers the
normalized low-rank latents for the whole sequence (1536 query, 512 KV and 64
RoPE-key values per token for DeepSeek V3), up-projects only its own heads, and
runs causal cuDNN attention. One all-to-all returns the context to sequence
sharding before the output projection; the backward reduce-scatters the latent
gradients. At sequence length 16,384 and CP8 this moves about 360 MB per layer
and microbatch per GPU, versus about 880 MB for exchanging up-projected Q/K/V.

## Routed MoE profiler

The profiler runs one or more production-shaped routed MoE layers. EP8 uses up
to eight layers and EP16 uses up to fifteen layers.

```bash
python -m maximoe.profile \
  --ep 8 \
  --num-layers 1 \
  --profile-dir profiles/ep8
```

## Distributed process environment

MaxiMoE initializes distributed JAX when `NNODES` is set. A multi-node launcher
must provide these variables to every process:

```bash
export NNODES=16
export NODE_RANK=0
export JAX_COORDINATOR_ADDRESS=host:port
export JAX_LOCAL_DEVICE_IDS=0,1,2,3
```

Use the process rank and local device IDs assigned to each node. The trainer
expects the number of visible devices to equal `pp * data * fsdp * ep` across
all processes.

`--data` replicates the training state and all-reduces gradients across
replicas. `--fsdp` shards the training state, and `--zero` selects what it
shards:

- `1`: master weights and Adam moments. Each step gathers the compute
  parameters once, and gradients are reduce-scattered after the pipeline.
- `2`: also accumulated gradients. Each microbatch reduce-scatters its
  gradients.
- `3` (default): also compute parameters. Each layer gathers its weights.

Weights outside the experts also shard their training state over `--ep`.

JaxPP's `scripts/slurm/launch` provides this environment for Slurm jobs. It
mounts the current JaxPP worktree and each run's output directory at `/output`.
It preserves the current project directory inside the container, so relative
setup, workload, and environment paths have the same meaning as they do on the
host. `scripts/jax-env.sh` writes runtime artifacts through `/output`. On the
host, runs stay under `examples/maximoe/logs/` by default. Pass `--num-nodes` to
select the node count. The default process mode starts one process per node and
lets that process control all visible GPUs. Use `--process-mode per_gpu` and set
`NTASKS_PER_NODE` to start one process per GPU instead.

The launcher requires Slurm with Pyxis container support. Set `CONTAINER_IMAGE`
to an image containing JaxPP, MaxiMoE, and their dependencies when the site does
not provide a default. `SLURM_LOG_DIR`, `SLURM_PARTITION`, `SLURM_ACCOUNT`,
`SLURM_TIME`, `SLURM_MAIL_USER`, and `EXTRA_SLURM_FLAGS` are also overridable.


## Runtime controls

- `MAXIMOE_ALL_TO_ALL_MODE=all_gather` replaces tiled all-to-all with
  all-gather plus a local slice for metadata paths that support it. The default
  is `all_to_all`.
- `RAGGED_DOT_SM_MARGIN` reserves SMs for communication alongside persistent
  ragged kernels.
- `RAGGED_DOT_PROFILE_DIR` enables the Mosaic in-kernel profiler.
- `--kernel-implementation {xla,sm90,sm100}` on the trainer and profiler selects
  the implementation for their MoE operations. Omit it to select the local
  platform's default (Mosaic on supported GPUs, XLA elsewhere).

The operation APIs take a keyword-only `implementation: str | None = None`, for
example `ragged_dot(lhs, rhs, sizes, implementation="xla")`. The resolved choice
is a static primitive parameter, shared by its forward and backward. Calls using
different implementations can coexist in one JIT. Training code can set
`Config(kernel_implementation="xla")`; there is no ambient backend context.

`RaggedDot`, `RaggedDotSwiGLU`, and `RaggedDotCombine` live in
`kernels/ragged_dot/ops.py`. These shared HiJAX classes own differentiation and
reference accumulation; their expansion and backward rules call one raw kernel
surface (`ragged_dot`, `ragged_dot_dx`, `ragged_dot_dw`, `ragged_dot_swiglu`,
`swiglu_backward`) that every backend module implements, including SM100's fused
SwiGLU epilogue. The standalone `xla` modules remain independently
differentiable specifications for testing.

Environment values used during tracing must be set before compilation.

## Tests

Run the project tests from `examples/maximoe`:

```bash
python -m pytest
```

Every test module is also an `absltest` program, so one module runs with
`python tests/ragged_dot_test.py`. GPU tests skip unless an SM90 or SM100 GPU is
visible, tests of one architecture's kernels skip on the other, and multi-device
tests skip when fewer GPUs are visible. The suite includes
abstract-sharding tests, routing and custom-VJP checks, trainer geometry tests,
and kernel correctness tests. Every kernel package (`maximoe.kernels.ragged_dot`
and `maximoe.kernels.moe_routing`) ships an `xla` module that is the
specification of its operations; `tests/ragged_dot_test.py` and
`tests/moe_routing_test.py` compare the Mosaic GPU kernels for the local
architecture against it, forward and backward.

Expert groups are packed contiguously. `RaggedArray.sizes` and the ragged-dot
`group_sizes` argument hold the actual number of rows per expert; there is no
separate `live_sizes` argument. Only the static allocation is tile-aligned.
Rows at or beyond `sum(group_sizes)` are unused and may contain arbitrary values,
so reductions and routing must exclude them. The same contract applies with
capacity dropping and with `--capacity-factor -1` for dropless routing.

## License

MaxiMoE is licensed under the Apache License 2.0. See `LICENSE`.

The SM90 ragged-dot kernels adapt Apache-2.0 code from Tokamax and JAX; the
SM100 kernels adapt JAX code. The shared helpers in
`kernels/ragged_dot/common.py` adapt code from both. These files retain the
upstream copyright notices in their headers.
