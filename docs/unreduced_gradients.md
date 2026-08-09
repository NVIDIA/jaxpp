---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
---

# Delaying gradient reductions

A JaxPP pipeline is a gradient accumulation loop: {func}`~jaxpp.api.treduce` runs the
microbatch function once per microbatch under a pipeline schedule and adds the
resulting gradients across microbatches (`jaxpp.Add`), like a `jax.lax.scan`
whose carry is the gradient accumulator.

That loop interacts with SPMD sharding. When a parameter is replicated over a
mesh axis that the batch is split over, which is the case for the
data-parallel axis, and for an FSDP parameter after the all-gather that
precedes its use, its gradient is a cross-device sum. For `h = x @ w` the
backward pass computes `dw = x.T @ dh`, a contraction over the batch
dimension: with the batch sharded, each device holds only its slice of `x`
and `dh`, so it produces a partial `dw`, and the actual gradient is the
sum of the partials across devices, an all-reduce. Differentiating each
microbatch separately therefore performs one parameter-sized all-reduce per
microbatch.

The reduction is itself a sum, so it can be swapped with the sum over
microbatches. Writing `g[r, m]` for device `r`'s partial gradient of
microbatch `m`:

```text
grad = sum_m ( sum_r g[r, m] )    one cross-device reduction per microbatch
     = sum_r ( sum_m g[r, m] )    local sums, then one reduction per step
```

The second arrangement adds each device's partial gradients locally across
microbatches and reduces once at the end of the step. The result is the same
up to floating-point rounding (the additions are reassociated).

The examples on this page are compiled for two GPUs but never run: the mesh
below is built from compile-only topology devices, so one attached GPU is
enough. `hlo_summary` trims a compiled function's HLO to its while loops and
all-reduces, enough to see whether a reduction runs inside the microbatch loop
or once after it:

```{code-cell} ipython3
import re

import jax
import jax.numpy as jnp
from jax.experimental import topologies

topo = topologies.get_topology_desc(platform="cuda", topology="1x1x2")
gpu_mesh = jax.make_mesh((2,), ("data",), devices=topo.devices)

def hlo_summary(jitted, *args):
    """Compiled HLO trimmed to its while loops and all-reduces."""
    hlo = jitted.lower(*args).compile().as_text()
    blocks = re.findall(r"^((?:ENTRY )?%?[\w.-]+) \([^\n]*\{\n(.*?)^\}", hlo, re.M | re.S)
    out = []
    for name, body in blocks:
        picked = [
            line.split(", metadata=")[0].split(", channel_id=")[0].strip()
            for line in body.splitlines()
            if re.search(r" (all-reduce(-start|-done)?|while)\(", line)
        ]
        if picked:
            out += [name + " {", "  ..."] + [f"  {p}" for p in picked] + ["  ...", "}"]
    return "\n".join(out)
```

## Why XLA cannot do it for JaxPP

Here is the straightforward arrangement: `w` replicated, the batch sharded
over `data`, every scan iteration's backward pass reducing its own gradient.
Yet the compiled program performs one all-reduce per step, in `main` after the
while loop: within a single compiled program this is XLA's job, done by its
GPU compiler in the `while-loop-all-reduce-code-motion` pass:

```{code-cell} ipython3
def loss(w, xs):
    return jnp.sum((xs @ w) ** 2)

w = jax.ShapeDtypeStruct((4, 4), jnp.float32,
    sharding=jax.NamedSharding(gpu_mesh, jax.P()))  # replicated
microbatches = jax.ShapeDtypeStruct((3, 2, 4), jnp.float32,
    sharding=jax.NamedSharding(gpu_mesh, jax.P(None, "data", None)))

def naive_step(w, microbatches):
    def add_grad(acc, mb):
        return acc + jax.grad(loss)(w, mb), None

    acc, _ = jax.lax.scan(add_grad, jnp.zeros_like(w), microbatches)
    return acc

with jax.set_mesh(gpu_mesh):
    hlo = hlo_summary(jax.jit(naive_step), w, microbatches)
print(hlo)

regions, entry = hlo.split("ENTRY")
assert "while(" in entry
assert "all-reduce" in entry and "all-reduce" not in regions  # hoisted
```

Disabling the pass shows the program as written, one all-reduce in the scan
body, executed once per microbatch:

```{code-cell} ipython3
no_hoist = jax.jit(
    naive_step,
    compiler_options={"xla_disable_hlo_passes": "while-loop-all-reduce-code-motion"},
)

with jax.set_mesh(gpu_mesh):
    hlo = hlo_summary(no_hoist, w, microbatches)
print(hlo)

regions, entry = hlo.split("ENTRY")
assert "while(" in entry
assert "all-reduce" in regions and "all-reduce" not in entry  # once per microbatch
```

A JaxPP program is not a single compiled program. {func}`~jaxpp.api.treduce` unrolls
the accumulation loop, and JaxPP cuts the unrolled program into tasks that are
compiled independently (see {doc}`compilation`); the loop itself survives only
as the schedule the runtime executes. Each microbatch's backward task then
ends with its own all-reduce, and no XLA pass ever sees two of them together,
let alone the loop. The reduction has to be delayed in the JAX program itself,
before it is cut into tasks.

How the per-device partial gradients are kept depends on the mesh axis types
of the `jax_mesh` passed to {class}`~jaxpp.api.MpmdMesh`:

* With **explicit** (`Explicit`) axes, keep them in the array's type, using
  the `reduced` and `unreduced` tags of `jax.P`.
* With **automatic** (`Auto`) axes, where shardings are not part of the
  types, keep them as an actual array dimension: put `jax.grad` inside
  `jax.vmap`, accumulate gradients with their mapped dimension intact, and
  sum that dimension once at the end.

## Explicit axes: `reduced` and `unreduced`

Explicit axes can describe a partial, not-yet-summed value directly in the
array's type, with two `jax.P` tags:

* `jax.P(unreduced={"data"})` types each device's buffer as one additive
  contribution: the logical value is the element-wise sum of the per-device
  buffers. This is the type of a partial gradient.
* `jax.P(reduced={"data"})` marks a value replicated over `data`. The tag
  does not change the array itself, only differentiation: the gradient with
  respect to a reduced parameter is returned unreduced, one contribution per
  device, instead of being summed on the spot.

Mark the parameter reduced, accumulate the unreduced gradients, and reduce
once with `jax.reshard` after the loop. The batch and loss are the ones from
above, and `jax.eval_shape` traces the step without running it:

```{code-cell} ipython3
w = jax.ShapeDtypeStruct((4, 4), jnp.float32,
    sharding=jax.NamedSharding(gpu_mesh, jax.P(reduced={"data"})))
print(f"{jax.typeof(w)=!s}")

@jax.jit
def step(w, microbatches):
    def add_grad(acc, mb):
        grad = jax.grad(loss)(w, mb)
        print(f"{jax.typeof(grad)=!s}")
        return acc + grad, None

    acc0 = jax.reshard(jnp.zeros_like(w), jax.P(unreduced={"data"}))
    acc, _ = jax.lax.scan(add_grad, acc0, microbatches)
    return jax.reshard(acc, jax.P())  # the one cross-device reduction

with jax.set_mesh(gpu_mesh):
    grads = jax.eval_shape(step, w, microbatches)
print(f"{jax.typeof(grads)=!s}")
```

The gradient of each microbatch comes back unreduced (`{U:data}`), the
additions in the loop are local, and the final `jax.reshard` compiles to a
single all-reduce after the loop, this time guaranteed by the types:

```{code-cell} ipython3
with jax.set_mesh(gpu_mesh):
    hlo = hlo_summary(step, w, microbatches)
print(hlo)

regions, entry = hlo.split("ENTRY")
assert "all-reduce" in entry and "all-reduce" not in regions  # once per step
```

JAX's own tests exercise this pattern and check that exactly
[one all-reduce](https://github.com/jax-ml/jax/blob/990e6a0b84138346e6a38785412f36356e0e5dc3/tests/pjit_test.py#L9434-L9497)
remains.

The accumulator is initialized with an explicit `reshard` because
`jnp.zeros_like(w)` has `w`'s own type, replicated and reduced, not the
gradient's unreduced type. Unreduced values do not mix with ordinary ones;
doing so fails during tracing:

```{code-cell} ipython3
@jax.jit
def bad_step(w, microbatches):
    def add_grad(acc, mb):
        return acc + jax.grad(loss)(w, mb), None

    acc, _ = jax.lax.scan(add_grad, jnp.zeros_like(w), microbatches)
    return jax.reshard(acc, jax.P())

try:
    with jax.set_mesh(gpu_mesh):
        jax.eval_shape(bad_step, w, microbatches)
except Exception as e:
    if type(e).__name__ != "ShardingTypeError":
        raise
    print(f"{type(e).__name__}: {e}")
else:
    raise AssertionError("expected tracing to fail")
```

The same applies to almost any operation on an unreduced array other than
adding another array unreduced over the same axes; even scaling by an ordinary
scalar is rejected. Normalize the loss before differentiation, or scale the
gradient after the final `reshard`.

This arrangement does not change what the program computes. The microbatch
keeps a single logical batch dimension whatever its layout, so an operation
that couples batch elements, such as batch-normalization statistics, is still
computed over the whole microbatch, with the communication that this requires.
What is unsupported fails during tracing, as above.

## FSDP: delaying the reduce-scatter

An FSDP parameter is stored sharded, all-gathered before use, and its gradient
is reduce-scattered back to the sharded layout. The gathered copy is
replicated, so the same pattern applies: gather once before the microbatch
loop, mark the gathered copy reduced, and reshard the accumulated gradient
back to the sharded layout once per step. That final reshard is the step's one
reduce-scatter:

```{code-cell} ipython3
fsdp_mesh = jax.make_mesh((2,), ("fsdp",), devices=topo.devices)

w_shards = jax.ShapeDtypeStruct((4, 4), jnp.float32,
    sharding=jax.NamedSharding(fsdp_mesh, jax.P("fsdp")))
microbatches = jax.ShapeDtypeStruct((3, 2, 4), jnp.float32,
    sharding=jax.NamedSharding(fsdp_mesh, jax.P(None, "fsdp", None)))

@jax.jit
def step(w_shards, microbatches):
    w = jax.reshard(w_shards, jax.P(reduced={"fsdp"}))  # the one all-gather

    def add_grad(acc, mb):
        return acc + jax.grad(loss)(w, mb), None

    acc0 = jax.reshard(jnp.zeros_like(w), jax.P(unreduced={"fsdp"}))
    acc, _ = jax.lax.scan(add_grad, acc0, microbatches)
    return jax.reshard(acc, jax.P("fsdp"))  # the one reduce-scatter

with jax.set_mesh(fsdp_mesh):
    grads = jax.eval_shape(step, w_shards, microbatches)
print(f"{jax.typeof(grads)=!s}")
```

The gather must sit outside the differentiated function. Differentiation
transposes every operation it passes through, and the transpose of an
all-gather is a reduce-scatter: a parameter gathered inside `loss` (or
inside the scanned function) puts that reduce-scatter back into every
microbatch's backward pass. Gathering once and differentiating with respect to
the gathered copy avoids that; the final `reshard` restores the layout the
optimizer expects.

Unlike the data-parallel case, delaying the FSDP reduction trades memory for
communication: the accumulator holds the full parameter shape on every device
instead of one shard, and the gathered parameter stays live across the whole
microbatch loop.

## Automatic axes: keep a per-replica dimension

The `reduced` and `unreduced` tags require explicit axes. With automatic
axes, keep the partial gradients as an actual array dimension instead.
`jax.grad` of a function that maps over a batch sums the per-element
gradients of any shared input; `jax.vmap` of `jax.grad` computes the same
gradients but keeps them separate, leaving the caller to decide when to sum:

```{code-cell} ipython3
def loss(w, x):
    return jnp.vdot(w, x)

w = jnp.arange(4.0)
xs = jnp.arange(12.0).reshape(3, 4)

total = jax.grad(lambda w: jax.vmap(loss, in_axes=(None, 0))(w, xs).sum())(w)
parts = jax.vmap(jax.grad(loss), in_axes=(None, 0))(w, xs)

assert parts.shape == (3, *w.shape)  # one gradient per batch element
assert jnp.array_equal(parts.sum(axis=0), total)
```

To use this across devices, group the batch of each microbatch as
`(replica, per-replica batch)` and map `jax.grad` over the replica
dimension. Each gradient then carries a leading dimension with one
parameter-sized entry per replica, sharded over `data`, so each device holds
exactly its own contribution and the deferred sum becomes the one cross-device
reduction of the step:

```{code-cell} ipython3
auto_mesh = jax.make_mesh((2,), ("data",), devices=topo.devices,
                          axis_types=(jax.sharding.AxisType.Auto,))

def loss(w, xs):
    return jnp.sum((xs @ w) ** 2)

per_replica_grad = jax.vmap(jax.grad(loss), in_axes=(None, 0), spmd_axis_name="data")

@jax.jit
def step(w, microbatches):
    def add_grad(acc, mb):
        return acc + per_replica_grad(w, mb), None

    acc0 = jax.lax.with_sharding_constraint(jnp.zeros((2, *w.shape)), jax.P("data"))
    acc, _ = jax.lax.scan(add_grad, acc0, microbatches)
    return acc.sum(axis=0)  # the one cross-device reduction

# the arrangement computes the same gradients, checked here on one device
acc = sum(per_replica_grad(jnp.ones((4, 4)), mb) for mb in jnp.ones((3, 2, 2, 4)))
expected = sum(jax.grad(loss)(jnp.ones((4, 4)), mb) for mb in jnp.ones((3, 4, 4)))
assert jnp.allclose(acc.sum(axis=0), expected)

w = jax.ShapeDtypeStruct((4, 4), jnp.float32,
    sharding=jax.NamedSharding(auto_mesh, jax.P()))  # replicated
# (microbatch, replica, per-replica batch, feature)
microbatches = jax.ShapeDtypeStruct((3, 2, 2, 4), jnp.float32,
    sharding=jax.NamedSharding(auto_mesh, jax.P(None, "data", None, None)))

with jax.set_mesh(auto_mesh):
    regions, entry = hlo_summary(step, w, microbatches).split("ENTRY")
assert "all-reduce" in entry and "all-reduce" not in regions  # once per step
```

Two details matter:

* `spmd_axis_name="data"` makes sharding constraints inside the mapped
  function account for the new leading dimension and keep it on `data`. This
  small example has no internal constraints, but real models do, and without
  it those constraints say nothing about the mapped dimension.
* The constraint on the accumulator keeps its leading dimension sharded over
  `data`. With automatic axes nothing else in the program forces that
  choice, and a replicated accumulator would reintroduce a cross-device
  transfer in every iteration.

The mapped dimension has one entry per replica, not one per example, so this
costs no extra memory in plain data parallelism: each device holds one
parameter-sized accumulator either way. It now contains the device's partial
sum instead of the reduced sum.

There is also a semantic restriction: mapping `jax.grad` over replicas is only
equivalent to differentiating the original function if batch elements do not
interact inside `loss`. An operation that couples them, such as batch
normalization computing statistics over the batch dimension, changes meaning
under `vmap`: each replica computes statistics over its own slice, nothing
raises, and the accumulated result is the gradient of a different function.
The explicit-axes arrangement above does not have this problem.

## Caveats

* Swapping the sums is only valid for state that accumulates additively.
  Quantities accumulated by overwriting or by a maximum, such as some
  quantization statistics, must keep their per-microbatch handling.
* The additions are reassociated, so results can differ from the
  per-microbatch reduction in the last bits of floating-point rounding.
* Reduce before anything that needs the actual gradient: the optimizer update,
  gradient clipping, or a gradient-norm log all read the logical sum, not one
  device's partial.
