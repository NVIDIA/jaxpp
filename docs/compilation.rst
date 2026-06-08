Reducing compilation time
=========================

To pipeline a program, JaxPP cuts its jaxpr into subjaxprs called *tasks* and
assigns them to pipeline stages; a stage runs several tasks, such as its forward
and backward passes. Wiring the tasks together requires the sharding of every
value that crosses a task boundary (its *intermediate sharding*). How JaxPP
obtains these shardings depends on the mesh axis types of the ``jax_mesh``
passed to :class:`~jaxpp.api.MpmdMesh`:

* With **automatic** (``Auto``) axes, the shardings of intermediate values are
  chosen by XLA's SPMD sharding propagation, a pass XLA runs only during a full
  compilation. JaxPP therefore compiles the program just to obtain those
  shardings, then discards the resulting executable.
* With **explicit** (``Explicit``) axes, the shardings are carried in the
  JAX-level types and are already known at trace time, so this compilation is
  skipped entirely (see `Explicit sharding`_).

**By default JaxPP runs that compilation over the whole computation at once.**
For large models this single compilation can dominate startup time. The
following environment variables make JaxPP compile the program one task at a
time and cache the result across tasks, which can
substantially reduce compilation time:

``JAXPP_ENABLE_LOCAL_PROPAGATION=1`` (defaults to ``0``)
   Compile each task separately to infer its shardings, instead of compiling the
   whole computation at once, and reuse the result across structurally equivalent
   tasks.

``JAXPP_FAST_INFER_SHARDINGS=1`` (defaults to ``0``)
   Run that sharding-inference compilation with cheap compiler flags (low
   optimization level, no autotuning, and so on). The flags do not change the
   inferred shardings, so they cut compile time without changing the result.

How it works
------------

Tasks are reused only when their structure matches exactly. For example, a task
with two layers is not shared with a task of two layers plus the loss
computation, because the two produce different jaxprs. This task-jaxpr
deduplication is controlled by ``JAXPP_ENABLE_TASK_JAXPR_DEDUPLICATION``
(defaults to ``1``); set it to ``0`` to compile every task separately, even
structurally identical ones.

Inferring the shardings is a separate compilation from the one that runs the
program, so each task is compiled twice: once to infer its shardings and once
for execution. The execution compilation reuses the same structural caching and
is parallelized across ranks, with each rank compiling only the tasks it owns.

Caveats
-------

Suboptimal shardings
~~~~~~~~~~~~~~~~~~~~~

Unlike the whole-computation compilation, the local pass propagates shardings one
task at a time rather than across the entire computation, so it can produce worse
shardings when the program is sparsely annotated.

Cache misses
~~~~~~~~~~~~

The cache is always safe and never changes results, but it can still miss when
two structurally identical tasks are not recognized as equivalent. The dedup
only follows params that are *directly* a jaxpr and compares the rest by
identity, so a miss happens when a task either:

* **Captures a Python object** that compares by identity. A ``jax.debug.callback``,
  for example, is a separate object at each call site, but any captured object
  can have the same effect.
* **Hides jaxprs in a container** the dedup does not descend into. ``pl.kernel``
  / ``plgpu.kernel`` lower to ``mpmd_map``, which keeps the kernel body in a
  ``jaxprs`` **tuple**, so the whole kernel is invisible to the dedup and is
  compiled once per task. ``pl.pallas_call`` keeps its body as a direct jaxpr
  param but nests the index map inside
  ``GridMapping`` / ``BlockMapping`` **dataclasses** as ``index_map_jaxpr``.

Wrapping the affected computation in a single ``jax.jit`` avoids this: it sits
behind a ``pjit`` boundary that is deduplicated, and the trace cache shares one
jaxpr across all call sites. It is worth doing by default, even when task
deduplication is not the concern: a kernel called from several sites is otherwise
re-traced *and* re-compiled at each one.

For example, an ``add_one`` kernel called twice (``interpret=True`` so it runs on
any backend):

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    def _add_one_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    @jax.jit  # one cached jaxpr, shared across call sites
    def add_one(x):
        return pl.pallas_call(
            _add_one_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            interpret=True,
        )(x)

    def f(x):
        return add_one(x) + add_one(x)  # two calls to the same kernel

    print(jax.make_jaxpr(f)(jnp.arange(4.0)))

Both call sites reference one shared kernel jaxpr, printed once as a ``let``
binding (note ``jaxpr=add_one`` on both calls) rather than inlining the kernel
body twice:

.. code-block:: text

    let add_one = { lambda ; a:f32[4]. let
        b:f32[4] = pallas_call[ ... jaxpr=jaxpr ... ] a
      in (b,) } in
    let jaxpr = { lambda ; c:Ref<default>{f32[4]} d:Ref<default>{f32[4]}. let
        e:f32[4] <- c[...]
        f:f32[4] = add e 1.0:f32[]
        g:f32[4], d[...] <- d[...], f
      in () } in
    { lambda ; h:f32[4]. let
        i:f32[4] = jit[name=add_one jaxpr=add_one] h
        j:f32[4] = jit[name=add_one jaxpr=add_one] h
        k:f32[4] = add i j
      in (k,) }

Without the ``jax.jit`` wrapper the printed jaxpr instead contains two separate
``pallas_call`` equations, each inlining the kernel body.

Explicit sharding
-----------------

Everything above applies only to ``Auto`` mesh axes, where shardings are left to
the compiler. When every mesh axis instead has type ``Explicit`` (JAX's
`explicit sharding mode <https://docs.jax.dev/en/latest/parallel.html>`_, also
called "sharding in types"), shardings are part of the JAX-level types and
propagate through each operation at trace time, as part of abstract evaluation.
The intermediate shardings are therefore already attached to the traced jaxpr,
queryable with ``jax.typeof``, before anything is compiled.

In that mode JaxPP reads the intermediate shardings straight off the jaxpr's types, so
the sharding-inference compilation does not happen at all. Because no XLA
propagation is involved, ``JAXPP_ENABLE_LOCAL_PROPAGATION`` and
``JAXPP_FAST_INFER_SHARDINGS`` have no effect.
