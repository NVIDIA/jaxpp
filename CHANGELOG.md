# Change log

<!-- jaxpp-release-begin: Update 26.10.15 -->
## Update 26.10.15

Updates include

- Runtime and communication
  - Synchronized transfers on the first MPMD call to avoid lazy CUDA initialization deadlocks, then resumed asynchronous transfers
  - Cached output sharding and mesh metadata across MPMD calls
- Compilation, sharding, and pipeline execution
  - Reused one compiled executable per task across equivalent Python scalar, NumPy scalar, and JAX array inputs
  - Traced MPMD functions through `jax.jit(...).trace` and aligned unspecified input/output sharding behavior with `jax.jit`
  - Added `tscan` with carry state and stacked microbatch outputs, and support for closed-over Ref accumulators in pipelined loops
  - Fixed typed PRNG-key size accounting during resharding and preserved loop effects and reduction debug metadata
- MaxiMoE
  - Unified XLA, SM90, and SM100 MoE operations behind explicit backend selection, shared differentiation rules, and XLA reference implementations
  - Updated training precision with FP32 master weights and router state, BF16 compute and stored Adam moments, selective pinned-host state offloading, and optional lossless expert-master compression
  - Added ZeRO modes for sharding training state, accumulated gradients, and compute parameters
  - Reworked context-parallel MLA attention around low-rank latent exchange and preserved per-sequence capacity and auxiliary-loss semantics across context shards
  - Expanded kernel, precision, offload, and training-equivalence tests and added H100 16K dropless training recipes
- Compatibility, containers, and documentation
  - Raised the minimum JAX version to 0.9.1 and updated compatibility handling for newer JAX tracing, state-discharge, and Jaxpr APIs
  - Updated container dependency setup, including CUDA wheel headers, Transformer Engine runtime checks, and the OpenSSH client
  - Documented the NCCL 2.30.7 symmetric-communication initialization issue and NCCL 2.31.2 workaround
<!-- jaxpp-release-end: Update 26.10.15 -->

<!-- jaxpp-release-begin: Update 26.09.17 -->
## Update 26.09.17

Updates include

- Activation offloading
  - Preserved host and device memory-space targets while rebinding named shardings to a new mesh
  - Added regression coverage for offload and reload targets, gradients, and flat and nested device placements
<!-- jaxpp-release-end: Update 26.09.17 -->

<!-- jaxpp-release-begin: Update 26.09.10 -->
## Update 26.09.10

Updates include

- Pipeline execution
  - Made the microbatch index implicit in `dax_pscan`, simplifying its caller interface and ensuring loop transformations handle the index consistently
- Runtime and communication
  - Replaced CUDA send callbacks with event polling to avoid GIL deadlocks and PJRT races during send-buffer release
  - Removed the now-unused direct `cuda-bindings` dependency
<!-- jaxpp-release-end: Update 26.09.10 -->

<!-- jaxpp-release-begin: Update 26.08.10 -->
## Update 26.08.10

Updates include

- JAX support
  - Added support for JAX **<= 0.11.0** and updated compatibility with JAX nightly
  - Updated compatibility with the latest JAXPR, inline parameter, transpose, and state-discharge APIs
- Runtime and communication
  - Reimplemented DIME using `cuda.core` and `nccl4py`, removing the CuPy dependency
  - Improved DLPack validation, version compatibility, device-state restoration, and stream handling
  - Moved transfer completion to first use, allowing communication to overlap more intervening computation
  - Added task state discharge support for internal Ref state
- Sharding and pipeline fixes
  - Fixed cross-MPMD sharding normalization while preserving reduction metadata and memory kind
  - Fixed cross-MPMD stacks under ambient concrete meshes
  - Preserved and rebound nested abstract meshes under `shard_map`
  - Fixed replicated `treduce` output indexing
  - Skipped symbolic `float0` values during explicit buffer deletion
- Documentation
  - Added executable MyST notebook documentation with build-time output validation
  - Added guidance for delaying gradient reductions with explicit and automatic sharding
  - Added committed rendered documentation for browsing generated examples and outputs
- Dependencies, build, and security
  - Updated Transformer Engine to 2.16.0 with the CUDA 13 package
  - Removed unused pandas and PyArrow dependencies
  - Added checksum verification and version pinning for the `uv` installer
  - Hardened GitHub documentation workflows with pinned actions, least-privilege permissions, concurrency controls, and timeouts
<!-- jaxpp-release-end: Update 26.08.10 -->
