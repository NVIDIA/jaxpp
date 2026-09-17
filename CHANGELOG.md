# Change log

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
