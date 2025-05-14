## Vocabulary

- Driver: the main python program
- Worker or Actor: a (possibly remote) `ray` Actor "owning" a set of devices.
  - Warning: parts of the codebase might mistakenly contain the term `node`
    which usually just means worker/actor.
    A node/host with `8` gpus can have 8 actors with 1 GPU each,
    4 actors with 2 GPUs each, 2 actors with 4 GPUs each or or 1 actor with 8 GPUs
    each.
  - For clusters with `N` nodes it is necessary to spawn 1 ray head process (`ray start --head`) in 1 node
    and `N - 1` processes in the remaining nodes (1 process per node).
    Once that is done there can be as many actors as number of total available
    GPUs (1 GPU per actor) or any combination.
    The only limitation is that 1 actor cannot have more GPUs than available
    on a single node.
- Device mesh: `jax.sharding.Mesh`, an nd-array of _unique_ Jax devices
  (`np.ndarray[Device]`) paired with axis_names for each of the axes.
  - **Addressable devices**/**Local device mesh**: devices that are owned by
    the actor
  - **Global device mesh**: all the devices available in the actors that
    have joined the XLA distributed context
- `Sharding`: a description of how an `Array` is sharded
- `OpSharding`: a lower-level serializeable description of a sharding
- `PartitionSpec`: describes a sharding when paired with a mesh
- Stage: a step of a pipelined computation. A stage can be
  scheduled in any device mesh of any worker.

## Runtime

- JaxPP allows running models in pipelined fashion across a cluster of
  resources without model-invasive changes to the model definition,
  differently from [GSPMD's pipeline parallelism Sec 3.3] and [praxis].
- Similarly to [Pathways], each pipeline stage of the model runs in a
  specific actor (or worker) managing a device mesh formed by its
  addressable devices.
- Actors are created through `ray` which automatically manages the assignment
  of gpus to actors (`num_gpus` argument).
- All the workers join the same XLA distributed context meaning that
  if `backend.devices()` returns all the devices available in the distributed
  context
- In the example below a mesh with 4 workers, 2 GPUs each forming a global
  device mesh composed of 8 devices is created. The global mesh shape is `(4, 1, 2)`.
  The shape of the mesh local to a specific actor is `(1, 2)`.
  Since the example runs on a single node, `"local"` can be passed as ray address
  without requiring to instantiate ray processes explicitly in the node (`ray start`).

    ```py
        mesh = RemoteMpmdMesh(
            4, (1, 2), ("data", "model"),
            ray_address="local", # or "ip:port" returned by `ray start --head`
        )
        pipe_train_step = jaxpp.pipelined(
            train_step,
            num_microbatches=num_microbatches,
            batch_argnums=(1,),
            donate_argnums=(0,),
            in_axis_resources=(state_pspecs, batch_pspec),
            out_axis_resources=(state_pspecs, None),
            shard_policy=args.shard_policy,
            schedule=args.schedule,
        )
    ```

- The JaxPP API call `pipelined` takes a traceable Jax function,
  traces it to a Jaxpr and "slices" it into stages that are to be run in
  remote device meshes, compiles them in the remote meshes and returns a function that when
  called will send a list of instructions to each worker to run the stages of the
  pipeline (a worker might run multiple stages with a interleaved/circular schedule).
- The arguments passed can either be numpy arrays (on the CPU) or `ArrayRef`s.
  - `ArrayRef`s are unique ids identifying an array on a remote mesh
- The results of the computation are `ArrayRef`s.
- Point-to-point communication between device meshes of different workers
  is performed as `CollectivePermute` operations
  - The collective permute is jitted on the "global device mesh"
    formed by the devices of the two workers the communication happens between
- The possible instructions are
  - `RunOp`: dispatches a (sharded) XLA executable on the local device mesh of the worker
    workers on the a "global device mesh" formed by the device meshes of the workers participating in the communication.
    To do so arrays are first expanded to global arrays # TODO
  - `AllReduceOp`: similar to `PpermuteOp` but performs a all-reduce instead of a send-receive
  - `DeleteOp`: deletes an array in the local store of the actor to release memory
  - `RenameOp`: when a function is compiled to a list of instructions, the instructions
    refer to argument and result array's for each instruction through unique ids (`UID`).
    These unique ids are always the same for each function call of the callable returned by
    `pipelined`.

## Sharp edges

- JaxPP driver programs must be run in a node that participates in the ray cluster.
  The driver program joins the distributed context
  partially occupying one of the GPUs possibly used by a worker.
  This is necessary so that the global device mesh (i.e devices) is available
  for `pjit` and lowering.
  TODO: Eventually we will want to move to "compile-only devices"
  (currently not landed for GPUs but exist for TPUs <https://github.com/openxla/xla/blob/d18b8bf83f395f462aee8fcd21da8ac0e8095ce3/xla/python/xla.cc#L957-L971>) which wouldn't require that and free the 300MB GPU mem occupied by the driver
- JaxPP programs must `jax.config.update("jax_platforms", "cpu")` at their start
  or before any Jax activity for the reason above
- There is monkey patching on `flax` for `with_sharding_constraint` because
  they wouldn't be emitted on a CPU backend.

[GSPMD's pipeline parallelism Sec 3.3]: https://arxiv.org/abs/2105.04663
[praxis]: https://github.com/google/praxis/blob/d21f379cdb787fba403ca9e56eb7b1aa33a28a36/praxis/layers/pipeline.py#L92-L161
[Pathways]: https://arxiv.org/abs/2203.12533
