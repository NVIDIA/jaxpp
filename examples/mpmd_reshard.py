# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example of using spmd_to_mpmd_reshard to reshard the data between SPMD and MPMD meshes.

Example usage:
N_PROCS=2 N_GPUS=4 COMMAND="python -u examples/mpmd_reshard.py" ./scripts/local_mc.sh
N_PROCS=4 N_GPUS=2 COMMAND="python -u examples/mpmd_reshard.py" ./scripts/local_mc.sh
"""

import functools

import jax

import jaxpp.api as jaxpp
import jaxpp.distributed_utils as jppdu


def main():
    spmd_mesh = jax.make_mesh((2, 2, 2), ("stage", "fsdp", "tensor"))
    mpmd_mesh = jaxpp.MpmdMesh(spmd_mesh, "stage")

    embed_dim = 1024
    hidden_dim = embed_dim * 4
    rng_key = jax.random.PRNGKey(0)

    shardings = [
        # NOTE: when initializing the weights, we want to shard also across the
        # "stage" axis to be _memory efficient_
        jax.sharding.NamedSharding(
            spmd_mesh, jax.sharding.PartitionSpec(("stage", "fsdp"), "tensor")
        ),
        jax.sharding.NamedSharding(
            spmd_mesh, jax.sharding.PartitionSpec("tensor", ("stage", "fsdp"))
        ),
        jax.sharding.NamedSharding(
            spmd_mesh, jax.sharding.PartitionSpec("fsdp", None, None)
        ),
    ]

    # Traditional SPMD initialization in SPMD JAX
    @functools.partial(jax.jit, out_shardings=shardings)
    def init_fn(rng_key):
        return [
            jax.random.normal(rng_key, (embed_dim, hidden_dim)),
            jax.random.normal(rng_key, (hidden_dim, embed_dim)),
            jax.random.normal(rng_key, (4, 128, embed_dim)),
        ]

    W1, W2, x = init_fn(rng_key)

    @jax.grad
    def forward_fn(weights, x):
        W1, W2 = weights
        h1 = x @ W1
        h1 = jaxpp.pipeline_enter_stage(h1)
        h2 = h1 @ W2
        h2 = jaxpp.pipeline_enter_stage(h2)
        return ((x - h2) ** 2).sum()

    @functools.partial(
        jaxpp.mpmd_jit_with_loop,
        mpmd_mesh=mpmd_mesh,
        # NOTE: `mpmd_mesh.lowering_mesh().shape["stage"] == 1`
        in_shardings=tuple(_.spec for _ in shardings) + (shardings[-1].spec,),
        out_shardings=tuple(_.spec for _ in shardings[:2]),
    )
    def accumulation_loop(W1, W2, x, unused):
        return jaxpp.treduce(
            functools.partial(forward_fn, (W1, W2)),
            x,
            schedule=jaxpp.Std1F1B(mpmd_mesh.mpmd_dim),
            operation=jaxpp.Add,
        )

    args_mpmd_shardings, kwargs_mpmd_shardings = accumulation_loop.trace_and_place(
        W1,
        W2,
        x,
        x,  # unused
    ).in_shardings

    print(args_mpmd_shardings)
    # NOTE: the partition specs of the shardings are the same as the one of the input shardings
    #  however the meshes is the one corresponding to the spmd mesh of this process
    # (
    #   MpmdSharding(mesh_ids={0}, sharding=NamedSharding(mesh=Mesh('stage': 1, 'fsdp': 2, 'tensor': 2, axis_types=(Auto, Auto, Auto)), spec=PartitionSpec(('stage', 'fsdp'), 'tensor'), memory_kind=device))
    #   MpmdSharding(mesh_ids={1}, sharding=NamedSharding(mesh=Mesh('stage': 1, 'fsdp': 2, 'tensor': 2, axis_types=(Auto, Auto, Auto)), spec=PartitionSpec('tensor', ('stage', 'fsdp')), memory_kind=device)),
    #   MpmdSharding(mesh_ids={0}, sharding=NamedSharding(mesh=Mesh('stage': 1, 'fsdp': 2, 'tensor': 2, axis_types=(Auto, Auto, Auto)), spec=PartitionSpec('fsdp', None, None), memory_kind=device))
    # )

    # This reshard makes
    _W1, _W2, _x, _unused = jaxpp.spmd_to_mpmd_reshard(
        mpmd_mesh,
        [W1, W2, x, x],
        list(args_mpmd_shardings),
    )

    # The arguments to spmd_to_mpmd_reshard are deleted and should not be used
    # further
    print(f"{W1.is_deleted()=}")
    print(f"{W2.is_deleted()=}")
    print(f"{x.is_deleted()=}")
    W1, W2, x = _W1, _W2, _x

    # W1 is present only on process 0
    print(f"{W1.is_partially_addressable=}")
    # 0: True
    # 1: False

    if W1.is_partially_addressable:
        print(f"{W1.first_mpmd_replica.sharding=}")

    # W2 is present only on process 1
    print(f"{W2.is_partially_addressable=}")
    # 0: False
    # 1: True
    if W2.is_partially_addressable:
        print(f"{W2.first_mpmd_replica.sharding=}")

    dW1, dW2 = accumulation_loop(W1, W2, x, _unused)

    dW1, dW2 = jaxpp.mpmd_to_spmd_reshard(mpmd_mesh, [dW1, dW2], shardings[:2])
    print(f"{dW1.sharding=}")
    print(f"{dW2.sharding=}")


if __name__ == "__main__":
    jppdu.distributed_main(main)
