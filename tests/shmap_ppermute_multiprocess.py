# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# == process 0 ==
#
# (0.4.10) root@cml-wa-dgx01:/workdir/jaxpp# CUDA_VISIBLE_DEVICES=0,1,2,3 python test/shmap_ppermute_multiprocess.py --process_id=0 --num_processes=2 --coord_addr="localhost:9876"
# Initializing...
# Initialized...
# a=Array([[[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8]],
#
#        [[ 9, 10, 11, 12],
#         [13, 14, 15, 16]],
#
#        [[17, 18, 19, 20],
#         [21, 22, 23, 24]],
#
#        [[25, 26, 27, 28],
#         [29, 30, 31, 32]],
#
#        [[33, 34, 35, 36],
#         [37, 38, 39, 40]],
#
#        [[41, 42, 43, 44],
#         [45, 46, 47, 48]],
#
#        [[49, 50, 51, 52],
#         [53, 54, 55, 56]],
#
#        [[57, 58, 59, 60],
#         [61, 62, 63, 64]]], dtype=int32) a.shape=(8, 2, 4)
# host local a.sharding=NamedSharding(mesh={'x': 2, 'y': 2}, spec=PartitionSpec('x', 'y', None)) (8, 2, 4)
# global a.sharding=NamedSharding(mesh={'x': 4, 'y': 2}, spec=PartitionSpec('x', 'y')) a.shape=(16, 2, 4)
# global res.sharding=NamedSharding(mesh={'x': 4, 'y': 2}, spec=PartitionSpec('x', 'y')) res.shape=(16, 2, 4)
# host local res=Array([[[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]]], dtype=int32) res.shape=(8, 2, 4)
#
# == process 1 ==
#
# (0.4.10) root@cml-wa-dgx01:/workdir/jaxpp# CUDA_VISIBLE_DEVICES=4,5,6,7 python test/shmap_ppermute_multiprocess.py --process_id=1 --num_processes=2 --coord_addr="localhost:9876"
# Initializing...
# Initialized...
# a=Array([[[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]],
#
#        [[0, 0, 0, 0],
#         [0, 0, 0, 0]]], dtype=int32) a.shape=(8, 2, 4)
# host local a.sharding=NamedSharding(mesh={'x': 2, 'y': 2}, spec=PartitionSpec('x', 'y', None)) (8, 2, 4)
# global a.sharding=NamedSharding(mesh={'x': 4, 'y': 2}, spec=PartitionSpec('x', 'y')) a.shape=(16, 2, 4)
# global res.sharding=NamedSharding(mesh={'x': 4, 'y': 2}, spec=PartitionSpec('x', 'y')) res.shape=(16, 2, 4)
# host local res=Array([[[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8]],
#
#        [[ 9, 10, 11, 12],
#         [13, 14, 15, 16]],
#
#        [[17, 18, 19, 20],
#         [21, 22, 23, 24]],
#
#        [[25, 26, 27, 28],
#         [29, 30, 31, 32]],
#
#        [[33, 34, 35, 36],
#         [37, 38, 39, 40]],
#
#        [[41, 42, 43, 44],
#         [45, 46, 47, 48]],
#
#        [[49, 50, 51, 52],
#         [53, 54, 55, 56]],
#
#        [[57, 58, 59, 60],
#         [61, 62, 63, 64]]], dtype=int32) res.shape=(8, 2, 4)

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import array
from jax.experimental.maps import xmap
from jax.experimental.multihost_utils import (
    global_array_to_host_local_array,
    host_local_array_to_global_array,
)
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.interpreters import xla
from jax.sharding import PartitionSpec


def get_args():
    parser = argparse.ArgumentParser()
    # pjit distributed flags
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--num_processes", default=2, type=int)
    parser.add_argument("--coord_addr", default="localhost:9876", type=str)
    return parser.parse_args()


def multiple_process_bidim_ppermute(args):
    print("Initializing...")
    jax.distributed.initialize(
        args.coord_addr, num_processes=args.num_processes, process_id=args.process_id
    )
    print("Initialized...")

    ldevs = jax.local_devices()
    n_ldevs = jax.local_device_count()
    gdevs = jax.devices()
    n_gdevs = jax.device_count()
    lmesh = jax.sharding.Mesh(np.array(ldevs).reshape(n_ldevs // 2, 2), ("x", "y"))
    # Only process 0 has real values.  Process 1 simply has a buffer full of zeros
    a = (
        np.arange(n_ldevs * 16) + 1
        if args.process_id == 0
        else np.zeros(n_ldevs * 16, dtype=jnp.int32)
    )
    a = a.reshape(n_ldevs * 2, 2, 4)
    # jax.device_put is optional.  host_local_array_to_global_array handles the sharding if needed.
    # a = jax.device_put(
    #     a,
    #     jax.sharding.NamedSharding(lmesh, jax.sharding.PartitionSpec("x", "y", None)),
    # )
    # print(f"host local {a.sharding=} {a.shape}")
    # jax.debug.visualize_array_sharding(a)
    print(f"{a=} {a.shape=}")

    # Send values from process 0 to process 1
    def f(a):
        return lax.ppermute(
            a, "x", zip(np.arange(0, n_ldevs // 2), np.arange(n_ldevs // 2, n_ldevs))
        )

    gmesh = jax.sharding.Mesh(np.array(gdevs).reshape(n_gdevs // 2, 2), ("x", "y"))
    shmap_f = shard_map(
        f,
        mesh=gmesh,
        in_specs=PartitionSpec("x", "y"),
        out_specs=PartitionSpec("x", "y"),
    )
    a = host_local_array_to_global_array(a, gmesh, jax.sharding.PartitionSpec("x", "y"))
    print(f"global {a.sharding=} {a.shape=}")
    # jax.debug.visualize_array_sharding(a)
    res = jax.jit(shmap_f)(a)
    print(f"global {res.sharding=} {res.shape=}")
    # jax.debug.visualize_array_sharding(res)

    # zeros are printed out on process 0 and values are printed out on process 1
    res = global_array_to_host_local_array(
        res, gmesh, jax.sharding.PartitionSpec("x", "y")
    )
    print(f"host local {res=} {res.shape=}")
    # jax.debug.visualize_array_sharding(res)


multiple_process_bidim_ppermute(get_args())
