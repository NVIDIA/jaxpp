# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from collections.abc import Callable
from typing import cast

import jax
import jax.util as ju
from jax.interpreters import partial_eval

from jaxpp.arrayref import ArrayRef
from jaxpp.compilation import pjit_to_serializeable_mesh_computation
from jaxpp.mesh import RemoteMpmdMesh
from jaxpp.ops import RunOp
from jaxpp.types import (
    UID,
    DistributedSharding,
    DistributedShardingPyTree,
    PutArg,
    SerializeableSharding,
    fresh_scalar_uid,
)
from jaxpp.utils import hbytes

logger = logging.getLogger(__name__)


def run_replicated_dced(
    fun: Callable,
    in_dist_shardings: DistributedShardingPyTree,
    out_dist_shardings: DistributedShardingPyTree,
    mesh: RemoteMpmdMesh,
    *args,
):
    closed_jaxpr, out_shapes = jax.make_jaxpr(fun, return_shape=True)(*args)
    (
        flat_out_dist_shardings,
        out_shardings_tree,
    ) = jax.tree_util.tree_flatten(out_dist_shardings)
    assert out_shardings_tree == jax.tree.structure(out_shapes)

    in_uids = list[UID](fresh_scalar_uid() for _ in closed_jaxpr.in_avals)
    out_uids = list[UID](fresh_scalar_uid() for _ in closed_jaxpr.out_avals)
    init_exec_uid = fresh_scalar_uid()

    futs = []
    for mpmd_idx in range(mesh.mpmd_dim):
        used_outputs = [
            mpmd_idx in dist_sh.mesh_ids for dist_sh in flat_out_dist_shardings
        ]
        subset_out_uids = [
            uid for uid, used in jax.util.safe_zip(out_uids, used_outputs) if used
        ]

        stage_init_jaxpr = closed_jaxpr.replace(
            jaxpr=partial_eval.dce_jaxpr(closed_jaxpr.jaxpr, used_outputs)[0]
        )
        size = hbytes(stage_init_jaxpr.out_avals)
        logger.info(f"State memory for {mpmd_idx=} {size}")

        in_shardings = tuple(
            cast(DistributedSharding, dist_sh).sharding
            for dist_sh in jax.tree_util.tree_leaves(in_dist_shardings)
        )
        out_shardings = tuple(
            cast(DistributedSharding, dist_sh).sharding
            for dist_sh, used in jax.util.safe_zip(
                flat_out_dist_shardings, used_outputs
            )
            if used
        )
        compiled = pjit_to_serializeable_mesh_computation(
            stage_init_jaxpr,
            in_axis_resources=in_shardings,
            out_axis_resources=out_shardings,
            name=f"{fun.__name__}_{mpmd_idx}",
            use_pgle=False,
        )

        put_args = [
            PutArg(
                uid=in_uid,
                value=value,
                sharding=SerializeableSharding(
                    cast(jax.sharding.NamedSharding, sharding)
                ),
                mpmd_idxs={mpmd_idx},
            )
            for (in_uid, value, sharding) in ju.safe_zip(
                in_uids, jax.tree_util.tree_leaves(args), in_shardings
            )
        ]
        futs.extend(mesh.put_tensors(put_args))
        futs.extend(mesh.put_mesh_computation(mpmd_idx, init_exec_uid, compiled))
        futs.extend(
            mesh.execute_instructions(
                mpmd_idx, [], [RunOp(init_exec_uid, in_uids, subset_out_uids)], []
            )
        )

    mesh.blocking_tree(futs)

    flat_state = [
        ArrayRef(dist_sh, out_uid, mesh, outvar.aval)
        for dist_sh, out_uid, outvar in jax.util.safe_zip(
            flat_out_dist_shardings, out_uids, closed_jaxpr.jaxpr.outvars
        )
    ]
    return jax.tree_util.tree_unflatten(out_shardings_tree, flat_state)
