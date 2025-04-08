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

import pickle
from pathlib import Path

import jax

from jaxpp.arrayref import ArrayRef
from jaxpp.mesh import RemoteMpmdMesh
from jaxpp.types import DistributedSharding, PyTree, fresh_scalar_uid


def save_state(path: Path | str, state_dict: PyTree, mesh: RemoteMpmdMesh):
    if not isinstance(path, Path):
        path = Path(path)

    def is_leaf(x):
        return isinstance(x, ArrayRef)

    arrayrefs: tuple[ArrayRef, ...]
    arrayrefs, _ = jax.tree_util.tree_flatten(state_dict, is_leaf=is_leaf)
    uids_by_mpmd_idx = [[] for _ in range(mesh.mpmd_dim)]
    for arrayref in arrayrefs:
        # Use only one element from sharding.mesh_ids as we want to serialize each array
        # only once by one of the workers.
        mpmd_idx = next(iter(arrayref.mpmd_idxs))
        for mubatch_uid in arrayref.uid:
            uids_by_mpmd_idx[mpmd_idx].append(mubatch_uid)

    def save():
        for mpmd_idx in range(mesh.mpmd_dim):
            yield mesh.save_arrays(mpmd_idx, uids_by_mpmd_idx[mpmd_idx], str(path))

    mesh.blocking(save)

    state_dict = jax.tree_util.tree_map(
        lambda x: (x.sharding, x.uid, x.aval), state_dict, is_leaf=is_leaf
    )

    path.mkdir(parents=True, exist_ok=True)
    with (path / "state_dict").open(mode="wb") as f:
        pickle.dump(state_dict, f)


def load_state(path: Path | str, mesh: RemoteMpmdMesh) -> PyTree:
    if not isinstance(path, Path):
        path = Path(path)

    def is_leaf(x):
        return (
            isinstance(x, tuple)
            and len(x) == 3
            and isinstance(x[0], DistributedSharding)
        )

    with (path / "state_dict").open(mode="rb") as f:
        state_dict: PyTree = pickle.load(f)

    # We can't reuse the uids from the old state_dict to create a new state_dict.
    # If we did that, we would run into an issue with garbage collection in a situation
    # like below.
    #
    #   state = ...
    #   state_dict = state.state_dict()
    #   jaxpp.save_state(path, state_dict, mesh)
    #   new_state_dict = jaxpp.load_state(path, mesh)
    #   new_state = ...
    #   new_state = new_state.restore_state(new_state_dict)
    #
    # At this point, each arrayref in state and the corresponding arrayref in new_state
    # have references to the same array.
    #
    #   del state
    #
    # When the old state object is garbage collected, each arrayref in the old state is
    # deleted along with the acutal array, which is still needed by the new state.
    #
    state_dict = jax.tree_util.tree_map(
        lambda x: (
            x[0],
            # a pair of a list of old mubatch_uids and a list of new mubatch_uids.
            (x[1], [fresh_scalar_uid() for _ in range(len(x[1]))]),
            x[2],
        ),
        state_dict,
        is_leaf=is_leaf,
    )
    arrayrefs, _ = jax.tree_util.tree_flatten(state_dict, is_leaf=is_leaf)
    shardings_by_mpmd_idx = [[] for _ in range(mesh.mpmd_dim)]
    for dist_sh, uid, _ in arrayrefs:
        dist_sh: DistributedSharding
        # Use all elements from dist_sh.mesh_ids as want to deserialize each array on
        # all workers that need it.
        for mpmd_idx in dist_sh.mesh_ids:
            # for each pair of an old mubatch_uid and a new mubatch_uid.
            for mubatch_uid in zip(*uid, strict=True):
                shardings_by_mpmd_idx[mpmd_idx].append((dist_sh.sharding, mubatch_uid))

    def load():
        for mpmd_idx in range(mesh.mpmd_dim):
            yield mesh.load_arrays(mpmd_idx, shardings_by_mpmd_idx[mpmd_idx], str(path))

    mesh.blocking(load)

    # Returning a new `state_dict`
    return jax.tree_util.tree_map(
        lambda x: ArrayRef(x[0], x[1][1], mesh, x[2]), state_dict, is_leaf=is_leaf
    )
