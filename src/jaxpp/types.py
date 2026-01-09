# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools as it
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

import jax
from jax.sharding import NamedSharding, PartitionSpec

from jaxpp.mesh import MpmdMesh

PyTree = Any
ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
MpmdShardingPyTree = Any
DLPackCapsule = Any


ScalarUid = NewType("ScalarUid", int)

MpmdIdx = NewType("MpmdIdx", int)


@dataclass(frozen=True)
class MpmdSharding:
    mpmd_mesh: MpmdMesh
    # NOTE: mesh_ids can be empty for unused arrays
    # NOTE: It's always converted to a frozenset
    mesh_ids: set[int]
    spec: PartitionSpec

    def __post_init__(self):
        object.__setattr__(self, "mesh_ids", frozenset(self.mesh_ids))

    @property
    def sharding(self) -> NamedSharding:
        """Construct a JAX NamedSharding spanning all MPMD groups in mesh_ids.
        The returned sharding's mesh is a submesh of the full MPMD mesh,
        containing only the devices from the specified MPMD group indices.
        This is used for resharding between SPMD and MPMD array layouts.

        Returns:
            A NamedSharding with a mesh spanning all devices in mesh_ids
            and the partition spec from this MpmdSharding.
        """
        mesh = self.mpmd_mesh.mpmd_submesh(list(self.mesh_ids)).jax_mesh
        return NamedSharding(mesh, self.spec)


UID = ScalarUid


_global_uid = it.count()


def fresh_scalar_uid() -> ScalarUid:
    return ScalarUid(next(_global_uid))


class TaskType(Enum):
    FWD = 1
    BWD = 2
    BWD_I = 3
    BWD_W = 4

    def __repr__(self):
        return "%s.%s" % (self.__class__.__name__, self._name_)

    @property
    def default_latency(self):
        if self is TaskType.BWD:
            latency = 2
        elif self in {TaskType.FWD, TaskType.BWD_I, TaskType.BWD_W}:
            latency = 1
        else:
            raise ValueError(f"Unexpected task type: {self}")
        return latency
