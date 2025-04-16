# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, NewType, TypeVar

import jax
import jax.numpy as jnp
from jax._src.sharding_impls import UnspecifiedValue
from jax.sharding import NamedSharding
from jax.sharding import Sharding as JSharding
from typing_extensions import Self

PyTree = Any
ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
DistributedShardingPyTree = Any
DLPackCapsule = Any

MaybeSharding = JSharding | UnspecifiedValue

ScalarUid = NewType("ScalarUid", int)


@dataclass(frozen=True)
class DistributedSharding:
    mesh_ids: set[int]
    sharding: NamedSharding


class SerializeableSharding:
    def __init__(
        self,
        named_sharding: NamedSharding | UnspecifiedValue,
    ):
        if isinstance(named_sharding, UnspecifiedValue):
            self.sharding = named_sharding
        else:
            assert isinstance(
                named_sharding, NamedSharding
            ), "Unsupported sharding type"
            self.sharding = NamedSharding(
                named_sharding.mesh.abstract_mesh, named_sharding.spec
            )

    def to_named_sharding(self, mesh):
        if isinstance(self.sharding, UnspecifiedValue):
            return self.sharding
        return NamedSharding(mesh, self.sharding.spec)


UID = ScalarUid


class Bind(NamedTuple):
    from_: UID
    to_: UID


class PutArg(NamedTuple):
    uid: UID
    value: jnp.ndarray
    sharding: SerializeableSharding
    mpmd_idxs: set[int]


DeviceId = NewType("DeviceId", int)
GlobalDeviceId = NewType("GlobalDeviceId", int)
HardwareDeviceId = NewType("HardwareDeviceId", int)

WorkerId = NewType("WorkerId", int)

NcclId = NewType("NcclId", bytes)
Rank = NewType("Rank", int)

OpSharding = Any

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT
else:
    SupportsRichComparisonT = TypeVar("SupportsRichComparisonT")


_global_uid = it.count()


def fresh_scalar_uid() -> ScalarUid:
    return ScalarUid(next(_global_uid))


class UniqueSortedSequence(
    Generic[SupportsRichComparisonT], tuple[SupportsRichComparisonT, ...]
):
    __slots__ = ()

    @classmethod
    def create(cls, es: Iterable[SupportsRichComparisonT]) -> Self:
        return cls(sorted(set(es)))


class UniqueGlobalDeviceIds(UniqueSortedSequence[GlobalDeviceId]):
    @classmethod
    def strict_create(cls, ids: Iterable[GlobalDeviceId]) -> "UniqueGlobalDeviceIds":
        ids = tuple(ids)
        res = cls.create(ids)
        assert len(res) == len(ids)
        return cls(res)

    @property
    def primary(self):
        return self[0]

    @property
    def ranks(self) -> Sequence[tuple[GlobalDeviceId, Rank]]:
        return [(gid, Rank(rank)) for rank, gid in enumerate(self)]

    def rank_of(self, gid: GlobalDeviceId) -> Rank:
        return dict(self.ranks)[gid]


class CommKeyWithNcclId(NamedTuple):
    device_ids: UniqueGlobalDeviceIds
    nccl_id: NcclId
    nccl_collnet_enable: str = "0"


class TaskType(Enum):
    FWD = 1
    BWD = 2
    BWD_I = 3
    BWD_W = 4


class Task(NamedTuple):
    stage_id: int
    mubatch_idx: int
    fwd_or_bwd: TaskType


ScheduleTasks = NewType("ScheduleTasks", list[list[Task | None]])
