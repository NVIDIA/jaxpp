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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NewType, TypeVar

import jax
from jax.sharding import NamedSharding

PyTree = Any
ArrayTree = jax.Array | Iterable["ArrayTree"] | Mapping[Any, "ArrayTree"]
DistributedShardingPyTree = Any
DLPackCapsule = Any


ScalarUid = NewType("ScalarUid", int)

MpmdIdx = NewType("MpmdIdx", int)


@dataclass(frozen=True)
class DistributedSharding:
    mesh_ids: set[int]
    sharding: NamedSharding


UID = ScalarUid


if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT
else:
    SupportsRichComparisonT = TypeVar("SupportsRichComparisonT")


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
