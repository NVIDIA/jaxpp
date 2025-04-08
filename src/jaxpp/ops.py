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

import abc
import dataclasses
import enum
import itertools as it
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from jaxpp.types import (
    UID,
    GlobalDeviceId,
    SerializeableSharding,
    UniqueGlobalDeviceIds,
)


class CommDesc(NamedTuple):
    uid: UID
    aval: Any
    sharding: SerializeableSharding
    from_dev_ids: Sequence[GlobalDeviceId]
    to_dev_ids: Sequence[GlobalDeviceId]


class Operator(enum.Enum):
    SUM = enum.auto()
    MAX = enum.auto()


@dataclass
class AllReduceDesc:
    within_dev_ids: Mapping[GlobalDeviceId, UniqueGlobalDeviceIds]
    uid: UID
    new_uid: UID
    operator: Operator = Operator.SUM


class Op(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def in_uids(self) -> list[UID]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_uids(self) -> list[UID]:
        raise NotImplementedError


@dataclass
class AllReduceOp(Op):
    descs: Sequence[AllReduceDesc]
    # The flag below controls whether it's ok to use SHARP for this collective.
    # Given that few SHARP communicators can be created we selectively
    # enable it only for DP all-reduces
    nccl_collnet_enable: str = "0"

    @property
    def in_uids(self) -> list[UID]:
        return [desc.uid for desc in self.descs]

    @property
    def out_uids(self) -> list[UID]:
        return [desc.new_uid for desc in self.descs]


@dataclass
class SendOp(Op):
    comm_desc: Sequence[CommDesc]

    @property
    def in_uids(self) -> list[UID]:
        return [desc.uid for desc in self.comm_desc]

    @property
    def out_uids(self) -> list[UID]:
        return []


@dataclass
class RecvOp(Op):
    comm_desc: Sequence[CommDesc]

    @property
    def in_uids(self) -> list[UID]:
        return []

    @property
    def out_uids(self) -> list[UID]:
        return [desc.uid for desc in self.comm_desc]


def get_comm_keys(mpmd_instructions: list[list[Op]]):
    comm_keys = dict[UniqueGlobalDeviceIds, str]()
    for instruction_list in mpmd_instructions:
        for instr in instruction_list:
            if isinstance(instr, (SendOp, RecvOp)):
                for desc in instr.comm_desc:
                    for sdid, ddid in zip(
                        desc.from_dev_ids, desc.to_dev_ids, strict=True
                    ):
                        comm_keys[UniqueGlobalDeviceIds.strict_create((sdid, ddid))] = (
                            "0"
                        )
            elif isinstance(instr, AllReduceOp):
                for desc in instr.descs:
                    comm_keys.update(
                        zip(
                            desc.within_dev_ids.values(),
                            it.repeat(instr.nccl_collnet_enable),
                        )
                    )

    return sorted(comm_keys.items())


@dataclass
class RunOp(Op):
    exec_uid: UID
    _in_uids: list[UID]
    _out_uids: list[UID]

    def __post_init__(self):
        if len(set(self.in_uids)) != len(self.in_uids):
            raise AssertionError(
                f"Duplicate input found {Counter(self.in_uids).most_common(10)}"
            )

    @property
    def in_uids(self) -> list[UID]:
        return self._in_uids

    @property
    def out_uids(self) -> list[UID]:
        return self._out_uids


@dataclass
class DeleteOp(Op):
    _in_uids: list[UID]
    _out_uids: list[UID] = dataclasses.field(default_factory=list)

    @property
    def in_uids(self) -> list[UID]:
        return self._in_uids

    @property
    def out_uids(self) -> list[UID]:
        return self._out_uids


def add_delete_ops(ops: Sequence[Op], must_live: set[UID]) -> list[Op]:
    def used(op: Op) -> set[UID]:
        return set(op.in_uids)

    def defined(op: Op) -> set[UID]:
        return set(op.out_uids)

    def last_uses(ops: Sequence[Op], must_live: set[UID]) -> Sequence[set[UID]]:
        alive_before_op: list[set[UID]] = [set()] * len(ops)
        alive_before_op.append(set(must_live))

        dead_after_op: list[set[UID]] = [set()] * len(ops)
        for idx, op in reversed(list(enumerate(ops))):
            alive_before_op[idx] = used(op) | (alive_before_op[idx + 1] - defined(op))
            dead = (alive_before_op[idx] | defined(op)) - alive_before_op[idx + 1]
            if isinstance(op, AllReduceOp):
                dead = dead - used(op)
            dead_after_op[idx] = dead
        return dead_after_op

    dead_after_op = last_uses(ops, must_live)

    res = []
    for op, dead in zip(ops, dead_after_op, strict=True):
        res.append(op)
        if len(dead) > 0:
            res.append(DeleteOp(sorted(dead)))
    return res
