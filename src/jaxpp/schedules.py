# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from jaxpp.types import MpmdIdx, TaskType


@dataclass(frozen=True)
class Task:
    stage_id: int
    mubatch_idx: int
    fwd_or_bwd: TaskType
    latency: int = field(hash=False, compare=False)

    @classmethod
    def make(
        cls,
        stage_id: int,
        mubatch_idx: int,
        fwd_or_bwd: TaskType,
        latency: int | None = None,
    ):
        if latency is None:
            latency = fwd_or_bwd.default_latency
        return cls(stage_id, mubatch_idx, fwd_or_bwd, latency)

    def __str__(self):
        return f"{mk_task_name(self.stage_id, self.fwd_or_bwd, self.mubatch_idx)}"


class FusedTask(tuple[Task, ...]):
    @property
    def latency(self):
        return sum(t.latency for t in self)

    def __str__(self):
        return f"{FusedTask.__name__}({', '.join(str(t) for t in self)})"


ScheduleTasks = list[list[Task | FusedTask | None]]

FWD = TaskType.FWD
BWD_A = TaskType.BWD_I
BWD_W = TaskType.BWD_W


class SequentialMicrobatchesIterator:
    def __init__(self):
        self.task_mubatch = defaultdict[tuple[int, TaskType], int](lambda: 0)

    def task(self, stage_id: int, task_type: TaskType):
        microbatch = self.task_mubatch[(stage_id, task_type)]
        res = Task.make(stage_id=stage_id, mubatch_idx=microbatch, fwd_or_bwd=task_type)
        self.task_mubatch[(stage_id, task_type)] += 1
        return res

    def fwd(self, stage_id: int) -> Task:
        return self.task(stage_id, FWD)

    def bwd(self, stage_id) -> FusedTask | Task:
        return self.task(stage_id, TaskType.BWD)

    def fwd_bwd(self, fwd_stage_id, bwd_stage_id) -> FusedTask:
        bwd = self.bwd(bwd_stage_id)
        if isinstance(bwd, Task):
            bwd = (bwd,)
        return FusedTask((self.fwd(fwd_stage_id), *bwd))


class ZBSequentialMicrobatchesIterator(SequentialMicrobatchesIterator):
    # Activation backwards
    def bwd_a(self, stage_id: int) -> Task:
        return self.task(stage_id=stage_id, task_type=BWD_A)

    # Weight backwards
    def bwd_w(self, stage_id: int) -> Task:
        return self.task(stage_id=stage_id, task_type=BWD_W)

    def bwd(self, stage_id) -> FusedTask:
        return FusedTask((self.bwd_a(stage_id), self.bwd_w(stage_id)))


def dualpipev_tasks(mpmd_dim: int, mpmd_idx: int, n_mubatches: int):
    # Adapted from https://github.com/deepseek-ai/DualPipe/blob/3da1bbea53606543d7f5f232338fc58096db30e3/dualpipe/dualpipev.py#L288
    it = ZBSequentialMicrobatchesIterator()

    # Each mpmd_idx has 2 stages to run: stage0 and stage1
    stage0 = mpmd_idx
    stage1 = mpmd_dim * 2 - mpmd_idx - 1
    mpmd_idx_tasks = []

    # Step 1: nF0
    section_tasks = (mpmd_dim - mpmd_idx - 1) * 2
    mpmd_idx_tasks.extend([it.fwd(stage0) for _ in range(section_tasks)])

    # Step 2: nF0F1
    section_tasks = mpmd_idx + 1
    for idx in range(section_tasks):
        mpmd_idx_tasks.extend([it.fwd(stage0), it.fwd(stage1)])

    # Step 3: nB1W1F1 (Use zero bubble)
    section_tasks = mpmd_dim - mpmd_idx - 1
    for idx in range(section_tasks):
        mpmd_idx_tasks.extend([it.bwd_a(stage1), it.bwd_w(stage1), it.fwd(stage1)])

    # Step 4 (Main step): nF0B1F1B0
    section_tasks = n_mubatches - mpmd_dim * 2 + mpmd_idx + 1
    for idx in range(section_tasks):
        if idx == 0:
            if mpmd_idx == mpmd_dim - 1:
                mpmd_idx_tasks.append(it.fwd(stage0))
                mpmd_idx_tasks.append(it.bwd(stage1))
            else:
                mpmd_idx_tasks.append(it.fwd_bwd(stage0, stage1))
        else:
            mpmd_idx_tasks.append(it.fwd_bwd(stage0, stage1))

        mpmd_idx_tasks.append(it.fwd_bwd(stage1, stage0))

    # Step 5: nB1F1B0
    section_tasks = mpmd_dim - mpmd_idx - 1
    for idx in range(section_tasks):
        mpmd_idx_tasks.append(it.bwd(stage1))
        mpmd_idx_tasks.append(it.fwd_bwd(stage1, stage0))

    # Step 6: nB1B0 (The second half of the chunks use zero bubble)
    section_tasks = mpmd_idx + 1
    for idx in range(section_tasks):
        # Reference: enable_zb switches at step_6 // 2 based on rank % 2
        enable_zb_at = section_tasks // 2

        # First backward (stage1)
        if idx >= enable_zb_at and mpmd_idx % 2 == 1:
            # Switch to zero bubble for odd ranks
            mpmd_idx_tasks.append(it.bwd_a(stage1))
        else:
            mpmd_idx_tasks.append(it.bwd(stage1))

        # Second backward (stage0)
        if (
            (
                # For the first stage (stage0 == 0), generate BWD_I and BWD_W together
                stage0 != 0
            )
            and idx >= enable_zb_at
            and mpmd_idx % 2 == 0
        ):
            # Switch to zero bubble for even ranks
            mpmd_idx_tasks.append(it.bwd_a(stage0))
        else:
            mpmd_idx_tasks.append(it.bwd(stage0))

    # Step 7: nWB0 (Use zero bubble)
    section_tasks = mpmd_dim - mpmd_idx - 1
    for idx in range(section_tasks):
        # For the first stage (stage0 == 0), generate BWD_I and BWD_W together
        if stage0 == 0:
            mpmd_idx_tasks.append(it.bwd(stage0))
        else:
            mpmd_idx_tasks.append(it.bwd_a(stage0))

    # Step 8: nW
    _ = {
        (stage_id, mubatch_idx)
        for (stage_id, task_type), mubatch_idx in it.task_mubatch.items()
        if task_type == TaskType.BWD_W
    }
    assert len(_) == 2
    dw_tasks = []
    for stage_id, mubatch_idx in _:
        for _ in range(mubatch_idx, n_mubatches):
            dw_tasks.append(it.bwd_w(stage_id))

    mpmd_idx_tasks.extend(sorted(dw_tasks, key=lambda x: (x.mubatch_idx, x.stage_id)))
    return mpmd_idx_tasks


@dataclass(eq=True, frozen=True)
class BaseSchedule(metaclass=ABCMeta):
    num_stages: int
    is_partial_bwd: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.num_stages <= 0:
            raise ValueError("The argument `num_stages` must be `>= 0`")

    def get_mpmd_idx(self, stage_id: int) -> MpmdIdx:
        return MpmdIdx(stage_id)

    @staticmethod
    def get_num_stages(num_tasks: int) -> int:
        # There are 2n - 1 tasks for n stages because fwd and bwd are fused for the
        # last stage.
        num_stages, rem = divmod(num_tasks, 2)
        # num_stages, rem = divmod(num_tasks + 1, 2)
        assert rem == 0
        return num_stages

    @abstractmethod
    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        raise NotImplementedError


@dataclass(eq=True, frozen=True)
class InterleavedBaseSchedule(BaseSchedule):
    mpmd_dim: int

    def __post_init__(self):
        super().__post_init__()
        if self.mpmd_dim <= 0:
            raise ValueError("The argument `mpmd_dim` must be `>= 0`")
        if self.num_stages % self.mpmd_dim != 0:
            raise ValueError(
                f"{self.num_stages=} can not be evenly divided by {self.mpmd_dim=}. "
                f"Remainder: {divmod(self.num_stages, self.mpmd_dim)=}"
            )

    def get_mpmd_idx(self, stage_id: int) -> MpmdIdx:
        return MpmdIdx(stage_id % self.mpmd_dim)


@dataclass(eq=True, frozen=True)
class Std1F1B(BaseSchedule):
    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        steps = n_mubatches + self.num_stages - 1
        schedule = ScheduleTasks([[None] * (steps * 2) for _ in range(self.num_stages)])

        stage_mubatch = [[0, 0] for _ in range(self.num_stages)]

        # warmup
        for step in range(self.num_stages):
            for stage_id in range(self.num_stages):
                if step >= stage_id:
                    mubatch_idx = stage_mubatch[stage_id][0]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[stage_id][step] = Task.make(
                            stage_id, mubatch_idx, TaskType.FWD
                        )
                        stage_mubatch[stage_id][0] += 1

        # steady state and cooldown
        for step in range(self.num_stages, 2 * steps):
            relative_step = step - self.num_stages
            for stage_id in range(self.num_stages):
                inv_stage = self.num_stages - stage_id - 1
                if relative_step >= inv_stage:
                    fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
                    task_type = TaskType.FWD if fwd_or_bwd == 0 else TaskType.BWD
                    mubatch_idx = stage_mubatch[stage_id][fwd_or_bwd]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[stage_id][step] = Task.make(
                            stage_id, mubatch_idx, task_type
                        )
                        stage_mubatch[stage_id][fwd_or_bwd] += 1

        return schedule


@dataclass(eq=True, frozen=True)
class Eager1F1B(BaseSchedule):
    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        steps = n_mubatches + self.num_stages - 1
        schedule = ScheduleTasks([[None] * (steps * 2) for _ in range(self.num_stages)])

        stage_mubatch = [[0, 0] for _ in range(self.num_stages)]

        # warmup
        for step in range(2 * self.num_stages - 1):
            for stage_id in range(self.num_stages):
                if (step < self.num_stages and step >= stage_id) or (
                    step >= self.num_stages
                    and step - self.num_stages < self.num_stages - 1 - stage_id
                ):
                    mubatch_idx = stage_mubatch[stage_id][0]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[stage_id][step] = Task.make(
                            stage_id, mubatch_idx, TaskType.FWD
                        )
                        stage_mubatch[stage_id][0] += 1

        # steady state and cooldown
        for step in range(self.num_stages, 2 * steps):
            relative_step = step - self.num_stages
            for stage_id in range(self.num_stages):
                inv_stage = self.num_stages - stage_id - 1
                if relative_step >= inv_stage:
                    fwd_or_bwd = 1 - (relative_step + inv_stage) % 2
                    task_type = TaskType.FWD if fwd_or_bwd == 0 else TaskType.BWD
                    mubatch_idx = stage_mubatch[stage_id][fwd_or_bwd]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[stage_id][step] = Task.make(
                            stage_id, mubatch_idx, task_type
                        )
                        stage_mubatch[stage_id][fwd_or_bwd] += 1
        return schedule


@dataclass(eq=True, frozen=True)
class GPipe(BaseSchedule):
    @staticmethod
    def get_num_stages(num_tasks: int) -> int:
        # There are 2n tasks for n stages.
        num_stages, rem = divmod(num_tasks, 2)
        assert rem == 0
        return num_stages

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        steps = n_mubatches + self.num_stages - 1
        schedule = ScheduleTasks([[None] * (steps * 2) for _ in range(self.num_stages)])
        for step in range(steps):
            for stage_id in range(self.num_stages):
                mubatch_idx = step - stage_id
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[stage_id][step] = Task.make(
                        stage_id, mubatch_idx, TaskType.FWD
                    )

        for step in range(steps, steps * 2):
            for stage_id in reversed(range(self.num_stages)):
                mubatch_idx = (step - steps) - (self.num_stages - stage_id - 1)
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[stage_id][step] = Task.make(
                        stage_id, mubatch_idx, TaskType.BWD
                    )

        return schedule


@dataclass(eq=True, frozen=True)
class ZeroBubble(BaseSchedule):
    def __post_init__(self):
        super().__post_init__()
        # Set self.is_partial_bwd with object.__setattr__ because
        # the class is a frozen dataclass.
        object.__setattr__(self, "is_partial_bwd", True)

    @staticmethod
    def get_num_stages(num_tasks: int) -> int:
        # There are 3n - 2 tasks for n stages because fwd and bwd_i are fused for the
        # last stage and bwd_i and bwd_w are fused for the first stage.
        num_stages, rem = divmod(num_tasks + 2, 3)
        assert rem == 0
        return num_stages

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        return self.build_schedule(n_mubatches)

    def build_schedule(self, n_mubatches: int) -> ScheduleTasks:
        assert (
            n_mubatches >= self.num_stages
        ), f"Expect num of microbatches >= num of stages, but {n_mubatches} microbatches and {self.num_stages} stages found"
        steps = n_mubatches * 3 + (self.num_stages - 1)
        schedule: ScheduleTasks = [([None] * steps) for _ in range(self.num_stages)]
        fwd_stage_mubatch = [0] * self.num_stages
        bwd_i_stage_mubatch = [0] * self.num_stages
        bwd_w_stage_mubatch = [0] * self.num_stages
        task_type = [TaskType.BWD_I] * self.num_stages

        # warmup - fwd
        for stage_id in range(self.num_stages):
            for step_id in range(self.num_stages):
                if step_id >= stage_id:
                    mubatch_idx = fwd_stage_mubatch[stage_id]
                    schedule[stage_id][step_id] = Task.make(
                        stage_id, mubatch_idx, TaskType.FWD
                    )
                    fwd_stage_mubatch[stage_id] += 1

        # warmup - fwd + bwd_i
        pivot = self.num_stages + self.num_stages
        for stage_id in range(self.num_stages):
            for step_id in range(pivot - 1 - stage_id, pivot - 1 + stage_id):
                cur_kind = task_type[stage_id]
                mubatch_idx = None
                if cur_kind is TaskType.BWD_I:
                    mubatch_idx = bwd_i_stage_mubatch[stage_id]
                    bwd_i_stage_mubatch[stage_id] += 1
                    next_kind = TaskType.FWD
                elif cur_kind is TaskType.FWD:
                    mubatch_idx = fwd_stage_mubatch[stage_id]
                    fwd_stage_mubatch[stage_id] += 1
                    next_kind = TaskType.BWD_I
                else:
                    raise ValueError(f"Unexpected stage type in warmup: {cur_kind}")
                schedule[stage_id][step_id] = Task.make(stage_id, mubatch_idx, cur_kind)
                task_type[stage_id] = next_kind

        # steady state and cooldown
        def get_next(kind, stage_id):
            # Rotate through BWD_I -> BWD_W -> FWD to find the task type whose
            # mubatch_idx is less than n_mubatches.
            # If kind is the same as next_kind, we know that all task types have been
            # tried.
            next_kind = kind
            while True:
                curr_kind = next_kind
                if curr_kind is TaskType.BWD_I:
                    next_kind = TaskType.BWD_W
                    if bwd_i_stage_mubatch[stage_id] < n_mubatches:
                        mubatch_idx = bwd_i_stage_mubatch[stage_id]
                        bwd_i_stage_mubatch[stage_id] += 1
                        return mubatch_idx, curr_kind, next_kind
                elif curr_kind is TaskType.BWD_W:
                    next_kind = TaskType.FWD
                    if bwd_w_stage_mubatch[stage_id] < n_mubatches:
                        mubatch_idx = bwd_w_stage_mubatch[stage_id]
                        bwd_w_stage_mubatch[stage_id] += 1
                        return mubatch_idx, curr_kind, next_kind
                elif curr_kind is TaskType.FWD:
                    next_kind = TaskType.BWD_I
                    if fwd_stage_mubatch[stage_id] < n_mubatches:
                        mubatch_idx = fwd_stage_mubatch[stage_id]
                        fwd_stage_mubatch[stage_id] += 1
                        return mubatch_idx, curr_kind, next_kind
                else:
                    raise ValueError(f"Unexpected stage type: {curr_kind}")
                if kind == next_kind:
                    raise ValueError(
                        "All tasks have been already scheduled for all mubatches"
                    )

        assert next_kind is TaskType.BWD_I
        for stage_id in range(self.num_stages):
            next_kind = TaskType.BWD_I
            for step_id in range(pivot - 1 + stage_id, steps):
                mubatch_idx, cur_kind, next_kind = get_next(next_kind, stage_id)
                schedule[stage_id][step_id] = Task.make(stage_id, mubatch_idx, cur_kind)

        return schedule


# From PyTorch: https://github.com/pytorch/pytorch/blob/e619c6bb90b9dedaccd3cbeed86a288993a4e33f/torch/distributed/pipelining/schedules.py#L2247-L2265
@dataclass(eq=True, frozen=True)
class Interleaved1F1B(InterleavedBaseSchedule):
    fuse_steady_state: bool = False

    def __post_init__(self):
        super().__post_init__()
        vp, _ = divmod(self.num_stages, self.mpmd_dim)
        if _ != 0:
            raise ValueError(
                f"{self.num_stages=} must be divisible by {self.mpmd_dim=}"
            )
        # Set self.vp and self.is_partial_bwd with object.__setattr__ because
        # the class is a frozen dataclass.
        object.__setattr__(self, "vp", vp)
        object.__setattr__(self, "is_partial_bwd", False)

    def microbatches_per_round(self, n_microbatches: int):
        number_of_rounds = max(1, n_microbatches // self.mpmd_dim)
        microbatches_per_round, _ = divmod(n_microbatches, number_of_rounds)
        if _ != 0:
            raise ValueError("n_microbatches must be divisible by mpmd_dim")
        return microbatches_per_round

    def _get_rank_warmup_ops(self, mpmd_idx, n_microbatches: int) -> int:
        microbatches_per_round = self.microbatches_per_round(n_microbatches)

        # Warms up operations for last stage
        warmups_ops_last_stage = (self.vp - 1) * microbatches_per_round
        # Increment warmup operations by 2 for each hop away from the last stage
        multiply_factor = 2
        warmup_ops = warmups_ops_last_stage + multiply_factor * (
            (self.mpmd_dim - 1) - mpmd_idx
        )
        return warmup_ops

    def get_rank_warmup_ops(self, mpmd_idx, n_microbatches: int) -> int:
        warmup_ops = self._get_rank_warmup_ops(mpmd_idx, n_microbatches)
        # We cannot have more warmup operations than there are number of microbatches,
        # so cap it there
        return min(warmup_ops, n_microbatches * self.vp)

    def forward_stage_index(
        self, mpmd_idx: int, step: int, microbatches_per_round: int
    ):
        # Get the local index from 0 to n_local_stages-1
        local_index = (step // microbatches_per_round) % self.vp
        return (local_index * self.mpmd_dim) + mpmd_idx

    def backward_stage_index(self, step, warmup_ops, microbatches_per_round, mpmd_idx):
        local_index = (
            self.vp - 1 - ((step - warmup_ops) // microbatches_per_round) % self.vp
        )
        return (local_index * self.mpmd_dim) + mpmd_idx

    def _tasks_for_rank(self, mpmd_idx: int, n_mubatches: int) -> ScheduleTasks:
        microbatch_ops = self.vp * n_mubatches
        warmup_ops = self.get_rank_warmup_ops(mpmd_idx, n_mubatches)
        fwd_bwd_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        it = (
            ZBSequentialMicrobatchesIterator()
            if self.is_partial_bwd
            else SequentialMicrobatchesIterator()
        )

        microbatches_per_round = self.microbatches_per_round(n_mubatches)
        tasks = []
        # Warmup
        for step in range(warmup_ops):
            tasks.append(
                it.fwd(self.forward_stage_index(mpmd_idx, step, microbatches_per_round))
            )
        # Steady state
        for step in range(warmup_ops, warmup_ops + fwd_bwd_ops):
            fwd_idx = self.forward_stage_index(mpmd_idx, step, microbatches_per_round)
            bwd_idx = self.backward_stage_index(
                step, warmup_ops, microbatches_per_round, mpmd_idx
            )
            fwd = it.fwd(fwd_idx)
            bwd = it.bwd(bwd_idx)

            if (not self.fuse_steady_state) and fwd_idx != self.num_stages - 1:
                tasks.extend([fwd, bwd])
            else:
                if isinstance(bwd, Task):
                    bwd = (bwd,)
                tasks.append(FusedTask((fwd, *bwd)))

        # Cooldown
        for step in range(
            warmup_ops + fwd_bwd_ops, warmup_ops + fwd_bwd_ops + cooldown_ops
        ):
            tasks.append(
                it.bwd(
                    self.backward_stage_index(
                        step, warmup_ops, microbatches_per_round, mpmd_idx
                    ),
                )
            )
        return tasks

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        return [
            self._tasks_for_rank(mpmd_idx, n_mubatches)
            for mpmd_idx in range(self.mpmd_dim)
        ]


class KimiK2(Interleaved1F1B):
    def _get_rank_warmup_ops(self, mpmd_idx, n_microbatches):
        return super()._get_rank_warmup_ops(mpmd_idx, n_microbatches) + 1


@dataclass(eq=True, frozen=True)
class InterleavedGPipe(InterleavedBaseSchedule):
    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        FWD, BWD = 0, 1
        half_steps = n_mubatches * 2 + self.mpmd_dim - 1
        n_steps = half_steps * 2
        schedule = ScheduleTasks([([None] * n_steps) for _ in range(self.mpmd_dim)])
        stage_mubatch = list[list[tuple[int, int, int]]](
            [(dim_id, 0, 0), (dim_id + self.mpmd_dim, 0, 0)]
            for dim_id in range(self.mpmd_dim)
        )

        def get_next(n_stages, mpmd_idx, fwd_or_bwd, value, stage_id, count):
            assert value == 1, f"Expect an update by increasing 1, but `{value}` found."
            count += value
            rem_stages = count % n_stages
            is_fwd = fwd_or_bwd == 0
            stage_id = (
                (mpmd_idx if is_fwd else mpmd_idx + self.mpmd_dim)
                if rem_stages < self.mpmd_dim
                else (mpmd_idx + self.mpmd_dim if is_fwd else mpmd_idx)
            )
            mubatch_idx = (count // n_stages) * self.mpmd_dim + (count % self.mpmd_dim)
            return (stage_id, mubatch_idx, count)

        # fwd: the first half
        for step in range(half_steps):
            for mpmd_idx in range(self.mpmd_dim):
                if step >= mpmd_idx:
                    stage_id, mubatch_idx, count = stage_mubatch[mpmd_idx][0]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[mpmd_idx][step] = Task.make(
                            stage_id, mubatch_idx, TaskType.FWD
                        )
                        stage_mubatch[mpmd_idx][0] = get_next(
                            self.num_stages, mpmd_idx, FWD, 1, stage_id, count
                        )

        # bwd: the second half
        for step in range(half_steps, n_steps):
            relative_step = step - half_steps
            for mpmd_idx in range(self.mpmd_dim):
                inv_step = self.mpmd_dim - mpmd_idx - 1
                if relative_step >= inv_step:
                    stage_id, mubatch_idx, count = stage_mubatch[mpmd_idx][BWD]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[mpmd_idx][step] = Task.make(
                            stage_id, mubatch_idx, TaskType.BWD
                        )
                        stage_mubatch[mpmd_idx][0] = get_next(
                            self.num_stages, mpmd_idx, BWD, 1, stage_id, count
                        )

        return schedule




@dataclass(eq=True, frozen=True)
class DualPipeV(InterleavedBaseSchedule):
    def __post_init__(self):
        super().__post_init__()
        q, r = divmod(self.num_stages, self.mpmd_dim)
        if q != 2 or r != 0:
            raise ValueError(
                f"{DualPipeV.__name__} only supports 2 * mpmd_dim stages,"
                f" {self.num_stages=} requested with {self.mpmd_dim=}"
            )
        # Set self.is_partial_bwd with object.__setattr__ because
        # the class is a frozen dataclass.
        object.__setattr__(self, "is_partial_bwd", True)

    def get_mpmd_idx(self, stage_id: int) -> MpmdIdx:
        q, r = divmod(stage_id, self.mpmd_dim)
        if q % 2 == 0:
            return r
        else:
            return (self.mpmd_dim - 1) - r

    def tasks(self, n_mubatches: int) -> list[list[Task | FusedTask]]:
        if not (n_mubatches > 0 and n_mubatches >= self.num_stages):
            raise ValueError(
                f"{DualPipeV.__name__} requires {n_mubatches=} >= "
                f"{self.num_stages=} ({self.mpmd_dim=})"
            )
        return [
            dualpipev_tasks(self.mpmd_dim, mpmd_idx, n_mubatches)
            for mpmd_idx in range(self.mpmd_dim)
        ]


def strip_nones(ts: list[Task | FusedTask | None]):
    return [t for t in ts if t is not None]


def unpack_fused_tasks_fn(ts: list[Task | FusedTask]):
    return [
        t
        for maybe_fused_task in ts
        for t in (
            maybe_fused_task
            if isinstance(maybe_fused_task, FusedTask)
            else [maybe_fused_task]
        )
    ]


def check_and_strip_adjecent_bwd_a_bwd_w(
    tasks: Sequence[Task | FusedTask], first_stage_id: Any
):
    def _check_and_strip_fused(task: FusedTask):
        tasks = check_and_strip_adjecent_bwd_a_bwd_w(list(task), first_stage_id)
        if len(tasks) == 1:
            return tasks[0]
        return FusedTask(tasks)

    res = []
    i = 0
    while i < len(tasks) - 1:
        offset = 1
        task = tasks[i]
        if isinstance(task, FusedTask):
            res.append(_check_and_strip_fused(task))
            i += 1
            continue

        next_task = tasks[i + 1]
        if task.stage_id == first_stage_id and task.fwd_or_bwd is TaskType.BWD_I:
            # We expect BWD_I to always be followed by BWD_W
            # in any schedule (as we don't know how to split the first stage BWD
            # into sensible BWD_I, BWD_W since we can't figure what activations are.
            # In the future we might want)
            if isinstance(next_task, FusedTask) or not (
                next_task.stage_id == first_stage_id
                and next_task.fwd_or_bwd is TaskType.BWD_W
            ):
                # TODO(first_stage)
                raise NotImplementedError(
                    f"{TaskType.BWD_I} is not followed by {TaskType.BWD_W} for "
                    f"first_stage {first_stage_id} at tasks {i} {tasks[i:i+2]}.\n"
                    f"{[str(_) for _ in tasks]}"
                )
            task = dataclasses.replace(task, latency=task.latency + next_task.latency)
            offset = 2

        res.append(task)
        i += offset

    # The last task might have not been fused
    if i < len(tasks):
        assert i == len(tasks) - 1
        task = tasks[i]
        if isinstance(task, FusedTask):
            res.append(_check_and_strip_fused(task))
        else:
            res.append(task)

    return res


def preprocess_schedule_tasks(
    schedule: list[list[Task | FusedTask | None]],
    first_stage_id,
    unpack_fused_tasks: bool,
):
    # TODO: remove None stripping once all the schedules have been updated
    # Strip `None`s

    tasks = [strip_nones(tl) for tl in schedule]
    if unpack_fused_tasks:
        tasks = [unpack_fused_tasks_fn(tasks) for tasks in tasks]

    return [
        check_and_strip_adjecent_bwd_a_bwd_w(_, first_stage_id=first_stage_id)
        for _ in tasks
    ]


def mk_task_name(stage_id, ty: TaskType, mubatch_idx: int | None = None):
    prefix = {
        TaskType.FWD: "fwd_",
        TaskType.BWD: "bwd_",
        TaskType.BWD_I: "bwdA_",
        TaskType.BWD_W: "bwdW_",
    }
    suffix = ""
    if mubatch_idx is not None:
        suffix = f"__{mubatch_idx}"
    return f"{prefix[ty]}{stage_id}{suffix}"
