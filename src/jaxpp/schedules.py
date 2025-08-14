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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from jaxpp.types import ScheduleTasks, Task, TaskType


@dataclass
class BaseSchedule(metaclass=ABCMeta):
    num_stages: int

    def __post_init__(self):
        if self.num_stages <= 0:
            raise ValueError("The argument `num_stages` must be `>= 0`")
        self.is_partial_bwd = False

    @staticmethod
    def get_num_stages(num_tasks: int) -> int:
        # There are 2n - 1 tasks for n stages because fwd and bwd are fused for the
        # last stage.
        num_stages, rem = divmod(num_tasks + 1, 2)
        assert rem == 0
        return num_stages

    @abstractmethod
    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        raise NotImplementedError


@dataclass
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
                        schedule[stage_id][step] = Task(
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
                        schedule[stage_id][step] = Task(
                            stage_id, mubatch_idx, task_type
                        )
                        stage_mubatch[stage_id][fwd_or_bwd] += 1

        return schedule


@dataclass
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
                        schedule[stage_id][step] = Task(
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
                        schedule[stage_id][step] = Task(
                            stage_id, mubatch_idx, task_type
                        )
                        stage_mubatch[stage_id][fwd_or_bwd] += 1
        return schedule


@dataclass
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
                    schedule[stage_id][step] = Task(stage_id, mubatch_idx, TaskType.FWD)

        for step in range(steps, steps * 2):
            for stage_id in reversed(range(self.num_stages)):
                mubatch_idx = (step - steps) - (self.num_stages - stage_id - 1)
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[stage_id][step] = Task(stage_id, mubatch_idx, TaskType.BWD)

        return schedule


@dataclass
class ZeroBubble(BaseSchedule):
    def __post_init__(self):
        super().__post_init__()
        self.is_partial_bwd = True

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
                    schedule[stage_id][step_id] = Task(
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
                schedule[stage_id][step_id] = Task(stage_id, mubatch_idx, cur_kind)
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
                schedule[stage_id][step_id] = Task(stage_id, mubatch_idx, cur_kind)

        return schedule


@dataclass
class Base_MPMD_DIM_Schedule(BaseSchedule):
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


@dataclass
class Interleaved1F1B(Base_MPMD_DIM_Schedule):
    @cached_property
    def vp(self):
        return self.num_stages // self.mpmd_dim

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        if n_mubatches % self.num_stages != 0:
            raise ValueError(f"{n_mubatches=} % num_stages={self.num_stages} != 0")

        FWD, BWD = 0, 1

        num_warmup_steps = (self.vp - 1) * self.mpmd_dim + (self.mpmd_dim - 1)
        num_steps = (2 * (self.mpmd_dim - 1)) + (self.vp * n_mubatches * 2)
        tail_pos = num_steps - (self.mpmd_dim * (self.vp - 1) + self.mpmd_dim)
        schedule = ScheduleTasks([([None] * num_steps) for _ in range(self.mpmd_dim)])
        stage_mubatch = list[list[tuple[int, int]]](
            [(dim_id, 0), (dim_id + (self.vp - 1) * self.mpmd_dim, 0)]
            for dim_id in range(self.mpmd_dim)
        )

        def get_next(stage_id: int, mubatch_idx: int, fwd_or_bwd):
            mubatch_idx += 1
            if mubatch_idx % self.mpmd_dim == 0:
                x, stage_id = divmod(
                    stage_id + self.mpmd_dim
                    if fwd_or_bwd == FWD
                    else stage_id - self.mpmd_dim,
                    self.vp * self.mpmd_dim,
                )
                if x == 0:
                    mubatch_idx -= self.mpmd_dim
            return stage_id, mubatch_idx

        def is_fwd_or_bwd(step, relative_step, offset):
            if step < num_warmup_steps + offset:
                return FWD
            if step > tail_pos:
                return BWD
            return relative_step % 2

        def should_leave_empty(step, offset):
            return (
                num_warmup_steps + offset <= step
                and step < num_warmup_steps + offset * 2
            )

        # warmup
        for step in range(num_warmup_steps):
            for mpmd_idx in range(self.mpmd_dim):
                if step >= mpmd_idx:
                    stage_id, mubatch_idx = stage_mubatch[mpmd_idx][FWD]
                    if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                        schedule[mpmd_idx][step] = Task(
                            stage_id, mubatch_idx, TaskType.FWD
                        )
                        stage_mubatch[mpmd_idx][FWD] = get_next(
                            stage_id, mubatch_idx, FWD
                        )
        # steady state and cooldown
        for step in range(num_warmup_steps, num_steps):
            relative_step = step - num_warmup_steps
            for mpmd_idx in range(self.mpmd_dim):
                offset = self.mpmd_dim - mpmd_idx - 1
                if should_leave_empty(step, offset):
                    continue
                fwd_or_bwd = is_fwd_or_bwd(step, relative_step, offset)
                stage_id, mubatch_idx = stage_mubatch[mpmd_idx][fwd_or_bwd]
                if mubatch_idx >= 0 and mubatch_idx < n_mubatches:
                    schedule[mpmd_idx][step] = Task(
                        stage_id,
                        mubatch_idx,
                        TaskType.FWD if fwd_or_bwd == 0 else TaskType.BWD,
                    )
                    stage_mubatch[mpmd_idx][fwd_or_bwd] = get_next(
                        stage_id, mubatch_idx, fwd_or_bwd
                    )

        return schedule


@dataclass
class InterleavedGPipe(Base_MPMD_DIM_Schedule):
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
                        schedule[mpmd_idx][step] = Task(
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
                        schedule[mpmd_idx][step] = Task(
                            stage_id, mubatch_idx, TaskType.BWD
                        )
                        stage_mubatch[mpmd_idx][0] = get_next(
                            self.num_stages, mpmd_idx, BWD, 1, stage_id, count
                        )

        return schedule


