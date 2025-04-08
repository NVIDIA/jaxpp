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

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        zero_bubble = InterleavedZeroBubble(self.num_stages, self.num_stages)
        return zero_bubble.build_schedule(n_mubatches, self.num_stages)


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


@dataclass
class InterleavedZeroBubble(Base_MPMD_DIM_Schedule):
    """
    ZB-H1, https://arxiv.org/abs/2401.10241
    """

    def __post_init__(self):
        super().__post_init__()
        self.is_partial_bwd = True

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        return self.build_schedule(n_mubatches, self.mpmd_dim)

    def build_schedule(self, n_mubatches: int, mpmd: int) -> ScheduleTasks:
        assert (
            n_mubatches >= mpmd
        ), f"Expect num of microbatches >= num of workers, but {n_mubatches} microbatches and {mpmd} workers found"
        num_repeats = self.num_stages // mpmd  # vp: number of repeats
        max_mubatches = num_repeats * n_mubatches
        steps = max_mubatches * 3 + (mpmd - 1)
        schedule: ScheduleTasks = [([None] * steps) for _ in range(mpmd)]
        fwd_stage_mubatch = [0] * mpmd
        bwd_i_stage_mubatch = [0] * mpmd
        bwd_w_stage_mubatch = [0] * mpmd
        task_type = [TaskType.BWD_I] * mpmd

        # warmup - fwd
        for mpmd_id in range(mpmd):
            for step_id in range(self.num_stages):
                if step_id >= mpmd_id and fwd_stage_mubatch[mpmd_id] < max_mubatches:
                    mubatch_idx = fwd_stage_mubatch[mpmd_id]
                    vp_id, mubatch_idx = divmod(mubatch_idx, mpmd)
                    stage_id = (vp_id % num_repeats) * mpmd + mpmd_id
                    mubatch_idx = (vp_id // num_repeats) * mpmd + mubatch_idx
                    schedule[mpmd_id][step_id] = Task(
                        stage_id, mubatch_idx, TaskType.FWD
                    )
                    fwd_stage_mubatch[mpmd_id] += 1

        # warmup - fwd + bwd_i
        pivot = self.num_stages + mpmd
        for mpmd_id in range(mpmd):
            for step_id in range(pivot - 1 - mpmd_id, pivot - 1 + mpmd_id):
                cur_kind = task_type[mpmd_id]
                mubatch_idx = None
                if cur_kind is TaskType.BWD_I:
                    if bwd_i_stage_mubatch[mpmd_id] < max_mubatches:
                        mubatch_idx = bwd_i_stage_mubatch[mpmd_id]
                        bwd_i_stage_mubatch[mpmd_id] += 1
                    next_kind = TaskType.FWD
                elif cur_kind is TaskType.FWD:
                    if fwd_stage_mubatch[mpmd_id] < max_mubatches:
                        mubatch_idx = fwd_stage_mubatch[mpmd_id]
                        fwd_stage_mubatch[mpmd_id] += 1
                    next_kind = TaskType.BWD_I
                else:
                    raise ValueError(f"Unexpected stage type in warmup: {cur_kind}")
                if mubatch_idx is not None:
                    vp_id, mubatch_idx = divmod(mubatch_idx, mpmd)
                    stage_id = (
                        vp_id % num_repeats
                        if cur_kind is TaskType.FWD
                        else num_repeats - vp_id % num_repeats - 1
                    ) * mpmd + mpmd_id
                    mubatch_idx = (vp_id // num_repeats) * mpmd + mubatch_idx
                    schedule[mpmd_id][step_id] = Task(stage_id, mubatch_idx, cur_kind)
                task_type[mpmd_id] = next_kind

        def get_cur_kind(kind, stage_id, bound, dep=3):
            # rotation order: BWD_I -> BWD_W -> FWD
            def find_next():
                mubatch_idx = None
                if kind is TaskType.BWD_I:
                    if bwd_i_stage_mubatch[stage_id] < bound:
                        mubatch_idx = bwd_i_stage_mubatch[stage_id]
                        bwd_i_stage_mubatch[stage_id] += 1
                    next_kind = TaskType.BWD_W
                elif kind is TaskType.BWD_W:
                    if bwd_w_stage_mubatch[stage_id] < bound:
                        mubatch_idx = bwd_w_stage_mubatch[stage_id]
                        bwd_w_stage_mubatch[stage_id] += 1
                    next_kind = TaskType.FWD
                elif kind is TaskType.FWD:
                    if fwd_stage_mubatch[stage_id] < bound:
                        mubatch_idx = fwd_stage_mubatch[stage_id]
                        fwd_stage_mubatch[stage_id] += 1
                    next_kind = TaskType.BWD_I
                else:
                    raise ValueError(f"Unexpected stage type: {kind}")
                return mubatch_idx, kind, next_kind

            if dep <= 0:
                return None, None, None
            mubatch_idx, cur_kind, next_kind = find_next()
            if mubatch_idx is None:
                return get_cur_kind(next_kind, stage_id, bound, dep - 1)
            return mubatch_idx, cur_kind, next_kind

        # steady + cooldown
        assert next_kind is TaskType.BWD_I
        for mpmd_id in range(mpmd):
            next_kind = TaskType.BWD_I
            for step_id in range(pivot - 1 + mpmd_id, steps):
                if next_kind is None:  # skip remaining items
                    break
                mubatch_idx, cur_kind, next_kind = get_cur_kind(
                    next_kind, mpmd_id, max_mubatches
                )
                if mubatch_idx is not None:
                    vp_id, mubatch_idx = divmod(mubatch_idx, mpmd)
                    stage_id = (
                        vp_id % num_repeats
                        if cur_kind is TaskType.FWD
                        else num_repeats - vp_id % num_repeats - 1
                    ) * mpmd + mpmd_id
                    mubatch_idx = (vp_id // num_repeats) * mpmd + mubatch_idx
                    schedule[mpmd_id][step_id] = Task(stage_id, mubatch_idx, cur_kind)

        return schedule
