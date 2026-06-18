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


def _microbatch_round_sizes(n_microbatches: int, mpmd_dim: int) -> tuple[int, ...]:
    number_of_rounds = max(1, n_microbatches // mpmd_dim)
    base_size, extra = divmod(n_microbatches, number_of_rounds)
    return tuple(
        base_size + (round_idx < extra) for round_idx in range(number_of_rounds)
    )


def _local_stage_sequence(
    vp: int, round_sizes: Sequence[int], *, reverse: bool
) -> tuple[int, ...]:
    local_stages = range(vp - 1, -1, -1) if reverse else range(vp)
    return tuple(
        local_stage
        for round_size in round_sizes
        for local_stage in local_stages
        for _ in range(round_size)
    )


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


def _next_zero_bubble_task_type(task_type: TaskType) -> TaskType:
    if task_type is TaskType.BWD_I:
        return TaskType.BWD_W
    if task_type is TaskType.BWD_W:
        return TaskType.FWD
    if task_type is TaskType.FWD:
        return TaskType.BWD_I
    raise ValueError(f"Unexpected zero-bubble task type: {task_type}")


def _zero_bubble_tasks_for_rank(
    *, num_stages: int, mpmd_dim: int, mpmd_idx: int, n_mubatches: int
) -> list[Task]:
    vp = num_stages // mpmd_dim
    microbatch_rounds = []
    round_start = 0
    for round_size in _microbatch_round_sizes(n_mubatches, mpmd_dim):
        microbatch_rounds.append(range(round_start, round_start + round_size))
        round_start += round_size

    def stage_microbatches(*, reverse: bool) -> tuple[tuple[int, int], ...]:
        local_stages = range(vp - 1, -1, -1) if reverse else range(vp)
        return tuple(
            (local_stage * mpmd_dim + mpmd_idx, microbatch)
            for microbatch_round in microbatch_rounds
            for local_stage in local_stages
            for microbatch in microbatch_round
        )

    forward_tasks = stage_microbatches(reverse=False)
    backward_tasks = stage_microbatches(reverse=True)
    max_tasks = len(forward_tasks)
    assert max_tasks == len(backward_tasks) == vp * n_mubatches
    task_count = {TaskType.FWD: 0, TaskType.BWD_I: 0, TaskType.BWD_W: 0}

    def maybe_task(task_type: TaskType) -> Task | None:
        task_idx = task_count[task_type]
        if task_idx >= max_tasks:
            return None
        task_count[task_type] += 1
        if task_type is TaskType.FWD:
            stage_id, microbatch = forward_tasks[task_idx]
        elif task_type in {TaskType.BWD_I, TaskType.BWD_W}:
            stage_id, microbatch = backward_tasks[task_idx]
        else:
            raise ValueError(f"Unexpected zero-bubble task type: {task_type}")
        return Task.make(stage_id, microbatch, task_type)

    def next_available_task(task_type: TaskType) -> tuple[Task | None, TaskType]:
        for _ in range(3):
            task = maybe_task(task_type)
            next_task_type = _next_zero_bubble_task_type(task_type)
            if task is not None:
                return task, next_task_type
            task_type = next_task_type
        return None, task_type

    tasks = []
    for _ in range(num_stages - mpmd_idx):
        tasks.append(maybe_task(TaskType.FWD))

    task_type = TaskType.BWD_I
    for _ in range(2 * mpmd_idx):
        tasks.append(maybe_task(task_type))
        if task_type is TaskType.BWD_I:
            task_type = TaskType.FWD
        elif task_type is TaskType.FWD:
            task_type = TaskType.BWD_I
        else:
            raise ValueError(f"Unexpected zero-bubble warmup task type: {task_type}")

    pivot = num_stages + mpmd_dim
    steps = max_tasks * 3 + (mpmd_dim - 1)
    task_type = TaskType.BWD_I
    for _ in range(pivot - 1 + mpmd_idx, steps):
        task, task_type = next_available_task(task_type)
        if task is None:
            break
        tasks.append(task)

    return [task for task in tasks if task is not None]


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
        assert n_mubatches >= self.num_stages, (
            "Expect num of microbatches >= num of stages, but "
            f"{n_mubatches} microbatches and {self.num_stages} stages found"
        )
        return [
            _zero_bubble_tasks_for_rank(
                num_stages=self.num_stages,
                mpmd_dim=self.num_stages,
                mpmd_idx=stage_id,
                n_mubatches=n_mubatches,
            )
            for stage_id in range(self.num_stages)
        ]


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

    def microbatch_round_sizes(self, n_microbatches: int) -> tuple[int, ...]:
        return _microbatch_round_sizes(n_microbatches, self.mpmd_dim)

    def microbatches_per_round(self, n_microbatches: int):
        return self.microbatch_round_sizes(n_microbatches)[0]

    def _uncapped_rank_warmup_ops(self, mpmd_idx, round_sizes: Sequence[int]) -> int:
        # Warms up operations for last stage
        warmups_ops_last_stage = (self.vp - 1) * round_sizes[0]
        # Increment warmup operations by 2 for each hop away from the last stage
        multiply_factor = 2
        return warmups_ops_last_stage + multiply_factor * (
            (self.mpmd_dim - 1) - mpmd_idx
        )

    def get_rank_warmup_ops(self, mpmd_idx, n_microbatches: int) -> int:
        warmup_ops = self._uncapped_rank_warmup_ops(
            mpmd_idx, self.microbatch_round_sizes(n_microbatches)
        )
        # We cannot have more warmup operations than there are number of microbatches,
        # so cap it there
        return min(warmup_ops, n_microbatches * self.vp)

    def _tasks_for_rank(
        self, mpmd_idx: int, n_mubatches: int
    ) -> list[Task | FusedTask]:
        round_sizes = self.microbatch_round_sizes(n_mubatches)
        forward_local_stages = _local_stage_sequence(
            self.vp, round_sizes, reverse=False
        )
        backward_local_stages = _local_stage_sequence(
            self.vp, round_sizes, reverse=True
        )
        microbatch_ops = len(forward_local_stages)
        assert microbatch_ops == len(backward_local_stages)
        assert microbatch_ops == self.vp * n_mubatches
        warmup_ops = self.get_rank_warmup_ops(mpmd_idx, n_mubatches)
        fwd_bwd_ops = microbatch_ops - warmup_ops

        it = (
            ZBSequentialMicrobatchesIterator()
            if self.is_partial_bwd
            else SequentialMicrobatchesIterator()
        )

        def stage_id(local_stage: int) -> int:
            return (local_stage * self.mpmd_dim) + mpmd_idx

        tasks = []
        # Warmup
        for step in range(warmup_ops):
            tasks.append(it.fwd(stage_id(forward_local_stages[step])))
        # Steady state
        for step in range(warmup_ops, warmup_ops + fwd_bwd_ops):
            fwd_idx = stage_id(forward_local_stages[step])
            bwd_idx = stage_id(backward_local_stages[step - warmup_ops])
            fwd = it.fwd(fwd_idx)
            bwd = it.bwd(bwd_idx)

            if (not self.fuse_steady_state) and fwd_idx != self.num_stages - 1:
                tasks.extend([fwd, bwd])
            else:
                if isinstance(bwd, Task):
                    bwd = (bwd,)
                tasks.append(FusedTask((fwd, *bwd)))

        # Cooldown
        for step in range(microbatch_ops, microbatch_ops + warmup_ops):
            tasks.append(it.bwd(stage_id(backward_local_stages[step - warmup_ops])))
        return tasks

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        return [
            self._tasks_for_rank(mpmd_idx, n_mubatches)
            for mpmd_idx in range(self.mpmd_dim)
        ]


@dataclass(eq=True, frozen=True)
class Eager1F1B(Interleaved1F1B):
    mpmd_dim: int | None = None
    fuse_steady_state: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.mpmd_dim is None:
            object.__setattr__(self, "mpmd_dim", self.num_stages)
        super().__post_init__()
        if self.mpmd_dim != self.num_stages:
            raise ValueError(
                f"{Eager1F1B.__name__} requires mpmd_dim == num_stages, got "
                f"{self.mpmd_dim=} and {self.num_stages=}"
            )

    def microbatch_round_sizes(self, n_microbatches: int) -> tuple[int, ...]:
        return (n_microbatches,)


@dataclass(eq=True, frozen=True)
class Std1F1B(Interleaved1F1B):
    mpmd_dim: int | None = None
    fuse_steady_state: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.mpmd_dim is None:
            object.__setattr__(self, "mpmd_dim", self.num_stages)
        super().__post_init__()
        if self.mpmd_dim != self.num_stages:
            raise ValueError(
                f"{Std1F1B.__name__} requires mpmd_dim == num_stages, got "
                f"{self.mpmd_dim=} and {self.num_stages=}"
            )

    def microbatch_round_sizes(self, n_microbatches: int) -> tuple[int, ...]:
        return (n_microbatches,)

    def _uncapped_rank_warmup_ops(self, mpmd_idx, round_sizes: Sequence[int]) -> int:
        return self.num_stages - mpmd_idx

    def _tasks_for_rank(
        self, mpmd_idx: int, n_mubatches: int
    ) -> list[Task | FusedTask]:
        round_sizes = self.microbatch_round_sizes(n_mubatches)
        forward_local_stages = _local_stage_sequence(
            self.vp, round_sizes, reverse=False
        )
        backward_local_stages = _local_stage_sequence(
            self.vp, round_sizes, reverse=True
        )
        microbatch_ops = len(forward_local_stages)
        assert microbatch_ops == len(backward_local_stages)
        assert microbatch_ops == self.vp * n_mubatches
        warmup_ops = self.get_rank_warmup_ops(mpmd_idx, n_mubatches)
        fwd_bwd_ops = microbatch_ops - warmup_ops

        it = SequentialMicrobatchesIterator()

        def stage_id(local_stage: int) -> int:
            return (local_stage * self.mpmd_dim) + mpmd_idx

        tasks = [
            it.fwd(stage_id(forward_local_stages[step])) for step in range(warmup_ops)
        ]

        for step in range(warmup_ops, warmup_ops + fwd_bwd_ops):
            tasks.append(it.bwd(stage_id(backward_local_stages[step - warmup_ops])))
            tasks.append(it.fwd(stage_id(forward_local_stages[step])))

        for step in range(microbatch_ops, microbatch_ops + warmup_ops):
            tasks.append(it.bwd(stage_id(backward_local_stages[step - warmup_ops])))

        return tasks


class KimiK2(Interleaved1F1B):
    def _uncapped_rank_warmup_ops(self, mpmd_idx, round_sizes):
        return super()._uncapped_rank_warmup_ops(mpmd_idx, round_sizes) + 1


@dataclass(eq=True, frozen=True)
class InterleavedGPipe(InterleavedBaseSchedule):
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "vp", self.num_stages // self.mpmd_dim)

    def microbatch_round_sizes(self, n_microbatches: int) -> tuple[int, ...]:
        return _microbatch_round_sizes(n_microbatches, self.mpmd_dim)

    def _tasks_for_rank(self, mpmd_idx: int, n_mubatches: int) -> list[Task]:
        round_sizes = self.microbatch_round_sizes(n_mubatches)
        forward_local_stages = _local_stage_sequence(
            self.vp, round_sizes, reverse=False
        )
        backward_local_stages = _local_stage_sequence(
            self.vp, round_sizes, reverse=True
        )
        assert len(forward_local_stages) == len(backward_local_stages)
        assert len(forward_local_stages) == self.vp * n_mubatches

        it = SequentialMicrobatchesIterator()

        def stage_id(local_stage: int) -> int:
            return (local_stage * self.mpmd_dim) + mpmd_idx

        tasks = [it.fwd(stage_id(local_stage)) for local_stage in forward_local_stages]
        tasks.extend(
            it.bwd(stage_id(local_stage)) for local_stage in backward_local_stages
        )
        return tasks

    def tasks(self, n_mubatches: int) -> ScheduleTasks:
        return [
            self._tasks_for_rank(mpmd_idx, n_mubatches)
            for mpmd_idx in range(self.mpmd_dim)
        ]


@dataclass(eq=True, frozen=True)
class GPipe(InterleavedGPipe):
    mpmd_dim: int | None = None

    def __post_init__(self):
        if self.mpmd_dim is None:
            object.__setattr__(self, "mpmd_dim", self.num_stages)
        super().__post_init__()
        if self.mpmd_dim != self.num_stages:
            raise ValueError(
                f"{GPipe.__name__} requires mpmd_dim == num_stages, got "
                f"{self.mpmd_dim=} and {self.num_stages=}"
            )

    def microbatch_round_sizes(self, n_microbatches: int) -> tuple[int, ...]:
        return (n_microbatches,)




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
