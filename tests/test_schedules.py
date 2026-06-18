# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import pytest
from parameterized import parameterized_class

import jaxpp
import jaxpp.schedules
from jaxpp.schedules import DualPipeV, FusedTask


def _unpack_fused_tasks(tasks):
    return [
        task
        for maybe_fused_task in tasks
        for task in (
            maybe_fused_task
            if isinstance(maybe_fused_task, FusedTask)
            else (maybe_fused_task,)
        )
    ]


def _stage_mubatches(tasks, stage_id, task_type):
    return [
        task.mubatch_idx
        for task in _unpack_fused_tasks(tasks)
        if task.stage_id == stage_id and task.fwd_or_bwd is task_type
    ]


@parameterized_class(
    [
        {"ScheduleCls": jaxpp.schedules.Std1F1B},
        {"ScheduleCls": jaxpp.schedules.Eager1F1B},
        {"ScheduleCls": jaxpp.schedules.Interleaved1F1B, "mpmd_dim": 1},
        {"ScheduleCls": jaxpp.schedules.ZeroBubble},
    ]
)
class TestSchedules(unittest.TestCase):
    """
    Tests for the `log_elapsed_time` context manager.
    """

    ScheduleCls: type | None = None
    mpmd_dim: int | None = None
    num_stages: int = 2
    n_mubatches: int = 4

    def get_schedule(
        self, num_stages: int, mpmd_dim: int | None = None
    ) -> jaxpp.schedules.BaseSchedule:
        assert self.ScheduleCls is not None
        assert issubclass(self.ScheduleCls, jaxpp.schedules.BaseSchedule)

        if mpmd_dim is not None:
            return self.ScheduleCls(num_stages=num_stages, mpmd_dim=mpmd_dim)

        return self.ScheduleCls(num_stages=num_stages)

    def test_correct_schedule(self):
        schedule = self.get_schedule(num_stages=self.num_stages, mpmd_dim=self.mpmd_dim)
        result = schedule.tasks(self.n_mubatches)

        assert isinstance(result, list)

        if self.mpmd_dim is not None:
            assert len(result) == self.mpmd_dim
        else:
            assert len(result) == self.num_stages

        assert all(isinstance(step, list) for step in result)

    def test_negative_num_stages(self):
        with pytest.raises(ValueError, match="The argument `num_stages` must be `>= 0"):
            _ = self.get_schedule(num_stages=-1, mpmd_dim=self.mpmd_dim)

    def test_negative_mpmd_dim(self):
        if not issubclass(self.ScheduleCls, jaxpp.schedules.InterleavedBaseSchedule):
            self.skipTest("Doesn't use `mpmd_dim` argument")

        with pytest.raises(ValueError, match="The argument `mpmd_dim` must be `>= 0"):
            _ = self.get_schedule(num_stages=self.num_stages, mpmd_dim=-1)

    def test_mismatch_num_stages_and_mpmd_dim(self):
        if not issubclass(self.ScheduleCls, jaxpp.schedules.InterleavedBaseSchedule):
            self.skipTest("Doesn't use `mpmd_dim` argument")

        with pytest.raises(ValueError, match="can not be evenly divided by"):
            _ = self.get_schedule(num_stages=2, mpmd_dim=3)


def test_validate_dualpipev_num_stages_less():
    with pytest.raises(ValueError) as exc:
        DualPipeV(num_stages=4, mpmd_dim=4)

    assert (
        f"{DualPipeV.__name__} only supports 2 * mpmd_dim stages, self.num_stages=4"
        f" requested with self.mpmd_dim=4"
    ) in str(exc.value)


def test_validate_dualpipev_num_stages_more():
    with pytest.raises(ValueError) as exc:
        DualPipeV(num_stages=12, mpmd_dim=4)

    assert (
        f"{DualPipeV.__name__} only supports 2 * mpmd_dim stages, self.num_stages=12"
        f" requested with self.mpmd_dim=4"
    ) in str(exc.value)


def test_validate_dualpipev_n_mubatches():
    with pytest.raises(ValueError) as exc:
        DualPipeV(num_stages=6, mpmd_dim=3).tasks(5)

    assert f"{DualPipeV.__name__} requires n_mubatches=5 >= self.num_stages=6" in str(
        exc.value
    )


def test_eager_1f1b_is_stage_per_rank_interleaved_schedule():
    schedule = jaxpp.schedules.Eager1F1B(num_stages=3)

    assert isinstance(schedule, jaxpp.schedules.Interleaved1F1B)
    assert schedule.mpmd_dim == schedule.num_stages
    assert [schedule.get_mpmd_idx(stage_id) for stage_id in range(3)] == [0, 1, 2]


def test_eager_1f1b_accepts_non_divisible_microbatches_and_keeps_fusion():
    schedule = jaxpp.schedules.Eager1F1B(num_stages=3)
    tasks = schedule.tasks(7)

    assert len(tasks) == 3
    for rank_tasks in tasks[:-1]:
        assert [
            task.mubatch_idx
            for task in rank_tasks
            if task.fwd_or_bwd is jaxpp.schedules.TaskType.FWD
        ] == list(range(7))
        assert [
            task.mubatch_idx
            for task in rank_tasks
            if task.fwd_or_bwd is jaxpp.schedules.TaskType.BWD
        ] == list(range(7))

    assert all(isinstance(task, FusedTask) for task in tasks[-1])
    assert [
        tuple(str(task) for task in fused_task)
        for fused_task in tasks[-1]
    ] == [(f"fwd_2__{i}", f"bwd_2__{i}") for i in range(7)]


def test_eager_1f1b_rejects_non_stage_per_rank_mpmd_dim():
    _ = jaxpp.schedules.Eager1F1B(num_stages=3, mpmd_dim=3)

    with pytest.raises(ValueError, match="requires mpmd_dim == num_stages"):
        _ = jaxpp.schedules.Eager1F1B(num_stages=3, mpmd_dim=1)


def test_std_1f1b_is_stage_per_rank_interleaved_schedule():
    schedule = jaxpp.schedules.Std1F1B(num_stages=3)

    assert isinstance(schedule, jaxpp.schedules.Interleaved1F1B)
    assert schedule.mpmd_dim == schedule.num_stages
    assert [schedule.get_mpmd_idx(stage_id) for stage_id in range(3)] == [0, 1, 2]


def test_std_1f1b_keeps_backward_before_forward_steady_state_order():
    schedule = jaxpp.schedules.Std1F1B(num_stages=3)

    assert [[str(task) for task in rank_tasks] for rank_tasks in schedule.tasks(5)] == [
        [
            "fwd_0__0",
            "fwd_0__1",
            "fwd_0__2",
            "bwd_0__0",
            "fwd_0__3",
            "bwd_0__1",
            "fwd_0__4",
            "bwd_0__2",
            "bwd_0__3",
            "bwd_0__4",
        ],
        [
            "fwd_1__0",
            "fwd_1__1",
            "bwd_1__0",
            "fwd_1__2",
            "bwd_1__1",
            "fwd_1__3",
            "bwd_1__2",
            "fwd_1__4",
            "bwd_1__3",
            "bwd_1__4",
        ],
        [
            "fwd_2__0",
            "bwd_2__0",
            "fwd_2__1",
            "bwd_2__1",
            "fwd_2__2",
            "bwd_2__2",
            "fwd_2__3",
            "bwd_2__3",
            "fwd_2__4",
            "bwd_2__4",
        ],
    ]


def test_std_1f1b_rejects_non_stage_per_rank_mpmd_dim():
    _ = jaxpp.schedules.Std1F1B(num_stages=3, mpmd_dim=3)

    with pytest.raises(ValueError, match="requires mpmd_dim == num_stages"):
        _ = jaxpp.schedules.Std1F1B(num_stages=3, mpmd_dim=1)


def test_interleaved_1f1b_keeps_divisible_schedule_order():
    schedule = jaxpp.schedules.Interleaved1F1B(num_stages=4, mpmd_dim=2)

    assert schedule.microbatch_round_sizes(4) == (2, 2)
    assert [[str(task) for task in rank_tasks] for rank_tasks in schedule.tasks(4)] == [
        [
            "fwd_0__0",
            "fwd_0__1",
            "fwd_2__0",
            "fwd_2__1",
            "fwd_0__2",
            "bwd_2__0",
            "fwd_0__3",
            "bwd_2__1",
            "fwd_2__2",
            "bwd_0__0",
            "fwd_2__3",
            "bwd_0__1",
            "bwd_2__2",
            "bwd_2__3",
            "bwd_0__2",
            "bwd_0__3",
        ],
        [
            "fwd_1__0",
            "fwd_1__1",
            "FusedTask(fwd_3__0, bwd_3__0)",
            "FusedTask(fwd_3__1, bwd_3__1)",
            "fwd_1__2",
            "bwd_1__0",
            "fwd_1__3",
            "bwd_1__1",
            "FusedTask(fwd_3__2, bwd_3__2)",
            "FusedTask(fwd_3__3, bwd_3__3)",
            "bwd_1__2",
            "bwd_1__3",
        ],
    ]


def test_interleaved_1f1b_accepts_non_divisible_microbatches():
    schedule = jaxpp.schedules.Interleaved1F1B(num_stages=4, mpmd_dim=2)
    tasks = schedule.tasks(5)

    assert schedule.microbatch_round_sizes(5) == (3, 2)
    for stage_id in range(4):
        rank_tasks = tasks[schedule.get_mpmd_idx(stage_id)]
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.FWD
        ) == list(range(5))
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.BWD
        ) == list(range(5))

    assert sum(isinstance(task, FusedTask) for task in tasks[-1]) == 5


def test_kimik2_accepts_non_divisible_microbatches():
    schedule = jaxpp.schedules.KimiK2(
        num_stages=8,
        mpmd_dim=4,
        fuse_steady_state=True,
    )
    tasks = schedule.tasks(9)

    assert schedule.microbatch_round_sizes(9) == (5, 4)
    for stage_id in range(8):
        rank_tasks = tasks[schedule.get_mpmd_idx(stage_id)]
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.FWD
        ) == list(range(9))
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.BWD
        ) == list(range(9))


def test_gpipe_is_stage_per_rank_interleaved_gpipe_schedule():
    schedule = jaxpp.schedules.GPipe(num_stages=3)

    assert isinstance(schedule, jaxpp.schedules.InterleavedGPipe)
    assert schedule.mpmd_dim == schedule.num_stages
    assert [schedule.get_mpmd_idx(stage_id) for stage_id in range(3)] == [0, 1, 2]


def test_gpipe_keeps_fwd_then_bwd_order():
    schedule = jaxpp.schedules.GPipe(num_stages=2)

    assert [[str(task) for task in rank_tasks] for rank_tasks in schedule.tasks(3)] == [
        [
            "fwd_0__0",
            "fwd_0__1",
            "fwd_0__2",
            "bwd_0__0",
            "bwd_0__1",
            "bwd_0__2",
        ],
        [
            "fwd_1__0",
            "fwd_1__1",
            "fwd_1__2",
            "bwd_1__0",
            "bwd_1__1",
            "bwd_1__2",
        ],
    ]


def test_gpipe_rejects_non_stage_per_rank_mpmd_dim():
    _ = jaxpp.schedules.GPipe(num_stages=3, mpmd_dim=3)

    with pytest.raises(ValueError, match="requires mpmd_dim == num_stages"):
        _ = jaxpp.schedules.GPipe(num_stages=3, mpmd_dim=1)


def test_interleaved_gpipe_accepts_non_divisible_microbatches():
    schedule = jaxpp.schedules.InterleavedGPipe(num_stages=4, mpmd_dim=2)
    tasks = schedule.tasks(5)

    assert schedule.microbatch_round_sizes(5) == (3, 2)
    assert [[str(task) for task in rank_tasks] for rank_tasks in tasks] == [
        [
            "fwd_0__0",
            "fwd_0__1",
            "fwd_0__2",
            "fwd_2__0",
            "fwd_2__1",
            "fwd_2__2",
            "fwd_0__3",
            "fwd_0__4",
            "fwd_2__3",
            "fwd_2__4",
            "bwd_2__0",
            "bwd_2__1",
            "bwd_2__2",
            "bwd_0__0",
            "bwd_0__1",
            "bwd_0__2",
            "bwd_2__3",
            "bwd_2__4",
            "bwd_0__3",
            "bwd_0__4",
        ],
        [
            "fwd_1__0",
            "fwd_1__1",
            "fwd_1__2",
            "fwd_3__0",
            "fwd_3__1",
            "fwd_3__2",
            "fwd_1__3",
            "fwd_1__4",
            "fwd_3__3",
            "fwd_3__4",
            "bwd_3__0",
            "bwd_3__1",
            "bwd_3__2",
            "bwd_1__0",
            "bwd_1__1",
            "bwd_1__2",
            "bwd_3__3",
            "bwd_3__4",
            "bwd_1__3",
            "bwd_1__4",
        ],
    ]

    for stage_id in range(4):
        rank_tasks = tasks[schedule.get_mpmd_idx(stage_id)]
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.FWD
        ) == list(range(5))
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.BWD
        ) == list(range(5))


def test_zero_bubble_keeps_schedule_order():
    schedule = jaxpp.schedules.ZeroBubble(num_stages=3)

    assert [[str(task) for task in rank_tasks] for rank_tasks in schedule.tasks(3)] == [
        [
            "fwd_0__0",
            "fwd_0__1",
            "fwd_0__2",
            "bwdA_0__0",
            "bwdW_0__0",
            "bwdA_0__1",
            "bwdW_0__1",
            "bwdA_0__2",
            "bwdW_0__2",
        ],
        [
            "fwd_1__0",
            "fwd_1__1",
            "bwdA_1__0",
            "fwd_1__2",
            "bwdA_1__1",
            "bwdW_1__0",
            "bwdA_1__2",
            "bwdW_1__1",
            "bwdW_1__2",
        ],
        [
            "fwd_2__0",
            "bwdA_2__0",
            "fwd_2__1",
            "bwdA_2__1",
            "fwd_2__2",
            "bwdA_2__2",
            "bwdW_2__0",
            "bwdW_2__1",
            "bwdW_2__2",
        ],
    ]


def test_interleaved_zero_bubble_keeps_schedule_order():
    schedule = jaxpp.schedules.InterleavedZeroBubble(num_stages=4, mpmd_dim=2)

    assert schedule.is_partial_bwd is True
    assert [[str(task) for task in rank_tasks] for rank_tasks in schedule.tasks(2)] == [
        [
            "fwd_0__0",
            "fwd_0__1",
            "fwd_2__0",
            "fwd_2__1",
            "bwdA_2__0",
            "bwdW_2__0",
            "bwdA_2__1",
            "bwdW_2__1",
            "bwdA_0__0",
            "bwdW_0__0",
            "bwdA_0__1",
            "bwdW_0__1",
        ],
        [
            "fwd_1__0",
            "fwd_1__1",
            "fwd_3__0",
            "bwdA_3__0",
            "fwd_3__1",
            "bwdA_3__1",
            "bwdW_3__0",
            "bwdA_1__0",
            "bwdW_3__1",
            "bwdA_1__1",
            "bwdW_1__0",
            "bwdW_1__1",
        ],
    ]


def test_interleaved_zero_bubble_accepts_non_divisible_microbatches():
    schedule = jaxpp.schedules.InterleavedZeroBubble(num_stages=4, mpmd_dim=2)
    tasks = schedule.tasks(5)

    for stage_id in range(4):
        rank_tasks = tasks[schedule.get_mpmd_idx(stage_id)]
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.FWD
        ) == list(range(5))
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.BWD_I
        ) == list(range(5))
        assert _stage_mubatches(
            rank_tasks, stage_id, jaxpp.schedules.TaskType.BWD_W
        ) == list(range(5))


if __name__ == "__main__":
    unittest.main()
