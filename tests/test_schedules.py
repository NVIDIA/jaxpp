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
from jaxpp.schedules import DualPipeV


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


if __name__ == "__main__":
    unittest.main()
