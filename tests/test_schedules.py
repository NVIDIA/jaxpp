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


@parameterized_class(
    [
        {"ScheduleCls": jaxpp.schedules.Std1F1B},
        {"ScheduleCls": jaxpp.schedules.Eager1F1B},
        {"ScheduleCls": jaxpp.schedules.GPipe},
        {"ScheduleCls": jaxpp.schedules.Interleaved1F1B, "mpmd_dim": 1},
        {"ScheduleCls": jaxpp.schedules.InterleavedGPipe, "mpmd_dim": 1},
    ]
)
class TestSchedules(unittest.TestCase):
    """
    Tests for the `log_elapsed_time` context manager.
    """

    ScheduleCls: type | None = None
    mpmd_dim: int | None = None
    num_stages: int = 2
    n_mubatches: int = 3

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
        if not issubclass(self.ScheduleCls, jaxpp.schedules.Base_MPMD_DIM_Schedule):
            self.skipTest("Doesn't use `mpmd_dim` argument")

        with pytest.raises(ValueError, match="The argument `mpmd_dim` must be `>= 0"):
            _ = self.get_schedule(num_stages=self.num_stages, mpmd_dim=-1)

    def test_mismatch_num_stages_and_mpmd_dim(self):
        if not issubclass(self.ScheduleCls, jaxpp.schedules.Base_MPMD_DIM_Schedule):
            self.skipTest("Doesn't use `mpmd_dim` argument")

        with pytest.raises(ValueError, match="can not be evenly divided by"):
            _ = self.get_schedule(num_stages=2, mpmd_dim=3)


if __name__ == "__main__":
    unittest.main()
