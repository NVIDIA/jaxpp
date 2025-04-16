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

import unittest

import jaxpp
import jaxpp.types


class TestScalarUid(unittest.TestCase):
    def test_scalar_uid(self):
        uid1 = jaxpp.types.fresh_scalar_uid()
        uid2 = jaxpp.types.fresh_scalar_uid()
        assert uid1 + 1 == uid2


class TestUniqueSortedSequence(unittest.TestCase):
    def test_create_unique_sorted_sequence(self):
        sequence = jaxpp.types.UniqueSortedSequence.create([3, 1, 2, 2])
        assert sequence == (1, 2, 3)


class TestUniqueGlobalDeviceIds(unittest.TestCase):
    def test_strict_create(self):
        ids = [jaxpp.types.GlobalDeviceId(idx) for idx in range(1, 4)]
        ugd = jaxpp.types.UniqueGlobalDeviceIds.strict_create(ids)
        assert ugd == (1, 2, 3)
        assert ugd.primary == 1

    def test_ranks(self):
        ids = [jaxpp.types.GlobalDeviceId(idx) for idx in range(1, 3)]
        ugd = jaxpp.types.UniqueGlobalDeviceIds.create(ids)
        assert ugd.ranks == [(1, jaxpp.types.Rank(0)), (2, jaxpp.types.Rank(1))]

    def test_rank_of(self):
        ids = [jaxpp.types.GlobalDeviceId(1), jaxpp.types.GlobalDeviceId(2)]
        ugd = jaxpp.types.UniqueGlobalDeviceIds.create(ids)
        assert ugd.rank_of(jaxpp.types.GlobalDeviceId(1)) == jaxpp.types.Rank(0)


if __name__ == "__main__":
    unittest.main()
