# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class JaxppUnitTest(unittest.TestCase):
    def setUp(self):
        self.verbose = False

    def assertEqual(self, ref, out):
        is_same = (ref == out).all()
        if not is_same:
            self.printDiff(ref, out)
            raise AssertionError

    def printDiff(self, ref, out):
        if self.verbose:
            print("--- ref ---")
            print(ref)
            print("--- out ---")
            print(out)
