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

import time
import unittest
from collections import OrderedDict
from unittest.mock import patch

import pytest
from parameterized import parameterized, parameterized_class

from jaxpp import utils


class TestUnzipMulti(unittest.TestCase):
    """
    Tests for the `unzip_multi` function.
    """

    def test_empty_list(self):
        """Test that `unzip_multi` returns empty lists when given an empty list."""
        result = utils.unzip_multi([])
        assert result == [[], []]

    def test_arity_3(self):
        """Test that `unzip_multi` correctly unzips lists of triples with arity=3."""
        input_data = [(1, 2, 3), (4, 5, 6)]
        result = utils.unzip_multi(input_data, arity=3)
        assert result == [[1, 4], [2, 5], [3, 6]]

    def test_mismatched_arity(self):
        """Test that `unzip_multi` raises an assertion error for mismatched arity."""
        with pytest.raises(AssertionError):
            utils.unzip_multi([(1, 2, 3), (4, 5)], arity=3)


class TestGroupby(unittest.TestCase):
    """
    Tests for the `groupby` function.
    """

    def test_empty_iterable(self):
        """Test `groupby` with an empty iterable should return an empty dict."""
        result = utils.groupby([])
        assert result == OrderedDict()

    def test_grouping(self):
        """Test `groupby` correctly groups elements by their key."""
        input_data = [("a", 1), ("b", 2), ("a", 3)]
        expected = OrderedDict([("a", [1, 3]), ("b", [2])])
        result = utils.groupby(input_data)
        assert result == expected

    def test_single_group(self):
        """Test `groupby` with elements that all have the same key."""
        input_data = [("a", 1), ("a", 2), ("a", 3)]
        expected = OrderedDict([("a", [1, 2, 3])])
        result = utils.groupby(input_data)
        assert result == expected


class TestPartition(unittest.TestCase):
    """
    Tests for the `partition` function.
    """

    def test_empty_iterable(self):
        """Test `partition` with an empty iterable should return two empty lists."""
        result = utils.partition(lambda x: x > 0, [])
        assert result == ([], [])

    def test_partitioning(self):
        """Test `partition` correctly partitions elements based on the predicate."""
        input_data = [1, -2, 3, -4, 5]
        result = utils.partition(lambda x: x > 0, input_data)
        assert result == ([1, 3, 5], [-2, -4])


class TestRichDict(unittest.TestCase):
    """
    Tests for the `RichDict` class.
    """

    def setUp(self):
        """Set up a `RichDict` instance for testing."""
        self.rd = utils.RichDict()

    def test_get_or_else_key_exists(self):
        """Test `get_or_else` returns the existing value if the key exists."""
        self.rd["a"] = 1
        result = self.rd.get_or_else("a", lambda: 2)
        assert result == 1

    def test_get_or_else_key_missing(self):
        """Test `get_or_else` returns the default value if the key is missing."""
        result = self.rd.get_or_else("a", lambda: 2)
        assert result == 2

    def test_get_or_else_update_key_exists(self):
        """Test `get_or_else_update` returns the existing value if the key exists."""
        self.rd["a"] = 1
        result = self.rd.get_or_else_update("a", lambda: 2)
        assert result == 1

    def test_get_or_else_update_key_missing(self):
        """Test `get_or_else_update` updates and returns the new value if the key is
        missing."""
        result = self.rd.get_or_else_update("a", lambda: 2)
        assert result == 2
        assert self.rd["a"] == 2

    def test_set_or_raise_if_present_key_absent(self):
        """Test `set_or_raise_if_present` sets the value if the key is absent."""
        self.rd.set_or_raise_if_present("a", 1)
        assert self.rd["a"] == 1

    def test_set_or_raise_if_present_key_present(self):
        """Test `set_or_raise_if_present` raises a ValueError if the key is present."""
        self.rd["a"] = 1
        with pytest.raises(KeyError, match="already present with value"):
            self.rd.set_or_raise_if_present("a", 2)


@parameterized_class(
    [
        {"message": "hello_world"},
        {"message": None},
    ]
)
class TestLogElapsedTime(unittest.TestCase):
    """
    Tests for the `log_elapsed_time` context manager.
    """

    message = None

    @patch("jaxpp.utils.logger.info")
    def test_elapsed_time_logging_default_values(self, mock_info):
        """Test `log_elapsed_time` logs the elapsed time in seconds."""
        with utils.log_elapsed_time("test_event"):
            time.sleep(0.001)  # Sleep for 1ms

        assert mock_info.call_count == 2
        args = mock_info.call_args[0]
        assert "test_event took" in args[0]
        assert "s" in args[0]

    @parameterized.expand(
        [
            ("seconds", "s", 1e-3, "s"),
            ("milliseconds", "ms", 1e-3, "ms"),
            ("microseconds", "us", 1e-6, "us"),
            ("nanoseconds", "ns", 1e-9, "ns"),
        ]
    )
    @patch("jaxpp.utils.logger.info")
    def test_elapsed_time_logging_ms(
        self, name, unit, sleep_time, expected_unit, mock_info
    ):
        """
        Test `log_elapsed_time` logs the elapsed time correctly for each unit.

        :param name: A descriptive name for the test.
        :param unit: The time unit to be tested.
        :param sleep_time: The amount of time to sleep inside the context manager.
        :param expected_unit: The expected unit in the log message.
        :param mock_info: The mock for the logger.info call.
        """
        with utils.log_elapsed_time("test_event", msg=self.message, unit=unit):
            time.sleep(sleep_time)

        assert mock_info.call_count == 2
        args = mock_info.call_args[0]
        assert "test_event took" in args[0]
        assert expected_unit in args[0]

    def test_invalid_unit(self):
        """Test `log_elapsed_time` raises ValueError for an invalid time unit."""
        with pytest.raises(ValueError, match="Unknown Unit"):
            with utils.log_elapsed_time(
                "test_event", msg=self.message, unit="invalid_unit"
            ):
                pass


if __name__ == "__main__":
    unittest.main()
