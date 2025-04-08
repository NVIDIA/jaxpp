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
from unittest.mock import call, patch

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.lib import xla_bridge
from parameterized import parameterized_class

from jaxpp.autoslicing import ManualSlicingStrategy, StableSlicingStrategy


def create_simple_layer():
    """Creates a simple JAX matmul layer."""
    key = random.PRNGKey(0)
    W = random.normal(key, (2, 2))

    def layer(x):
        return jnp.dot(x, W)

    return layer


@parameterized_class(("strategy_name"), [("manual",), ("stable",)])
class AutoSlicingStrategyTest(unittest.TestCase):
    def setUp(self):
        """Clear the environment and define layers."""
        xla_bridge.get_backend.cache_clear()

        self.layers = [create_simple_layer() for _ in range(9)]

    def test_bypass_slicing_single_stage(self):
        """Test StableSlicingStrategy with a single stage using JAX layers."""
        if self.strategy_name == "manual":
            strategy = ManualSlicingStrategy(self.layers, stage_sizes=[9])
        else:
            strategy = StableSlicingStrategy(self.layers, num_stages=1)

        assert strategy.num_stages == 1
        assert strategy.layers == self.layers

        x = jnp.ones((2, 2))
        for layer, expected_layer in zip(strategy, self.layers, strict=True):
            jax.tree_util.tree_all(jnp.allclose(layer(x), expected_layer(x)))

    def test_slicing_with_multiple_stages(self):
        """Test slicing with multiple stages using JAX layers."""

        with patch("jaxpp.autoslicing.logger.info") as mock_log:
            if self.strategy_name == "manual":
                strategy = ManualSlicingStrategy(self.layers, stage_sizes=[3, 3, 3])
            else:
                strategy = StableSlicingStrategy(self.layers, num_stages=3)

            assert strategy.num_stages == 3
            assert len(strategy.layers) == 3
            assert len(strategy.layers[0]) == 3
            assert len(strategy.layers[1]) == 3
            assert len(strategy.layers[2]) == 3

            x = jnp.ones((2, 2))

            for layer in strategy:
                x = layer(x)
                assert x.shape == (2, 2)

        expected_calls = [
            call("[Auto Slicing] The model will be splitted in 3 pipeline stages."),
            call("\t- [Stage ID 01/03] - 03 layers/blocks."),
            call("\t- [Stage ID 02/03] - 03 layers/blocks."),
            call("\t- [Stage ID 03/03] - 03 layers/blocks."),
            call("[Auto Slicing] Entering Stage: 01"),
            call("[Auto Slicing] Entering Stage: 02"),
            call("[Auto Slicing] Entering Stage: 03"),
        ]
        assert mock_log.call_args_list == expected_calls

    def test_slicing_with_invalid_stage_sizes(self):
        """Test slicing with invalid argument."""

        if self.strategy_name == "manual":
            with pytest.raises(ValueError, match="Received a list of"):
                _ = ManualSlicingStrategy(self.layers, stage_sizes=[3, 3, 2])
        else:
            with pytest.raises(ValueError, match="Unsufficient number of layers "):
                _ = StableSlicingStrategy(self.layers, num_stages=11)


class StableSlicingStrategyTest(unittest.TestCase):
    def setUp(self):
        """Clear the environment."""
        xla_bridge.get_backend.cache_clear()

    def test_stable_strategy_edge_case_overflow_1(self):
        layers = [create_simple_layer() for _ in range(10)]
        strategy = StableSlicingStrategy(layers, 3)

        assert strategy.num_stages == 3
        assert len(strategy.layers) == 3
        assert len(strategy.layers[0]) == 4
        assert len(strategy.layers[1]) == 3
        assert len(strategy.layers[2]) == 3

    def test_stable_strategy_edge_case_overflow_2(self):
        layers = [create_simple_layer() for _ in range(11)]
        strategy = StableSlicingStrategy(layers, 3)

        assert strategy.num_stages == 3
        assert len(strategy.layers) == 3
        assert len(strategy.layers[0]) == 4
        assert len(strategy.layers[1]) == 4
        assert len(strategy.layers[2]) == 3

    def test_stable_strategy_edge_case_underflow_1(self):
        layers = [create_simple_layer() for _ in range(8)]
        strategy = StableSlicingStrategy(layers, 3)

        assert strategy.num_stages == 3
        assert len(strategy.layers) == 3
        assert len(strategy.layers[0]) == 3
        assert len(strategy.layers[1]) == 3
        assert len(strategy.layers[2]) == 2

    def test_stable_strategy_edge_case_underflow_2(self):
        layers = [create_simple_layer() for _ in range(7)]
        strategy = StableSlicingStrategy(layers, 3)

        assert strategy.num_stages == 3
        assert len(strategy.layers) == 3
        assert len(strategy.layers[0]) == 3
        assert len(strategy.layers[1]) == 2
        assert len(strategy.layers[2]) == 2


if __name__ == "__main__":
    unittest.main()
