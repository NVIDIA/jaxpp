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

import logging
from abc import ABCMeta, abstractmethod

import jaxpp
import jaxpp.api

logger = logging.getLogger(__name__)


class BaseSlicingStrategy(metaclass=ABCMeta):
    def __init__(self, layers: list, *args, **kwargs):
        self._layers, self._num_stages = self._split_layer_list(layers, *args, **kwargs)
        assert self._num_stages >= 1, f"Invalid number of stages: {self.num_stages}"

        if self.num_stages > 1:
            logger.info(
                f"[Auto Slicing] The model will be splitted in {self.num_stages} "
                "pipeline stages."
            )
            for stage_id, stage_list in enumerate(self.layers):
                logger.info(
                    f"\t- [Stage ID {stage_id + 1:02d}/{self.num_stages:02d}] - "
                    f"{len(stage_list):02d} layers/blocks."
                )

    def __iter__(self):
        if self.num_stages == 1:
            yield from self._layers

        else:
            logger.info("[Auto Slicing] Entering Stage: 01")
            yield from self.__sliced_iter__()

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def layers(self):
        return self._layers

    def __sliced_iter__(self):
        # Defines a layer generator that automatically inserts
        # `jaxpp.api.pipeline_enter_stage(...)` when the transition needs to be
        # executed.
        for stage_id, staged_layers in enumerate(self._layers):
            for stage_rank, layer in enumerate(staged_layers):
                if stage_rank == 0 and stage_id != 0:
                    yield self.wrap_layer_with_jaxpp_stage(layer, stage_id)
                else:
                    yield layer

    def wrap_layer_with_jaxpp_stage(self, layer, stage_id):
        def wrapped(x, *args, **kwargs):
            logger.info(f"[Auto Slicing] Entering Stage: {stage_id + 1:02d}")
            x = jaxpp.api.pipeline_enter_stage(x)
            return layer(x, *args, **kwargs)

        return wrapped

    @abstractmethod
    def _split_layer_list(self, *args, **kwargs) -> tuple[list, int]:
        raise NotImplementedError  # pragma: no cover


class StableSlicingStrategy(BaseSlicingStrategy):
    """This auto-slicing utility iterator aims to divide a list of layers/blocks,
    into `num_stages` equal parts."""

    def _split_layer_list(self, layers: list, num_stages: int) -> tuple[list, int]:
        if num_stages > len(layers):
            raise ValueError(
                "Unsufficient number of layers received. "
                f"Expected {num_stages}, received: {len(layers)}"
            )

        if num_stages > 1:
            # Split the layers into stages with equal number of layers
            k, m = divmod(len(layers), num_stages)

            return [
                layers[stage * k + min(stage, m) : (stage + 1) * k + min(stage + 1, m)]
                for stage in range(num_stages)
            ], num_stages

        return layers, num_stages


class ManualSlicingStrategy(BaseSlicingStrategy):
    """This auto-slicing utility iterator aims to divide a list of layers/blocks,
    into N stages as specified by `stage_sizes` which is a list of stage sizes."""

    def _split_layer_list(self, layers: list, stage_sizes: list) -> tuple[list, int]:
        if sum(stage_sizes) != len(layers):
            raise ValueError(
                f"Received a list of {len(layers)=}, but stages only mention, "
                f"{stage_sizes} stages for a total of {sum(stage_sizes)=}"
            )

        if len(stage_sizes) == 1:
            return layers, 1

        staged_layers = []
        idx = 0
        for stage_size in stage_sizes:
            stage_layers = layers[idx : idx + stage_size]
            idx += stage_size
            staged_layers.append(stage_layers)

        return staged_layers, len(stage_sizes)
