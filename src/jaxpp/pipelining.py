# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import contextmanager
from typing import Any

import jax

from jaxpp.jax_primitives import pipeline_yield_p
from jaxpp.types import TaskType


class PipelineStageContext:
    _is_tracing = False
    _current_stage: int = 0

    def __init__(self) -> None:
        raise RuntimeError("This class can not be instanciated.")

    @classmethod
    def next_stage(cls) -> tuple[int, int]:
        from_stage = cls._current_stage
        cls._current_stage += 1
        return from_stage, cls._current_stage

    @classmethod
    def reset(cls):
        cls._current_stage = 0

    @classmethod
    @contextmanager
    def tracing_scope(cls):
        cls._is_tracing = True
        _current_stage = cls._current_stage
        cls.reset()

        yield

        cls._is_tracing = False
        cls._current_stage = _current_stage

    @classmethod
    def mark_stage_switch(cls, anchor: Any) -> Any:
        if not cls._is_tracing:
            return anchor

        from_stage_id, current_stage = cls.next_stage()

        anchor_flat, tree = jax.tree_util.tree_flatten(anchor)
        return jax.tree_util.tree_unflatten(
            tree,
            pipeline_yield_p.bind(
                *anchor_flat,
                name=f"stage_{current_stage:03d}",
                task_type=TaskType.FWD,
                from_stage_id=from_stage_id,
                to_stage_id=current_stage,
            ),
        )


def pipeline_enter_stage(
    anchor: Any, name: str | None = None, stage_id: int | None = None
) -> Any:
    return PipelineStageContext.mark_stage_switch(anchor)
