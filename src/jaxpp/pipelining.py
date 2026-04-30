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

import warnings
from contextlib import contextmanager
from typing import Any

import jax

from jaxpp.jax_primitives import pipeline_yield_p
from jaxpp.types import TaskType

_current_stage = None


@contextmanager
def yield_scope(enabled: bool = True):
    global _current_stage
    prev_stage = _current_stage
    _current_stage = 0 if enabled else None
    try:
        yield
    finally:
        _current_stage = prev_stage


def pipeline_enter_stage(anchor: Any, name: str | None = None) -> Any:
    warnings.warn(
        "pipeline_enter_stage is deprecated, use mark_stage_end instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mark_stage_end(anchor, name)


def mark_stage_end(anchor: Any, name: str | None = None) -> Any:
    """Marks the approximate end of a pipeline stage.

    JaxPP splits the computation at each stage boundary and dispatches
    the resulting stages to different MPMD devices according to the chosen
    pipeline schedule (e.g. ``Interleaved1F1B``).

    Must be called inside a function passed to :func:`treduce` /
    :func:`treduce_i`.

    Args:
        anchor: Arbitrary pytree used to define the approximate split point
        for stages. The arrays actually communicated between stages may be a
        subset of *anchor*, or even arrays that *anchor* depends on -- not
        necessarily *anchor* itself.
        name: Optional human-readable label for debugging and profiling.

    Returns:
        A pytree with the same structure as *anchor*.
    """
    global _current_stage
    if _current_stage is None:
        return anchor

    stage_id = _current_stage
    _current_stage += 1

    anchor_flat, tree = jax.tree_util.tree_flatten(anchor)
    return jax.tree_util.tree_unflatten(
        tree,
        pipeline_yield_p.bind(
            *anchor_flat, name=name, task_type=TaskType.FWD, stage_id=stage_id
        ),
    )
