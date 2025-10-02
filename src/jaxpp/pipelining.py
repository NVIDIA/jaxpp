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


# TODO(from,to)
def pipeline_enter_stage(anchor: Any, name: str | None = None) -> Any:
    global _current_stage
    if _current_stage is None:
        return anchor

    from_stage_id = _current_stage
    _current_stage += 1

    anchor_flat, tree = jax.tree_util.tree_flatten(anchor)
    return jax.tree_util.tree_unflatten(
        tree,
        pipeline_yield_p.bind(
            *anchor_flat,
            name=f"stage_{name}_{_current_stage:03d}",
            task_type=TaskType.FWD,
            from_stage_id=from_stage_id,
            to_stage_id=_current_stage,
        ),
    )
