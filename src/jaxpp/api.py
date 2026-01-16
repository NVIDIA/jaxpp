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

# Deactivate Removal of Unused Imports
# ruff: noqa: F401

from jaxpp import __version__
from jaxpp.array import (
    MpmdArray,
    _to_global_jax_array,
    mpmd_to_spmd_reshard,
    spmd_to_mpmd_reshard,
)
from jaxpp.core import mpmd_jit_by_yield, mpmd_jit_with_loop
from jaxpp.jax_primitives import add_multi_p, gather_multi_p
from jaxpp.mesh import MpmdMesh
from jaxpp.pipelining import pipeline_enter_stage
from jaxpp.schedules import (
    BaseSchedule,
    DualPipeV,
    Eager1F1B,
    Interleaved1F1B,
    Std1F1B,
    ZeroBubble,
)
from jaxpp.training import Add, Concat, Max, treduce, treduce_i


def cross_mpmd_all_reduce(*args):
    first = args[0]
    if not all(first.dtype == arg.dtype for arg in args):
        raise AssertionError(
            f"All arguments must have the same dtype, got {[a.dtype for a in args]}"
        )
    return add_multi_p.bind(*args)


def cross_mpmd_stack(arrays, axis: int = 0):
    import jax

    expanded = [jax.numpy.expand_dims(a, axis=axis) for a in arrays]
    return gather_multi_p.bind(*expanded, axis=axis)
