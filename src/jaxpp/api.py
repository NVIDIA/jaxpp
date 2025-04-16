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
from jaxpp.arrayref import ArrayRef
from jaxpp.core import pipelined
from jaxpp.jax_primitives import add_multi_p
from jaxpp.loop_output import LoopOutput
from jaxpp.mesh import MpmdMesh, RemoteMpmdMesh
from jaxpp.pipelining import pipeline_enter_stage
from jaxpp.replication import run_replicated_dced
from jaxpp.schedules import BaseSchedule, Eager1F1B, Interleaved1F1B, Std1F1B
from jaxpp.training import accumulate_grads


def cross_mpmd_all_reduce(*args):
    first = args[0]
    if not all(first.dtype == arg.dtype for arg in args):
        raise AssertionError(
            f"All arguments must have the same dtype, got {[a.dtype for a in args]}"
        )
    return add_multi_p.bind(*args)
