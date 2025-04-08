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
from jaxpp.autoslicing import ManualSlicingStrategy, StableSlicingStrategy
from jaxpp.compilation import pjit_to_serializeable_mesh_computation
from jaxpp.core import pipelined
from jaxpp.loop_output import LoopOutput
from jaxpp.mesh import RemoteMpmdMesh
from jaxpp.pipelining import pipeline_enter_stage
from jaxpp.replication import run_replicated_dced
from jaxpp.schedules import (
    BaseSchedule,
    Eager1F1B,
    GPipe,
    Interleaved1F1B,
    InterleavedGPipe,
    InterleavedZeroBubble,
    Std1F1B,
    ZeroBubble,
)
from jaxpp.training import accumulate_grads
