# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import numpy as np
from jax.sharding import Mesh

from jaxpp.mesh import MpmdMesh


class _FakeClient:
    platform = "gpu"

    def process_index(self):
        return 0


class _FakeMemory:
    def __init__(self, kind: str, device):
        self.kind = kind
        self.device = device

    def __eq__(self, other):
        if isinstance(other, _FakeMemory):
            return self.kind == other.kind
        return self.kind == other

    def __hash__(self):
        return hash(self.kind)

    def __str__(self):
        return self.kind

    def __repr__(self):
        return f"_FakeMemory({self.kind})"


class _FakeDevice:
    platform = "gpu"
    device_kind = "fake gpu"
    core_count = 1
    client = _FakeClient()

    def __init__(self, device_id: int):
        self.id = device_id
        self.process_index = 0
        self._memories = {
            kind: _FakeMemory(kind, self) for kind in ("device", "pinned_host")
        }

    def addressable_memories(self):
        return list(self._memories.values())

    def memory(self, kind: str):
        return self._memories[kind]

    def default_memory(self):
        return self.memory("device")

    def __repr__(self):
        return f"_FakeDevice({self.id})"


def make_mesh(
    shape: tuple[int, ...], axis_names: tuple[str, ...], mpmd_axis: str
) -> MpmdMesh:
    required_devices = int(np.prod(shape))
    devices = [_FakeDevice(i) for i in range(required_devices)]
    devices = np.asarray(devices, dtype=object).reshape(shape)
    return MpmdMesh(Mesh(devices, axis_names), mpmd_axis)
