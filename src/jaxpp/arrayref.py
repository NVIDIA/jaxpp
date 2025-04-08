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

import sys
from dataclasses import dataclass

import jax
import numpy as np

from jaxpp.mesh import RemoteMpmdMesh
from jaxpp.types import UID


@dataclass
class ArrayRefSharding:
    mesh_ids: set[int]
    sharding: jax.sharding.NamedSharding


# reference to a remote array
class ArrayRef:
    def __init__(
        self,
        sharding: ArrayRefSharding,
        uid: UID,
        mesh: RemoteMpmdMesh,
        aval,
    ):
        self.sharding = sharding
        self.uid = uid
        self.mesh = mesh
        self.aval = aval
        self.deleted = False

    def __str__(self):
        return f"ArrayRef(mpmd_idxs={self.mpmd_idxs}, uid={self.uid}, aval={self.aval})"

    def __del__(self):
        if sys.is_finalizing():
            return
        # TODO: use one single remote call per worker
        try:
            if not self.deleted:
                for mpmd_idx in self.mpmd_idxs:
                    self.mesh.delete(mpmd_idx, self.uid)
        except TypeError:
            # Developer note:
            # More efficient to ask for forgiveness (only during shutdown)
            # Than to perform a check at every single iteration
            pass

    @property
    def mpmd_idxs(self):
        return self.sharding.mesh_ids

    @property
    def _value(self):
        assert not self.deleted, "Unsafe use of deleted buffer"
        fut_res = self.mesh.get_tensor(next(iter(self.mpmd_idxs)), self.uid)
        res = self.mesh.blocking_tree([fut_res])[0]
        # FIXME: this is for MC runtime describing abstract value
        if res is None:
            return np.array(None, dtype=np.float32)
        return res

    def __format__(self, format_spec):
        # Simulates behavior of https://github.com/numpy/numpy/pull/9883
        if self.ndim == 0:
            return format(self._value[()], format_spec)

        return str(self._value)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._value.shape

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def is_replicated(self):
        return len(self.mpmd_idxs) > 1

    def __array__(self, dtype=None) -> np.ndarray:
        # Follow the pattern numpy expects: https://numpy.org/doc/stable/reference/generated/numpy.array.html
        return np.array(self._value).astype(dtype)

    def block_until_ready(self):
        # Fetch the value and does nothing. Forces the value to be ready.
        # TODO: Actually verify that the value is ready to be fetched instead
        # of actually fetching it.
        _ = self._value
