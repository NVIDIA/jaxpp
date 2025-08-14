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

from jaxpp.mesh import MpmdMesh, RemoteMpmdMesh
from jaxpp.types import UID


@dataclass
class ArrayRefSharding:
    mesh_ids: set[int]
    sharding: jax.sharding.NamedSharding


def mpmd_sharding(
    mpmd_mesh: MpmdMesh, mesh_ids: set[int], sharding: jax.sharding.NamedSharding
):
    jax_mesh = mpmd_mesh.as_mpmd_mesh.mpmd_submesh(sorted(mesh_ids)).jax_mesh
    return jax.sharding.NamedSharding(
        jax_mesh, sharding.spec, memory_kind=sharding.memory_kind
    )


# reference to a remote array
class ArrayRef:
    def __init__(
        self,
        sharding: ArrayRefSharding,
        uid: UID,
        mesh: MpmdMesh | RemoteMpmdMesh,
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

        # We need to delete the array on the remote mesh only under
        # the remote single-controller runtime. We still have to remove
        # its reference in the store.
        if isinstance(self.mesh, MpmdMesh):
            if self.mesh.my_mpmd_axis_index in self.mpmd_idxs and not self.deleted:
                self.mesh.store.pop_array(self.uid)
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
        if isinstance(self.mesh, MpmdMesh):
            if self.mesh.my_mpmd_axis_index in self.mpmd_idxs:
                return self.mesh.get_tensor(self.mesh.my_mpmd_axis_index, self.uid)
            else:
                if (
                    enable_empty_arrays := getattr(
                        jax._src.config, "enable_empty_arrays", None
                    )
                ) is not None and enable_empty_arrays.value:
                    sharding = mpmd_sharding(
                        self.mesh, self.mpmd_idxs, self.sharding.sharding
                    )
                    return jax.make_array_from_single_device_arrays(
                        self.shape, sharding, [], dtype=self.dtype
                    )
                else:
                    return np.array(np.nan, dtype=np.float32)

        fut_res = self.mesh.get_tensor(next(iter(self.mpmd_idxs)), self.uid)
        res = self.mesh.blocking_tree([fut_res])[0]
        return res

    def __format__(self, format_spec):
        # Simulates behavior of https://github.com/numpy/numpy/pull/9883
        if self.ndim == 0:
            return format(self._value[()], format_spec)

        return str(self._value)

    @property
    def ndim(self):
        return self.aval.ndim

    @property
    def shape(self):
        return self.aval.shape

    @property
    def dtype(self):
        return self.aval.dtype

    def block_until_ready(self):
        # Fetch the value and does nothing. Forces the value to be ready.
        # TODO: Actually verify that the value is ready to be fetched instead
        # of actually fetching it.
        _ = self._value
