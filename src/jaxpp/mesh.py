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

import dataclasses
from collections.abc import Mapping
from functools import cached_property
from typing import ClassVar

import jax
import numpy as np

from jaxpp import env_vars


@dataclasses.dataclass(frozen=True)
class MpmdMesh:
    """A JAX mesh partitioned into MPMD (Multiple Program Multiple Data) groups.

    MpmdMesh wraps a standard JAX mesh and designates one axis as the "MPMD axis".
    The mesh is conceptually split into multiple independent groups along this axis,
    where each group can execute different computations (e.g., pipeline stages).

    For example, with a mesh of shape {'mpmd': 4, 'data': 2, 'model': 2} and
    mpmd_axis_name='mpmd', the mesh is split into 4 MPMD groups, each containing
    4 devices (2 data x 2 model). Each group runs its own computation, and arrays
    can be distributed across one or more groups.

    Key concepts:
        - MPMD group: A slice of the mesh along the MPMD axis. Each group
          has an index from 0 to mpmd_dim - 1.
        - Submesh: A subset of MPMD groups combined into a single mesh.
          Used when arrays are replicated across multiple groups.
        - Lowering mesh: The mesh used for XLA compilation, which is the
          local process's MPMD group mesh in multi-process settings.

    In multi-process execution, each process belongs to exactly one MPMD group.
    Arrays may be replicated across multiple groups when needed as inputs by
    multiple pipeline stages (common for constants and loop invariants).

    Attributes:
        jax_mesh: The underlying JAX mesh containing all devices.
        mpmd_axis_name: Name of the axis used to partition into MPMD groups.
    """

    jax_mesh: jax.sharding.Mesh
    mpmd_axis_name: str
    mesh_stack: ClassVar[list["MpmdMesh"]] = []

    def __enter__(self):
        MpmdMesh.mesh_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.mesh_stack.pop()

    def __post_init__(self):
        if not self.jax_mesh.is_multi_process:
            return

        mpmd_idx_by_process = dict[int, int]()
        for d in self.jax_mesh._flat_devices_set:
            if (mpmd_idx := mpmd_idx_by_process.get(d.process_index)) is not None:
                if self.device_mpmd_idx[d] != mpmd_idx:
                    raise AssertionError(
                        f"Process {d.process_index} found in two mpmd indices: "
                        f"{mpmd_idx} {self.device_coords[d][self.mpmd_axis]}"
                        f"{jax.local_devices()=} {self.device_coords}"
                    )
            else:
                mpmd_idx_by_process[d.process_index] = self.device_mpmd_idx[d]

    @cached_property
    def device_coords(self) -> Mapping[jax.Device, tuple[int, ...]]:
        return {
            device: coord for coord, device in np.ndenumerate(self.jax_mesh.devices)
        }

    @cached_property
    def device_mpmd_idx(self) -> Mapping[jax.Device, int]:
        return {
            device: self.device_coords[device][self.mpmd_axis]
            for device in self.jax_mesh.devices.flat
        }

    @cached_property
    def mpmd_dim(self):
        return self.jax_mesh.shape[self.mpmd_axis_name]

    @cached_property
    def mpmd_axis(self) -> int:
        return self.jax_mesh.axis_names.index(self.mpmd_axis_name)

    @cached_property
    def unstack(self) -> list[jax.sharding.Mesh]:
        return [
            jax.sharding.Mesh(mpmd_group_devices, self.jax_mesh.axis_names)
            for mpmd_group_devices in np.split(
                self.jax_mesh.devices, self.mpmd_dim, self.mpmd_axis
            )
        ]

    @property
    def mpmd_idx_for_mesh(self) -> Mapping[jax.sharding.Mesh, int]:
        return {mesh: idx for idx, mesh in enumerate(self.unstack)}

    @cached_property
    def my_mpmd_axis_index(self) -> int:
        if (
            not env_vars.jaxpp_debug_force_mpmdify.value
            and not self.jax_mesh.is_multi_process
        ):
            raise ValueError(
                "my_mpmd_axis_index is supported only in multi-process meshes"
            )
        local_devices = set(jax.local_devices())
        my_devices_coord = {
            self.device_coords[d][self.mpmd_axis]
            for d in self.jax_mesh._flat_devices_tuple
            if d in local_devices
        }
        (mpmd_axis_index,) = my_devices_coord
        return mpmd_axis_index

    @cached_property
    def my_mpmd_group_mesh(self) -> jax.sharding.Mesh:
        if not self.jax_mesh.is_multi_process:
            raise ValueError(
                "my_mpmd_group_mesh is supported only in multi-process meshes"
            )
        return jax.sharding.Mesh(
            np.expand_dims(
                np.take(self.jax_mesh.devices, self.my_mpmd_axis_index, self.mpmd_axis),
                axis=self.mpmd_axis,
            ),
            self.jax_mesh.axis_names,
        )

    def lowering_mesh(self) -> jax.sharding.Mesh:
        if not self.jax_mesh.is_multi_process:
            return self.mpmd_submesh([0]).jax_mesh
        return self.my_mpmd_group_mesh

    def mpmd_submesh(self, mpmd_indices: list[int]) -> "MpmdMesh":
        assert isinstance(mpmd_indices, list)
        jax_mesh = jax.sharding.Mesh(
            np.take(self.jax_mesh.devices, mpmd_indices, self.mpmd_axis),
            self.jax_mesh.axis_names,
        )
        return MpmdMesh(jax_mesh, self.mpmd_axis_name)
