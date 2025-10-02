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

from collections import OrderedDict

import jax
import jax._src.array as jarray
import jax._src.core as jcore

from jaxpp.mesh import MpmdMesh
from jaxpp.utils import get_named_sharding


class MpmdArray:
    def __init__(
        self,
        partially_addressable_arrays: list[jax.Array],
        mpmd_mesh: MpmdMesh,
        mpmd_idxs: frozenset[int],
        spec: jax.sharding.PartitionSpec | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: jax.numpy.dtype | None = None,
    ):
        self._mpmd_mesh = mpmd_mesh
        self._mpmd_idxs = tuple(sorted(mpmd_idxs))

        if mpmd_mesh.jax_mesh.is_multi_process:
            assert len(partially_addressable_arrays) <= 1

        partially_addressable_arrays_map = {}
        for idx, arr in enumerate(partially_addressable_arrays):
            mesh = get_named_sharding(arr).mesh
            if (mpmd_idx := mpmd_mesh.mpmd_idx_for_mesh.get(mesh)) is None:
                raise ValueError(
                    f"Argument array {idx} {arr.shape} is not on a mesh that is part "
                    f"mpmd_mesh={mpmd_mesh.jax_mesh}"
                )
            if mpmd_idx not in mpmd_idxs:
                raise ValueError(
                    f"Argument array's ({idx} {arr.shape}) mpmd_idx={mpmd_idx} not "
                    "in mpmd_idxs={mpmd_idxs}"
                )
            partially_addressable_arrays_map[mpmd_idx] = arr

        self._partially_addressable_arrays: OrderedDict[int, jax.Array] = OrderedDict(
            sorted(partially_addressable_arrays_map.items(), key=lambda x: x[0])
        )

        if len(self._partially_addressable_arrays) == 0:
            assert spec is not None
            assert shape is not None
            assert dtype is not None
        else:
            first_value = list(self._partially_addressable_arrays.values())[0]
            spec = spec if spec is not None else get_named_sharding(first_value).spec
            shape = shape if shape is not None else first_value.shape
            dtype = dtype if dtype is not None else first_value.dtype

            shapes = [a.shape for a in self._partially_addressable_arrays.values()]
            assert all(_ == shape for _ in shapes), (shape, shapes)
            dtypes = [a.dtype for a in self._partially_addressable_arrays.values()]
            assert all(_ == dtype for _ in dtypes), (dtype, dtypes)
            specs = [
                get_named_sharding(a).spec
                for a in self._partially_addressable_arrays.values()
            ]
            assert all(_ == spec for _ in specs), (spec, specs)

        self.spec = spec
        self._sharding = jax.sharding.NamedSharding(
            mpmd_mesh.mpmd_submesh(list(self._mpmd_idxs)).jax_mesh, spec
        )
        self.aval = jcore.ShapedArray(shape, dtype, weak_type=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.aval.shape

    @property
    def dtype(self) -> jax.numpy.dtype:
        return self.aval.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def sharding(self) -> jax.sharding.NamedSharding:
        """
        NOTE: this is different from self.to_mpmd_local_array.sharding
          if self.is_mpmd_replicated
        """
        return self._sharding

    def __repr__(self):
        return (
            f"MpmdArray(shape={self.shape}, dtype={self.dtype}, "
            f"mpmd_idxs={self._mpmd_idxs}, sharding={self._sharding})"
        )

    @property
    def is_mpmd_replicated(self) -> bool:
        """
        Returns True if the array is replicated in more than one mpmd rank.
        """
        return len(self._mpmd_idxs) > 1

    @property
    def is_partially_addressable(self) -> bool:
        """
        Returns True if the array is partially addressable in the mpmd rank
        this process participates in.
        An array is partially addressable at this rank if this rank holds a shard of
        the array (the shard can potentially be replicated across multiple mpmd ranks).
        """
        return len(self._partially_addressable_arrays) > 0

    @property
    def to_mpmd_local_array(self) -> jax.Array | list[jax.Array] | None:
        """
        Returns a jax.Array if the array is partially addressable in the mpmd rank
        this process participates in.
        Otherwise, returns None.
        Returns a list of arrays when it's a single process, multiple-devices mesh.
        """
        if not self.is_partially_addressable:
            return None
        els = list(self._partially_addressable_arrays.values())
        if len(els) == 1:
            return els[0]
        return els

    @property
    def first_mpmd_replica(self) -> jax.Array | None:
        if not self.is_partially_addressable:
            return None

        mpmd_idx, array = next(iter(self._partially_addressable_arrays.items()))
        if mpmd_idx == self._mpmd_idxs[0]:
            return array
        return None

    def __int__(self):
        assert self.is_partially_addressable, "Array is not partially addressable"
        return int(self.to_mpmd_local_array)

    def __format__(self, format_spec):
        assert self.is_partially_addressable, "Array is not partially addressable"
        return format(self.to_mpmd_local_array, format_spec)


jcore.pytype_aval_mappings[MpmdArray] = jarray._get_aval_array


@property
def _to_global_jax_array(mpmd_array: MpmdArray) -> jax.Array | None:
    if not mpmd_array.is_partially_addressable:
        if getattr(
            jax.config, "jax_enable_empty_arrays", False
        ) or jax.__version_info__ >= (0, 7, 1):
            return jax.make_array_from_single_device_arrays(
                shape=mpmd_array.shape,
                sharding=mpmd_array._sharding,
                arrays=[],
                dtype=mpmd_array.dtype,
            )
        return None

    return jax.make_array_from_single_device_arrays(
        shape=mpmd_array.shape,
        sharding=mpmd_array._sharding,
        arrays=[
            shard.data
            for arr in mpmd_array._partially_addressable_arrays.values()
            for shard in arr.addressable_shards
        ],
        dtype=mpmd_array.dtype,
    )
