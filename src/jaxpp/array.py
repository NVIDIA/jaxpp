# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import OrderedDict, defaultdict
from typing import Any, cast

import jax
import numpy as np

from jaxpp.jax_compat import core as jcore
from jaxpp.mesh import MpmdMesh
from jaxpp.types import MpmdSharding
from jaxpp.utils import filter_axes, get_named_sharding, update_named_sharding


class MpmdArray:
    """An array distributed across one or more MPMD groups.

    MpmdArray represents a logical array that exists in a subset of MPMD groups
    within an :class:`~jaxpp.mesh.MpmdMesh`.
    Unlike standard JAX SPMD arrays where all processes have a shard of the corresponding
    array, an MpmdArray is only "partially addressable", it exists as sharded in only
    one or more MPMD groups (potentially all of them too).

    Properties:
        - `mpmd_idxs`: The set of MPMD group indices where this array exists.
          Most computed arrays exist in a single group (len=1), but constants
          and loop invariants may be replicated across multiple groups when
          needed as inputs by multiple pipeline stages.
        - `is_partially_addressable`: A process can only access array shards if
          it belongs to one of the MPMD groups in mpmd_idxs.
          Use `to_mpmd_local_array` to get the SPMD JAX array
          (potentially spanning multiple devices) for this MPMD group.
        - `is_mpmd_replicated`: When `len(mpmd_idxs) > 1`, the array data is replicated
          across those groups.

    The `sharding` property returns a NamedSharding whose mesh spans all devices
    in mpmd_idxs, useful for resharding between SPMD and MPMD layouts.

    Example:
        An array with mpmd_idxs={0, 2} on a 4-group MPMD mesh exists in groups
        0 and 2. Processes in groups 1 and 3 cannot access this array's data
        (is_partially_addressable=False for them).

    Attributes:
        spec: The PartitionSpec describing how the array is sharded within
            each MPMD group.
        aval: The abstract value (shape and dtype) of the array.
    """

    def __init__(
        self,
        partially_addressable_arrays: list[jax.Array],
        mpmd_sharding: MpmdSharding,
        shape: tuple[int, ...] | None = None,
        dtype: jax.numpy.dtype | None = None,
    ):
        mpmd_mesh = mpmd_sharding.mpmd_mesh
        mpmd_idxs = mpmd_sharding.mesh_ids
        spec = mpmd_sharding.spec

        self._mpmd_sharding = mpmd_sharding
        self._mpmd_mesh = mpmd_mesh
        self._mpmd_idxs = tuple(sorted(mpmd_idxs))

        if mpmd_mesh.jax_mesh.is_multi_process:
            assert len(partially_addressable_arrays) <= 1

        partially_addressable_arrays_map = {}
        for idx, arr in enumerate(partially_addressable_arrays):
            mesh = get_named_sharding(arr).mesh
            if (mpmd_idx := mpmd_mesh.mpmd_idx_for_mesh.get(mesh)) is None:
                raise ValueError(
                    f"Argument array {idx} {arr.shape} is not on a mesh that is part"
                    f" of mpmd_mesh={mpmd_mesh.jax_mesh}"
                )

            if mpmd_idx not in mpmd_idxs:
                raise ValueError(
                    f"Argument array's ({idx} {arr.shape}) mpmd_idx={mpmd_idx} not "
                    "in mpmd_idxs={mpmd_idxs}"
                )

            if mpmd_idx in partially_addressable_arrays_map:
                raise ValueError(
                    f"Argument array {idx} {arr.shape} already has a "
                    f"mpmd_idx={mpmd_idx}"
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
            shape = shape if shape is not None else first_value.shape
            dtype = dtype if dtype is not None else first_value.dtype

            shapes = [a.shape for a in self._partially_addressable_arrays.values()]
            assert all(_ == shape for _ in shapes), (shape, shapes)
            dtypes = [a.dtype for a in self._partially_addressable_arrays.values()]
            assert all(_ == dtype for _ in dtypes), (dtype, dtypes)
            mpmd_axis = mpmd_sharding.mpmd_mesh.mpmd_axis_name
            specs = [
                filter_axes(get_named_sharding(a).spec, {mpmd_axis})
                for a in self._partially_addressable_arrays.values()
            ]
            assert all(_ == mpmd_sharding.spec for _ in specs), (
                mpmd_sharding.spec,
                specs,
            )

        self._spec = mpmd_sharding.spec
        self._sharding = mpmd_sharding.sharding

        # TODO: maybe add sharding/vma/memory_space to aval
        self.aval = jcore.ShapedArray(shape, dtype, weak_type=False)

    @property
    def spec(self) -> jax.sharding.PartitionSpec:
        return self._spec

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

    @property
    def _mpmd_local_sharding(self) -> jax.sharding.NamedSharding:
        return jax.sharding.NamedSharding(self._mpmd_mesh.lowering_mesh(), self.spec)

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

    def delete(self):
        assert self.is_partially_addressable, "Array is not partially addressable"
        assert not self.is_deleted(), "Array is deleted"
        for arr in self._partially_addressable_arrays.values():
            arr.delete()

    def is_deleted(self) -> bool:
        assert self.is_partially_addressable, "Array is not partially addressable"
        if len(self._partially_addressable_arrays) == 1:
            return next(iter(self._partially_addressable_arrays.items()))[
                1
            ].is_deleted()

        _ = [a.is_deleted() for a in self._partially_addressable_arrays.values()]
        deleted = any(_)
        assert deleted == all(_)
        return deleted

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

        assert not self.is_deleted(), "Array is deleted"

        els = list(self._partially_addressable_arrays.values())
        if len(els) == 1:
            return els[0]
        return els

    @property
    def first_mpmd_replica(self) -> jax.Array | None:
        if not self.is_partially_addressable:
            return None

        assert not self.is_deleted(), "Array is deleted"

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

    def block_until_ready(self):
        for arr in self._partially_addressable_arrays.values():
            arr.block_until_ready()
        return self


def pytype_aval_mapping(self: MpmdArray) -> jcore.AbstractValue:
    aval = self.aval
    if hasattr(aval, "sharding"):
        return jcore.update_aval_with_sharding(self.aval, self._mpmd_local_sharding)
    return aval


jcore.pytype_aval_mappings[MpmdArray] = pytype_aval_mapping


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


def _id(*xs):
    return xs


def _spmd_to_mpmd_reshard(
    mpmd_mesh: MpmdMesh,
    spmd_values: list[jax.Array],
    dist_shardings: list[MpmdSharding],
    donate: list[bool] | None = None,
) -> list[MpmdArray]:
    if donate is None:
        donate = [False] * len(spmd_values)

    for spmd_value, dist_sharding in zip(spmd_values, dist_shardings):
        assert isinstance(
            spmd_value.sharding,
            (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding),
        ), f"Unsupported sharding type: {spmd_value.sharding}"

    # NOTE: We filter out the mpmd axis from the sharding so that
    # the output is replicated across all mpmd ranks.
    _actual_shardings = tuple(
        update_named_sharding(
            filter_axes(
                cast(jax.sharding.NamedSharding, dist_sharding.sharding),
                {mpmd_mesh.mpmd_axis_name},
            ),
            mesh=mpmd_mesh.jax_mesh,
        )
        for dist_sharding in dist_shardings
    )

    res: list[jax.Array] = jax.jit(_id, out_shardings=_actual_shardings)(*spmd_values)

    for spmd_value, donated in zip(spmd_values, donate, strict=True):
        if donated:
            spmd_value.delete()

    if not mpmd_mesh.jax_mesh.is_multi_process:
        # TODO: return an jaxpp.MpmdArray instead of a list of jax.Array
        _res = []
        for _, dsh in zip(res, dist_shardings, strict=True):
            shards = []
            for s in _.addressable_shards:
                if mpmd_mesh.device_mpmd_idx[s.device] in dsh.mesh_ids:
                    shards.append(s.data)
                else:
                    s.data.delete()

            _arr = jax.make_array_from_single_device_arrays(
                _.shape,
                jax.sharding.NamedSharding(
                    mpmd_mesh.mpmd_submesh(list(dsh.mesh_ids)).jax_mesh,
                    _.sharding.spec,
                ),
                shards,
            )

            _res.append(_arr)
        return _res

    _res = []
    for arr, dsh in zip(res, dist_shardings, strict=True):
        mesh_ids = dsh.mesh_ids
        # MpmdSharding.__post_init__ canonicalizes the spec by filtering out
        # the mpmd axis, so we can just use dsh.spec directly
        mpmd_sharding = MpmdSharding(
            mpmd_mesh=dsh.mpmd_mesh, mesh_ids=dsh.mesh_ids, spec=dsh.spec
        )

        if mpmd_mesh.my_mpmd_axis_index not in mesh_ids:
            _res.append(
                MpmdArray(
                    partially_addressable_arrays=[],
                    mpmd_sharding=mpmd_sharding,
                    shape=arr.shape,
                    dtype=arr.dtype,
                )
            )
            arr.delete()
        else:
            new_arr = jax.make_array_from_single_device_arrays(
                arr.shape,
                jax.sharding.NamedSharding(
                    mpmd_mesh.my_mpmd_group_mesh, arr.sharding.spec
                ),
                [s.data for s in arr.addressable_shards],
            )
            _res.append(
                MpmdArray(
                    partially_addressable_arrays=[new_arr], mpmd_sharding=mpmd_sharding
                )
            )
    return _res


def _get_working_memory_threshold() -> int:
    """Get the minimum available working memory across all devices globally."""
    min_available = float("inf")
    for d in jax.local_devices():
        stats = d.memory_stats()
        available = stats["bytes_limit"] - stats["peak_bytes_in_use"]
        min_available = min(min_available, available)

    # Ensure all processes use the same threshold by computing global minimum
    if jax.process_count() > 1:
        # Use process_allgather to collect all local minimums, then compute global min
        # Note: process_allgather requires an array, not a scalar
        local_min_array = jax.numpy.array(min_available, dtype=jax.numpy.float32)
        all_mins = jax.experimental.multihost_utils.process_allgather(
            local_min_array, tiled=True
        )
        min_available = float(jax.numpy.min(all_mins))

    return int(min_available // 3)


def _build_mpmd_interleaved_order(
    arrays: list[jax.Array],
    shardings: list[MpmdSharding],
) -> list[int]:
    """Build array order interleaved by mpmd_idx, largest first within each idx."""
    by_mpmd_idx: dict[int, list[int]] = defaultdict(list)
    for i, dsh in enumerate(shardings):
        by_mpmd_idx[min(dsh.mesh_ids)].append(i)

    # Sort each mpmd_idx group by size (largest last for popping)
    by_mpmd_idx = {
        mpmd_idx: sorted(vs, key=lambda i: arrays[i].addressable_shards[0].data.nbytes)
        for mpmd_idx, vs in by_mpmd_idx.items()
    }

    # Build order by round-robin popping largest from each mpmd_idx
    order: list[int] = []
    while any(by_mpmd_idx.values()):
        for mpmd_idx in list(by_mpmd_idx.keys()):
            if by_mpmd_idx[mpmd_idx]:
                order.append(by_mpmd_idx[mpmd_idx].pop())

    return order


def spmd_to_mpmd_reshard(
    mpmd_mesh: MpmdMesh, spmd_arrays, mpmd_shardings, threshold: int | None = None
):
    """
    Reshards a pytree of SPMD arrays to MPMD arrays.

    This function redistributes data from a Single Program Multiple Data (SPMD)
    layout to a Multiple Program Multiple Data (MPMD) layout. It handles
    memory constraints by grouping arrays and processing them in chunks.
    It's the caller's responsibility to not use the input spmd_arrays after calling
    this function as they will be consumed by this function.

    The specs of the returned arrays will _not_ have `mpmd_mesh.mpmd_axis_name` in
    them.

    Limitations: same constraints as jax.jit apply (e.g. _device_assignment must be the
    same for all arrays)

    Args:
        mpmd_mesh: The MPMD mesh definition.
        spmd_arrays: A pytree of source SPMD arrays.
        mpmd_shardings: A pytree of target MPMD shardings, matching the structure of
                        spmd_arrays.
        threshold: Memory threshold in bytes for grouping operations.
                   If None, calculated based on available memory.

    Returns:
        A pytree of MpmdArray objects with the same structure as spmd_arrays.
    """
    spmd_arrays_with_path, spmd_tree_def = jax.tree.flatten_with_path(spmd_arrays)
    mpmd_shardings_flat, mpmd_tree_def = jax.tree.flatten(mpmd_shardings)

    # For unused arrays (len(dsh.mesh_ids) == 0), we default their placement
    # to mpmd rank 0
    mpmd_shardings_flat = [
        MpmdSharding(mpmd_mesh=dsh.mpmd_mesh, mesh_ids={0}, spec=dsh.spec)
        if len(dsh.mesh_ids) == 0
        else dsh
        for dsh in mpmd_shardings_flat
    ]

    assert spmd_tree_def == mpmd_tree_def

    _, spmd_arrays_flat = jax._src.util.unzip2(spmd_arrays_with_path)
    spmd_arrays_flat_list = list(spmd_arrays_flat)

    # Build interleaved order by mpmd_idx (largest first within each idx)
    order = _build_mpmd_interleaved_order(spmd_arrays_flat_list, mpmd_shardings_flat)
    ordered_arrays = [spmd_arrays_flat_list[i] for i in order]

    # Group by memory threshold
    threshold = threshold if threshold is not None else _get_working_memory_threshold()
    groups = _group_by_size_threshold(
        [
            a.addressable_shards[0].data.nbytes * mpmd_mesh.mpmd_dim
            for a in ordered_arrays
        ],
        threshold,
    )

    resharded_arrays_by_index: dict[int, MpmdArray] = {}
    for group_idx, group_indices in enumerate(groups):
        # Map group indices back to original indices
        orig_indices = [order[i] for i in group_indices]
        group_arrays = [spmd_arrays_flat_list[i] for i in orig_indices]
        group_mpmd_shardings = [mpmd_shardings_flat[i] for i in orig_indices]

        group_results = _spmd_to_mpmd_reshard(
            mpmd_mesh,
            group_arrays,
            group_mpmd_shardings,
            donate=[True] * len(group_arrays),  # FIXME: Maybe make it a kwarg
        )
        for orig_idx, result in zip(orig_indices, group_results):
            resharded_arrays_by_index[orig_idx] = result

    resharded_flat = [
        resharded_arrays_by_index[i] for i in range(len(spmd_arrays_flat_list))
    ]
    return jax.tree.unflatten(spmd_tree_def, resharded_flat)


def _group_by_size_threshold(
    sizes: list[int],
    threshold: int,
) -> list[list[int]]:
    """Group indices by size threshold.

    Groups are formed sequentially - entries are added to the current group
    until adding another would exceed the threshold, then a new group starts.
    Returns groups of indices into the input sizes list.
    """
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_size = 0

    for i, size in enumerate(sizes):
        if current_size + size > threshold and current_group:
            groups.append(current_group)
            current_group = []
            current_size = 0
        current_group.append(i)
        current_size += size

    if current_group:
        groups.append(current_group)

    return groups


def _axis_name_in_spec(axis_name: str, spec) -> bool:
    for elem in spec:
        if elem == axis_name:
            return True
        if isinstance(elem, tuple) and axis_name in elem:
            return True
    return False


def logically_stacked(
    array: jax.Array,
    comm_mesh: jax.sharding.Mesh,
    mesh_axis_name: str,
    array_axis: int = 0,
    strict: bool = False,
):
    """
    Logically stacks an array along a new axis corresponding to the MPMD dimension.

    This function expands the input array's dimensions and reshards it across
    the communication mesh, effectively treating distributed shards as a single
    logical array with an extra dimension.
    """
    assert isinstance(array.sharding, jax.sharding.NamedSharding)

    if strict:
        spec = array.sharding.spec
        assert not _axis_name_in_spec(
            mesh_axis_name, spec
        ), f"axis_name {mesh_axis_name!r} already exists in spec {spec}"
    else:
        spec = filter_axes(array.sharding.spec, {mesh_axis_name})

    expanded_array = jax.numpy.expand_dims(array, array_axis)
    in_sharding = jax.sharding.NamedSharding(
        comm_mesh,
        jax.sharding.PartitionSpec(
            *spec[:array_axis], mesh_axis_name, *spec[array_axis:]
        ),
    )

    global_array = jax.make_array_from_single_device_arrays(
        (
            *array.shape[:array_axis],
            comm_mesh.shape[mesh_axis_name],
            *array.shape[array_axis:],
        ),
        in_sharding,
        [s.data for s in expanded_array.addressable_shards],
    )
    return global_array


def _select_mpmd_slice(arrays, mpmd_idxs):
    """Selector function to pick the slice corresponding to the MPMD index."""
    return tuple(array[idx] for array, idx in zip(arrays, mpmd_idxs, strict=True))


def mpmd_to_spmd_reshard(
    mpmd_mesh: MpmdMesh, mpmd_arrays, spmd_shardings, threshold: int | None = None
) -> jax.Array:
    """
    Reshards a pytree of MPMD arrays to SPMD arrays.

    This function redistributes data from a Multiple Program Multiple Data (MPMD)
    layout back to a Single Program Multiple Data (SPMD) layout. It reconstructs
    global arrays from distributed MPMD shards.
    It's the caller's responsibility to not use the input mpmd_arrays after calling
    this function as they will be consumed by this function.

    Args:
        mpmd_mesh: The MPMD mesh definition.
        mpmd_arrays: A pytree of source MPMD arrays.
        spmd_shardings: A pytree of target SPMD shardings.
        threshold: Memory threshold in bytes for grouping operations.
                   If None, calculated based on available memory.

    Returns:
        A pytree of JAX arrays with the same structure as mpmd_arrays.
    """

    if not mpmd_mesh.jax_mesh.is_multi_process:
        return jax.device_put(
            jax.tree.map(lambda _: _.first_mpmd_replica, mpmd_arrays), spmd_shardings
        )

    mpmd_arrays_with_path, mpmd_tree_def = jax.tree.flatten_with_path(mpmd_arrays)
    mpmd_arrays_with_path: list[tuple[Any, MpmdArray]]
    spmd_shardings_flat, spmd_tree_def = jax.tree.flatten(spmd_shardings)
    assert mpmd_tree_def == spmd_tree_def

    donate = [True] * len(mpmd_arrays_with_path)  # FIXME: Maybe make it a kwarg

    # Collect metadata without building stacked arrays yet
    mpmd_arr_list = [mpmd_arr for _, mpmd_arr in mpmd_arrays_with_path]
    mpmd_idxs = [mpmd_arr._mpmd_idxs[0] for mpmd_arr in mpmd_arr_list]

    def get_shard_size(mpmd_arr: MpmdArray) -> int:
        return int(
            np.prod(mpmd_arr._mpmd_local_sharding.shard_shape(mpmd_arr.shape))
            * np.dtype(mpmd_arr.dtype).itemsize
        )

    sizes = [get_shard_size(mpmd_arr) for mpmd_arr in mpmd_arr_list]
    # Sort by size (largest first) for better memory efficiency
    order = sorted(range(len(mpmd_arr_list)), key=lambda i: sizes[i], reverse=True)
    groups = _group_by_size_threshold(
        [sizes[i] * mpmd_mesh.mpmd_dim for i in order],
        threshold if threshold is not None else _get_working_memory_threshold(),
    )

    resharded_arrays_by_index: dict[int, jax.Array] = {}
    for group_indices in groups:
        orig_indices = [order[i] for i in group_indices]

        # Build stacked arrays for this group
        group_stacked = []
        for orig_idx in orig_indices:
            mpmd_arr = mpmd_arr_list[orig_idx]
            local_array = mpmd_arr.first_mpmd_replica
            if local_array is None:
                # Create zeros if this rank doesn't hold data for this array
                local_array = jax.jit(
                    jax.numpy.zeros,
                    static_argnums=(0, 1),
                    out_shardings=mpmd_arr._mpmd_local_sharding,
                )(mpmd_arr.shape, mpmd_arr.dtype)

            stacked = logically_stacked(
                local_array, mpmd_mesh.jax_mesh, mpmd_mesh.mpmd_axis_name
            )
            # logically_stacked creates a new array, so we can delete the local array
            if donate[orig_idx]:
                local_array.delete()
            group_stacked.append(stacked)

        group_stacked = tuple(group_stacked)
        group_mpmd_idxs = tuple(mpmd_idxs[i] for i in orig_indices)
        group_spmd_shardings = tuple(spmd_shardings_flat[i] for i in orig_indices)

        in_shardings = (tuple(_.sharding for _ in group_stacked),)
        group_results = jax.jit(
            _select_mpmd_slice,
            in_shardings=in_shardings,
            out_shardings=group_spmd_shardings,
            static_argnums=(1,),
        )(group_stacked, group_mpmd_idxs)

        for i, orig_idx in enumerate(orig_indices):
            resharded_arrays_by_index[orig_idx] = group_results[i]
            group_stacked[i].delete()

    resharded_flat = [resharded_arrays_by_index[i] for i in range(len(mpmd_arr_list))]
    return jax.tree.unflatten(spmd_tree_def, resharded_flat)
