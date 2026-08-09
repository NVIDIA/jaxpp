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

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, overload

import jax
from jax.sharding import NamedSharding, PartitionSpec

from jaxpp import jax_compat as jc
from jaxpp.mesh import MpmdMesh, _require_mpmd_indices
from jaxpp.utils import update_named_sharding


def make_array_from_addressable_shards(
    shape: tuple[int, ...],
    sharding: jax.sharding.Sharding,
    arrays: Iterable[jax.Array],
    *,
    dtype: Any,
) -> jax.Array:
    if isinstance(sharding, NamedSharding) and sharding.spec.unreduced:
        # JAX does not expose index maps for unreduced shardings here.
        addressable_devices = sharding.addressable_devices
        devices = tuple(
            device
            for device in sharding.mesh.devices.flat
            if device in addressable_devices
        )
    else:
        devices = tuple(sharding.addressable_devices_indices_map(shape))
    addressable_devices = set(devices)
    shard_by_device = {}
    for array in arrays:
        for shard in array.addressable_shards:
            if shard.device not in addressable_devices:
                continue
            if shard.device in shard_by_device:
                raise ValueError(f"duplicate shard for device {shard.device}")
            shard_by_device[shard.device] = shard.data

    missing_devices = [device for device in devices if device not in shard_by_device]
    if missing_devices:
        raise ValueError(f"missing shards for devices {missing_devices}")
    return jax.make_array_from_single_device_arrays(
        shape, sharding, [shard_by_device[device] for device in devices], dtype=dtype
    )


@overload
def axis_index(spec: PartitionSpec, axis_name: str) -> int | None: ...


@overload
def axis_index(spec: NamedSharding, axis_name: str) -> int | None: ...


def axis_index(spec: PartitionSpec | NamedSharding, axis_name: str) -> int | None:
    if isinstance(spec, NamedSharding):
        spec = spec.spec
    for idx, part in enumerate(jc.spec_partitions(spec)):
        if axis_name in jax.tree_util.tree_leaves(part):
            return idx
    return None


def normalize_index_groups(
    groups, *, mpmd_mesh: MpmdMesh, name: str
) -> tuple[tuple[int, ...], ...]:
    normalized_groups = []
    for group_idx, group in enumerate(groups):
        group_name = f"{name}[{group_idx}]"
        if isinstance(group, int):
            normalized_groups.append((group,))
            continue
        if not isinstance(group, Sequence) or isinstance(group, str):
            raise TypeError(f"{group_name} must be an int or a sequence of ints")
        if len(group) == 0:
            raise ValueError(f"{group_name} must not be empty")
        if not all(isinstance(mpmd_idx, int) for mpmd_idx in group):
            raise TypeError(f"{group_name} must contain only ints")
        normalized_groups.append(tuple(group))

    if len(normalized_groups) == 0:
        raise ValueError(f"{name} must not be empty")
    validate_index_groups(normalized_groups, mpmd_mesh=mpmd_mesh, name=name)
    return tuple(normalized_groups)


def validate_index_groups(
    groups: Sequence[tuple[int, ...]], *, mpmd_mesh: MpmdMesh, name: str
) -> tuple[int, ...]:
    all_indices = []
    seen = set()
    for group in groups:
        for idx in group:
            if idx in seen:
                raise ValueError(f"{name} must not contain duplicate MPMD indices")
            if idx < 0 or idx >= mpmd_mesh.mpmd_dim:
                raise ValueError(f"{name} contains out-of-range MPMD index {idx}")
            seen.add(idx)
            all_indices.append(idx)
    return tuple(all_indices)


class StackAxisInfo(NamedTuple):
    tail_shape: tuple[int, ...]
    per_index_extent: int
    spec: PartitionSpec
    memory_kind: str | None


def stack_shape_and_sharding(
    shapes: Sequence[tuple[int, ...]],
    shardings: Sequence[NamedSharding],
    *,
    mpmd_mesh: MpmdMesh,
    axis: int = 0,
) -> tuple[tuple[NamedSharding, ...], NamedSharding, tuple[int, ...], tuple[bool, ...]]:
    """Infer stack metadata within an MPMD coordinate scope.

    For stack, `mpmd_mesh` is the scope of the output value. It may be the
    ambient global mesh, or a submesh such as the union of the input meshes or a
    collective group. Input sharding meshes are interpreted relative to this
    scope, and the output sharding uses `mpmd_mesh.jax_mesh`.
    """
    if len(shapes) == 0:
        raise ValueError("stack expects at least one argument")
    if len(shapes) != len(shardings):
        raise ValueError("stack metadata must have matching lengths")

    axis_name = mpmd_mesh.mpmd_axis_name
    index_groups = tuple(
        _require_mpmd_indices(mpmd_mesh, sharding.mesh, name=f"stack argument {idx}")
        for idx, sharding in enumerate(shardings)
    )
    validate_index_groups(
        index_groups, mpmd_mesh=mpmd_mesh, name="stack argument meshes"
    )

    existing_axis_infos = []
    for idx, (shape, sharding, indices) in enumerate(
        zip(shapes, shardings, index_groups, strict=True)
    ):
        spec_axis = axis_index(sharding, axis_name)
        if spec_axis is None:
            if len(indices) != 1:
                raise ValueError(
                    "stack arguments must not mix replicated values from multiple "
                    f"MPMD indices; argument {idx} without {axis_name!r} in its "
                    "PartitionSpec must use a single MPMD index"
                )
            continue
        spec = jc.spec_partitions(sharding.spec)
        if spec_axis != axis or spec[axis] != axis_name:
            raise ValueError(
                "stack arguments must use "
                f"{axis_name!r} only as PartitionSpec axis {axis}"
            )
        if len(shape) <= axis:
            raise ValueError(f"stack argument {idx} must have an MPMD axis dimension")
        if shape[axis] % len(indices) != 0:
            raise ValueError(
                "stack argument leading dimension must be divisible by the number "
                "of MPMD indices"
            )
        existing_axis_infos.append(
            StackAxisInfo(
                tail_shape=shape[:axis] + shape[axis + 1 :],
                per_index_extent=shape[axis] // len(indices),
                spec=sharding.spec,
                memory_kind=sharding.memory_kind,
            )
        )

    if existing_axis_infos:
        first_info = existing_axis_infos[0]
        tail_shape = first_info.tail_shape
        per_index_extent = first_info.per_index_extent
        out_spec = first_info.spec
        memory_kind = first_info.memory_kind
        for info in existing_axis_infos[1:]:
            if info.tail_shape != tail_shape:
                raise ValueError("stack arguments must have the same non-MPMD shape")
            if info.per_index_extent != per_index_extent:
                raise ValueError(
                    "stack arguments must have the same per-MPMD-index extent"
                )
            if info.spec != out_spec:
                raise ValueError("stack arguments must have the same PartitionSpec")
            if info.memory_kind != memory_kind:
                raise ValueError("stack arguments must have the same memory kind")

        logical_in_shardings = []
        expand_inputs = []
        out_spec_parts = jc.spec_partitions(out_spec)
        no_axis_spec = out_spec.update(
            partitions=out_spec_parts[:axis] + out_spec_parts[axis + 1 :]
        )
        for idx, (shape, sharding) in enumerate(zip(shapes, shardings, strict=True)):
            if axis_index(sharding, axis_name) is not None:
                logical_in_shardings.append(sharding)
                expand_inputs.append(False)
                continue
            if shape != tail_shape or per_index_extent != 1:
                raise ValueError(
                    f"stack argument {idx} must have {axis_name!r} in its "
                    "PartitionSpec to stack an existing MPMD dimension"
                )
            if sharding.spec != no_axis_spec:
                raise ValueError(
                    "stack replicated arguments must match the non-MPMD "
                    "PartitionSpec"
                )
            if sharding.memory_kind != memory_kind:
                raise ValueError("stack arguments must have the same memory kind")
            logical_in_shardings.append(sharding)
            expand_inputs.append(True)
        out_shape = (
            *tail_shape[:axis],
            per_index_extent * mpmd_mesh.mpmd_dim,
            *tail_shape[axis:],
        )
    else:
        tail_shape = shapes[0]
        spec = shardings[0].spec
        spec_parts = jc.spec_partitions(spec)
        out_spec = spec.update(
            partitions=spec_parts[:axis] + (axis_name,) + spec_parts[axis:]
        )
        memory_kind = shardings[0].memory_kind
        for shape, sharding in zip(shapes[1:], shardings[1:], strict=True):
            if shape != tail_shape:
                raise ValueError("stack arguments must have the same shape")
            if sharding.spec != shardings[0].spec:
                raise ValueError("stack arguments must have the same PartitionSpec")
            if sharding.memory_kind != memory_kind:
                raise ValueError("stack arguments must have the same memory kind")
        logical_in_shardings = list(shardings)
        expand_inputs = [True] * len(shardings)
        out_shape = (*tail_shape[:axis], mpmd_mesh.mpmd_dim, *tail_shape[axis:])

    out_sharding = update_named_sharding(
        shardings[0], mesh=mpmd_mesh.jax_mesh, spec=out_spec
    )
    return tuple(logical_in_shardings), out_sharding, out_shape, tuple(expand_inputs)


def stack_arrays_with_shardings(
    arrays: Sequence[jax.Array],
    shardings: Sequence[NamedSharding],
    *,
    mpmd_mesh: MpmdMesh,
    axis: int = 0,
) -> jax.Array:
    if len(arrays) == 0:
        raise ValueError("stack expects at least one argument")
    _, out_sharding, out_shape, expand_inputs = stack_shape_and_sharding(
        tuple(array.shape for array in arrays),
        shardings,
        mpmd_mesh=mpmd_mesh,
        axis=axis,
    )
    cast_arrays = []
    for array, expand in zip(arrays, expand_inputs, strict=True):
        if expand:
            # NOTE(ambient-mesh): the caller may hold jax.set_mesh of a mesh
            # that does not contain this array's devices (e.g. a train loop
            # pinned to one stage's mesh), and eager dispatch rejects arrays
            # outside the ambient mesh. Expand under the array's own mesh.
            with jax.set_mesh(array.sharding.mesh):
                cast_arrays.append(jax.numpy.expand_dims(array, axis=axis))
        else:
            cast_arrays.append(array)
    return make_array_from_addressable_shards(
        out_shape, out_sharding, cast_arrays, dtype=arrays[0].dtype
    )


def local_stack_array(
    array: jax.Array,
    *,
    out_shape: tuple[int, ...],
    out_sharding: NamedSharding,
    expand: bool,
    axis: int = 0,
) -> jax.Array:
    if expand:
        # NOTE(ambient-mesh)
        with jax.set_mesh(array.sharding.mesh):
            array = jax.numpy.expand_dims(array, axis=axis)
    return make_array_from_addressable_shards(
        out_shape, out_sharding, (array,), dtype=array.dtype
    )


def slice_shape_and_shardings(
    shape: tuple[int, ...],
    in_sharding: NamedSharding,
    groups: Sequence[tuple[int, ...]],
    *,
    mpmd_mesh: MpmdMesh,
) -> tuple[tuple[tuple[int, ...], ...], tuple[NamedSharding, ...]]:
    """Infer slice outputs in the coordinate scope used by `groups`.

    For slice, `mpmd_mesh` defines the numbering for `groups`; output shardings
    are built from `mpmd_mesh.mpmd_submesh(group)`. It is not necessarily the
    ambient global mesh, but it must contain the input sharding mesh and all
    requested group indices.
    """
    axis_name = mpmd_mesh.mpmd_axis_name
    mpmd_axis = axis_index(in_sharding, axis_name)
    if mpmd_axis is not None and (
        mpmd_axis != 0 or jc.spec_partitions(in_sharding.spec)[0] != axis_name
    ):
        raise ValueError(
            "slice argument must use "
            f"{axis_name!r} only as the leading PartitionSpec axis"
        )

    validate_index_groups(groups, mpmd_mesh=mpmd_mesh, name="slice groups")
    in_indices = _require_mpmd_indices(
        mpmd_mesh, in_sharding.mesh, name="slice argument"
    )
    requested_indices = tuple(idx for group in groups for idx in group)
    if not set(requested_indices).issubset(in_indices):
        raise ValueError("slice groups must be contained in the input mesh")

    if mpmd_axis == 0:
        if len(shape) == 0:
            raise ValueError("slice argument must have a leading MPMD dimension")
        if shape[0] % len(in_indices) != 0:
            raise ValueError(
                "slice argument leading dimension must be divisible by the number "
                "of MPMD indices"
            )
        per_index_extent = shape[0] // len(in_indices)
        out_shapes = tuple(
            (per_index_extent * len(group), *shape[1:]) for group in groups
        )
    else:
        out_shapes = (shape,) * len(groups)

    out_shardings = tuple(
        update_named_sharding(
            in_sharding, mesh=mpmd_mesh.mpmd_submesh(list(group)).jax_mesh
        )
        for group in groups
    )
    return out_shapes, out_shardings


def slice_arrays(
    array: jax.Array,
    *,
    in_sharding: NamedSharding,
    groups: Sequence[tuple[int, ...]],
    mpmd_mesh: MpmdMesh,
) -> tuple[jax.Array, ...]:
    out_shapes, out_shardings = slice_shape_and_shardings(
        array.shape, in_sharding, groups, mpmd_mesh=mpmd_mesh
    )
    return tuple(
        make_array_from_addressable_shards(
            out_shape, out_sharding, (array,), dtype=array.dtype
        )
        for out_shape, out_sharding in zip(out_shapes, out_shardings, strict=True)
    )


def local_slice_arrays(arg, *, in_sharding, out_shardings):
    return tuple(
        make_array_from_addressable_shards(
            local_slice_out_shape(arg.shape, in_sharding, out_sharding),
            out_sharding,
            (arg,),
            dtype=arg.dtype,
        )
        for out_sharding in out_shardings
    )


def local_slice_out_shape(
    shape: tuple[int, ...], in_sharding: NamedSharding, out_sharding: NamedSharding
) -> tuple[int, ...]:
    spec = jc.spec_partitions(in_sharding.spec)
    if len(spec) == 0 or not isinstance(spec[0], str):
        return shape
    axis_name = spec[0]
    mesh_extent = in_sharding.mesh.shape[axis_name]
    assert shape[0] % mesh_extent == 0, (shape, mesh_extent, in_sharding)
    per_index_extent = shape[0] // mesh_extent
    return (per_index_extent * out_sharding.mesh.shape[axis_name], *shape[1:])
