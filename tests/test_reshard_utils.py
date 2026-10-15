# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import numpy as np
from jax.sharding import PartitionSpec as P
from parameterized import parameterized

import jaxpp.distributed_utils as jppdu
from jaxpp.array import filter_axes, mpmd_to_spmd_reshard, spmd_to_mpmd_reshard
from jaxpp.mesh import MpmdMesh
from jaxpp.types import MpmdSharding


class ReshardMetadataTest(unittest.TestCase):
    def test_dtype_itemsize_metadata(self):
        """Check byte accounting without claiming custom-dtype transport support."""
        cases = (
            ("numpy_dtype", np.dtype("float32"), 4),
            ("numpy_compatible", SimpleNamespace(dtype=np.dtype("float32")), 4),
            (
                "none_itemsize",
                SimpleNamespace(dtype=np.dtype("float32"), itemsize=None),
                4,
            ),
            ("typed_key", jax.random.key(0, impl="unsafe_rbg").dtype, 16),
        )
        mesh = SimpleNamespace(
            jax_mesh=SimpleNamespace(is_multi_process=True), mpmd_dim=2
        )
        for name, dtype, itemsize in cases:
            with self.subTest(dtype=name):
                mpmd_array = SimpleNamespace(
                    dtype=dtype,
                    shape=(4,),
                    _mpmd_idxs=(0,),
                    _mpmd_local_sharding=SimpleNamespace(
                        shard_shape=lambda shape: shape
                    ),
                )
                # Stop after size collection, before constructing distributed arrays.
                with mock.patch(
                    "jaxpp.array._group_by_size_threshold",
                    side_effect=RuntimeError("size metadata checked"),
                ) as group:
                    with self.assertRaisesRegex(RuntimeError, "size metadata checked"):
                        mpmd_to_spmd_reshard(
                            mesh, [mpmd_array], [object()], threshold=256
                        )
                group.assert_called_once_with([4 * itemsize * mesh.mpmd_dim], 256)


class ReshardUtilsTest(jppdu.JaxDistributedTest):
    def _check_mpmd_reshard_result(
        self, mpmd_array, mpmd_mesh, mpmd_target_sharding, global_data
    ):
        process_index = jax.process_index()
        if process_index == 0:
            self.assertTrue(mpmd_array.is_partially_addressable)
            local_arr = mpmd_array.to_mpmd_local_array
            self.assertIsNotNone(local_arr)

            np.testing.assert_array_equal(np.array(local_arr), global_data)
            self.assertEqual(
                local_arr.sharding,
                filter_axes(mpmd_target_sharding, {mpmd_mesh.mpmd_axis_name}),
            )
        else:
            self.assertFalse(mpmd_array.is_partially_addressable)
            self.assertEqual(
                mpmd_array._mpmd_local_sharding.spec,
                filter_axes(mpmd_target_sharding, {mpmd_mesh.mpmd_axis_name}).spec,
            )

    @parameterized.expand(
        [
            ("mpmd_data_None", P(("mpmd", "data"), None)),
            ("None_model_mpmd", P(None, ("model", "mpmd"))),
            ("model_data", P("model", "data")),
            ("None_model_data", P(None, ("model", "data"))),
        ]
    )
    def test_spmd_to_mpmd_reshard(self, name, spmd_pspec):
        process_count = jax.process_count()
        local_device_count = jax.local_device_count()

        print(f"{local_device_count=}")
        print(f"{spmd_pspec=}")

        devices = np.array(jax.devices()).reshape(process_count, 2, -1)
        jax_mesh = jax.sharding.Mesh(devices, axis_names=("mpmd", "data", "model"))
        mpmd_mesh = MpmdMesh(jax_mesh, mpmd_axis_name="mpmd")

        print(f"{jax_mesh.shape=}")
        print(f"{mpmd_mesh.mpmd_dim=}")

        global_shape = (8, 16)
        global_data = np.arange(np.prod(global_shape), dtype=np.float32).reshape(
            global_shape
        )

        spmd_sharding = jax.sharding.NamedSharding(jax_mesh, spmd_pspec)
        print(f"{spmd_sharding.spec=}")

        spmd_array = jax.make_array_from_callback(
            global_shape, spmd_sharding, lambda idx: global_data[idx]
        )

        print(f"{spmd_array.shape=}")
        print(f"{spmd_array.sharding=}")
        num_shards = len(spmd_array.addressable_shards)
        print(f"{num_shards=}")

        target_mesh_ids = {0}
        submesh = mpmd_mesh.mpmd_submesh(list(target_mesh_ids)).jax_mesh
        mpmd_target_sharding = jax.sharding.NamedSharding(submesh, spmd_pspec)
        dist_sharding = MpmdSharding(
            mpmd_mesh=mpmd_mesh, mesh_ids=target_mesh_ids, spec=spmd_pspec
        )

        [mpmd_array] = spmd_to_mpmd_reshard(mpmd_mesh, [spmd_array], [dist_sharding])
        self.assertTrue(spmd_array.is_deleted())

        self._check_mpmd_reshard_result(
            mpmd_array, mpmd_mesh, mpmd_target_sharding, global_data
        )

        [_foo] = mpmd_to_spmd_reshard(mpmd_mesh, [mpmd_array], [spmd_sharding])

    def test_reshard_with_grouping(self):
        """Test spmd_to_mpmd and mpmd_to_spmd reshard with multiple arrays and forced grouping."""
        process_count = jax.process_count()
        process_index = jax.process_index()

        devices = np.array(jax.devices()).reshape(process_count, 2, -1)
        jax_mesh = jax.sharding.Mesh(devices, axis_names=("mpmd", "data", "model"))
        mpmd_mesh = MpmdMesh(jax_mesh, mpmd_axis_name="mpmd")

        # Create multiple arrays of different sizes to test grouping
        shapes = [(8, 16), (16, 32), (4, 8), (32, 64)]
        spmd_pspec = P("data", "model")
        spmd_sharding = jax.sharding.NamedSharding(jax_mesh, spmd_pspec)

        spmd_arrays = []
        global_datas = []
        for shape in shapes:
            global_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            global_datas.append(global_data)
            spmd_array = jax.make_array_from_callback(
                shape, spmd_sharding, lambda idx, gd=global_data: gd[idx]
            )
            spmd_arrays.append(spmd_array)

        # Create different target mesh_ids for each array:
        # Array 0: {0}, Array 1: empty, Array 2: {1}, Array 3: {0, 1}
        dist_shardings = [
            MpmdSharding(mpmd_mesh=mpmd_mesh, mesh_ids={0}, spec=spmd_pspec),
            MpmdSharding(mpmd_mesh=mpmd_mesh, mesh_ids=set(), spec=spmd_pspec),
            MpmdSharding(mpmd_mesh=mpmd_mesh, mesh_ids={1}, spec=spmd_pspec),
            MpmdSharding(mpmd_mesh=mpmd_mesh, mesh_ids={0, 1}, spec=spmd_pspec),
        ]

        addressable_by_process = [{0}, {0}, {1}, {0, 1}]

        # Leads to "single element" groups
        small_threshold = 1
        print(f"Testing with {len(spmd_arrays)} arrays and threshold={small_threshold}")

        mpmd_arrays = spmd_to_mpmd_reshard(
            mpmd_mesh, spmd_arrays, dist_shardings, threshold=small_threshold
        )

        for arr in spmd_arrays:
            self.assertTrue(arr.is_deleted())

        for i, (mpmd_array, global_data) in enumerate(zip(mpmd_arrays, global_datas)):
            should_be_addressable = process_index in addressable_by_process[i]
            if should_be_addressable:
                self.assertTrue(mpmd_array.is_partially_addressable)
                local_arr = mpmd_array.to_mpmd_local_array
                self.assertIsNotNone(local_arr)
                np.testing.assert_array_equal(
                    np.array(local_arr), global_data, err_msg=f"Array {i} mismatch"
                )
            else:
                self.assertFalse(mpmd_array.is_partially_addressable)

        spmd_shardings = [spmd_sharding] * len(mpmd_arrays)
        result_arrays = mpmd_to_spmd_reshard(
            mpmd_mesh, mpmd_arrays, spmd_shardings, threshold=small_threshold
        )

        for i, (result_arr, global_data) in enumerate(zip(result_arrays, global_datas)):
            self.assertEqual(result_arr.shape, global_data.shape)

    @parameterized.expand(
        [
            (f"{impl}_{shape}_{owner}_{threshold}", impl, shape, owner, threshold)
            for impl in ("threefry2x32", "rbg", "unsafe_rbg")
            for shape in ("scalar", "batched")
            for owner in ("first", "last")
            for threshold in (1, 1024)
        ]
    )
    def test_typed_key_reshard_round_trip(self, name, impl, shape, owner, threshold):
        """Typed keys and numeric state survive ownership changes and grouping."""
        process_count = jax.process_count()
        devices = np.array(jax.devices()).reshape(process_count, -1)
        mesh = jax.sharding.Mesh(devices, ("mpmd", "data"))
        mpmd_mesh = MpmdMesh(mesh, mpmd_axis_name="mpmd")
        key_owner = 0 if owner == "first" else process_count - 1
        batch_size = 2 * jax.local_device_count()

        def make_key():
            key = jax.random.key(17, impl=impl)
            return jax.random.split(key, batch_size) if shape == "batched" else key

        key_spec = P("data") if shape == "batched" else P()
        key_sharding = jax.sharding.NamedSharding(mesh, key_spec)
        replicated_sharding = jax.sharding.NamedSharding(mesh, P())
        param_sharding = jax.sharding.NamedSharding(mesh, P("data"))
        expected_key_data = np.asarray(jax.random.key_data(make_key())).copy()
        expected_param = np.arange(batch_size, dtype=np.float32)

        state = {
            "key": jax.jit(make_key, out_shardings=key_sharding)(),
            "step": jax.jit(
                lambda: jax.numpy.array(7, dtype=jax.numpy.uint32),
                out_shardings=replicated_sharding,
            )(),
            "param": jax.jit(
                lambda: jax.numpy.arange(batch_size, dtype=jax.numpy.float32),
                out_shardings=param_sharding,
            )(),
        }
        key_dtype, key_shape = state["key"].dtype, state["key"].shape
        shardings = {
            "key": key_sharding,
            "step": replicated_sharding,
            "param": param_sharding,
        }
        mpmd_shardings = {
            "key": MpmdSharding(mpmd_mesh, {key_owner}, key_spec),
            "step": MpmdSharding(mpmd_mesh, {0}, P()),
            "param": MpmdSharding(mpmd_mesh, {0}, P("data")),
        }
        mpmd_state = spmd_to_mpmd_reshard(
            mpmd_mesh, state, mpmd_shardings, threshold=threshold
        )
        self.assertTrue(all(array.is_deleted() for array in state.values()))
        self.assertEqual(
            mpmd_state["key"].is_partially_addressable, jax.process_index() == key_owner
        )

        result = mpmd_to_spmd_reshard(
            mpmd_mesh, mpmd_state, shardings, threshold=threshold
        )
        self.assertEqual(result["key"].dtype, key_dtype)
        self.assertEqual(result["key"].shape, key_shape)
        for name, array in result.items():
            self.assertTrue(
                array.sharding.is_equivalent_to(shardings[name], array.ndim)
            )
            for shard in array.addressable_shards:
                if name == "key":
                    np.testing.assert_array_equal(
                        np.asarray(jax.random.key_data(shard.data)),
                        expected_key_data[shard.index],
                    )
                elif name == "param":
                    np.testing.assert_array_equal(
                        np.asarray(shard.data), expected_param[shard.index]
                    )
                else:
                    self.assertEqual(int(shard.data), 7)

    @parameterized.expand(
        [
            ("data_None", P("data", None)),
            ("None_model", P(None, "model")),
            ("model_data", P("model", "data")),
            ("None_model_data", P(None, ("model", "data"))),
        ]
    )
    def test_spmd_to_mpmd_reshard_different_mesh_shapes(self, name, spmd_pspec):
        process_count = jax.process_count()
        local_device_count = jax.local_device_count()

        print(f"{local_device_count=}")
        print(f"{spmd_pspec=}")

        all_devices = np.array(jax.devices())
        model_axis_size = min(2, local_device_count)

        source_jax_mesh = jax.sharding.Mesh(
            all_devices.reshape(1, -1, model_axis_size), ("mpmd", "data", "model")
        )

        target_jax_mesh = jax.sharding.Mesh(
            all_devices.reshape(process_count, 2, -1), ("mpmd", "data", "model")
        )
        mpmd_mesh = MpmdMesh(target_jax_mesh, mpmd_axis_name="mpmd")

        print(f"{source_jax_mesh.shape=}")
        print(f"{target_jax_mesh.shape=}")
        print(f"{mpmd_mesh.mpmd_dim=}")

        global_shape = (8, 16)
        global_data = np.arange(np.prod(global_shape), dtype=np.float32).reshape(
            global_shape
        )

        source_spmd_sharding = jax.sharding.NamedSharding(source_jax_mesh, spmd_pspec)
        print(f"{source_spmd_sharding.spec=}")

        spmd_array = jax.make_array_from_callback(
            global_shape, source_spmd_sharding, lambda idx: global_data[idx]
        )

        print(f"{spmd_array.shape=}")
        print(f"{spmd_array.sharding=}")
        print(f"{len(spmd_array.addressable_shards)=}")

        target_mesh_ids = {0}
        submesh = mpmd_mesh.mpmd_submesh(list(target_mesh_ids)).jax_mesh
        mpmd_target_sharding = jax.sharding.NamedSharding(submesh, spmd_pspec)
        dist_sharding = MpmdSharding(
            mpmd_mesh=mpmd_mesh, mesh_ids=target_mesh_ids, spec=spmd_pspec
        )

        [mpmd_array] = spmd_to_mpmd_reshard(mpmd_mesh, [spmd_array], [dist_sharding])
        self.assertTrue(spmd_array.is_deleted())

        self._check_mpmd_reshard_result(
            mpmd_array, mpmd_mesh, mpmd_target_sharding, global_data
        )

        target_spmd_sharding = jax.sharding.NamedSharding(target_jax_mesh, spmd_pspec)
        [_foo] = mpmd_to_spmd_reshard(mpmd_mesh, [mpmd_array], [target_spmd_sharding])


if __name__ == "__main__":
    jppdu.distributed_main(unittest.main)
