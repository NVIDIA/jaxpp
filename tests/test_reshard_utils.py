# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import jax
import numpy as np
from jax.sharding import PartitionSpec as P
from parameterized import parameterized

import jaxpp.distributed_utils as jppdu
from jaxpp.array import filter_axes, mpmd_to_spmd_reshard, spmd_to_mpmd_reshard
from jaxpp.mesh import MpmdMesh
from jaxpp.types import DistributedSharding


class ReshardUtilsTest(jppdu.JaxDistributedTest):
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
        process_index = jax.process_index()
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
        dist_sharding = DistributedSharding(
            mesh_ids=target_mesh_ids, sharding=mpmd_target_sharding
        )

        [mpmd_array] = spmd_to_mpmd_reshard(mpmd_mesh, [spmd_array], [dist_sharding])
        self.assertTrue(spmd_array.is_deleted())

        if process_index == 0:
            self.assertTrue(mpmd_array.is_partially_addressable)
            local_arr = mpmd_array.to_mpmd_local_array
            self.assertIsNotNone(local_arr)
            print(f"{local_arr.shape=}")

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

        target_mesh_ids = {0}
        submesh = mpmd_mesh.mpmd_submesh(list(target_mesh_ids)).jax_mesh
        mpmd_target_sharding = jax.sharding.NamedSharding(submesh, spmd_pspec)
        dist_shardings = [
            DistributedSharding(mesh_ids=target_mesh_ids, sharding=mpmd_target_sharding)
            for _ in shapes
        ]

        # Leads to "single element" groups
        small_threshold = 1
        print(f"Testing with {len(spmd_arrays)} arrays and threshold={small_threshold}")

        mpmd_arrays = spmd_to_mpmd_reshard(
            mpmd_mesh, spmd_arrays, dist_shardings, threshold=small_threshold
        )

        for arr in spmd_arrays:
            self.assertTrue(arr.is_deleted())

        for i, (mpmd_array, global_data) in enumerate(zip(mpmd_arrays, global_datas)):
            if process_index == 0:
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


if __name__ == "__main__":
    jppdu.distributed_main(unittest.main)
