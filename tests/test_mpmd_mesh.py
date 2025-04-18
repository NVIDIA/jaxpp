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
import jax.experimental.topologies as topologies
import numpy as np
from jax.sharding import Mesh

from jaxpp.mesh import MpmdMesh

target_config = """
gpu_device_info {
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: 233472
  threads_per_core_limit: 2048
  core_count: 132
  fpus_per_core: 128
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: 3352320000000
  l2_cache_size: 52428800
  clock_rate_ghz: 1.98
  device_memory_size: 84942979072
  shared_memory_per_block_optin: 232448
  cuda_compute_capability {
    major: 9
  }
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info {
  major: 9
  minor: 6
}
device_description_str: "NVIDIA H100 80GB HBM3"
"""


class TestPutArg(unittest.TestCase):
    def setUp(self):
        """Reset and Set up the logger instance before each test."""
        jax.config
        topo = topologies.get_topology_desc(
            "topo",
            "cuda",
            target_config=target_config,
            topology="4x4x1",
        )
        self.devices = topo.devices

    def test_putarg_initialization(self):
        mesh = MpmdMesh(
            Mesh(
                np.array(self.devices).reshape((4, 2, 1, 2, 1)),
                ("stage", "data", "sequence", "tensor", "expert"),
            ),
            "stage",
            strict=False,
        )

        assert mesh.mpmd_axis == 0

        assert mesh.device_coords[self.devices[0]] == (0, 0, 0, 0, 0)
        assert mesh.device_coords[self.devices[1]] == (0, 0, 0, 1, 0)

        expected_shape = (2, 1, 2, 1)
        meshes = mesh.unstack
        for idx, m in enumerate(meshes):
            assert m.axis_names == ("data", "sequence", "tensor", "expert")
            assert m.devices.shape == expected_shape
            start_idx = idx * np.prod(expected_shape)
            assert (
                np.arange(start_idx, start_idx + np.prod(expected_shape)).reshape(
                    expected_shape
                )
                == m.device_ids
            ).all()


if __name__ == "__main__":
    unittest.main()
