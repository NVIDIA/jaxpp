# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import jax.lib
import jax.numpy as jnp
import jaxlib.mlir.ir
import numpy as np
import pytest
from jax import core as jcore
from jax._src import sharding_impls
from jax._src.lib import cuda_versions
from jax.interpreters.pxla import MeshComputation
from jax.lib import xla_bridge
from jaxlib.mlir.ir import Module as ir_Module
from packaging.version import Version

from jaxpp.compilation import (
    SerializeableMeshComputation,
    pjit_to_serializeable_mesh_computation,
)
from jaxpp.types import SerializeableSharding


class _BaseCompilationTest(unittest.TestCase):
    N_DEVICES = 8

    def setUp(self):
        """Set up a basic environment before each test with 8 GPU devices."""
        xla_bridge.get_backend.cache_clear()

        actual_gpu_devices = cuda_versions.cuda_device_count()

        if actual_gpu_devices < 1:
            # Skip the test if no GPU are available
            pytest.skip()

        num_clients, remainder = divmod(self.N_DEVICES, actual_gpu_devices)
        if remainder > 0:
            num_clients += 1

        if Version(jax.__version__) <= Version("0.4.30"):
            jax.config.update("use_mock_gpu_client", True)
            jax.config.update("mock_num_gpus", num_clients)

        elif Version(jax.__version__) == Version("0.4.31"):
            jax.config.update("mock_num_processes", num_clients)

        else:
            jax.config.update("mock_num_gpu_processes", num_clients)

        self.devices = jax.devices()[: self.N_DEVICES]

        assert len(self.devices) == self.N_DEVICES

        self.mesh = jax.sharding.Mesh(np.array(self.devices).reshape(2, 4), ("x", "y"))
        self.array = jnp.ones((8, 8))


class TestSerializeableSharding(_BaseCompilationTest):
    def test_to_named_sharding_with_unspecified(self):
        """Test to_named_sharding when the input is UnspecifiedValue."""
        obj = SerializeableSharding(sharding_impls.UNSPECIFIED)

        result = obj.to_named_sharding(self.mesh)

        assert isinstance(result, sharding_impls.UnspecifiedValue)

    def test_to_named_sharding_with_valid_sharding(self):
        """Test to_named_sharding with valid device assignment."""
        sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec("x", "y")
        )
        obj = SerializeableSharding(sharding)

        result = obj.to_named_sharding(self.mesh)

        assert isinstance(result, jax.sharding.NamedSharding)
        assert len(result._device_assignment) == len(self.devices[: self.N_DEVICES])


class TestSerializeableMeshComputation(_BaseCompilationTest):
    def setUp(self):
        """Set up a basic environment before each test with 8 CPU devices."""
        super().setUp()

        # Set up a simple jaxpr for testing
        @jax.jit
        def simple_fn(x):
            return x * 2

        x = jnp.ones((8, 8))
        self.jaxpr = jax.make_jaxpr(simple_fn)(x)
        self.closed_jaxpr = jcore.ClosedJaxpr(self.jaxpr.jaxpr, self.jaxpr.literals)

    def _create_dummy_mesh_computation(self, mesh=None, devices=None):
        """Helper method to create a dummy MeshComputation."""
        if mesh is None:
            mesh = self.mesh

        if devices is None:
            devices = self.devices

        with jaxlib.mlir.ir.Context():
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
            return MeshComputation(
                name="dummy_computation",
                hlo=ir_Module.parse("module {}"),
                donated_invars=[False],
                in_shardings=[sharding],
                out_shardings=[sharding],
                device_assignment=devices,
                backend=jax.lib.xla_bridge.get_backend(),
                in_avals=[jcore.ShapedArray(self.array.shape, self.array.dtype)],
                out_avals=[jcore.ShapedArray(self.array.shape, self.array.dtype)],
                platforms=("cpu",),
            )

    def test_initialization(self):
        """Test the initialization of SerializeableMeshComputation."""
        # Dummy MeshComputation object
        sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec("x")
        )

        obj = pjit_to_serializeable_mesh_computation(
            closed_jaxpr=self.closed_jaxpr,
            in_axis_resources=[sharding],
            out_axis_resources=[sharding],
            name="test_mesh_computation",
            donate_invars=None,
            mesh=self.mesh,
            compiler_options={"xla_gpu_enable_async_collectives": True},
            use_pgle=False,
        )

        assert obj.name == "test_mesh_computation"
        assert obj.compiler_options is not None
        assert "xla_gpu_enable_async_collectives" in obj.compiler_options
        print(f"{obj.device_assignment_ids=}")
        assert len(obj.device_assignment_ids) == len(self.devices)

    def test_pjit_to_serializeable_mesh_computation(self):
        """Test the pjit_to_serializeable_mesh_computation function."""
        axis_resources = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec("x")
        )

        result = pjit_to_serializeable_mesh_computation(
            self.closed_jaxpr,
            in_axis_resources=(axis_resources,),
            out_axis_resources=(axis_resources,),
            mesh=self.mesh,
        )

        assert isinstance(result, SerializeableMeshComputation)
        assert len(result.in_shardings) == len(self.closed_jaxpr.in_avals)
        assert len(result.out_shardings) == len(self.closed_jaxpr.out_avals)


if __name__ == "__main__":
    unittest.main()
