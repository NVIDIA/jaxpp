import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

import jax
import jax.numpy as jnp

from jaxpp.api import MpmdArray, MpmdMesh


class TestMpmdArray(unittest.TestCase):
    def setUp(self):
        assert jax.device_count() == 6

        self.mesh = jax.make_mesh((3, 2), ("stage", "model"))
        self.mpmd_mesh = MpmdMesh(self.mesh, "stage")
        self.sharding0 = jax.sharding.NamedSharding(
            self.mpmd_mesh.mpmd_submesh([0]).jax_mesh,
            jax.sharding.PartitionSpec("model"),
        )
        self.sharding2 = jax.sharding.NamedSharding(
            self.mpmd_mesh.mpmd_submesh([2]).jax_mesh,
            jax.sharding.PartitionSpec("model"),
        )
        self.array_at_submesh_0 = jax.device_put(jnp.zeros(2), self.sharding0)
        self.array_at_submesh_2 = jax.device_put(jnp.zeros(2), self.sharding2)

    def test_mpmd_non_replicated_array(self):
        # For non-replicated arrays
        mpmd_non_replicated_array1_p0 = MpmdArray(
            [self.array_at_submesh_0], self.mpmd_mesh, frozenset({0})
        )
        # The second process contains only the metadata, no physical arrays
        mpmd_non_replicated_array1_p1 = MpmdArray(
            [],
            self.mpmd_mesh,
            frozenset({0}),
            spec=self.array_at_submesh_0.sharding.spec,
            shape=self.array_at_submesh_0.shape,
            dtype=self.array_at_submesh_0.dtype,
        )

        assert not mpmd_non_replicated_array1_p0.is_mpmd_replicated
        assert (
            mpmd_non_replicated_array1_p0.sharding.mesh
            == self.mpmd_mesh.mpmd_submesh([0]).jax_mesh
        )
        assert (
            mpmd_non_replicated_array1_p0.to_mpmd_local_array.sharding.mesh
            == mpmd_non_replicated_array1_p0.sharding.mesh
        )
        # The second process contains only the metadata so it returns no array
        assert not mpmd_non_replicated_array1_p1.is_partially_addressable
        assert mpmd_non_replicated_array1_p1.to_mpmd_local_array is None

    def test_mpmd_replicated_array(self):
        # Multi-process case where the array is replicated across 0 and 2
        mpmd_replicated_array2_p0 = MpmdArray(
            [self.array_at_submesh_0], self.mpmd_mesh, frozenset({0, 2})
        )
        # Process 1 contains only metadata, no physical arrays
        mpmd_replicated_array2_p1 = MpmdArray(
            [],
            self.mpmd_mesh,
            frozenset({0, 2}),
            spec=self.array_at_submesh_0.sharding.spec,
            shape=self.array_at_submesh_0.shape,
            dtype=self.array_at_submesh_0.dtype,
        )
        mpmd_replicated_array2_p2 = MpmdArray(
            [self.array_at_submesh_2], self.mpmd_mesh, frozenset({0, 2})
        )
        assert mpmd_replicated_array2_p0.is_mpmd_replicated
        assert mpmd_replicated_array2_p1.is_mpmd_replicated
        assert not mpmd_replicated_array2_p1.is_partially_addressable
        assert mpmd_replicated_array2_p2.is_mpmd_replicated

        submesh_0_2 = self.mpmd_mesh.mpmd_submesh([0, 2])

        assert mpmd_replicated_array2_p0.sharding.mesh == submesh_0_2.jax_mesh
        assert mpmd_replicated_array2_p1.sharding.mesh == submesh_0_2.jax_mesh
        assert mpmd_replicated_array2_p2.sharding.mesh == submesh_0_2.jax_mesh

        assert (
            mpmd_replicated_array2_p0.to_mpmd_local_array.sharding.mesh
            != mpmd_replicated_array2_p0.sharding.mesh
        )
        assert mpmd_replicated_array2_p1.to_mpmd_local_array is None
        assert (
            mpmd_replicated_array2_p2.to_mpmd_local_array.sharding.mesh
            != mpmd_replicated_array2_p2.sharding.mesh
        )

        assert mpmd_replicated_array2_p0.first_mpmd_replica is not None
        assert mpmd_replicated_array2_p1.first_mpmd_replica is None
        assert mpmd_replicated_array2_p2.first_mpmd_replica is None

    def test_single_process_case(self):
        # Single process case
        mpmd_array = MpmdArray(
            [self.array_at_submesh_0, self.array_at_submesh_2],
            self.mpmd_mesh,
            frozenset({0, 2}),
        )
        assert mpmd_array.is_mpmd_replicated
        assert mpmd_array.sharding.mesh == self.mpmd_mesh.mpmd_submesh([0, 2]).jax_mesh
        # The sharding of to_mpmd_local_array
        assert isinstance(mpmd_array.to_mpmd_local_array, list)
        assert (
            mpmd_array.to_mpmd_local_array[0].sharding.mesh
            == self.mpmd_mesh.mpmd_submesh([0]).jax_mesh
        )
        assert (
            mpmd_array.to_mpmd_local_array[1].sharding.mesh
            == self.mpmd_mesh.mpmd_submesh([2]).jax_mesh
        )


if __name__ == "__main__":
    unittest.main()
