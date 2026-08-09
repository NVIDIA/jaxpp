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

import ctypes
import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from cuda.bindings import runtime as cuda_runtime
from jax._src.dlpack import to_dlpack
from jax.experimental.buffer_callback import buffer_callback
from jax.experimental.layout import Layout, with_layout_constraint
from parameterized import parameterized

from jaxpp.dlpack import (
    DLManagedTensorVersioned,
    DLPackVersion,
    capsule_name,
    dlpack_nccl_args,
    dltensor_from_capsule,
)


def cuda_memcpy_to_host(device_ptr: int, num_bytes: int) -> bytes:
    host_buffer = (ctypes.c_uint8 * num_bytes)()
    err = cuda_runtime.cudaMemcpy(
        host_buffer,
        device_ptr,
        num_bytes,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    if isinstance(err, tuple):
        err = err[0]
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy failed with {err!r}")
    return bytes(host_buffer)


class TestDlpackExport(unittest.TestCase):
    @parameterized.expand(
        [
            ("float32", jnp.float32, np.float32),
            ("bfloat16", jnp.bfloat16, ml_dtypes.bfloat16),
            ("float8_e4m3fn", jnp.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            ("float8_e5m2", jnp.float8_e5m2, ml_dtypes.float8_e5m2),
        ]
    )
    def test_dlpack_export(self, name, jax_dtype, np_dtype):
        x = jnp.array([1, 2, 3], dtype=jax_dtype)
        capsule = to_dlpack(x)
        self.assertEqual(capsule_name(capsule), "dltensor")
        data_ptr, count, _nccl_dtype = dlpack_nccl_args(capsule)

        self.assertEqual(count, 3)

        itemsize = np.dtype(np_dtype).itemsize
        raw_bytes = cuda_memcpy_to_host(data_ptr, count * itemsize)
        values = np.frombuffer(raw_bytes, dtype=np_dtype)

        np.testing.assert_array_equal(values, np.array([1, 2, 3], dtype=np_dtype))

    def test_unsupported_dtype(self):
        x = jnp.array([1, 2, 3], dtype=jnp.float8_e4m3b11fnuz)
        capsule = to_dlpack(x)
        with self.assertRaises(ValueError) as ctx:
            dlpack_nccl_args(capsule)
        self.assertIn("Unsupported DLPack dtype", str(ctx.exception))

    def test_jax_array_dlpack_rejects_non_standard_layout(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        x = with_layout_constraint(x, Layout((1, 0)))
        x.block_until_ready()

        capsule = x.__dlpack__()
        self.assertEqual(capsule_name(capsule), "dltensor")

        dltensor = dltensor_from_capsule(capsule)
        self.assertEqual([dltensor.shape[i] for i in range(dltensor.ndim)], [2, 3])
        self.assertIsNotNone(dltensor.strides)
        self.assertEqual([dltensor.strides[i] for i in range(dltensor.ndim)], [1, 2])

        with self.assertRaises(ValueError) as ctx:
            dlpack_nccl_args(capsule)
        self.assertIn("non-contiguous DLPack tensor", str(ctx.exception))

    def test_versioned_dlpack_capsule(self):
        seen = []

        def callback(_ctx, _out, x):
            capsule = x.__dlpack__(stream=None, max_version=(1, 0))
            seen.append((capsule_name(capsule), dlpack_nccl_args(capsule)))

        f = buffer_callback(
            callback, jax.ShapeDtypeStruct((0,), jnp.int32), has_side_effect=True
        )
        f(jnp.array([1, 2, 3], dtype=jnp.float32)).block_until_ready()

        [(name, (data_ptr, count, nccl_dtype))] = seen
        self.assertEqual(name, "dltensor_versioned")
        self.assertGreater(data_ptr, 0)
        self.assertEqual(count, 3)
        self.assertEqual(nccl_dtype, 7)

    @parameterized.expand([(0,), (2,)])
    def test_versioned_dlpack_capsule_rejects_mismatched_major(self, major):
        managed = DLManagedTensorVersioned()
        managed.version = DLPackVersion(major, 0)

        ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        capsule = ctypes.pythonapi.PyCapsule_New(
            ctypes.addressof(managed), b"dltensor_versioned", None
        )

        with self.assertRaisesRegex(
            ValueError, f"unsupported DLPack version {major}.0"
        ):
            dltensor_from_capsule(capsule)


if __name__ == "__main__":
    unittest.main()
