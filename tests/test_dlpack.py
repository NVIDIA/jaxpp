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

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.dlpack import to_dlpack
from parameterized import parameterized

from jaxpp.dlpack import capsule_name, dlpack_nccl_args

_libcudart = ctypes.CDLL("libcudart.so")
_libcudart.cudaMemcpy.argtypes = [
    ctypes.c_void_p,  # dst
    ctypes.c_void_p,  # src
    ctypes.c_size_t,  # count (bytes)
    ctypes.c_int,  # kind
]
_libcudart.cudaMemcpy.restype = ctypes.c_int
_cudaMemcpyDeviceToHost = 2


def cuda_memcpy_to_host(device_ptr: int, num_bytes: int) -> bytes:
    host_buffer = (ctypes.c_uint8 * num_bytes)()
    err = _libcudart.cudaMemcpy(
        host_buffer, device_ptr, num_bytes, _cudaMemcpyDeviceToHost
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpy failed with error {err}")
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
        data_ptr, count, nccl_dtype = dlpack_nccl_args(capsule)

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
        self.assertIn("Unsupported dtype", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
