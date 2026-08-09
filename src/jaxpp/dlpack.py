# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utilities for reading NCCL send/recv arguments from DLPack capsules."""

import ctypes
import math
from typing import TypeAlias

# A DLPack capsule is an opaque CPython PyCapsule. There is no narrower static
# type for it.
DLPackCapsule: TypeAlias = object

_C_STR_DLTENSOR = b"dltensor"
_C_STR_DLTENSOR_VERSIONED = b"dltensor_versioned"
_C_STR_USED_DLTENSOR = b"used_dltensor"
_C_STR_USED_DLTENSOR_VERSIONED = b"used_dltensor_versioned"

# DLPack major version whose DLManagedTensorVersioned layout we parse.
_DLPACK_SUPPORTED_MAJOR = 1

# DLDeviceType values that name GPU memory. NCCL needs a device-resident ptr.
_K_DLCUDA = 2
_K_DLCUDA_MANAGED = 13

# DLPack DLDataTypeCode values.
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
_K_DLINT = 0
_K_DLUINT = 1
_K_DLFLOAT = 2
_K_DLBfloat = 4
_K_DLCOMPLEX = 5
_K_DLBOOL = 6
_K_DLFLOAT8_E4M3FN = 10
_K_DLFLOAT8_E5M2 = 12

# (type_code, bits, lanes) -> dtype name.
_DLPACK_TO_NAME = {
    (_K_DLBOOL, 8, 1): "bool",
    (_K_DLINT, 8, 1): "int8",
    (_K_DLINT, 16, 1): "int16",
    (_K_DLINT, 32, 1): "int32",
    (_K_DLINT, 64, 1): "int64",
    (_K_DLUINT, 8, 1): "uint8",
    (_K_DLUINT, 16, 1): "uint16",
    (_K_DLUINT, 32, 1): "uint32",
    (_K_DLUINT, 64, 1): "uint64",
    (_K_DLFLOAT, 16, 1): "float16",
    (_K_DLBfloat, 16, 1): "bfloat16",
    (_K_DLFLOAT, 32, 1): "float32",
    (_K_DLFLOAT, 64, 1): "float64",
    (_K_DLCOMPLEX, 64, 1): "complex64",
    (_K_DLCOMPLEX, 128, 1): "complex128",
    (_K_DLFLOAT8_E4M3FN, 8, 1): "float8_e4m3fn",
    (_K_DLFLOAT8_E5M2, 8, 1): "float8_e5m2",
}

# dtype name -> ncclDataType_t value.
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
# These integers are exactly nccl4py's nccl.bindings.DataType values, so they
# feed straight into nccl.bindings.send/recv.
_NAME_TO_NCCL = {
    "bool": 1,  # ncclUint8
    "int8": 0,
    "uint8": 1,
    "int32": 2,
    "uint32": 3,
    "int64": 4,
    "uint64": 5,
    "float16": 6,
    "float32": 7,
    "float64": 8,
    "bfloat16": 9,
    "float8_e4m3fn": 10,
    "float8_e5m2": 11,
}


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLManagedTensorVersioned(ctypes.Structure):
    # DLPack 1.x layout: `version` + `flags` precede `dl_tensor`.
    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


_DLManagedTensorPtr = ctypes.POINTER(DLManagedTensor)
_DLManagedTensorVersionedPtr = ctypes.POINTER(DLManagedTensorVersioned)

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]


def capsule_name(capsule: DLPackCapsule) -> str:
    name = ctypes.pythonapi.PyCapsule_GetName(capsule)
    if name is None:
        raise ValueError("capsule has no name")
    return name.decode("utf-8")


def _capsule_is_valid(capsule: DLPackCapsule, name: bytes) -> bool:
    return bool(ctypes.pythonapi.PyCapsule_IsValid(capsule, name))


def _capsule_pointer(capsule: DLPackCapsule, name: bytes) -> int:
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, name)
    if not ptr:
        raise ValueError(f"DLPack capsule {name!r} holds a NULL pointer")
    return ptr


def dltensor_from_capsule(capsule: DLPackCapsule) -> DLTensor:
    """Return the DLTensor descriptor from a DLPack capsule.

    Supports both the legacy unversioned DLManagedTensor and the DLPack 1.x
    DLManagedTensorVersioned layout. This only reads the descriptor; it does
    not consume the capsule, so a receive capsule can still be handed to
    dlpack_managed_tensor_to_buffer afterwards.
    """
    if _capsule_is_valid(capsule, _C_STR_DLTENSOR):
        ptr = _capsule_pointer(capsule, _C_STR_DLTENSOR)
        return ctypes.cast(ptr, _DLManagedTensorPtr).contents.dl_tensor

    if _capsule_is_valid(capsule, _C_STR_DLTENSOR_VERSIONED):
        ptr = _capsule_pointer(capsule, _C_STR_DLTENSOR_VERSIONED)
        managed = ctypes.cast(ptr, _DLManagedTensorVersionedPtr).contents
        # `version` is the first field by DLPack design, so it is safe to read
        # before trusting the rest of the possibly future layout.
        if managed.version.major != _DLPACK_SUPPORTED_MAJOR:
            raise ValueError(
                f"unsupported DLPack version {managed.version.major}."
                f"{managed.version.minor}; this parser understands the v"
                f"{_DLPACK_SUPPORTED_MAJOR}.x DLManagedTensorVersioned layout"
            )
        return managed.dl_tensor

    if _capsule_is_valid(capsule, _C_STR_USED_DLTENSOR) or _capsule_is_valid(
        capsule, _C_STR_USED_DLTENSOR_VERSIONED
    ):
        raise ValueError(
            "DLPack capsule was already consumed; a DLPack capsule may only be "
            "consumed once"
        )

    raise ValueError(
        "expected an unconsumed DLPack capsule named b'dltensor' or "
        "b'dltensor_versioned'"
    )


def _shape(tensor: DLTensor) -> list[int]:
    return [tensor.shape[i] for i in range(tensor.ndim)]


def _validate_contiguous(tensor: DLTensor, shape: list[int]) -> None:
    # NCCL send/recv move count * contiguous elements. DLPack strides are in
    # elements. A NULL strides pointer means implicitly row-major/compact, while
    # JAX fills in explicit strides. Either way require C-contiguity so that
    # `count` elements from `data_ptr` are exactly the buffer. Size-1 dims carry
    # a don't-care stride and are skipped.
    if not tensor.strides:
        return

    expected = 1
    for i in range(tensor.ndim - 1, -1, -1):
        if tensor.shape[i] > 1 and tensor.strides[i] != expected:
            strides = [tensor.strides[j] for j in range(tensor.ndim)]
            raise ValueError(
                f"non-contiguous DLPack tensor (shape={shape}, strides={strides}); "
                "NCCL send/recv require contiguous data"
            )
        expected *= tensor.shape[i]


def dlpack_nccl_args(capsule: DLPackCapsule) -> tuple[int, int, int]:
    """Read (data_ptr, n_elements, ncclDataType) out of a DLPack capsule."""
    tensor = dltensor_from_capsule(capsule)

    if tensor.device.device_type not in (_K_DLCUDA, _K_DLCUDA_MANAGED):
        raise ValueError(
            f"expected a CUDA buffer (DLDeviceType {_K_DLCUDA}), got "
            f"device_type={tensor.device.device_type}"
        )

    shape = _shape(tensor)
    _validate_contiguous(tensor, shape)

    # `tensor.data` is a ctypes c_void_p field, which evaluates to None for a
    # NULL pointer. A live device buffer always has a non-NULL address; the
    # fallback only avoids None + int for degenerate NULL-data tensors.
    data_ptr = (tensor.data or 0) + tensor.byte_offset
    nelems = math.prod(shape)

    key = (tensor.dtype.type_code, tensor.dtype.bits, tensor.dtype.lanes)
    dtype_name = _DLPACK_TO_NAME.get(key)
    if dtype_name is None:
        raise ValueError(f"Unsupported DLPack dtype (type_code, bits, lanes)={key}")
    nccl_dtype = _NAME_TO_NCCL.get(dtype_name)
    if nccl_dtype is None:
        raise ValueError(f"dtype {dtype_name!r} has no NCCL equivalent")
    return data_ptr, nelems, nccl_dtype
