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

# NOTE: Keep dlpack in sync when changing jax versions
# https://raw.githubusercontent.com/dmlc/dlpack/2a7e9f1256ddc48186c86dff7a00e189b47e5310/apps/numpy_dlpack/dlpack/dlpack.py

import ctypes
import math

_c_str_dltensor = b"dltensor"


class DLDeviceType(ctypes.c_int):
    """The enum that encodes the type of the device where
    DLTensor memory is allocated.
    """
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLCUDAManaged = 13
    kDLOneAPI = 14

    def __str__(self):
        return {
            self.kDLCPU : "CPU",
            self.kDLCUDA: "CUDA",
            self.kDLCUDAHost: "CUDAHost",
            self.kDLOpenCL: "OpenCL",
            self.kDLVulkan: "Vulkan",
            self.kDLMetal: "Metal",
            self.kDLVPI: "VPI",
            self.kDLROCM: "ROCM",
            self.kDLROCMHost: "ROMCHost",
            self.kDLCUDAManaged: "CUDAManaged",
            self.kDLOneAPI: "oneAPI",
            }[self.value]


class DLDevice(ctypes.Structure):
    """Represents the device where DLTensor memory is allocated.
    The device is represented by the pair of fields:
       device_type: DLDeviceType
       device_id: c_int
    """
    _fields_ = [
        ("device_type", DLDeviceType),
        ("device_id", ctypes.c_int),
    ]

# https://github.com/dmlc/dlpack/blob/6ea9b3eb64c881f614cd4537f95f0e125a35555c/include/dlpack/dlpack.h#L141-L182
class DLDataTypeCode(ctypes.c_uint8):
    """An integer that encodes the category of DLTensor elements' data type."""
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6
    # FP8 data types
    kDLFloat8_e3m4 = 7
    kDLFloat8_e4m3 = 8
    kDLFloat8_e4m3b11fnuz = 9
    kDLFloat8_e4m3fn = 10
    kDLFloat8_e4m3fnuz = 11
    kDLFloat8_e5m2 = 12
    kDLFloat8_e5m2fnuz = 13
    kDLFloat8_e8m0fnu = 14
    # FP6 data types
    kDLFloat6_e2m3fn = 15
    kDLFloat6_e3m2fn = 16
    # FP4 data types
    kDLFloat4_e2m1fn = 17

    def __str__(self):
        return {
            self.kDLInt: "int",
            self.kDLUInt: "uint",
            self.kDLFloat: "float",
            self.kDLOpaqueHandle: "void_p",
            self.kDLBfloat: "bfloat",
            self.kDLComplex: "complex",
            self.kDLBool: "bool",
            self.kDLFloat8_e3m4: "float8_e3m4",
            self.kDLFloat8_e4m3: "float8_e4m3",
            self.kDLFloat8_e4m3b11fnuz: "float8_e4m3b11fnuz",
            self.kDLFloat8_e4m3fn: "float8_e4m3fn",
            self.kDLFloat8_e4m3fnuz: "float8_e4m3fnuz",
            self.kDLFloat8_e5m2: "float8_e5m2",
            self.kDLFloat8_e5m2fnuz: "float8_e5m2fnuz",
            self.kDLFloat8_e8m0fnu: "float8_e8m0fnu",
            self.kDLFloat6_e2m3fn: "float6_e2m3fn",
            self.kDLFloat6_e3m2fn: "float6_e3m2fn",
            self.kDLFloat4_e2m1fn: "float4_e2m1fn",
        }[self.value]


class DLDataType(ctypes.Structure):
    """Descriptor of data type for elements of DLTensor.
    The data type is described by a triple, `DLDataType.type_code`,
    `DLDataType.bits`, and `DLDataType.lanes`.

    The element is understood as packed `lanes` repetitions of
    elements from `type_code` data-category of width `bits`.
    """
    _fields_ = [
        ("type_code", DLDataTypeCode),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]
    TYPE_MAP = {
        "bool": (DLDataTypeCode.kDLBool, 8, 1),
        "int8": (DLDataTypeCode.kDLInt, 8, 1),
        "int16": (DLDataTypeCode.kDLInt, 16, 1),
        "int32": (DLDataTypeCode.kDLInt, 32, 1),
        "int64": (DLDataTypeCode.kDLInt, 64, 1),
        "uint8": (DLDataTypeCode.kDLUInt, 8, 1),
        "uint16": (DLDataTypeCode.kDLUInt, 16, 1),
        "uint32": (DLDataTypeCode.kDLUInt, 32, 1),
        "uint64": (DLDataTypeCode.kDLUInt, 64, 1),
        "float16": (DLDataTypeCode.kDLFloat, 16, 1),
        "bfloat16": (DLDataTypeCode.kDLBfloat, 16, 1),
        "float32": (DLDataTypeCode.kDLFloat, 32, 1),
        "float64": (DLDataTypeCode.kDLFloat, 64, 1),
        "complex64": (DLDataTypeCode.kDLComplex, 64, 1),
        "complex128": (DLDataTypeCode.kDLComplex, 128, 1),
        # FP4 types
        "float4_e2m1fn": (DLDataTypeCode.kDLFloat4_e2m1fn, 4, 1),
        # FP8 types
        "float8_e3m4": (DLDataTypeCode.kDLFloat8_e3m4, 8, 1),
        "float8_e4m3": (DLDataTypeCode.kDLFloat8_e4m3, 8, 1),
        "float8_e4m3b11fnuz": (DLDataTypeCode.kDLFloat8_e4m3b11fnuz, 8, 1),
        "float8_e4m3fn": (DLDataTypeCode.kDLFloat8_e4m3fn, 8, 1),
        "float8_e4m3fnuz": (DLDataTypeCode.kDLFloat8_e4m3fnuz, 8, 1),
        "float8_e5m2": (DLDataTypeCode.kDLFloat8_e5m2, 8, 1),
        "float8_e5m2fnuz": (DLDataTypeCode.kDLFloat8_e5m2fnuz, 8, 1),
        "float8_e8m0fnu": (DLDataTypeCode.kDLFloat8_e8m0fnu, 8, 1),
    }

    REV_MAP = {v: k for k, v in TYPE_MAP.items()}


class DLTensor(ctypes.Structure):
    """Structure describing strided layout of DLTensor.
    Fields are:
       data:  void pointer
       device: DLDevice
       ndim: number of indices needed to reference an
             element of the tensor
       dtype: data type descriptor
       shape: tuple with lengths of the corresponding
              tensor dimensions
       strides: tuple of numbers of array elements to
                step in each dimension when traversing
                the tensor
       byte_offset: data + byte_offset gives the address of
                tensor element with index (0,) * ndim
    """
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
    """Structure storing the pointer to the tensor descriptor,
    deleter callable for the tensor descriptor, and pointer to
    some additional data. These are stored in fields `dl_tensor`,
    `deleter`, and `manager_ctx`."""
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


DLManagedTensorPtr = ctypes.POINTER(DLManagedTensor)


class NcclDataType(ctypes.c_uint8):
    # https://github.com/NVIDIA/nccl/blob/1e0c869c39bb33f1034cb9920bd2a8a8406f04a3/src/nccl.h.in#L328-L341
    ncclInt8 = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclFloat64 = 8
    ncclBfloat16 = 9
    ncclFloat8e4m3 = 10
    ncclFloat8e5m2 = 11

    TYPE_MAP = {
        "bool": ncclUint8,
        "int8": ncclInt8,
        "uint8": ncclUint8,
        "int32": ncclInt32,
        "uint32": ncclUint32,
        "int64": ncclInt64,
        "uint64": ncclUint64,
        "float16": ncclFloat16,
        "float32": ncclFloat32,
        "float64": ncclFloat64,
        "bfloat16": ncclBfloat16,
        "float8_e4m3fn": ncclFloat8e4m3,
        "float8_e5m2": ncclFloat8e5m2,
    }


ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p

def capsule_name(cap):
    return ctypes.pythonapi.PyCapsule_GetName(ctypes.py_object(cap)).decode('utf-8')

RawDataPointer = ctypes.c_void_p


def dlpack_nccl_args(dla) -> tuple[RawDataPointer, int, NcclDataType]:
    dlmanaged_tensor_ptr = ctypes.cast(
        ctypes.pythonapi.PyCapsule_GetPointer(dla, _c_str_dltensor), DLManagedTensorPtr
    )
    dltensor = dlmanaged_tensor_ptr.contents.dl_tensor

    data_ptr = dltensor.data + dltensor.byte_offset

    nelems = math.prod(dltensor.shape[i] for i in range(dltensor.ndim))

    dtype_key = (
        dltensor.dtype.type_code.value,
        dltensor.dtype.bits,
        dltensor.dtype.lanes,
    )

    if dtype_key not in DLDataType.REV_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_key}")

    dtype_name = DLDataType.REV_MAP[dtype_key]
    if dtype_name not in NcclDataType.TYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    nccl_dtype = NcclDataType.TYPE_MAP[dtype_name]

    return RawDataPointer(data_ptr), nelems, NcclDataType(nccl_dtype)
