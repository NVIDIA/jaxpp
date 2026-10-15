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

"""GPU architecture detection for MaxiMoE kernels."""

import enum
import os

import jax
from jax.extend import backend

# Read once because JIT caches are not invalidated by later environment
# changes. The reservation is shared by multiple kernels.
MOSAIC_SM_MARGIN = int(os.environ.get("RAGGED_DOT_SM_MARGIN", "0"))


class GpuArchitecture(enum.StrEnum):
    """GPU architecture families supported by the MaxiMoE Mosaic kernels."""

    SM90 = "sm90"
    SM100 = "sm100"


def architecture(device: jax.Device | None = None) -> GpuArchitecture:
    """Return the supported architecture family for ``device``."""
    if device is None:
        device = backend.get_default_device()
    if device.platform != "gpu":
        raise NotImplementedError(
            f"MaxiMoE Mosaic kernels require a GPU, got platform {device.platform!r}."
        )

    compute_capability = device.compute_capability
    capability = float(compute_capability)
    if 9.0 <= capability < 10.0:
        return GpuArchitecture.SM90
    if 10.0 <= capability < 11.0:
        return GpuArchitecture.SM100
    raise NotImplementedError(
        "MaxiMoE Mosaic kernels support SM90 and SM100 GPUs, got "
        f"{device.device_kind!r} with compute capability {compute_capability}."
    )
