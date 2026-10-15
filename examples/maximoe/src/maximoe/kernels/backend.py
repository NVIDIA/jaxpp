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

"""Implementation selection shared by the MaxiMoE kernel packages.

Every operation ships an ``xla`` module written with ``jax.numpy`` and
``jax.lax`` next to its Mosaic GPU kernels. The ``xla`` module is the
specification: it is small, runs on any backend, and JAX differentiates it
without custom VJPs. Kernel tests compare each Mosaic implementation against
it, and the whole model can run on it for debugging.

Operations take a keyword-only implementation argument. None selects the local
platform default; resolved choices are static parameters of their HiJAX primitives.
"""

import enum

import jax

from maximoe.kernels import gpu


class Implementation(enum.StrEnum):
    """Kernel implementations selectable for every MaxiMoE operation."""

    XLA = "xla"
    SM90 = gpu.GpuArchitecture.SM90.value
    SM100 = gpu.GpuArchitecture.SM100.value


def resolve(implementation: str | None = None) -> Implementation:
    """Validate an explicit choice, or select the local platform's default."""
    if implementation is not None:
        return Implementation(implementation)
    return (
        Implementation(gpu.architecture().value)
        if jax.default_backend() == "gpu"
        else Implementation.XLA
    )
