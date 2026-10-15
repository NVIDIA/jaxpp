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

"""Token permutation and combination for MoE routing.

The ``xla`` module is the specification of every operation. ``mosaic_gpu``
implements the same functions as Mosaic GPU kernels for Hopper and Blackwell.
Dispatch follows :func:`maximoe.kernels.backend.resolve`.

The wrappers below are plain Python. They call the selected module's
module-level functions, which ``mosaic_gpu`` wraps in JIT boundaries once at
import, instead of creating new custom-VJP or JIT wrappers while tracing. JaxPP
needs that stability to share kernels across pipeline-stage traces.
"""

import types

import jax

from maximoe.kernels import backend
from maximoe.kernels.moe_routing import mosaic_gpu, xla


def get_implementation(
    selection: backend.Implementation | str | None = None,
) -> types.ModuleType:
    """Return the stable implementation module for ``selection``.

    ``None`` resolves through :func:`maximoe.kernels.backend.resolve`. Every
    Mosaic architecture maps to the architecture-neutral ``mosaic_gpu`` module.
    """
    selection = backend.resolve(selection)
    return xla if selection is backend.Implementation.XLA else mosaic_gpu


def combine(
    buffer: jax.Array,
    top_k_weights: jax.Array | None,
    positions: jax.Array,
    activations_in_float32: bool,
    group_sizes: jax.Array | None = None,
    *,
    implementation: str | None = None,
) -> jax.Array:
    """Sum each token's weighted expert rows; see ``xla.combine``."""
    return get_implementation(implementation).combine(
        buffer, top_k_weights, positions, activations_in_float32, group_sizes
    )


def combine_transpose(
    grad_out: jax.Array,
    positions: jax.Array,
    extent: int,
    cost_estimate_num_rows: int | None = None,
    group_sizes: jax.Array | None = None,
    *,
    implementation: str | None = None,
) -> jax.Array:
    """Scatter token cotangents to expert rows; see ``xla.combine_transpose``."""
    return get_implementation(implementation).combine_transpose(
        grad_out, positions, extent, cost_estimate_num_rows, group_sizes
    )


def weight_expert_rows(
    buffer: jax.Array,
    top_k_weights: jax.Array,
    positions: jax.Array,
    route_dot_values: jax.Array | None = None,
    cost_estimate_num_rows: int | None = None,
    *,
    implementation: str | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Scale live expert rows by route weights; see ``xla.weight_expert_rows``."""
    return get_implementation(implementation).weight_expert_rows(
        buffer, top_k_weights, positions, route_dot_values, cost_estimate_num_rows
    )


def permute(
    inputs: jax.Array,
    positions: jax.Array,
    packed_size: int,
    *,
    implementation: str | None = None,
) -> jax.Array:
    """Scatter token rows to their packed positions; see ``xla.permute``."""
    return get_implementation(implementation).permute(inputs, positions, packed_size)


def gather_sum(
    buffer: jax.Array, positions: jax.Array, *, implementation: str | None = None
) -> jax.Array:
    """Gather buffer rows and sum over routes; see ``xla.gather_sum``."""
    return get_implementation(implementation).gather_sum(buffer, positions)
