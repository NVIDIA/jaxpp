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

"""Training-step scaffolding for the MaxiMoE trainer tests.

Kept apart from :mod:`maximoe.test_util` so kernel tests don't import the
trainer and JaxPP.
"""

import functools

import jax
import jaxpp.api as jaxpp
import numpy as np

from maximoe import train


def local_arrays(value):
    """The per-stage local arrays of an MPMD array, as a list."""
    local = value.to_mpmd_local_array
    return local if isinstance(local, list) else [local]


def prepare_step(config, devices):
    """Compile a training step for ``config`` and build its state and batch.

    Returns the compiled step, the initialized train state, and a batch of
    deterministic tokens placed on ``devices``.
    """
    mpmd_mesh = jaxpp.MpmdMesh(train.make_mesh(config, devices), "stage")
    mesh = mpmd_mesh.lowering_mesh()
    excluded = {axis for axis in ("data", "fsdp", "expert") if mesh.shape[axis] == 1}
    params = train.filter_axes(train.parameter_shardings(config), excluded)
    specs = train.train_state_specs(config)
    partitions = train.filter_axes(train.train_state_shardings(config), excluded)
    shapes = train.shape_tree(
        specs, partitions, mesh, host_offload_regex=config.host_offload_regex
    )
    batch_shapes = train.batch_shapes(config, mesh)
    step = jaxpp.mpmd_jit_with_loop(
        functools.partial(train.train_step, config, params),
        mpmd_mesh=mpmd_mesh,
        in_shardings=(
            jax.tree.map(lambda x: x.sharding, shapes),
            jax.tree.map(lambda x: x.sharding, batch_shapes),
        ),
        out_shardings=(
            jax.tree.map(lambda x: x.sharding, shapes),
            train.Metrics(
                *(jax.NamedSharding(mesh, jax.P()) for _ in train.Metrics._fields)
            ),
        ),
        donate_argnums=(0,),
    ).compile(shapes, batch_shapes)
    (state_shardings, batch_shardings), _ = step.in_shardings
    state = train.init_train_state(config, mpmd_mesh, specs, state_shardings)
    tokens = (
        np.arange(np.prod(batch_shapes.inputs.shape), dtype=np.int32).reshape(
            batch_shapes.inputs.shape
        )
        % config.vocab_size
    )
    batch = train.put_batch(
        train.Batch(
            tokens, (tokens + 1) % config.vocab_size, np.ones(tokens.shape, np.float32)
        ),
        batch_shapes,
        batch_shardings,
        mpmd_mesh,
    )
    return step, jax.block_until_ready(state), batch
