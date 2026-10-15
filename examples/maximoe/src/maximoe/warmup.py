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

"""Initialize CUDA tasks before posting pipeline transfers."""

import logging
import time

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jaxpp import jax_compat as jc
from jaxpp.core import GlobalMpmdFunction, ScalarMpmdFunction
from jaxpp.jax_primitives import PjitKwargs, apply_task, task_p
from jaxpp.mesh import _resolve_placement

logger = logging.getLogger(__name__)


def warmup_tasks(step: ScalarMpmdFunction | GlobalMpmdFunction):
    """Initialize task specializations using disposable inputs before transfers.

    Stage participants execute their tasks in schedule order. The final global
    barrier prevents a fast stage from posting receives during another stage's
    initialization. Real train state must not exist yet because task donation
    and temporary buffers can otherwise double the startup memory requirement.
    """
    jaxpr = (
        step.local_jaxpr if isinstance(step, ScalarMpmdFunction) else step.closed_jaxpr
    )
    tasks = {}
    for eqn in jaxpr.eqns:
        if eqn.primitive is not task_p:
            continue
        p = eqn.params
        _, mesh = _resolve_placement(
            step.mpmd_mesh, p["mpmd_idx"], name="task mpmd_idx"
        )
        params = PjitKwargs.from_task(
            call_jaxpr=p["call_jaxpr"],
            in_shardings=p["in_shardings"],
            out_shardings=p["out_shardings"],
            donate_invars=p["donate_invars"],
            ctx_mesh=mesh,
            name=p.get("compile_name") or p["task_name"],
        )
        tasks.setdefault(params, None)
    for params in tasks:
        # Refs allocated inside the task are disposable, like its dummy inputs.
        effects = params.jaxpr.effects - {jc.core.internal_mutable_array_effect}
        if effects:
            raise NotImplementedError(
                f"task warmup cannot replay effects in {params.name}: {effects}"
            )
    multihost_utils.sync_global_devices("maximoe-task-warmup-start")
    started = time.perf_counter()
    with step.mpmd_mesh:
        for index, params in enumerate(tasks, 1):
            logger.info("Warming task %d/%d: %s", index, len(tasks), params.name)
            with jax.set_mesh(params.ctx_mesh):
                # Later invocations also exercise command-buffer recording.
                for iteration in range(3):
                    inputs = [
                        jax.device_put(
                            jnp.zeros(aval.shape, aval.dtype, out_sharding=sharding),
                            sharding,
                        )
                        for aval, sharding in zip(
                            params.jaxpr.in_avals, params.in_shardings, strict=True
                        )
                    ]
                    jax.block_until_ready(inputs)
                    outputs = apply_task(jc.jit_p, *inputs, params=params)
                    jax.block_until_ready(outputs)
                    del inputs, outputs
                    logger.info("Warmed %s invocation %d/3", params.name, iteration + 1)
    jax.effects_barrier()
    multihost_utils.sync_global_devices("maximoe-task-warmup-done")
    logger.info(
        "Task warmup completed before pipeline communication in %.3f seconds",
        time.perf_counter() - started,
    )
