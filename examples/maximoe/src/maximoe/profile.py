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

"""Production-routed DeepSeek V3 MoE profile for EP8 and folded EP16."""

import argparse
import dataclasses
import functools
import logging
import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging as absl_logging
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding

from maximoe.train import (
    Config,
    MoEParams,
    filter_axes,
    layer_remat_policy,
    log_memory_stats,
    make_mesh,
    parameter_shardings,
    parameter_specs,
    routed_moe_apply,
)

GLOBAL_TOKENS = 32_768
NUM_STEPS = 15
WARMUP_STEPS = 3
logger = logging.getLogger("maximoe.profile")


class ArraySignature(NamedTuple):
    finite: jax.Array
    total: jax.Array
    squared_total: jax.Array
    maximum: jax.Array


class NumericalSignatures(NamedTuple):
    loss: jax.Array
    output: ArraySignature
    input_grad: ArraySignature
    gate_grad: ArraySignature
    wi_grad: ArraySignature
    wo_grad: ArraySignature


def make_config(
    ep: int,
    profile_dir: str,
    num_layers: int = 1,
    *,
    implementation: str | None = None,
    use_ring_of_experts: bool = False,
) -> Config:
    if ep not in (8, 16):
        raise ValueError("the folded MoE profile supports only EP8 and EP16")
    max_layers = 8 if ep == 8 else 15
    if not 1 <= num_layers <= max_layers:
        raise ValueError(f"EP{ep} supports between 1 and {max_layers} layers")
    return dataclasses.replace(
        Config(),
        # This profiler owns compute weights only; it has no optimizer masters.
        param_dtype=jnp.bfloat16,
        kernel_implementation=implementation,
        pp=1,
        data=1,
        fsdp=1,
        ep=ep,
        schedule="1f1b",
        num_pipeline_repeats=1,
        pipeline_stage_sizes=None,
        fuse_steady_state=False,
        use_ring_of_experts=use_ring_of_experts,
        per_device_batch_size=1,
        num_microbatches=1,
        seq_length=GLOBAL_TOKENS // ep,
        num_train_steps=NUM_STEPS,
        profile_dir=profile_dir,
        num_layers=num_layers,
        num_dense_layers=0,
        num_routing_groups=1,
        num_selected_groups=1,
        moe_aux_loss_weight=0.0,
        router_bias_update_rate=0.0,
        mtp_num_layers=0,
    )


def calculate_routed_training_flops_per_device(config: Config) -> float:
    tokens = GLOBAL_TOKENS // config.ep
    return (
        3
        * config.num_layers
        * (
            2 * tokens * config.emb_dim * config.num_experts
            + 6
            * tokens
            * config.emb_dim
            * config.moe_mlp_dim
            * config.num_experts_per_tok
        )
    )


def profile_random_keys(
    seed: int, num_layers: int
) -> tuple[tuple[tuple[jax.Array, ...], ...], jax.Array, jax.Array]:
    root_key = jax.random.key(seed)
    # Separate fold-in domains keep canonical arrays stable across stack depths.
    parameter_key = jax.random.fold_in(root_key, 0)
    layer_keys = []
    for layer in range(num_layers):
        layer_key = jax.random.fold_in(parameter_key, layer)
        layer_keys.append(
            tuple(jax.random.fold_in(layer_key, component) for component in range(3))
        )
    return (
        tuple(layer_keys),
        jax.random.fold_in(root_key, 1),
        jax.random.fold_in(root_key, 2),
    )


@functools.partial(jax.checkpoint, policy=layer_remat_policy, prevent_cse=True)
def checkpointed_routed_moe_apply(
    config: Config,
    gate: jax.Array,
    wi: jax.Array,
    wo: jax.Array,
    router_bias: jax.Array,
    inputs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return routed_moe_apply(config, gate, wi, wo, router_bias, inputs)


def profile_step(
    config: Config,
    gates: tuple[jax.Array, ...],
    wis: tuple[jax.Array, ...],
    wos: tuple[jax.Array, ...],
    inputs: jax.Array,
    cotangent: jax.Array,
    router_biases: tuple[jax.Array, ...],
):
    def loss_fn(gates, wis, wos, inputs):
        output = inputs
        aux_loss = jnp.zeros((), dtype=jnp.float32)
        counts = jnp.zeros((config.num_experts,), dtype=jnp.int32)
        for gate, wi, wo, router_bias in zip(
            gates, wis, wos, router_biases, strict=True
        ):
            routed, layer_aux_loss, layer_counts = checkpointed_routed_moe_apply(
                config, gate, wi, wo, router_bias, output
            )
            # The residual keeps later routers at the scale seen in production.
            output = output + routed
            aux_loss += layer_aux_loss
            counts += layer_counts
        loss = jnp.sum(
            output.astype(jnp.float32) * cotangent.astype(jnp.float32)
        ) / jnp.asarray(GLOBAL_TOKENS, dtype=jnp.float32)
        return loss + aux_loss, (output, counts)

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3), has_aux=True)(
        gates, wis, wos, inputs
    )


def numerical_signatures(result: Any) -> NumericalSignatures:
    (loss, (output, counts)), (gate_grads, wi_grads, wo_grads, input_grad) = result
    num_layers = len(gate_grads)
    expected_specs = (
        (jax.P(), (jax.P("expert", None, None), jax.P(None))),
        (
            (jax.P(None, None, unreduced={"expert"}),) * num_layers,
            (jax.P("expert", None, None),) * num_layers,
            (jax.P("expert", None, None),) * num_layers,
            jax.P("expert", None, None),
        ),
    )
    actual_specs = jax.tree.map(lambda value: jax.typeof(value).sharding.spec, result)
    if actual_specs != expected_specs:
        raise ValueError(
            f"unexpected routed profile result shardings: {actual_specs!r}"
        )

    gate_grads = tuple(
        jax.reshard(grad, jax.typeof(grad).sharding.update(spec=jax.P()))
        for grad in gate_grads
    )

    def signature(values: Any) -> ArraySignature:
        values = tuple(value.astype(jnp.float32) for value in jax.tree.leaves(values))
        return ArraySignature(
            finite=jnp.stack([jnp.all(jnp.isfinite(value)) for value in values]).all(),
            total=sum(
                (jnp.sum(value) for value in values),
                start=jnp.zeros((), dtype=jnp.float32),
            ),
            squared_total=sum(
                (jnp.sum(jnp.square(value)) for value in values),
                start=jnp.zeros((), dtype=jnp.float32),
            ),
            maximum=jnp.stack([jnp.max(jnp.abs(value)) for value in values]).max(),
        )

    return NumericalSignatures(
        loss=loss,
        output=signature(output),
        input_grad=signature(input_grad),
        gate_grad=signature(gate_grads),
        wi_grad=signature(wi_grads),
        wo_grad=signature(wo_grads),
    )


def initialize_profile_arrays(config: Config, mesh: jax.sharding.Mesh):
    specs = tuple(layer.ffn for layer in parameter_specs(config).layers)
    shardings = filter_axes(
        tuple(layer.ffn for layer in parameter_shardings(config).layers),
        {"data", "fsdp"},
    )
    if not all(isinstance(spec, MoEParams) for spec in specs) or not all(
        isinstance(sharding, MoEParams) for sharding in shardings
    ):
        raise ValueError("the profile layer must be a routed MoE layer")

    layer_keys, input_key, cotangent_key = profile_random_keys(
        config.param_seed, config.num_layers
    )
    gates = tuple(
        spec.gate.initializer(
            keys[0],
            spec.gate.shape,
            spec.gate.dtype,
            NamedSharding(mesh, sharding.gate),
        )
        for spec, sharding, keys in zip(specs, shardings, layer_keys, strict=True)
    )
    wis = tuple(
        spec.wi.initializer(
            keys[1], spec.wi.shape, spec.wi.dtype, NamedSharding(mesh, sharding.wi)
        )
        for spec, sharding, keys in zip(specs, shardings, layer_keys, strict=True)
    )
    wos = tuple(
        spec.wo.initializer(
            keys[2], spec.wo.shape, spec.wo.dtype, NamedSharding(mesh, sharding.wo)
        )
        for spec, sharding, keys in zip(specs, shardings, layer_keys, strict=True)
    )

    flat_sharding = NamedSharding(mesh, jax.P("expert", None))
    activation_sharding = NamedSharding(mesh, jax.P("expert", None, None))
    flat_shape = (GLOBAL_TOKENS, config.emb_dim)
    inputs = jax.random.normal(
        input_key, flat_shape, config.dtype, out_sharding=flat_sharding
    )
    cotangent = jax.random.normal(
        cotangent_key, flat_shape, config.dtype, out_sharding=flat_sharding
    )
    activation_shape = (config.ep, config.seq_length, config.emb_dim)
    inputs = jax.reshard(inputs.reshape(activation_shape), activation_sharding)
    cotangent = jax.reshard(cotangent.reshape(activation_shape), activation_sharding)
    router_biases = tuple(
        jnp.zeros(
            (config.num_experts,),
            dtype=jnp.float32,
            out_sharding=NamedSharding(mesh, jax.P()),
        )
        for _ in range(config.num_layers)
    )
    return gates, wis, wos, inputs, cotangent, router_biases


def log_numerical_results(config: Config, result: Any) -> None:
    logger.info(
        "Materializing global numerical diagnostics outside the timed profile window"
    )
    started_at = time.perf_counter()
    signatures = jax.jit(numerical_signatures)(result)
    jax.block_until_ready(signatures)
    if any(
        jax.typeof(value).sharding.spec != jax.P()
        for value in jax.tree.leaves(signatures)
    ):
        raise ValueError("numerical diagnostics must be globally replicated scalars")
    host_signatures = jax.tree.map(
        lambda value: np.asarray(value.addressable_data(0)).item(), signatures
    )
    logger.info(
        "Materialized global numerical diagnostics in %.3f seconds",
        time.perf_counter() - started_at,
    )
    logger.info("Numerical loss=%g", host_signatures.loss)
    for name in ("output", "input_grad", "gate_grad", "wi_grad", "wo_grad"):
        signature = getattr(host_signatures, name)
        logger.info(
            "Numerical signature %s: finite=%s, sum=%g, squared_sum=%g, max_abs=%g",
            name,
            signature.finite,
            signature.total,
            signature.squared_total,
            signature.maximum,
        )

    counts = result[0][1][1]
    expert_counts = np.asarray(counts.addressable_data(0), dtype=np.int64)
    expected_routes = config.num_layers * GLOBAL_TOKENS * config.num_experts_per_tok
    if int(expert_counts.sum()) != expected_routes:
        raise ValueError(
            f"expected {expected_routes} requested routes, got {expert_counts.sum()}"
        )
    logger.info(
        "Requested expert counts: total=%d, mean=%.3f, std=%.3f, min=%d, "
        "max=%d, values=%s",
        expert_counts.sum(),
        expert_counts.mean(),
        expert_counts.std(),
        expert_counts.min(),
        expert_counts.max(),
        expert_counts.tolist(),
    )
    if not config.use_ring_of_experts:
        logger.info(
            "Dynamic deduplicated ragged-all-to-all send sizes are unavailable: "
            "the production routed API exposes requested expert totals, while "
            "destination-deduplicated token counts remain internal to dispatch"
        )


def run_profile(config: Config) -> None:
    setup_started_at = time.perf_counter()
    mesh_started_at = time.perf_counter()
    mesh = make_mesh(config)
    logger.info(
        "Created mesh in %.3f seconds: shape=%s, devices=%s",
        time.perf_counter() - mesh_started_at,
        mesh.shape,
        [device.id for device in mesh.devices.flat],
    )

    with jax.set_mesh(mesh):
        init_started_at = time.perf_counter()
        gates, wis, wos, inputs, cotangent, router_biases = initialize_profile_arrays(
            config, mesh
        )
        jax.block_until_ready((gates, wis, wos, inputs, cotangent, router_biases))
        logger.info(
            "Initialized reusable profile arrays in %.3f seconds",
            time.perf_counter() - init_started_at,
        )
        logger.info(
            "Profile geometry: layers=%d, global_activation=%s, local_activation=%s, "
            "gate=%s, wi=%s, local_wi=%s, wo=%s, local_wo=%s",
            config.num_layers,
            inputs.shape,
            (1, config.seq_length, config.emb_dim),
            gates[0].shape,
            wis[0].shape,
            (config.num_experts // config.ep, config.emb_dim, 2 * config.moe_mlp_dim),
            wos[0].shape,
            (config.num_experts // config.ep, config.moe_mlp_dim, config.emb_dim),
        )
        row_mib = config.emb_dim * jnp.dtype(config.dtype).itemsize / 2**20
        if config.use_ring_of_experts:
            gathered_rows = config.ep * config.seq_length
            logger.info(
                "Per-layer static ring-of-experts allocations: gathered_rows=%d, "
                "gathered=%.3f MiB, routed_calls_per_stack=%d",
                gathered_rows,
                gathered_rows * row_mib,
                config.num_layers,
            )
        else:
            dedup_send_rows = config.seq_length * min(
                config.num_experts_per_tok, config.ep
            )
            dedup_recv_rows = config.ep * config.seq_length
            logger.info(
                "Per-layer static deduplicated exchange allocations: "
                "send_rows=%d, send=%.3f MiB, recv_rows=%d, recv=%.3f MiB, "
                "routed_calls_per_stack=%d",
                dedup_send_rows,
                dedup_send_rows * row_mib,
                dedup_recv_rows,
                dedup_recv_rows * row_mib,
                config.num_layers,
            )
        log_memory_stats("After MoE profile initialization")

        compile_started_at = time.perf_counter()
        step = jax.jit(functools.partial(profile_step, config))
        compiled = step.lower(
            gates, wis, wos, inputs, cotangent, router_biases
        ).compile()
        logger.info(
            "Compiled routed MoE profile in %.3f seconds",
            time.perf_counter() - compile_started_at,
        )
        logger.info("Compiled memory analysis: %s", compiled.memory_analysis())
        log_memory_stats("After routed MoE compilation")
        logger.info(
            "Profile setup completed in %.3f seconds",
            time.perf_counter() - setup_started_at,
        )

        profile_path = os.path.join(
            config.profile_dir, f"proc_{jax.process_index():06}_step_{NUM_STEPS:06}"
        )
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.raise_error_on_start_failure = True
        flops_per_device = calculate_routed_training_flops_per_device(config)
        local_step_seconds = []
        local_window_seconds = None
        steady_started_at = None

        for step_number in range(1, NUM_STEPS + 1):
            # Keep only one call's outputs live so BFC can reuse gradient storage.
            result = None
            if step_number == WARMUP_STEPS + 1:
                multihost_utils.sync_global_devices("moe_profile_steady_ready")
                steady_started_at = time.perf_counter()

            profile_this_step = step_number == NUM_STEPS
            if profile_this_step:
                multihost_utils.sync_global_devices("moe_profile_trace_ready")
                logger.info("Profiling MoE step %d to %s", step_number, profile_path)
                jax.profiler.start_trace(
                    profile_path, profiler_options=profiler_options
                )
                multihost_utils.sync_global_devices("moe_profile_trace_started")

            started_at = time.perf_counter()
            with jax.profiler.StepTraceAnnotation("moe_core", step_num=step_number):
                result = compiled(gates, wis, wos, inputs, cotangent, router_biases)
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - started_at

            if step_number == 1:
                log_memory_stats("After first routed MoE execution")

            if step_number <= WARMUP_STEPS:
                logger.info(
                    "Completed MoE warmup step %d, local_seconds=%.6f, "
                    "max_process_seconds=None, TFLOP/s/device=None",
                    step_number,
                    elapsed,
                )
            elif not profile_this_step:
                local_step_seconds.append(elapsed)

            if step_number == WARMUP_STEPS:
                log_memory_stats("After routed MoE warmup")

            if step_number == NUM_STEPS - 1:
                if steady_started_at is None:
                    raise ValueError("steady timing did not start")
                if len(local_step_seconds) != NUM_STEPS - WARMUP_STEPS - 1:
                    raise ValueError(
                        "steady timing did not observe every unprofiled step"
                    )
                local_window_seconds = time.perf_counter() - steady_started_at
                timing = np.asarray(
                    [local_window_seconds, *local_step_seconds], dtype=np.float32
                )
                timing_by_process = np.asarray(
                    multihost_utils.process_allgather(timing, tiled=False)
                ).reshape(jax.process_count(), timing.size)
                max_step_seconds = timing_by_process[:, 1:].max(axis=0)
                for index, (local_seconds, max_seconds) in enumerate(
                    zip(local_step_seconds, max_step_seconds, strict=True),
                    start=WARMUP_STEPS + 1,
                ):
                    logger.info(
                        "Completed MoE step %d, local_seconds=%.6f, "
                        "max_process_seconds=%.6f, TFLOP/s/device=%.3f",
                        index,
                        local_seconds,
                        max_seconds,
                        flops_per_device / max_seconds / 1e12,
                    )
                max_window_seconds = float(timing_by_process[:, 0].max())
                mean_seconds = max_window_seconds / len(local_step_seconds)
                median_seconds = float(np.median(max_step_seconds))
                p90_seconds = float(np.percentile(max_step_seconds, 90))
                max_seconds = float(np.max(max_step_seconds))
                mean_seconds_per_layer = mean_seconds / config.num_layers
                topology_normalized_seconds_per_layer = mean_seconds_per_layer * (
                    2 if config.ep == 16 else 1
                )
                logger.info(
                    "Steady MoE window: first_step=%d, last_step=%d, steps=%d, "
                    "local_seconds=%.6f, max_process_seconds=%.6f, "
                    "seconds/stack=%.6f, seconds/layer=%.6f, "
                    "trajectory_median_seconds=%.6f, "
                    "trajectory_p90_seconds=%.6f, trajectory_max_seconds=%.6f, "
                    "TFLOP/s/device=%.3f, "
                    "PP4-normalized-seconds/layer=%.6f",
                    WARMUP_STEPS + 1,
                    NUM_STEPS - 1,
                    len(local_step_seconds),
                    local_window_seconds,
                    max_window_seconds,
                    mean_seconds,
                    mean_seconds_per_layer,
                    median_seconds,
                    p90_seconds,
                    max_seconds,
                    flops_per_device / mean_seconds / 1e12,
                    topology_normalized_seconds_per_layer,
                )

            if profile_this_step:
                multihost_utils.sync_global_devices("moe_profile_trace_done")
                jax.profiler.stop_trace()
                logger.info(
                    "Completed MoE step %d, local_seconds=None, "
                    "max_process_seconds=None, TFLOP/s/device=None, profile=%s",
                    step_number,
                    profile_path,
                )

        if result is None or local_window_seconds is None:
            raise ValueError("profile loop did not complete")
        log_memory_stats("After routed MoE profile, before diagnostics")
        log_numerical_results(config, result)
        log_memory_stats("After out-of-window numerical diagnostics")


def parse_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Profile the production routed MoE core at EP8 or folded EP16."
    )
    parser.add_argument("--ep", type=int, choices=(8, 16), required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--kernel-implementation", choices=("xla", "sm90", "sm100"))
    parser.add_argument(
        "--use-ring-of-experts",
        action="store_true",
        help="Gather tokens and reduce-scatter outputs instead of the "
        "deduplicated ragged all-to-all.",
    )
    args = parser.parse_args(argv)
    return make_config(
        args.ep,
        args.profile_dir,
        args.num_layers,
        implementation=args.kernel_implementation,
        use_ring_of_experts=args.use_ring_of_experts,
    )


def main(argv: list[str] | None = None) -> None:
    absl_logging.use_python_logging()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y%m%d_%H%M%S",
        level=logging.INFO,
        force=True,
    )
    absl_logging.set_verbosity(absl_logging.INFO)
    started_at = time.perf_counter()
    config = parse_config(argv)
    distributed = "NNODES" in os.environ
    if distributed:
        distributed_started_at = time.perf_counter()
        jax.distributed.initialize(
            coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["NNODES"]),
            process_id=int(os.environ["NODE_RANK"]),
            local_device_ids=[
                int(device) for device in os.environ["JAX_LOCAL_DEVICE_IDS"].split(",")
            ],
            initialization_timeout=300,
        )
        logger.info(
            "Initialized distributed JAX in %.3f seconds",
            time.perf_counter() - distributed_started_at,
        )
    logger.info("Configuration: %s", config)
    # Profile under the trainer's config (see ``train.main``).
    with jax.remove_size_one_mesh_axis_from_type(True):
        run_profile(config)
    logger.info("Program completed in %.3f seconds", time.perf_counter() - started_at)
    if distributed:
        shutdown_started_at = time.perf_counter()
        jax.distributed.shutdown()
        logger.info(
            "Shut down distributed JAX in %.3f seconds",
            time.perf_counter() - shutdown_started_at,
        )


if __name__ == "__main__":
    main()
