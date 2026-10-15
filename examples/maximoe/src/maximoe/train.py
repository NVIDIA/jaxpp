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

"""Minimal physical-axis trainer for the MLPerf DeepSeek V3 benchmark."""

import argparse
import dataclasses
import functools
import itertools
import logging
import math
import os
import re
import time
from collections.abc import Callable
from typing import Any, Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import jaxpp.api as jaxpp
import numpy as np
from absl import logging as absl_logging
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding

from maximoe.kernels import backend, gpu
from maximoe.model import (
    AttentionLayerParams,
    YarnRope,
    context_to_expert_batch,
    expert_batch_to_context,
    ffn,
    optimized_routed_ep,
    rms_norm,
    self_attention,
    sum_context_shards,
    take_along_last_dim_no_scatter_add,
)
from maximoe.warmup import warmup_tasks

T = TypeVar("T")
Initializer = Callable[[jax.Array, tuple[int, ...], Any, NamedSharding], jax.Array]
logger = logging.getLogger("maximoe.train")


def save_cudnn_flash_attention(primitive, *_, **__) -> bool:
    return (
        primitive
        is jax._src.cudnn.fused_attention_stablehlo._dot_product_attention_fwd_p_wrapper
    )


layer_remat_policy = jax.checkpoint_policies.save_from_both_policies(
    jax.checkpoint_policies.save_only_these_names(
        "out_proj", "dedup_dispatch_metadata", "query_wa_proj"
    ),
    save_cudnn_flash_attention,
)


class Batch(NamedTuple, Generic[T]):
    inputs: T
    targets: T
    loss_mask: T


class DenseParams(NamedTuple, Generic[T]):
    wi_0: T
    wi_1: T
    wo: T


class MoEParams(NamedTuple, Generic[T]):
    gate: T
    wi: T
    wo: T
    shared: DenseParams[T]


class LayerParams(NamedTuple, Generic[T]):
    attention: AttentionLayerParams[T]
    ffn: DenseParams[T] | MoEParams[T]


class MtpParams(NamedTuple, Generic[T]):
    enorm: T
    hnorm: T
    eh_proj: T
    layer: LayerParams[T]
    final_norm: T


class ModelParams(NamedTuple, Generic[T]):
    embedding: T
    layers: tuple[LayerParams[T], ...]
    final_norm: T
    lm_head: T
    mtp: MtpParams[T] | None


class AdamState(NamedTuple, Generic[T]):
    count: T
    mu: ModelParams[T]
    nu: ModelParams[T]


class RouterState(NamedTuple, Generic[T]):
    biases: tuple[T, ...]


class TrainState(NamedTuple, Generic[T]):
    params: ModelParams[T]
    router: RouterState[T]
    opt: AdamState[T]


class AdamUpdate(NamedTuple, Generic[T]):
    param: T
    mu: T
    nu: T


class ExpertMasterResidual(NamedTuple, Generic[T]):
    low_bits: T
    high_correction: T


class Metrics(NamedTuple, Generic[T]):
    loss: T
    lm_loss: T
    mtp_loss: T
    moe_aux_loss: T
    grad_norm: T
    learning_rate: T


class MicrobatchAux(NamedTuple, Generic[T]):
    lm_loss: T
    mtp_loss: T
    moe_aux_loss: T
    router_counts: tuple[T, ...]


@dataclasses.dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: Any
    initializer: Initializer


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class Config:
    pp: int = 4
    data: int = 1
    fsdp: int = 1
    ep: int = 64
    zero: int = 3
    kernel_implementation: str | None = None
    host_offload_regex: str | None = None
    schedule: str = "1f1b"
    num_pipeline_repeats: int = 1
    pipeline_stage_sizes: tuple[int, ...] | None = None
    fuse_steady_state: bool = False
    expert_gradient_accumulation: bool = True
    compress_expert_master_params: bool = False
    use_ring_of_experts: bool = False
    ep_as_context: bool = False

    per_device_batch_size: float = 64.0
    num_microbatches: int = 256
    seq_length: int = 4_096
    num_train_steps: int = 100
    profile_dir: str | None = None
    warmup_tasks: bool = True

    num_layers: int = 61
    num_dense_layers: int = 3
    emb_dim: int = 7_168
    dense_mlp_dim: int = 18_432
    moe_mlp_dim: int = 2_048
    num_experts: int = 256
    num_experts_per_tok: int = 8
    capacity_factor: float = -1.0
    num_routing_groups: int = 8
    num_selected_groups: int = 4
    vocab_size: int = 129_280

    num_query_heads: int = 128
    q_lora_rank: int = 1_536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    normalization_layer_epsilon: float = 1e-6
    rope_theta: float = 10_000.0
    original_max_position_embeddings: int = 4_096
    rope_factor: float = 40.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0

    routed_scaling_factor: float = 2.5
    moe_aux_loss_weight: float = 0.01
    router_bias_update_rate: float = 0.001
    mtp_num_layers: int = 1
    mtp_loss_weight: float = 0.1

    beta_1: float = 0.9
    beta_2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    min_learning_rate: float = 1e-8
    warmup_steps: int = 4
    decay_steps: int = 11_996

    init_stddev: float = 0.006
    param_seed: int = 1_234
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32
    moment_dtype: Any = jnp.bfloat16

    def __post_init__(self):
        if self.kernel_implementation is not None:
            backend.resolve(self.kernel_implementation)
        if min(self.pp, self.data, self.fsdp, self.ep) < 1:
            raise ValueError("pp, data, fsdp, and ep must be positive")
        if self.zero not in (1, 2, 3):
            raise ValueError("zero must be 1, 2, or 3")
        if self.compress_expert_master_params:
            if jnp.dtype(self.param_dtype) != jnp.dtype(jnp.float32) or jnp.dtype(
                self.dtype
            ) != jnp.dtype(jnp.bfloat16):
                raise ValueError(
                    "expert master compression requires FP32 masters and BF16 compute"
                )
        if self.num_pipeline_repeats < 1:
            raise ValueError("num_pipeline_repeats must be positive")
        if self.num_pipeline_stages > self.num_layers:
            raise ValueError("pipeline stages cannot exceed num_layers")
        if self.pipeline_stage_sizes is not None and (
            len(self.pipeline_stage_sizes) != self.num_pipeline_stages
            or min(self.pipeline_stage_sizes) < 1
            or sum(self.pipeline_stage_sizes) != self.num_layers
        ):
            raise ValueError(
                "pipeline_stage_sizes must contain one positive size per pipeline "
                "stage and sum to num_layers"
            )
        if self.schedule == "1f1b" and self.num_pipeline_repeats != 1:
            raise ValueError("1f1b requires num_pipeline_repeats=1")
        if self.schedule == "1f1b" and self.fuse_steady_state:
            raise ValueError("1f1b does not support fuse_steady_state")
        if self.schedule not in ("1f1b", "kimik2"):
            raise ValueError(f"unsupported pipeline schedule: {self.schedule}")
        if not 0 <= self.num_dense_layers <= self.num_layers:
            raise ValueError("num_dense_layers must be between zero and num_layers")
        if self.num_experts % self.ep:
            raise ValueError("num_experts must be divisible by ep")
        if self.ep_as_context and self.seq_length % self.ep:
            raise ValueError("ep-as-context requires seq_length to be divisible by ep")
        if self.ep_as_context and self.num_query_heads % self.ep:
            raise ValueError(
                "ep-as-context requires num_query_heads to be divisible by ep"
            )
        if self.num_experts % self.num_routing_groups:
            raise ValueError("num_experts must be divisible by num_routing_groups")
        if self.num_selected_groups > self.num_routing_groups:
            raise ValueError("num_selected_groups cannot exceed num_routing_groups")
        if self.num_experts_per_tok > self.num_selected_groups * self.experts_per_group:
            raise ValueError("selected routing groups do not contain enough experts")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive")
        if self.num_microbatches < 1:
            raise ValueError("num_microbatches must be positive")
        global_batch_size = self.per_device_batch_size * self.num_devices
        if not float(global_batch_size).is_integer():
            raise ValueError(
                "per_device_batch_size times the device count must be an integer"
            )
        if int(global_batch_size) % self.num_microbatches:
            raise ValueError("global batch size must be divisible by num_microbatches")
        if self.microbatch_size % self.data_parallel_size:
            raise ValueError(
                "microbatch size must be divisible by the data-parallel size"
            )
        if self.mtp_num_layers not in (0, 1):
            raise ValueError("mtp_num_layers must be zero or one")
        if self.mtp_num_layers and self.seq_length < 2:
            raise ValueError("seq_length must be at least two for MTP")
        if self.num_train_steps < 1:
            raise ValueError("num_train_steps must be positive")
        if self.profile_dir is not None and self.num_train_steps < 4:
            raise ValueError("profiling requires at least three preceding train steps")
        if self.host_offload_regex is not None:
            try:
                re.compile(self.host_offload_regex)
            except re.error as error:
                raise ValueError(f"invalid host-offload-regex: {error}") from error
        if len(set(self.stage_ends)) != self.num_pipeline_stages:
            raise ValueError("pipeline stages must have distinct layer boundaries")

    @property
    def data_parallel_size(self) -> int:
        return self.data * self.fsdp * (1 if self.ep_as_context else self.ep)

    @property
    def sequence_shards(self) -> int:
        return self.ep if self.ep_as_context else 1

    @property
    def num_devices(self) -> int:
        return self.pp * self.data * self.fsdp * self.ep

    @property
    def global_batch_size(self) -> int:
        return int(self.per_device_batch_size * self.num_devices)

    @property
    def microbatch_size(self) -> int:
        return self.global_batch_size // self.num_microbatches

    @property
    def num_pipeline_stages(self) -> int:
        return self.pp * self.num_pipeline_repeats

    @property
    def stage_ends(self) -> tuple[int, ...]:
        if self.pipeline_stage_sizes is not None:
            return tuple(itertools.accumulate(self.pipeline_stage_sizes))
        layers_per_stage, remainder = divmod(self.num_layers, self.num_pipeline_stages)
        return tuple(
            (stage + 1) * layers_per_stage + min(stage + 1, remainder)
            for stage in range(self.num_pipeline_stages)
        )

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.num_routing_groups


def calculate_training_flops_per_device(config: Config) -> float:
    """Estimate matmul FLOPs per device for one forward and backward train step."""
    tokens = config.per_device_batch_size * config.seq_length
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    attention_projection_flops = (
        2
        * tokens
        * (
            config.emb_dim * config.q_lora_rank
            + config.q_lora_rank * config.num_query_heads * qk_head_dim
            + config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
            + config.kv_lora_rank
            * config.num_query_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
            + config.num_query_heads * config.v_head_dim * config.emb_dim
        )
    )
    causal_attention_flops = (
        tokens
        * config.seq_length
        * config.num_query_heads
        * (qk_head_dim + config.v_head_dim)
    )
    attention_flops = (attention_projection_flops + causal_attention_flops) * (
        config.num_layers + config.mtp_num_layers
    )
    dense_flops = (
        6 * tokens * config.emb_dim * config.dense_mlp_dim * config.num_dense_layers
    )
    moe_flops = (
        tokens
        * (
            2 * config.emb_dim * config.num_experts
            + 6 * config.emb_dim * config.moe_mlp_dim * (config.num_experts_per_tok + 1)
        )
        * (config.num_layers - config.num_dense_layers + config.mtp_num_layers)
    )
    mtp_projection_flops = 4 * tokens * config.emb_dim**2 * config.mtp_num_layers
    logits_flops = (
        2 * tokens * config.emb_dim * config.vocab_size * (1 + config.mtp_num_layers)
    )
    return 3 * (
        attention_flops + dense_flops + moe_flops + mtp_projection_flops + logits_flops
    )


def make_mesh(config: Config, devices: list[jax.Device] | None = None) -> Mesh:
    devices = jax.devices() if devices is None else devices
    expected = config.num_devices
    if len(devices) != expected:
        raise ValueError(f"expected {expected} devices, got {len(devices)}")
    return Mesh(
        np.asarray(devices, dtype=object).reshape(
            config.pp, config.data, config.fsdp, config.ep
        ),
        ("stage", "data", "fsdp", "expert"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def move_to_reduced(shardings: T, axes: set[str]) -> T:
    def move(sharding: jax.sharding.PartitionSpec | NamedSharding):
        named = isinstance(sharding, NamedSharding)
        spec = sharding.spec if named else sharding
        moved_axes = axes & set(jax.tree.leaves(spec.partitions))
        if not moved_axes:
            return sharding
        partitions = []
        for partition in spec.partitions:
            names = partition if isinstance(partition, tuple) else (partition,)
            names = tuple(name for name in names if name not in moved_axes)
            partitions.append(names[0] if len(names) == 1 else names or None)
        spec = spec.update(partitions=partitions, reduced=spec.reduced | moved_axes)
        return sharding.update(spec=spec) if named else spec

    return jax.tree.map(move, shardings)


def reshard_if_needed(value: jax.Array, sharding: jax.sharding.Sharding) -> jax.Array:
    # JaxPP task boundaries materialize identity reshards as fresh loop invariants.
    if jax.typeof(value).sharding == sharding:
        return value
    return jax.reshard(value, sharding)


def reshard_params_for_compute(
    params: ModelParams[jax.Array], param_shardings: ModelParams[jax.P]
) -> ModelParams[jax.Array]:
    """Gather optimizer shards into compute layout, FSDP before EP.

    A direct reshard from an optimizer spec such as P(("fsdp", "expert"), ...)
    to a replicated compute spec is one collective over the composite group,
    which spans nodes when FSDP does. Gathering FSDP first keeps the second
    reshard within the node-local EP domain. Under ZeRO-3 the compute spec still
    shards FSDP, and each layer gathers its weights instead.
    """

    def prepare(param, spec):
        if "fsdp" not in jax.tree.leaves(spec.partitions):
            param = reshard_if_needed(
                param, move_to_reduced(jax.typeof(param).sharding, {"fsdp"})
            )
        return reshard_if_needed(param, jax.typeof(param).sharding.update(spec=spec))

    return jax.tree.map(prepare, params, param_shardings)


def shard_gradient_over_fsdp(
    gradient: jax.Array, optimizer_sharding: NamedSharding
) -> jax.Array:
    """Reduce-scatter a ZeRO-2 microbatch gradient over FSDP only."""
    spec = jax.typeof(gradient).sharding.spec
    partitions = [*spec.partitions, *(None,) * (gradient.ndim - len(spec))]
    for dim, partition in enumerate(optimizer_sharding.spec.partitions):
        if "fsdp" in jax.tree.leaves(partition):
            kept = [axis for axis in jax.tree.leaves(partitions[dim]) if axis != "fsdp"]
            partitions[dim] = (*kept, "fsdp") if kept else "fsdp"
    # A fresh spec also clears unreduced_kind when FSDP was the last such axis.
    target = jax.P(
        *partitions, reduced=spec.reduced, unreduced=spec.unreduced - {"fsdp"}
    )
    return reshard_if_needed(
        gradient, jax.typeof(gradient).sharding.update(spec=target)
    )


def reduce_gradient_for_optimizer(
    gradient: jax.Array, optimizer_sharding: NamedSharding
) -> jax.Array:
    """Reduce-scatter into the optimizer layout, then all-reduce over DATA.

    DATA replicates training state, so its contributions stay local until the
    gradient has its FSDP/EP partitioning and the all-reduce moves only shards.
    """
    sharding = jax.typeof(gradient).sharding
    spec = optimizer_sharding.spec
    local = spec.update(unreduced=sharding.spec.unreduced & {"data"})
    gradient = reshard_if_needed(gradient, sharding.update(spec=local))
    return reshard_if_needed(gradient, sharding.update(spec=spec))


def cast_if_needed(value: jax.Array, dtype: Any) -> jax.Array:
    # Preserve object identity so an identity conversion cannot become a new
    # loop-spanning buffer after JaxPP cuts the program into tasks.
    if value.dtype == jnp.dtype(dtype):
        return value
    return value.astype(dtype)


def pack_expert_master(
    master: jax.Array, compute: jax.Array
) -> ExpertMasterResidual[jax.Array]:
    """Save an exact FP32 master using three bytes plus its existing BF16 copy.

    Related work: Intel's Split SGD [1] stores the high and low 16-bit halves
    of FP32 weights; FlashOptim [2, Sec. 3.1] describes weight splitting with
    quantized residuals. Here we preserve the existing rounded BF16 compute
    value and store the low 16 bits plus an 8-bit high-word correction. The
    combined five-byte representation preserves every FP32 bit, including
    NaN payloads and signed zero.

    [1] https://intel.github.io/intel-extension-for-pytorch/xpu/1.13.120%2Bxpu/tutorials/technical_details/split_sgd.html
    [2] https://arxiv.org/pdf/2602.23349
    """
    if jax.typeof(master).memory_space == jax.memory.Space.Host:
        raise ValueError("compressed expert masters cannot be host-offloaded")
    bits = jax.lax.bitcast_convert_type(master, jnp.uint32)
    high = (bits >> 16).astype(jnp.int32)
    compute_high = jax.lax.bitcast_convert_type(compute, jnp.uint16).astype(jnp.int32)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    is_compute_zero = (compute_high & 0x7FFF) == 0
    # NaN conversions may change payload and sign. Subnormal conversions may
    # flush to zero. For both, encode the original high payload/magnitude and
    # sign directly instead of relying on a small rounding correction.
    special = (high & 0x7F) | ((high >> 8) & 0x80)
    correction = jnp.where(is_nan | is_compute_zero, special, high - compute_high)
    return ExpertMasterResidual(bits.astype(jnp.uint16), correction.astype(jnp.int8))


def unpack_expert_master(
    compute: jax.Array, residual: ExpertMasterResidual[jax.Array]
) -> jax.Array:
    """Reconstruct every original FP32 bit, including NaN payloads and signed zero."""
    compute = reshard_if_needed(compute, jax.typeof(residual.low_bits).sharding)
    high = jax.lax.bitcast_convert_type(compute, jnp.uint16).astype(jnp.int32)
    code = jax.lax.bitcast_convert_type(residual.high_correction, jnp.uint8).astype(
        jnp.int32
    )
    special = (code & 0x7F) | ((code & 0x80) << 8)
    is_nan = (high & 0x7FFF) > 0x7F80
    is_zero = (high & 0x7FFF) == 0
    high = jnp.where(
        is_nan,
        special | 0x7F80,
        jnp.where(is_zero, special, high + residual.high_correction.astype(jnp.int32)),
    )
    bits = (high.astype(jnp.uint32) << 16) | residual.low_bits.astype(jnp.uint32)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def pack_expert_masters(params, compute_params):
    residuals = []
    for master, compute in zip(
        jax.tree.leaves(params, is_leaf=lambda value: isinstance(value, MoEParams)),
        jax.tree.leaves(
            compute_params, is_leaf=lambda value: isinstance(value, MoEParams)
        ),
        strict=True,
    ):
        if isinstance(master, MoEParams):
            residuals.append(
                (
                    pack_expert_master(master.wi, compute.wi),
                    pack_expert_master(master.wo, compute.wo),
                )
            )
    return tuple(residuals)


def unpack_expert_masters(params, compute_params, residuals):
    residuals = iter(residuals)

    def restore(master, compute):
        if isinstance(master, MoEParams):
            wi, wo = next(residuals)
            return master._replace(
                wi=unpack_expert_master(compute.wi, wi),
                wo=unpack_expert_master(compute.wo, wo),
            )
        return master

    return jax.tree.map(
        restore,
        params,
        compute_params,
        is_leaf=lambda value: isinstance(value, MoEParams),
    )


def filter_axes(shardings: T, excluded_axes: set[str]) -> T:
    def filter_spec(spec: jax.P):
        partitions = []
        for partition in spec.partitions:
            names = partition if isinstance(partition, tuple) else (partition,)
            names = tuple(name for name in names if name not in excluded_axes)
            partitions.append(names[0] if len(names) == 1 else names or None)
        return spec.update(
            partitions=partitions,
            reduced=spec.reduced - excluded_axes,
            unreduced=spec.unreduced - excluded_axes,
        )

    return jax.tree.map(filter_spec, shardings)


def parameter_specs(config: Config) -> ModelParams[ArraySpec]:
    normal = jax.nn.initializers.normal(config.init_stddev)
    ones = jax.nn.initializers.ones

    def spec(shape, initializer=normal):
        return ArraySpec(shape, config.param_dtype, initializer)

    attention = AttentionLayerParams(
        pre_attn_norm_scale=spec((config.emb_dim,), ones),
        wq_a=spec((config.emb_dim, config.q_lora_rank)),
        q_norm_scale=spec((config.q_lora_rank,), ones),
        wq_b=spec(
            (
                config.q_lora_rank,
                config.num_query_heads,
                config.qk_nope_head_dim + config.qk_rope_head_dim,
            )
        ),
        wkv_a=spec((config.emb_dim, config.kv_lora_rank + config.qk_rope_head_dim)),
        kv_norm_scale=spec((config.kv_lora_rank,), ones),
        wkv_b=spec(
            (
                config.kv_lora_rank,
                config.num_query_heads,
                config.qk_nope_head_dim + config.v_head_dim,
            )
        ),
        out_kernel=spec((config.num_query_heads, config.v_head_dim, config.emb_dim)),
        post_attn_norm_scale=spec((config.emb_dim,), ones),
    )
    dense = DenseParams(
        wi_0=spec((config.emb_dim, config.dense_mlp_dim)),
        wi_1=spec((config.emb_dim, config.dense_mlp_dim)),
        wo=spec((config.dense_mlp_dim, config.emb_dim)),
    )
    moe = MoEParams(
        gate=spec((config.emb_dim, config.num_experts)),
        wi=spec((config.num_experts, config.emb_dim, 2 * config.moe_mlp_dim)),
        wo=spec((config.num_experts, config.moe_mlp_dim, config.emb_dim)),
        shared=DenseParams(
            wi_0=spec((config.emb_dim, config.moe_mlp_dim)),
            wi_1=spec((config.emb_dim, config.moe_mlp_dim)),
            wo=spec((config.moe_mlp_dim, config.emb_dim)),
        ),
    )

    layers = (LayerParams(attention, dense),) * config.num_dense_layers + (
        LayerParams(attention, moe),
    ) * (config.num_layers - config.num_dense_layers)
    mtp = (
        MtpParams(
            enorm=spec((config.emb_dim,), ones),
            hnorm=spec((config.emb_dim,), ones),
            eh_proj=spec((2 * config.emb_dim, config.emb_dim)),
            layer=LayerParams(attention, moe),
            final_norm=spec((config.emb_dim,), ones),
        )
        if config.mtp_num_layers
        else None
    )
    return ModelParams(
        embedding=spec((config.vocab_size, config.emb_dim)),
        layers=layers,
        final_norm=spec((config.emb_dim,), ones),
        lm_head=spec((config.emb_dim, config.vocab_size)),
        mtp=mtp,
    )


def parameter_shardings(config: Config) -> ModelParams[jax.P]:
    # Context-parallel attention up-projects only each EP rank's heads.
    up_proj = (
        jax.P("fsdp", "expert", None, reduced={"data"})
        if config.ep_as_context
        else jax.P("fsdp", None, None, reduced={"data", "expert"})
    )
    attention = AttentionLayerParams(
        pre_attn_norm_scale=jax.P(None, reduced={"data", "fsdp", "expert"}),
        wq_a=jax.P("fsdp", None, reduced={"data", "expert"}),
        q_norm_scale=jax.P(None, reduced={"data", "fsdp", "expert"}),
        wq_b=up_proj,
        wkv_a=jax.P("fsdp", None, reduced={"data", "expert"}),
        kv_norm_scale=jax.P(None, reduced={"data", "fsdp", "expert"}),
        wkv_b=up_proj,
        out_kernel=jax.P("fsdp", None, None, reduced={"data", "expert"}),
        post_attn_norm_scale=jax.P(None, reduced={"data", "fsdp", "expert"}),
    )
    dense = DenseParams(
        wi_0=jax.P("fsdp", None, reduced={"data", "expert"}),
        wi_1=jax.P("fsdp", None, reduced={"data", "expert"}),
        wo=jax.P("fsdp", None, reduced={"data", "expert"}),
    )
    moe = MoEParams(
        gate=jax.P("fsdp", None, reduced={"data", "expert"}),
        wi=jax.P("expert", "fsdp", None, reduced={"data"}),
        wo=jax.P("expert", None, "fsdp", reduced={"data"}),
        shared=DenseParams(
            wi_0=jax.P("fsdp", None, reduced={"data", "expert"}),
            wi_1=jax.P("fsdp", None, reduced={"data", "expert"}),
            wo=jax.P("fsdp", None, reduced={"data", "expert"}),
        ),
    )

    layers = (LayerParams(attention, dense),) * config.num_dense_layers + (
        LayerParams(attention, moe),
    ) * (config.num_layers - config.num_dense_layers)
    mtp = (
        MtpParams(
            enorm=jax.P(None, reduced={"data", "fsdp", "expert"}),
            hnorm=jax.P(None, reduced={"data", "fsdp", "expert"}),
            eh_proj=jax.P("fsdp", None, reduced={"data", "expert"}),
            layer=LayerParams(attention, moe),
            final_norm=jax.P(None, reduced={"data", "fsdp", "expert"}),
        )
        if config.mtp_num_layers
        else None
    )
    params = ModelParams(
        embedding=jax.P("fsdp", None, reduced={"data", "expert"}),
        layers=layers,
        final_norm=jax.P(None, reduced={"data", "fsdp", "expert"}),
        lm_head=jax.P("fsdp", None, reduced={"data", "expert"}),
        mtp=mtp,
    )

    if config.zero < 3:
        params = move_to_reduced(params, {"fsdp"})
    return params


def optimizer_shardings(config: Config) -> ModelParams[jax.P]:
    # Keep head-sharded up-projections on their EP rank; only FSDP is reduced.
    up_proj = (
        jax.P("fsdp", "expert", None)
        if config.ep_as_context
        else jax.P(("fsdp", "expert"), None, None)
    )
    attention = AttentionLayerParams(
        pre_attn_norm_scale=jax.P("fsdp"),
        wq_a=jax.P(("fsdp", "expert"), None),
        q_norm_scale=jax.P("fsdp"),
        wq_b=up_proj,
        wkv_a=jax.P(("fsdp", "expert"), None),
        kv_norm_scale=jax.P("fsdp"),
        wkv_b=up_proj,
        out_kernel=jax.P(("fsdp", "expert"), None, None),
        post_attn_norm_scale=jax.P("fsdp"),
    )
    dense = DenseParams(
        wi_0=jax.P(("fsdp", "expert"), None),
        wi_1=jax.P(("fsdp", "expert"), None),
        wo=jax.P(("fsdp", "expert"), None),
    )
    moe = MoEParams(
        gate=jax.P(("fsdp", "expert"), None),
        wi=jax.P("expert", "fsdp", None),
        wo=jax.P("expert", None, "fsdp"),
        shared=DenseParams(
            wi_0=jax.P(("fsdp", "expert"), None),
            wi_1=jax.P(("fsdp", "expert"), None),
            wo=jax.P(("fsdp", "expert"), None),
        ),
    )

    layers = (LayerParams(attention, dense),) * config.num_dense_layers + (
        LayerParams(attention, moe),
    ) * (config.num_layers - config.num_dense_layers)
    mtp = (
        MtpParams(
            enorm=jax.P("fsdp"),
            hnorm=jax.P("fsdp"),
            eh_proj=jax.P(("fsdp", "expert"), None),
            layer=LayerParams(attention, moe),
            final_norm=jax.P("fsdp"),
        )
        if config.mtp_num_layers
        else None
    )
    params = ModelParams(
        embedding=jax.P(("fsdp", "expert"), None),
        layers=layers,
        final_norm=jax.P("fsdp"),
        lm_head=jax.P(("fsdp", "expert"), None),
        mtp=mtp,
    )

    return params


def train_state_specs(config: Config) -> TrainState[ArraySpec]:
    params = parameter_specs(config)
    zeros = jax.nn.initializers.zeros
    layers = params.layers + (() if params.mtp is None else (params.mtp.layer,))
    moe_layers = tuple(
        layer.ffn for layer in layers if isinstance(layer.ffn, MoEParams)
    )

    def moments(spec):
        return dataclasses.replace(spec, dtype=config.moment_dtype, initializer=zeros)

    count = ArraySpec((), jnp.int32, zeros)
    return TrainState(
        params=params,
        router=RouterState(
            biases=tuple(
                ArraySpec((ffn.gate.shape[-1],), jnp.float32, zeros)
                for ffn in moe_layers
            )
        ),
        opt=AdamState(
            count=count,
            mu=jax.tree.map(moments, params),
            nu=jax.tree.map(moments, params),
        ),
    )


def train_state_shardings(config: Config) -> TrainState[jax.P]:
    optimizer = optimizer_shardings(config)
    params = optimizer
    layers = optimizer.layers + (
        () if optimizer.mtp is None else (optimizer.mtp.layer,)
    )
    moe_layers = tuple(
        layer.ffn for layer in layers if isinstance(layer.ffn, MoEParams)
    )
    return TrainState(
        params=params,
        router=RouterState(biases=tuple(jax.P() for _ in moe_layers)),
        opt=AdamState(count=jax.P(), mu=optimizer, nu=optimizer),
    )


def shape_tree(specs, shardings, mesh: Mesh, *, host_offload_regex: str | None = None):
    pattern = re.compile(host_offload_regex) if host_offload_regex is not None else None
    matched = 0

    def make_shape(path, spec, sharding):
        nonlocal matched
        target = NamedSharding(mesh, sharding)
        if pattern is not None and pattern.search(jax.tree_util.keystr(path)):
            target = target.with_memory_kind("pinned_host")
            matched += 1
        return jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=target)

    shapes = jax.tree.map_with_path(make_shape, specs, shardings)
    if pattern is not None and not matched:
        raise ValueError(
            f"host-offload-regex {host_offload_regex!r} matched no train-state leaves"
        )
    return shapes


def on_device(value: jax.Array) -> jax.Array:
    """Reload an offloaded leaf without changing its partitioning or dtype."""
    if jax.typeof(value).memory_space == jax.memory.Space.Host:
        return jax.device_put(value, jax.memory.Space.Device)
    return value


def batch_shapes(config: Config, mesh: Mesh) -> Batch[jax.ShapeDtypeStruct]:
    shape = (config.num_microbatches, config.microbatch_size, config.seq_length)
    batch_axes = tuple(
        axis
        for axis in ("data", "fsdp", "expert")
        if mesh.shape.get(axis, 1) > 1
        and (axis != "expert" or not config.ep_as_context)
    )
    batch_axis = batch_axes[0] if len(batch_axes) == 1 else batch_axes or None
    sequence_axis = (
        "expert" if config.ep_as_context and mesh.shape.get("expert", 1) > 1 else None
    )
    sharding = NamedSharding(mesh, jax.P(None, batch_axis, sequence_axis))
    tokens = jax.ShapeDtypeStruct(shape, jnp.int32, sharding=sharding)
    loss_mask = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=sharding)
    return Batch(inputs=tokens, targets=tokens, loss_mask=loss_mask)


def init_train_state(
    config: Config, mpmd_mesh: jaxpp.MpmdMesh, specs: TrainState[ArraySpec], shardings
) -> TrainState[jaxpp.MpmdArray]:
    path_specs, tree = jax.tree.flatten_with_path(specs)
    flat_shardings = tree.flatten_up_to(shardings)
    arrays = []
    root_key = jax.random.key(config.param_seed)
    process_index = jax.process_index()

    for index, ((_, spec), sharding) in enumerate(
        zip(path_specs, flat_shardings, strict=True)
    ):
        local = []
        for stage in sharding.mesh_ids:
            stage_mesh = mpmd_mesh.unstack[stage]
            if any(
                device.process_index == process_index
                for device in stage_mesh.devices.flat
            ):
                target = NamedSharding(
                    stage_mesh, sharding.spec, memory_kind=sharding.memory_kind
                )
                with jax.set_mesh(stage_mesh):
                    value = spec.initializer(
                        jax.random.fold_in(root_key, index),
                        spec.shape,
                        spec.dtype,
                        (
                            target.with_memory_kind("device")
                            if target.memory_kind == "pinned_host"
                            else target
                        ),
                    )
                    if target.memory_kind == "pinned_host":
                        value = jax.device_put(value, target)
                        # Release this temporary device shard before initializing
                        # the next leaf, rather than staging the whole state in HBM.
                        value.block_until_ready()
                    local.append(value)
        arrays.append(
            jaxpp.MpmdArray(local, sharding, shape=spec.shape, dtype=spec.dtype)
        )
    return tree.unflatten(arrays)


def put_batch(
    host_batch: Batch[np.ndarray],
    shapes: Batch[jax.ShapeDtypeStruct],
    shardings,
    mpmd_mesh: jaxpp.MpmdMesh,
) -> Batch[jaxpp.MpmdArray]:
    host_leaves, tree = jax.tree.flatten(host_batch)
    shape_leaves = tree.flatten_up_to(shapes)
    sharding_leaves = tree.flatten_up_to(shardings)
    arrays = []
    process_index = jax.process_index()

    for host, shape, sharding in zip(
        host_leaves, shape_leaves, sharding_leaves, strict=True
    ):
        local = []
        for stage in sharding.mesh_ids:
            stage_mesh = mpmd_mesh.unstack[stage]
            if any(
                device.process_index == process_index
                for device in stage_mesh.devices.flat
            ):
                target = NamedSharding(
                    stage_mesh, sharding.spec, memory_kind=sharding.memory_kind
                )
                local.append(
                    jax.make_array_from_process_local_data(
                        target, host, global_shape=shape.shape
                    )
                )
        arrays.append(
            jaxpp.MpmdArray(local, sharding, shape=shape.shape, dtype=shape.dtype)
        )
    return tree.unflatten(arrays)


def attention_apply(
    config: Config,
    params: AttentionLayerParams[jax.Array],
    inputs: jax.Array,
    positions: jax.Array,
    segment_ids: jax.Array,
    rope: YarnRope,
) -> tuple[jax.Array, jax.Array]:
    dtype = params.wq_a.dtype
    normed = rms_norm(
        inputs,
        params.pre_attn_norm_scale,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
    )
    qk_head_dim = params.wq_b.shape[-1]
    yarn_mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
    softmax_scale = yarn_mscale**2 * qk_head_dim**-0.5

    def apply_attention(inputs, positions, segment_ids, params, yarn_rope):
        return self_attention(
            inputs,
            positions,
            segment_ids,
            params,
            epsilon=config.normalization_layer_epsilon,
            kv_lora_rank=params.wkv_b.shape[0],
            qk_nope_head_dim=qk_head_dim - yarn_rope.embedding_dims,
            softmax_scale=softmax_scale,
            dtype=dtype,
            yarn_rope=yarn_rope,
            context_parallel_axis="expert" if config.ep_as_context else None,
        )

    attention = jax.shard_map(
        apply_attention, out_specs=jax.typeof(normed).sharding.spec
    )(normed, positions, segment_ids, params, rope)
    residual = inputs + attention
    return residual, rms_norm(
        residual,
        params.post_attn_norm_scale,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
    )


def route(
    config: Config, gate: jax.Array, bias: jax.Array, inputs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    activation_spec = jax.typeof(inputs).sharding.spec
    num_experts = gate.shape[-1]
    experts_per_group = num_experts // config.num_routing_groups
    raw_logits = jnp.einsum(
        "bsd,de->bse",
        inputs.astype(jnp.float32),
        gate.astype(jnp.float32),
        out_sharding=activation_spec,
    )
    unbiased = jax.nn.sigmoid(raw_logits)
    selection = unbiased + bias.astype(jnp.float32)
    grouped = selection.reshape(
        *selection.shape[:-1], config.num_routing_groups, experts_per_group
    )
    group_scores = jax.lax.top_k(grouped, min(2, experts_per_group))[0].sum(axis=-1)
    selected_groups = jax.lax.top_k(group_scores, config.num_selected_groups)[1]
    group_mask = jax.nn.one_hot(
        selected_groups, config.num_routing_groups, dtype=jnp.int32
    ).sum(axis=-2)
    expert_mask = jnp.repeat(group_mask > 0, experts_per_group, axis=-1)
    selection = jnp.where(expert_mask, selection, -jnp.inf)

    aux_scores = unbiased / (unbiased.sum(axis=-1, keepdims=True) + 1e-20)
    aux_indices = jax.lax.top_k(aux_scores, config.num_experts_per_tok)[1]
    aux_counts = jax.nn.one_hot(aux_indices, num_experts, dtype=jnp.float32).sum(
        axis=(1, 2)
    )
    score_sums = aux_scores.sum(axis=1)
    if config.ep_as_context:
        # Routing sees [B * EP, S / EP, ...]. Balance the original sequences by
        # summing their small per-expert summaries across the context shards.
        aux_counts = sum_context_shards(aux_counts, sequence_shards=config.ep)
        score_sums = sum_context_shards(score_sums, sequence_shards=config.ep)
    sequence_length = inputs.shape[1] * config.sequence_shards
    aux_loss = (score_sums * jax.lax.stop_gradient(aux_counts)).sum(axis=-1)
    aux_loss *= num_experts / (
        aux_indices.shape[-1] * sequence_length * sequence_length
    )
    with jax.named_scope("top_k"):
        _, top_k_indices = jax.lax.top_k(selection, config.num_experts_per_tok)
        top_k_weights = take_along_last_dim_no_scatter_add(unbiased, top_k_indices)
        top_k_weights = (
            top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-20)
        ) * (config.routed_scaling_factor)
    return top_k_weights, top_k_indices, config.moe_aux_loss_weight * aux_loss.mean()


def gather_fsdp(tree: T) -> T:
    """Gather each FSDP-sharded compute weight in ``tree`` (ZeRO-3).

    Other weights are returned as is: an identity reshard still creates an AD
    boundary, which would force a gradient value before the expert weights'
    Ref update.
    """
    return jax.tree.map(
        lambda value: reshard_if_needed(
            value, move_to_reduced(jax.typeof(value).sharding, {"fsdp"})
        ),
        tree,
    )


def mlp_apply(params: DenseParams[jax.Array], inputs: jax.Array) -> jax.Array:
    return ffn(
        params.wi_0,
        params.wi_1,
        params.wo,
        inputs,
        dtype=params.wo.dtype,
        out_sharding=jax.typeof(inputs).sharding.spec,
    )


def layer_apply(
    config: Config,
    layer: LayerParams[jax.Array],
    router_bias: jax.Array | None,
    inputs: jax.Array,
    positions: jax.Array,
    segment_ids: jax.Array,
    rope: YarnRope,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Apply one decoder layer: attention, then a dense or routed MoE MLP.

    Returns the layer output, the MoE auxiliary loss, and the requested route
    count per expert (``None`` for dense layers, which take no router bias).
    """
    attention, mlp = gather_fsdp(layer)
    residual, normed = attention_apply(
        config, attention, inputs, positions, segment_ids, rope
    )
    if isinstance(mlp, DenseParams):
        return residual + mlp_apply(mlp, normed), jnp.zeros((), jnp.float32), None
    routed, aux_loss, counts = routed_moe_apply(
        config, mlp.gate, mlp.wi, mlp.wo, router_bias, normed
    )
    return residual + routed + mlp_apply(mlp.shared, normed), aux_loss, counts


checkpointed_layer_apply = jax.checkpoint(
    layer_apply, policy=layer_remat_policy, prevent_cse=True
)


def routed_moe_apply(
    config: Config,
    gate: jax.Array,
    wi: jax.Array,
    wo: jax.Array,
    router_bias: jax.Array,
    inputs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the routed expert path with the trainer's exact routing contract."""
    if config.ep_as_context:
        inputs = context_to_expert_batch(inputs, sequence_shards=config.ep)

    top_k_weights, top_k_indices, aux_loss = route(config, gate, router_bias, inputs)
    routed, routing_stats = optimized_routed_ep(
        inputs,
        top_k_weights,
        top_k_indices,
        wi,
        wo,
        capacity_factor=config.capacity_factor,
        sequence_shards=config.sequence_shards,
        implementation=config.kernel_implementation,
        use_ring_of_experts=config.use_ring_of_experts,
        dtype=wi.dtype,
    )
    if config.ep_as_context:
        routed = expert_batch_to_context(routed, sequence_shards=config.ep)
    counts = routing_stats.requested_per_seq_expert.sum(axis=0)
    return routed, aux_loss, counts


def lookup_embedding(weights: jax.Array, tokens: jax.Array) -> jax.Array:
    """Keep replica-local scatter cotangents distinct until scan finishes."""
    token_spec = jax.typeof(tokens).sharding.spec
    output_spec = token_spec.update(partitions=token_spec.partitions + (None,))
    weight_spec = jax.typeof(weights).sharding.spec
    if not weight_spec.reduced:
        return weights.at[tokens].get(out_sharding=output_spec)

    def lookup(weights, tokens):
        weights = jax.lax.pcast(
            weights, tuple(jax.typeof(weights).manual_axis_type.reduced), to="varying"
        )
        return weights.at[tokens].get()

    return jax.shard_map(lookup, out_specs=output_spec)(weights, tokens)


def mtp_apply(
    config: Config,
    params: MtpParams[jax.Array],
    embedding: jax.Array,
    lm_head: jax.Array,
    router_bias: jax.Array,
    inputs: jax.Array,
    tokens: jax.Array,
    positions: jax.Array,
    segment_ids: jax.Array,
    rope: YarnRope,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    enorm, hnorm, eh_proj, final_norm, embedding = gather_fsdp(
        (params.enorm, params.hnorm, params.eh_proj, params.final_norm, embedding)
    )
    mtp_inputs = jnp.roll(tokens, -1, axis=-1).at[:, -1].set(0)
    mtp_positions = jnp.roll(positions, -1, axis=-1).at[:, -1].set(0)
    mtp_embedding = lookup_embedding(embedding, mtp_inputs)
    mtp_embedding = rms_norm(
        mtp_embedding,
        enorm,
        epsilon=config.normalization_layer_epsilon,
        dtype=mtp_embedding.dtype,
    )
    hidden = rms_norm(
        inputs, hnorm, epsilon=config.normalization_layer_epsilon, dtype=inputs.dtype
    )
    hidden = jnp.einsum(
        "bsd,df->bsf",
        jnp.concatenate((mtp_embedding, hidden), axis=-1),
        eh_proj,
        out_sharding=jax.typeof(inputs).sharding.spec,
    )
    hidden, aux_loss, counts = checkpointed_layer_apply(
        config, params.layer, router_bias, hidden, mtp_positions, segment_ids, rope
    )
    hidden = rms_norm(
        hidden,
        final_norm,
        epsilon=config.normalization_layer_epsilon,
        dtype=hidden.dtype,
    )
    lm_head = gather_fsdp(lm_head)
    return jnp.einsum("bsd,dv->bsv", hidden, lm_head), aux_loss, counts


def model_apply(
    config: Config,
    params: ModelParams[jax.Array],
    router: RouterState[jax.Array],
    batch: Batch[jax.Array],
    positions: jax.Array,
    segment_ids: jax.Array,
    rope: YarnRope,
) -> tuple[jax.Array, jax.Array | None, jax.Array, tuple[jax.Array, ...]]:
    embedding = gather_fsdp(params.embedding)
    inputs = lookup_embedding(embedding, batch.inputs)
    aux_loss = jnp.zeros((), jnp.float32)
    router_counts = []
    # Router biases follow the MoE layers in order, ending with the MTP layer.
    router_biases = iter(router.biases)
    stage_ends = set(config.stage_ends[:-1])
    for index, layer in enumerate(params.layers, start=1):
        router_bias = next(router_biases) if isinstance(layer.ffn, MoEParams) else None
        inputs, layer_aux, counts = checkpointed_layer_apply(
            config, layer, router_bias, inputs, positions, segment_ids, rope
        )
        aux_loss += layer_aux
        if counts is not None:
            router_counts.append(counts)
        if index in stage_ends:
            inputs, aux_loss = jaxpp.mark_stage_end(
                (inputs, aux_loss), name=f"layers_{index}"
            )

    final_norm, lm_head = gather_fsdp((params.final_norm, params.lm_head))
    main_hidden = rms_norm(
        inputs,
        final_norm,
        epsilon=config.normalization_layer_epsilon,
        dtype=inputs.dtype,
    )
    logits = jnp.einsum("bsd,dv->bsv", main_hidden, lm_head)
    if params.mtp is None:
        return logits, None, aux_loss, tuple(router_counts)

    mtp_logits, mtp_aux, mtp_counts = mtp_apply(
        config,
        params.mtp,
        params.embedding,
        params.lm_head,
        next(router_biases),
        inputs,
        batch.inputs,
        positions,
        segment_ids,
        rope,
    )
    aux_loss += mtp_aux
    router_counts.append(mtp_counts)
    return logits, mtp_logits, aux_loss, tuple(router_counts)


def cross_entropy(
    logits: jax.Array, targets: jax.Array, loss_mask: jax.Array, token_count: jax.Array
) -> jax.Array:
    logits = logits.astype(jnp.float32)
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    losses = jax.nn.logsumexp(logits, axis=-1) - target_logits
    return (losses * loss_mask).sum() / token_count


def learning_rate(config: Config, step: jax.Array, global_batch_size: int) -> jax.Array:
    peak = round(2.4e-5 * math.sqrt(global_batch_size / 16_384), 9)
    step = step.astype(jnp.float32)
    warmup = jnp.minimum(step / config.warmup_steps, 1.0)
    progress = jnp.clip((step - config.warmup_steps) / config.decay_steps, 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    decay = config.min_learning_rate + (peak - config.min_learning_rate) * cosine
    return jnp.where(step < config.warmup_steps, peak * warmup, decay)


def update_router_biases(
    config: Config, state: RouterState[jax.Array], counts: tuple[jax.Array, ...]
) -> RouterState[jax.Array]:
    return RouterState(
        biases=tuple(
            (
                bias
                + config.router_bias_update_rate
                * jnp.sign(layer_counts.mean() - layer_counts)
            )
            for bias, layer_counts in zip(state.biases, counts, strict=True)
        )
    )


def adam_update(
    config: Config,
    param: jax.Array,
    grad: jax.Array,
    mu: jax.Array,
    nu: jax.Array,
    *,
    count: jax.Array,
    learning_rate: jax.Array,
) -> AdamUpdate[jax.Array]:
    """Update master weights in FP32, rounding moments only for storage."""
    grad = grad.astype(jnp.float32)
    mu_fp32 = config.beta_1 * mu.astype(jnp.float32) + (1.0 - config.beta_1) * grad
    nu_fp32 = config.beta_2 * nu.astype(jnp.float32) + (
        1.0 - config.beta_2
    ) * jnp.square(grad)
    mu_hat = mu_fp32 / (1.0 - config.beta_1**count)
    nu_hat = nu_fp32 / (1.0 - config.beta_2**count)
    update = mu_hat / (jnp.sqrt(nu_hat) + config.adam_epsilon)
    param_fp32 = param.astype(jnp.float32)
    if param.ndim > 1:
        update += config.weight_decay * param_fp32
    return AdamUpdate(
        (param_fp32 - learning_rate * update).astype(param.dtype),
        mu_fp32.astype(mu.dtype),
        nu_fp32.astype(nu.dtype),
    )


def train_step(
    config: Config,
    param_shardings: ModelParams[jax.P],
    state: TrainState[jax.Array],
    batch: Batch[jax.Array],
) -> tuple[TrainState[jax.Array], Metrics[jax.Array]]:
    # Differentiate BF16 compute copies, including shared embedding/head weights,
    # so gradient accumulation stays in compute precision. Keep FP32 masters
    # separately in their optimizer sharding for the update after the pipeline.
    compute_params = jax.tree.map(
        lambda param: cast_if_needed(on_device(param), config.dtype), state.params
    )
    expert_master_residuals = (
        pack_expert_masters(state.params, compute_params)
        if config.compress_expert_master_params
        else None
    )
    params = reshard_params_for_compute(compute_params, param_shardings)
    router = jax.tree.map(on_device, state.router)
    lm_token_count = batch.loss_mask.sum()
    mtp_token_count = batch.loss_mask[..., 1:].sum() if config.mtp_num_layers else None

    # These values are loop-invariant and consumed by every pipeline stage. By
    # creating them before treduce, JaxPP's placement pass materializes them on
    # every consuming MPMD rank instead of transferring them from stage zero.
    batch_token_spec = jax.typeof(batch.inputs).sharding.spec
    microbatch_token_spec = jax.P(*batch_token_spec.partitions[1:])
    microbatch_token_shape = batch.inputs.shape[1:]
    positions = jnp.broadcast_to(
        jnp.arange(microbatch_token_shape[-1], dtype=jnp.int32),
        microbatch_token_shape,
        out_sharding=microbatch_token_spec,
    )
    segment_ids = jnp.ones(
        microbatch_token_shape, dtype=jnp.int32, out_sharding=microbatch_token_spec
    )
    first_attention = params.layers[0].attention
    rope = YarnRope.mk(
        embedding_dims=(
            first_attention.wkv_a.shape[-1] - first_attention.wkv_b.shape[0]
        ),
        rope_theta=config.rope_theta,
        original_max_position_embeddings=config.original_max_position_embeddings,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
        rope_factor=config.rope_factor,
    )

    def loss_fn(params, microbatch):
        logits, mtp_logits, moe_aux_loss, router_counts = model_apply(
            config, params, router, microbatch, positions, segment_ids, rope
        )
        lm_loss = cross_entropy(
            logits, microbatch.targets, microbatch.loss_mask, lm_token_count
        )
        mtp_loss = (
            jnp.zeros((), jnp.float32)
            if mtp_logits is None
            else cross_entropy(
                mtp_logits[:, :-1],
                microbatch.targets[:, 1:],
                microbatch.loss_mask[:, 1:],
                mtp_token_count,
            )
        )
        moe_aux_loss /= batch.inputs.shape[0]
        loss = lm_loss + config.mtp_loss_weight * mtp_loss + moe_aux_loss
        loss, lm_loss, mtp_loss, moe_aux_loss = jaxpp.mark_stage_end(
            (loss, lm_loss, mtp_loss, moe_aux_loss), name="loss"
        )
        aux = MicrobatchAux(
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            moe_aux_loss=moe_aux_loss,
            router_counts=router_counts,
        )
        return loss, aux

    # With expert gradient accumulation, the ragged kernels add each
    # microbatch's expert dW into Refs that treduce threads through the loop.
    # Shared embedding/head gradients must remain values so JaxPP can split their
    # accumulations between the pipeline stages using them.
    def gradient_ref(param, accumulate_in_place):
        if not accumulate_in_place:
            return jax.ad.GradValue()
        sharding = jax.typeof(param).to_ct_aval().sharding
        return jax.new_ref(jnp.zeros(param.shape, param.dtype, out_sharding=sharding))

    # ZeRO-2 reduce-scatters each microbatch before accumulating; its sharded
    # accumulator cannot take a kernel's full expert dW.
    shard_gradients = config.zero == 2 and config.fsdp > 1

    def expert_weights(node):
        in_place = (
            isinstance(node, MoEParams)
            and config.expert_gradient_accumulation
            and not shard_gradients
        )
        mask = jax.tree.map(lambda _: False, node)
        return mask._replace(wi=True, wo=True) if in_place else mask

    grad_refs = jax.tree.map(
        gradient_ref,
        params,
        jax.tree.map(
            expert_weights, params, is_leaf=lambda x: isinstance(x, MoEParams)
        ),
    )

    def microbatch_grad(microbatch):
        loss, pullback, aux = jax.vjp(
            functools.partial(loss_fn, microbatch=microbatch), params, has_aux=True
        )
        (grads,) = pullback.with_refs(grad_refs)(jnp.ones_like(loss))
        if expert_master_residuals is not None:
            # JaxPP places before-loop outputs through their loop consumers.
            # The residuals are read only by the optimizer, so associate them
            # with each expert layer's router counts without a runtime read.
            def place_residuals(counts, residuals):
                for residual in jax.tree.leaves(residuals):
                    counts = jaxpp.place_with(counts, residual)
                return counts

            aux = aux._replace(
                router_counts=tuple(
                    place_residuals(counts, residuals)
                    for counts, residuals in zip(
                        aux.router_counts, expert_master_residuals, strict=True
                    )
                )
            )
        if shard_gradients:
            grads = jax.tree.map(
                lambda grad, mu: shard_gradient_over_fsdp(
                    grad, jax.typeof(mu).sharding
                ),
                grads,
                state.opt.mu,
            )
        return aux, grads

    if config.schedule == "1f1b":
        schedule = jaxpp.Std1F1B(config.pp)
    else:
        schedule = jaxpp.KimiK2(
            config.num_pipeline_stages, config.pp, config.fuse_steady_state
        )
    aux, grads = jaxpp.treduce(
        microbatch_grad, batch, schedule=schedule, operation=(jaxpp.Concat(), jaxpp.Add)
    )
    # Gradients accumulated into Refs come back as empty GradRef placeholders.
    grads = jax.tree.map(
        lambda ref, grad: (
            grad if isinstance(ref, jax.ad.GradValue) else jax.freeze(ref)
        ),
        grad_refs,
        grads,
    )
    aux = jax.tree.map(functools.partial(jnp.sum, axis=0), aux)
    loss = aux.lm_loss + config.mtp_loss_weight * aux.mtp_loss + aux.moe_aux_loss
    # Compute parameters reduced over DATA/FSDP/EP give gradients unreduced over
    # those axes, so JaxPP adds microbatch gradients locally (ZeRO-2 already
    # holds FSDP shards) and they are reduced once into the optimizer layout.
    grads = jax.tree.map(
        lambda grad, mu: reduce_gradient_for_optimizer(grad, jax.typeof(mu).sharding),
        grads,
        state.opt.mu,
    )
    if jnp.dtype(config.param_dtype) == jnp.dtype(config.dtype):
        # Reconstruct optimizer shards from the complete compute values, allowing
        # JaxPP to release the original parameter shards before the pipeline loop.
        optimizer_params = jax.tree.map(
            lambda param, mu: reshard_if_needed(param, jax.typeof(mu).sharding),
            params,
            state.opt.mu,
        )
    else:
        # Recover exact expert masters from their compute copies and residuals.
        optimizer_params = (
            unpack_expert_masters(state.params, params, expert_master_residuals)
            if config.compress_expert_master_params
            else state.params
        )
    # Issue host reloads after the pipeline and before the global norm barrier,
    # so they can overlap the cross-stage reduction rather than wait for clipping.
    optimizer_params, optimizer_mu, optimizer_nu = jax.tree.map(
        on_device, (optimizer_params, state.opt.mu, state.opt.nu)
    )
    old_count = on_device(state.opt.count)
    squared_norms = tuple(
        jnp.sum(jnp.square(grad.astype(jnp.float32))) for grad in jax.tree.leaves(grads)
    )
    grad_norm = jnp.sqrt(jaxpp.cross_mpmd_all_reduce(*squared_norms))
    clip = jnp.minimum(1.0, config.gradient_clip / (grad_norm + 1e-6))
    grads = jax.tree.map(lambda grad: grad * clip, grads)
    count = old_count + 1
    global_batch_size = batch.inputs.shape[0] * batch.inputs.shape[1]
    lr = learning_rate(config, old_count, global_batch_size)

    update = functools.partial(adam_update, config, count=count, learning_rate=lr)

    updated = jax.tree.map(update, optimizer_params, grads, optimizer_mu, optimizer_nu)

    def is_update(value):
        return isinstance(value, AdamUpdate)

    new_params = jax.tree.map(lambda value: value[0], updated, is_leaf=is_update)
    mu = jax.tree.map(lambda value: value[1], updated, is_leaf=is_update)
    nu = jax.tree.map(lambda value: value[2], updated, is_leaf=is_update)
    router = update_router_biases(config, router, aux.router_counts)
    new_state = TrainState(
        params=new_params, router=router, opt=AdamState(count=count, mu=mu, nu=nu)
    )
    new_state = jax.tree.map(
        lambda value, previous: (
            jax.device_put(value, jax.memory.Space.Host)
            if jax.typeof(previous).memory_space == jax.memory.Space.Host
            else value
        ),
        new_state,
        state,
    )
    metrics = Metrics(
        loss=loss,
        lm_loss=aux.lm_loss,
        mtp_loss=aux.mtp_loss,
        moe_aux_loss=aux.moe_aux_loss,
        grad_norm=grad_norm,
        learning_rate=lr,
    )
    return new_state, metrics


def observe_metrics(
    step_index: int,
    metrics: Metrics[jaxpp.MpmdArray],
    step_seconds: float | None,
    flops_per_device: float,
    tokens_per_device: float,
) -> None:
    jax.block_until_ready(metrics)

    def local_scalar(value: jaxpp.MpmdArray):
        if not value.is_partially_addressable:
            return None
        local = value.to_mpmd_local_array
        if isinstance(local, list):
            local = local[0]
        return np.asarray(local.addressable_data(0)).item()

    local_metrics = jax.tree.map(local_scalar, metrics)
    if step_seconds is None:
        seconds = tflops_per_second = tokens_per_second = None
    else:
        seconds = f"{step_seconds:.3f}"
        tflops_per_second = f"{flops_per_device / step_seconds / 1e12:.3f}"
        tokens_per_second = f"{tokens_per_device / step_seconds:.3f}"
    logger.info(
        "Completed train step %d, seconds=%s, TFLOP/s/device=%s, "
        "Tokens/s/device=%s, loss=%s, lm_loss=%s, mtp_loss=%s, "
        "moe_aux_loss=%s, grad_norm=%s, learning_rate=%s",
        step_index,
        seconds,
        tflops_per_second,
        tokens_per_second,
        None if local_metrics.loss is None else f"{local_metrics.loss:.6f}",
        None if local_metrics.lm_loss is None else f"{local_metrics.lm_loss:.6f}",
        None if local_metrics.mtp_loss is None else f"{local_metrics.mtp_loss:.6f}",
        (
            None
            if local_metrics.moe_aux_loss is None
            else f"{local_metrics.moe_aux_loss:.6f}"
        ),
        None if local_metrics.grad_norm is None else f"{local_metrics.grad_norm:.6f}",
        (
            None
            if local_metrics.learning_rate is None
            else f"{local_metrics.learning_rate:.3e}"
        ),
    )


def log_memory_stats(label: str):
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        values = []
        for key in (
            "bytes_in_use",
            "peak_bytes_in_use",
            "bytes_limit",
            "largest_free_block_bytes",
            "largest_alloc_size",
        ):
            value = stats.get(key)
            values.append(None if value is None else f"{value / 2**30:.2f} GiB")
        logger.info(
            "%s memory: process=%d, device=%s, in_use=%s, peak=%s, limit=%s, "
            "largest_free=%s, largest_alloc=%s",
            label,
            jax.process_index(),
            device,
            *values,
        )


def train_loop(config: Config, host_batch: Batch[np.ndarray]):
    setup_started_at = time.perf_counter()
    mesh_started_at = time.perf_counter()
    mesh = make_mesh(config)
    mpmd_mesh = jaxpp.MpmdMesh(mesh, "stage")
    local_mesh = mpmd_mesh.lowering_mesh()
    logger.info(
        "Created the training mesh in %.3f seconds",
        time.perf_counter() - mesh_started_at,
    )

    metadata_started_at = time.perf_counter()
    specs = train_state_specs(config)
    excluded_axes = {
        axis
        for axis in ("data", "fsdp", "expert")
        if local_mesh.shape.get(axis, 1) == 1
    }
    param_shardings = filter_axes(parameter_shardings(config), excluded_axes)
    state_spec_shardings = filter_axes(train_state_shardings(config), excluded_axes)
    state_shapes = shape_tree(
        specs,
        state_spec_shardings,
        local_mesh,
        host_offload_regex=config.host_offload_regex,
    )
    if config.host_offload_regex is not None:
        offloaded = [
            shape
            for shape in jax.tree.leaves(state_shapes)
            if shape.sharding.memory_kind == "pinned_host"
        ]
        logger.info(
            "Host offload: regex=%r, leaves=%d, global_state=%.3f GiB",
            config.host_offload_regex,
            len(offloaded),
            sum(
                math.prod(shape.shape) * np.dtype(shape.dtype).itemsize
                for shape in offloaded
            )
            / 2**30,
        )
    data_shapes = batch_shapes(config, local_mesh)
    state_shardings = jax.tree.map(lambda shape: shape.sharding, state_shapes)
    data_shardings = jax.tree.map(lambda shape: shape.sharding, data_shapes)
    metric_shardings = Metrics(
        *(NamedSharding(local_mesh, jax.P()) for _ in Metrics._fields)
    )
    traceable = jaxpp.mpmd_jit_with_loop(
        functools.partial(train_step, config, param_shardings),
        mpmd_mesh=mpmd_mesh,
        in_shardings=(state_shardings, data_shardings),
        out_shardings=(state_shardings, metric_shardings),
        donate_argnums=(0,),
    )
    num_parameters = sum(
        math.prod(spec.shape) for spec in jax.tree.leaves(specs.params)
    )
    logger.info(
        "Prepared train metadata in %.3f seconds, parameters=%.3f billion",
        time.perf_counter() - metadata_started_at,
        num_parameters / 1e9,
    )

    prepare_started_at = time.perf_counter()
    step = traceable.compile(state_shapes, data_shapes)
    (in_shardings, kwargs_shardings) = step.in_shardings
    if kwargs_shardings:
        raise ValueError("train step unexpectedly inferred keyword inputs")
    state_input_shardings, batch_input_shardings = in_shardings
    logger.info(
        "Prepared the train step in %.3f seconds",
        time.perf_counter() - prepare_started_at,
    )

    if config.warmup_tasks:
        warmup_tasks(step)

    init_started_at = time.perf_counter()
    state = init_train_state(config, mpmd_mesh, specs, state_input_shardings)
    jax.block_until_ready(state)
    logger.info(
        "Initialized the train state in %.3f seconds",
        time.perf_counter() - init_started_at,
    )
    log_memory_stats("After train-state initialization")

    batch_started_at = time.perf_counter()
    batch = put_batch(host_batch, data_shapes, batch_input_shardings, mpmd_mesh)
    jax.block_until_ready(batch)
    logger.info(
        "Prepared the reusable device batch in %.3f seconds",
        time.perf_counter() - batch_started_at,
    )

    flops_per_device = calculate_training_flops_per_device(config)
    tokens_per_device = config.per_device_batch_size * config.seq_length
    logger.info(
        "Trainer setup completed in %.3f seconds, training_work=%.3f "
        "TFLOP/device/step, tokens=%.0f/device/step",
        time.perf_counter() - setup_started_at,
        flops_per_device / 1e12,
        tokens_per_device,
    )
    logger.info(
        "Per-step performance uses host dispatch cadence with one-step metric buffering"
    )
    logger.info(
        "The first three train step timings include lazy XLA compilation, "
        "dispatch startup, and the first buffered metric observation"
    )
    profile_step = config.num_train_steps if config.profile_dir is not None else None
    if profile_step is not None:
        profile_path = os.path.join(
            config.profile_dir, f"proc_{jax.process_index():06}_step_{profile_step:06}"
        )
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.raise_error_on_start_failure = True
        logger.info(
            "Train step %d will be profiled after three startup steps and %d "
            "steady train steps, and excluded from performance timing",
            profile_step,
            profile_step - 4,
        )

    pending = None
    last_dispatch = time.perf_counter()
    last_timed_step = config.num_train_steps - int(profile_step is not None)
    steady_step_count = max(0, last_timed_step - 3)
    steady_started_at = None
    local_steady_training_seconds = None
    steady_training_seconds = None

    for step_index in range(config.num_train_steps):
        step_number = step_index + 1
        profile_this_step = step_number == profile_step
        if step_number == 4 and steady_step_count:
            _, previous_metrics, _ = pending
            jax.block_until_ready((state, previous_metrics))
            multihost_utils.sync_global_devices("maximoe_steady_ready")
            steady_started_at = time.perf_counter()
            last_dispatch = steady_started_at

        if profile_this_step:
            if pending is None:
                jax.block_until_ready(state)
            else:
                _, previous_metrics, _ = pending
                jax.block_until_ready((state, previous_metrics))
            multihost_utils.sync_global_devices("maximoe_profile_ready")
            logger.info("Profiling train step %d to %s", step_number, profile_path)
            jax.profiler.start_trace(profile_path, profiler_options=profiler_options)
            multihost_utils.sync_global_devices("maximoe_profile_started")

        with jax.profiler.StepTraceAnnotation("train", step_num=step_number):
            state, metrics = step(state, batch)

        if profile_this_step:
            jax.block_until_ready((state, metrics))
            multihost_utils.sync_global_devices("maximoe_profile_done")
            jax.profiler.stop_trace()
            step_seconds = None
            logger.info("Saved train step %d profile to %s", step_number, profile_path)
        else:
            dispatched_at = time.perf_counter()
            step_seconds = dispatched_at - last_dispatch
            last_dispatch = dispatched_at

        if pending is not None:
            previous_step, previous_metrics, previous_seconds = pending
            observe_metrics(
                previous_step,
                previous_metrics,
                previous_seconds,
                flops_per_device,
                tokens_per_device,
            )

        pending = step_number, metrics, step_seconds
        if step_number == last_timed_step and steady_started_at is not None:
            terminal_step, terminal_metrics, terminal_seconds = pending
            jax.block_until_ready((state, terminal_metrics))
            local_finished_at = time.perf_counter()
            local_steady_training_seconds = local_finished_at - steady_started_at
            elapsed_by_process = multihost_utils.process_allgather(
                np.asarray(local_steady_training_seconds, dtype=np.float32), tiled=False
            )
            steady_training_seconds = float(np.max(elapsed_by_process))
            observe_metrics(
                terminal_step,
                terminal_metrics,
                terminal_seconds,
                flops_per_device,
                tokens_per_device,
            )
            pending = None

    if pending is not None:
        previous_step, previous_metrics, previous_seconds = pending
        jax.block_until_ready((state, previous_metrics))
        observe_metrics(
            previous_step,
            previous_metrics,
            previous_seconds,
            flops_per_device,
            tokens_per_device,
        )

    mean_step_time = (
        None
        if steady_training_seconds is None
        else steady_training_seconds / steady_step_count
    )

    logger.info(
        "Steady train window: first_step=%s, last_step=%s, steps=%d, "
        "local_seconds=%s, max_process_seconds=%s, seconds/step=%s, "
        "TFLOP/s/device=%s, Tokens/s/device=%s",
        4 if steady_step_count else None,
        last_timed_step if steady_step_count else None,
        steady_step_count,
        (
            None
            if local_steady_training_seconds is None
            else f"{local_steady_training_seconds:.3f}"
        ),
        None if steady_training_seconds is None else f"{steady_training_seconds:.3f}",
        None if mean_step_time is None else f"{mean_step_time:.3f}",
        (
            None
            if mean_step_time is None
            else f"{flops_per_device / mean_step_time / 1e12:.3f}"
        ),
        (
            None
            if mean_step_time is None
            else f"{tokens_per_device / mean_step_time:.3f}"
        ),
    )
    log_memory_stats("After training loop")
    return state


def tiny_config() -> Config:
    """Return a small DeepSeek V3 trainer smoke-test configuration."""
    return Config(
        pp=1,
        data=1,
        fsdp=1,
        ep=1,
        per_device_batch_size=1,
        num_microbatches=1,
        seq_length=128,
        num_train_steps=2,
        num_layers=2,
        num_dense_layers=1,
        emb_dim=256,
        dense_mlp_dim=512,
        moe_mlp_dim=256,
        num_experts=8,
        num_experts_per_tok=2,
        num_routing_groups=1,
        num_selected_groups=1,
        vocab_size=1_024,
        num_query_heads=8,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mtp_num_layers=0,
    )


def nullable_str(value: str) -> str | None:
    return None if value == "None" else value


def main():
    absl_logging.use_python_logging()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y%m%d_%H%M%S",
        level=logging.INFO,
        force=True,
    )
    absl_logging.set_verbosity(absl_logging.INFO)
    program_started_at = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run the physical-axis DeepSeek V3 trainer on synthetic data.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preset",
        choices=("tiny",),
        help=(
            "Start from a local smoke-test configuration; explicit flags still "
            "override it."
        ),
    )
    parser.add_argument("--kernel-implementation", choices=("xla", "sm90", "sm100"))
    parser.add_argument("--pp", type=int)
    parser.add_argument(
        "--data", type=int, help="Replicas of the training state (gradient all-reduce)."
    )
    parser.add_argument(
        "--fsdp", type=int, help="Ranks sharding the training state under --zero."
    )
    parser.add_argument("--ep", type=int)
    parser.add_argument(
        "--zero",
        type=int,
        choices=(1, 2, 3),
        help=(
            "What FSDP shards: 1=optimizer state, 2=also accumulated gradients, "
            "3=also compute parameters, gathered within each layer (default: 3)."
        ),
    )
    parser.add_argument(
        "--host-offload-regex",
        type=nullable_str,
        metavar="REGEX|None",
        help=(
            "Store matching train-state leaves in pinned host memory between uses. "
            "Uses re.search on JAX key paths, e.g. '^\\.(params|opt)\\.' for master "
            "parameters and Adam state, or '^\\.opt\\.' for Adam state only. "
            "Pass None to disable offloading."
        ),
    )
    parser.add_argument("--schedule", choices=("1f1b", "kimik2"))
    parser.add_argument("--num-pipeline-repeats", type=int)
    parser.add_argument(
        "--pipeline-stage-sizes",
        type=lambda value: tuple(int(size) for size in value.split(",")),
    )
    parser.add_argument("--fuse-steady-state", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--expert-gradient-accumulation",
        action=argparse.BooleanOptionalAction,
        help="Use gradient Refs to accumulate expert dW in the ragged kernels.",
    )
    parser.add_argument(
        "--compress-expert-master-params",
        action=argparse.BooleanOptionalAction,
        help=(
            "Losslessly encode on-device FP32 expert masters across the pipeline "
            "using three bytes per parameter plus existing BF16 compute weights. "
            "Preserves expert gradient Refs and gathers weights once per step. "
            "See FlashOptim (https://arxiv.org/abs/2602.23349, Sec. 3.1) for "
            "related weight splitting; this encoding preserves exact FP32 bits."
        ),
    )
    parser.add_argument("--use-ring-of-experts", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--ep-as-context",
        action=argparse.BooleanOptionalAction,
        help=(
            "Shard sequence over the expert axis and transpose sequence to "
            "attention heads around cuDNN attention."
        ),
    )
    parser.add_argument("--per-device-batch-size", type=float)
    parser.add_argument("--num-microbatches", type=int)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--profile-dir", type=nullable_str, metavar="PATH|None")
    parser.add_argument(
        "--warmup-tasks",
        action=argparse.BooleanOptionalAction,
        help="Execute disposable task inputs before initializing train state.",
    )
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-dense-layers", type=int)
    parser.add_argument("--emb-dim", type=int)
    parser.add_argument("--dense-mlp-dim", type=int)
    parser.add_argument("--moe-mlp-dim", type=int)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--num-experts-per-tok", type=int)
    parser.add_argument(
        "--capacity-factor",
        type=float,
        help=(
            "Bound padded route-buffer memory by dropping excess assignments per "
            "original sequence/expert (shared across --ep-as-context shards); "
            "negative disables dropping."
        ),
    )
    parser.add_argument("--num-routing-groups", type=int)
    parser.add_argument("--num-selected-groups", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--num-query-heads", type=int)
    parser.add_argument("--q-lora-rank", type=int)
    parser.add_argument("--kv-lora-rank", type=int)
    parser.add_argument("--qk-nope-head-dim", type=int)
    parser.add_argument("--qk-rope-head-dim", type=int)
    parser.add_argument("--v-head-dim", type=int)
    parser.add_argument("--moe-aux-loss-weight", type=float)
    parser.add_argument("--router-bias-update-rate", type=float)
    parser.add_argument("--mtp-num-layers", type=int)
    parser.add_argument(
        "--param-dtype",
        choices=("fp32", "bf16"),
        help="Master-weight storage dtype (default: fp32); compute remains BF16.",
    )
    args = vars(parser.parse_args())
    if "param_dtype" in args:
        args["param_dtype"] = {"fp32": jnp.float32, "bf16": jnp.bfloat16}[
            args["param_dtype"]
        ]
    preset = args.pop("preset", None)
    config = dataclasses.replace(
        tiny_config() if preset == "tiny" else Config(), **args
    )
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
    gpu_arch = gpu.architecture()
    logger.info("Selected GPU architecture: %s", gpu_arch.value)
    logger.info("Configuration: %s", config)
    logger.info(
        "Batch geometry: per_device_batch_size=%g, global_batch_size=%d, "
        "microbatch_size=%d, num_microbatches=%d",
        config.per_device_batch_size,
        config.global_batch_size,
        config.microbatch_size,
        config.num_microbatches,
    )

    data_started_at = time.perf_counter()
    shape = (config.num_microbatches, config.microbatch_size, config.seq_length)
    token_shape = (*shape[:-1], shape[-1] + 1)
    tokens = np.arange(math.prod(token_shape), dtype=np.int32).reshape(token_shape)
    tokens %= config.vocab_size
    batch = Batch(
        inputs=tokens[..., :-1],
        targets=tokens[..., 1:],
        loss_mask=np.ones(shape, dtype=np.float32),
    )
    logger.info(
        "Prepared the synthetic batch in %.3f seconds",
        time.perf_counter() - data_started_at,
    )
    # The trainer traces with size-one mesh axes dropped from array types.
    with jax.remove_size_one_mesh_axis_from_type(True):
        state = train_loop(config, batch)
    jax.block_until_ready(state)
    logger.info(
        "Program completed in %.3f seconds", time.perf_counter() - program_started_at
    )
    if distributed:
        shutdown_started_at = time.perf_counter()
        jax.distributed.shutdown()
        logger.info(
            "Shut down distributed JAX in %.3f seconds",
            time.perf_counter() - shutdown_started_at,
        )


if __name__ == "__main__":
    main()
