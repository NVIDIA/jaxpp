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

"""Minimal MPMD 1F1B transformer example.

The jaxpp MPMD APIs support both automatic and explicit mesh axes. This
example chooses explicit mesh axes so it can also demonstrate JAX reduced axes:
FSDP-sharded weights are marked reduced after all-gather, while accumulated
grads stay unreduced across microbatches so each stage delays its FSDP
reduce-scatter until its final backward task.

Example usage:
XLA_FLAGS=--xla_force_host_platform_device_count=2 python examples/mpmd.py

For visible FSDP collectives on four local devices:
python examples/mpmd.py --num_procs=2 --gpus_per_proc=2 --fsdp=2
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.api import VJP
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpp.experimental import mpmd


@jax.tree_util.register_static
@dataclass(frozen=True)
class Config:
    pp: int = 2
    fsdp: int = 1
    num_layers: int = 6
    batch_size: int = 64
    seq_length: int = 512
    embed_dim: int = 1024
    num_heads: int = 8
    mlp_dim: int = 4096
    dtype: Any = jnp.float32
    param_seed: int = 12738
    data_seed: int = 4289
    dump_jaxprs: bool = False

    def __post_init__(self):
        if self.pp < 1:
            raise ValueError("pp must be positive")
        if self.fsdp < 1:
            raise ValueError("fsdp must be positive")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.num_layers % self.pp:
            raise ValueError(f"num_layers must be divisible by pp={self.pp}")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

    @property
    def layers_per_stage(self) -> int:
        return self.num_layers // self.pp

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def from_argv(cls, argv=None) -> "Config":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--pp", type=int, default=cls.pp)
        parser.add_argument("--fsdp", type=int, default=cls.fsdp)
        parser.add_argument("--num-layers", type=int, default=cls.num_layers)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--seq-length", type=int, default=cls.seq_length)
        parser.add_argument("--embed-dim", type=int, default=cls.embed_dim)
        parser.add_argument("--num-heads", type=int, default=cls.num_heads)
        parser.add_argument("--mlp-dim", type=int, default=cls.mlp_dim)
        parser.add_argument("--param-seed", type=int, default=cls.param_seed)
        parser.add_argument("--data-seed", type=int, default=cls.data_seed)
        parser.add_argument("--dump_jaxprs", "--dump-jaxprs", action="store_true")
        args = parser.parse_args(argv)
        return cls(
            pp=args.pp,
            fsdp=args.fsdp,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            param_seed=args.param_seed,
            data_seed=args.data_seed,
            dump_jaxprs=args.dump_jaxprs,
        )


class TransformerLayerParams(NamedTuple):
    qkv: Any
    attn_out: Any
    mlp_in: Any
    mlp_out: Any


def rms_norm(x):
    scale = jax.lax.rsqrt(
        jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6
    )
    return x * scale.astype(x.dtype)


def transformer_layer(params, activation, *, qkv_out_pspec, feature_out_pspec):
    qkv = jnp.einsum(
        "bsd,3dkh->bs3kh", activation, params.qkv, out_sharding=qkv_out_pspec
    )
    attn = jax.nn.dot_product_attention(
        qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :], is_causal=True
    )
    attn = jnp.einsum(
        "bskh,khd->bsd", attn, params.attn_out, out_sharding=feature_out_pspec
    )
    activation = rms_norm(activation + attn)

    hidden = jnp.einsum(
        "bsd,dh->bsh", activation, params.mlp_in, out_sharding=feature_out_pspec
    )
    hidden = jax.nn.gelu(hidden)
    hidden = jnp.einsum(
        "bsh,hd->bsd", hidden, params.mlp_out, out_sharding=feature_out_pspec
    )
    return rms_norm(activation + hidden)


def transformer_stack(params, activation, *, qkv_out_pspec, feature_out_pspec):
    for layer_params in params:
        activation = transformer_layer(
            layer_params,
            activation,
            qkv_out_pspec=qkv_out_pspec,
            feature_out_pspec=feature_out_pspec,
        )
    return activation


def loss_and_output_grad(out, target):
    diff = out.astype(jnp.float32) - target.astype(jnp.float32)
    loss = jnp.mean(diff * diff)
    out_grad = (jnp.asarray(2, diff.dtype) / out.size) * diff
    return loss, out_grad.astype(out.dtype)


def to_reduced(sharding_or_pspec, axis_name):
    def is_sharding_or_pspec(value):
        return isinstance(value, (NamedSharding, P))

    def reduce_partition_spec(spec):
        def erase_axis(partition):
            if partition is None or partition is P.UNCONSTRAINED:
                return partition
            if isinstance(partition, tuple):
                axes = tuple(axis for axis in partition if axis != axis_name)
                return axes if axes else None
            return None if partition == axis_name else partition

        return P(
            *(erase_axis(partition) for partition in spec.partitions),
            unreduced=spec.unreduced - {axis_name},
            reduced=spec.reduced | {axis_name},
        )

    def reduce_leaf(leaf):
        if isinstance(leaf, NamedSharding):
            return leaf.update(spec=reduce_partition_spec(leaf.spec))
        return reduce_partition_spec(leaf)

    if is_sharding_or_pspec(sharding_or_pspec):
        return reduce_leaf(sharding_or_pspec)
    return jax.tree_util.tree_map(
        reduce_leaf, sharding_or_pspec, is_leaf=is_sharding_or_pspec
    )


def folded_keys(seed):
    root_key = jax.random.key(seed)
    idx = 0
    while True:
        yield jax.random.fold_in(root_key, idx)
        idx += 1


def process_has_sharding(sharding):
    process_index = jax.process_index()
    return any(
        device.process_index == process_index for device in sharding.mesh.devices.flat
    )


def empty_array(shape_dtype):
    return jax.make_array_from_single_device_arrays(
        shape=shape_dtype.shape,
        sharding=shape_dtype.sharding,
        arrays=[],
        dtype=shape_dtype.dtype,
    )


def sharded_init(initializer, key, shape_dtype):
    if not process_has_sharding(shape_dtype.sharding):
        return empty_array(shape_dtype)
    with jax.set_mesh(shape_dtype.sharding.mesh):
        return initializer(
            key, shape_dtype.shape, shape_dtype.dtype, shape_dtype.sharding
        )


def normal_init(key, shape, dtype, sharding):
    return jax.random.normal(key, shape, dtype, out_sharding=sharding)


def init_param_state(config, param_shapes):
    he_init = jax.nn.initializers.he_normal(1, 1)
    keys = folded_keys(config.param_seed)

    def init_layer(layer_shapes):
        return TransformerLayerParams(
            qkv=sharded_init(he_init, next(keys), layer_shapes.qkv),
            attn_out=sharded_init(he_init, next(keys), layer_shapes.attn_out),
            mlp_in=sharded_init(he_init, next(keys), layer_shapes.mlp_in),
            mlp_out=sharded_init(he_init, next(keys), layer_shapes.mlp_out),
        )

    return tuple(
        tuple(init_layer(layer_shapes) for layer_shapes in stage_shapes)
        for stage_shapes in param_shapes
    )


def init_arrays(keys, array_shapes):
    return tuple(
        sharded_init(normal_init, next(keys), shape_dtype)
        for shape_dtype in array_shapes
    )


def init_inputs(config, input_shapes):
    param_shapes, xs_shapes, target_shapes = input_shapes
    data_keys = folded_keys(config.data_seed)
    return (
        init_param_state(config, param_shapes),
        init_arrays(data_keys, xs_shapes),
        init_arrays(data_keys, target_shapes),
    )


def main(argv=None):
    config = Config.from_argv(argv)
    if config.pp != 2:
        raise ValueError(f"examples/mpmd.py only supports pp=2, got pp={config.pp}")
    assert jax.process_count() == 2, f"expected 2 processes, got {jax.process_count()}"
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.size < 2:
        raise RuntimeError(
            "examples/mpmd.py requires at least two devices. On CPU, run with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2."
        )

    ep = devices.size // (config.pp * config.fsdp)
    if ep < 1:
        raise RuntimeError(
            "examples/mpmd.py requires at least "
            f"{config.pp * config.fsdp} devices for "
            f"pp={config.pp}, fsdp={config.fsdp}"
        )
    devices = devices[: config.pp * config.fsdp * ep].reshape(
        (1, config.pp, config.fsdp, ep)
    )
    mpmd_mesh = mpmd.MpmdMesh(
        Mesh(
            devices,
            ("replica", "pp", "fsdp", "ep"),
            axis_types=(AxisType.Explicit,) * 4,
        ),
        "pp",
    )

    replicated_pspec = P()
    fsdp_3d_pspec = P("fsdp", None, None)
    qkv_out_pspec = P("fsdp", None, None, None, None)
    qkv_param_pspec = P(None, "fsdp", None, None)
    mlp_in_param_pspec = P("fsdp", None)
    mlp_out_param_pspec = P(None, "fsdp")

    stage_0_activation_sharding = NamedSharding(mpmd_mesh.unstack[0], fsdp_3d_pspec)
    stage_1_activation_sharding = NamedSharding(mpmd_mesh.unstack[1], fsdp_3d_pspec)
    stage_1_replicated_sharding = NamedSharding(mpmd_mesh.unstack[1], replicated_pspec)

    def layer_param_shardings(mesh):
        return TransformerLayerParams(
            qkv=NamedSharding(mesh, qkv_param_pspec),
            attn_out=NamedSharding(mesh, fsdp_3d_pspec),
            mlp_in=NamedSharding(mesh, mlp_in_param_pspec),
            mlp_out=NamedSharding(mesh, mlp_out_param_pspec),
        )

    def layer_shapes(shardings):
        return TransformerLayerParams(
            qkv=jax.ShapeDtypeStruct(
                (3, config.embed_dim, config.num_heads, config.head_dim),
                config.dtype,
                sharding=shardings.qkv,
            ),
            attn_out=jax.ShapeDtypeStruct(
                (config.num_heads, config.head_dim, config.embed_dim),
                config.dtype,
                sharding=shardings.attn_out,
            ),
            mlp_in=jax.ShapeDtypeStruct(
                (config.embed_dim, config.mlp_dim),
                config.dtype,
                sharding=shardings.mlp_in,
            ),
            mlp_out=jax.ShapeDtypeStruct(
                (config.mlp_dim, config.embed_dim),
                config.dtype,
                sharding=shardings.mlp_out,
            ),
        )

    stage_0_param_shardings, stage_1_param_shardings = (
        layer_param_shardings(mpmd_mesh.unstack[0]),
        layer_param_shardings(mpmd_mesh.unstack[1]),
    )
    params = (
        tuple(
            layer_shapes(stage_0_param_shardings)
            for _ in range(config.layers_per_stage)
        ),
        tuple(
            layer_shapes(stage_1_param_shardings)
            for _ in range(config.layers_per_stage)
        ),
    )
    param_shardings = jax.tree.map(lambda shape_dtype: shape_dtype.sharding, params)
    params_bf16_shardings = (
        to_reduced(param_shardings, "fsdp") if config.fsdp > 1 else param_shardings
    )
    params_bf16_shapes = jax.tree.map(
        lambda shape_dtype, sharding: jax.ShapeDtypeStruct(
            shape_dtype.shape, jnp.bfloat16, sharding=sharding
        ),
        params,
        params_bf16_shardings,
    )

    def fwd(params, activation):
        return transformer_stack(
            params,
            activation,
            qkv_out_pspec=qkv_out_pspec,
            feature_out_pspec=fsdp_3d_pspec,
        )

    fwd_ad, bwd_ad = jax.fwd_and_bwd(fwd, argnums=(0, 1), jitted=False)

    def stage_forward(params, activation):
        out, vjp_residuals = fwd_ad(params, activation)
        return out, tuple(vjp_residuals.opaque_residuals)

    def stage_backward_grads(
        vjp_template: VJP, params, activation, saved_residuals, dact_out
    ):
        # Keep primal args explicit across tasks, then restore JAX's VJP object
        # at the backward boundary.
        vjp_residuals: VJP = replace(
            vjp_template,
            args_res=[params, activation],
            opaque_residuals=list(saved_residuals),
        )
        param_grad, dact_in = bwd_ad(vjp_residuals, dact_out)
        param_grad = jax.tree.map(lambda grad: grad.astype(jnp.float32), param_grad)
        return param_grad, dact_in

    activation_shape = (config.batch_size, config.seq_length, config.embed_dim)
    stage_0_activation_shape = jax.ShapeDtypeStruct(
        activation_shape, config.dtype, sharding=stage_0_activation_sharding
    )
    stage_1_activation_shape = jax.ShapeDtypeStruct(
        activation_shape, config.dtype, sharding=stage_1_activation_sharding
    )
    xs = (stage_0_activation_shape, stage_0_activation_shape, stage_0_activation_shape)
    targets = (
        stage_1_activation_shape,
        stage_1_activation_shape,
        stage_1_activation_shape,
    )
    input_shapes = (params, xs, targets)
    in_shardings = jax.tree.map(lambda shape_dtype: shape_dtype.sharding, input_shapes)

    def concrete_sharding_like(shape_dtype, mesh):
        sharding = shape_dtype.sharding
        if sharding is None:
            return NamedSharding(mesh, P(*(None for _ in shape_dtype.shape)))
        return NamedSharding(mesh, sharding.spec, memory_kind=sharding.memory_kind)

    stage_0_vjp_template: VJP
    stage_1_vjp_template: VJP
    with jax.set_mesh(mpmd_mesh.unstack[0]):
        _, stage_0_vjp_template = jax.eval_shape(
            fwd_ad, params_bf16_shapes[0], stage_0_activation_shape
        )
    with jax.set_mesh(mpmd_mesh.unstack[1]):
        _, stage_1_vjp_template = jax.eval_shape(
            fwd_ad, params_bf16_shapes[1], stage_1_activation_shape
        )
    stage_0_saved_residual_shapes = tuple(stage_0_vjp_template.opaque_residuals)
    stage_1_saved_residual_shapes = tuple(stage_1_vjp_template.opaque_residuals)
    # Demonstrate host offload without moving the full opaque AD residual tree.
    stage_0_offloaded_residual_count = 15
    stage_0_residual_shardings = tuple(
        concrete_sharding_like(shape_dtype, mpmd_mesh.unstack[0])
        for shape_dtype in stage_0_saved_residual_shapes
    )
    stage_1_residual_shardings = tuple(
        concrete_sharding_like(shape_dtype, mpmd_mesh.unstack[1])
        for shape_dtype in stage_1_saved_residual_shapes
    )
    stage_0_offloaded_residual_shardings = stage_0_residual_shardings[
        :stage_0_offloaded_residual_count
    ]
    stage_0_retained_residual_shardings = stage_0_residual_shardings[
        stage_0_offloaded_residual_count:
    ]
    stage_0_host_residual_shardings = jax.tree.map(
        lambda sharding: sharding.with_memory_kind("pinned_host"),
        stage_0_offloaded_residual_shardings,
    )

    def bwd0_grads_from_residuals(params, activation, saved_residuals, dact_out):
        return stage_backward_grads(
            stage_0_vjp_template, params, activation, saved_residuals, dact_out
        )

    def bwd1_grads_from_residuals(params, activation, saved_residuals, dact_out):
        return stage_backward_grads(
            stage_1_vjp_template, params, activation, saved_residuals, dact_out
        )

    def bwd0(grad_acc, params, activation, saved_residuals, dact_out):
        param_grad, dact_in = stage_backward_grads(
            stage_0_vjp_template, params, activation, saved_residuals, dact_out
        )
        return jax.tree.map(jnp.add, grad_acc, param_grad), dact_in

    def bwd1(grad_acc, params, activation, saved_residuals, dact_out):
        param_grad, dact_in = stage_backward_grads(
            stage_1_vjp_template, params, activation, saved_residuals, dact_out
        )
        return jax.tree.map(jnp.add, grad_acc, param_grad), dact_in

    with jax.set_mesh(mpmd_mesh.unstack[0]):
        stage_0_param_grad_shape, _ = jax.eval_shape(
            bwd0_grads_from_residuals,
            params_bf16_shapes[0],
            stage_0_activation_shape,
            stage_0_saved_residual_shapes,
            stage_0_activation_shape,
        )
    with jax.set_mesh(mpmd_mesh.unstack[1]):
        stage_1_param_grad_shape, _ = jax.eval_shape(
            bwd1_grads_from_residuals,
            params_bf16_shapes[1],
            stage_1_activation_shape,
            stage_1_saved_residual_shapes,
            stage_1_activation_shape,
        )

    stage_0_grad_shardings = jax.tree.map(
        lambda shape_dtype: concrete_sharding_like(shape_dtype, mpmd_mesh.unstack[0]),
        stage_0_param_grad_shape,
    )
    stage_1_grad_shardings = jax.tree.map(
        lambda shape_dtype: concrete_sharding_like(shape_dtype, mpmd_mesh.unstack[1]),
        stage_1_param_grad_shape,
    )
    grad_acc_specs = jax.tree.map(
        lambda sharding: sharding.spec, stage_0_grad_shardings
    )
    stage_0_bwd_out_shardings = (stage_0_grad_shardings, stage_0_activation_sharding)
    stage_1_fwd_bwd_out_shardings = (
        stage_1_replicated_sharding,
        stage_1_residual_shardings,
        stage_1_activation_sharding,
        stage_1_grad_shardings,
        stage_1_activation_sharding,
    )

    @mpmd.mpmd(mpmd_mesh, in_shardings=in_shardings, infer_donation=True)
    def two_stage_1f1b(params, xs, targets):
        act0_0, act0_1, act0_2 = xs
        target0, target1, target2 = targets

        def zero_grad_acc(params):
            return jax.tree.map(
                lambda param, spec: jnp.zeros_like(
                    param, dtype=jnp.float32, out_sharding=spec
                ),
                params,
                grad_acc_specs,
            )

        acc0 = mpmd.task(
            zero_grad_acc,
            name="zero_stage_0_grad_acc",
            out_shardings=stage_0_grad_shardings,
        )(params[0])
        acc1 = mpmd.task(
            zero_grad_acc,
            name="zero_stage_1_grad_acc",
            out_shardings=stage_1_grad_shardings,
        )(params[1])

        @mpmd.task(
            out_shardings=(
                params_bf16_shardings[0],
                stage_0_activation_sharding,
                stage_0_residual_shardings,
            )
        )
        def fwd0_0(params, act0_0):
            # all-gather params over "fsdp" after the cast.
            params = jax.reshard(
                jax.tree.map(lambda param: param.astype(jnp.bfloat16), params),
                params_bf16_shardings[0],
            )
            act1_0, res0_0 = stage_forward(params, act0_0)
            return params, act1_0, res0_0

        @mpmd.task(
            out_shardings=(
                stage_0_activation_sharding,
                stage_0_offloaded_residual_shardings,
                stage_0_retained_residual_shardings,
            )
        )
        def fwd0_1(params, act0_1):
            act1_1, res0_1 = stage_forward(params, act0_1)
            return (
                act1_1,
                res0_1[:stage_0_offloaded_residual_count],
                res0_1[stage_0_offloaded_residual_count:],
            )

        @mpmd.task(
            out_shardings=(
                stage_0_activation_sharding,
                stage_0_residual_shardings,
                stage_0_host_residual_shardings,
            )
        )
        def fwd0_2(params, act0_2, res0_1_offload):
            # Overlap offloading previous microbatch residuals with current compute.
            res0_1_cpu = jax.device_put(res0_1_offload, stage_0_host_residual_shardings)
            act1_2, res0_2 = stage_forward(params, act0_2)
            return act1_2, res0_2, res0_1_cpu

        params0, act1_0_s0, res0_0 = fwd0_0(params[0], act0_0)
        act1_0_fut = mpmd.transfer(act1_0_s0, out_shardings=stage_1_activation_sharding)

        act1_1_s0, res0_1_offload, res0_1_kept = fwd0_1(params0, act0_1)
        act1_1_fut = mpmd.transfer(act1_1_s0, out_shardings=stage_1_activation_sharding)

        act1_2_s0, res0_2, res0_1_cpu = fwd0_2(params0, act0_2, res0_1_offload)
        act1_2_fut = mpmd.transfer(act1_2_s0, out_shardings=stage_1_activation_sharding)

        act1_0 = act1_0_fut.done()

        @mpmd.task(
            out_shardings=(
                params_bf16_shardings[1],
                stage_1_replicated_sharding,
                stage_1_residual_shardings,
                stage_1_activation_sharding,
            )
        )
        def fwd1_0(params, act1_0, target0):
            # all-gather params over "fsdp".
            params = jax.reshard(
                jax.tree.map(lambda param: param.astype(jnp.bfloat16), params),
                params_bf16_shardings[1],
            )
            out, res1_0 = stage_forward(params, act1_0)
            loss1_0, dout1_0 = loss_and_output_grad(out, target0)
            return params, loss1_0, res1_0, dout1_0

        params1, loss1_0, res1_0, dout1_0 = fwd1_0(params[1], act1_0, target0)

        act1_1 = act1_1_fut.done()

        def fwd1_bwd1(
            params, acc1, act1, target, prev_activation, prev_res1, prev_dout1
        ):
            out, res1 = stage_forward(params, act1)
            acc1, dact1_prev = bwd1(
                acc1, params, prev_activation, prev_res1, prev_dout1
            )
            loss1, dout1 = loss_and_output_grad(out, target)
            return (loss1, res1, dout1, acc1, dact1_prev)

        (loss1_1, res1_1, dout1_1, acc1, dact1_0_s1) = mpmd.task(
            fwd1_bwd1, name="fwd1_1_bwd1_0", out_shardings=stage_1_fwd_bwd_out_shardings
        )(params1, acc1, act1_1, target1, act1_0, res1_0, dout1_0)
        dact1_0_fut = mpmd.transfer(
            dact1_0_s1, out_shardings=stage_0_activation_sharding
        )

        act1_2 = act1_2_fut.done()

        (loss1_2, res1_2, dout1_2, acc1, dact1_1_s1) = mpmd.task(
            fwd1_bwd1, name="fwd1_2_bwd1_1", out_shardings=stage_1_fwd_bwd_out_shardings
        )(params1, acc1, act1_2, target2, act1_1, res1_1, dout1_1)
        dact1_1_fut = mpmd.transfer(
            dact1_1_s1, out_shardings=stage_0_activation_sharding
        )

        def bwd0_0_and_onload_res0_1(
            acc0, params, activation, res0_0, dact1_0, res0_1_cpu
        ):
            res0_1_offload = tuple(
                jax.device_put(res0_1_cpu, stage_0_offloaded_residual_shardings)
            )
            acc0, dact0_0 = bwd0(acc0, params, activation, res0_0, dact1_0)
            return acc0, dact0_0, res0_1_offload

        dact1_0_s0 = dact1_0_fut.done()
        acc0, dact0_0, res0_1_offload = mpmd.task(
            bwd0_0_and_onload_res0_1,
            name="bwd0_0",
            out_shardings=(
                stage_0_grad_shardings,
                stage_0_activation_sharding,
                stage_0_offloaded_residual_shardings,
            ),
        )(acc0, params0, act0_0, res0_0, dact1_0_s0, res0_1_cpu)

        res0_1 = res0_1_offload + res0_1_kept
        dact1_1_s0 = dact1_1_fut.done()
        acc0, dact0_1 = mpmd.task(
            bwd0, name="bwd0_1", out_shardings=stage_0_bwd_out_shardings
        )(acc0, params0, act0_1, res0_1, dact1_1_s0)

        def bwd1_2(acc1, params, activation, res1_2, dout1_2):
            acc1, dact1_2 = bwd1(acc1, params, activation, res1_2, dout1_2)
            # reduce-scatter params over "fsdp".
            return jax.reshard(acc1, param_shardings[1]), dact1_2

        grad1, dact1_2_s1 = mpmd.task(
            bwd1_2, out_shardings=(param_shardings[1], stage_1_activation_sharding)
        )(acc1, params1, act1_2, res1_2, dout1_2)
        dact1_2_fut = mpmd.transfer(
            dact1_2_s1, out_shardings=stage_0_activation_sharding
        )

        dact1_2_s0 = dact1_2_fut.done()

        def bwd0_2(acc0, params, activation, res0_2, dact1_2):
            acc0, dact0_2 = bwd0(acc0, params, activation, res0_2, dact1_2)
            # reduce-scatter params over "fsdp".
            return jax.reshard(acc0, param_shardings[0]), dact0_2

        grad0, dact0_2 = mpmd.task(
            bwd0_2, out_shardings=(param_shardings[0], stage_0_activation_sharding)
        )(acc0, params0, act0_2, res0_2, dact1_2_s0)

        return (
            (loss1_0, loss1_1, loss1_2),
            (grad0, grad1),
            (dact0_0, dact0_1, dact0_2),
        )

    if config.dump_jaxprs:
        global_jaxpr = jax.make_jaxpr(two_stage_1f1b)(*input_shapes).jaxpr
        print("====== Global Jaxpr ======")
        print(global_jaxpr)
    lowered = two_stage_1f1b.lower(*input_shapes)
    if mpmd_mesh.jax_mesh.is_multi_process:
        if config.dump_jaxprs:
            print("====== Local Jaxpr ======")
            print(lowered._local_jaxpr)
        train_step = lowered
    else:
        if config.dump_jaxprs:
            print("====== Local Jaxpr ======")
            for mpmd_idx, local_jaxpr in enumerate(lowered.local_jaxprs):
                print(f"------ MPMD idx {mpmd_idx} ------")
                print(local_jaxpr)
        train_step = two_stage_1f1b

    args = init_inputs(config, input_shapes)
    profile_root = Path("./profile")
    profile_dir = profile_root / f"process_{jax.process_index()}"

    with jax.profiler.StepTraceAnnotation("mpmd_lowered", step_num=0):
        outputs = train_step(*args)
    jax.block_until_ready(outputs)

    print(f"profiling MPMD lowered step to {profile_root}")
    multihost_utils.sync_global_devices("mpmd_profile_ready")
    jax.profiler.start_trace(str(profile_dir))
    multihost_utils.sync_global_devices("mpmd_profile_started")
    with jax.profiler.StepTraceAnnotation("mpmd_lowered", step_num=1):
        outputs = train_step(*args)
    jax.block_until_ready(outputs)
    multihost_utils.sync_global_devices("mpmd_profile_done")
    jax.profiler.stop_trace()


def _launcher_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_procs", "--num-procs", type=int, default=2)
    parser.add_argument("--gpus_per_proc", "--gpus-per-proc", type=int, default=1)
    parser.add_argument(
        "--coordinator_address", "--coordinator-address", default="127.0.0.1:5678"
    )
    return parser.parse_known_args(argv)


def _worker_main(
    process_id, num_processes, coordinator_address, local_device_ids, argv
):
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
        cluster_detection_method="deactivate",
    )
    try:
        main(argv)
    finally:
        jax.distributed.shutdown()


def _run_local_multiprocess(argv=None):
    launcher_args, model_args = _launcher_args(argv)
    num_processes = launcher_args.num_procs
    local_devices_per_process = launcher_args.gpus_per_proc
    coordinator_address = launcher_args.coordinator_address
    context = mp.get_context("spawn")
    processes = []
    exit_code = 0

    try:
        for process_id in range(num_processes):
            first_device = process_id * local_devices_per_process
            local_device_ids = list(
                range(first_device, first_device + local_devices_per_process)
            )
            process = context.Process(
                target=_worker_main,
                args=(
                    process_id,
                    num_processes,
                    coordinator_address,
                    local_device_ids,
                    model_args,
                ),
            )
            process.start()
            processes.append(process)

        while any(process.is_alive() for process in processes):
            for process in processes:
                if process.exitcode not in (None, 0):
                    exit_code = process.exitcode
                    raise SystemExit(exit_code)
            time.sleep(0.1)

        for process in processes:
            if process.exitcode:
                exit_code = process.exitcode
                break
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    _run_local_multiprocess()
