# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools

import jax
import jax.numpy as jnp

try:
    from jax._src.cudnn import fused_attention_stablehlo as fa_hlo
    from jax._src.lib import cuda_versions

except ImportError:
    fa_hlo = None


def _dot_product_attention_fwd(
    query,
    key,
    value,
    bias,
    mask,
    q_seqlen,
    kv_seqlen,
    scale,
    seed,
    dropout_rate,
    variadic_args,
    mask_type,
    layout,
    is_training,
):
    outputs = fa_hlo._dot_product_attention_fwd_p.bind(
        query,
        key,
        value,
        bias,
        mask,
        q_seqlen,
        kv_seqlen,
        scale=scale,
        seed=seed,
        dropout_rate=dropout_rate,
        variadic_args=variadic_args,
        mask_type=mask_type,
        layout=layout,
        is_training=is_training,
    )
    output = outputs[0]
    return output


def _dot_product_attention_fwd_rule(
    query,
    key,
    value,
    bias,
    mask,
    q_seqlen,
    kv_seqlen,
    scale,
    seed,
    dropout_rate,
    variadic_args,
    mask_type,
    layout,
    is_training,
):
    outputs = fa_hlo._dot_product_attention_fwd_p.bind(
        query,
        key,
        value,
        bias,
        mask,
        q_seqlen,
        kv_seqlen,
        scale=scale,
        seed=seed,
        dropout_rate=dropout_rate,
        variadic_args=variadic_args,
        mask_type=mask_type,
        layout=layout,
        is_training=is_training,
    )
    res = (
        (query, key, value, bias, mask, q_seqlen, kv_seqlen, outputs[1], outputs[0])
        if is_training
        else None
    )
    return outputs[0], res


def _dot_product_attention_bwd_rule(
    scale,
    seed,
    dropout_rate,
    variadic_args,
    mask_type,
    layout,
    is_training,
    res,
    grad_output,
):
    (query, key, value, bias, mask, q_seqlen, kv_seqlen, activation, fwd_output) = res
    grads = fa_hlo._dot_product_attention_bwd_p.bind(
        query,
        key,
        value,
        bias,
        mask,
        q_seqlen,
        kv_seqlen,
        activation,
        fwd_output,
        grad_output,
        scale=scale,
        seed=seed,
        dropout_rate=dropout_rate,
        variadic_args=variadic_args,
        mask_type=mask_type,
        layout=layout,
    )
    grads = (*grads,) + (None,) * (7 - len(grads))
    return grads


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: jax.Array,
    mask: jax.Array,
    q_seqlen: jax.Array,
    kv_seqlen: jax.Array,
    scale: float,
    seed: int,
    dropout_rate: float,
    variadic_args: tuple[bool, ...],
    mask_type: bool,
    layout: int,
    is_training: bool,
):
    output = _dot_product_attention_fwd(
        query,
        key,
        value,
        bias,
        mask,
        q_seqlen,
        kv_seqlen,
        scale=scale,
        seed=seed,
        dropout_rate=dropout_rate,
        variadic_args=variadic_args,
        mask_type=mask_type,
        layout=layout,
        is_training=is_training,
    )
    return output


_dot_product_attention.defvjp(
    _dot_product_attention_fwd_rule, _dot_product_attention_bwd_rule
)

if fa_hlo is None or jax.__version_info__ < (0, 4, 29):

    def self_attention(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        bias: jax.Array,
        mask: jax.Array | None,
        scale: float,
        seed: int,
        dropout_rate: float,
        layout: int,
        is_training: bool,
    ):
        raise ValueError("hlo cudnn self_attention called but is not supported")
else:

    def self_attention(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        bias: jax.Array,
        mask: jax.Array | None,
        scale: float,
        seed: int,
        dropout_rate: float,
        is_training: bool,
    ):
        if mask is not None:
            raise NotImplementedError

        mask = jnp.zeros(0, dtype=query.dtype)
        cudnn_version = cuda_versions.cudnn_get_version()

        layout = fa_hlo._normalize_layout("BTNH")
        has_bias = False
        has_mask = False
        has_dbias = (
            has_bias
            and is_training
            and fa_hlo.should_export_dbias(bias.shape, query.shape, layout)
        )
        variadic_args = (has_bias, has_mask, has_dbias)
        mask_type = fa_hlo.MaskType.CAUSAL
        fa_hlo.check_layout(query, key, value, None, None, None, layout)
        # check if flash attention is supported for this attention pattern
        fa_hlo.check_is_flash_attention(
            query, key, layout, cudnn_version, False, is_training
        )

        return _dot_product_attention(
            query=query,
            key=key,
            value=value,
            bias=jnp.zeros(0, dtype=query.dtype),
            mask=mask,
            q_seqlen=jnp.zeros(0, dtype=jnp.int32),
            kv_seqlen=jnp.zeros(0, dtype=jnp.int32),
            scale=scale,
            seed=seed,
            dropout_rate=dropout_rate,
            variadic_args=variadic_args,
            mask_type=mask_type,
            layout=layout,
            is_training=is_training,
        )
