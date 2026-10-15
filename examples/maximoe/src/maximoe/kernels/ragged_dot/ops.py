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

"""Backend-independent ragged-dot operations and their differentiation rules.

Three HiJAX primitives compose the raw kernels of one backend module into
differentiable operations: :class:`RaggedDot`, :class:`RaggedDotSwiGLU` (the
gate/up projection with its activation epilogue), and :class:`RaggedDotCombine`
(the down projection with the routing ``combine`` epilogue). The primitives own
the backward rules, so backends only implement forward kernels, gradient
kernels, and in-place weight-gradient accumulation. When JAX hands a backward
rule a Ref accumulator for the weights, the dW kernel adds into it directly.

Every backend module exposes the same surface: ``ragged_dot``, ``ragged_dot_dx``,
``ragged_dot_dw``, ``ragged_dot_swiglu``, and ``swiglu_backward``. The ``xla``
module is also the specification: JAX differentiates it independently of these
rules, and tests compare each Mosaic backend against it.
"""

import types

import jax
import jax.numpy as jnp
from jax import ad_checkpoint as adc
from jax._src.hijax import fake_linear_op
from jax.experimental import hijax

from maximoe.kernels import backend, moe_routing
from maximoe.kernels.ragged_dot import sm90, sm100, xla

# HiPrim was named VJPHiPrimitive in the pinned JAX 0.11.1 revision.
HiPrim = getattr(hijax, "HiPrim", None) or hijax.VJPHiPrimitive

IMPLEMENTATIONS = types.MappingProxyType(
    {
        backend.Implementation.XLA: xla,
        backend.Implementation.SM90: sm90,
        backend.Implementation.SM100: sm100,
    }
)


def get_implementation(implementation: str | None = None) -> types.ModuleType:
    """Return the backend module; ``None`` selects the local platform's default."""
    return IMPLEMENTATIONS[backend.resolve(implementation)]


def materialize_weights(rhs):
    """Cast weights that are ``reduced`` over replica axes to ``varying``.

    Expert weights reach these operations reduced over the non-expert mesh axes.
    Casting inside the primitive keeps every kernel replica-local, and the
    weight cotangent is then ``unreduced`` over those axes, which
    ``ragged_dot_dw`` produces through ``output_manual_axis_type``.
    :func:`promote_varying` mirrors the cast in the output type.
    """
    return jax.lax.pcast(
        rhs, tuple(jax.typeof(rhs).manual_axis_type.reduced), to="varying"
    )


def weight_varying(rhs):
    """Manual axes a kernel output varies on because of its weight operand."""
    return rhs.manual_axis_type.varying | rhs.manual_axis_type.reduced


def promote_varying(template, shape, dtype, varying):
    """Output type derived from ``template`` that also varies on ``varying``."""
    axis_type = template.manual_axis_type
    varying = axis_type.varying | varying
    return hijax.ShapedArray(
        shape,
        dtype,
        sharding=template.sharding.update(spec=jax.P()),
        manual_axis_type=axis_type.update(
            varying=varying,
            reduced=axis_type.reduced - varying,
            unreduced=axis_type.unreduced - varying,
        ),
    )


class RaggedOperation(HiPrim):
    """Kernel lookup, binding, weight accumulation, and remat shared by all ops."""

    @classmethod
    def apply(cls, *args, **params):
        """Construct the primitive from the types of ``args`` and apply it."""
        return cls(*jax.tree.map(jax.typeof, args), **params)(*args)

    @property
    def kernels(self) -> types.ModuleType:
        return IMPLEMENTATIONS[self.implementation]

    def input_gradient(self, dout, rhs, group_sizes):
        return self.kernels.ragged_dot_dx(
            dout,
            materialize_weights(rhs),
            group_sizes,
            cost_estimate_num_rows=self.cost_estimate_num_rows,
        )

    def weight_gradient(self, acc, lhs, dout, group_sizes, dtype):
        """dW for the weight accumulator ``acc``.

        A caller-owned Ref is updated in place by the kernel and ``None`` is
        returned. Otherwise the gradient is returned in ``dtype`` for ``acc``.
        """
        if isinstance(acc, hijax.NullAccum):
            return None
        ref = acc.inst().ref if isinstance(acc, hijax.RefAccum) else None
        dw = self.kernels.ragged_dot_dw(
            lhs,
            dout,
            group_sizes,
            output_manual_axis_type=acc.aval.manual_axis_type,
            accumulator=ref,
            cost_estimate_num_rows=self.cost_estimate_num_rows,
        )
        return None if dw is None else dw.astype(dtype)

    def jvp(self, primals, tangents):
        """Keep the accumulator-aware backward under legacy checkpoint.

        ``jax.checkpoint`` linearizes through JVP. Like custom_vjp, this is a
        transpose-only bridge to ``vjp_bwd`` rather than a derivative of the
        kernels.
        """
        tangents = jax.tree.leaves(
            tangents, is_leaf=lambda x: isinstance(x, hijax.Zero)
        )
        nzs = [not isinstance(t, hijax.Zero) for t in tangents]
        out, residuals = self.vjp_fwd(nzs, *primals)
        (tangent,) = fake_linear_op(self, nzs, [True], residuals, None, *tangents)
        return out, tangent


class RaggedDot(RaggedOperation):
    """Grouped matmul with direct accumulation of weight cotangents."""

    def __init__(
        self,
        lhs,
        rhs,
        group_sizes,
        cost_estimate_num_rows=None,
        output_dtype=None,
        *,
        implementation=None,
    ):
        self.in_avals = (lhs, rhs, group_sizes)
        output_dtype = jnp.dtype(lhs.dtype if output_dtype is None else output_dtype)
        self.out_aval = promote_varying(
            lhs,
            self.output_shape(lhs, rhs),
            output_dtype,
            weight_varying(rhs) | group_sizes.manual_axis_type.varying,
        )
        self.params = dict(
            cost_estimate_num_rows=cost_estimate_num_rows,
            output_dtype=output_dtype,
            implementation=backend.resolve(implementation),
        )
        super().__init__()

    def output_shape(self, lhs, rhs):
        return (lhs.shape[0], rhs.shape[-1])

    def expand(self, lhs, rhs, group_sizes):
        return self.kernels.ragged_dot(
            lhs,
            materialize_weights(rhs),
            group_sizes,
            self.cost_estimate_num_rows,
            output_dtype=self.output_dtype,
        )

    def vjp_fwd(self, nzs, lhs, rhs, group_sizes):
        return self.expand(lhs, rhs, group_sizes), (lhs, rhs, group_sizes)

    def vjp_bwd(self, residuals, dout, lhs_acc, rhs_acc, sizes_acc):
        lhs, rhs, group_sizes = residuals
        drhs = self.weight_gradient(
            rhs_acc, lhs, dout.astype(lhs.dtype), group_sizes, rhs.dtype
        )
        if drhs is not None:
            # Schedule the weight gradient before the input gradient.
            dout, rhs, drhs = jax.lax.optimization_barrier((dout, rhs, drhs))
            rhs_acc.accum(drhs)
        if not isinstance(lhs_acc, hijax.NullAccum):
            lhs_acc.accum(self.input_gradient(dout, rhs, group_sizes).astype(lhs.dtype))


class RaggedDotSwiGLU(RaggedDot):
    """Grouped gate/up projection and SwiGLU, allowing a fused backend epilogue."""

    def output_shape(self, lhs, rhs):
        return (lhs.shape[0], rhs.shape[-1] // 2)

    def project(self, lhs, rhs, group_sizes, *, save_preact):
        return self.kernels.ragged_dot_swiglu(
            lhs,
            materialize_weights(rhs),
            group_sizes,
            self.output_dtype,
            self.cost_estimate_num_rows,
            save_preact=save_preact,
        )

    def expand(self, lhs, rhs, group_sizes):
        return self.project(lhs, rhs, group_sizes, save_preact=False)

    def vjp_fwd(self, nzs, lhs, rhs, group_sizes):
        out, preact = self.project(lhs, rhs, group_sizes, save_preact=True)
        return out, (lhs, rhs, group_sizes, preact)

    def vjp_bwd(self, residuals, dout, lhs_acc, rhs_acc, sizes_acc):
        lhs, rhs, group_sizes, preact = residuals
        dpreact = self.kernels.swiglu_backward(
            preact, dout, group_sizes, self.cost_estimate_num_rows
        )
        super().vjp_bwd((lhs, rhs, group_sizes), dpreact, lhs_acc, rhs_acc, sizes_acc)


class RaggedDotCombine(RaggedOperation):
    """Grouped down projection followed by the routing ``combine``.

    The backward composes ``combine_transpose`` and ``weight_expert_rows`` around
    the projection's gradient kernels, so the projected expert rows are never
    stored and dW accumulates in place. With ``token_group_sizes`` only each
    batch element's live token prefix of the output is defined.

    ``eager_weight_gradient`` makes the input gradient wait for dW. Nothing
    else consumes dW, so XLA may otherwise defer it to the end of the backward
    pass and keep every layer's expert cotangent alive until then.
    """

    def __init__(
        self,
        intermediate,
        wo,
        group_sizes,
        top_k_weights,
        positions,
        token_group_sizes,
        activations_in_float32,
        cost_estimate_num_rows,
        *,
        implementation=None,
        eager_weight_gradient=False,
    ):
        self.in_avals = (
            intermediate,
            wo,
            group_sizes,
            top_k_weights,
            positions,
            token_group_sizes,
        )
        # Combine promotes the projection to every routing operand's varying axes.
        varying = weight_varying(wo).union(
            *(
                aval.manual_axis_type.varying
                for aval in self.in_avals[2:]
                if aval is not None
            )
        )
        self.out_aval = promote_varying(
            intermediate,
            (*positions.shape[:-1], wo.shape[-1]),
            jnp.dtype(jnp.float32 if activations_in_float32 else intermediate.dtype),
            varying,
        )
        self.params = dict(
            activations_in_float32=activations_in_float32,
            cost_estimate_num_rows=cost_estimate_num_rows,
            implementation=backend.resolve(implementation),
            eager_weight_gradient=eager_weight_gradient,
        )
        super().__init__()

    def project(
        self, intermediate, wo, group_sizes, top_k_weights, positions, token_group_sizes
    ):
        expert_output = self.kernels.ragged_dot(
            intermediate,
            materialize_weights(wo),
            group_sizes,
            self.cost_estimate_num_rows,
        )
        expert_output = adc.checkpoint_name(expert_output, "mlpwo")
        return moe_routing.combine(
            expert_output,
            top_k_weights,
            positions,
            self.activations_in_float32,
            token_group_sizes,
            implementation=self.implementation,
        )

    def expand(self, *args):
        return self.project(*args)

    def vjp_fwd(self, nzs, *args):
        return self.project(*args), args

    def vjp_bwd(
        self, residuals, dout, x_acc, wo_acc, sizes_acc, weights_acc, *route_accs
    ):
        intermediate, wo, group_sizes, top_k_weights, positions, token_group_sizes = (
            residuals
        )
        rows = self.cost_estimate_num_rows
        expert_cotangent = moe_routing.combine_transpose(
            dout.astype(intermediate.dtype),
            positions,
            intermediate.shape[0],
            rows,
            token_group_sizes,
            implementation=self.implementation,
        )
        z = self.input_gradient(expert_cotangent, wo, group_sizes).astype(
            intermediate.dtype
        )
        expert_cotangent, z = jax.lax.optimization_barrier((expert_cotangent, z))
        dintermediate, dweights = moe_routing.weight_expert_rows(
            z,
            top_k_weights,
            positions,
            intermediate,
            rows,
            implementation=self.implementation,
        )
        weighted_cotangent = moe_routing.weight_expert_rows(
            expert_cotangent,
            top_k_weights,
            positions,
            None,
            rows,
            implementation=self.implementation,
        )
        dwo = self.weight_gradient(
            wo_acc, intermediate, weighted_cotangent, group_sizes, wo.dtype
        )
        if self.eager_weight_gradient and not isinstance(wo_acc, hijax.NullAccum):
            # An in-place kernel updated the Ref; its value marks completion.
            done = wo_acc.inst().ref[...] if dwo is None else dwo
            done, dintermediate = jax.lax.optimization_barrier((done, dintermediate))
            dwo = None if dwo is None else done
        if dwo is not None:
            wo_acc.accum(dwo)
        x_acc.accum(dintermediate)
        weights_acc.accum(dweights)


def ragged_dot(
    lhs,
    rhs,
    group_sizes,
    cost_estimate_num_rows=None,
    *,
    output_dtype=None,
    implementation: str | None = None,
):
    """``out[m] = lhs[m] @ rhs[g(m)]`` for live rows; see ``xla.ragged_dot``."""
    return RaggedDot.apply(
        lhs,
        rhs,
        group_sizes,
        cost_estimate_num_rows=cost_estimate_num_rows,
        output_dtype=output_dtype,
        implementation=implementation,
    )


def ragged_dot_swiglu(
    lhs,
    rhs_interleaved,
    group_sizes,
    out_dtype=None,
    cost_estimate_num_rows=None,
    *,
    implementation: str | None = None,
):
    """Project at output precision and apply SwiGLU; see ``xla.ragged_dot_swiglu``."""
    return RaggedDotSwiGLU.apply(
        lhs,
        rhs_interleaved,
        group_sizes,
        cost_estimate_num_rows=cost_estimate_num_rows,
        output_dtype=out_dtype,
        implementation=implementation,
    )


def ragged_dot_combine(
    intermediate,
    wo,
    group_sizes,
    top_k_weights,
    positions,
    token_group_sizes,
    activations_in_float32,
    cost_estimate_num_rows,
    *,
    implementation: str | None = None,
    eager_weight_gradient: bool = False,
):
    """Down-project expert rows and combine them into weighted token rows.

    With ``token_group_sizes``, only each batch element's live token prefix is
    defined. ``eager_weight_gradient`` orders dW before the input gradient; see
    :class:`RaggedDotCombine`.
    """
    return RaggedDotCombine.apply(
        intermediate,
        wo,
        group_sizes,
        top_k_weights,
        positions,
        token_group_sizes,
        activations_in_float32=activations_in_float32,
        cost_estimate_num_rows=cost_estimate_num_rows,
        implementation=implementation,
        eager_weight_gradient=eager_weight_gradient,
    )


def ragged_dot_dx(
    dout,
    rhs,
    group_sizes,
    *,
    cost_estimate_num_rows=None,
    implementation: str | None = None,
):
    """``dx[m] = dout[m] @ rhs[g(m)].T`` for live rows; see ``xla.ragged_dot_dx``."""
    return get_implementation(implementation).ragged_dot_dx(
        dout,
        materialize_weights(rhs),
        group_sizes,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )


def ragged_dot_dw(
    lhs,
    dout,
    group_sizes,
    *,
    output_manual_axis_type=None,
    accumulator=None,
    cost_estimate_num_rows=None,
    implementation: str | None = None,
):
    """Return dW or accumulator + dW; a caller-owned Ref is updated and returns None."""
    return get_implementation(implementation).ragged_dot_dw(
        lhs,
        dout,
        group_sizes,
        output_manual_axis_type=output_manual_axis_type,
        accumulator=accumulator,
        cost_estimate_num_rows=cost_estimate_num_rows,
    )
