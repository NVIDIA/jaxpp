# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from functools import partial

import jax
import jax._src.util as ju
import numpy as np
from jax._src import dtypes

from jaxpp.types import ArrayTree

MUBATCH_AXIS = 1


@dataclass(frozen=True)
class LoopOutput:
    """Specifies how to accumulate outputs of a loop body.

    This class is typically used as the return type for the loop body function
    passed to utilities like `jaxpp.accumulate_grads`. It allows specifying
    different accumulation strategies for various outputs alongside the primary
    output (e.g., gradients).

    Attributes:
        sum: A pytree of arrays. Corresponding elements from each loop iteration
            will be summed together. Finalized using `jnp.sum(axis=0)`.
        cat: A pytree of arrays. Corresponding elements from each loop iteration
            will be concatenated along a new leading axis (the microbatch axis).
            Finalized state retains the concatenated arrays.
        max: A pytree of arrays. The element-wise maximum across all loop iterations
            will be kept. Finalized using `jnp.max(axis=0)`.
        last: A pytree of arrays. The elements from the final loop iteration
            will be kept. Finalized using `jnp.take(indices=-1, axis=0)`.

    Any of the attributes can be None if that accumulation strategy is not needed.
    """

    sum: ArrayTree | None
    cat: ArrayTree | None
    max: ArrayTree | None = None
    last: ArrayTree | None = None

    def __iter__(self):
        return iter((self.sum, self.cat, self.max, self.last))

    def state(self, n_iters: int) -> "LoopOutput":
        def maybe_tree_state(state_init, loop_output):
            if loop_output is not None:
                return jax.tree_util.tree_map(state_init, loop_output)
            return None

        def max_state(array):
            small = (
                -np.inf
                if dtypes.supports_inf(array.dtype)
                else dtypes.finfo(array.dtype).min
            )
            return jax.lax.full_like(array, small)

        leaves = jax.tree_util.tree_leaves(
            LoopOutput(
                sum=maybe_tree_state(jax.lax.zeros_like_array, self.sum),
                cat=maybe_tree_state(
                    lambda a: jax.numpy.zeros(
                        shape=ju.tuple_insert(a.shape, MUBATCH_AXIS, n_iters),
                        dtype=a.dtype,
                    ),
                    self.cat,
                ),
                max=maybe_tree_state(max_state, self.max),
                last=maybe_tree_state(jax.lax.zeros_like_array, self.last),
            )
        )
        return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(self), leaves)

    def update(self, iteration: int, update: "LoopOutput"):
        def maybe_tree_update(operator, state, update):
            if state is not None:
                return jax.tree_util.tree_map(operator, state, update)
            return None

        cat = maybe_tree_update(
            partial(
                jax.lax.dynamic_update_index_in_dim, index=iteration, axis=MUBATCH_AXIS
            ),
            self.cat,
            update.cat,
        )

        leaves = jax.tree_util.tree_leaves(
            LoopOutput(
                sum=maybe_tree_update(jax.lax.add, self.sum, update.sum),
                cat=cat,
                max=maybe_tree_update(jax.lax.max, self.max, update.max),
                last=maybe_tree_update(
                    (lambda state, update: jax.lax.select(True, update, state)),
                    self.last,
                    update.last,
                ),
            )
        )
        # NOTE: by unflattening as the `self` structure we make
        #  sure to passthrough auxiliary data that users
        #  might have added through inheritance
        return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(self), leaves)

    def finalize(self):
        def maybe_tree_update(operator, state):
            if state is not None:
                return jax.tree_util.tree_map(operator, state)
            return None

        leaves = jax.tree_util.tree_leaves(
            LoopOutput(
                sum=maybe_tree_update(
                    # lambda a: jax.lax.reduce_sum_p.bind(a, axes=(0,)),
                    # NOTE: the `jnp.sum` below will cast to fp32
                    #  corresponding to megatron's `accumulate_allreduce_grads_in_fp32=True`
                    #  For the `accumulate_allreduce_grads_in_fp32=False`
                    #  version use the line above
                    partial(jax.numpy.sum, axis=0),
                    self.sum,
                ),
                cat=self.cat,
                max=maybe_tree_update(partial(jax.numpy.max, axis=0), self.max),
                last=maybe_tree_update(
                    partial(jax.numpy.take, indices=-1, axis=0), self.last
                ),
            )
        )

        # NOTE: by unflattening as the `self` structure we make
        #  sure to passthrough auxiliary data that users
        #  might have added through inheritance
        return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(self), leaves)


def LoopOutput_tree_flatten(loop_output: LoopOutput):
    return ((loop_output.sum, loop_output.cat, loop_output.max, loop_output.last), None)


def LoopOutput_tree_unflatten(aux, children):
    return LoopOutput(*children)


jax.tree_util.register_pytree_node(
    LoopOutput, LoopOutput_tree_flatten, LoopOutput_tree_unflatten
)
