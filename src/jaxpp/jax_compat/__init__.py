"""JAX compatibility wrappers for internal APIs.

This module provides stable access to JAX internal APIs that may change
between versions, with version-conditional imports where needed.
"""

import jax

# Usage:
#   from jaxpp import jax_compat as jc           # for jc.remat_p, jc.cache, etc.
#   from jaxpp.jax_compat import core as jcore   # for jcore.Var, jcore.Jaxpr, etc.
#
# core - re-export as module for namespace alias usage
from jax._src import core  # noqa: F401

# ad_checkpoint
from jax._src.ad_checkpoint import remat_p

# api_util
from jax._src.api_util import _ensure_inbounds

# custom_transpose
from jax._src.custom_transpose import tree_broadcast

# debugging
from jax._src.debugging import debug_effect, inspect_sharding_p

# distributed
from jax._src.distributed import global_state

# dtypes
from jax._src.dtypes import finfo, supports_inf

# interpreters/ad
from jax._src.interpreters.ad import (
    call_transpose,
    call_transpose_param_updaters,
)

# interpreters/partial_eval
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    close_jaxpr,
    convert_constvars_jaxpr,
)

# lib
from jax._src.lib import _jax
from jax._src.pjit import _parse_jit_arguments

# shard_map
from jax._src.shard_map import shard_map_p

# sharding_impls
from jax._src.sharding_impls import UnspecifiedValue

# tree_util
from jax._src.tree_util import equality_errors_pytreedef

# jutil (from jax._src.util)
# util
from jax._src.util import (
    OrderedSet,
    cache,
    partition_list,
    safe_map,
    safe_zip,
    unzip2,
    weakref_lru_cache,
)

# op_shardings - version-conditional
if jax.__version_info__ < (0, 7, 2):
    from jax._src.op_shardings import are_op_shardings_equal as are_hlo_shardings_equal
else:
    from jax._src.op_shardings import are_hlo_shardings_equal

# pjit - version-conditional
# jit_p was renamed from pjit_p in JAX 0.7.0
if jax.__version_info__ < (0, 7, 0):
    from jax._src.pjit import pjit_p as jit_p
else:
    from jax._src.pjit import jit_p

# _infer_params was renamed to _trace_for_jit in JAX 0.8.3
if jax.__version_info__ < (0, 8, 3) or jax.__version_info__ > (0, 9, 0):
    from jax._src.pjit import _infer_params
else:
    from jax._src.pjit import _trace_for_jit as _infer_params


def set_mesh(mesh: jax.sharding.Mesh):
    """Return a context manager that sets the mesh.

    JAX >= 0.8 requires ``jax.set_mesh`` for the mesh to be visible to
    ``jax.jit``; older versions only support ``with mesh:``.
    """
    if jax.__version_info__ >= (0, 8):
        return jax.set_mesh(mesh)
    return mesh


def map_dynamic_args(args, kwargs, static_argnums, static_argnames, fn):
    static_argnums = static_argnums or ()
    static_argnames = static_argnames or ()

    if static_argnums:
        num_args = len(args)
        static_argnums = _ensure_inbounds(True, num_args, static_argnums)
    dyn_argnums = tuple(i for i in range(len(args)) if i not in static_argnums)
    dyn_args = tuple(args[i] for i in dyn_argnums)
    dyn_kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}

    flat_dyn, in_tree = jax.tree.flatten((dyn_args, dyn_kwargs))
    flat_transformed = [fn(x) for x in flat_dyn]
    transformed_dyn_args, transformed_dyn_kwargs = jax.tree.unflatten(
        in_tree, flat_transformed
    )

    dyn_iter = iter(transformed_dyn_args)
    new_args = tuple(
        next(dyn_iter) if i in dyn_argnums else args[i] for i in range(len(args))
    )
    new_kwargs = {
        k: (transformed_dyn_kwargs[k] if k not in static_argnames else v)
        for k, v in kwargs.items()
    }

    return new_args, new_kwargs


__all__ = [
    # ad_checkpoint
    "remat_p",
    # custom_transpose
    "tree_broadcast",
    # debugging
    "debug_effect",
    "inspect_sharding_p",
    # distributed
    "global_state",
    # dtypes
    "finfo",
    "supports_inf",
    # jutil
    "OrderedSet",
    "cache",
    "partition_list",
    "safe_map",
    "safe_zip",
    "unzip2",
    # lib
    "_jax",
    # op_shardings
    "are_hlo_shardings_equal",
    # pjit
    "jit_p",
    "_infer_params",
    "_parse_jit_arguments",
    # shard_map
    "shard_map_p",
    # sharding_impls
    "UnspecifiedValue",
    # tree_util
    "equality_errors_pytreedef",
    # util
    "weakref_lru_cache",
    # interpreters/ad
    "call_transpose",
    "call_transpose_param_updaters",
    # interpreters/partial_eval
    "DynamicJaxprTrace",
    "DynamicJaxprTracer",
    "close_jaxpr",
    "convert_constvars_jaxpr",
    # utilities
    "map_dynamic_args",
    "set_mesh",
]
