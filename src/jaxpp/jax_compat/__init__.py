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
from jax._src import dtypes as _dtypes
from jax._src import op_shardings as _op_shardings

# get_aval was removed in JAX 0.10.0; jax.typeof is the equivalent.
# Use (0, 10) — pre-release versions like 0.10.1rc0 parse to (0, 10).
if jax.__version_info__ >= (0, 10):
    get_aval = jax.typeof
else:
    get_aval = core.get_aval

if jax.__version_info__ < (0, 8, 2):
    from jax._src import effects as _effects

    def eqn_effects(jaxpr, invars):
        del invars
        # Input effects are indexed to include jaxpr.constvars, but the eqn
        # should have effects indexed only on its explicit arguments.
        return {
            (
                e.replace(input_index=e.input_index - len(jaxpr.constvars))
                if isinstance(e, _effects.JaxprInputEffect)
                else e
            )
            for e in jaxpr.effects
        }

elif jax.__version_info__ < (0, 10, 2):

    def eqn_effects(jaxpr, invars):
        del invars
        return core.eqn_effects(jaxpr)

else:
    eqn_effects = core.eqn_effects


# ad_checkpoint
from jax._src.ad_checkpoint import remat_p

# api_util
from jax._src.api_util import _ensure_inbounds, _ensure_index_tuple

if jax.__version_info__ >= (0, 10):
    from jax._src.api_util import flatten_axis_resources
else:
    from jax._src.pjit import flatten_axis_resources

# debugging
from jax._src.debugging import debug_effect, inspect_sharding_p

# distributed
from jax._src.distributed import global_state

# dtypes
from jax._src.dtypes import finfo, supports_inf

if jax.__version_info__ >= (0, 10, 2):
    register_canonicalize_value_handler = _dtypes.register_canonicalize_value_handler
else:

    def register_canonicalize_value_handler(pytype, handler):
        _dtypes.canonicalize_value_handlers[pytype] = handler


# interpreters/ad
from jax._src.interpreters.ad import call_transpose, call_transpose_param_updaters

# interpreters/partial_eval
# has_effects lives in jax._src.interpreters.partial_eval across all supported
# versions, so import it from the private module directly.
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    close_jaxpr,
    convert_constvars_jaxpr,
    has_effects,
)

# lib
from jax._src.lib import _jax
from jax._src.pjit import _parse_jit_arguments as _jax_parse_jit_arguments

# shard_map
from jax._src.shard_map import shard_map_p

# sharding_impls
from jax._src.sharding_impls import UNSPECIFIED, UnspecifiedValue

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

# op_shardings
_are_hlo_shardings_equal = _op_shardings.are_hlo_shardings_equal

# pjit
from jax._src.pjit import jit_p

# _infer_params was briefly renamed to _trace_for_jit in JAX 0.9.0 and 0.9.0.1.
if (0, 9, 0) <= jax.__version_info__ < (0, 9, 1):
    from jax._src.pjit import _trace_for_jit as _infer_params
else:
    from jax._src.pjit import _infer_params

if jax.__version_info__ < (0, 10):
    from jax._src.custom_transpose import tree_broadcast
else:
    from jax.tree_util import tree_broadcast as _tree_broadcast_prefix
    from jax.tree_util import tree_unflatten as _tree_unflatten

    def tree_broadcast(full_treedef, tree, is_leaf=None):
        full_tree = _tree_unflatten(full_treedef, [0] * full_treedef.num_leaves)
        return _tree_broadcast_prefix(tree, full_tree, is_leaf=is_leaf)


def init_tracer(tracer, trace, aval):
    if jax.__version_info__ >= (0, 10):
        core.Tracer.__init__(tracer, trace, aval)
    else:
        core.Tracer.__init__(tracer, trace)
        tracer.aval = aval


def _parse_jit_arguments(*args, **kwargs):
    if jax.__version_info__ <= (0, 8, 1):
        kwargs.setdefault("abstracted_axes", None)
    else:
        kwargs.pop("abstracted_axes", None)
    return _jax_parse_jit_arguments(*args, **kwargs)


def bind_with_trace(primitive, trace, args, avals, params):
    if jax.__version_info__ >= (0, 9, 2):
        return primitive.bind_with_trace(trace, args, avals, params)
    return primitive.bind_with_trace(trace, args, params)


def aval_to_shape_dtype_struct(aval):
    """Convert an abstract value to a ``jax.ShapeDtypeStruct``."""
    # vma was renamed to manual_axis_type in JAX 0.10.0
    if jax.__version_info__ >= (0, 10):
        return jax.ShapeDtypeStruct(
            aval.shape,
            aval.dtype,
            sharding=aval.sharding,
            manual_axis_type=aval.manual_axis_type,
        )
    return jax.ShapeDtypeStruct(
        aval.shape, aval.dtype, sharding=aval.sharding, vma=aval.vma
    )


def shardings_are_equivalent(
    old: jax.sharding.Sharding,
    new: jax.sharding.Sharding,
    ndim: int,
    *,
    compare_memkind: bool,
) -> bool:
    hlo_equal = old == new or _are_hlo_shardings_equal(
        old._to_xla_hlo_sharding(ndim), new._to_xla_hlo_sharding(ndim)
    )
    if not hlo_equal:
        return False
    return not compare_memkind or old.memory_kind == new.memory_kind


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
    "register_canonicalize_value_handler",
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
    # pjit
    "jit_p",
    "_infer_params",
    "_parse_jit_arguments",
    # shard_map
    "shard_map_p",
    # sharding_impls
    "UNSPECIFIED",
    "UnspecifiedValue",
    # tree_util
    "equality_errors_pytreedef",
    "flatten_axis_resources",
    "_ensure_inbounds",
    "_ensure_index_tuple",
    # util
    "weakref_lru_cache",
    # interpreters/ad
    "call_transpose",
    "call_transpose_param_updaters",
    # core
    "bind_with_trace",
    "eqn_effects",
    # interpreters/partial_eval
    "DynamicJaxprTrace",
    "DynamicJaxprTracer",
    "close_jaxpr",
    "convert_constvars_jaxpr",
    "has_effects",
    # utilities
    "aval_to_shape_dtype_struct",
    "init_tracer",
    "map_dynamic_args",
    "shardings_are_equivalent",
]
