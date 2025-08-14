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

import copy
import itertools as it
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any

import jax
import jax._src.core as jcore
import numpy as np
from jax._src import profiler
from jax._src.interpreters.pxla import _create_da_object
from jax._src.pjit import _pjit_lower
from jax._src.sharding_impls import UNSPECIFIED, UnspecifiedValue
from jax.interpreters import mlir, pxla
from jaxlib.mlir.ir import Module as ir_Module

from jaxpp.types import UID, MaybeSharding, SerializeableSharding, fresh_scalar_uid


# TODO: move to https://github.com/google/jax/blob/b71829f/jax/experimental/serialize_executable.py#L24-L59
# eventually when serialization is properly supported in jax
class SerializeableMeshComputation:
    """
    A serializeable version of jax's `MeshComputation` object which contains
    a mlir `ir_Module` and additionally compiler arguments.
    """

    _hlo: str | bytes

    def __init__(
        self,
        mesh_computation: pxla.MeshComputation,
        name: str = "",
        compiler_options: dict[str, Any] | None = None,
        use_pgle=False,
    ):
        assert isinstance(mesh_computation, pxla.MeshComputation)
        self.name = name or mesh_computation._name
        self.compiler_options = compiler_options
        self._donated_invars = mesh_computation._donated_invars
        self.use_pgle = use_pgle

        hlo = mesh_computation._hlo
        if not isinstance(hlo, ir_Module):
            raise TypeError(f"Unknown hlo format {type(hlo)}")

        self._hlo = mlir.module_to_string(hlo)

        compile_args = copy.copy(mesh_computation.compile_args)

        self.in_shardings = jax.util.safe_map(
            SerializeableSharding, compile_args["in_shardings"]
        )
        self.out_shardings = jax.util.safe_map(
            SerializeableSharding, compile_args["out_shardings"]
        )
        self.device_assignment_ids = [d.id for d in compile_args["device_assignment"]]
        del compile_args["in_shardings"]
        del compile_args["out_shardings"]
        del compile_args["intermediate_shardings"]  # TODO: since 0.4.31 maybe keep ?
        del compile_args["context_mesh"]
        del compile_args["backend"]
        del compile_args["device_assignment"]
        del compile_args["all_args_info"]
        self.compile_args = compile_args

    def to_mesh_executable(self, mesh: jax.sharding.Mesh) -> pxla.MeshExecutable:
        hlo: ir_Module
        with mlir.make_ir_context():
            hlo = ir_Module.parse(self._hlo)

        in_named_shardings = tuple(s.to_named_sharding(mesh) for s in self.in_shardings)
        out_named_shardings = tuple(
            s.to_named_sharding(mesh) for s in self.out_shardings
        )

        compiler_options = self.compiler_options

        if self.use_pgle:
            # PGLE trace
            compiler_options = compiler_options if compiler_options else {}
            compiler_options["xla_gpu_enable_latency_hiding_scheduler"] = True
            mesh_computation = pxla.MeshComputation(
                self.name,
                hlo=hlo,
                compiler_options_kvs=(),
                donated_invars=self._donated_invars,
                platforms=(mesh._flat_devices_tuple[0].client.platform,),
                **self.compile_args,
                backend=mesh._flat_devices_tuple[0].client,
                in_shardings=in_named_shardings,
                out_shardings=out_named_shardings,
                device_assignment=_create_da_object(mesh._flat_devices_tuple),
            ).compile(compiler_options)

            @partial(jax.jit, out_shardings=in_named_shardings)
            def gen_dummy_inputs():
                return tuple(
                    jax.random.normal(
                        jax.random.PRNGKey(42), shape=in_aval.shape
                    ).astype(in_aval.dtype)
                    for in_aval in mesh_computation.in_avals
                )

            retries = 10
            pgle_profiler = profiler.PGLEProfiler(retries, percentile=90)
            for _ in range(retries):
                dummy_inputs = gen_dummy_inputs()
                with profiler.PGLEProfiler.trace(pgle_profiler):
                    dummy_outputs = mesh_computation.call(*dummy_inputs)
                for a in it.chain(dummy_outputs, dummy_inputs):
                    a.delete()

            fdo_profile = pgle_profiler.consume_fdo_profile()
            assert fdo_profile is not None
            compiler_options.update({"fdo_profile": fdo_profile})

        return pxla.MeshComputation(
            self.name,
            hlo=hlo,
            donated_invars=self._donated_invars,
            compiler_options_kvs=(),
            platforms=(mesh._flat_devices_tuple[0].client.platform,),
            **self.compile_args,
            backend=mesh._flat_devices_tuple[0].client,
            in_shardings=in_named_shardings,
            out_shardings=out_named_shardings,
            device_assignment=_create_da_object(mesh._flat_devices_tuple),
        ).compile(compiler_options)


def pjit_lower(
    jaxpr: jcore.ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts,  #: pxla.MaybeLayout,
    out_layouts,  #: pxla.MaybeLayout,
    donated_invars,
    ctx_mesh,
    name: str,
    keep_unused: bool,
    inline: bool,
    compiler_options_kvs: tuple[tuple[str, Any], ...],
    *,
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    pgle_profiler: profiler.PGLEProfiler | None,
):
    args = (
        jaxpr,
        in_shardings,
        out_shardings,
        in_layouts,
        out_layouts,
        donated_invars,
        ctx_mesh,
        name,
        keep_unused,
        inline,
        compiler_options_kvs,
    )
    kwargs = dict(
        lowering_platforms=lowering_platforms,
        lowering_parameters=lowering_parameters,
        pgle_profiler=pgle_profiler,
    )
    if jax.__version_info__ < (0, 5, 3):
        from jax._src.mesh import ResourceEnv

        args = ()
        kwargs = dict(
            jaxpr=jaxpr,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            in_layouts=in_layouts,
            out_layouts=out_layouts,
            donated_invars=donated_invars,
            resource_env=ResourceEnv(physical_mesh=ctx_mesh),
            name=name,
            keep_unused=keep_unused,
            inline=inline,
            compiler_options_kvs=compiler_options_kvs,
            lowering_platforms=lowering_platforms,
            lowering_parameters=lowering_parameters,
            pgle_profiler=pgle_profiler,
        )

    return _pjit_lower(*args, **kwargs)


class LoweringCache:
    def __init__(self):
        self._cache = dict[Any, tuple[UID, SerializeableMeshComputation]]()

    def pjit_to_serializeable_mesh_computation(
        self,
        closed_jaxpr: jcore.ClosedJaxpr,
        in_axis_resources: Sequence[MaybeSharding],
        out_axis_resources: Sequence[MaybeSharding] | UnspecifiedValue = UNSPECIFIED,
        in_layouts=None,
        out_layouts=None,
        name: str = "",
        mesh: jax.sharding.Mesh | None = None,
        donate_invars: Sequence[bool] | None = None,
        compiler_options: Sequence[tuple[str, Any]] | None = None,
        use_pgle=False,
    ):
        kwargs = OrderedDict(
            closed_jaxpr=closed_jaxpr,
            in_axis_resources=tuple(in_axis_resources),
            out_axis_resources=tuple(out_axis_resources),
            in_layouts=in_layouts,
            out_layouts=out_layouts,
            name=name,
            mesh=mesh,
            donate_invars=donate_invars,
            compiler_options=compiler_options,
            use_pgle=use_pgle,
        )
        key = tuple(kwargs.items())
        if (res := self._cache.get(key)) is None:
            res = (fresh_scalar_uid(), pjit_to_serializeable_mesh_computation(**kwargs))
            self._cache[key] = res
        return res


def pjit_to_serializeable_mesh_computation(
    closed_jaxpr: jcore.ClosedJaxpr,
    in_axis_resources: Sequence[MaybeSharding],
    out_axis_resources: Sequence[MaybeSharding] | UnspecifiedValue = UNSPECIFIED,
    in_layouts=None,
    out_layouts=None,
    name: str = "",
    mesh: jax.sharding.Mesh | None = None,
    donate_invars: Sequence[bool] | None = None,
    compiler_options: Sequence[tuple[str, Any]] | None = None,
    use_pgle=False,
) -> SerializeableMeshComputation:
    """
    Lowers the `closed_jaxpr` to a `SerializeableMeshComputation` object comprising
    of the mlir lowring (`ir_Module`) of the jaxpr and additional
    compiler arguments needed.
    This can be safely sent to a remote worker for compilation.
    """
    if mesh is None:
        # Set `mesh` to an empty mesh.
        mesh = jax.sharding.Mesh(np.empty((), dtype=object), ())

    if isinstance(out_axis_resources, UnspecifiedValue):
        out_axis_resources = (UNSPECIFIED,) * len(closed_jaxpr.out_avals)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Some donated buffers were not usable.*"
        )
        lowering = pjit_lower(
            closed_jaxpr,
            ctx_mesh=mesh,
            in_shardings=tuple(in_axis_resources),
            out_shardings=tuple(out_axis_resources),
            in_layouts=in_layouts or (None,) * len(closed_jaxpr.in_avals),
            out_layouts=out_layouts or (None,) * len(closed_jaxpr.out_avals),
            donated_invars=donate_invars or (False,) * len(closed_jaxpr.in_avals),
            name=name,
            keep_unused=True,
            inline=False,
            compiler_options_kvs=tuple(compiler_options.items())
            if compiler_options is not None
            else (),
            lowering_platforms=("cuda",),
            lowering_parameters=mlir.LoweringParameters(),
            pgle_profiler=None,  # FIXME
        )

    assert isinstance(lowering, pxla.MeshComputation)
    return SerializeableMeshComputation(lowering, name, compiler_options, use_pgle)
