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

"""Shared scaffolding for the MaxiMoE tests."""

import jax
import numpy as np
from jax._src import jaxpr_util
from jax._src import test_util as jtu

from maximoe.kernels import gpu


def kernel_launches(jaxpr):
    """Every Pallas kernel launch reachable from ``jaxpr``, in program order."""
    return [
        eqn
        for _, eqn in jaxpr_util.all_eqns(jaxpr)
        if eqn.primitive.name in ("mpmd_map", "pallas_call")
    ]


def kernel_names(jaxpr):
    """Names of every Pallas kernel launch reachable from ``jaxpr``."""
    return [eqn.params["name"] for eqn in kernel_launches(jaxpr)]


@jtu.with_config(jax_numpy_dtype_promotion="standard", jax_numpy_rank_promotion="allow")
class MaximoeTestCase(jtu.JaxTestCase):
    """``jtu.JaxTestCase`` under the promotion rules MaxiMoE relies on.

    JAX's own tests default to strict dtype promotion and reject rank
    promotion. The kernel specifications multiply float32 products by bfloat16
    route weights and compare token ranges against per-batch prefix sizes, so
    the library needs NumPy semantics for both.
    """

    def skip_unless_gpus(
        self, count: int = 1, architecture: gpu.GpuArchitecture | str | None = None
    ) -> None:
        """Skip unless ``count`` CUDA GPUs of a supported architecture are visible.

        Supported architectures are those ``gpu.architecture`` recognizes, so a
        GPU outside the SM90 and SM100 families, such as SM120, never counts as
        either. ``architecture`` narrows the requirement to one family.
        """
        if architecture is None:
            family = "SM90 or SM100"
        else:
            architecture = gpu.GpuArchitecture(architecture)
            family = architecture.name
        if count == 1:
            requirement = f"requires an {family} GPU"
        else:
            requirement = f"requires {count} {family} GPUs"
        if not jtu.test_device_matches(["cuda"]) or jax.device_count() < count:
            self.skipTest(requirement)
        try:
            local = gpu.architecture()
        except NotImplementedError:
            self.skipTest(requirement)
        if architecture is not None and local is not architecture:
            self.skipTest(requirement)

    def assertClose(self, actual, expected, rows=None, *, rtol=2e-2, atol_scale=2e-2):
        """Compare ``rows`` in float32 with ``atol`` scaled to ``expected``.

        The defaults cover bfloat16 rounding of float32 accumulations. Poisoned
        rows leaking into a kernel show up as non-finite values here.
        """
        actual = np.asarray(actual, np.float32)
        expected = np.asarray(expected, np.float32)
        if rows is not None:
            actual, expected = actual[rows], expected[rows]
        self.assertTrue(np.isfinite(actual).all(), "kernel produced non-finite values")
        atol = atol_scale * max(float(np.max(np.abs(expected), initial=0.0)), 1e-3)
        self.assertAllClose(actual, expected, atol=atol, rtol=rtol)
