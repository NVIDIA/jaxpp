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

"""Host offloading of train-state leaves selected by ``host_offload_regex``."""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import AbstractMesh, AxisType

from maximoe import test_util, train, train_test_util

jax.config.parse_flags_with_absl()


# Trace under the trainer's config (see ``train.main``).
@jtu.with_config(jax_remove_size_one_mesh_axis_from_type=True)
class HostOffloadTest(test_util.MaximoeTestCase):
    @parameterized.parameters(
        None,
        r"^\.opt\.(mu|nu)\.",
        r"^\.(params|opt)\.",
        r"\.layers\[1\]\.ffn\.",
        r"\.mtp\.",
    )
    def test_offload_selection_preserves_shapes_dtypes_and_partitioning(self, pattern):
        config = dataclasses.replace(train.tiny_config(), mtp_num_layers=1)
        mesh = AbstractMesh(
            (1, 2, 1, 2),
            ("stage", "data", "fsdp", "expert"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        specs = train.train_state_specs(config)
        shardings = train.train_state_shardings(config)
        shapes = train.shape_tree(specs, shardings, mesh, host_offload_regex=pattern)
        for (path, shape), spec, partition in zip(
            jax.tree.leaves_with_path(shapes),
            jax.tree.leaves(specs),
            jax.tree.leaves(shardings),
            strict=True,
        ):
            selected = pattern is not None and re.search(
                pattern, jax.tree_util.keystr(path)
            )
            self.assertEqual(
                shape.sharding.memory_kind == "pinned_host", bool(selected)
            )
            self.assertEqual(shape.shape, spec.shape)
            self.assertEqual(shape.dtype, spec.dtype)
            self.assertEqual(shape.sharding.spec, partition)

    def test_invalid_offload_regex_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid host-offload-regex"):
            dataclasses.replace(train.tiny_config(), host_offload_regex="[")

    def test_offload_regex_matching_no_leaves_is_rejected(self):
        config = train.tiny_config()
        mesh = AbstractMesh((1,), ("data",), axis_types=(AxisType.Explicit,))
        specs = train.train_state_specs(config)
        shardings = jax.tree.map(lambda _: jax.P(), specs)
        with self.assertRaisesRegex(ValueError, "matched no train-state leaves"):
            train.shape_tree(specs, shardings, mesh, host_offload_regex=r"^\.typo\.")

    @parameterized.named_parameters(
        (
            "fp32_selected",
            jnp.float32,
            r"^\.params\.layers\[1\]\.|^\.opt\.(mu|nu)\.",
            0,
        ),
        (
            "bf16_selected",
            jnp.bfloat16,
            r"^\.params\.layers\[1\]\.|^\.opt\.(mu|nu)\.",
            0,
        ),
        ("all_state_mtp", jnp.float32, r"^\.", 1),
    )
    def test_offloaded_training_matches_device_state_across_steps(
        self, param_dtype, pattern, mtp_num_layers
    ):
        # Four GPUs cover the pipeline and the FSDP-sharded optimizer.
        self.skip_unless_gpus(4)
        # JaxPP's top-level jaxpr types offloaded inputs without their host
        # memory space, so the ``jax_enable_checks`` typecheck in its
        # ``eval_jaxpr`` rejects the pinned-host state.
        self.enterContext(jax.enable_checks(False))
        config = dataclasses.replace(
            train.tiny_config(),
            pp=2,
            fsdp=2,
            zero=1,
            num_microbatches=2,
            param_dtype=param_dtype,
            mtp_num_layers=mtp_num_layers,
            # Reach a visible update even with the tiny batch's scaled learning rate.
            warmup_steps=1,
            decay_steps=1,
            min_learning_rate=1e-3,
        )
        baseline, expected, baseline_batch = train_test_util.prepare_step(
            config, jax.devices()[:4]
        )
        offloaded, actual, offloaded_batch = train_test_util.prepare_step(
            dataclasses.replace(config, host_offload_regex=pattern), jax.devices()[:4]
        )
        initial_weights = np.asarray(
            train_test_util.local_arrays(expected.params.layers[1].ffn.wi)[0]
        ).copy()

        def check_state():
            for (path, got), want in zip(
                jax.tree.leaves_with_path(actual),
                jax.tree.leaves(expected),
                strict=True,
            ):
                selected = re.search(pattern, jax.tree_util.keystr(path)) is not None
                self.assertEqual(got.sharding.memory_kind == "pinned_host", selected)
                for got_local, want_local in zip(
                    train_test_util.local_arrays(got),
                    train_test_util.local_arrays(want),
                    strict=True,
                ):
                    self.assertEqual(
                        got_local.sharding.memory_kind == "pinned_host", selected
                    )
                    self.assertAllClose(
                        np.asarray(got_local, dtype=np.float32),
                        np.asarray(want_local, dtype=np.float32),
                        rtol=1e-5,
                        atol=1e-6,
                    )

        check_state()
        for _ in range(3):
            expected, expected_metrics = jax.block_until_ready(
                baseline(expected, baseline_batch)
            )
            actual, actual_metrics = jax.block_until_ready(
                offloaded(actual, offloaded_batch)
            )
            check_state()
            for got, want in zip(
                jax.tree.leaves(actual_metrics),
                jax.tree.leaves(expected_metrics),
                strict=True,
            ):
                for got_local, want_local in zip(
                    train_test_util.local_arrays(got),
                    train_test_util.local_arrays(want),
                    strict=True,
                ):
                    self.assertAllClose(
                        np.asarray(got_local),
                        np.asarray(want_local),
                        rtol=1e-5,
                        atol=1e-6,
                    )
        self.assertFalse(
            np.array_equal(
                initial_weights,
                np.asarray(
                    train_test_util.local_arrays(actual.params.layers[1].ffn.wi)[0]
                ),
            )
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
