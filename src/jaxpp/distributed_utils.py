# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import os
import unittest

import jax

from jaxpp.dime2 import get_distributed_client


def distributed_main(fn):
    add_kwargs = {}
    init_parameters = inspect.signature(jax.distributed.initialize).parameters
    if "heartbeat_timeout_seconds" in init_parameters:
        add_kwargs["heartbeat_timeout_seconds"] = 5
    if "shutdown_timeout_seconds" in init_parameters:
        add_kwargs["shutdown_timeout_seconds"] = 5
    if "initialization_timeout" in init_parameters:
        add_kwargs["initialization_timeout"] = 5

    jax.distributed.initialize(
        coordinator_address=f"{os.environ['JAX_COORDINATOR_IP']}:{os.environ['JAX_COORDINATOR_PORT']}",
        num_processes=int(os.environ["NNODES"]),
        process_id=int(os.environ["NODE_RANK"]),
        local_device_ids=list(range(int(os.environ["N_GPUS"]))),
        **add_kwargs,
    )
    fn()
    jax.distributed.shutdown()


class JaxDistributedTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        if jax.process_count() == 1:
            self.skipTest("Test requires multiple processes.")
        get_distributed_client().wait_at_barrier(f"{self._testMethodName}_start", 1000)

    def tearDown(self):
        get_distributed_client().wait_at_barrier(f"{self._testMethodName}_end", 1000)
        super().tearDown()
