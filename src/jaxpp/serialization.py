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

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

import jax
import tensorstore as ts
from jax._src import array, sharding, typing
from jax.experimental.array_serialization import serialization

logger = logging.getLogger(__name__)


async def async_serialize(
    arr_inp, tensorstore_spec, commit_future=None, context=serialization.TS_CONTEXT
):
    if not serialization._spec_has_metadata(tensorstore_spec):
        tensorstore_spec["metadata"] = serialization._get_metadata(arr_inp)

    # In the future, when we support multiple nodes per worker, we will have to ensure
    # that the tensorstore is created by only one of the nodes.
    # That is, only one of the nodes performs ts.open with create=True as below.
    # Then every node performs ts.open with assume_metadata=True to be able to write
    # their own shards.
    # Alternatively, each node can have its own tensorstore for local shards.
    t = await ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=context,
    )

    async def _write_array(shard):
        write_future = t[shard.index].write(shard.data)
        if commit_future is not None:
            assert isinstance(commit_future, list)
            commit_future.append(write_future.commit)
            await write_future.copy
        else:
            await write_future.commit

    local_shards = arr_inp.addressable_shards
    future_write_state = jax.tree_util.tree_map(_write_array, local_shards)
    return await asyncio.gather(*future_write_state)


class ArrayRefAsyncCheckpointManager(serialization.GlobalAsyncCheckpointManager):
    """Responsible for serializing arrays via TensorStore."""

    def _thread_func(self):
        try:
            current_process = jax.process_index()
            logger.info(
                "Starting commit to storage layer by process: %s", current_process
            )
            thread_start_time = time.time()
            if self._commit_futures:
                for future in self._commit_futures:
                    future.result()
            logger.info(
                "Finished committing to storage layer by process: %s", current_process
            )

            self._on_commit_callback()
            logger.info("on_commit_callback successfully ran!")

            jax.monitoring.record_event_duration_secs(
                "/jax/checkpoint/write/async/thread_duration_sec",
                time.time() - thread_start_time,
            )

        except Exception as e:
            logger.critical(f"{type(e)}: {e}")
            self._exception = e

    def wait_until_finished(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            logger.info("Thread joined successfully")

        self.check_for_errors()
        logger.info("Error check finished successfully")

    def serialize(self, arrays, tensorstore_specs, *, on_commit_callback):
        """Serializes Arrays or Arrays via TensorStore asynchronously.

        TensorStore writes to a storage layer in 2 steps:
        - Reading/copying from the source after which the source can be modified.
        => Returns a copy future.
        - Writing/committing to the storage layer.
        => Returns a commit future.

        In asynchronous mode, the serialization waits for the commit future to
        finish in a separate thread allowing other computation to proceed.

        Args:
        - arrays: Arrays or Arrays that should be serialized.
        - tensorstore_specs: TensorStore specs that are used to serialize GDAs or
        Arrays.
        - on_commit_callback: This callback will be executed after all processes
        have finished writing their checkpoints to disk. Filesystems where
        atomic rename operations are supported, you can rename from the
        temporary directory to the final directory. On GCS, you write to the
        final directory directly and in `on_commit_callback` you write a
        success file indicating that the serialization was successful because
        GCS does not support atomic rename operations.
        """
        logger.info("Waiting for previous serialization to finish.")
        self.wait_until_finished()

        commit_futures = [[] for _ in range(len(tensorstore_specs))]

        async def _run_serializer():
            future_writer = jax.tree_util.tree_map(
                async_serialize, arrays, tensorstore_specs, commit_futures
            )
            return await asyncio.gather(*future_writer)

        asyncio.run(_run_serializer())

        self._add_futures(jax.tree_util.tree_flatten(commit_futures)[0])

        # Used in wait_until_finished to check on process != 0, if the checkpoint
        # has finished writing.
        self._start_async_commit(on_commit_callback)

    def deserialize(
        self,
        shardings: Sequence[sharding.Sharding],
        tensorstore_specs: Sequence[dict[str, Any]],
        global_shapes: Sequence[array.Shape] | None = None,
        dtypes: Sequence[typing.DTypeLike] | None = None,
        concurrent_gb: int = 32,
    ):
        self.wait_until_finished()
        return serialization.run_deserialization(
            shardings, tensorstore_specs, global_shapes, dtypes, concurrent_gb
        )
