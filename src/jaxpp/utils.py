# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

import jax

logger = logging.getLogger(__name__)


def unzip_multi(xs, arity=2):
    if len(xs) == 0:
        return [[] for _ in range(arity)]
    assert all(len(xs_arr) == arity for xs_arr in xs)
    return jax.util.safe_map(list, jax.util.safe_zip(*xs))


_T = TypeVar("_T")
_Key = TypeVar("_Key")


def groupby(elements: Iterable[tuple[_Key, _T]]) -> dict[_Key, list[_T]]:
    # Result is OrderedDict as keys are seen in the iterable
    groups = OrderedDict()
    for key, elem in elements:
        group = groups.get(key, None)
        if group is None:
            group = []
            groups[key] = group
        group.append(elem)
    return groups


def partition(
    predicate: Callable[[_T], bool], elements: Iterable[_T]
) -> tuple[list[_T], list[_T]]:
    groups = groupby((predicate(e), e) for e in elements)
    return groups.get(True, []), groups.get(False, [])


class _Sentinel:
    pass


SENTINEL = _Sentinel()


class RichDict(dict[_Key, _T], Generic[_Key, _T]):
    def get_or_else(self, k: _Key, f: Callable[[], _T]) -> _T:
        maybe_res = self.get(k, SENTINEL)
        if isinstance(maybe_res, _Sentinel):
            return f()
        return maybe_res

    def get_or_else_update(self, k: _Key, f: Callable[[], _T]) -> _T:
        maybe_res = self.get(k, SENTINEL)
        if isinstance(maybe_res, _Sentinel):
            res = f()
            self[k] = res
            return res
        return maybe_res

    def set_or_raise_if_present(self, k: _Key, v: _T) -> None:
        if k in self:
            raise KeyError(f"Key `{k}` already present with value: `{self[k]}`")
        self[k] = v


@contextlib.contextmanager
def log_elapsed_time(event: str, msg: str | None = None, unit="s"):
    valid_units = ["s", "ms", "us", "ns"]

    if unit not in valid_units:
        raise ValueError(f"Unknown Unit: `{unit}`. Accepted: {valid_units}.")

    start_time = time.perf_counter_ns()
    logger.info(f"[start] {event}")
    yield
    elapsed_time = time.perf_counter_ns() - start_time

    match unit:
        case "ns":
            elapsed_time = float(elapsed_time)
        case "us":
            elapsed_time = elapsed_time / 1e3
        case "ms":
            elapsed_time = elapsed_time / 1e6
        case "s":
            elapsed_time = elapsed_time / 1e9

    if msg is not None:
        logger.info(f"[  end] {event} took {elapsed_time:.5}{unit}: {msg}")
    else:
        logger.info(f"[  end] {event} took {elapsed_time:.5}{unit}")
