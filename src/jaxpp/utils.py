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
import functools
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

import jax
from jax._src.sharding_impls import UnspecifiedValue

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OverwriteableVar(Generic[T]):
    MISSING = object()

    def __init__(self, default_value: T | None = None):
        self.default_value = default_value
        self._set_value = OverwriteableVar.MISSING

    def __bool__(self):
        raise ValueError("Variable used as truthy value")

    @contextlib.contextmanager
    def set(self, to: T):
        prev_value = self._set_value
        self._set_value = to
        try:
            yield
        finally:
            self._set_value = prev_value

    @property
    def value(self):
        if self._set_value is not OverwriteableVar.MISSING:
            return self._set_value
        assert self.default_value
        return self.default_value


class EnvVar(ABC, Generic[T], OverwriteableVar[T]):
    def __init__(self, env_key: str, default_value: T | None = None):
        super().__init__(default_value)
        self.default_value = default_value
        self.env_key = env_key

    @functools.cached_property
    def value(self) -> T:
        if self._set_value is not OverwriteableVar.MISSING:
            return self._set_value

        set_v = os.getenv(self.env_key)

        if set_v is None:
            assert self.default_value is not None
            return self.default_value

        parsed_v = self.parse(set_v)
        if parsed_v is None:
            raise ValueError(f"Unsupported value {self.env_key}={set_v}")

        return parsed_v

    @abstractmethod
    def parse(self, value: str) -> T | None: ...


def parse_bool(s):
    if s.lower() in ["true", "1", "t", "y", "yes"]:
        return True
    elif s.lower() in ["false", "0", "f", "n", "no"]:
        return False
    return None


class BoolEnvVar(EnvVar[bool]):
    def parse(self, value: str) -> bool | None:
        return parse_bool(value)


class StrEnvVar(EnvVar[str]):
    def parse(self, value: str) -> str | None:
        return value


class IntEnvVar(EnvVar[int]):
    def parse(self, value: str) -> int | None:
        try:
            return int(value)
        except ValueError:
            pass


def unzip_multi(xs, arity=2):
    if len(xs) == 0:
        return [[] for _ in range(arity)]
    assert all(len(xs_arr) == arity for xs_arr in xs)
    return jax._src.util.safe_map(list, jax._src.util.safe_zip(*xs))


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


def array_bytes(avals: Iterable[jax.Array]) -> int:
    return sum(aval.size * aval.dtype.itemsize for aval in avals)


def format_bytes(n_bytes: int) -> str:
    power_labels = {0: "B", 1: "KiB", 2: "MiB", 3: "GiB", 4: "TiB"}

    curr = float(n_bytes)
    n = 0
    while curr > 2**10:
        curr /= 2**10
        n += 1
    return f"{curr:.2f}{power_labels[n]}"


def hbytes(avals: Iterable[jax.Array]) -> str:
    return format_bytes(array_bytes(avals))


def get_named_sharding(a: jax.Array):
    assert isinstance(a.sharding, jax.sharding.NamedSharding)
    return a.sharding


def updated_named_sharding_mesh(
    shardings: Iterable[jax.sharding.NamedSharding | UnspecifiedValue | None], new_mesh
):
    res = []
    for s in shardings:
        if s is None or isinstance(s, UnspecifiedValue):
            res.append(s)
            continue

        assert isinstance(s, jax.sharding.NamedSharding)
        new_sharding = s
        if not isinstance(s.mesh, jax.sharding.AbstractMesh):
            new_sharding = jax.sharding.NamedSharding(new_mesh, s.spec)
        res.append(new_sharding)
    return res


@contextlib.contextmanager
def print_memstats(label: str, enabled: bool = False):
    if not enabled:
        yield
        return
    print(f"\nBefore: {label}:")
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"] / 2**30
        limit = stats["bytes_limit"] / 2**30
        peak_size = stats["peak_bytes_in_use"] / 2**30
        print(
            f"\tUsing (GB) {used:.2f} / {limit:.2f} ({used/limit:%}) ({peak_size=:.2f} GiB) on {d}",
            flush=True,
        )

    yield

    print(f"\nAfter: {label}:")
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"] / 2**30
        limit = stats["bytes_limit"] / 2**30
        peak_size = stats["peak_bytes_in_use"] / 2**30
        print(
            f"\tUsing (GB) {used:.2f} / {limit:.2f} ({used/limit:%}) ({peak_size=:.2f} GiB) on {d}",
            flush=True,
        )
