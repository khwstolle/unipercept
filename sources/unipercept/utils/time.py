r"""
Time utilities
==============

This module provides utilities for working with timestamps and profiling code.
"""

import contextlib
import enum as E
import time
import typing as T
from datetime import datetime
from typing import override

import pandas as pd
import torch
from tensordict import TensorDict

from unipercept.types import Device

__all__ = ["get_timestamp", "ProfileAccumulator", "profile"]

#######################
# T I M E S T A M P S #
#######################


class TimestampFormat(E.StrEnum):
    UNIX = E.auto()
    ISO = E.auto()
    LOCALE = E.auto()
    SHORT_YMD_HMS = E.auto()


def get_timestamp(*, format: str | TimestampFormat = TimestampFormat.ISO) -> str:
    """
    Returns a timestamp in the given format.
    """
    now = datetime.now()

    match format:
        case TimestampFormat.UNIX:
            return str(int(now.timestamp()))
        case TimestampFormat.ISO:
            return now.isoformat(timespec="seconds")
        case TimestampFormat.LOCALE:
            return now.strftime(r"%c")
        case TimestampFormat.SHORT_YMD_HMS:
            return now.strftime(r"%y%j%H%M%S")
        case _:
            msg = f"Invalid timestamp format: {format}"
            raise ValueError(msg)


######################
# P R O F I L I N G  #
######################

type TimingsMemory = T.MutableMapping[str, int]

PROFILE_KEY_SEPARATOR: T.Final[str] = "."


# ----------------------- #
# Profile context manager #
# ----------------------- #


class profile:
    """
    Context manager that profiles a block of code and stores the elapsed time.

    If the `dest` is `None` and the strict flag is set to `True`, an error is raised.
    Otherwise, the the null context manager is returned.
    """

    __slots__ = ("memory", "key", "_time", "_start")

    memory: TimingsMemory | None
    key: str
    _time: int
    _start: int | None

    @T.overload
    def __new__(
        cls,
        dest: None,
        key: str,
        /,
        *,
        strict: T.Literal[False],
    ) -> contextlib.AbstractContextManager[None]: ...

    @T.overload
    def __new__(
        cls,
        dest: None,
        key: str,
        /,
        *,
        strict: T.Literal[True],
    ) -> T.NoReturn: ...

    @T.overload
    def __new__(
        cls,
        dest: TimingsMemory,
        key: str,
        /,
        *,
        strict: bool = False,
    ) -> T.Self: ...

    def __new__(
        cls,
        dest: T.MutableMapping[str, int] | None,
        key: str,
        /,
        *,
        strict: bool = False,
    ) -> contextlib.AbstractContextManager[None] | T.Self:
        if dest is None:
            if strict:
                msg = f"No memory was passed and strict mode is enabled ({strict=})."
                raise RuntimeError(msg)
            return contextlib.nullcontext()

        # Construct the class
        self = super().__new__(cls)
        self.memory = dest
        self.key = key
        self._start = None

        return self

    @property
    def _running(self) -> bool:
        return self._start is not None

    @property
    def _delta(self) -> int:
        if self._start is None:
            return 0
        return time.perf_counter_ns() - self._start

    def __enter__(self) -> "ProfileHandle":
        if self.memory is not None:
            self._start = time.perf_counter_ns()
        return ProfileHandle(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.memory is None:
            return
        if not self._running:
            return
        self.memory[self.key] = self._delta
        self._start = None


# ---------------------- #
# Profile session handle #
# ---------------------- #


class ProfileHandle(T.MutableMapping[str, int]):
    """
    Redirects all __setitem__ and __getitem__ operations to the parent mapping, adding
    a prefix to the key.
    """

    __slots__ = ("_ref",)

    def __init__(self, ref: profile):
        self._ref = ref

    @property
    def _prefix(self):
        return self._ref.key + PROFILE_KEY_SEPARATOR

    @property
    def memory(self):
        mem = self._ref.memory
        if mem is None:
            msg = "Leaf is not attached to a profile context with memory."
            raise ValueError(msg)
        return mem

    def cancel(self):
        r"""
        Cancels the ongoing profiling context.
        """
        self._ref._start = None

    @override
    def __setitem__(self, key: str, value: int):
        self.memory[self._prefix + key] = value

    @override
    def __getitem__(self, key: str) -> int:
        return self.memory[self._prefix + key]

    @override
    def __delitem__(self, key: str):
        del self.memory[self._prefix + key]

    @override
    def __iter__(self) -> T.Iterator[str]:
        return iter(self.memory)

    @override
    def __len__(self) -> int:
        return len(self.memory)

    @override
    def keys(self) -> T.KeysView[str]:
        return self.memory.keys()

    @override
    def values(self) -> T.ValuesView[int]:
        return self.memory.values()

    @override
    def items(self) -> T.ItemsView[str, int]:
        return self.memory.items()


# ------------------------------- #
# Profiling utility for iterables #
# ------------------------------- #


@T.overload
def profile_iter[_R: T.Any](
    dest: TimingsMemory | None,
    key: str,
    src: T.Iterator[_R] | T.Iterable[_R],
    *,
    warmup: int,
    strict: T.Literal[False],
) -> T.Iterable[_R]: ...


@T.overload
def profile_iter[_R: T.Any](
    dest: None,
    key: str,
    src: T.Iterator[_R] | T.Iterable[_R],
    *,
    warmup: int,
    strict: T.Literal[True],
) -> T.NoReturn: ...


@T.overload
def profile_iter[_R: T.Any](
    dest: TimingsMemory,
    key: str,
    src: T.Iterator[_R] | T.Iterable[_R],
    *,
    warmup: int,
    strict: bool = False,
) -> T.Iterable[_R]: ...


def profile_iter[_R: T.Any](
    dest: TimingsMemory | None,
    key: str,
    src: T.Iterator[_R] | T.Iterable[_R],
    *,
    warmup: int = 0,
    strict: bool = False,
) -> T.Iterable[_R]:
    r"""
    Profile the time it takes to get each item from the iterable, and yield the profile
    leaf and the item.

    Parameters
    ----------
    timings : TimingsMemory | None
        The memory to store the timings in.
    source : Iterator[_R] | Iterble[_R]
        The source to iterate over. If it is not already an iterator, we turn it into
        one using `iter(source)`.
    key : str
        The key to store the timings under in the target memory.
    warmup : int
        The number of iterations to skip before recording the timings.
    """
    assert warmup >= 0, f"Warmup ({warmup=}) must be a non-negative integer."
    # Check whether the timings memory is present, if not, then we redirect immediately
    # to a generator that returns pairs of `None` and each item from the source.
    if dest is None:
        if strict:
            msg = f"No memory was passed and strict mode is enabled ({strict=})."
            raise RuntimeError(msg)
        yield from src

    # Check whether the source is an iterator or an iterable, in the latter case
    # we need to turn it into an iterator first
    if not isinstance(src, T.Iterator):
        src = iter(src)

    assert warmup >= 0, "Warmup must be a non-negative integer."
    # Infinite loop such that we can have more fine-grained control over
    # the logic involved in stoping the iteration and to profile exactly the
    # time involved in getting an item from the source.
    while True:
        with profile(dest, key, strict=True) as ph:
            # Fetch an item from the source iterator
            try:
                item = next(src)
            except StopIteration:
                # Cancel the profiling context if the source is depleted
                ph.cancel()
                break

            # Handle warmup, i.e. skip the first `warmup` iterations
            if warmup >= 0:
                ph.cancel()
                warmup -= 1
        yield item


# ---------------- #
# Profiling memory #
# ---------------- #


class ProfileRecord(T.NamedTuple):
    r"""
    A single long-format record used to store profiling data.
    """

    key: str
    time: int | float


class ProfileAccumulator(T.MutableMapping[str, int]):
    """
    Mapping that accumulates values can be used in combination with the `profile`
    context manager.
    """

    __slots__ = ("records",)

    _KEY_TIME: T.Final[str] = "seconds"
    _KEY_NAME: T.Final[str] = "name"

    records: list[ProfileRecord]

    def __init__(self):
        self.records = []

    def run(
        self, warmup: int = 0, incomplete_ok: bool = True, enabled: bool = True
    ) -> T.Iterator[dict[str, int] | None]:
        """
        Returns an infinite iterator of dicts to which profiling data can be written.
        The first dict that is successfully returned is defines the keys that will be
        read.
        """
        if not enabled:
            while True:
                yield None
        skip = warmup
        while True:
            target = {}
            yield target
            if len(target.keys()) == 0:
                continue

            new_keys = frozenset(target.keys())
            cur_keys = frozenset(self.keys())
            if len(self.records) > 0 and new_keys != cur_keys and not incomplete_ok:
                msg = f"Received {new_keys - cur_keys} keys that were not present in the previous iteration."
                raise ValueError(msg)
            if skip > 0:
                skip -= 1
            else:
                for k, v in target.items():
                    self[k] = v
            del target

    @override
    def __getitem__(self, key):
        return self.to_means()[key]

    @override
    def __setitem__(self, key, value: int):
        assert isinstance(value, int), "Value must be time in nanoseconds"
        rec = ProfileRecord(key, value)
        self.records.append(rec)

    @override
    def __delitem__(self, key):
        idxs_to_remove = [i for i, r in enumerate(self.records) if r.key == key]
        for i in reversed(idxs_to_remove):
            del self.records[i]

    @override
    def __iter__(self):
        return iter(self.to_means)

    @override
    def __len__(self):
        return len(self.records)

    def reset(self):
        self.records.clear()

    def to_records(self, *, unit: bool = True) -> T.Iterable[ProfileRecord]:
        return (
            ProfileRecord(r.key, r.time * 1e-9 if unit else r.time)
            for r in self.records
        )

    def to_dataframe(self, *, unit: bool = True) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            self.to_records(unit=unit),
            columns=[self._KEY_NAME, self._KEY_TIME],
        )

    def to_tensordict(
        self, *, device: Device | None = None, unit: bool = True
    ) -> TensorDict:
        return TensorDict(
            {
                key: torch.tensor(
                    times,
                    device=device,
                    dtype=torch.double if unit else torch.long,
                    requires_grad=False,
                )
                for key, times in self.to_groups(unit=unit).items()
            },
            device=device,
            batch_size=[],
        )

    def to_summary(self, *, unit: bool = True) -> pd.DataFrame:
        """
        Return summarizing statistics for every key in the accumulator.
        """
        df = self.to_dataframe(unit=unit)

        return df.groupby(self._KEY_NAME)[self._KEY_TIME].agg(
            ["count", "sum", "mean", "std", "min", "max", "median"]
        )

    def to_means(self):
        return {k: sum(v) / len(v) for k, v in self.to_groups().items()}

    def to_groups(self, *, unit: bool = True) -> dict[str, list[int | float]]:
        """
        Group the records by their key.
        """
        groups = {}
        for r in self.records:
            t = r.time * 1e-9 if unit else r.time
            groups.setdefault(r.key, []).append(t)
        return groups

    @override
    def items(self):
        return self.to_means().items()

    @override
    def keys(self):
        return frozenset(r.key for r in self.records)

    @override
    def values(self):
        return self.to_means().values()

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(self.keys())})"
