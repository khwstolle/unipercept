r"""
This file contains the base class for writers.
"""

import abc
import pathlib
import typing as T

import expath
from tensordict import TensorDict, TensorDictBase

__all__ = ["DataWriter"]


class DataWriter(metaclass=abc.ABCMeta):
    """
    Base class for writers that store data coming out of a model.
    """

    def __init__(
        self,
        *,
        path: expath.PathType,
        total_size: int,
        local_size: int,
        local_offset: int,
    ):
        """
        Parameters
        ----------
        path : str
            The path to the MemmapTensor directory.
        size : int
            The size of the first dimension of the results.
        """
        self._path = expath.locate(path)
        self._total_size = total_size
        self._local_size = local_size
        self._local_offset = local_offset

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def total_size(self) -> int:
        return self._total_size

    @property
    def local_size(self) -> int:
        return self._local_size

    @property
    def local_offset(self) -> int:
        return self._local_offset

    @property
    @abc.abstractmethod
    def is_closed(self) -> bool: ...

    @T.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self._path!r}, size={self.total_size})"

    @abc.abstractmethod
    def commit(self):
        """
        Write the results to disk. This method blocks until all results are written.

        Notes
        -----
        This method **does not** close the queue. To close the queue, use the
        :meth:`close` method.
        """
        ...

    @abc.abstractmethod
    def add(
        self, data: TensorDict, *, timings: T.MutableMapping[str, int] | None = None
    ):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """
        ...

    @abc.abstractmethod
    def close(self): ...

    @abc.abstractmethod
    def read(self) -> TensorDictBase: ...
