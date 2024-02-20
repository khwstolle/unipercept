r"""
Implements a DataWriter that uses a TensorDict to store the data in memory.
"""

import concurrent.futures
import gc
import json
import typing as T

import torch
import torch.utils.data
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase

from unipercept.state import check_main_process
from unipercept.utils.time import profile

from ._base import DataWriter


class MemmapWriter(DataWriter):
    """
    Stores each data item in a list of TensorDicts, then read them by creating a
    LazyStackedTensorDict from that list.
    """

    def __init__(
        self,
        *,
        max_threads: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._writes: set[concurrent.futures.Future[T.Any]] = set()
        self._cursor: int = self.local_offset
        self._closed = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_threads, thread_name_prefix=self.__class__.__name__
        )

        # If we are the main process, then also write the `meta.json` file for the
        # LazyStackedTensorDict
        if check_main_process():
            meta_path = self._path / "meta.json"
            if not meta_path.exists():
                meta = {"_type": str(LazyStackedTensorDict), "stack_dim": 0}
                meta_path.write_text(json.dumps(meta))

    @property
    @T.override
    def is_closed(self) -> bool:
        return self._closed

    @T.override
    def add(
        self, data: TensorDict, *, timings: T.MutableMapping[str, int] | None = None
    ):
        if self.is_closed:
            msg = "The writer has been closed, so no more data can be added."
            raise RuntimeError(msg)
        if data.batch_dims >= 1:
            for item in data.unbind(0):
                self.add(item, timings=timings)
            return
        with profile(timings, "copy"):
            data = data.to("cpu")  # , non_blocking=True)
        with profile(timings, "memmap"):
            fut = self._executor.submit(
                data.memmap_,
                str(self._path / str(self._cursor)),
                num_threads=0,
                return_early=False,
            )
            fut.add_done_callback(lambda fut: self._writes.remove(fut))

            self._writes.add(fut)
            self._cursor += 1

    @T.override
    def close(self):
        if self.is_closed:
            return  # already closed
        self._closed = True

    @T.override
    def commit(self):
        self.close()

        _, self._writes = concurrent.futures.wait(
            self._writes, timeout=None, return_when=concurrent.futures.ALL_COMPLETED
        )

        # Sanity check that no futures are still pending
        if len(self._writes) > 0:
            msg = f"{len(self._writes)} futures were not completed."
            raise RuntimeError(msg)

        torch.cuda.empty_cache()
        gc.collect()

    @T.override
    def read(self) -> TensorDictBase:
        if self._closed is False:
            msg = "The writer has not been closed, so the data is not ready to be read."
            raise RuntimeError(msg)
        reader = LazyStackedTensorDict.load_memmap(self._path)
        return reader
