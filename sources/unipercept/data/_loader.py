"""Defines functions for creating dataloaders for training and validation, using the common dataset format."""

import abc
import dataclasses as D
import enum
import functools
import itertools
import math
import operator
import random
import typing as T
import warnings
from typing import override

import laco
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    Sampler,
    get_worker_info,
)

from unipercept.data.sets import PerceptionDataqueue, PerceptionDataset, QueueGenerator
from unipercept.data.transforms import apply_dataset
from unipercept.log import create_table, get_logger
from unipercept.model import InputData
from unipercept.state import cpus_available, get_process_count, get_process_index

__all__ = [
    "DataLoaderConfig",
    "DataLoaderFactory",
    "DatasetInterface",
    "TrainingSampler",
    "InferenceSampler",
    "SamplerFactory",
]

_logger = get_logger(__name__)


_P = T.ParamSpec("_P")
_I = T.TypeVar("_I", covariant=True)


def _distribute_batch_size(total: int) -> int:
    """Given a total batch size, distribute it evenly across all GPUs."""
    world_size = get_process_count()
    if total == 0 or total % world_size != 0:
        msg = f"Batch size ({total=}) must be divisible by {world_size=}."
        raise ValueError(msg)
    per_device = total // world_size

    return per_device


@D.dataclass(slots=True)
class ProcessInfo:
    """
    Tuple representing the total number of distributed processes and the index of the active process.
    """

    count: int
    index: int

    def __post_init__(self):
        self.count = max(self.count, 1)


def to_tensordict(data: InputData):
    if isinstance(data, InputData):
        data = data.to_tensordict(retain_none=False)
    return data


class BaseSampler(Sampler[_I], T.Generic[_I], metaclass=abc.ABCMeta):
    @staticmethod
    def get_dist_info(dist_num: int | None, dist_idx: int | None) -> ProcessInfo:
        """
        Returns the number of distributed processes (e.g. GPUs) and the index of the current process.
        If no value is provided, it is determined from the global state.

        In case parameters are provided, they must both be an integer.

        Parameters
        ----------
        dist_num
            The number of distributed processes.
        dist_idx
            The index of the current process.

        Returns
        -------
        dist_num
            The number of distributed processes.
        dist_idx
            The index of the current process.

        Raises
        ------
        ValueError
            If either ``dist_num`` or ``dist_idx`` is not an integer.
        """

        if not dist.is_available():
            msg = "Distributed data sampler requires torch.distributed to be available."
            raise RuntimeError(msg)

        if isinstance(dist_num, int) and isinstance(dist_idx, int):
            return ProcessInfo(count=dist_num, index=dist_idx)

        if dist_num is None and dist_idx is None:
            return ProcessInfo(
                count=get_process_count() or 1, index=get_process_index() or 0
            )

        msg = f"Both `{dist_num=}` and `{dist_idx=}` must be integers."
        raise ValueError(msg)

    _process_index: T.Final[int]
    _process_count: T.Final[int]

    def __init__(
        self,
        queue: PerceptionDataqueue,
        *,
        process_index: int | None = None,
        process_count: int | None = None,
        epoch=0,
    ):
        assert epoch >= 0, f"Epoch must be non-negative, but got {epoch=}."

        info = self.get_dist_info(process_index, process_count)

        self._process_count, self._process_index = info.count, info.index
        self._queue_size = len(queue)
        self._epoch = epoch

        _logger.debug(
            f"Initialized sampler {self._process_index + 1} of {self._process_count}"
        )

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        # _logger.debug(f"Sampler epoch set to {value}")
        self._epoch = value

    @property
    def process_index(self) -> int:
        return self._process_index

    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def queue_size(self) -> int:
        return self._queue_size

    @property
    @abc.abstractmethod
    def indices(self) -> T.Iterator[_I]: ...

    @property
    @abc.abstractmethod
    def sample_count(self) -> int: ...

    @property
    @abc.abstractmethod
    def total_count(self) -> int: ...

    @property
    def generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self._epoch)

    @override
    def __iter__(self):
        yield from self.indices

    def __len__(self):
        raise NotImplementedError


class TrainingSampler(BaseSampler[int]):
    def __init__(
        self,
        *args,
        shuffle=True,
        repeat_factor: float = 2,
        selected_round=0,
        selected_ratio=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._epoch = 0
        self._shuffle = shuffle
        self._repeat_factor = repeat_factor

        if not selected_ratio:
            selected_ratio = self._process_count
        if selected_round:
            assert selected_round > self.queue_size, (
                f"{self.queue_size=} <= {selected_round=}."
            )
            self._selected_count = int(
                math.floor(
                    self.queue_size // selected_round * selected_round / selected_ratio
                )
            )
        else:
            self._selected_count = int(math.ceil(self.queue_size / selected_ratio))

    @functools.cached_property
    @override
    def sample_count(self):
        return int(
            math.ceil(self.queue_size * self._repeat_factor / self.process_count)
        )

    @functools.cached_property
    @override
    def total_count(self):
        return self.sample_count * self.process_count

    @property
    @override
    def indices(self):
        # Shuffle if needed
        if self._shuffle:
            idc = torch.randperm(self.queue_size, generator=self.generator)
        else:
            idc = torch.arange(start=0, end=self.queue_size)

        # Produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        rep = self._repeat_factor
        if isinstance(rep, float) and not rep.is_integer():
            rep_size = math.ceil(rep * self.queue_size)
            idc = idc[torch.tensor([int(i // rep) for i in range(rep_size)])]
        else:
            idc = torch.repeat_interleave(idc, repeats=int(rep), dim=0)

        idc = idc.tolist()
        # Add extra samples to make it evenly divisible
        pad_size = self.total_count - len(idc)
        if pad_size > 0:
            idc += idc[:pad_size]
        assert len(idc) == self.total_count

        # Subsample per process
        idc = idc[self.process_index : self.total_count : self.process_count]
        assert len(idc) == self.sample_count

        # Generate samples from the subsampled indices
        yield from iter(idc[: self._selected_count])

        self.epoch += 1

    @override
    def __len__(self):
        return min(self.sample_count, self._selected_count)


class InferenceSampler(BaseSampler[str]):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    @staticmethod
    def create_indices(size: int, p_num: int, p_idx: int):
        shard_len = size // p_num
        shard_rem = size % p_num
        shard_sizes = [shard_len + int(r < shard_rem) for r in range(p_num)]

        i_start = sum(shard_sizes[:p_idx])
        i_end = min(sum(shard_sizes[: p_idx + 1]), size)

        return list(range(i_start, i_end))

    def __init__(self, queue: PerceptionDataqueue, *args, **kwargs):
        if "epoch" in kwargs:
            warnings.warn("Epoch argument is ignored in InferenceSampler.", UserWarning)
            del kwargs["epoch"]
        super().__init__(queue, *args, **kwargs)

        # We need to make sure that all samples that belong to the same sequence
        # are proceesed  by the same distributed process. Therefore, we need to first
        # first create groups of samples.

        # Group by sequence
        sequence_keys = {}
        for key, item in iter(queue):
            assert isinstance(key, str), f"Expected key to be a string, but got {key=}."
            sequence_keys.setdefault(item["sequence"], []).append(
                (key, float(item["frame"]))
            )

        # Sort by frame
        for k in sequence_keys:
            sequence_keys[k] = sorted(sequence_keys[k], key=lambda x: x[1])

        # Create indices for each process, where each index points to a sequence id
        keys_list = [
            list(map(operator.itemgetter(0), ks)) for ks in sequence_keys.values()
        ]
        key_indices = set(
            self.create_indices(len(keys_list), self.process_count, self.process_index)
        )

        # Map each tuple (key, frame_num) to (key), then store the flattened list of keys
        self._indices = list(itertools.chain(*[keys_list[k] for k in key_indices]))
        # print(f"Indices (keys) for process {self.process_index}: {list(self._indices)}")
        if not all(isinstance(i, str) for i in self._indices):
            msg = f"Expected all indices to be strings! Got: {self._indices}"
            raise RuntimeError(msg)

    @property
    @override
    def epoch(self):
        raise ValueError("Epoch is not defined for InferenceSampler.")

    @property
    @override
    def indices(self) -> T.Iterable[str]:
        yield from iter(self._indices)

    @property
    @override
    def sample_count(self):
        return len(self._indices)

    @property
    @override
    def total_count(self):
        return self.queue_size

    @override
    def __len__(self):
        return self.sample_count


class SamplerType(enum.StrEnum):
    TRAINING = enum.auto()
    INFERENCE = enum.auto()


_SAMPLER_CLASS_MAP = {
    SamplerType.TRAINING: TrainingSampler,
    SamplerType.INFERENCE: InferenceSampler,
}


class SamplerFactory:
    __slots__ = ("_fn",)

    def __init__(
        self,
        sampler: SamplerType | str | T.Callable[T.Concatenate[int, _P], Sampler],
        **kwargs,
    ):
        if isinstance(sampler, (str, SamplerType)):
            init_fn = _SAMPLER_CLASS_MAP[SamplerType(sampler)]
        elif isinstance(sampler, type) and issubclass(sampler, Sampler):
            init_fn = sampler
        elif callable(sampler):
            _logger.warning(
                (
                    f"Could not explicitly determine whether `sampler` (type: {type(sampler)}) is a Sampler subclass or "
                    "name, assuming it is a callable that returns a subclass of `torch.utils.data.Sampler`. "
                    "This may lead to unexpected behavior. Please use `SamplerFactory` with a `SamplerType` or a "
                    "`torch.utils.data.Sampler` subclass instead."
                ),
                stacklevel=2,
            )
            init_fn = sampler

        self._fn = functools.partial(init_fn, **kwargs)

    def __call__(self, queue: PerceptionDataqueue) -> Sampler:
        return self._fn(queue)


@D.dataclass()
class DataLoaderConfig:
    """
    Configuration parameters passed to the PyTorch dataoader
    """

    drop_last: bool = False
    pin_memory: bool = D.field(
        default_factory=lambda: laco.get_env(
            bool,
            "UP_DATALOADER_PIN_MEMORY",
            default=True,
        )
    )
    num_workers: int = D.field(
        default_factory=lambda: laco.get_env(
            int,
            "UP_DATALOADER_WORKERS",
            default=min(cpus_available() // 2, 4),
        )
    )
    prefetch_factor: int | None = D.field(
        default_factory=lambda: laco.get_env(
            int,
            "UP_DATALOADER_PREFETCH_FACTOR",
            default=2,
        )
    )
    persistent_workers: bool | None = False


##################
# Loader factory #
##################


@D.dataclass(slots=True, frozen=True)
class DataLoaderFactory:
    """
    Factory for creating dataloaders.

    Attributes
    ----------
    dataset
        The dataset to use.
    actions
        The actions to apply to the dataset (see: ``ops.py``).
    sampler
        The sampler to use.
    config
        The dataloader configuration to use.
    shard_sampler
        See: ``DatasetInterface``. Passing ``None`` uses that class's default.
    shard_chunk_size
        See: ``DatasetInterface``. Passing ``None`` uses that class's default.

    """

    dataset: PerceptionDataset
    sampler: SamplerFactory
    actions: T.Sequence[nn.Module] = D.field(default_factory=list)
    gatherer: QueueGenerator | None = D.field(
        default=None,
        metadata={
            "help": "The gatherer to use to collect items from the dataset into a queue. Defaults to extracting individual frames."
        },
    )
    config: DataLoaderConfig = D.field(default_factory=DataLoaderConfig)
    iterable: bool = D.field(
        default=False,
        metadata={
            "help": (
                "Whether to turn a MapDataset (i.e. a dataset that is not iterable) into an IterableDataset. "
                "See PyTorch DataLoader docs for more information."
            )
        },
    )

    @classmethod
    def with_training_defaults(cls, dataset: PerceptionDataset, **kwargs) -> T.Self:
        """Create a loader factory with default settings for inference mode."""
        if "actions" in kwargs:
            actions = kwargs.pop("actions")
        else:
            actions = []
        if "sampler" in kwargs:
            sampler = kwargs.pop("sampler")
            sampler["sampler"] = "training"
        else:
            sampler = {"sampler": "training"}
        if "config" in kwargs:
            config = kwargs.pop("config")
            config.drop_last = True
        else:
            config = DataLoaderConfig(drop_last=True)
        return cls(
            dataset=dataset,
            sampler=SamplerFactory(**sampler),
            config=config,
            actions=[op.train() for op in actions],
            **kwargs,
        )

    @classmethod
    def with_inference_defaults(cls, dataset: PerceptionDataset, **kwargs) -> T.Self:
        """Create a loader factory with default settings for training mode."""
        if "actions" in kwargs:
            actions = kwargs.pop("actions")
        else:
            actions = []
        if "sampler" in kwargs:
            sampler = kwargs.pop("sampler")
            sampler["sampler"] = "inference"
        else:
            sampler = {"sampler": "inference"}
        if "config" in kwargs:
            config = kwargs.pop("config")
            config["drop_last"] = False
        else:
            config = {"drop_last": False}
        return cls(
            dataset=dataset,
            sampler=SamplerFactory(**sampler),
            config=DataLoaderConfig(**config),
            actions=[op.eval() for op in actions],
            **kwargs,
        )

    def __call__(
        self, batch_size: int | None = None, /, use_distributed: bool = True
    ) -> DataLoader:
        from unipercept.data import SamplerFactory
        from unipercept.data.sets import PerceptionDataset
        from unipercept.model import InputData

        assert isinstance(self.dataset, PerceptionDataset), type(self.dataset)
        assert isinstance(self.sampler, SamplerFactory), type(self.sampler)
        assert isinstance(self.config, DataLoaderConfig), type(self.config)
        assert isinstance(self.actions, T.Sequence), type(self.actions)

        _logger.info("Wrapping dataset: %s", str(self.dataset))

        # Keyword arguments for the loader
        loader_kwargs = {
            k: v for k, v in D.asdict(self.config).items() if v is not None
        }

        # Instantiate sampler
        sampler_kwargs = {}
        if not use_distributed:
            sampler_kwargs["process_count"] = 1
            sampler_kwargs["process_index"] = 0

        queue, pipe = self.dataset(self.gatherer)

        sampler = self.sampler(queue)
        pipe = apply_dataset(pipe, self.actions + [to_tensordict])

        # Create a dataset inteface for the dataloader
        if self.iterable:
            interface_kwargs = {}
            # TODO: find a way to pass these arguments in a way that is mutually exclusive `config.make_iterable`
            # if self.config.shard_sampler is not None:
            #     interface_kwargs["shard_sampler"] = self.shard_sampler
            # if self.config.shard_chunk_size is not None:
            #     interface_kwargs["shared_chunk_size"] = self.shard_chunk_size
            if isinstance(pipe, IterableDataset):
                raise ValueError(
                    f"Dataset {self.dataset} is already an iterable dataset, cannot wrap it in another iterable dataset!"
                )
            interface = DatasetInterface(ifc, sampler, **interface_kwargs)
            _logger.debug(
                "Transformed map-style dataset to iterable-style dataset: %s",
                str(interface),
            )
            loader_kwargs["sampler"] = None
        else:
            interface = pipe
            loader_kwargs["sampler"] = sampler

        # Loader
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["collate_fn"] = functools.partial(
            self.wrap_collate_with_replacement,
            collate_fn=loader_kwargs.get("collate_fn", InputData.collate),
            dataset=interface,
        )
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs.setdefault("worker_init_fn", _worker_init_fn)
        elif "worker_init_fn" in loader_kwargs:
            msg = f"Worker init function is set, but num_workers is {loader_kwargs['num_workers']}."
            raise ValueError(msg)

        _logger.debug(
            "Creating dataloader (%d queued; %d × %d items):\n%s",
            len(queue),
            len(interface),
            batch_size,
            create_table(loader_kwargs, format="long"),
            # tabulate(loader_kwargs.items(), tablefmt="simple"),
        )

        return DataLoader(interface, **loader_kwargs)

    @staticmethod
    def wrap_collate_with_replacement(
        batch, *, collate_fn: T.Callable, dataset: Dataset
    ):
        """
        Collate function that replaces missing items (``None``) with a random item in
        the dataset.

        This only applies to map-style datasets (where the sample cannot be skipped
        during iteration).

        Parameters
        ----------
        batch
            The batch to collate.
        dataset
            The **map-style** dataset to use, e.g. with a ``__getitem__`` and
            a ``__len__`` method.

        Returns
        -------
        dict
            The collated batch.
        """
        batch = [
            (
                item
                if item is not None
                else dataset[
                    random.randint(
                        0,
                        len(dataset) - 1,  # type: ignore
                    )
                ]
            )
            for item in batch
        ]
        return collate_fn(batch)


#######################
# Dataset Preparation #
#######################


class DatasetInterface(IterableDataset):
    """
    Use a map-style dataset as an iterable dataset.

    Based on Detectron2 implementation: `detectron2.data.ToIterableDataset`.
    """

    @staticmethod
    def _roundrobin(*iterables):
        """Roundrobin('ABC', 'D', 'EF') --> A D E B F C."""
        from itertools import cycle, islice

        num_active = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    @staticmethod
    def _worker(iterable, *, chunk_size=1, strategy=_roundrobin):
        from itertools import islice

        # Shard the iterable if we're currently inside pytorch dataloader worker.
        worker_info = get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            # do nothing
            yield from iterable
        else:
            # worker0: 0, 1, ..., chunk_size-1, num_workers*chunk_size, num_workers*chunk_size+1, ...
            # worker1: chunk_size, chunk_size+1, ...
            # worker2: 2*chunk_size, 2*chunk_size+1, ...
            # ...
            yield from strategy(
                *[
                    islice(
                        iterable,
                        worker_info.id * chunk_size + chunk_i,
                        None,
                        worker_info.num_workers * chunk_size,
                    )
                    for chunk_i in range(chunk_size)
                ]
            )

    __slots__ = ("dataset", "sampler", "shard_sampler", "shard_chunk_size")

    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        shard_sampler: bool = True,
        shard_chunk_size: int = 1,
    ):
        """
        Parameters
        ----------
        dataset
            A map-style dataset.
        sampler
            A cheap iterable that produces indices to be applied on ``dataset``.
        shard_sampler
            Whether to shard the sampler based on the current pytorch data loader worker id.
            When an IterableDataset is forked by pytorch's DataLoader into multiple workers, it is responsible for
            sharding its data based on worker id so that workers don't produce identical data.
            Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
            and this argument should be set to True.
            But certain samplers may be already
            sharded, in that case this argument should be set to False.
        shard_chunk_size:
            When sharding the sampler, each worker will only produce 1/N of the ids
        """
        assert not isinstance(dataset, IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler
        self.shard_chunk_size = shard_chunk_size

    @override
    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = self._worker(self.sampler, chunk_size=self.shard_chunk_size)
        for idx in sampler:
            item = self.dataset[idx]
            if item is None:
                continue
            yield item

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore


####################
# Worker Functions #
####################


def _worker_init_fn(worker_id: int) -> None:
    """Worker init function that resets the random seed."""
    import os
    import random

    import numpy as np
    import torch

    seed = (torch.initial_seed() % 2**31) + worker_id

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
