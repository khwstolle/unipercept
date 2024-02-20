"""
Data operations for input data.
"""

import random
import typing as T
from typing import override

import torch
import torch.nn
import torch.types
import torch.utils.data as torch_data
import torchvision.ops
import torchvision.transforms.v2.functional
from tensordict import TensorDictBase
from torch import nn

from unipercept import tensors
from unipercept.log import logger
from unipercept.types import Tensor
from unipercept.utils.pickle import as_picklable
from unipercept.utils.tensorclass import Tensorclass, is_tensordict_like


class TransformReject(Exception):
    """
    Exception that is raised when a operation fails to modify the input data in a way
    that is acceptable.

    The current data point is discarded and the loader will continue with the next one.
    """

    def __init__(self, message: str):
        super().__init__(message)


class TransformSkip(Exception):
    """
    Exception that is raised when a operation flags itself to be skipped.
    """

    def __init__(self, message: str):
        super().__init__(message)


def transform_inputs(
    inputs: TensorDictBase,
    ops: T.Iterable[nn.Module],
) -> T.Iterator[TensorDictBase | Tensorclass]:
    ops = list(ops)
    res: T.Any = inputs
    for index, fn in enumerate(ops):
        # Try to apply each operation in the sequence.
        try:
            res = fn(res)
        except TransformSkip as e:
            logger.debug("Transform was skipped: %s", e)
            continue
        except TransformReject as e:
            logger.debug("Transform rejected sample: %s", e)
            return

        # If the operation returns None, skip the rest of the sequence.
        if res is None:
            logger.warning("Transformed data is None, skipping!", stacklevel=2)
            return

        # If the operation returns a sequence, split it and apply the operations to
        # each item.
        if is_tensordict_like(res):
            continue
        if isinstance(res, T.Iterable):
            for item in res:
                yield from transform_inputs(item, ops[index + 1 :])
            return
        msg = f"Non-TensorDictBase object is not iterable, got {type(res)}!"
        raise ValueError(msg)
    if is_tensordict_like(res):
        yield res
    elif isinstance(res, T.Iterable):
        yield from res
    msg = f"Transformed data is not a TensorDictBase, got {type(res)}!"
    raise ValueError(msg)


class _TransformedIterable[_Q, _R: TensorDictBase](torch_data.IterableDataset):
    """Applies a sequence of transformations to an iterable dataset."""

    __slots__ = ("_set", "_fns")

    def __init__(
        self, dataset: torch_data.Dataset[tuple[_Q, _R]], fns: T.Sequence[nn.Module]
    ):
        self._set = dataset
        self._fns = list(as_picklable(fn) for fn in fns)

        assert len(self) >= 0

    def __len__(self):
        if not isinstance(self._set, T.Sized):
            raise ValueError(f"Dataset {self._set} must be sized!")
        return len(self._set)

    def __getnewargs__(
        self,
    ) -> tuple[torch_data.Dataset[tuple[_Q, _R]], list[nn.Module]]:
        return self._set, self._fns

    @override
    def __str__(self):
        return f"{str(self._set)} ({len(self._fns)} transforms)"

    @override
    def __repr__(self):
        return f"<{repr(self._set)} x {len(self._fns)} transforms>"

    @override
    def __iter__(self) -> T.Iterator[tuple[_Q, _R]]:
        it = iter(self._set)
        while True:
            try:
                item, data = next(it)
            except StopIteration:
                return
            with tensors.helpers.transform_context():
                data_list = list(transform_inputs(data, self._fns))
            for data in data_list:
                yield item, data


class _TransformedMap[_Q, _R: TensorDictBase](torch_data.Dataset[tuple[_Q, _R]]):
    """Applies a sequence of transformations to an iterable dataset."""

    __slots__ = ("_set", "_fns", "_retry", "_fallback_candidates")

    def __init__(
        self,
        dataset: torch_data.Dataset[tuple[_Q, _R]],
        fns: T.Sequence[nn.Module],
        *,
        max_retry: int = 100,
    ):
        self._set = dataset
        self._fns = list(as_picklable(fn) for fn in fns)

        assert len(self) >= 0
        self._retry = max_retry
        self._fallback_candidates: set[int | str] = set()
        self._random = random.Random(42)

    def __len__(self):
        if not isinstance(self._set, T.Sized):
            raise ValueError(f"Dataset {self._set} must be sized!")
        return len(self._set)

    # def __getnewargs__(self) -> tuple[_D, list[nn.Module]]:
    #    return self._set, self._fns

    @override
    def __str__(self):
        return f"{str(self._set)} ({len(self._fns)} transforms)"

    @override
    def __repr__(self):
        return f"<{repr(self._set)} x {len(self._fns)} transforms>"

    @override
    def __getitem__(self, idx: int | str) -> tuple[_Q, _R]:
        for _ in range(self._retry):
            item, data = self._set[idx]
            with tensors.helpers.transform_context():
                data = next(transform_inputs(data, self._fns), None)
            if data is None:
                self._fallback_candidates.discard(idx)
                if len(self._fallback_candidates) == 0:
                    idx = self._random.randint(0, len(self) - 1)
                else:
                    idx = self._random.sample(list(self._fallback_candidates), k=1)[0]
            else:
                self._fallback_candidates.add(idx)
                return item, data

        raise RuntimeError(f"Failed to apply transforms after {self._retry} retries!")


def apply_dataset[_Q, _R: TensorDictBase](
    dataset: torch_data.Dataset[tuple[_Q, _R]], actions: T.Sequence[nn.Module]
) -> _TransformedMap[_Q, _R] | _TransformedIterable[_Q, _R]:
    """Map a function over the elements in a dataset."""
    if isinstance(dataset, torch_data.IterableDataset):
        return _TransformedIterable(dataset, actions)
    return _TransformedMap(dataset, actions)


class PadToDivisible(nn.Module):
    """
    Pads the input to be divisible by a given number.
    """

    __constants__ = ("divisor",)

    def __init__(self, divisor: int):
        super().__init__()
        self.divisor = divisor
        self.fill_values = {
            tensors.PanopticTensor: tensors.PanopticTensor.IGNORE,
            tensors.ImageTensor: 127,
        }

    @override
    def forward(self, *data: Tensor) -> Tensor:
        assert len(data) > 0
        h, w = data[0].shape[-2:]

        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor

        if pad_h == 0 and pad_w == 0:
            return data

        def apply_padding(x: torch.Tensor) -> torch.Tensor:
            return torchvision.transforms.v2.functional.pad(
                x,
                [0, 0, pad_w, pad_h],
                fill=next(
                    pad_value
                    for pad_value in (
                        self.fill_values.get(type(x)),
                        next(
                            (
                                v
                                for t, v in self.fill_values.items()
                                if isinstance(x, t)
                            ),
                            None,
                        ),
                        0,
                    )
                    if pad_value is not None
                ),
            )

        return tuple(apply_padding(x) for x in data)


class BoxesFromMasks(nn.Module):
    """
    Adds bounding boxes for each ground truth mask in the input segmentation.
    """

    def __init__(self):
        super().__init__()

    @override
    def forward(
        self, segmentation: tensors.PanopticTensor
    ) -> list[tensors.BoundingBoxes]:
        T, H, W = segmentation.shape

        boxes = [
            torchvision.ops.masks_to_boxes(
                torch.stack(
                    [m for _, m in tensors.PanopticTensor.get_instance_masks(seg)]
                )
            )
            for seg in segmentation.unbind(0)
        ]

        return [
            tensors.BoundingBoxes(
                b, format=tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W)
            )
            for b in boxes
        ]
