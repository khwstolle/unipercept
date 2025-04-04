"""
Wrap and register torchvision's tensor types.
"""

import enum as E
import typing as T

import expath
import safetensors
import torch
import torchvision.io
from torchvision.tv_tensors import (
    BoundingBoxes,
    BoundingBoxFormat,
)
from torchvision.tv_tensors import (
    Image as ImageTensor,
)
from torchvision.tv_tensors import (
    Mask as MaskTensor,
)

from unipercept.types import Tensor

from .registry import pixel_maps

__all__ = [
    "ImageTensor",
    "MaskTensor",
    "MaskFormat",
    "MaskMeta",
    "load_mask",
    "save_mask",
    "BoundingBoxes",
    "BoundingBoxFormat",
    "load_image",
]

########
# Mask #
########

pixel_maps.add(MaskTensor)


class MaskFormat(E.StrEnum):
    PNG_L = E.auto()
    PNG_LA = E.auto()
    PNG_L16 = E.auto()
    PNG_LA16 = E.auto()
    TORCH = E.auto()
    SAFETENSORS = E.auto()


class MaskMeta(T.TypedDict):
    format: T.NotRequired[MaskFormat]


def load_mask(
    input: expath.PathType | torch.Tensor, /, **kwargs: MaskMeta
) -> MaskTensor:
    format = kwargs["format"]

    match format:
        case MaskFormat.PNG_L | MaskFormat.PNG_L16:
            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            return torchvision.io.decode_png(
                input, mode=torchvision.io.ImageReadMode.GRAY
            ).as_subclass(MaskTensor)
        case MaskFormat.PNG_LA | MaskFormat.PNG_LA16:
            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            return torchvision.io.decode_png(
                input, mode=torchvision.io.ImageReadMode.GRAY
            ).as_subclass(MaskTensor)
        case MaskFormat.TORCH:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            with expath.open(input, "rb") as fh:
                return torch.load(fh).as_subclass(MaskTensor)
        case MaskFormat.SAFETENSORS:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            input = str(expath.locate(input))
            return safetensors.torch.load(input).as_subclass(MaskTensor)
        case _:
            msg = f"Unsupported format: {format}"
            raise NotImplementedError(msg)


def save_mask(
    tensor: Tensor | MaskTensor, path: expath.PathType, /, **kwargs: MaskMeta
) -> MaskTensor:
    raise NotImplementedError("Not implemented yet.")


#########
# Image #
#########

pixel_maps.add(ImageTensor)


def load_image(input: expath.PathType | torch.Tensor) -> ImageTensor:
    """Reads an image from the given path."""
    from torchvision.transforms.v2.functional import to_dtype, to_image

    if not isinstance(input, torch.Tensor):
        input = str(expath.locate(input))
    img = torchvision.io.decode_image(input)
    img = to_image(img)
    img = to_dtype(img, torch.float32, scale=True)

    assert img.shape[0] == 3, f"Expected image to have 3 channels, got {img.shape[0]}!"

    return img.as_subclass(ImageTensor)
