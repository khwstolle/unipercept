"""
Wrap and register torchvision's tensor types.
"""

import enum as E
import typing as T

import PIL.Image as pil_image
import safetensors
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
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

from unipercept.file_io import get_local_path
from unipercept.types import Pathable, Tensor

from .helpers import read_pixels
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


def load_mask(path: Pathable, /, **kwargs: MaskMeta) -> MaskTensor:
    format = kwargs["format"]
    path = get_local_path(path)

    match format:
        case MaskFormat.PNG_L | MaskFormat.PNG_L16:
            return read_pixels(path, color=False, alpha=False).as_subclass(MaskTensor)
        case MaskFormat.PNG_LA | MaskFormat.PNG_LA16:
            return read_pixels(path, color=False, alpha=True).as_subclass(MaskTensor)
        case MaskFormat.TORCH:
            return torch.load(path).as_subclass(MaskTensor)
        case MaskFormat.SAFETENSORS:
            return safetensors.torch.load(path).as_subclass(MaskTensor)
        case _:
            msg = f"Unsupported format: {format}"
            raise NotImplementedError(msg)


def save_mask(
    tensor: Tensor | MaskTensor, path: Pathable, /, **kwargs: MaskMeta
) -> MaskTensor:
    raise NotImplementedError("Not implemented yet.")


#########
# Image #
#########

pixel_maps.add(ImageTensor)


def load_image(path: Pathable) -> ImageTensor:
    """Reads an image from the given path."""
    path = get_local_path(str(path))

    with pil_image.open(path) as img_pil:
        img_pil = img_pil.convert("RGB")
        img = to_image(img_pil)
        img = to_dtype(img, torch.float32, scale=True)

    assert img.shape[0] == 3, f"Expected image to have 3 channels, got {img.shape[0]}!"

    return img.as_subclass(ImageTensor)
