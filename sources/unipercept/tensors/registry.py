"""
Register tensors belonging to certain categories.

The following categories are available:

- `pixel_maps`: Pixel maps are tensors that represent 2D (image/pixel) data
- `point_maps`: Point maps are tensors that represent 3D point cloud data
"""

import torch

from unipercept.types import Tensor
from unipercept.utils.registry import AnonymousRegistry

__all__ = ["pixel_maps", "point_maps"]


def _check_tensorsubclass(cls: type) -> bool:
    return issubclass(cls, torch.Tensor)


pixel_maps = AnonymousRegistry[type[Tensor]](check=_check_tensorsubclass)
point_maps = AnonymousRegistry[type[Tensor]](check=_check_tensorsubclass)
