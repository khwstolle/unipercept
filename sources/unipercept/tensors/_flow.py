r"""
Flow tensors
============

Provides support for optical and scene flow tensors.
"""

from torchvision.tv_tensors import Mask

from .registry import pixel_maps

__all__ = ["OpticalFlow", "SceneFlow"]


@pixel_maps.register()
class OpticalFlow(Mask):
    pass

@pixel_maps.register()
class SceneFlow(Mask):
    pass
