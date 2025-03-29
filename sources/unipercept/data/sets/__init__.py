"""This module contains the dataset modules."""

# from unipercept.utils.module import lazy_module_factory

from ._base import *
from ._manifest import *
from ._metadata import *


# __getattr__, __dir__ = lazy_module_factory(
#     __name__,
#     [
#         "cityscapes",
#         "coco",
#         "huggingface",
#         "kitti_360",
#         "kitti_dvps",
#         "kitti_step",
#         "pascal_voc",
#         "pattern",
#         "vistas",
#         "wilddash",
#     ],
# )

# del lazy_module_factory
