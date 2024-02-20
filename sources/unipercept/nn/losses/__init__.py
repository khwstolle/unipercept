"""
This module hosts various losses for perception tasks.
"""

import unipercept.utils.module

__all__ = []
__getattr__, __dir__ = unipercept.utils.module.lazy_module_factory(
    __name__,
    [
        "contrastive",
        "depth",
        "focal",
        "functional",
        "guided",
        "image",
        "mixins",
        "panoptic",
    ],
)
