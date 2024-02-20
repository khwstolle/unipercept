r"""
Generic COCO-like dataset.
"""

from __future__ import annotations

import typing_extensions as TX

from . import (
    Metadata,
    PerceptionDataset,
)


@D.dataclass(kw_only=True)
class GenericCOCODataset(
    PerceptionDataset, id=None, info=None, use_manifest_cache=False
):
    """A generic dataset that uses COCO-like dataset JSON files."""

    path: str
    like: str

    @TX.override
    def read_info(self) -> Metadata:
        from . import catalog

        return catalog.get_info(self.like)
