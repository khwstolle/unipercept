r"""
WebDataset
==========

A format for efficient storage and streaming of large datasets. Uses simple
tar archives to store files and metadata.

See Also
--------

- `WebDataset Format Specification <https://docs.google.com/document/d/18OdLjruFNX74ILmgrdiCI9J1fQZuhzzRBCHV9URWto0/edit?usp=sharing>`_
- `WebDataset GitHub repository <https://github.com/webdataset/webdataset>`_
"""

from typing import override

import webdataset as wds

from unipercept.data.sets.builder import Dataset, DatasetBuilder
from unipercept.file_io import Path, Pathable


class WebDatasetManifest:
    r"""
    Represents the manifest of a WebDataset.
    """

    def __init__(self, path: Pathable):
        self.path = Path(path)


class WebDatasetBuilder[Config, Manifest, Metadata](
    DatasetBuilder[Config, WebDatasetManifest, Metadata]
):
    r"""
    Utility class for interacting with WebDataset structures.
    """

    @override
    def build(self, config: Config) -> Manifest:
        pass

    @override
    def describe(cls, config: Config) -> Metadata:
        pass


class WebDataset[Metadata](Dataset[WebDatasetManifest, Metadata]):
    def __init__(self, *sources: WebDatasetManifest, metadata: _I):
        self.data = wds.WebDataset()
