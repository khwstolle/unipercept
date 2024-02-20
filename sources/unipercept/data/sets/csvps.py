r"""
Cityscapes VPS
==============

This module provides the Cityscapes VPS dataset.
This requires manual installation according to
    `these instructions <https://huggingface.co/datasets/khwstolle/csvps>`_.

See Also
--------

- `Cityscapes Dataset <https://www.cityscapes-dataset.com/>`_
- `Video Panoptic Segmentation <https://github.com/mcahny/vps>`_
- `Hugging Face Hub <https://huggingface.co/datasets/khwstolle/csvps>`_
"""

from dataclasses import KW_ONLY, dataclass

from unipercept.file_io import Path

from .webdataset import WebDatasetBuilder


@dataclass
class CSVPSConfig(WebDatasetConfig):
    r"""
    Configuration object for CSVPS.
    """

    split: str
    _: KW_ONLY
    root: str = "datasets://csvps"


class CSVPSDatasetBuilder(
    WebDatasetBuilder[CSVPSConfig, CSVPSDataset, CSVPSInfo, CSVPSManifest]
):
    r"""
    Builder class for CSVPS.
    """

    def check(self):
        path = Path(self.config.root)
        if not path.exists():
            raise FileNotFoundError(f"CSVPS root directory not found: {path}")

    def build(self):
        pass

    @classmethod
    def describe(cls):
        pass
