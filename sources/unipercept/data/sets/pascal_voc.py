"""
Pascal VOC dataset, with panoptic segmentation.

See: https://pytorch.org/vision/0.10/_modules/torchvision/datasets/voc.html#VOCSegmentation
"""

from __future__ import annotations

import typing as T
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, override

import expath
import typing_extensions as TX
from tqdm import tqdm

from unipercept.data.pseudolabeler import PseudoGenerator
from unipercept.render import colormap

from . import (
    RGB,
    Metadata,
    PerceptionDataset,
    SClass,
    SType,
)

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = ["PascalVOCDataset"]


# ----------- #
# Static info #
# ----------- #


def get_info() -> up.data.sets.Metadata:
    class_names = [
        "void",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    cmap = colormap(rgb=True, maximum=255)

    sem_list = []
    for i, name in enumerate(class_names):
        sem_list.append(
            SClass(
                color=RGB(*cmap[i]),
                kind=SType.THING if i > 0 else SType.VOID,
                dataset_id=i,
                unified_id=i - 1,
                name=name,
            )
        )

    return Metadata.from_parameters(
        sem_list,
        depth_max=80.0,
        fps=15.0,
    )


# -------------------- #
# Dataset and variants #
# -------------------- #


DEPTH_FORMAT = "tiff"
PANOPTIC_FORMAT = "safetensors"


@D.dataclass(frozen=True)
class PascalVOCDataset(PerceptionDataset, id="pascal-voc", info=get_info):
    """
    Pascal VOC w/ panoptic segmentation.

    Panoptic segmentation ground truths are generated on the fly, and cached to disk.  See: ``_build_panoptic``.
    """

    root: str = "//unipercept/datasets/voc"
    split: Literal["train", "val", "trainval", "test"] = "train"
    year: str = "2012"

    def _get_torchvision_dataset(self, *, download=False):
        from torchvision.datasets import VOCSegmentation

        root = expath.locate(self.root)

        return VOCSegmentation(
            root, year=self.year, image_set=self.split, download=download
        )

    @TX.override
    def download(self, *, force: bool = False) -> None:
        dataset = self._get_torchvision_dataset(download=True)
        del dataset

    @classmethod
    @TX.override
    def options(cls):
        return {
            "split": ["train", "val", "trainval", "test"],
            "year": ["2007", "2012"],
        }

    def _build_panoptic(self) -> list[tuple[str, str, str]]:
        dataset = self._get_torchvision_dataset()
        src_img: list[str] = dataset.images
        src_seg: list[str] = dataset.targets
        src_pan = [
            t.replace("SegmentationClass", "SegmentationPanoptic").replace(
                dataset._TARGET_FILE_EXT, f".{PANOPTIC_FORMAT}"
            )
            for t in src_seg
        ]
        src_dep: list[str] = [
            t.replace("SegmentationClass", "Depth").replace(
                dataset._TARGET_FILE_EXT, f".{DEPTH_FORMAT}"
            )
            for t in src_seg
        ]
        del dataset

        with PseudoGenerator() as pseudo_gen:
            for i in tqdm(range(len(src_img)), desc="Building (pseudo) labels"):
                pan_path = expath.locate(src_pan[i])
                if not pan_path.is_file():
                    seg_path = Path(src_seg[i])
                    ins_path = Path(
                        src_seg[i].replace("SegmentationClass", "SegmentationObject")
                    )
                    pseudo_gen.add_panoptic_merge_task(seg_path, ins_path, pan_path)

                img_path = expath.locate(src_img[i])
                dep_path = expath.locate(src_dep[i])
                if not dep_path.is_file():
                    pseudo_gen.add_depth_generator_task(img_path, dep_path)

        return list(zip(src_img, src_pan, src_dep, strict=False))

    @override
    def _build_manifest(self) -> up.data.types.Manifest:
        paths = self._build_panoptic()

        # Convert to mapping of string -> dt.CaptureRecord
        sequences: Mapping[str, up.data.types.ManifestSequence] = {}
        for i, (img_path, pan_path, dep_path) in enumerate(paths):
            captures: list[up.data.types.CaptureRecord] = [
                {
                    "primary_key": str(i),
                    "sources": {
                        "image": {
                            "path": str(img_path),
                        },
                        "panoptic": {
                            "path": str(pan_path),
                            "meta": {
                                "format": PANOPTIC_FORMAT,
                            },
                        },
                        "depth": {
                            "path": str(dep_path),
                            "meta": {
                                "format": DEPTH_FORMAT,
                            },
                        },
                    },
                }
            ]

            # Create sequence item
            seq_item: up.data.types.ManifestSequence = {
                "camera": {
                    "focal_length": (1, 1),
                    "principal_point": (1, 1),
                    "rotation": (0, 0, 0),
                    "translation": (0, 0, 0),
                    "image_size": (512, 512),
                },
                "fps": -1,
                "captures": captures,
            }
            sequences[i] = seq_item

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": f"{self.year}-panoptic",
            "sequences": sequences,
        }
