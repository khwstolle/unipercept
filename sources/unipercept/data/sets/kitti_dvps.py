"""
Semantic KITTI-DVPS dataset
===========================

Expects the dataset to be installed and in the following format:

.. code-block:: none
    $UNICORE_DATA
        |── semkitti-dvps
        │   ├── video_sequence
        │   │   ├── train
        │   │   │   ├── 000000_000000_leftImg8bit.png
        │   │   │   ├── 000000_000000_gtFine_class.png
        │   │   │   ├── 000000_000000_gtFine_instance.png
        │   │   │   ├── 000000_000000_depth_718.8560180664062.png
        │   │   │   ├── ...
        │   │   ├── val
        │   │   │   ├── ...

The number before the ``.png`` extension for the depth label is the focal length of
the camera.
"""

import concurrent.futures
import dataclasses as D
import functools as F
import os
import typing as T
import zipfile

import expath
from tqdm import tqdm

from unipercept.state import cpus_available
from unipercept.utils.image import size as get_image_size
from unipercept.utils.time import get_timestamp

from ._base import PerceptionDataset
from ._manifest import (
    CameraModelParameters,
    CaptureRecord,
    CaptureSources,
    Manifest,
    ManifestSequence,
)
from ._metadata import RGB, Metadata, SClass, SType

__all__ = ["KITTIDVPSDataset"]

DOWNLOAD_URL: T.Final = (
    "https://huggingface.co/HarborYuan/PolyphonicFormer/resolve/main/semkitti-dvps.zip"
)
DEFAULT_FOCAL_LENGTH: T.Final = 718.8560180664062
CAPTURE_FPS: T.Final = 17.0


def get_info() -> Metadata:
    sem_list = [
        SClass(
            color=RGB(0, 0, 0),
            kind=SType.VOID,
            dataset_id=255,
            unified_id=-1,
            name="void",
        ),
        SClass(
            color=RGB(245, 150, 100),
            kind=SType.THING,
            dataset_id=0,
            unified_id=0,
            name="car",
        ),
        SClass(
            color=RGB(245, 230, 100),
            kind=SType.THING,
            dataset_id=1,
            unified_id=1,
            name="bicycle",
        ),
        SClass(
            color=RGB(150, 60, 30),
            kind=SType.THING,
            dataset_id=2,
            unified_id=2,
            name="motorcycle",
        ),
        SClass(
            color=RGB(180, 30, 80),
            kind=SType.THING,
            dataset_id=3,
            unified_id=3,
            name="truck",
        ),
        SClass(
            color=RGB(255, 0, 0),
            kind=SType.THING,
            dataset_id=4,
            unified_id=4,
            name="other-vehicle",
        ),
        SClass(
            color=RGB(30, 30, 255),
            kind=SType.THING,
            dataset_id=5,
            unified_id=5,
            name="person",
        ),
        SClass(
            color=RGB(200, 40, 255),
            kind=SType.THING,
            dataset_id=6,
            unified_id=6,
            name="bicyclist",
        ),
        SClass(
            color=RGB(90, 30, 150),
            kind=SType.THING,
            dataset_id=7,
            unified_id=7,
            name="motorcyclist",
        ),
        SClass(
            color=RGB(255, 0, 255),
            kind=SType.STUFF,
            dataset_id=8,
            unified_id=8,
            name="road",
        ),
        SClass(
            color=RGB(255, 150, 255),
            kind=SType.STUFF,
            dataset_id=9,
            unified_id=9,
            name="parking",
        ),
        SClass(
            color=RGB(75, 0, 75),
            kind=SType.STUFF,
            dataset_id=10,
            unified_id=10,
            name="sidewalk",
        ),
        SClass(
            color=RGB(75, 0, 175),
            kind=SType.STUFF,
            dataset_id=11,
            unified_id=11,
            name="other-ground",
        ),
        SClass(
            color=RGB(0, 200, 255),
            kind=SType.STUFF,
            dataset_id=12,
            unified_id=12,
            name="building",
        ),
        SClass(
            color=RGB(50, 120, 255),
            kind=SType.STUFF,
            dataset_id=13,
            unified_id=13,
            name="fence",
        ),
        SClass(
            color=RGB(0, 175, 0),
            kind=SType.STUFF,
            dataset_id=14,
            unified_id=14,
            name="vegetation",
        ),
        SClass(
            color=RGB(0, 60, 135),
            kind=SType.STUFF,
            dataset_id=15,
            unified_id=15,
            name="trunk",
        ),
        SClass(
            color=RGB(80, 240, 150),
            kind=SType.STUFF,
            dataset_id=16,
            unified_id=16,
            name="terrain",
        ),
        SClass(
            color=RGB(150, 240, 255),
            kind=SType.STUFF,
            dataset_id=17,
            unified_id=17,
            name="pole",
        ),
        SClass(
            color=RGB(0, 0, 255),
            kind=SType.STUFF,
            dataset_id=18,
            unified_id=18,
            name="traffic-sign",
        ),
    ]

    return Metadata.from_parameters(
        sem_list,
        depth_max=80.0,
        fps=17.0,
    )


@D.dataclass(kw_only=True)
class KITTIDVPSDataset(
    PerceptionDataset, info=get_info, id="kitti-dvps", version="3.0"
):
    """
    Implements the KITTISemanticDVPS dataset introduced by *ViP-DeepLab: [...]* (Qiao et al, 2021).

    Paper: https://arxiv.org/abs/2106.10867
    """

    split: T.Literal["train", "val"]
    root: str = "//unipercept/datasets/semkitti-dvps"
    pseudo: bool = True

    @classmethod
    @T.override
    def options(cls) -> dict[str, list[str]]:
        return {
            "split": ["train", "val"],
        }

    @T.override
    def download(self, *, force: bool = False) -> None:
        """
        Download and extract the dataset.

        The default download URL is provided by the authors of PolyphonicFormer.
        """

        if expath.is_dir(self.root) and not force:
            return

        archive_path = expath.locate(DOWNLOAD_URL)
        with zipfile.ZipFile(archive_path) as zip:
            zip.extractall(self.root)
        expath.rm(archive_path)

    @T.override
    def _build_manifest(self) -> Manifest:
        cap_root = expath.locate(self.root) / "video_sequence" / self.split
        if not cap_root.exists():
            msg = f"Captures path {cap_root} does not exist!"
            raise RuntimeError(msg)

        captures = list(_discover_captures(cap_root).items())

        if len(captures) == 0:
            msg = f"No images found in {cap_root}"
            raise RuntimeError(msg)

        with (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=cpus_available(),
            ) as pool,
        ):
            sequences: dict[str, ManifestSequence] = {}
            for seq_id, cap in tqdm(
                pool.map(
                    F.partial(_build_capture_record, root=cap_root),
                    captures,
                    chunksize=32,
                ),
                total=len(captures),
                desc="Building capture records",
            ):
                # Ensure that the sequence exists
                seq = sequences.setdefault(
                    seq_id,
                    {
                        "camera": None,
                        "fps": CAPTURE_FPS,
                        "captures": [],
                        "motions": [],
                    },
                )
                seq["captures"].append(cap)

            # Build the camera records
            for seq_id, cam in tqdm(
                pool.map(_build_camera_record, list(sequences.items())),
                total=len(sequences),
                desc="Building camera records",
            ):
                sequences[seq_id]["camera"] = cam

        return {
            "timestamp": get_timestamp(),
            "version": self.version,
            "sequences": sequences,
        }


####################
# Helper Functions #
####################


def _discover_captures(
    directory: expath.locate, *, unknown_ok: bool = True
) -> dict[tuple[str, str], list[str | None]]:
    files_map: dict[tuple[str, str], list[str | None]] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            seq_id, frame_name, *_ = entry.name.split("_")
            files = files_map.setdefault((seq_id, frame_name), [None, None, None, None])
            if entry.name.endswith("_leftImg8bit.png"):
                files[0] = entry.name
            elif entry.name.endswith("_gtFine_class.png"):
                files[1] = entry.name
            elif entry.name.endswith("_gtFine_instance.png"):
                files[2] = entry.name
            elif entry.name.endswith(".png") and "_depth_" in entry.name:
                files[3] = entry.name
            elif not unknown_ok:
                msg = f"Unknown file found: {entry.path}"
                raise ValueError(msg)

    return files_map


def _build_camera_record(
    item: tuple[str, ManifestSequence],
) -> tuple[str, CameraModelParameters]:
    key, sequence = item
    try:
        cap_path = sequence["captures"][0]["sources"]["image"]["path"]
        assert isinstance(cap_path, str), f"Capture path is not a string: {cap_path}"
    except KeyError as err:
        msg = f"Capture path not found in sequence: {key}"
        raise KeyError(msg) from err

    dep_src = sequence["captures"][0]["sources"].get("depth")
    if dep_src is not None:
        dep_path = dep_src["path"]
        assert isinstance(dep_path, str), f"Depth path is not a string: {dep_path}"
        focal_length = float(expath.locate(dep_path).stem.split("_")[-1])
    else:
        focal_length = DEFAULT_FOCAL_LENGTH

    image_size = get_image_size(cap_path)
    cam: CameraModelParameters = {
        "focal_length": (focal_length, focal_length),
        "principal_point": (
            image_size.height // 2,
            image_size.width // 2,
        ),
        "rotation": (0.0, 0.0, 0.0),
        "translation": (0.0, 0.0, 0.0),
        "image_size": (image_size.height, image_size.width),
        "convention": "iso8855",
    }

    return key, cam


def _build_capture_record(
    item: tuple[tuple[str, str], list[str]], *, root: expath.locate
) -> tuple[str, CaptureRecord]:
    (seq_id, frame_name), files = item

    if len(files) != 4:  # noqa: PLR2004
        msg = f"Expected 4 files, got {len(files)} for {seq_id}_{frame_name}"
        raise ValueError(msg)

    # Read the file names from the list
    img_file, sem_file, ins_file, dep_file = files

    # Build the primary key
    primary_key = f"{seq_id}_{frame_name}"

    # Image file is required
    if img_file is None:
        msg = f"Image file not found for {primary_key}"
        raise ValueError(msg)

    # Create sources record
    sources: CaptureSources = {
        "image": {
            "path": str(root / img_file),
        }
    }

    # Add optional sources
    if dep_file is not None:
        sources["depth"] = {
            "path": str(root / dep_file),
            "meta": {
                "format": "depth_int16",
            },
        }
    if sem_file is not None:
        sources["semantic"] = {
            "path": str(root / sem_file),
            "meta": {"format": "png_l16"},
        }

    if ins_file is not None:
        sources["instance"] = {
            "path": str(root / ins_file),
            "meta": {"format": "png_l16"},
        }

    # Create the capture record
    rec: CaptureRecord = {"primary_key": primary_key, "sources": sources}

    return seq_id, rec
