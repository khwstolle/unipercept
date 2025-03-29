r"""
Mask tensors
============

Implements mask tensors for semantic and instance segmentation maps, and their
(panoptic segmentation) joint representation.
"""

import enum as E
import torchvision.io
import typing as T

import expath
import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
from typing_extensions import deprecated
from unipercept.data.types.coco import COCOResultPanopticSegment
from unipercept.tensors.helpers import write_png_l16, write_png_rgb
from unipercept.tensors.registry import pixel_maps
from unipercept.types import Tensor

from ._torchvision import MaskTensor

__all__ = [
    "PanopticTensor",
    "PanopticFormat",
    "PanopticTensorLike",
    "save_panoptic",
    "load_panoptic",
]

_L = T.TypeVar("_L", int, Tensor)
_BYTE_OFFSET: T.Final = 256


class PanopticFormat(E.StrEnum):
    """
    Enumerates the different formats of labels that are supported. Uses the name of
    the dataset that introduced the format.
    """

    CITYSCAPES = E.auto()
    CITYSCAPES_VPS = E.auto()
    CITYSCAPES_DVPS = E.auto()
    KITTI = E.auto()
    VISTAS = E.auto()
    WILD_DASH = E.auto()
    TORCH = E.auto()
    SAFETENSORS = E.auto()
    PNG_UINT16 = E.auto()
    PNG_L16 = E.auto()


@pixel_maps.register()
class PanopticTensor(MaskTensor):
    """
    Implements a panoptic segmentation map, where each pixel has the value:
        category_id * divisor + instance_id.
    """

    DIVISOR: T.ClassVar[int] = int(2**15)  # same for all datasets
    IGNORE: T.ClassVar[int] = -1

    @classmethod
    @torch.no_grad()
    def read(cls, path: expath.PathType, **meta_kwds) -> T.Self:
        return load_panoptic(path, **meta_kwds).as_subclass(cls)

    def save(
        self,
        path: expath.PathType,
        *,
        format: PanopticFormat | str | None = None,
        **meta_kwds,
    ) -> None:
        return save_panoptic(self, path, format=format, **meta_kwds)

    @classmethod
    def default(cls, shape: torch.Size, device: torch.device | str = "cpu") -> T.Self:
        return torch.full(
            shape, cls.IGNORE * cls.DIVISOR, dtype=torch.long, device=device
        ).as_subclass(cls)

    @classmethod
    def default_like(cls, other: Tensor) -> T.Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(
            torch.full_like(
                other, fill_value=cls.IGNORE * cls.DIVISOR, dtype=torch.long
            )
        )

    @classmethod
    @T.override
    def wrap_like(
        cls,
        other: T.Self,
        tensor: Tensor,
    ) -> T.Self:
        return tensor.to(dtype=torch.long, non_blocking=True).as_subclass(cls)

    @classmethod
    def from_semantic(cls, semantic: Tensor) -> T.Self:
        """
        Create an instance from a semantic segmentation map by setting the instance IDs to 0.
        """
        if not torch.compiler.is_compiling():
            cls.must_be_semantic_map(semantic)
        return (semantic * cls.DIVISOR).as_subclass(PanopticTensor)

    @classmethod
    def from_parts(cls, semantic: Tensor, instance: Tensor) -> T.Self:
        """
        Create an instance from a semantic segmentation and instance segmentation map by combining them
        using the global ``divisor``.
        """
        if not torch.compiler.is_compiling():
            if semantic.shape != instance.shape:
                msg = f"Expected tensors of the same shape, got {semantic.shape} and {instance.shape}"
                raise ValueError(msg)
            cls.must_be_semantic_map(semantic)
            cls.must_be_instance_map(instance)

        semantic = semantic.to(dtype=torch.long, non_blocking=True)
        instance = instance.to(dtype=torch.long, non_blocking=True)

        ignore_mask = semantic == cls.IGNORE
        panoptic = instance + semantic * cls.DIVISOR
        panoptic[ignore_mask] = cls.IGNORE

        return panoptic.as_subclass(PanopticTensor)

    @classmethod
    def from_combined(cls, encoded_map: Tensor | T.Any, divisor: int) -> T.Self:
        """
        Decompose an encoded map into a semantic segmentation and instance segmentation map, then combine
        again using the global ``divisor``.
        """
        encoded_map = torch.as_tensor(encoded_map)
        assert encoded_map.dtype in (torch.int32, torch.int64), encoded_map.dtype

        sem_id = torch.floor_divide(encoded_map, divisor)
        ins_id = torch.remainder(encoded_map, divisor)
        ins_id = torch.where(encoded_map >= 0, ins_id, 0)

        return PanopticTensor.from_parts(sem_id, ins_id)

    @T.overload
    def to_parts(self: Tensor) -> Tensor: ...

    @T.overload
    def to_parts(self: Tensor, as_tuple=True) -> tuple[Tensor, Tensor]: ...

    def to_parts(
        self: Tensor, as_tuple: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Split the semantic and instance segmentation maps, returing a tensor of size [..., 2].
        The first channel contains the semantic segmentation map, the second channel contains the instance
        id that is NOT UNIQUE for each class.
        """
        ignore_mask = self == PanopticTensor.IGNORE
        sem = torch.floor_divide(self, PanopticTensor.DIVISOR)
        ins = torch.remainder(self, PanopticTensor.DIVISOR)
        ins[ignore_mask] = 0
        if as_tuple:
            return sem.as_subclass(MaskTensor), ins.as_subclass(MaskTensor)
        return torch.stack((sem, ins), dim=-1).as_subclass(MaskTensor)

    @staticmethod
    def parse_label(label: _L) -> tuple[_L, _L]:
        """
        Parse a label into a semantic and instance ID.
        """
        sem_id: _L
        ins_id: _L

        if isinstance(label, int):
            sem_id = label // PanopticTensor.DIVISOR
            ins_id = label % PanopticTensor.DIVISOR if label >= 0 else 0
        else:
            sem_id = torch.floor_divide(label, PanopticTensor.DIVISOR)
            ins_id = torch.remainder(label, PanopticTensor.DIVISOR)
            ins_id = torch.where(label >= 0, ins_id, 0)

        return sem_id, ins_id

    @classmethod
    @T.overload
    def is_void(cls, label: int) -> bool: ...

    @classmethod
    @T.overload
    def is_void(cls, label: Tensor) -> Tensor: ...

    @classmethod
    def is_void(cls, label: Tensor | int) -> Tensor | bool:
        return label < 0

    def get_semantic_map(self: Tensor) -> MaskTensor:
        return torch.floor_divide(self, PanopticTensor.DIVISOR).as_subclass(MaskTensor)

    def get_semantic_masks(self: Tensor) -> T.Iterator[tuple[int, MaskTensor]]:
        """Return a list of masks, one for each semantic class."""
        sem_map = PanopticTensor.get_semantic_map(self)
        uq = torch.unique(sem_map)
        yield from (
            (int(u), (sem_map == u).as_subclass(MaskTensor))
            for u in uq
            if u != PanopticTensor.IGNORE
        )

    def get_semantic_mask(self, class_id: int) -> MaskTensor:
        """Return a mask for the specified semantic class."""
        return (self.get_semantic_map() == class_id).as_subclass(MaskTensor)

    def unique_semantics(self) -> Tensor:
        """Count the number of unique semantic classes."""
        uq = torch.unique(self.get_semantic_map())
        return uq[uq >= 0]

    def get_instance_map(self: Tensor, return_labels: bool = True) -> MaskTensor:
        # old: does not support same sub-id for different classes
        ins_ids = torch.remainder(self, PanopticTensor.DIVISOR)
        if return_labels:
            mask = torch.where((ins_ids > 0) & (self != PanopticTensor.IGNORE), self, 0)
        else:
            mask = torch.where(self != PanopticTensor.IGNORE, ins_ids, 0)
        return mask.as_subclass(MaskTensor)

    def get_instance_masks(self: Tensor) -> T.Iterator[tuple[int, MaskTensor]]:
        """Return a list of masks, one for each instance."""
        ins_map = PanopticTensor.get_instance_map(self)
        uq = torch.unique(ins_map)
        yield from (
            (int(u), (ins_map == u).as_subclass(MaskTensor)) for u in uq if u > 0
        )

    def get_instance_mask(self: Tensor, instance_id: int) -> MaskTensor:
        """Return a mask for the specified instance."""
        return (PanopticTensor.get_instance_map(self) == instance_id).as_subclass(
            MaskTensor
        )

    def get_masks_by_label(
        self: Tensor, *, with_void: bool = False, as_tensor: bool = False
    ) -> T.Iterator[tuple[Tensor, MaskTensor]]:
        """
        Iterate pairs of labels and masks, where each masks corresponds to a unique
        label.
        """
        for pan_id in self.unique():
            if not as_tensor:
                pan_id = pan_id.detach().item()
            if PanopticTensor.is_void(pan_id) and not with_void:
                continue
            yield pan_id, (self == pan_id).as_subclass(MaskTensor)

    def get_masks(
        self: Tensor, **kwargs
    ) -> T.Iterator[tuple[Tensor, Tensor, MaskTensor]]:
        """Return a mask for each semantic class and instance (if any)."""
        for pan_id, mask in PanopticTensor.get_masks_by_label(self, **kwargs):
            sem_id, ins_id = PanopticTensor.parse_label(pan_id)
            yield sem_id, ins_id, mask

    def unique_instances(self: Tensor) -> Tensor:
        """Count the number of unique instances for each semantic class."""
        ins_mask = PanopticTensor.get_instance_map(self) != PanopticTensor.IGNORE
        return torch.unique(self[ins_mask])

    def remove_instances_(self: Tensor, semantic_list: T.Iterable[int]) -> None:
        """Remove instances for the specified semantic classes."""
        sem_map, ins_map = PanopticTensor.to_parts(self, as_tuple=True)

        # Compute candidate map where all pixels that are not in the semantic list are set to -1
        can_map = torch.where(ins_map > 0, sem_map, PanopticTensor.IGNORE)

        # Set all pixels that are not in the semantic list to 0
        for class_ in semantic_list:
            self[can_map == class_] = class_ * PanopticTensor.DIVISOR

    def remove_instances(self: Tensor, semantic_list: T.Iterable[int]) -> Tensor:
        sem_map, _ = PanopticTensor.to_parts(self, as_tuple=True)
        replace = torch.tensor(
            list(semantic_list), dtype=sem_map.dtype, device=sem_map.device
        )
        return self.where(
            ~torch.isin(sem_map, replace), sem_map * PanopticTensor.DIVISOR
        )

    def translate_semantic_(
        self: Tensor,
        translation: T.Mapping[int, int],
        inverse: bool = False,
        *,
        raises: bool = False,
    ) -> Tensor:
        """
        DEPRECATED

        Use `recode_all_` instead.
        """
        sem_map, ins_map = PanopticTensor.to_parts(self, as_tuple=True)
        self.fill_(PanopticTensor.IGNORE)

        if raises:
            self_ids = set(map(int, torch.unique(sem_map.detach().cpu()).tolist()))
            keys_ids = set(map(int, translation.keys()))
            miss_ids = self_ids - keys_ids

            if len(miss_ids) > 0:
                msg = f"Missing {miss_ids} in translations {translation}"
                raise ValueError(msg)

        for (
            old_id,
            new_id,
        ) in translation.items():
            if inverse:
                old_id, new_id = new_id, old_id  # noqa: PLW2901
            mask = sem_map == old_id
            self[mask] = new_id * PanopticTensor.DIVISOR + ins_map[mask]

        return self

    def recode(
        self: Tensor,
        old_cat: int,
        new_cat: int,
        *,
        missing_ok: bool = False,
        inplace=False,
    ) -> Tensor:
        sem_map, ins_map = PanopticTensor.to_parts(self, as_tuple=True)
        mask = sem_map == old_cat

        # Check if there are any pixels for the given semantic category
        if not missing_ok and not mask.any():
            msg = f"No pixels found for semantic category {old_cat}"
            raise ValueError(msg)

        # Inplace variant updates entries at the given mask
        if inplace:
            self[mask] = new_cat * PanopticTensor.DIVISOR + ins_map[mask]
            return self

        # Non-inplace variant uses `Tensor.where(mask, otherwise)` to update entries
        # in the semantic map, and then creates a new panoptic map
        sem_map = sem_map.where(~mask, new_cat)
        return PanopticTensor.from_parts(sem_map, ins_map)

    def recode_(self: Tensor, *translation, **kwargs) -> Tensor:
        return PanopticTensor.recode(self, *translation, inplace=True, **kwargs)

    def recode_all(
        self: Tensor,
        pairs: T.Mapping[int, int],
        *,
        reverse: bool = False,
        fill: int = IGNORE,
        inplace: bool = False,
    ) -> Tensor:
        """
        Recode the semantic classes of the map using the given translation mapping
        """

        # Reverse the mapping of `old_id` -> `new_id` to `new_id` -> `old_id`
        if reverse:
            pairs = {v: k for k, v in pairs.items()}

        # Split the panoptic map into semantic and instance parts
        sem_map, ins_map = PanopticTensor.to_parts(self, as_tuple=True)

        # Convert the fill value to a PanopticID
        fill = fill * PanopticTensor.DIVISOR

        # Result panoptic map
        if inplace:
            result = self
            result.fill_(fill)
        else:
            result = torch.full_like(self, fill, dtype=self.dtype)

        # Recode each semantic class
        for old_cat, new_cat in pairs.items():
            new_id = new_cat * PanopticTensor.DIVISOR
            if inplace:
                mask = sem_map == old_cat
                result[mask] = new_id + ins_map[mask]
            else:
                result = result.where(sem_map != old_cat, new_id + ins_map)
        return result

    def recode_all_(
        self: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""
        In-place version of :meth:`recode_all`.
        """
        return PanopticTensor.recode_all(self, *args, inplace=True, **kwargs)

    def get_nonempty(self: Tensor) -> MaskTensor:
        """Return a new instance with only the non-empty pixels."""
        return self[self >= 0].as_subclass(MaskTensor)

    def to_coco(self) -> tuple[pil_image.Image, list[COCOResultPanopticSegment]]:
        segm = torch.zeros_like(self, dtype=torch.int32)

        segments_info = []

        for i, (sem_id, ins_id, mask) in enumerate(self.get_masks()):
            coco_id = i + 1
            segm[mask] = coco_id
            segments_info.append(
                COCOResultPanopticSegment(id=coco_id, category_id=sem_id)
            )

        img = pil_image.fromarray(segm.numpy().astype("uint8"), mode="L")

        return img, segments_info

    @classmethod
    def must_be_semantic_map(cls, t: Tensor):
        if t.ndim < 2:
            msg = f"Expected 2D tensor, got {t.ndim}D tensor"
        elif t.dtype not in (torch.int32, torch.int64):
            msg = f"Expected int32 or int64 tensor, got {t.dtype}"
        elif (t_min := t.min()) < cls.IGNORE:
            msg = f"Expected non-negative values other than {cls.IGNORE}, got {t_min.item()}"
        elif (t_max := t.max()) >= cls.DIVISOR:
            msg = f"Expected values < {cls.DIVISOR}, got {t_max.item()}"
        else:
            return
        raise ValueError(msg)

    @classmethod
    def is_semantic_map(cls, t: Tensor) -> bool:
        try:
            cls.must_be_semantic_map(t)
        except ValueError:
            return False
        return True

    @classmethod
    def must_be_instance_map(cls, t: Tensor):
        if t.ndim < 2:
            msg = f"Expected 2D tensor, got {t.ndim}D tensor"
        elif t.dtype not in (torch.int32, torch.int64):
            msg = f"Expected int32 or int64 tensor, got {t.dtype}"
        elif (t_min := t.min()) < cls.IGNORE:
            msg = f"Expected non-negative values, got {t_min.item()}"
        elif (t_max := t.max()) >= cls.DIVISOR:
            msg = f"Expected values < {cls.DIVISOR}, got {t_max.item()}"
        else:
            return
        raise ValueError(msg)

    @classmethod
    def is_instance_map(cls, t: Tensor) -> bool:
        try:
            cls.must_be_instance_map(t)
        except ValueError:
            return False
        return True


PanopticTensorLike: T.TypeAlias = PanopticTensor | Tensor


def load_panoptic(input: expath.PathType | Tensor, /, **meta_kwds) -> PanopticTensor:
    """Read a panoptic map from a file."""
    from .helpers import get_kwd, read_pixels

    data_format = get_kwd(meta_kwds, "format", PanopticFormat)

    match data_format:
        case PanopticFormat.SAFETENSORS:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            input = str(expath.locate(input))
            labels = safetensors.load_file(input)["data"].as_subclass(PanopticTensor)
        case PanopticFormat.TORCH:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            with expath.open(input, "rb") as fh:
                labels = torch.load(fh, map_location=torch.device("cpu"))
            labels = labels.as_subclass(PanopticTensor)
            assert labels is not None
            assert isinstance(labels, (PanopticTensor, torch.Tensor)), type(labels)
        case PanopticFormat.CITYSCAPES:
            divisor = 1000
            void_id = 255
            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            img = torchvision.io.decode_image(
                input, mode=torchvision.io.ImageReadMode.RGB
            )
            assert img.ndim == 3, f"Expected 3D tensor, got {img.ndim}D tensor"

            map_ = (
                img[:, :, 0]
                + _BYTE_OFFSET * img[:, :, 1]
                + _BYTE_OFFSET * _BYTE_OFFSET * img[:, :, 2]
            )
            map_ = torch.where(map_ > 0, map_, void_id)
            map_ = torch.where(map_ < divisor, map_ * divisor, map_ + 1)

            labels = PanopticTensor.from_combined(map_, divisor)
        case PanopticFormat.CITYSCAPES_VPS:
            divisor = 1000
            void_id = 255

            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            img = torchvision.io.decode_image(
                input, mode=torchvision.io.ImageReadMode.GRAY
            )
            assert img.ndim == 2, f"Expected 2D tensor, got {img.ndim}D tensor"

            has_instance = img >= divisor

            ids = torch.where(has_instance, (img % divisor) + 1, 0)
            sem = torch.where(has_instance, img // divisor, img)
            sem[sem == void_id] = -1

            labels = PanopticTensor.from_parts(sem, ids)
        case PanopticFormat.CITYSCAPES_DVPS:
            divisor = 1000
            void_id = 32

            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            img = torchvision.io.decode_image(
                input, mode=torchvision.io.ImageReadMode.GRAY
            )
            assert img.ndim == 2, f"Expected 2D tensor, got {img.ndim}D tensor"

            has_instance = img >= divisor

            ids = (img % divisor) + 1
            sem = img // divisor
            sem[sem == void_id] = -1

            labels = PanopticTensor.from_parts(sem, ids)
        case PanopticFormat.KITTI:
            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            img = torchvision.io.decode_image(
                input, mode=torchvision.io.ImageReadMode.RGB
            )
            assert img.ndim == 3, f"Expected 3D tensor, got {img.ndim}D tensor"

            sem = img[:, :, 0]  # R-channel
            ids = torch.add(
                img[:, :, 1] * _BYTE_OFFSET,  # G channel
                img[:, :, 2],  # B channel
            )

            labels = PanopticTensor.from_parts(sem, ids)
        case PanopticFormat.VISTAS:
            divisor = 1000

            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            img = torchvision.io.decode_image(
                input, mode=torchvision.io.ImageReadMode.GRAY_ALPHA
            )
            assert img.ndim == 2, f"Expected 2D tensor, got {img.ndim}D tensor"
            assert img.dtype == torch.int32, img.dtype

            labels = PanopticTensor.from_combined(img, divisor)
        case PanopticFormat.WILD_DASH:
            annotations = get_kwd(meta_kwds, "annotations", list[dict[str, T.Any]])

            divisor = int(1e8)
            void_id = 255

            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            img = torchvision.io.decode_image(
                input, mode=torchvision.io.ImageReadMode.RGB
            )
            assert img.ndim == 3, f"Expected 3D tensor, got {img.ndim}D tensor"
            img = (
                img[:, :, 0].to(torch.long) * _BYTE_OFFSET * _BYTE_OFFSET
                + img[:, :, 1].to(torch.long) * _BYTE_OFFSET
                + img[:, :, 2].to(torch.long)
            )
            sem = torch.full_like(img, void_id, dtype=torch.long)
            for ann in annotations:
                id = ann["id"]
                category_id = ann["category_id"]
                mask = img == id
                sem[mask] = category_id

            ids = torch.full_like(img, 0, dtype=torch.long)  # TODO

            labels = PanopticTensor.from_parts(sem, ids)
            labels.translate_semantic_(
                translation=translations,
            )
        case _:
            msg = f"Could not read labels from {input!r} ({data_format=})"
            raise NotImplementedError(msg)

    assert labels.ndim == 2, f"Expected 2D tensor, got {labels.ndim}D tensor"

    assert labels is not None, (
        f"No labels were read from '{input}' (format: {data_format})"
    )

    if len(meta_kwds) > 0:
        raise TypeError(f"Unexpected keyword arguments: {tuple(meta_kwds.keys())}")

    assert labels.ndim == 2, f"Expected 2D tensor, got {labels.ndim}D tensor"
    return labels


def save_panoptic(
    self: Tensor,
    path: expath.PathType,
    *,
    format: PanopticFormat | str | None = None,
    **kwargs,
) -> None:
    """
    Save the panoptic map to a file.
    """
    if self.ndim != 2:
        msg = f"Expected 2D tensor, got {self.ndim}D tensor ({self.shape})"
        raise ValueError(msg)

    path = expath.locate(path)

    # Check whether the format can be inferred from the filename
    if format is None:
        match path.suffix.lower():
            case ".pth", ".pt":
                format = PanopticFormat.TORCH
            case ".safetensors":
                format = PanopticFormat.SAFETENSORS
            case _:
                msg = f"Could not infer labels format from path: {path}"
                raise ValueError(msg)

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the labels
    match PanopticFormat(format):
        case PanopticFormat.SAFETENSORS:
            assert path.suffix.lower() == ".safetensors", (
                f"Expected SAFETENSORS file, got {path}"
            )
            safetensors.save_file({"data": torch.as_tensor(self)}, path)
        case PanopticFormat.TORCH:
            assert path.suffix.lower() in (
                ".pth",
                ".pt",
            ), f"Expected PT file, got {path}"
            torch.save(torch.as_tensor(self), path)
        case PanopticFormat.PNG_UINT16 | PanopticFormat.PNG_L16:
            assert path.suffix.lower() == ".png", f"Expected PNG file, got {path}"
            sem, ins = PanopticTensor.to_parts(self, as_tuple=True)

            void_id = kwargs.get("void_id", 64)
            max_ins = kwargs.get("divisor", 1000)

            assert not torch.any(sem == void_id), (
                f"Found void ID in semantic map, {void_id=}"
            )
            assert torch.all(ins < max_ins), f"Found instance ID >= {max_ins=}"

            pan = sem.where(sem >= 0, void_id) * max_ins + ins
            write_png_l16(path, pan)
        case PanopticFormat.CITYSCAPES:
            divisor = kwargs.get("divisor", 1000)
            void_id = kwargs.get("void_id", 255)
            img = torch.empty((*self.shape, 3), dtype=torch.uint8)
            img[:, :, 0] = self % _BYTE_OFFSET
            img[:, :, 1] = self // _BYTE_OFFSET
            img[:, :, 2] = self // _BYTE_OFFSET // _BYTE_OFFSET
            img = torch.where(self == void_id * divisor, 0, img)
            img = torch.where(self < divisor, img * divisor, img - 1)

            write_png_rgb(path, img)
        case PanopticFormat.CITYSCAPES_VPS:
            divisor = 1000
            void_id = 255
            sem, ids = PanopticTensor.to_parts(self, as_tuple=True)
            img = torch.where(ids > 0, ids - 1 + sem * divisor, sem)
            img = torch.where(img == void_id, 0, img)

            write_png_l16(path, img)
        case PanopticFormat.CITYSCAPES_DVPS:
            # https://github.com/joe-siyuan-qiao/ViP-DeepLab/tree/master/cityscapes-dvps
            max_ins = 255
            void_id = 32

            sem, ids = PanopticTensor.to_parts(self.detach().cpu(), as_tuple=True)
            assert (sem < void_id).all(), sem.unique().tolist()
            sem = sem.where(sem >= 0, void_id)
            assert (ids < max_ins).all(), ids.unique().tolist()

            img = torch.zeros((*self.shape[-2:], 3), dtype=torch.uint8)
            img[:, :, 0] = sem
            img[:, :, 1] = ids

            write_png_rgb(path, img)
        case PanopticFormat.KITTI:  # KITTI STEP
            img = torch.empty((*self.shape[-2:], 3), dtype=torch.uint8)

            sem, ids = PanopticTensor.to_parts(self, as_tuple=True)
            img[:, :, 0] = sem
            img[:, :, 1] = ids // _BYTE_OFFSET
            img[:, :, 2] = ids % _BYTE_OFFSET

            write_png_rgb(path, img)
        case PanopticFormat.VISTAS:
            divisor = 1000
            void_id = 255
            img = torch.empty((*self.shape, 3), dtype=torch.uint8)
            img[:, :, 0] = self % _BYTE_OFFSET
            img[:, :, 1] = self // _BYTE_OFFSET
            img[:, :, 2] = self // _BYTE_OFFSET // _BYTE_OFFSET

            write_png_rgb(path, img)
        case PanopticFormat.WILD_DASH:
            divisor = int(1e8)
            void_id = 255
            img = torch.empty((*self.shape, 3), dtype=torch.uint8)
            img[:, :, 0] = self // (_BYTE_OFFSET * _BYTE_OFFSET)
            img[:, :, 1] = (self // _BYTE_OFFSET) % _BYTE_OFFSET
            img[:, :, 2] = self % _BYTE_OFFSET
            img = torch.where(self == void_id * divisor, 0, img)

            write_png_rgb(path, img)
        case _:
            msg = f"Could not save labels to {path!r} ({format=})"
            raise NotImplementedError(msg)
