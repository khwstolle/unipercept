r"""
Depth tensors
=============

Provides support for depth tensors, e.g. disparity maps, depth maps, etc.
"""

import enum as E
import functools
import io
import json
import typing as T

import expath
import numpy as np
import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
import torchvision.io
from einops import rearrange
from tensordict import MemoryMappedTensor
from torch.nn.functional import interpolate
from torch.types import Device
from torchvision.transforms.v2.functional import register_kernel
from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size
from torchvision.tv_tensors import Mask

from unipercept.tensors.helpers import get_kwd, read_pixels, write_png_l16
from unipercept.tensors.registry import pixel_maps
from unipercept.types import Device, DType, Pathable, Tensor
from unipercept.utils.inspect import locate_object

__all__ = [
    "DepthTensor",
    "DepthMode",
    "DepthFormat",
    "load_depthmap",
    "save_depthmap",
    "downsample_depthmap",
    "resize_depthmap",
    "absolute_to_normalized_depth",
    "normalized_to_absolute_depth",
]

DEFAULT_DEPTH_DTYPE: T.Final = torch.float32


class DepthFormat(E.StrEnum):
    r"""
    Enum class for depth map file formats and their respective mode.
    """

    TIFF = E.auto()
    DEPTH_INT16 = E.auto()
    DISPARITY_INT16 = E.auto()
    TORCH = E.auto()
    SAFETENSORS = E.auto()
    MEMMAP = E.auto()


class DepthMode(E.StrEnum):
    r"""
    Enum class for depth prediction modes.
    """

    ABSOLUTE = E.auto()
    DISPARITY = E.auto()


@pixel_maps.register()
class DepthTensor(Mask):
    r"""
    Represents a two-dimensional depth map as a PyTorch tensor.
    """

    def __new__(
        cls,
        data: T.Any,
        *,
        dtype: DType | None = None,
        device: Device = None,
        requires_grad: bool | None = None,
    ) -> T.Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        return tensor.as_subclass(cls)

    @classmethod
    def default_like(cls, other: Tensor) -> T.Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(torch.full_like(other, fill_value=0, dtype=torch.float32))

    @classmethod
    def default(cls, shape: torch.Size, device: Device = "cpu") -> T.Self:
        """Returns a default instance of this class with the given shape."""
        return cls(torch.zeros(shape, device=device, dtype=torch.float32))  # type: ignore


@torch.no_grad()
def save_depthmap(
    data: Tensor, path: Pathable, format: DepthFormat | str | None = None
) -> None:
    import expath

    data = data.detach().cpu()

    path = expath.locate(path)
    if format is None:
        match path.suffix.lower():
            case ".tiff":
                format = DepthFormat.TIFF
            case ".pth", ".pt":
                format = DepthFormat.TORCH
            case ".safetensors":
                format = DepthFormat.SAFETENSORS
            case ".memmap":
                format = DepthFormat.MEMMAP
            case _:
                msg = f"Could not infer depth format from path: {path}"
                raise ValueError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)

    match DepthFormat(format):
        case DepthFormat.TIFF:
            depth_image = pil_image.fromarray(data.float().squeeze_(0).cpu().numpy())
            if depth_image.mode != "F":
                msg = f"Expected image format 'F'; Got {depth_image.mode!r}"
                raise ValueError(msg)
            assert path.suffix.lower() == ".tiff", path
            depth_image.save(path, format="TIFF")
        case DepthFormat.SAFETENSORS:
            assert path.suffix.lower() == ".safetensors", path
            safetensors.save_file({"data": torch.as_tensor(data)}, path)
        case DepthFormat.TORCH:
            torch.save(torch.as_tensor(data), path)
        case DepthFormat.DEPTH_INT16:
            # depth_image = (self * float(2**8)).numpy().astype(np.uint16)
            # image = pil_image.fromarray(depth_image, mode="I;16")
            # image.save(path)
            assert path.suffix.lower() == ".png", path
            write_png_l16(path, data * float(2**8))

        case _:
            msg = f"Unsupported depth format: {format}"
            raise NotImplementedError(msg)


@torch.no_grad()
def load_depthmap(
    input: expath.PathType | Tensor,
    dtype: torch.dtype = DEFAULT_DEPTH_DTYPE,
    **meta_kwds: T.Any,
) -> DepthTensor:
    if not isinstance(input, torch.Tensor):
        input = str(expath.locate(input))
    # Switch by depth format
    format = get_kwd(meta_kwds, "format", DepthFormat | str)
    match DepthFormat(format):  # type: ignore
        case DepthFormat.TIFF:
            if isinstance(input, torch.Tensor):
                fp_open = functools.partial(io.BytesIO, input.numpy().tobytes())
            else:
                fp_open = functools.partial(expath.open, input, "rb")
            with fp_open() as fp, pil_image.open(fp) as img:
                assert img.mode == "F", img.mode
                dm_np = np.array(img, copy=True)
            dm = torch.from_numpy(dm_np)
            assert dm.ndim == 2, dm.shape
            assert dm.dtype == torch.float32, dm.dtype
            dm = dm.unsqueeze(0)
        case DepthFormat.DEPTH_INT16:
            if not isinstance(input, torch.Tensor):
                input = str(expath.locate(input))
            dm = torchvision.io.read_image(
                input, mode=torchvision.io.ImageReadMode.UNCHANGED
            )
            assert dm.ndim == 3, dm.shape
            assert dm.dtype == torch.uint16, dm.dtype
            dm = dm.to(dtype=dtype) / float(2**8)
        case DepthFormat.DISPARITY_INT16:
            dm = load_depthmap_from_disparitymap(input, **meta_kwds)
        case DepthFormat.SAFETENSORS:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            input = str(expath.locate(input))
            dm = safetensors.load_file(input)["data"]
        case DepthFormat.TORCH:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            input = str(expath.locate(input))
            dm = torch.load(input, map_location="cpu")
        case DepthFormat.MEMMAP:
            if isinstance(input, torch.Tensor):
                msg = "Expected a path to a file, not a tensor."
                raise TypeError(msg)
            input = expath.locate(input)
            tensor_name = input.stem
            path_meta = input.parent / "meta.json"
            with path_meta.open("r") as f:
                meta = json.load(f)[tensor_name]

            dm = MemoryMappedTensor.from_filename(
                filename=input,
                dtype=locate_object(meta["dtype"]),
                shape=torch.Size(meta["shape"]),
            ).contiguous()
        case _:
            msg = f"Unsupported depth format: {format}"
            raise NotImplementedError(msg)

    # TODO: Add angular FOV compensation via metadata
    dm = dm.to(dtype=dtype)
    dm[dm == torch.inf] = 0.0
    dm[dm == torch.nan] = 0.0
    if dm.ndim != 3:
        msg = (
            f"Depth map has {dm.ndim} dimensions, expected  3 (CHW or HWC format). "
            f"Got {dm.shape} instead."
        )
        raise ValueError(msg)

    return dm.as_subclass(DepthTensor)


def load_depthmap_from_disparitymap(
    path: str,
    camera_baseline: float,
    camera_fx: float,
) -> DepthTensor:
    # Get machine epsilon for the given dtype, used to check for invalid values
    eps = torch.finfo(torch.float32).eps

    # Read disparity map
    disp = read_pixels(path, False)
    assert disp.dtype == torch.int32, disp.dtype

    # Convert disparity from 16-bit to float
    disp = disp.to(dtype=torch.float32, copy=False)
    disp -= 1
    disp[disp >= eps] /= 256.0

    # Infer depth using camera parameters and disparity
    valid_mask = disp >= eps

    depth = torch.zeros_like(disp)
    depth[valid_mask] = (camera_baseline * camera_fx) / disp[valid_mask]

    # Set invalid depth values to 0
    depth[depth == torch.inf] = 0
    depth[depth == torch.nan] = 0

    return T.cast(DepthTensor, depth.as_subclass(DepthTensor))


class DepthDownsampleMethod(E.StrEnum):
    MEDIAN = E.auto()
    NEAREST = E.auto()


def downsample_depthmap(
    depth_map: Tensor,
    size: tuple[int, int] | torch.Size,
    method: DepthDownsampleMethod | str = DepthDownsampleMethod.MEDIAN,
) -> Tensor:
    """
    Downsampling of depth maps.

    Parameters
    ----------
    depth_map: Tensor[..., H_old, W_old]
        The input depth map.
    size : tuple[H_new, W_new]
        The target size of the downsampled depth map.
    method : DepthDownsampleMethod or str
        The method used for downsampling. Default: ```DepthDownsampleMethod.MEDIAN``.

    Returns
    -------
    Tensor[..., H, W]
        The downsampled depth map.
    """
    # Check that the shape is actually divisible by the target size, else do NN downsample first to the closest size
    h_old, w_old = depth_map.shape[-2:]
    h_new, w_new = size

    match DepthDownsampleMethod(method):
        case DepthDownsampleMethod.MEDIAN:
            if h_old % h_new != 0 or w_old % w_new != 0:
                h_new = h_old // round(h_old / h_new)
                w_new = w_old // round(w_old / w_new)
                depth_map = interpolate_depth(depth_map, (h_new, w_new))

            # Perform median pooling
            depth_map = rearrange(
                depth_map,
                "... (h1 h2) (w1 w2) -> ... h1 w1 (h2 w2)",
                h1=h_new,
                w1=w_new,
            )
            depth_map[depth_map <= 0] = torch.nan
            depth_map = torch.nanmedian(depth_map, dim=-1).values

            # Set invalid depth values to 0
            depth_map[~torch.isfinite(depth_map)] = 0
        case DepthDownsampleMethod.NEAREST:
            depth_map = interpolate_depth(depth_map, size)

    return depth_map


@register_kernel(functional="resize", tv_tensor_cls=DepthTensor)
def resize_depthmap(
    image: DepthTensor,
    size: list[int],
    interpolation: T.Any = None,  # noqa: U100
    max_size: int | None = None,
    antialias: T.Any = False,  # noqa: U100
    use_rescale: bool = False,
) -> Tensor:
    shape = image.shape
    h_old, w_old = shape[-2:]
    h_new, w_new = _compute_resized_output_size(
        (h_old, w_old), size=size, max_size=max_size
    )

    if h_new <= h_old and w_new <= w_old:
        res = downsample_depthmap(image, (h_new, w_new))
    else:
        res = interpolate_depth(image, (h_new, w_new))

    if use_rescale:
        d_min = image.min()
        d_max = image.max()
        scale = (h_old / h_new + w_old / w_new) / 2

        res = (res * scale).clamp(d_min, d_max)

    return res.as_subclass(DepthTensor)


def interpolate_depth(
    depth_map: Tensor,
    size: tuple[int, int] | torch.Size,
) -> Tensor:
    """
    Quick wrapper for 2D nearest neighbor interpolation, if the input does not have enough
    dimensions, then these are added before interpolation and removed at the end.
    """
    ndim = depth_map.ndim
    while depth_map.ndim < 4:
        depth_map = depth_map.unsqueeze(0)
    depth_map = interpolate(depth_map, size=size, mode="nearest-exact")
    while depth_map.ndim > ndim:
        depth_map = depth_map.squeeze(0)

    return depth_map


def clamp_absolute_depth(
    value: Tensor,
    min_depth: float | Tensor,
    max_depth: float | Tensor,
) -> Tensor:
    """
    Clamp depth values to the specified range. Uses ReLU instead of ``torch.clamp``
    to avoid backpropagation through the gradient of the clamp function, which results
    in numerical inaccuracy for PyTorch version < 2.3.0.

    Parameters
    ----------
    value : Tensor[..., N, H, W]
        The depth tensor.
    min_depth : float or Tensor[..., N]
        The minimum depth value.
    max_depth : float or Tensor[..., N]
        The maximum depth value.

    Returns
    -------
    Tensor[..., N, H, W]
        The clamped depth tensor.
    """
    result = (value - min_depth).relu() + min_depth
    result = max_depth - (max_depth - result).relu()
    return result


def normalized_to_absolute_depth(
    value: Tensor,
    min_depth: float | Tensor,
    max_depth: float | Tensor,
    mode: DepthMode | str = DepthMode.ABSOLUTE,
) -> Tensor:
    """
    Convert depth from normalized values to absolute range.

    Notes
    -----
    The range of values in the input is not strictly enfoced. Users should ensure
    that the input values are within the normalized range.

    Parameters
    ----------
    value : Tensor[..., N, H, W] in (0, 1)
        The normalized depth tensor.
    min_depth : float or Tensor[..., N]
        The minimum depth value.
    max_depth : float or Tensor[..., N]
        The maximum depth value.
    mode : DepthMode
        The depth prediction mode.

    Returns
    -------
    Tensor[..., N, H, W] in (min_depth, max_depth)
        The absolute depth tensor.
    """
    if mode == DepthMode.ABSOLUTE:
        result = value * (max_depth - min_depth) + min_depth
    elif mode == DepthMode.DISPARITY:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * value
        result = 1 / scaled_disp
    else:
        msg = f"Invalid prediction mode: {mode}"
        raise NotImplementedError(msg)
    result = torch.nan_to_num(result, nan=0, posinf=0, neginf=0)
    return result


def absolute_to_normalized_depth(
    value: Tensor,
    min_depth: float | Tensor,
    max_depth: float | Tensor,
    mode: DepthMode | str,
) -> Tensor:
    """
    Convert depth from absolute range to normalized values.

    Notes
    -----
    The range of values in the input is not strictly enfoced. Users should ensure
    that the input values are within the absolute range.

    Parameters
    ----------
    value : Tensor[..., N, H, W] in (min_depth, max_depth)
        The absolute depth tensor.
    min_depth : float or Tensor[..., N]
        The minimum depth value.
    max_depth : float or Tensor[..., N]
        The maximum depth value.
    mode : DepthMode
        The depth prediction mode.

    Returns
    -------
    Tensor[..., N, H, W] in (0, 1)
        The absolute depth tensor.
    """
    if mode == DepthMode.ABSOLUTE:
        result = (value - min_depth) / (max_depth - min_depth)
    elif mode == DepthMode.DISPARITY:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = 1 / value
        result = (scaled_disp - min_disp) / (max_disp - min_disp)
    else:
        msg = f"Invalid prediction mode: {mode}"
        raise NotImplementedError(msg)
    result = torch.nan_to_num(result, nan=0, posinf=0, neginf=0)
    return result
