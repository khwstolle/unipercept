from __future__ import annotations
import typing
import enum as E
import torch.fx
import torch

from unipercept.types import Device, DType, Size, Tensor


class GridMode(E.StrEnum):
    INDEX = E.auto()
    NORMALIZED = E.auto()
    PIXEL_CENTER = E.auto()
    PIXEL_NOISE = E.auto()


if typing.TYPE_CHECKING:
    def _arange_like(tensor: Tensor, n: int) -> Tensor: ...
    def _stack_grids(x_g: Tensor, y_g: Tensor) -> Tensor: ...
else:

    @torch.fx.wrap
    def _stack_grids(x_g: Tensor, y_g: Tensor) -> Tensor:
        return torch.stack((x_g, y_g), dim=-1)

    @torch.fx.wrap
    def _arange_like(tensor: Tensor, n: int) -> Tensor:
        return torch.arange(n, dtype=tensor.dtype, device=tensor.device)


@torch.no_grad()
def _grid_from_indices(
    x_i: Tensor,
    y_i: Tensor,
    mode: GridMode | str = "index",
) -> Tensor:
    # Normalize before generating the grid
    if mode == GridMode.NORMALIZED:
        x_i = x_i / (x_i.max() - x_i.min()) * 2 - 1
        y_i = y_i / (y_i.max() - y_i.min()) * 2 - 1

    # Generate a meshgrid of x and y coordinates
    # gx, gy = torch.broadcast_tensors(x.unsqueeze(0), y.unsqueeze(1))
    x_g, y_g = torch.meshgrid(x_i, y_i, indexing="xy")

    # Stack the x and y coordinates along the last dimension
    coords = _stack_grids(x_g, y_g)

    # Apply the specified mode
    match mode:
        case GridMode.NORMALIZED | GridMode.INDEX:
            pass
        case GridMode.PIXEL_CENTER:
            coords = coords + 0.5
        case GridMode.PIXEL_NOISE:
            coords += torch.rand_like(coords)
        case _:
            msg = f"Invalid {mode=}. Choose from {list(GridMode)}"
            raise ValueError(msg)
    return coords


@torch.no_grad()
def generate_coord_grid_like(
    tensor: Tensor,
    mode: GridMode | str = "index",
    *,
    stride: int = 1,
) -> Tensor:
    r"""
    Generate pixel coordinates grid like the input tensor.

    Parameters
    ----------
    tensor:
        Input tensor.
    stride:
        The stride of the input tensor with respect to the canvas size.
    mode:
        Grid mode to use. By default, `GridMode.INDEX`.

    Returns
    -------
    Tensor[H, W, 2]
        Pixel coordinates grid.
    """
    H, W = tensor.shape[-2:]
    x_i = _arange_like(tensor, W * stride)
    y_i = _arange_like(tensor, H * stride)
    return _grid_from_indices(x_i, y_i, mode)


@torch.no_grad()
def generate_coord_grid_as(
    tensor: Tensor,
    canvas_size: Size | tuple[int, int],
    *,
    mode: GridMode | str = "index",
) -> Tensor:
    r"""
    Generate pixel coordinates grid with the same device and dtype as the given tensor.

    Parameters
    ----------
    tensor:
        Input tensor.
    canvas_size:
        Size of the canvas.
    mode:
        Grid mode to use. By default, `GridMode.INDEX`.

    Returns
    -------
    Tensor[H, W, 2]
        Pixel coordinates grid.
    """
    h, w = canvas_size
    x_idx = _arange_like(tensor, w)
    y_idx = _arange_like(tensor, h)
    return _grid_from_indices(x_idx, y_idx, mode)


@torch.no_grad()
def generate_coord_grid(
    canvas_size: torch.Size | tuple[int, int],
    mode: GridMode | str = "index",
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Generate pixel coordinates grid.

    Parameters
    ----------
    canvas_size:
        Size of the canvas.
    device:
        Device to use.
    dtype:
        Data type to use.
    mode:
        Grid mode to use. By default, `GridMode.INDEX`.

    Returns
    -------
    Tensor[H, W, 2]
        Pixel coordinates grid.
    """
    h, w = canvas_size
    if dtype is None:
        dtype = torch.float32
    x_idx = torch.arange(w, device=device, dtype=dtype)
    y_idx = torch.arange(h, device=device, dtype=dtype)
    return _grid_from_indices(x_idx, y_idx, mode)
