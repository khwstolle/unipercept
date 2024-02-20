r"""
Filling operations for inpainting and completion tasks.
"""

import math

import torch
import torch.fx
from scipy.ndimage import distance_transform_edt

from unipercept.types import Tensor
from unipercept.utils.check import assert_shape, assert_tensor
from unipercept.vision.coord import generate_coord_grid
from unipercept.vision.filter import filter2d


def _distance_conv(
    input_map: torch.Tensor, kernel_size: int = 3, h: float = 0.35
) -> torch.Tensor:
    r"""
    Very simple distance transform using convolutions. This is a naive implementation
    and may be removed in the future in favor of a more efficient implementation.

    Parameters
    ----------
    input_map: torch.Tensor
        Input tensor of shape (N, C, H, W) or (N, H, W).
    kernel_size: int
        Size of the convolution kernel. Must be an odd number.
    h: float
        Parameter for the exponential function.

    Returns
    -------
    torch.Tensor
        Distance transform of the input tensor.

    Examples
    --------
    >>> input_map = torch.rand(1, 1, 5, 5)
    >>> dist_map = distance_transform(input_map)
    >>> print(dist_map.shape)
    torch.Size([1, 1, 5, 5])
    """
    assert_tensor(input_map)
    assert_shape(input_map, (..., "C", "H", "W"))
    assert kernel_size % 2 == 1, kernel_size
    assert kernel_size > 0, kernel_size

    ndim = input_map.ndim
    if ndim == 3:  # noqa: PLR2004
        input_map = input_map.unsqueeze(1)  # add channel

    *BATCH, C, H, W = input_map.shape
    input_map = input_map.view(-1, C, H, W)

    if not input_map.is_floating_point():
        input_map = input_map.float()
    grid = generate_coord_grid(
        (kernel_size, kernel_size),
        device=input_map.device,
        dtype=input_map.dtype,
    ).unsqueeze(0)

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    dist_map = torch.zeros_like(input_map)
    boundary = input_map.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(math.ceil(max(H, W) / math.floor(kernel_size / 2))):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        dist_map += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    dist_map = dist_map.view(*BATCH, C, H, W)
    if ndim == 3:  # noqa: PLR2004
        dist_map = dist_map.squeeze(1)
    return dist_map


@torch.library.custom_op("unipercept::inpaint2d_euclidean_", mutates_args={"image"})
def inpaint2d_euclidean_(image: torch.Tensor, valid_mask: torch.Tensor) -> None:
    """
    In-place version of :func:`inpaint2d_euclidean`.
    """
    spatial_size = image.shape[-2:]
    image = image.view(-1, *spatial_size)
    mask_np = (  # True for invalid pixels
        (~valid_mask).view(-1, *spatial_size).cpu().numpy()
    )

    for i in range(image.size(0)):
        coords_np = distance_transform_edt(
            mask_np[i], return_distances=False, return_indices=True
        )
        coords = torch.from_numpy(coords_np).to(valid_mask.device)

        image[i] = image[i][tuple(coords)]


@torch.library.custom_op("unipercept::inpaint2d_euclidean", mutates_args=())
def inpaint2d_euclidean(image: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Fill sparse regions in the input tensor using the nearest valid values.
    """
    spatial_size = image.shape[-2:]
    result = image.clone()
    result_flat = result.view(-1, *spatial_size)
    image = image.view(-1, *spatial_size)
    mask_np = (  # True for invalid pixels
        (~valid_mask).reshape(-1, *spatial_size).cpu().numpy()
    )

    for i in range(image.size(0)):
        coords_np = distance_transform_edt(
            mask_np[i], return_distances=False, return_indices=True
        )
        coords = torch.from_numpy(coords_np).to(valid_mask.device)

        result_flat[i] = image[i][tuple(coords)]

    return result


@inpaint2d_euclidean.register_fake  # type: ignore[misc]
def _(image: Tensor, valid_mask: Tensor) -> Tensor:  # noqa: ARG001
    return image
