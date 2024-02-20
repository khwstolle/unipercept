r"""
Filter
======

The :mod:`unipercept.vision.filter` module includes functions and classes to apply
filters to images.

Notes
-----
Parts of this module are generated via `GitHub Copilot<https://copilot.github.com/>`_
and require manual review and testing. Contributions are welcome.
"""

import enum as E
import math

import torch
import typing_extensions as TX
from torch import nn, stack, tensor, where, zeros, zeros_like

from unipercept.types import Device, DType, Tensor
from unipercept.utils.check import assert_shape, assert_tensor


class FilterBorder(E.StrEnum):
    r"""Border types for padding."""

    CONSTANT = E.auto()
    REFLECT = E.auto()
    REPLICATE = E.auto()
    CIRCULAR = E.auto()


class FilterPadding(E.StrEnum):
    r"""Padding types."""

    VALID = E.auto()
    SAME = E.auto()


class FilterKind(E.StrEnum):
    r"""Filter kind."""

    CONVOLUTION = "conv"
    CORRELATION = "corr"


def _compute_padding(kernel_size: list[int]) -> list[int]:
    if len(kernel_size) < 2:  # noqa: PLR2004
        msg = f"Kernel size must be at least 2, got {len(kernel_size)}."
        raise ValueError(msg)
    ks_unity = [k - 1 for k in kernel_size]
    res = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        z = ks_unity[-(i + 1)]
        p_x = z // 2
        p_y = z - p_x

        res[2 * i + 0] = p_x
        res[2 * i + 1] = p_y

    return res


def filter2d(
    input: Tensor,
    kernel: Tensor,
    border_type: FilterBorder | str = "reflect",
    normalized: bool = False,
    padding: FilterPadding | str = "same",
    kind: FilterKind | str = "corr",
) -> Tensor:
    r"""
    Apply a 2D filter to an input tensor.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel: Tensor
        The kernel tensor with shape :math:`(B, H, W)`.
    border_type: FilterBorder or str, optional
        The border type for padding. Default is "reflect".
    normalized: bool, optional
        Whether to normalize the kernel. Default is False.
    padding: FilterPadding or str, optional
        The padding type. Default is "same".
    kind: FilterKind or str, optional
        The filter kind. Default is "corr".

    Returns
    -------
    Tensor
        The filtered tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import filter2d, get_box_kernel2d
    >>> input = torch.rand(1, 1, 5, 5)
    >>> kernel = get_box_kernel2d((3, 3))
    >>> output = filter2d(input, kernel)
    >>> print(output.shape)
    torch.Size([1, 1, 5, 5])
    """
    assert_tensor(input)
    assert_shape(input, ("B", "C", "H", "W"))
    assert_tensor(kernel)
    assert_shape(kernel, ("B", "H", "W"))

    # prepare kernel
    b, c, h, w = input.shape
    if str(kind).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(
            device=input.device, dtype=input.dtype
        )
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: list[int] = _compute_padding([height, width])
        input = nn.functional.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    input: Tensor,
    kernel_x: Tensor,
    kernel_y: Tensor,
    border_type: FilterBorder | str = "reflect",
    normalized: bool = False,
    padding: FilterPadding | str = "same",
) -> Tensor:
    r"""
    Apply a separable 2D filter to an input tensor.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_x: Tensor
        The kernel tensor for the x-axis with shape :math:`(B, H, 1)`.
    kernel_y: Tensor
        The kernel tensor for the y-axis with shape :math:`(B, 1, W)`.
    border_type: FilterBorder or str, optional
        The border type for padding. Default is "reflect".
    normalized: bool, optional
        Whether to normalize the kernel. Default is False.
    padding: FilterPadding or str, optional
        The padding type. Default is "same".

    Returns
    -------
    Tensor
        The filtered tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import filter2d_separable, get_box_kernel1d
    >>> input = torch.rand(1, 1, 5, 5)
    >>> kernel_x = get_box_kernel1d(3)
    >>> kernel_y = get_box_kernel1d(3)
    >>> output = filter2d_separable(input, kernel_x, kernel_y)
    >>> print(output.shape)
    torch.Size([1, 1, 5, 5])
    """
    out_x = filter2d(input, kernel_x[..., None, :], border_type, normalized, padding)
    out = filter2d(out_x, kernel_y[..., None], border_type, normalized, padding)
    return out


def filter3d(
    input: Tensor,
    kernel: Tensor,
    border_type: FilterBorder | str = "replicate",
    normalized: bool = False,
) -> Tensor:
    r"""
    Apply a 3D filter to an input tensor.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, D, H, W)`.
    kernel: Tensor
        The kernel tensor with shape :math:`(B, D, H, W)`.
    border_type: FilterBorder or str, optional
        The border type for padding. Default is "replicate".
    normalized: bool, optional
        Whether to normalize the kernel. Default is False.

    Returns
    -------
    Tensor
        The filtered tensor with shape :math:`(B, C, D, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import filter3d, get_box_kernel2d
    >>> input = torch.rand(1, 1, 3, 5, 5)
    >>> kernel = get_box_kernel2d((3, 3))
    >>> output = filter3d(input, kernel)
    >>> print(output.shape)
    torch.Size([1, 1, 3, 5, 5])
    """
    assert_tensor(input)
    assert_shape(input, ("B", "C", "D", "H", "W"))
    assert_tensor(kernel)
    assert_shape(kernel, ("B", "D", "H", "W"))

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(
            tmp_kernel
        )

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = nn.functional.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(
        -1,
        tmp_kernel.size(0),
        input_pad.size(-3),
        input_pad.size(-2),
        input_pad.size(-1),
    )

    # convolve the tensor with the kernel.
    output = nn.functional.conv3d(
        input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    return output.view(b, c, d, h, w)


class BlurPool2D(nn.Module):
    r"""
    Compute blur (i.e. anti-aliasing) and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Parameters
    ----------
    kernel_size:
        The kernel size.
    stride:
        The stride value.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(N, C, H_\text{out}, W_\text{out})`, where

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{kernel\_size//2}[0] -
            \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{kernel\_size//2}[1] -
            \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import BlurPool2D
    >>> input = torch.rand(1, 1, 5, 5)
    >>> blur_pool = BlurPool2D(kernel_size=3, stride=2)
    >>> output = blur_pool(input)
    >>> print(output.shape)
    torch.Size([1, 1, 2, 2])
    """

    def __init__(self, kernel_size: tuple[int, int] | int, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        self.kernel = torch.as_tensor(
            self.kernel, device=input.device, dtype=input.dtype
        )
        return _blur_pool_by_kernel2d(
            input, self.kernel.repeat((input.shape[1], 1, 1, 1)), self.stride
        )


class MaxBlurPool2D(nn.Module):
    r"""
    Compute pools and blurs and downsample a given feature map.

    Equivalent to ```nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))```

    See :cite:`zhang2019shiftinvar` for more details.

    Parameters
    ----------
    kernel_size:
        The kernel size.
    stride:
        The stride value.
    max_pool_size:
        The maximum pool size.
    ceil_mode:
        Boolean that should be true to match output size of conv2d with same kernel
        size.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H / stride, W / stride)`

    Returns
    -------
    Tensor:
        The resulting tensor.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import MaxBlurPool2D
    >>> input = torch.rand(1, 1, 5, 5)
    >>> max_blur_pool = MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False)
    >>> output = max_blur_pool(input)
    >>> print(output.shape)
    torch.Size([1, 1, 2, 2])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        stride: int = 2,
        max_pool_size: int = 2,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        self.kernel = torch.as_tensor(
            self.kernel, device=input.device, dtype=input.dtype
        )
        return _max_blur_pool_by_kernel2d(
            input,
            self.kernel.repeat((input.size(1), 1, 1, 1)),
            self.stride,
            self.max_pool_size,
            self.ceil_mode,
        )


class EdgeAwareBlurPool2D(nn.Module):
    r"""
    Compute edge-aware blur and downsample a given feature map.

    Parameters
    ----------
    kernel_size:
        The kernel size.
    edge_threshold:
        The edge threshold value.
    edge_dilation_kernel_size:
        The edge dilation kernel size.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import EdgeAwareBlurPool2D
    >>> input = torch.rand(1, 1, 5, 5)
    >>> edge_aware_blur_pool = EdgeAwareBlurPool2D(kernel_size=3, edge_threshold=1.25, edge_dilation_kernel_size=3)
    >>> output = edge_aware_blur_pool(input)
    >>> print(output.shape)
    torch.Size([1, 1, 5, 5])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        edge_threshold: float = 1.25,
        edge_dilation_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.edge_threshold = edge_threshold
        self.edge_dilation_kernel_size = edge_dilation_kernel_size

    @TX.override
    def forward(self, input: Tensor, eps: float = 1e-6) -> Tensor:
        return edge_aware_blur_pool2d(
            input,
            self.kernel_size,
            self.edge_threshold,
            self.edge_dilation_kernel_size,
            eps,
        )


def blur_pool2d(
    input: Tensor, kernel_size: tuple[int, int] | int, stride: int = 2
) -> Tensor:
    r"""
    Compute blur (i.e. anti-aliasing) and downsample a given feature map.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The kernel size.
    stride: int, optional
        The stride value. Default is 2.

    Returns
    -------
    Tensor
        The blurred and downsampled tensor with shape :math:`(B, C, H_{out}, W_{out})`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import blur_pool2d
    >>> input = torch.eye(5)[None, None]
    >>> output = blur_pool2d(input, 3)
    >>> print(output)
    tensor([[[[0.3125, 0.0625, 0.0000],
              [0.0625, 0.3750, 0.0625],
              [0.0000, 0.0625, 0.3125]]]])
    """
    kernel = get_pascal_kernel_2d(
        kernel_size, norm=True, device=input.device, dtype=input.dtype
    ).repeat((input.size(1), 1, 1, 1))
    return _blur_pool_by_kernel2d(input, kernel, stride)


def max_blur_pool2d(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    stride: int = 2,
    max_pool_size: int = 2,
    ceil_mode: bool = False,
) -> Tensor:
    r"""
    Compute max pooling, blur (i.e. anti-aliasing), and downsample a given feature map.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The kernel size.
    stride: int, optional
        The stride value. Default is 2.
    max_pool_size: int, optional
        The maximum pool size. Default is 2.
    ceil_mode: bool, optional
        Whether to use ceil mode for max pooling. Default is False.

    Returns
    -------
    Tensor
        The max pooled, blurred, and downsampled tensor with shape :math:`(B, C, H_{out}, W_{out})`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import max_blur_pool2d
    >>> input = torch.eye(5)[None, None]
    >>> output = max_blur_pool2d(input, 3)
    >>> print(output)
    tensor([[[[0.5625, 0.3125],
              [0.3125, 0.8750]]]])
    """
    assert_shape(input, ("B", "C", "H", "W"))

    kernel = get_pascal_kernel_2d(
        kernel_size, norm=True, device=input.device, dtype=input.dtype
    ).repeat((input.shape[1], 1, 1, 1))
    return _max_blur_pool_by_kernel2d(input, kernel, stride, max_pool_size, ceil_mode)


def _blur_pool_by_kernel2d(input: Tensor, kernel: Tensor, stride: int) -> Tensor:
    r"""
    Compute blur_pool by a given :math:`CxC_{out}xNxN` kernel.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel: Tensor
        The kernel tensor with shape :math:`(C, C_{out}, N, N)`.
    stride: int
        The stride value.

    Returns
    -------
    Tensor
        The blurred and downsampled tensor with shape :math:`(B, C, H_{out}, W_{out})`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _blur_pool_by_kernel2d, get_pascal_kernel_2d
    >>> input = torch.eye(5)[None, None]
    >>> kernel = get_pascal_kernel_2d(3).repeat((1, 1, 1, 1))
    >>> output = _blur_pool_by_kernel2d(input, kernel, 2)
    >>> print(output)
    tensor([[[[0.3125, 0.0625, 0.0000],
              [0.0625, 0.3750, 0.0625],
              [0.0000, 0.0625, 0.3125]]]])
    """
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return nn.functional.conv2d(
        input, kernel, padding=padding, stride=stride, groups=input.shape[1]
    )


def _max_blur_pool_by_kernel2d(
    input: Tensor, kernel: Tensor, stride: int, max_pool_size: int, ceil_mode: bool
) -> Tensor:
    r"""
    Compute max pooling, blur (i.e. anti-aliasing), and downsample a given feature map by a given :math:`CxC_{out}xNxN` kernel.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel: Tensor
        The kernel tensor with shape :math:`(C, C_{out}, N, N)`.
    stride: int
        The stride value.
    max_pool_size: int
        The maximum pool size.
    ceil_mode: bool
        Whether to use ceil mode for max pooling.

    Returns
    -------
    Tensor
        The max pooled, blurred, and downsampled tensor with shape :math:`(B, C, H_{out}, W_{out})`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _max_blur_pool_by_kernel2d, get_pascal_kernel_2d
    >>> input = torch.eye(5)[None, None]
    >>> kernel = get_pascal_kernel_2d(3).repeat((1, 1, 1, 1))
    >>> output = _max_blur_pool_by_kernel2d(input, kernel, 2, 2, False)
    >>> print(output)
    tensor([[[[0.5625, 0.3125],
              [0.3125, 0.8750]]]])
    """
    input = nn.functional.max_pool2d(
        input, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode
    )
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return nn.functional.conv2d(
        input, kernel, padding=padding, stride=stride, groups=input.size(1)
    )


def edge_aware_blur_pool2d(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    edge_threshold: float = 1.25,
    edge_dilation_kernel_size: int = 3,
    eps: float = 1e-6,
) -> Tensor:
    r"""
    Blur the input tensor while maintaining its edges.

    Parameters
    ----------
    input: Tensor
        Input image to blur with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        Kernel size for max pooling.
    edge_threshold: float
        Positive threshold for the edge detection.
    edge_dilation_kernel_size: int
        Kernel size for dilating edges.
    eps: float
        Epsilon value added before taking the log-2 for numerical stability.

    Returns
    -------
    Tensor
        The blurred tensor of shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import edge_aware_blur_pool2d
    >>> input = torch.rand(1, 1, 5, 5)
    >>> output = edge_aware_blur_pool2d(input, kernel_size=3, edge_threshold=1.25, edge_dilation_kernel_size=3)
    >>> print(output.shape)
    torch.Size([1, 1, 5, 5])
    """
    assert_shape(input, ("B", "C", "H", "W"))

    # Pad to avoid artifacts near physical edges
    input = nn.functional.pad(input, (2, 2, 2, 2), mode="reflect")

    blurred_input = blur_pool2d(input, kernel_size=kernel_size, stride=1)

    # Calculate the edges (add eps to avoid taking the log of 0)
    log_input, log_thresh = (input + eps).log2(), (tensor(edge_threshold)).log2()
    edges_x = log_input[..., :, 4:] - log_input[..., :, :-4]
    edges_y = log_input[..., 4:, :] - log_input[..., :-4, :]
    edges_x, edges_y = (
        edges_x.mean(dim=-3, keepdim=True),
        edges_y.mean(dim=-3, keepdim=True),
    )
    edges_x_mask, edges_y_mask = (
        edges_x.abs() > log_thresh.to(edges_x),
        edges_y.abs() > log_thresh.to(edges_y),
    )
    edges_xy_mask = (edges_x_mask[..., 2:-2, :] + edges_y_mask[..., :, 2:-2]).type_as(
        input
    )

    # Dilate the content edges to have a soft mask of edges
    dilated_edges = nn.functional.max_pool3d(
        edges_xy_mask, edge_dilation_kernel_size, 1, edge_dilation_kernel_size // 2
    )

    # Slice the padded regions
    input = input[..., 2:-2, 2:-2]
    blurred_input = blurred_input[..., 2:-2, 2:-2]

    # Fuse the input image on edges and blurry input everywhere else
    return dilated_edges * input + (1.0 - dilated_edges) * blurred_input


def box_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    border_type: FilterBorder | str = "reflect",
    separable: bool = False,
) -> Tensor:
    r"""
    Applies a box filter to the input tensor.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Parameters
    ----------
    input: Tensor
        The image to blur with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The blurring kernel size.
    border_type: FilterBorder or str, optional
        The padding mode to be applied before convolving. Default is "reflect".
    separable: bool, optional
        Run as composition of two 1D-convolutions. Default is False.

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import box_blur
    >>> input = torch.rand(2, 4, 5, 7)
    >>> output = box_blur(input, (3, 3))
    >>> print(output.shape)
    torch.Size([2, 4, 5, 7])
    """
    assert_tensor(input)

    if separable:
        ky, kx = _unpack_2d_ks(kernel_size)
        kernel_y = get_box_kernel1d(ky, device=input.device, dtype=input.dtype)
        kernel_x = get_box_kernel1d(kx, device=input.device, dtype=input.dtype)
        out = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel = get_box_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
        out = filter2d(input, kernel, border_type)

    return out


class BoxBlur(nn.Module):
    r"""
    Apply a box filter to the input tensor.

    Parameters
    ----------
    kernel_size:
        The blurring kernel size.
    border_type:
        The padding mode to be applied before convolving.
    separable:
        Run as composition of two 1D-convolutions.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import BoxBlur
    >>> input = torch.rand(2, 4, 5, 7)
    >>> blur = BoxBlur((3, 3))
    >>> output = blur(input)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 7])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        border_type: FilterBorder | str = "reflect",
        separable: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.separable = separable

        if separable:
            ky, kx = _unpack_2d_ks(self.kernel_size)
            self.register_buffer("kernel_y", get_box_kernel1d(ky))
            self.register_buffer("kernel_x", get_box_kernel1d(kx))
            self.kernel_y: Tensor
            self.kernel_x: Tensor
        else:
            self.register_buffer("kernel", get_box_kernel2d(kernel_size))
            self.kernel: Tensor

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        assert_tensor(input)
        if self.separable:
            return filter2d_separable(
                input, self.kernel_x, self.kernel_y, self.border_type
            )
        return filter2d(input, self.kernel, self.border_type)


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


def _unpack_3d_ks(kernel_size: tuple[int, int, int] | int) -> tuple[int, int, int]:
    if isinstance(kernel_size, int):
        kz = ky = kx = kernel_size
    else:
        kz, ky, kx = kernel_size

    kz = int(kz)
    ky = int(ky)
    kx = int(kx)

    return (kz, ky, kx)


def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""
    Normalize a 2D kernel.

    Parameters
    ----------
    input: Tensor
        The input kernel tensor with shape :math:`(..., H, W)`.

    Returns
    -------
    Tensor
        The normalized kernel tensor with shape :math:`(..., H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import normalize_kernel2d
    >>> kernel = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> normalized_kernel = normalize_kernel2d(kernel)
    >>> print(normalized_kernel)
    tensor([[0.1000, 0.2000],
            [0.3000, 0.4000]])
    """
    assert_shape(input, (..., "H", "W"))

    norm = input.abs().sum(dim=-1).sum(dim=-1)

    return input / (norm[..., None, None])


def gaussian(
    window_size: int,
    sigma: Tensor | float,
    *,
    mean: Tensor | float | None = None,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Compute the Gaussian values based on the window and sigma values.

    Parameters
    ----------
    window_size: int
        The size which drives the filter amount.
    sigma: Tensor or float
        Gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`.
    mean: Tensor or float, optional
        Gaussian mean. Default is ``window_size // 2``. If a tensor, should be in a shape :math:`(B, 1)`.
    device: Device, optional
        Device. If ``None``, then it will be inferred from sigma.
    dtype: DType, optional
        DType. If ``None``, then it will be inferred from sigma.

    Returns
    -------
    Tensor
        A tensor with shape :math:`(B, \text{kernel_size})`, with Gaussian values.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import gaussian
    >>> window_size = 5
    >>> sigma = torch.tensor([[1.0]])
    >>> gauss = gaussian(window_size, sigma)
    >>> print(gauss)
    tensor([[0.1353, 0.6065, 1.0000, 0.6065, 0.1353]])
    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)
        batch_size = 1
    else:
        assert_tensor(sigma)
        assert_shape(sigma, ("B", "1"))
        batch_size = sigma.shape[0]
        device = device if device is not None else sigma.device
        dtype = dtype if dtype is not None else sigma.dtype

    mean = float(window_size // 2) if mean is None else mean
    if isinstance(mean, float):
        mean = torch.tensor([[mean]], device=device, dtype=dtype)
    else:
        assert_tensor(mean)
        assert_shape(mean, ("B", "1"))

    x = (torch.arange(window_size, device=device, dtype=dtype) - mean).expand(
        batch_size, -1
    )

    if window_size % 2 == 0:
        x = x + 0.5

    k = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return k / k.sum(-1, keepdim=True)


def gaussian_discrete_erf(
    window_size: int,
    sigma: Tensor | float,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Discrete Gaussian by interpolating the error function.

    Parameters
    ----------
    window_size: int
        The size which drives the filter amount.
    sigma: Tensor or float
        Gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`.
    device: Device, optional
        This value will be used if sigma is a float. Device desired to compute.
    dtype: DType, optional
        This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
    Tensor
        A tensor with shape :math:`(B, \text{kernel_size})`, with discrete Gaussian values computed by approximation of the error function.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import gaussian_discrete_erf
    >>> window_size = 5
    >>> sigma = torch.tensor([[1.0]])
    >>> gauss = gaussian_discrete_erf(window_size, sigma)
    >>> print(gauss)
    tensor([[0.1353, 0.6065, 1.0000, 0.6065, 0.1353]])
    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    assert_shape(sigma, ("B", "1"))
    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    t = 0.70710678 / sigma.abs()
    # t = torch.tensor(2, device=sigma.device, dtype=sigma.dtype).sqrt() / (sigma.abs() * 2)

    k = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    k = k.clamp(min=0)

    return k / k.sum(-1, keepdim=True)


def _modified_bessel_0(x: Tensor) -> Tensor:
    r"""
    Compute the modified Bessel function of the first kind of order 0.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The modified Bessel function of the first kind of order 0.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _modified_bessel_0
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel = _modified_bessel_0(x)
    >>> print(bessel)
    tensor([1.2661, 2.2796, 4.8808])
    """
    ax = torch.abs(x)
    out = zeros_like(x)
    idx_a = ax < 3.75  # noqa: PLR2004
    if idx_a.any():
        y = (x[idx_a] / 3.75) * (x[idx_a] / 3.75)
        out[idx_a] = 1.0 + y * (
            3.5156229
            + y
            * (
                3.0899424
                + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))
            )
        )

    idx_b = ~idx_a
    if idx_b.any():
        y = 3.75 / ax[idx_b]
        ans = 0.916281e-2 + y * (
            -0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))
        )
        coef = 0.39894228 + y * (
            0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans))
        )
        out[idx_b] = (ax[idx_b].exp() / ax[idx_b].sqrt()) * coef

    return out


def _modified_bessel_1(x: Tensor) -> Tensor:
    r"""
    Compute the modified Bessel function of the first kind of order 1.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The modified Bessel function of the first kind of order 1.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _modified_bessel_1
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel = _modified_bessel_1(x)
    >>> print(bessel)
    tensor([0.5652, 1.5906, 3.9534])
    """
    ax = torch.abs(x)
    out = zeros_like(x)
    idx_a = ax < 3.75  # noqa: PLR2004
    if idx_a.any():
        y = (x[idx_a] / 3.75) * (x[idx_a] / 3.75)
        ans = 0.51498869 + y * (
            0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))
        )
        out[idx_a] = ax[idx_a] * (0.5 + y * (0.87890594 + y * ans))

    idx_b = ~idx_a
    if idx_b.any():
        y = 3.75 / ax[idx_b]
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
        ans = 0.39894228 + y * (
            -0.3988024e-1
            + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans)))
        )
        ans = ans * ax[idx_b].exp() / ax[idx_b].sqrt()
        out[idx_b] = where(x[idx_b] < 0, -ans, ans)

    return out


def _modified_bessel_i(n: int, x: Tensor) -> Tensor:
    r"""
    Compute the modified Bessel function of the first kind of order n.

    Parameters
    ----------
    n: int
        The order of the Bessel function.
    x: Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The modified Bessel function of the first kind of order n.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _modified_bessel_i
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> bessel = _modified_bessel_i(2, x)
    >>> print(bessel)
    tensor([0.5652, 1.5906, 3.9534])
    """
    if (x == 0.0).all():
        return x

    batch_size = x.shape[0]

    tox = 2.0 / x.abs()
    ans = zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    bip = zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    bi = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

    m = int(2 * (n + int(sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        idx = bi.abs() > 1.0e10

        if idx.any():
            ans[idx] = ans[idx] * 1.0e-10
            bi[idx] = bi[idx] * 1.0e-10
            bip[idx] = bip[idx] * 1.0e-10

        if j == n:
            ans = bip

    out = ans * _modified_bessel_0(x) / bi

    if (n % 2) == 1:
        out = where(x < 0.0, -out, out)

    # TODO: skip the previous computation for x == 0, instead of forcing here
    out = where(x == 0.0, x, out)

    return out


def gaussian_discrete(
    window_size: int,
    sigma: Tensor | float,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Discrete Gaussian kernel based on the modified Bessel functions.

    Parameters
    ----------
    window_size: int
        The size which drives the filter amount.
    sigma: Tensor or float
        Gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`.
    device: Device, optional
        This value will be used if sigma is a float. Device desired to compute.
    dtype: DType, optional
        This value will be used if sigma is a float. DType desired for compute.

    Returns
    -------
    Tensor
        A tensor with shape :math:`(B, \text{kernel_size})`, with discrete Gaussian values computed by modified Bessel function.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import gaussian_discrete
    >>> window_size = 5
    >>> sigma = torch.tensor([[1.0]])
    >>> gauss = gaussian_discrete(window_size, sigma)
    >>> print(gauss)
    tensor([[0.1353, 0.6065, 1.0000, 0.6065, 0.1353]])
    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    assert_shape(sigma, ("B", "1"))

    sigma2 = sigma * sigma
    tail = int(window_size // 2) + 1
    bessels = [
        _modified_bessel_0(sigma2),
        _modified_bessel_1(sigma2),
        *(_modified_bessel_i(k, sigma2) for k in range(2, tail)),
    ]
    out = torch.cat(bessels[:0:-1] + bessels, -1) * sigma2.exp()

    return out / out.sum(-1, keepdim=True)


def laplacian_1d(
    window_size: int, *, device: Device | None = None, dtype: DType = torch.float32
) -> Tensor:
    r"""
    Compute the Laplacian of a 1D kernel.

    Parameters
    ----------
    window_size: int
        The size of the kernel.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        The Laplacian of the 1D kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import laplacian_1d
    >>> window_size = 5
    >>> laplacian = laplacian_1d(window_size)
    >>> print(laplacian)
    tensor([ 1.,  1., -4.,  1.,  1.])
    """
    filter_1d = torch.ones(window_size, device=device, dtype=dtype)
    middle = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_box_kernel1d(
    kernel_size: int, *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 1D box filter.

    Parameters
    ----------
    kernel_size: int
        The size of the kernel.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A tensor with shape :math:`(1, \text{kernel_size})`, filled with the value :math:`\frac{1}{\text{kernel_size}}`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_box_kernel1d
    >>> kernel_size = 3
    >>> box_kernel = get_box_kernel1d(kernel_size)
    >>> print(box_kernel)
    tensor([[0.3333, 0.3333, 0.3333]])
    """
    scale = torch.tensor(1.0 / kernel_size, device=device, dtype=dtype)
    return scale.expand(1, kernel_size)


def get_box_kernel2d(
    kernel_size: tuple[int, int] | int,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 2D box filter.

    Parameters
    ----------
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A tensor with shape :math:`(1, \text{kernel_size}[0], \text{kernel_size}[1])`, filled with the value :math:`\frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]}`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_box_kernel2d
    >>> kernel_size = (3, 3)
    >>> box_kernel = get_box_kernel2d(kernel_size)
    >>> print(box_kernel)
    tensor([[[0.1111, 0.1111, 0.1111],
             [0.1111, 0.1111, 0.1111],
             [0.1111, 0.1111, 0.1111]]])
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    scale = torch.tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(1, ky, kx)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int,
    *,
    device: Device | None = None,
    dtype: DType = torch.float32,
) -> Tensor:
    r"""
    Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.

    Parameters
    ----------
    window_size: tuple[int, int] | int
        The size of the window.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A binary kernel with shape :math:`(\text{window\_size}[0] * \text{window\_size}[1], 1, \text{window\_size}[0], \text{window\_size}[1])`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_binary_kernel2d
    >>> window_size = (3, 3)
    >>> binary_kernel = get_binary_kernel2d(window_size)
    >>> print(binary_kernel.shape)
    torch.Size([9, 1, 3, 3])
    """
    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def get_sobel_kernel_3x3(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 3x3 Sobel kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 3x3 Sobel kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_sobel_kernel_3x3
    >>> sobel_kernel = get_sobel_kernel_3x3()
    >>> print(sobel_kernel)
    tensor([[-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.]])
    """
    return torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def get_sobel_kernel_5x5_2nd_order(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 5x5 second-order Sobel kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 5x5 second-order Sobel kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_sobel_kernel_5x5_2nd_order
    >>> sobel_kernel = get_sobel_kernel_5x5_2nd_order()
    >>> print(sobel_kernel)
    tensor([[-1.,  0.,  2.,  0., -1.],
            [-4.,  0.,  8.,  0., -4.],
            [-6.,  0., 12.,  0., -6.],
            [-4.,  0.,  8.,  0., -4.],
            [-1.,  0.,  2.,  0., -1.]])
    """
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _get_sobel_kernel_5x5_2nd_order_xy(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 5x5 second-order Sobel kernel for the xy direction.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 5x5 second-order Sobel kernel for the xy direction.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _get_sobel_kernel_5x5_2nd_order_xy
    >>> sobel_kernel = _get_sobel_kernel_5x5_2nd_order_xy()
    >>> print(sobel_kernel)
    tensor([[-1., -2.,  0.,  2.,  1.],
            [-2., -4.,  0.,  4.,  2.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 2.,  4.,  0., -4., -2.],
            [ 1.,  2.,  0., -2., -1.]])
    """
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel_3x3(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 3x3 first-order difference kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 3x3 first-order difference kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_diff_kernel_3x3
    >>> diff_kernel = get_diff_kernel_3x3()
    >>> print(diff_kernel)
    tensor([[ 0.,  0.,  0.],
            [-1.,  0.,  1.],
            [ 0.,  0.,  0.]])
    """
    return torch.tensor(
        [[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def get_diff_kernel3d(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 3x3x3 first-order difference kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 3x3x3 first-order difference kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_diff_kernel3d
    >>> diff_kernel = get_diff_kernel3d()
    >>> print(diff_kernel)
    tensor([[[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [-0.5,  0.,  0.5],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0., -0.5,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.5,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0., -0.5,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.5,  0.],
              [ 0.,  0.,  0.]]]])
    """
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_diff_kernel3d_2nd_order(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 3x3x3 second-order difference kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 3x3x3 second-order difference kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_diff_kernel3d_2nd_order
    >>> diff_kernel = get_diff_kernel3d_2nd_order()
    >>> print(diff_kernel)
    tensor([[[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 1., -2.,  1.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  1.,  0.],
              [ 0., -2.,  0.],
              [ 0.,  1.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0., -2.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 1.,  0., -1.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [-1.,  0.,  1.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  1.,  0.],
              [ 0.,  0.,  0.],
              [ 0., -1.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0., -1.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  1.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 1.,  0., -1.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [-1.,  0.,  1.],
              [ 0.,  0.,  0.]]]])
    """
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel[:, None, ...]


def get_sobel_kernel2d(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 2D Sobel kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D Sobel kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_sobel_kernel2d
    >>> sobel_kernel = get_sobel_kernel2d()
    >>> print(sobel_kernel)
    tensor([[[-1.,  0.,  1.],
             [-2.,  0.,  2.],
             [-1.,  0.,  1.]],
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]])
    """
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])


def get_diff_kernel2d(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 2D first-order difference kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D first-order difference kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_diff_kernel2d
    >>> diff_kernel = get_diff_kernel2d()
    >>> print(diff_kernel)
    tensor([[[-0.,  0.,  0.],
             [-1.,  0.,  1.],
             [-0.,  0.,  0.]],
            [[-0., -1., -0.],
             [ 0.,  0.,  0.],
             [ 0.,  1.,  0.]]])
    """
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 2D second-order Sobel kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D second-order Sobel kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_sobel_kernel2d_2nd_order
    >>> sobel_kernel = get_sobel_kernel2d_2nd_order()
    >>> print(sobel_kernel)
    tensor([[[-1.,  0.,  2.,  0., -1.],
             [-4.,  0.,  8.,  0., -4.],
             [-6.,  0., 12.,  0., -6.],
             [-4.,  0.,  8.,  0., -4.],
             [-1.,  0.,  2.,  0., -1.]],
            [[-1., -2.,  0.,  2.,  1.],
             [-2., -4.,  0.,  4.,  2.],
             [ 0.,  0.,  0.,  0.,  0.],
             [ 2.,  4.,  0., -4., -2.],
             [ 1.,  2.,  0., -2., -1.]],
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]])
    """
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order(
    *, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Utility function that returns a 2D second-order difference kernel.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D second-order difference kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_diff_kernel2d_2nd_order
    >>> diff_kernel = get_diff_kernel2d_2nd_order()
    >>> print(diff_kernel)
    tensor([[[-0.,  0.,  0.],
             [-1.,  0.,  1.],
             [-0.,  0.,  0.]],
            [[-0., -1., -0.],
             [ 0.,  0.,  0.],
             [ 0.,  1.,  0.]]])
    """
    gxx = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype
    )
    gyy = gxx.transpose(0, 1)
    gxy = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    return stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(
    mode: str,
    order: int,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Kernel for 1st or 2nd order image gradients.

    Parameters
    ----------
    mode: str
        The operator to use. Should be either "sobel" or "diff".
    order: int
        The order of the gradient.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        The kernel for the spatial gradient.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_spatial_gradient_kernel2d
    >>> kernel = get_spatial_gradient_kernel2d("sobel", 1)
    >>> print(kernel)
    tensor([[[-1.,  0.,  1.],
             [-2.,  0.,  2.],
             [-1.,  0.,  1.]],
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]])
    """
    if mode == "sobel" and order == 1:
        return get_sobel_kernel2d(device=device, dtype=dtype)
    if mode == "sobel" and order == 2:  # noqa: PLR2004
        return get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    if mode == "diff" and order == 1:
        return get_diff_kernel2d(device=device, dtype=dtype)
    if mode == "diff" and order == 2:  # noqa: PLR2004
        return get_diff_kernel2d_2nd_order(device=device, dtype=dtype)
    msg = f"Not implemented: {order=} on {mode=}"
    raise NotImplementedError(msg)


def get_spatial_gradient_kernel3d(
    mode: str,
    order: int,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Kernel for 1st or 2nd order scale pyramid gradients.

    Parameters
    ----------
    mode: str
        The operator to use. Should be either "sobel" or "diff".
    order: int
        The order of the gradient.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        The kernel for the spatial gradient.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_spatial_gradient_kernel3d
    >>> kernel = get_spatial_gradient_kernel3d("sobel", 1)
    >>> print(kernel)
    tensor([[[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [-0.5,  0.,  0.5],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0., -0.5,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.5,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0., -0.5,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]],
             [[ 0.,  0.,  0.],
              [ 0.,  0.5,  0.],
              [ 0.,  0.,  0.]]]])
    """
    if mode == "diff" and order == 1:
        return get_diff_kernel3d(device=device, dtype=dtype)
    if mode == "diff" and order == 2:  # noqa: PLR2004
        return get_diff_kernel3d_2nd_order(device=device, dtype=dtype)
    msg = f"Not implemented: {order=} on {mode=}"
    raise NotImplementedError(msg)


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 1D Gaussian kernel.

    Parameters
    ----------
    kernel_size: int
        The size of the kernel.
    sigma: float or Tensor
        The standard deviation of the Gaussian distribution.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 1D Gaussian kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_gaussian_kernel1d
    >>> kernel_size = 5
    >>> sigma = torch.tensor([[1.0]])
    >>> gauss_kernel = get_gaussian_kernel1d(kernel_size, sigma)
    >>> print(gauss_kernel)
    tensor([[0.1353, 0.6065, 1.0000, 0.6065, 0.1353]])
    """
    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_discrete_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 1D discrete Gaussian kernel.

    Parameters
    ----------
    kernel_size: int
        The size of the kernel.
    sigma: float or Tensor
        The standard deviation of the Gaussian distribution.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 1D discrete Gaussian kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_gaussian_discrete_kernel1d
    >>> kernel_size = 5
    >>> sigma = torch.tensor([[1.0]])
    >>> gauss_kernel = get_gaussian_discrete_kernel1d(kernel_size, sigma)
    >>> print(gauss_kernel)
    tensor([[0.1353, 0.6065, 1.0000, 0.6065, 0.1353]])
    """
    return gaussian_discrete(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_erf_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 1D Gaussian kernel using the error function.

    Parameters
    ----------
    kernel_size: int
        The size of the kernel.
    sigma: float or Tensor
        The standard deviation of the Gaussian distribution.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 1D Gaussian kernel using the error function.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_gaussian_erf_kernel1d
    >>> kernel_size = 5
    >>> sigma = torch.tensor([[1.0]])
    >>> gauss_kernel = get_gaussian_erf_kernel1d(kernel_size, sigma)
    >>> print(gauss_kernel)
    tensor([[0.1353, 0.6065, 1.0000, 0.6065, 0.1353]])
    """
    return gaussian_discrete_erf(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 2D Gaussian kernel.

    Parameters
    ----------
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    sigma: tuple[float, float] | Tensor
        The standard deviation of the Gaussian distribution.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D Gaussian kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_gaussian_kernel2d
    >>> kernel_size = (5, 5)
    >>> sigma = torch.tensor([[1.0, 1.0]])
    >>> gauss_kernel = get_gaussian_kernel2d(kernel_size, sigma)
    >>> print(gauss_kernel)
    tensor([[[0.1353, 0.6065, 1.0000, 0.6065, 0.1353],
             [0.6065, 2.7183, 4.4817, 2.7183, 0.6065],
             [1.0000, 4.4817, 7.3891, 4.4817, 1.0000],
             [0.6065, 2.7183, 4.4817, 2.7183, 0.6065],
             [0.1353, 0.6065, 1.0000, 0.6065, 0.1353]]])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    assert_tensor(sigma)
    assert_shape(sigma, ("B", "2"))

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, device=device, dtype=dtype)[
        ..., None
    ]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, device=device, dtype=dtype)[
        ..., None
    ]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel3d(
    kernel_size: tuple[int, int, int] | int,
    sigma: tuple[float, float, float] | Tensor,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 3D Gaussian kernel.

    Parameters
    ----------
    kernel_size: tuple[int, int, int] | int
        The size of the kernel.
    sigma: tuple[float, float, float] | Tensor
        The standard deviation of the Gaussian distribution.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 3D Gaussian kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_gaussian_kernel3d
    >>> kernel_size = (3, 3, 3)
    >>> sigma = torch.tensor([[1.0, 1.0, 1.0]])
    >>> gauss_kernel = get_gaussian_kernel3d(kernel_size, sigma)
    >>> print(gauss_kernel)
    tensor([[[[0.1353, 0.6065, 1.0000],
              [0.6065, 2.7183, 4.4817],
              [1.0000, 4.4817, 7.3891]],
             [[0.6065, 2.7183, 4.4817],
              [2.7183, 12.1825, 20.0855],
              [4.4817, 20.0855, 33.1155]],
             [[1.0000, 4.4817, 7.3891],
              [4.4817, 20.0855, 33.1155],
              [7.3891, 33.1155, 54.5982]]]])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    assert_tensor(sigma)
    assert_shape(sigma, ("B", "3"))

    ksize_z, ksize_y, ksize_x = _unpack_3d_ks(kernel_size)
    sigma_z, sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None], sigma[:, 2, None]

    kernel_z = get_gaussian_kernel1d(ksize_z, sigma_z, device=device, dtype=dtype)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, device=device, dtype=dtype)
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, device=device, dtype=dtype)

    return (
        kernel_z.view(-1, ksize_z, 1, 1)
        * kernel_y.view(-1, 1, ksize_y, 1)
        * kernel_x.view(-1, 1, 1, ksize_x)
    )


def get_laplacian_kernel1d(
    kernel_size: int, *, device: Device | None = None, dtype: DType = torch.float32
) -> Tensor:
    r"""
    Utility function that returns a 1D Laplacian kernel.

    Parameters
    ----------
    kernel_size: int
        The size of the kernel.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 1D Laplacian kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_laplacian_kernel1d
    >>> kernel_size = 5
    >>> laplacian_kernel = get_laplacian_kernel1d(kernel_size)
    >>> print(laplacian_kernel)
    tensor([ 1.,  1., -4.,  1.,  1.])
    """
    return laplacian_1d(kernel_size, device=device, dtype=dtype)


def get_laplacian_kernel2d(
    kernel_size: tuple[int, int] | int,
    *,
    device: Device | None = None,
    dtype: DType = torch.float32,
) -> Tensor:
    r"""
    Utility function that returns a 2D Laplacian kernel.

    Parameters
    ----------
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D Laplacian kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_laplacian_kernel2d
    >>> kernel_size = (3, 3)
    >>> laplacian_kernel = get_laplacian_kernel2d(kernel_size)
    >>> print(laplacian_kernel)
    tensor([[ 1.,  1.,  1.],
            [ 1., -8.,  1.],
            [ 1.,  1.,  1.]])
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    kernel = torch.ones((ky, kx), device=device, dtype=dtype)
    mid_x = kx // 2
    mid_y = ky // 2

    kernel[mid_y, mid_x] = 1 - kernel.sum()
    return kernel


def get_pascal_kernel_2d(
    kernel_size: tuple[int, int] | int,
    norm: bool = True,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Utility function that returns a 2D Pascal kernel.

    Parameters
    ----------
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    norm: bool, optional
        Whether to normalize the kernel. Default is True.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A 2D Pascal kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_pascal_kernel_2d
    >>> kernel_size = (3, 3)
    >>> pascal_kernel = get_pascal_kernel_2d(kernel_size)
    >>> print(pascal_kernel)
    tensor([[0.1111, 0.1111, 0.1111],
            [0.1111, 0.1111, 0.1111],
            [0.1111, 0.1111, 0.1111]])
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    ax = get_pascal_kernel_1d(kx, device=device, dtype=dtype)
    ay = get_pascal_kernel_1d(ky, device=device, dtype=dtype)

    filt = ay[:, None] * ax[None, :]
    if norm:
        filt = filt / torch.sum(filt)
    return filt


def get_pascal_kernel_1d(
    kernel_size: int,
    norm: bool = False,
    *,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Generate Yang Hui triangle (Pascal's triangle) by a given number.

    Parameters
    ----------
    kernel_size: int
        Kernel size. Should be positive.
    norm: bool, optional
        Whether to normalize the kernel. Default is False.
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        Pascal kernel with shape :math:`(kernel_size,)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_pascal_kernel_1d
    >>> kernel_size = 5
    >>> pascal_kernel = get_pascal_kernel_1d(kernel_size)
    >>> print(pascal_kernel)
    tensor([ 1.,  4.,  6.,  4.,  1.])
    """
    pre: list[float] = []
    cur: list[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_canny_nms_kernel(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Kernel used in Canny-NMS.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A Canny-NMS kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_canny_nms_kernel
    >>> canny_nms_kernel = get_canny_nms_kernel()
    >>> print(canny_nms_kernel)
    tensor([[[[ 0.,  0.,  0.],
              [ 0.,  1., -1.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0., -1.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0., -1.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [-1.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [-1.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[[-1.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0., -1.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0., -1.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]]])
    """
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hysteresis_kernel(
    device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Kernel used in Canny hysteresis.

    Parameters
    ----------
    device: Device, optional
        The desired device of returned tensor.
    dtype: DType, optional
        The desired data type of returned tensor.

    Returns
    -------
    Tensor
        A Canny hysteresis kernel.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_hysteresis_kernel
    >>> hysteresis_kernel = get_hysteresis_kernel()
    >>> print(hysteresis_kernel)
    tensor([[[[ 0.,  0.,  0.],
              [ 0.,  0.,  1.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  1.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  1.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 1.,  0.,  0.]]],
            [[[ 0.,  0.,   0.],
              [ 1.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 1.,  0.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  1.,  0.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  1.],
              [ 0.,  0.,  0.],
              [ 0.,  0.,  0.]]]])
    """
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )


def get_hanning_kernel1d(
    kernel_size: int, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    r"""
    Hanning kernel.

    .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    See Also
    --------
    - `NumPy Documentation <https://numpy.org/doc/stable/reference/generated/numpy.hanning.html>`_

    Parameters
    ----------
    kernel_size: int
        The size the of the kernel. It should be positive.
    device: Device, optional
        Tensor device.
    dtype: DType, optional
        Tensor data type.

    Returns
    -------
    Tensor
        1D tensor with Hanning filter coefficients.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_hanning_kernel1d
    >>> kernel_size = 5
    >>> hanning_kernel = get_hanning_kernel1d(kernel_size)
    >>> print(hanning_kernel)
    tensor([0.0000, 0.5000, 1.0000, 0.5000, 0.0000])
    """
    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = 0.5 - 0.5 * torch.cos(2.0 * math.pi * x / float(kernel_size - 1))
    return x


def get_hanning_kernel2d(
    kernel_size: tuple[int, int] | int,
    device: Device | None = None,
    dtype: DType | None = None,
) -> Tensor:
    r"""
    Returns 2D Hanning kernel, used in signal processing and KCF tracker.

    Parameters
    ----------
    kernel_size: tuple[int, int] | int
        The size of the kernel for the filter. It should be positive.
    device: Device, optional
        Tensor device.
    dtype: DType, optional
        Tensor dtype.

    Returns
    -------
    Tensor
        2D tensor with Hanning filter coefficients.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import get_hanning_kernel2d
    >>> kernel_size = (5, 5)
    >>> hanning_kernel = get_hanning_kernel2d(kernel_size)
    >>> print(hanning_kernel)
    tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.2500, 0.5000, 0.2500, 0.0000],
            [0.0000, 0.5000, 1.0000, 0.5000, 0.0000],
            [0.0000, 0.2500, 0.5000, 0.2500, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
    """
    kernel_size = _unpack_2d_ks(kernel_size)
    ky = get_hanning_kernel1d(kernel_size[0], device, dtype)[None].T
    kx = get_hanning_kernel1d(kernel_size[1], device, dtype)[None]
    return ky @ kx


def _preprocess_fast_guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    subsample: int = 1,
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    ky, kx = _unpack_2d_ks(kernel_size)
    if subsample > 1:
        s = 1 / subsample
        guidance_sub = nn.functional.interpolate(
            guidance, scale_factor=s, mode="nearest"
        )
        input_sub = (
            guidance_sub
            if input is guidance
            else nn.functional.interpolate(input, scale_factor=s, mode="nearest")
        )
        ky, kx = ((k - 1) // subsample + 1 for k in (ky, kx))
    else:
        guidance_sub = guidance
        input_sub = input
    return guidance_sub, input_sub, (ky, kx)


def _guided_blur_grayscale_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: FilterBorder | str = "reflect",
    subsample: int = 1,
) -> Tensor:
    r"""
    Apply guided blur with grayscale guidance.

    Parameters
    ----------
    guidance: Tensor
        The guidance tensor with shape :math:`(B, 1, H, W)`.
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    eps: float | Tensor
        The regularization parameter.
    border_type: FilterBorder | str, optional
        The border type for padding. Default is "reflect".
    subsample: int, optional
        The subsampling factor. Default is 1.

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _guided_blur_grayscale_guidance
    >>> guidance = torch.rand(1, 1, 5, 5)
    >>> input = torch.rand(1, 1, 5, 5)
    >>> output = _guided_blur_grayscale_guidance(guidance, input, (3, 3), 0.1)
    >>> print(output.shape)
    torch.Size([1, 1, 5, 5])
    """
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(
        guidance, input, kernel_size, subsample
    )

    mean_I = box_blur(guidance_sub, kernel_size, border_type)
    corr_I = box_blur(guidance_sub.square(), kernel_size, border_type)
    var_I = corr_I - mean_I.square()

    if input is guidance:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type)
        corr_Ip = box_blur(guidance_sub * input_sub, kernel_size, border_type)
        cov_Ip = corr_Ip - mean_I * mean_p

    if isinstance(eps, torch.Tensor):
        eps = eps.view(-1, 1, 1, 1)  # N -> NCHW

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_blur(a, kernel_size, border_type)
    mean_b = box_blur(b, kernel_size, border_type)

    if subsample > 1:
        mean_a = nn.functional.interpolate(
            mean_a, scale_factor=subsample, mode="bilinear"
        )
        mean_b = nn.functional.interpolate(
            mean_b, scale_factor=subsample, mode="bilinear"
        )

    return mean_a * guidance + mean_b


def _guided_blur_multichannel_guidance(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: FilterBorder | str = "reflect",
    subsample: int = 1,
) -> Tensor:
    r"""
    Apply guided blur with multichannel guidance.

    Parameters
    ----------
    guidance: Tensor
        The guidance tensor with shape :math:`(B, C, H, W)`.
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    eps: float | Tensor
        The regularization parameter.
    border_type: FilterBorder | str, optional
        The border type for padding. Default is "reflect".
    subsample: int, optional
        The subsampling factor. Default is 1.

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _guided_blur_multichannel_guidance
    >>> guidance = torch.rand(1, 3, 5, 5)
    >>> input = torch.rand(1, 3, 5, 5)
    >>> output = _guided_blur_multichannel_guidance(guidance, input, (3, 3), 0.1)
    >>> print(output.shape)
    torch.Size([1, 3, 5, 5])
    """
    guidance_sub, input_sub, kernel_size = _preprocess_fast_guided_blur(
        guidance, input, kernel_size, subsample
    )
    B, C, H, W = guidance_sub.shape

    mean_I = box_blur(guidance_sub, kernel_size, border_type).permute(0, 2, 3, 1)
    II = (guidance_sub.unsqueeze(1) * guidance_sub.unsqueeze(2)).flatten(1, 2)
    corr_I = box_blur(II, kernel_size, border_type).permute(0, 2, 3, 1)
    var_I = corr_I.reshape(B, H, W, C, C) - mean_I.unsqueeze(-2) * mean_I.unsqueeze(-1)

    if guidance is input:
        mean_p = mean_I
        cov_Ip = var_I

    else:
        mean_p = box_blur(input_sub, kernel_size, border_type).permute(0, 2, 3, 1)
        Ip = (input_sub.unsqueeze(1) * guidance_sub.unsqueeze(2)).flatten(1, 2)
        corr_Ip = box_blur(Ip, kernel_size, border_type).permute(0, 2, 3, 1)
        cov_Ip = corr_Ip.reshape(B, H, W, C, -1) - mean_p.unsqueeze(
            -2
        ) * mean_I.unsqueeze(-1)

    if isinstance(eps, torch.Tensor):
        _eps = torch.eye(C, device=guidance.device, dtype=guidance.dtype).view(
            1, 1, 1, C, C
        ) * eps.view(-1, 1, 1, 1, 1)
    else:
        _eps = guidance.new_full((C,), eps).diag().view(1, 1, 1, C, C)
    a = torch.linalg.solve(var_I + _eps, cov_Ip)  # B, H, W, C_guidance, C_input
    b = mean_p - (mean_I.unsqueeze(-2) @ a).squeeze(-2)  # B, H, W, C_input

    mean_a = box_blur(a.flatten(-2).permute(0, 3, 1, 2), kernel_size, border_type)
    mean_b = box_blur(b.permute(0, 3, 1, 2), kernel_size, border_type)

    if subsample > 1:
        mean_a = nn.functional.interpolate(
            mean_a, scale_factor=subsample, mode="bilinear"
        )
        mean_b = nn.functional.interpolate(
            mean_b, scale_factor=subsample, mode="bilinear"
        )
    mean_a = mean_a.view(B, C, -1, H * subsample, W * subsample)

    # einsum might not be contiguous, thus mean_b is the first argument
    return mean_b + torch.einsum("BCHW,BCcHW->BcHW", guidance, mean_a)


def guided_blur(
    guidance: Tensor,
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    eps: float | Tensor,
    border_type: FilterBorder | str = "reflect",
    subsample: int = 1,
) -> Tensor:
    r"""
    Edge-preserving image smoothing filter.
    Guidance and input can have different number of channels.

    See Also
    --------
    - Original papers :cite:`he2010guided` and :cite:`he2015fast`.

    Parameters
    ----------
    guidance: Tensor
        The guidance tensor with shape :math:`(B, C, H, W)`.
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    eps: float | Tensor
        The regularization parameter. Smaller values preserve more edges.
    border_type: FilterBorder | str, optional
        The border type for padding. Default is "reflect".
    subsample: int, optional
        The subsampling factor for Fast Guided filtering. Default is 1 (no subsampling).

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import guided_blur
    >>> guidance = torch.rand(2, 3, 5, 5)
    >>> input = torch.rand(2, 4, 5, 5)
    >>> output = guided_blur(guidance, input, 3, 0.1)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """
    assert_tensor(guidance)
    assert_shape(guidance, ("B", "C", "H", "W"))
    if input is not guidance:
        assert_tensor(input)
        assert_shape(input, ("B", "C", "H", "W"))

    if guidance.shape[1] == 1:
        return _guided_blur_grayscale_guidance(
            guidance, input, kernel_size, eps, border_type, subsample
        )
    return _guided_blur_multichannel_guidance(
        guidance, input, kernel_size, eps, border_type, subsample
    )


class GuidedBlur(nn.Module):
    r"""
    Apply guided blur to the input tensor.

    Parameters
    ----------
    kernel_size:
        The size of the kernel.
    eps:
        The regularization parameter. Smaller values preserve more edges.
    border_type:
        The border type for padding.
    subsample:
        The subsampling factor for Fast Guided filtering. Default is 1 (no subsampling).

    Shape
    -----
    - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import GuidedBlur
    >>> guidance = torch.rand(2, 3, 5, 5)
    >>> input = torch.rand(2, 4, 5, 5)
    >>> blur = GuidedBlur(3, 0.1)
    >>> output = blur(guidance, input)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        eps: float,
        border_type: FilterBorder | str = "reflect",
        subsample: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.border_type = border_type
        self.subsample = subsample

    @TX.override
    def forward(self, guidance: Tensor, input: Tensor) -> Tensor:
        return guided_blur(
            guidance,
            input,
            self.kernel_size,
            self.eps,
            self.border_type,
            self.subsample,
        )


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    r"""
    Compute the zero padding for a given kernel size.

    Parameters
    ----------
    kernel_size: tuple[int, int] | int
        The size of the kernel.

    Returns
    -------
    tuple[int, int]
        The zero padding for the given kernel size.

    Examples
    --------
    >>> from unipercept.vision.filter import _compute_zero_padding
    >>> kernel_size = (3, 3)
    >>> padding = _compute_zero_padding(kernel_size)
    >>> print(padding)
    (1, 1)
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def median_blur(input: Tensor, kernel_size: tuple[int, int] | int) -> Tensor:
    r"""
    Blur an image using the median filter.

    Parameters
    ----------
    input: Tensor
        Input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        Blur kernel size.

    Returns
    -------
    Tensor
        Output tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import median_blur
    >>> input = torch.rand(2, 4, 5, 7)
    >>> output = median_blur(input, (3, 3))
    >>> print(output.shape)
    torch.Size([2, 4, 5, 7])
    """
    assert_tensor(input)
    assert_shape(input, ("B", "C", "H", "W"))

    padding = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: Tensor = get_binary_kernel2d(
        kernel_size, device=input.device, dtype=input.dtype
    )
    b, c, h, w = input.shape

    # map the local window to single vector
    features: Tensor = nn.functional.conv2d(
        input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    return features.median(dim=2)[0]


class MedianBlur(nn.Module):
    r"""
    Blur an image using the median filter.

    Parameters
    ----------
    kernel_size:
        The kernel size.

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import MedianBlur
    >>> input = torch.rand(2, 4, 5, 7)
    >>> blur = MedianBlur((3, 3))
    >>> output = blur(input)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: tuple[int, int] | int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        return median_blur(input, self.kernel_size)


def _bilateral_blur(
    input: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: FilterBorder | str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""
    Apply bilateral blur to the input tensor.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    guidance: Tensor or None
        The guidance tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    sigma_color: float | Tensor
        The standard deviation for intensity/color Gaussian kernel.
    sigma_space: tuple[float, float] | Tensor
        The standard deviation for spatial Gaussian kernel.
    border_type: FilterBorder | str, optional
        The border type for padding. Default is "reflect".
    color_distance_type: str, optional
        The type of distance to calculate intensity/color difference. Default is "l1".

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _bilateral_blur
    >>> input = torch.rand(1, 1, 5, 5)
    >>> output = _bilateral_blur(input, None, (3, 3), 0.1, (1.5, 1.5))
    >>> print(output.shape)
    torch.Size([1, 1, 5, 5])
    """
    assert_tensor(input)
    assert_shape(input, ("B", "C", "H", "W"))
    if guidance is not None:
        assert_tensor(guidance)
        assert_shape(guidance, ("B", "C", "H", "W"))

    if isinstance(sigma_color, torch.Tensor):
        assert_shape(sigma_color, ("B"))
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(
            -1, 1, 1, 1, 1
        )

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = nn.functional.pad(
        input, (pad_x, pad_x, pad_y, pad_y), mode=border_type
    )
    unfolded_input = (
        padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)
    )  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = nn.functional.pad(
            guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type
        )
        unfolded_guidance = (
            padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)
        )  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        msg = f"{color_distance_type=} should be either 'l1' or 'l2'."
        raise ValueError(msg)
    color_kernel = (
        -0.5 / sigma_color**2 * color_distance_sq
    ).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(
        kernel_size, sigma_space, device=input.device, dtype=input.dtype
    )
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: FilterBorder | str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""
    Blur a tensor using a Bilateral filter.

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    sigma_color: float | Tensor
        The standard deviation for intensity/color Gaussian kernel. Smaller values preserve more edges.
    sigma_space: tuple[float, float] | Tensor
        The standard deviation for spatial Gaussian kernel. This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type: FilterBorder | str, optional
        The padding mode to be applied before convolving. Default is "reflect".
    color_distance_type: str, optional
        The type of distance to calculate intensity/color difference. Only ``'l1'`` or ``'l2'`` is allowed. Default is "l1".

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import bilateral_blur
    >>> input = torch.rand(2, 4, 5, 5)
    >>> output = bilateral_blur(input, (3, 3), 0.1, (1.5, 1.5))
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """
    return _bilateral_blur(
        input,
        None,
        kernel_size,
        sigma_color,
        sigma_space,
        border_type,
        color_distance_type,
    )


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: FilterBorder | str = "reflect",
    color_distance_type: str = "l1",
) -> Tensor:
    r"""
    Blur a tensor using a Joint Bilateral filter.

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :func:`bilateral_blur()` for more information.

    Parameters
    ----------
    input: Tensor
        The input tensor with shape :math:`(B, C, H, W)`.
    guidance: Tensor
        The guidance tensor with shape :math:`(B, C, H, W)`.
    kernel_size: tuple[int, int] | int
        The size of the kernel.
    sigma_color: float | Tensor
        The standard deviation for intensity/color Gaussian kernel. Smaller values preserve more edges.
    sigma_space: tuple[float, float] | Tensor
        The standard deviation for spatial Gaussian kernel. This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type: FilterBorder | str, optional
        The padding mode to be applied before convolving. Default is "reflect".
    color_distance_type: str, optional
        The type of distance to calculate intensity/color difference. Only ``'l1'`` or ``'l2'`` is allowed. Default is "l1".

    Returns
    -------
    Tensor
        The blurred tensor with shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import joint_bilateral_blur
    >>> input = torch.rand(2, 4, 5, 5)
    >>> guidance = torch.rand(2, 4, 5, 5)
    >>> output = joint_bilateral_blur(input, guidance, (3, 3), 0.1, (1.5, 1.5))
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """
    return _bilateral_blur(
        input,
        guidance,
        kernel_size,
        sigma_color,
        sigma_space,
        border_type,
        color_distance_type,
    )


class _BilateralBlur(nn.Module):
    r"""
    Base class for Bilateral and Joint Bilateral blur.

    Parameters
    ----------
    kernel_size:
        The size of the kernel.
    sigma_color:
        The standard deviation for intensity/color Gaussian kernel. Smaller values preserve more edges.
    sigma_space:
        The standard deviation for spatial Gaussian kernel. This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type:
        The padding mode to be applied before convolving. Default is "reflect".
    color_distance_type:
        The type of distance to calculate intensity/color difference. Only ``'l1'`` or ``'l2'`` is allowed. Default is "l1".

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import _BilateralBlur
    >>> input = torch.rand(2, 4, 5, 5)
    >>> blur = _BilateralBlur((3, 3), 0.1, (1.5, 1.5))
    >>> output = blur(input)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: FilterBorder | str = "reflect",
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    r"""
    Blur a tensor using a Bilateral filter.

    The operator is an edge-preserving image smoothing filter. The weight
    for each pixel in a neighborhood is determined not only by its distance
    to the center pixel, but also the difference in intensity or color.

    Parameters
    ----------
    kernel_size:
        The size of the kernel.
    sigma_color:
        The standard deviation for intensity/color Gaussian kernel. Smaller values preserve more edges.
    sigma_space:
        The standard deviation for spatial Gaussian kernel. This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type:
        The padding mode to be applied before convolving. Default is "reflect".
    color_distance_type:
        The type of distance to calculate intensity/color difference. Only ``'l1'`` or ``'l2'`` is allowed. Default is "l1".

    Shape
    -----
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import BilateralBlur
    >>> input = torch.rand(2, 4, 5, 5)
    >>> blur = BilateralBlur((3, 3), 0.1, (1.5, 1.5))
    >>> output = blur(input)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        return bilateral_blur(
            input,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )


class JointBilateralBlur(_BilateralBlur):
    r"""
    Blur a tensor using a Joint Bilateral filter.

    This operator is almost identical to a Bilateral filter. The only difference
    is that the color Gaussian kernel is computed based on another image called
    a guidance image. See :class:`BilateralBlur` for more information.

    Parameters
    ----------
    kernel_size:
        The size of the kernel.
    sigma_color:
        The standard deviation for intensity/color Gaussian kernel. Smaller values preserve more edges.
    sigma_space:
        The standard deviation for spatial Gaussian kernel. This is similar to ``sigma`` in :func:`gaussian_blur2d()`.
    border_type:
        The padding mode to be applied before convolving. Default is "reflect".
    color_distance_type:
        The type of distance to calculate intensity/color difference. Only ``'l1'`` or ``'l2'`` is allowed. Default is "l1".

    Shape
    -----
    - Input: :math:`(B, C, H, W)`, :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

    Examples
    --------
    >>> import torch
    >>> from unipercept.vision.filter import JointBilateralBlur
    >>> input = torch.rand(2, 4, 5, 5)
    >>> guidance = torch.rand(2, 4, 5, 5)
    >>> blur = JointBilateralBlur((3, 3), 0.1, (1.5, 1.5))
    >>> output = blur(input, guidance)
    >>> print(output.shape)
    torch.Size([2, 4, 5, 5])
    """

    @TX.override
    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )
