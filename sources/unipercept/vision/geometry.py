r"""
Various utilities for working with geometry transformations and conversions.

See Also
--------
- `Quaternions <https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html>`_
- `PyTorch3D <https://pytorch3d.readthedocs.io/>`_
"""

import enum as E
import typing

import torch
import torch.fx
from torch import nn

from unipercept.types import Device, DType, Size, Tensor
from unipercept.vision.coord import (
    GridMode,
    generate_coord_grid,
    generate_coord_grid_as,
)


class AxesConvention(E.StrEnum):
    r"""
    Enum for conventions on how to orient the world coordinate system:

        - ``OPENCV`` : ``(+x, +y, +z) = (right   , down  , fwd    )`` (right-handed)
        - ``ISO8855``: ``(+x, +y, +z) = (fwd     , left  , up     )`` (right-handed)
        - ``OPENGL`` : ``(+x, +y, +z) = (right   , up    , bwd    )`` (right-handed)
        - ``OPEN3D`` : ``(+x, +y, +z) = (right   , up    , bwd    )`` (right-handed)
        - ``DIRECTX``: ``(+x, +y, +z) = (right   , up    , fwd    )`` (left-handed)
    Notes
    -----
    The camera coordinate system is always oriented as ``OPENCV``, and pixels are
    oriented as ``(+u, +v) = (right, down)`` in the image plane, with (0, 0) at the
    top-left corner.

    See Also
    --------
    - `OpenCV documentation <https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html>`_
    - `Cityscapes calibration <https://github.com/mcordts/cityscapesScripts>`_
    - `Learn OpenGL <https://learnopengl.com/Getting-started/Coordinate-Systems>`_
    """

    ISO8855 = E.auto()
    OPENCV = E.auto()
    OPENGL = E.auto()
    OPEN3D = E.auto()
    DIRECTX = E.auto()


def _transform_to_opencv(
    convention: AxesConvention | str,
    device: Device | None = None,
    dtype: DType = torch.float32,
) -> Tensor:
    match convention:
        case AxesConvention.OPENCV:
            # (X, Y, Z) = [right, down, fwd]
            return torch.eye(4, device=device, dtype=dtype)
        case AxesConvention.ISO8855:
            # (X, Y, Z) = [fwd, left, up]
            return torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],  # X -> Z
                    [-1.0, 0.0, 0.0, 0.0],  # Y -> -X
                    [0.0, -1.0, 0.0, 0.0],  # Z -> -Y
                    [0.0, 0.0, 0.0, 1.0],  # 1 -> 1 (homogeneous)
                ],
                device=device,
                dtype=dtype,
            )
        case AxesConvention.OPENGL | AxesConvention.OPEN3D:
            # (X, Y, Z) = [right, up, bw]
            return torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],  # X -> X
                    [0.0, -1.0, 0.0, 0.0],  # Y -> -Y
                    [0.0, 0.0, -1.0, 0.0],  # Z -> -Z
                    [0.0, 0.0, 0.0, 1.0],  # 1 -> 1 (homogeneous)
                ],
                device=device,
                dtype=dtype,
            )
        case _:
            msg = f"Unknown convention {convention!r}!"
            raise ValueError(msg)


def _mapping_to_opencv(
    convention: AxesConvention | str,
    device: Device | None = None,
    dtype: DType = torch.float32,
) -> Tensor:
    return _transform_to_opencv(convention, device=device, dtype=dtype)[:3, :3]


def convert_extrinsics(
    extrinsics: Tensor,
    *,
    src: AxesConvention | str = AxesConvention.OPENCV,
    tgt: AxesConvention | str = AxesConvention.OPENCV,
) -> Tensor:
    r"""
    Changes an extrinsic transformation to a different world axes convention.

    This is useful in cases where a camera/dataset provides the extrinsic parameters
    in a different world convention than ours. For example, the Cityscapes dataset
    provides their extrinsics in ISO8855 contention, while we use OpenCV. This function
    allows using the direct parameters from the dataset to build the extrinsics matrix,
    which can subsequently be converted to OpenCV for use in our system.

    Parameters
    ----------
    extrinsics: Tensor[*, 4, 4]
        Input matrix.
    src: AxesConvention or str
        Current convention. See :class:`AxesConvention` for possible values.
    tgt: AxesConvention or str
        Target convention. See :class:`AxesConvention` for possible values.

    Returns
    -------
    Tensor[*, 4, 4]
        Output matrix.

    Examples
    --------
    >>> extrinsics = torch.eye(4)
    >>> converted_extrinsics = convert_extrinsics(
    ...     extrinsics, src=AxesConvention.ISO8855, tgt=AxesConvention.OPENCV
    ... )
    >>> print(converted_extrinsics)
    tensor([[ 0.,  0.,  1.,  0.],
            [-1.,  0.,  0.,  0.],
            [ 0., -1.,  0.,  0.],
            [ 0.,  0.,  0.,  1.]])
    """

    if src == tgt:
        return extrinsics

    device = extrinsics.device
    dtype = extrinsics.dtype
    T_src = _transform_to_opencv(src, device=device, dtype=dtype).T
    T_tgt = _transform_to_opencv(tgt, device=device, dtype=dtype)

    return extrinsics @ T_src @ T_tgt


def convert_points(
    points: torch.Tensor,
    *,
    src: AxesConvention | str = AxesConvention.OPENCV,
    tgt: AxesConvention | str = AxesConvention.OPENCV,
) -> torch.Tensor:
    r"""
    Converts points from one world axes convention to another.

    This is useful in cases where we want to display points in a different world
    convention than the one they were projected into.
    For example, if we have a projection onto OpenCV coordinates (the default) and we
    want to display a point cloud in Matplotlib (which uses Z-up coordinates), then
    we can convert the projected points to ISO8855 to get the correct orientation.

    Parameters
    ----------
    points: Tensor[*, 3]
        Input points.
    src: AxesConvention or str
        Current convention. See :class:`AxesConvention` for possible values.
    tgt: AxesConvention or str
        Target convention. See :class:`AxesConvention` for possible values.

    Returns
    -------
    Tensor[*, 3]
        Output points.

    Examples
    --------
    >>> points = torch.tensor([[1.0, 2.0, 3.0]])
    >>> converted_points = convert_points(
    ...     points, src=AxesConvention.ISO8855, tgt=AxesConvention.OPENCV
    ... )
    >>> print(converted_points)
    tensor([[ 3., -1., -2.]])
    """
    if src == tgt:
        return points

    device = points.device
    dtype = points.dtype
    R_src = _mapping_to_opencv(src, device=device, dtype=dtype)
    R_tgt = _mapping_to_opencv(tgt, device=device, dtype=dtype).T
    return points @ R_src @ R_tgt


def extrinsics_from_parameters(
    angles: list[tuple[float, float, float]] | Tensor,
    translation: list[tuple[float, float, float]] | Tensor,
    convention: AxesConvention | str = AxesConvention.ISO8855,
) -> Tensor:
    r"""
    Build the extrinsic matrix (R|t) using an angles vector and translations.

    Parameters
    ----------
    angles: list[tuple[float, float, float]] or Tensor
        List of tuples or tensor containing the rotation angles.
    translation: list[tuple[float, float, float]] or Tensor
        List of tuples or tensor containing the translation vectors.
    convention: AxesConvention or str
        Convention to use for the extrinsic matrix. See :class:`AxesConvention` for possible values.

    Returns
    -------
    Tensor[*, 4, 4]
        Extrinsic matrix.

    Examples
    --------
    >>> angles = [(0.0, 0.0, 0.0)]
    >>> translation = [(0.0, 0.0, 0.0)]
    >>> extrinsics = extrinsics_from_parameters(
    ...     angles, translation, convention=AxesConvention.ISO8855
    ... )
    >>> print(extrinsics)
    tensor([[ 0.,  0.,  1.,  0.],
            [-1.,  0.,  0.,  0.],
            [ 0., -1.,  0.,  0.],
            [ 0.,  0.,  0.,  1.]])
    """

    rotation = torch.as_tensor(angles).float()
    translation = torch.as_tensor(translation).float()

    # Amount of cameras
    ndim = rotation.ndim
    assert ndim == translation.ndim
    if ndim == 1:
        rotation = rotation.unsqueeze(0)
        translation = translation.unsqueeze(0)

    N = rotation.size(0)

    # Value checks
    assert rotation.shape[-1] == 3, rotation.shape
    assert translation.shape[-1] == 3, translation.shape

    # Create extrinsic matrix [R|t] + [0 0 0 1]
    extrinsic_matrix = torch.zeros(N, 4, 4, device=rotation[0].device)
    extrinsic_matrix[:, :3, :3] = axis_angle_to_rotation(rotation)
    for i in range(N):
        extrinsic_matrix[i, :3, 3] = translation[i]
        extrinsic_matrix[i, 3, 3] = 1.0

    if ndim == 1:
        extrinsic_matrix = extrinsic_matrix.squeeze(0)

    return convert_extrinsics(
        extrinsic_matrix, src=convention, tgt=AxesConvention.OPENCV
    )


def intrinsics_from_parameters(
    focal_length: list[tuple[float, float]] | Tensor,
    principal_point: list[tuple[float, float]] | Tensor,
    orthographic: bool,
) -> Tensor:
    r"""
    Build the calibration matrix (K-matrix) using the focal lengths and
    principal points.

    Parameters
    ----------
    focal_length: list[tuple[float, float]] or Tensor
        List of tuples or tensor containing the focal lengths.
    principal_point: list[tuple[float, float]] or Tensor
        List of tuples or tensor containing the principal points.
    orthographic: bool
        Whether to use orthographic projection.

    Returns
    -------
    Tensor[*, 4, 4]
        Calibration matrix.

    Examples
    --------
    >>> focal_length = [(1000.0, 1000.0)]
    >>> principal_point = [(500.0, 500.0)]
    >>> K = intrinsics_from_parameters(
    ...     focal_length, principal_point, orthographic=False
    ... )
    >>> print(K)
    tensor([[1000.,    0.,  500.,    0.],
            [   0., 1000.,  500.,    0.],
            [   0.,    0.,    1.,    1.],
            [   0.,    0.,    0.,    1.]])
    """

    # Focal length, shape (), (N, 1) or (N, 2)
    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.as_tensor(focal_length).float()
    if focal_length.shape[-1] == 1:
        fx = fy = focal_length
    else:
        fx, fy = focal_length.unbind(-1)

    # Principal point
    if not isinstance(principal_point, torch.Tensor):
        principal_point = torch.as_tensor(principal_point).float()
    px, py = principal_point.unbind(-1)

    # Batched/unbatched
    ndim = focal_length.ndim
    assert ndim == principal_point.ndim
    if ndim == 1:
        fx = fx.unsqueeze(0)
        fy = fy.unsqueeze(0)
        px = px.unsqueeze(0)
        py = py.unsqueeze(0)

    # Amount of cameras
    N = fx.size(0)

    # Create calibration matrix
    K: Tensor = fx.new_zeros(N, 4, 4)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    if orthographic:
        K[:, 0, 3] = px
        K[:, 1, 3] = py
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0
    else:
        K[:, 0, 2] = px
        K[:, 1, 2] = py
        K[:, 3, 2] = 1.0
        K[:, 2, 3] = 1.0

    if ndim == 1:
        K = K.squeeze(0)

    return K


# @torch.fx.wrap
@torch.autocast("cuda", enabled=False)
def unsafe_inverse(x: Tensor) -> Tensor:
    r"""
    Compute the inverse of the input tensor. The input is cast to FP32 and back
    to the input datatype if it is not supported by the inversion function.

    Parameters
    ----------
    x: Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Inverse of the input tensor.

    Examples
    --------
    >>> x = torch.eye(3)
    >>> x_inv = unsafe_inverse(x)
    >>> print(x_inv)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    """
    x_inv = typing.cast(Tensor, torch.linalg.inv_ex(x.float()).inverse)  # type: ignore[attr-defined]
    return x_inv.type_as(x)


def rad2deg(tensor: Tensor) -> Tensor:
    r"""
    Function that converts angles from radians to degrees.

    Parameters
    ----------
    tensor: Tensor
        Tensor of arbitrary shape.

    Returns
    -------
    Tensor
        Tensor with same shape as input.

    Examples
    --------
    >>> input = torch.tensor(3.1415926535)
    >>> rad2deg(input)
    tensor(180.)
    """
    return 180.0 * tensor / torch.pi


def deg2rad(tensor: Tensor) -> Tensor:
    r"""
    Function that converts angles from degrees to radians.

    Parameters
    ----------
    tensor: Tensor
        Tensor of arbitrary shape.

    Returns
    -------
    Tensor
        Tensor with same shape as input.

    Examples
    --------
    >>> input = torch.tensor(180.0)
    >>> deg2rad(input)
    tensor(3.1416)
    """
    return tensor * torch.pi / 180.0


def pol2cart(rho: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Function that converts polar coordinates to cartesian coordinates.

    Parameters
    ----------
    rho: Tensor
        Tensor of arbitrary shape.
    phi: Tensor
        Tensor of same arbitrary shape.

    Returns
    -------
    tuple[Tensor, Tensor]
        - x: Tensor with same shape as input.
        - y: Tensor with same shape as input.

    Examples
    --------
    >>> rho = torch.rand(1, 3, 3)
    >>> phi = torch.rand(1, 3, 3)
    >>> x, y = pol2cart(rho, phi)
    """
    x = rho * phi.cos()
    y = rho * phi.sin()
    return x, y


def cart2pol(x: Tensor, y: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    r"""
    Function that converts cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x: Tensor
        Tensor of arbitrary shape.
    y: Tensor
        Tensor of same arbitrary shape.
    eps: float
        To avoid division by zero.

    Returns
    -------
    tuple[Tensor, Tensor]
        - rho: Tensor with same shape as input.
        - phi: Tensor with same shape as input.

    Examples
    --------
    >>> x = torch.rand(1, 3, 3)
    >>> y = torch.rand(1, 3, 3)
    >>> rho, phi = cart2pol(x, y)
    """
    rho = torch.sqrt(x**2 + y**2 + eps)
    phi = torch.atan2(y, x)
    return rho, phi


def euclidean_to_homogeneous_points(points: Tensor) -> Tensor:
    r"""
    Convert Euclidean points to homogeneous coordinates.

    Usually this is done for a set of 2D points before reprojecting them to 3D points.
    This simply entails adding a column of ones to the input points.

    Parameters
    ----------
    points: Tensor[*, N, D]
        Input points to homogenize.

    Returns
    -------
    Tensor[*, N, D+1]
        Homogeneous coordinates for each point.

    Examples
    --------
    >>> points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> homogeneous_points = euclidean_to_homogeneous_points(points)
    >>> print(homogeneous_points)
    tensor([[1., 2., 1.],
            [3., 4., 1.]])
    """
    return nn.functional.pad(points, (0, 1), "constant", value=1.0)


def homogeneous_to_euclidean_points(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""
    Converts homogeneous points to Euclidean space.

    Semantically, this inverts the result of :func:`_euclidean_to_homogeneous_points`.

    Parameters
    ----------
    points: Tensor[*, N, D+1]
        Points to convert.
    eps: float, optional
        Small value to avoid division by zero.

    Returns
    -------
    Tensor[*, N, D]
        Euclidean coordinates for each point.

    Examples
    --------
    >>> points = torch.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]])
    >>> euclidean_points = homogeneous_to_euclidean_points(points)
    >>> print(euclidean_points)
    tensor([[1., 2.],
            [3., 4.]])
    """

    z_vec = points[..., -1:]
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), 1.0)

    return scale * points[..., :-1]


def convert_affinematrix_to_homography(A: Tensor) -> Tensor:
    r"""
    Function that converts batch of affine matrices.

    Parameters
    ----------
    A: Tensor[*, 2, 3]
        Affine matrix.

    Returns
    -------
    Tensor[*, 3, 3]
        Homography matrix.

    Examples
    --------
    >>> A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> H = convert_affinematrix_to_homography(A)
    >>> print(H)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    """
    H: Tensor = nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def axis_angle_to_rotation(axis_angle: Tensor) -> Tensor:
    r"""
    Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.

    Parameters
    ----------
    axis_angle: Tensor[*, 3]
        Axis-angle rotations in radians.

    Returns
    -------
    Tensor[*, 3, 3]
        Rotation matrix

    Examples
    --------
    >>> input = torch.tensor([[0.0, 0.0, 0.0]])
    >>> axis_angle_to_rotation(input)
    tensor([[[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]])

    >>> input = torch.tensor([[1.5708, 0.0, 0.0]])
    >>> axis_angle_to_rotation(input)
    tensor([[[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
             [ 0.0000e+00, -3.6200e-06, -1.0000e+00],
             [ 0.0000e+00,  1.0000e+00, -3.6200e-06]]])
    """

    def _compute_rotation(
        axis_angle: Tensor, theta2: Tensor, eps: float = 1e-6
    ) -> Tensor:
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = axis_angle / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation.view(-1, 3, 3)

    def _compute_rotation_taylor(axis_angle: Tensor) -> Tensor:
        rx, ry, rz = torch.chunk(axis_angle, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation.view(-1, 3, 3)

    _axis_angle = torch.unsqueeze(axis_angle, dim=1)
    theta2 = torch.matmul(_axis_angle, _axis_angle.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # Compute rotation matrices
    rotation_normal = _compute_rotation(axis_angle, theta2)
    rotation_taylor = _compute_rotation_taylor(axis_angle)

    # Create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    # Create output pose matrix with masked values
    rotation = mask_pos * rotation_normal + mask_neg * rotation_taylor
    return rotation  # Nx3x3


def rotation_to_axis_angle(rotation: Tensor) -> Tensor:
    r"""
    Convert 3x3 rotation matrix to Rodrigues vector in radians.

    Parameters
    ----------
    rotation: Tensor[*, 3, 3]
        Rotation matrix.

    Returns
    -------
    Tensor[*, 3]
        Rodrigues vector transformation.

    Examples
    --------
    >>> input = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> rotation_to_axis_angle(input)
    tensor([0., 0., 0.])

    >>> input = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    >>> rotation_to_axis_angle(input)
    tensor([1.5708, 0.0000, 0.0000])
    """
    quaternion: Tensor = rotation_to_quaternion(rotation)
    return quaternion_to_axis_angle(quaternion)


def rotation_to_quaternion(rotation: Tensor, eps: float = 1e-8) -> Tensor:
    r"""
    Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) format.

    Parameters
    ----------
    rotation: Tensor[*, 3, 3]
        The rotation matrix to convert.
    eps: float
        Small value to avoid zero division.

    Returns
    -------
    Tensor[*, 4]
        The rotation in quaternion.

    Examples
    --------
    >>> rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> quaternion = rotation_to_quaternion(rotation)
    >>> print(quaternion)
    tensor([1., 0., 0., 0.])
    """

    def safe_zero_division(numerator: Tensor, denominator: Tensor) -> Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_vec: Tensor = rotation.reshape(*rotation.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_vec, chunks=9, dim=-1
    )

    trace: Tensor = m00 + m11 + m22

    def trace_positive_cond() -> Tensor:
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1() -> Tensor:
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2() -> Tensor:
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3() -> Tensor:
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(quaternion: Tensor, eps: float = 1e-12) -> Tensor:
    r"""
    Normalize a quaternion.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Parameters
    ----------
    quaternion: Tensor
        A tensor containing a quaternion to be normalized.
        The tensor can be of shape :math:`(*, 4)`.
    eps: float
        Small value to avoid division by zero.

    Returns
    -------
    Tensor
        The normalized quaternion of shape :math:`(*, 4)`.

    Examples
    --------
    >>> quaternion = torch.tensor((1.0, 0.0, 1.0, 0.0))
    >>> normalize_quaternion(quaternion)
    tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    return nn.functional.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def quaternion_to_rotation(quaternion: Tensor) -> Tensor:
    r"""
    Convert a quaternion to a rotation matrix.

    The quaternion should be in (w, x, y, z) format.

    Parameters
    ----------
    quaternion: Tensor
        A tensor containing a quaternion to be converted.
        The tensor can be of shape :math:`(*, 4)`.

    Returns
    -------
    Tensor[*, 3, 3]
        The rotation matrix.

    Examples
    --------
    >>> quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    >>> rotation = quaternion_to_rotation(quaternion)
    >>> print(rotation)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}"
        )

    # normalize the input quaternion
    quaternion_norm: Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w = quaternion_norm[..., 0]
    x = quaternion_norm[..., 1]
    y = quaternion_norm[..., 2]
    z = quaternion_norm[..., 3]

    # compute the actual conversion
    tx: Tensor = 2.0 * x
    ty: Tensor = 2.0 * y
    tz: Tensor = 2.0 * z
    twx: Tensor = tx * w
    twy: Tensor = ty * w
    twz: Tensor = tz * w
    txx: Tensor = tx * x
    txy: Tensor = ty * x
    txz: Tensor = tz * x
    tyy: Tensor = ty * y
    tyz: Tensor = tz * y
    tzz: Tensor = tz * z
    one: Tensor = torch.tensor(1.0, device=quaternion.device, dtype=quaternion.dtype)

    matrix_flat: Tensor = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    )

    # this slightly awkward construction of the output shape is to satisfy torchscript
    output_shape = [*list(quaternion.shape[:-1]), 3, 3]
    matrix = matrix_flat.reshape(output_shape)

    return matrix


def quaternion_to_axis_angle(quaternion: Tensor) -> Tensor:
    r"""
    Convert quaternion vector to axis angle of rotation in radians.

    The quaternion should be in (w, x, y, z) format.

    Parameters
    ----------
    quaternion: Tensor
        Tensor with quaternions.

    Returns
    -------
    Tensor
        Tensor with axis angle of rotation.

    Examples
    --------
    >>> quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    >>> axis_angle = quaternion_to_axis_angle(quaternion)
    >>> print(axis_angle)
    tensor([0., 0., 0.])
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(f"Input type is not a Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}"
        )

    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta: Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: Tensor = torch.sqrt(sin_squared_theta)
    two_theta: Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: Tensor = two_theta / sin_theta
    k_neg: Tensor = 2.0 * torch.ones_like(sin_theta)
    k: Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    axis_angle: Tensor = torch.zeros_like(quaternion)[..., :3]
    axis_angle[..., 0] += q1 * k
    axis_angle[..., 1] += q2 * k
    axis_angle[..., 2] += q3 * k
    return axis_angle


def quaternion_log_to_exp(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    r"""
    Apply exponential map to log quaternion.

    The quaternion should be in (w, x, y, z) format.

    Parameters
    ----------
    quaternion: Tensor
        A tensor containing a quaternion to be converted.
        The tensor can be of shape :math:`(*, 3)`.
    eps: float
        A small number for clamping.

    Returns
    -------
    Tensor
        The quaternion exponential map of shape :math:`(*, 4)`.

    Examples
    --------
    >>> quaternion = torch.tensor([0.0, 1.0, 0.0])
    >>> quaternion_exp = quaternion_log_to_exp(quaternion)
    >>> print(quaternion_exp)
    tensor([0.5403, 0.0000, 0.8415, 0.0000])
    """
    # compute quaternion norm
    norm_q: Tensor = torch.norm(quaternion, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # compute scalar and vector
    quaternion_vector: Tensor = quaternion * torch.sin(norm_q) / norm_q
    quaternion_scalar: Tensor = torch.cos(norm_q)

    # compose quaternion and return
    quaternion_exp = torch.cat((quaternion_scalar, quaternion_vector), dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    r"""
    Apply the log map to a quaternion.

    The quaternion should be in (w, x, y, z) format.

    Parameters
    ----------
    quaternion: Tensor
        A tensor containing a quaternion to be converted.
        The tensor can be of shape :math:`(*, 4)`.
    eps: float
        A small number for clamping.

    Returns
    -------
    Tensor
        The quaternion log map of shape :math:`(*, 3)`.

    Examples
    --------
    >>> quaternion = torch.tensor([0.5403, 0.0, 0.8415, 0.0])
    >>> quaternion_log = quaternion_exp_to_log(quaternion)
    >>> print(quaternion_log)
    tensor([0.0000, 1.0000, 0.0000])
    """
    # Unpack quaternion vector and scalar
    quaternion_scalar = quaternion[..., 0:1]
    quaternion_vector = quaternion[..., 1:4]

    # Compute quaternion norm
    norm_q: Tensor = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True).clamp(
        min=eps
    )

    # Apply log map
    quaternion_log: Tensor = (
        quaternion_vector
        * torch.acos(torch.clamp(quaternion_scalar, min=-1.0, max=1.0))
        / norm_q
    )

    return quaternion_log


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    r"""
    Convert an axis angle to a quaternion.

    The quaternion vector has components in (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Parameters
    ----------
    axis_angle: Tensor
        Tensor with axis angle in radians.

    Returns
    -------
    Tensor
        Tensor with quaternion.

    Examples
    --------
    >>> axis_angle = torch.tensor((0.0, 1.0, 0.0))
    >>> axis_angle_to_quaternion(axis_angle)
    tensor([0.8776, 0.0000, 0.4794, 0.0000])
    """
    if not torch.is_tensor(axis_angle):
        raise TypeError(f"Input type is not a Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(
            f"Input must be a tensor of shape Nx3 or 3. Got {axis_angle.shape}"
        )

    # unpack input and compute conversion
    a0: Tensor = axis_angle[..., 0:1]
    a1: Tensor = axis_angle[..., 1:2]
    a2: Tensor = axis_angle[..., 2:3]
    theta_squared: Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: Tensor = torch.sqrt(theta_squared)
    half_theta: Tensor = theta * 0.5

    mask: Tensor = theta_squared > 0.0
    ones: Tensor = torch.ones_like(half_theta)

    k_neg: Tensor = 0.5 * ones
    k_pos: Tensor = torch.sin(half_theta) / theta
    k: Tensor = torch.where(mask, k_pos, k_neg)
    w: Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: Tensor = torch.zeros(
        size=(*axis_angle.shape[:-1], 4),
        dtype=axis_angle.dtype,
        device=axis_angle.device,
    )
    quaternion[..., 1:2] = a0 * k
    quaternion[..., 2:3] = a1 * k
    quaternion[..., 3:4] = a2 * k
    quaternion[..., 0:1] = w
    return quaternion


def euler_from_quaternion(
    w: Tensor, x: Tensor, y: Tensor, z: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Convert a quaternion coefficients to Euler angles.

    Returned angles are in radians in XYZ convention.

    Parameters
    ----------
    w: Tensor
        Quaternion :math:`q_w` coefficient.
    x: Tensor
        Quaternion :math:`q_x` coefficient.
    y: Tensor
        Quaternion :math:`q_y` coefficient.
    z: Tensor
        Quaternion :math:`q_z` coefficient.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A tuple with euler angles `roll`, `pitch`, `yaw`.

    Examples
    --------
    >>> w, x, y, z = (
    ...     torch.tensor(1.0),
    ...     torch.tensor(0.0),
    ...     torch.tensor(0.0),
    ...     torch.tensor(0.0),
    ... )
    >>> roll, pitch, yaw = euler_from_quaternion(w, x, y, z)
    >>> print(roll, pitch, yaw)
    tensor(0.) tensor(0.) tensor(0.)
    """
    yy = y * y

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + yy)
    roll = sinr_cosp.atan2(cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(min=-1.0, max=1.0)
    pitch = sinp.asin()

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (yy + z * z)
    yaw = siny_cosp.atan2(cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(
    roll: Tensor, pitch: Tensor, yaw: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Convert Euler angles to quaternion coefficients.

    Euler angles are assumed to be in radians in XYZ convention.

    Parameters
    ----------
    roll: Tensor
        The roll euler angle.
    pitch: Tensor
        The pitch euler angle.
    yaw: Tensor
        The yaw euler angle.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple with quaternion coefficients in order of `wxyz`.

    Examples
    --------
    >>> roll, pitch, yaw = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    >>> w, x, y, z = quaternion_from_euler(roll, pitch, yaw)
    >>> print(w, x, y, z)
    tensor(1.) tensor(0.) tensor(0.) tensor(0.)
    """
    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    yaw_half = yaw * 0.5

    cy = yaw_half.cos()
    sy = yaw_half.sin()
    cp = pitch_half.cos()
    sp = pitch_half.sin()
    cr = roll_half.cos()
    sr = roll_half.sin()

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return qw, qx, qy, qz


def normalize_pixel_coordinates(
    pixel_coordinates: Tensor, height: int, width: int, eps: float = 1e-8
) -> Tensor:
    r"""
    Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Parameters
    ----------
    pixel_coordinates: Tensor
        The grid with pixel coordinates. Size can be :math:`(*, 2)`.
    width: int
        The maximum width in the x-axis.
    height: int
        The maximum height in the y-axis.
    eps: float
        Safe division by zero.

    Returns
    -------
    Tensor[*, 2]
        Normalized pixel coordinates with shape :math:`(*, 2)`.

    Examples
    --------
    >>> pixel_coordinates = torch.tensor([[0.0, 0.0]])
    >>> normalized_coordinates = normalize_pixel_coordinates(pixel_coordinates, 100, 50)
    >>> print(normalized_coordinates)
    tensor([[-1., -1.]])
    """
    dtype = pixel_coordinates.dtype
    with pixel_coordinates.device:
        hw: Tensor = torch.stack(
            [
                torch.tensor(width, dtype=dtype),
                torch.tensor(height, dtype=dtype),
            ]
        )
        factor: Tensor = torch.tensor(2.0, dtype=dtype) / (hw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates(
    pixel_coordinates: Tensor, height: int, width: int, eps: float = 1e-8
) -> Tensor:
    r"""
    Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1).

    Parameters
    ----------
    pixel_coordinates: Tensor
        The normalized grid coordinates. Size can be :math:`(*, 2)`.
    width: int
        The maximum width in the x-axis.
    height: int
        The maximum height in the y-axis.
    eps: float
        Safe division by zero.

    Returns
    -------
    Tensor[*, 2]
        The denormalized pixel coordinates.

    Examples
    --------
    >>> coords = torch.tensor([[-1.0, -1.0]])
    >>> denormalized_coords = denormalize_pixel_coordinates(coords, 100, 50)
    >>> print(denormalized_coords)
    tensor([[0., 0.]])
    """
    dtype = pixel_coordinates.dtype
    with pixel_coordinates.device:
        hw: Tensor = torch.stack(
            [torch.tensor(width, dtype=dtype), torch.tensor(height, dtype=dtype)]
        )

    factor: Tensor = torch.tensor(2.0) / (hw - 1).clamp(eps)

    return 1.0 / factor * (pixel_coordinates + 1)


def normalize_pixel_coordinates3d(
    pixel_coordinates: Tensor, depth: int, height: int, width: int, eps: float = 1e-8
) -> Tensor:
    r"""
    Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Parameters
    ----------
    pixel_coordinates: Tensor
        The grid with pixel coordinates. Size can be :math:`(*, 3)`.
    depth: int
        The maximum depth in the z-axis.
    height: int
        The maximum height in the y-axis.
    width: int
        The maximum width in the x-axis.
    eps: float
        Safe division by zero.

    Returns
    -------
    Tensor[*, 3]
        The normalized pixel coordinates.

    Examples
    --------
    >>> pixel_coordinates = torch.tensor([[0.0, 0.0, 0.0]])
    >>> normalized_coordinates = normalize_pixel_coordinates3d(
    ...     pixel_coordinates, 10, 100, 50
    ... )
    >>> print(normalized_coordinates)
    tensor([[-1., -1., -1.]])
    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(
            f"Input pixel_coordinates must be of shape (*, 3). Got {pixel_coordinates.shape}"
        )
    # compute normalization factor
    dhw: Tensor = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: Tensor = 2.0 / (dhw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates3d(
    pixel_coordinates: Tensor, depth: int, height: int, width: int, eps: float = 1e-8
) -> Tensor:
    r"""
    Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1).

    Parameters
    ----------
    pixel_coordinates: Tensor
        The normalized grid coordinates. Size can be :math:`(*, 3)`.
    depth: int
        The maximum depth in the z-axis.
    height: int
        The maximum height in the y-axis.
    width: int
        The maximum width in the x-axis.
    eps: float
        Safe division by zero.

    Returns
    -------
    Tensor[*, 3]
        The denormalized pixel coordinates.

    Examples
    --------
    >>> coords = torch.tensor([[-1.0, -1.0, -1.0]])
    >>> denormalized_coords = denormalize_pixel_coordinates3d(coords, 10, 100, 50)
    >>> print(denormalized_coords)
    tensor([[0., 0., 0.]])
    """
    dtype = pixel_coordinates.dtype
    with pixel_coordinates.device:
        dhw = torch.stack(
            [
                torch.tensor(depth, dtype=dtype),
                torch.tensor(width, dtype=dtype),
                torch.tensor(height, dtype=dtype),
            ]
        )
    factor: Tensor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


def angle_to_rotation(angle: Tensor) -> Tensor:
    r"""
    Create a rotation matrix out of angles in degrees.

    Parameters
    ----------
    angle: Tensor
        Tensor of angles in degrees, any shape :math:`(*)`.

    Returns
    -------
    Tensor[*, 2, 2]
        Tensor of rotation matrices.

    Examples
    --------
    >>> input = torch.tensor([45.0])
    >>> rotation_matrix = angle_to_rotation(input)
    >>> print(rotation_matrix)
    tensor([[[ 0.7071,  0.7071],
             [-0.7071,  0.7071]]])
    """
    ang_rad = deg2rad(angle)
    cos_a: Tensor = torch.cos(ang_rad)
    sin_a: Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def normalize_homography(
    dst_pix_trans_src_pix: Tensor,
    dsize_src: tuple[int, int],
    dsize_dst: tuple[int, int],
) -> Tensor:
    r"""
    Normalize a given homography in pixels to [-1, 1].

    Parameters
    ----------
    dst_pix_trans_src_pix: Tensor[*, 3, 3]
        Homography/ies from source to destination to be normalized.
    dsize_src: tuple[int, int]
        Size of the source image (height, width).
    dsize_dst: tuple[int, int]
        Size of the destination image (height, width).

    Returns
    -------
    Tensor[*, 3, 3]
        The normalized homography.

    Examples
    --------
    >>> dst_pix_trans_src_pix = torch.eye(3)
    >>> dsize_src = (100, 100)
    >>> dsize_dst = (200, 200)
    >>> normalized_homography = normalize_homography(
    ...     dst_pix_trans_src_pix, dsize_src, dsize_dst
    ... )
    >>> print(normalized_homography)
    tensor([[ 2.0000,  0.0000, -1.0000],
            [ 0.0000,  2.0000, -1.0000],
            [ 0.0000,  0.0000,  1.0000]])
    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(
            f"Input type is not a Tensor. Got {type(dst_pix_trans_src_pix)}"
        )

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}"
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )

    src_pix_trans_src_norm = unsafe_inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )

    # compute chain transformations
    dst_norm_trans_src_norm: Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""
    Compute the normalization matrix from image size in pixels to [-1, 1].

    Parameters
    ----------
    height: int
        Image height.
    width: int
        Image width.
    eps: float
        Epsilon to prevent divide-by-zero errors.
    device: torch.device, optional
        Device to use.
    dtype: torch.dtype, optional
        Data type to use.

    Returns
    -------
    Tensor[1, 3, 3]
        Normalized transform.

    Examples
    --------
    >>> height, width = 100, 100
    >>> transform = normal_transform_pixel(height, width)
    >>> print(transform)
    tensor([[[ 0.0200,  0.0000, -1.0000],
             [ 0.0000,  0.0200, -1.0000],
             [ 0.0000,  0.0000,  1.0000]]])
    """
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normal_transform_pixel3d(
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-14,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""
    Compute the normalization matrix from image size in pixels to [-1, 1].

    Parameters
    ----------
    depth: int
        Image depth.
    height: int
        Image height.
    width: int
        Image width.
    eps: float
        Epsilon to prevent divide-by-zero errors.
    device: torch.device, optional
        Device to use.
    dtype: torch.dtype, optional
        Data type to use.

    Returns
    -------
    Tensor[1, 4, 4]
        Normalized transform.

    Examples
    --------
    >>> depth, height, width = 10, 100, 100
    >>> transform = normal_transform_pixel3d(depth, height, width)
    >>> print(transform)
    tensor([[[ 0.2000,  0.0000,  0.0000, -1.0000],
             [ 0.0000,  0.0200,  0.0000, -1.0000],
             [ 0.0000,  0.0000,  0.0200, -1.0000],
             [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    """
    tr_mat = unsafe_inverse(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
    )  # 4x4

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0
    depth_denom: float = eps if depth == 1 else depth - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    tr_mat[2, 2] = tr_mat[2, 2] * 2.0 / depth_denom

    return tr_mat.unsqueeze(0)  # 1x4x4


def denormalize_homography(
    dst_pix_trans_src_pix: Tensor,
    dsize_src: tuple[int, int],
    dsize_dst: tuple[int, int],
) -> Tensor:
    r"""
    De-normalize a given homography in pixels from [-1, 1] to actual height and width.

    Parameters
    ----------
    dst_pix_trans_src_pix: Tensor[*, 3, 3]
        Homography/ies from source to destination to be denormalized.
    dsize_src: tuple[int, int]
        Size of the source image (height, width).
    dsize_dst: tuple[int, int]
        Size of the destination image (height, width).

    Returns
    -------
    Tensor[*, 3, 3]
        The denormalized homography.

    Examples
    --------
    >>> dst_pix_trans_src_pix = torch.eye(3)
    >>> dsize_src = (100, 100)
    >>> dsize_dst = (200, 200)
    >>> denormalized_homography = denormalize_homography(
    ...     dst_pix_trans_src_pix, dsize_src, dsize_dst
    ... )
    >>> print(denormalized_homography)
    tensor([[ 0.5000,  0.0000,  0.5000],
            [ 0.0000,  0.5000,  0.5000],
            [ 0.0000,  0.0000,  1.0000]])
    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(
            f"Input type is not a Tensor. Got {type(dst_pix_trans_src_pix)}"
        )

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}"
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )

    dst_norm_trans_dst_pix: Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )
    dst_denorm_trans_dst_pix = unsafe_inverse(dst_norm_trans_dst_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: Tensor = dst_denorm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_norm_trans_src_pix
    )
    return dst_norm_trans_src_norm


def normalize_homography3d(
    dst_pix_trans_src_pix: Tensor,
    dsize_src: tuple[int, int, int],
    dsize_dst: tuple[int, int, int],
) -> Tensor:
    r"""
    Normalize a given homography in pixels to [-1, 1].

    Parameters
    ----------
    dst_pix_trans_src_pix: Tensor[*, 4, 4]
        Homography/ies from source to destination to be normalized.
    dsize_src: tuple[int, int, int]
        Size of the source image (depth, height, width).
    dsize_dst: tuple[int, int, int]
        Size of the destination image (depth, height, width).

    Returns
    -------
    Tensor[*, 4, 4]
        The normalized homography.

    Examples
    --------
    >>> dst_pix_trans_src_pix = torch.eye(4)
    >>> dsize_src = (10, 100, 100)
    >>> dsize_dst = (20, 200, 200)
    >>> normalized_homography = normalize_homography3d(
    ...     dst_pix_trans_src_pix, dsize_src, dsize_dst
    ... )
    >>> print(normalized_homography)
    tensor([[ 2.0000,  0.0000,  0.0000, -1.0000],
            [ 0.0000,  2.0000,  0.0000, -1.0000],
            [ 0.0000,  0.0000,  2.0000, -1.0000],
            [ 0.0000,  0.0000,  0.0000,  1.0000]])
    """
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(
            f"Input type is not a Tensor. Got {type(dst_pix_trans_src_pix)}"
        )

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (4, 4)
    ):
        raise ValueError(
            f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}"
        )

    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    dst_d, dst_h, dst_w = dsize_dst
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: Tensor = normal_transform_pixel3d(src_d, src_h, src_w).to(
        dst_pix_trans_src_pix
    )

    src_pix_trans_src_norm = unsafe_inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: Tensor = normal_transform_pixel3d(dst_d, dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )
    # compute chain transformations
    dst_norm_trans_src_norm: Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def normalize_points_with_intrinsics(point_2d: Tensor, camera_matrix: Tensor) -> Tensor:
    r"""
    Normalizes points with intrinsics. Useful for conversion of keypoints to be used with essential matrix.

    Parameters
    ----------
    point_2d: Tensor[*, 2]
        Points in the image pixel coordinates. The shape of the tensor can be :math:`(*, 2)`.
    camera_matrix: Tensor[*, 3, 3] or Tensor[*, 4, 4]
        Tensor containing the intrinsics camera matrix.

    Returns
    -------
    Tensor[*, 2]
        Tensor of (u, v) cam coordinates.

    Examples
    --------
    >>> point_2d = torch.tensor([[100.0, 200.0]])
    >>> camera_matrix = torch.tensor(
    ...     [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]
    ... )
    >>> normalized_points = normalize_points_with_intrinsics(point_2d, camera_matrix)
    >>> print(normalized_points)
    tensor([[-0.4000, -0.3000]])
    """
    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: Tensor = point_2d[..., 0]
    v_coord: Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: Tensor = camera_matrix[..., 0, 0]
    fy: Tensor = camera_matrix[..., 1, 1]
    cx: Tensor = camera_matrix[..., 0, 2]
    cy: Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: Tensor = (u_coord - cx) / fx
    y_coord: Tensor = (v_coord - cy) / fy

    xy: Tensor = torch.stack([x_coord, y_coord], dim=-1)
    return xy


def denormalize_points_with_intrinsics(
    point_2d_norm: Tensor, camera_matrix: Tensor
) -> Tensor:
    r"""
    Normalizes points with intrinsics.
    Useful for conversion of keypoints to be used with essential matrix.

    Parameters
    ----------
    point_2d_norm: Tensor[*, 2]
        Tensor containing the 2d points in the image pixel coordinates. The shape of the tensor can be :math:`(*, 2)`.
    camera_matrix: Tensor[*, 3, 3] or Tensor[*, 4, 4]
        Tensor containing the intrinsics camera matrix.

    Returns
    -------
    Tensor[*, 2]
        Tensor of (u, v) cam coordinates.

    Examples
    --------
    >>> point_2d_norm = torch.tensor([[-0.4000, -0.3000]])
    >>> camera_matrix = torch.tensor(
    ...     [[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]]
    ... )
    >>> denormalized_points = denormalize_points_with_intrinsics(
    ...     point_2d_norm, camera_matrix
    ... )
    >>> print(denormalized_points)
    tensor([[100., 200.]])
    """
    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X + cx
    # v = fy * Y + cy

    # unpack coordinates
    x_coord: Tensor = point_2d_norm[..., 0]
    y_coord: Tensor = point_2d_norm[..., 1]

    # unpack intrinsics
    fx: Tensor = camera_matrix[..., 0, 0]
    fy: Tensor = camera_matrix[..., 1, 1]
    cx: Tensor = camera_matrix[..., 0, 2]
    cy: Tensor = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    u_coord: Tensor = x_coord * fx + cx
    v_coord: Tensor = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def motion_to_extrinsics(R: Tensor, t: Tensor) -> Tensor:
    r"""
    Convert rotation matrix and translation vector to extrinsics matrix.

    Parameters
    ----------
    R: Tensor[*, 3, 3]
        Rotation matrix.
    t: Tensor[*, 3, 1]
        Translation vector.

    Returns
    -------
    Tensor[*, 4, 4]
        Extrinsic matrix.

    Examples
    --------
    >>> R = torch.eye(3)
    >>> t = torch.tensor([[0.0], [0.0], [0.0]])
    >>> extrinsics = motion_to_extrinsics(R, t)
    >>> print(extrinsics)
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    """
    Rt = torch.cat([R, t], dim=2)
    return convert_affinematrix_to_homography(Rt)


def extrinsics_to_motion(extrinsics: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Extracts rotation matrix and translation vector from extrinsics matrix.

    Parameters
    ----------
    extrinsics: Tensor[*, 4, 4]
        Pose matrix.

    Returns
    -------
    tuple[Tensor, Tensor]
        - Rotation matrix, :math:`(*, 3, 3)`.
        - Translation matrix :math:`(*, 3, 1)`.

    Examples
    --------
    >>> extrinsics = torch.eye(4)
    >>> R, t = extrinsics_to_motion(extrinsics)
    >>> print(R)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> print(t)
    tensor([[0.],
            [0.],
            [0.]])
    """
    return extrinsics[..., :3, :3], extrinsics[..., :3, 3:]


def reverse_motion(R: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Reverses a camera-to-world transformation to world-to-camera transformation, or
    vice versa.

    Parameters
    ----------
    R: Tensor[*, 3, 3]
        Rotation matrix.
    t: Tensor[*, 3, 1]
        Translation matrix.

    Returns
    -------
    tuple[Tensor, Tensor]
        - Rotation matrix, :math:`(*, 3, 3)`.
        - Translation matrix :math:`(*, 3, 1)`.

    Examples
    --------
    >>> R = torch.eye(3)
    >>> t = torch.tensor([[0.0], [0.0], [0.0]])
    >>> R_inv, t_inv = reverse_motion(R, t)
    >>> print(R_inv)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> print(t_inv)
    tensor([[0.],
            [0.],
            [0.]])
    """

    R_inv = R.transpose(1, 2)
    new_t: Tensor = -R_inv @ t

    return (R_inv, new_t)


def reverse_extrinsics(extrinsics: Tensor) -> Tensor:
    r"""
    Reverses a camera-to-world transformation to world-to-camera transformation, or
    vice versa.

    Parameters
    ----------
    extrinsics: Tensor[*, 4, 4]
        Pose matrix.

    Returns
    -------
    Tensor[*, 4, 4]
        Pose matrix.

    Examples
    --------
    >>> extrinsics = torch.eye(4)
    >>> reversed_extrinsics = reverse_extrinsics(extrinsics)
    >>> print(reversed_extrinsics)
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    """
    return motion_to_extrinsics(*reverse_motion(*extrinsics_to_motion(extrinsics)))


def vector_to_skew_symmetric_matrix(vec: Tensor) -> Tensor:
    r"""
    Converts a vector to a skew symmetric matrix.

    A vector :math:`(v1, v2, v3)` has a corresponding skew-symmetric matrix, which is of the form:

    .. math::
        \begin{bmatrix} 0 & -v3 & v2 \\
        v3 & 0 & -v1 \\
        -v2 & v1 & 0\end{bmatrix}

    Parameters
    ----------
    vec: Tensor[*, 3]
        Input vector.

    Returns
    -------
    Tensor[*, 3, 3]
        The skew-symmetric matrix.

    Examples
    --------
    >>> vec = torch.tensor([1.0, 2.0, 3.0])
    >>> skew_symmetric_matrix = vector_to_skew_symmetric_matrix(vec)
    >>> print(skew_symmetric_matrix)
    tensor([[ 0., -3.,  2.],
            [ 3.,  0., -1.],
            [-2.,  1.,  0.]])
    """

    v1, v2, v3 = vec[..., 0], vec[..., 1], vec[..., 2]
    zeros = torch.zeros_like(v1)
    return torch.stack(
        [
            torch.stack([zeros, -v3, v2], dim=-1),
            torch.stack([v3, zeros, -v1], dim=-1),
            torch.stack([-v2, v1, zeros], dim=-1),
        ],
        dim=-2,
    )


####################################################
# Various helper methods for geometric projections #
####################################################


def apply_points(transform: Tensor, points: Tensor) -> Tensor:
    r"""
    Apply a transformation on a set of points.

    Parameters
    ----------
    transform: Tensor[..., D+1, D+1]
        Transformation matrix.
    points: Tensor[..., N, D]
        Points to transform.

    Returns
    -------
    Tensor[*, N, D]
        Transformed points.

    Examples
    --------
    >>> transform = torch.eye(4)
    >>> points = torch.tensor([[1.0, 2.0, 3.0]])
    >>> transformed_points = apply_points(transform, points)
    >>> print(transformed_points)
    tensor([[1., 2., 3.]])
    """

    batch_size = points.shape[:-2]
    # Accept variable number of dimensions. This reshape operation combines
    # both unsqueezing (in case of no batch) and flattening (in case of multiple
    # leading dimensions).
    points = points.reshape(-1, points.shape[-2], points.shape[-1])
    transform = transform.reshape(-1, transform.shape[-2], transform.shape[-1])

    # Repeat the transform matrix (which may be unit) to match the number of points
    transform = torch.repeat_interleave(
        transform, repeats=points.shape[0] // transform.shape[0], dim=0
    )

    points_homo = euclidean_to_homogeneous_points(points)

    result_homo = torch.bmm(points_homo, transform.permute(0, 2, 1))
    result_homo = torch.squeeze(result_homo, dim=-1)
    result = homogeneous_to_euclidean_points(result_homo)
    return result.unflatten(0, batch_size)


#######################
# Rendering utilities #
#######################


@torch.no_grad()
def generate_fovmap(
    focal_length: Tensor, principal_point: Tensor, like: Size | Tensor
) -> Tensor:
    r"""
    Encode the camera intrinsic parameters (focal length and principle point) to a map
    with channels for the distance to the principle point and the field of view.

    Parameters
    ----------
    focal_length: Tensor
        Focal length of the camera.
    principal_point: Tensor
        Principal point of the camera.
    like: Size or Tensor
        Size of the output map.

    Returns
    -------
    Tensor
        Field of view map.

    Examples
    --------
    >>> focal_length = torch.tensor([1000.0, 1000.0])
    >>> principal_point = torch.tensor([500.0, 500.0])
    >>> like = (100, 100)
    >>> fovmap = generate_fovmap(focal_length, principal_point, like)
    >>> print(fovmap.shape)
    torch.Size([100, 100, 4])
    """
    if isinstance(like, torch.Tensor):
        height, width = like.shape[-2:]
    else:
        height, width = like[-2:]

    batch_shape = focal_length.shape[:-1]
    assert principal_point.shape == focal_length.shape

    if len(batch_shape) == 0:
        principal_point = principal_point.unsqueeze(0)
        focal_length = focal_length.unsqueeze(0)
    elif len(batch_shape) > 1:
        focal_length = focal_length.flatten(0, -2)
        principal_point = principal_point.flatten(0, -2)

    device = focal_length.device
    dtype = focal_length.dtype if focal_length.is_floating_point() else torch.float32

    results = []
    for focal_length, principal_point in zip(
        focal_length, principal_point, strict=False
    ):
        fx, fy = focal_length.unbind(-1)
        u0, v0 = principal_point.unbind(-1)

        # Distance to principal point
        x_row = torch.arange(0, width, dtype=dtype, device=device)
        x_row_center_norm = (x_row - u0) / (width * 2)
        x_center = torch.tile(x_row_center_norm, (height, 1))  # [H, W]

        y_col = torch.arange(0, height, dtype=dtype, device=device)
        y_col_center_norm = (y_col - v0) / (height * 2)
        y_center = torch.tile(y_col_center_norm, (width, 1)).T

        # FoV
        fov_x = torch.arctan(x_center / (fx / width))
        fov_y = torch.arctan(y_center / (fy / height))

        res = torch.stack([x_center, y_center, fov_x, fov_y], dim=2)
        results.append(res)

    res = torch.stack(results)

    return res.reshape(*batch_shape, height, width, 4)

@torch.no_grad()
def generate_directions_on_grid(
    transform: Tensor,
    coords: Tensor,
    *,
    normalize: bool = True,
    flatten: bool = True,
) -> Tensor:
    r"""
    Generate directions for each pixel in the grid.

    Parameters
    ----------
    transform: Tensor
        Transformation matrix. Shape (B, 4, 4).
    grid: Tensor
        Grid of pixel coordinates. Shape (H, W, 2).
    normalize: bool
        Whether to normalize the directions.
    flatten: bool
        Whether to flatten the output tensor.

    Returns
    -------
    Tensor
        Directions for each pixel.
    """
    canvas_size = coords.shape[:-1]

    coords = coords.flatten(0, 1)  # (H*W, 2)
    coords = euclidean_to_homogeneous_points(coords)  # (H*W, 3)
    coords = coords.repeat(transform.shape[0], 1, 1)  # (B, H*W, 3)

    proj = unsafe_inverse(transform)  # (B, 4, 4)
    dirs = apply_points(proj, coords)  # (B, H*W, 3)
    if normalize:
        dirs = nn.functional.normalize(dirs.mT, dim=-2).mT
    return dirs if flatten else dirs.unflatten(-2, canvas_size)

@torch.no_grad()
def generate_directions(
    transform: Tensor,
    canvas_size: tuple[int, int],
    *,
    grid_mode: GridMode | str = GridMode.PIXEL_CENTER,
    normalize: bool = True,
    flatten: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""
    Generate directions for each pixel in the canvas.

    Parameters
    ----------
    transform: Tensor
        Transformation matrix.
    canvas_size: tuple[int, int]
        Size of the canvas.
    grid_mode: GridMode
        Grid generator mode. By default pixel center.
    normalize: bool
        Whether to normalize the directions.
    flatten: bool
        Whether to flatten the output tensor.

    Returns
    -------
    Tensor
        Directions for each pixel.

    Examples
    --------
    >>> transform = torch.eye(4)
    >>> canvas_size = (100, 100)
    >>> directions = generate_directions(transform, canvas_size)
    >>> print(directions.shape)
    torch.Size([10000, 3])
    """

    # Generate the coordinate grid (H, W, 2)
    if device is None and dtype is None:
        coords = generate_coord_grid_as(transform, canvas_size, mode=grid_mode)
    else:
        if device is None:
            device = transform.device
        if dtype is None:
            dtype = transform.dtype
        coords = generate_coord_grid(
            canvas_size,
            device=device,
            dtype=dtype,
            mode=grid_mode,
        )

    # Generate the directions for each pixel on the grid
    return generate_directions_on_grid(
        transform, coords, normalize=normalize, flatten=flatten
    )


def directions_to_angles(dirs: Tensor) -> Tensor:
    r"""
    Convert directions to angles.

    Parameters
    ----------
    dirs: Tensor
        Directions tensor.

    Returns
    -------
    Tensor
        Angles tensor.

    Examples
    --------
    >>> dirs = torch.tensor([[1.0, 0.0, 0.0]])
    >>> angles = directions_to_angles(dirs)
    >>> print(angles)
    tensor([[1.5708, 1.5708]])
    """
    theta = torch.atan2(dirs[..., 0], dirs[..., -1])
    phi = torch.acos(dirs[..., 1])
    return torch.stack([theta, phi], dim=-1)


def spherical_zbuffer_to_euclidean(spherical_tensor: Tensor) -> Tensor:
    r"""
    Convert spherical coordinates with z-buffer depth to Euclidean coordinates.

    Parameters
    ----------
    spherical_tensor: Tensor
        Spherical coordinates tensor.

    Returns
    -------
    Tensor
        Euclidean coordinates tensor.

    Examples
    --------
    >>> spherical_tensor = torch.tensor([[1.5708, 1.5708, 1.0]])
    >>> euclidean_tensor = spherical_zbuffer_to_euclidean(spherical_tensor)
    >>> print(euclidean_tensor)
    tensor([[1.0000, 0.0000, 1.0000]])
    """
    theta, phi, z = spherical_tensor.unbind(-1)  # polar, azim, depth

    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def spherical_to_euclidean(spherical_tensor: Tensor) -> Tensor:
    r"""
    Convert spherical coordinates to Euclidean coordinates.

    Parameters
    ----------
    spherical_tensor: Tensor
        Spherical coordinates tensor.

    Returns
    -------
    Tensor
        Euclidean coordinates tensor.

    Examples
    --------
    >>> spherical_tensor = torch.tensor([[1.5708, 1.5708, 1.0]])
    >>> euclidean_tensor = spherical_to_euclidean(spherical_tensor)
    >>> print(euclidean_tensor)
    tensor([[1.0000, 0.0000, 0.0000]])
    """
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    r = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    x = r * torch.sin(phi) * torch.sin(theta)
    y = r * torch.cos(phi)
    z = r * torch.cos(theta) * torch.sin(phi)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def euclidean_to_spherical(spherical_tensor: Tensor) -> Tensor:
    r"""
    Convert Euclidean coordinates to spherical coordinates.

    Parameters
    ----------
    spherical_tensor: Tensor
        Euclidean coordinates tensor.

    Returns
    -------
    Tensor
        Spherical coordinates tensor.

    Examples
    --------
    >>> euclidean_tensor = torch.tensor([[1.0, 0.0, 0.0]])
    >>> spherical_tensor = euclidean_to_spherical(euclidean_tensor)
    >>> print(spherical_tensor)
    tensor([[1.5708, 1.5708, 1.0000]])
    """
    x = spherical_tensor[..., 0]  # polar angle
    y = spherical_tensor[..., 1]  # azimuth angle
    z = spherical_tensor[..., 2]  # radius
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(x / r, z / r)
    phi = torch.acos(y / r)

    euclidean_tensor = torch.stack((theta, phi, r), dim=-1)
    return euclidean_tensor


def euclidean_to_spherical_zbuffer(euclidean_tensor: Tensor) -> Tensor:
    r"""
    Convert Euclidean coordinates to spherical coordinates with z-buffer depth.

    Parameters
    ----------
    euclidean_tensor: Tensor
        Euclidean coordinates tensor.

    Returns
    -------
    Tensor
        Spherical coordinates tensor.

    Examples
    --------
    >>> euclidean_tensor = torch.tensor([[1.0, 0.0, 1.0]])
    >>> spherical_tensor = euclidean_to_spherical_zbuffer(euclidean_tensor)
    >>> print(spherical_tensor)
    tensor([[0.0000, 1.5708, 1.0000]])
    """
    pitch = torch.asin(euclidean_tensor[..., 1])
    yaw = torch.atan2(euclidean_tensor[..., 0], euclidean_tensor[..., -1])
    z = euclidean_tensor[..., 2]  # Extract zbuffer depth
    euclidean_tensor = torch.stack((pitch, yaw, z), dim=-1)
    return euclidean_tensor
