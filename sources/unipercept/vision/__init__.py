r"""
This module contains various implementations of computer vision algorithms,
conventions, and utilities.

Submodules
----------
- `coord`: Coordinate transformations and grid generation utilities.
- `filter`: Image filtering and convolution operations.
- `geometry`: Geometric transformations and conversions.
- `knn_points`: K-Nearest Neighbors (KNN) operations for point clouds.
- `pointrend`: Point sampling and selection methods.
- `inpaint`: Image inpainting and completion methods.

Examples
--------
Here are some examples of how to use the submodules:

1. Coordinate transformations and grid generation:
    >>> from unipercept.vision.coord import generate_coord_grid, GridMode
    >>> grid = generate_coord_grid((5, 5), mode=GridMode.PIXEL_CENTER)
    >>> print(grid)

2. Image filtering and convolution operations:
    >>> import torch
    >>> from unipercept.vision.filter import filter2d, get_box_kernel2d
    >>> input = torch.rand(1, 1, 5, 5)
    >>> kernel = get_box_kernel2d((3, 3))
    >>> output = filter2d(input, kernel)
    >>> print(output)

3. Geometric transformations and conversions:
    >>> import torch
    >>> from unipercept.vision.geometry import (
    ...     axis_angle_to_rotation,
    ...     rotation_to_axis_angle,
    ... )
    >>> axis_angle = torch.tensor([0.0, 1.0, 0.0])
    >>> rotation_matrix = axis_angle_to_rotation(axis_angle)
    >>> print(rotation_matrix)
    >>> axis_angle_converted = rotation_to_axis_angle(rotation_matrix)
    >>> print(axis_angle_converted)

5. Point sampling and selection methods:
    >>> import torch
    >>> from unipercept.vision.pointrend import random_points
    >>> source = torch.rand(1, 1, 5, 5)
    >>> points = random_points(source, n_points=10)
    >>> print(points)

6. Image inpainting and completion methods:
    >>> import torch
    >>> from unipercept.vision.inpaint import inpaint2d
    >>> input = torch.rand(1, 1, 5, 5)
    >>> mask = torch.zeros(1, 1, 5, 5)
    >>> mask[:, :, 1:4, 1:4] = 1
    >>> output = inpaint2d(input, mask)
    >>> print(output)
"""

from . import coord, filter, geometry, inpaint, pointrend
