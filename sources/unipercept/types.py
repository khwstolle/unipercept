r"""
Defines commin typings used throughout all submodules.
"""

import datetime
import os
import pathlib

import torch
import torch.nn
import torch.types

type Tensor = torch.Tensor
type Device = torch.device | torch.types.Device
type DType = torch.dtype
type StateDict = dict[str, Tensor]
type Size = torch.Size | tuple[int, ...]
type Buffer = bytes | bytearray | memoryview
type Pathable = str | pathlib.Path | os.PathLike
type Primitive = int | float | str | bytes | bytearray | memoryview
type Datetime = datetime.datetime
