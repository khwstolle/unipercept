"""Modules for working with sequentual layers."""

import typing as T

import torch
import torch.nn


class SequentialList(torch.nn.Sequential):
    """A nn.Sequential that takes a list of tensors as input and returns a list of tensors as output."""

    @T.override
    def forward(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        for module in self:
            input = module(input)
        return input
