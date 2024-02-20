import torch
from torch import nn

from unipercept.nn.layers.transformer.attention import LinearProject


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, proj: LinearProject = nn.Linear):
        super().__init__()

        self.proj_g = proj(dim, hidden_dim, bias=False)
        self.proj_o = proj(hidden_dim, dim, bias=False)
        self.proj_i = proj(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_o(nn.functional.silu(self.proj_g(x)) * self.proj_i(x))  # type: ignore
