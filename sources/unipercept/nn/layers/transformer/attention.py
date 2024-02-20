"""
Generic attention and variants.
"""

import typing as T

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

type LinearProject = nn.Module | T.Callable[[int, int], nn.Module]


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, groups: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Used in Grouped Query Attention (GQA)
    """
    keys = torch.repeat_interleave(keys, repeats=groups, dim=dim)
    values = torch.repeat_interleave(values, repeats=groups, dim=dim)
    return keys, values


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_heads: int | None = None,
        *,
        num_heads: int = 8,
        num_groups: int = 2,
        num_sinks: int = 0,
        gate: LinearProject | None = None,
        proj: LinearProject = nn.Linear,
    ):
        r"""
        The most basic kind of MHA
        """
        super().__init__()

        if dim_heads is None:
            assert dim_model % num_heads == 0, (dim_model, num_heads)
            dim_heads = dim_model // num_heads

        assert num_heads > 0
        assert num_groups > 0

        hidden_dim = dim_heads * num_heads

        assert hidden_dim % num_groups == 0, (hidden_dim, num_groups)

        self.num_heads = num_heads
        self.num_groups = num_groups

        self.dim_heads = dim_heads
        self.dim_model = dim_model

        self.proj_q = proj(dim_model, hidden_dim, bias=False)
        self.proj_kv = proj(dim_model, 2 * hidden_dim // num_groups, bias=False)
        self.proj_o = proj(hidden_dim, dim_model, bias=False)

        # Gate
        if gate is not None:
            self.proj_g = proj(dim_model, hidden_dim)
        else:
            self.register_module("proj_g", None)

        # Sinks
        assert num_sinks >= 0
        self.num_sinks = num_sinks
        if num_sinks > 0:
            self.sinks = nn.Parameter(
                torch.randn(num_heads // dim_heads, num_sinks, dim_heads)
                / dim_heads**0.5
            )
        else:
            self.register_parameter("sinks", None)

    def shape_inputs(self, x: Tensor, num_heads: int, head_dim: int) -> Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)

    @T.override
    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        x
            Query, keys and values of shape (B, N, C)

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, C)
        """

        qrys = self.proj_q(inputs)
        qrys = self.shape_inputs(qrys, self.num_heads, self.dim_heads)

        keys, vals = self.proj_kv(inputs).chunk(2, dim=-1)
        keys, vals = (
            self.shape_inputs(
                x,
                self.num_heads // self.num_groups,
                self.dim_heads // self.num_groups,
            )
            for x in (keys, vals)
        )

        if self.sinks is not None:
            sinks = self.sinks.unsqueeze(0).expand(B, -1, -1, -1)
            qrys, keys, vals = (
                torch.cat((sinks, x), dim=-2) for x in (qrys, keys, vals)
            )

        keys, vals = repeat_kv(keys, vals, self.num_groups, dim=1)

        outs = scaled_dot_product_attention(
            qrys, qrys, vals, enable_gqa=self.num_groups > 1
        )  # (N, H, L, E)
        outs = outs.reshape(B, -1, self.dim_heads * self.num_heads)

        if self.proj_g is not None:
            outs = self.proj_g(inputs) * outs

        return self.proj_o(outs)
