from math import ceil, floor

import torch


def scalar_quantile(  # noqa: PLR0913
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """

    Memory-efficient implementation of torch.quantile for scalars.

    Parameters
    ----------
    input: torch.Tensor
        The input tensor.
    q: float | torch.Tensor
        The quantile to compute. 0 <= q <= 1.
    dim: int | None
        The dimension to reduce. Default is None.
    keepdim: bool
        Whether to keep the reduced dimension. Default is False.
    interpolation: str
        The interpolation method to use. Default is 'nearest'.
    out: torch.Tensor | None
        The output tensor. Default is None.

    Returns
    -------
    torch.Tensor
        The computed quantile.

    """
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().item()
    q = float(q)
    if not (0 <= q <= 1):
        msg = f"Expected 0<={q=}<=1"
        raise ValueError(msg)

    if dim_was_none := dim is None:
        # Cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Check if interpolation is supported
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        msg = "Supported interpolations are 'nearest', 'lower', 'higher' (got {interpolation})!"
        raise NotImplementedError(msg)

    if out is not None:
        msg = f"Only None value is currently supported for out (got {out})!"
        raise ValueError(msg)

    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    return out.squeeze(dim)
