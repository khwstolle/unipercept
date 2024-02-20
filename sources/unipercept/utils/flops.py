r"""
Implementation of a context manager to measure flops of a module.

Inspired by the discussion on [the ideal flop counter](https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505)
on the PyTorch forum.
"""

import logging
import operator
import typing as T
from collections import defaultdict
from functools import reduce

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import PyTree, tree_map

aten: torch._ops._OpNamespace = torch.ops.aten
type InputType = torch.Tensor | bool | tuple[bool, ...]


def _matmul_flop_count(
    inputs: tuple[torch.Tensor, ...], _outputs: tuple[T.T.Any, ...]
) -> int:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = inputs[0].numel() * input_shapes[-1][-1]
    return flop


def _addmm_flop_count(inputs: tuple[torch.Tensor], _outputs: tuple[T.Any, ...]) -> int:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def _bmm_flop_count(inputs: tuple[torch.Tensor], _outputs: tuple[T.Any, ...]) -> int:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def _conv_flop_count(
    x_shape: Union[torch.Size, list[int]],
    w_shape: Union[torch.Size, list[int]],
    out_shape: Union[torch.Size, list[int]],
    transposed: bool = False,
) -> int:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as

    ::math
        flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Parameters
        ----------
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = (
        batch_size
        * reduce(operator.mul, w_shape, 1)
        * reduce(operator.mul, conv_shape, 1)
    )
    return flop


def _conv_flop_count(
    inputs: list[Union[torch.Tensor, bool, tuple[bool]]],
    outputs: tuple[torch.Tensor],
) -> int:
    """
    Count flops for convolution.
    """
    x: torch.Tensor = T.cast(torch.Tensor, inputs[0])
    w: torch.Tensor = T.cast(torch.Tensor, inputs[1])
    x_shape, w_shape, out_shape = (x.shape, w.shape, outputs[0].shape)
    transposed: bool = T.cast(bool, inputs[6])

    return _conv_flop_count(
        list(x_shape), list(w_shape), list(out_shape), transposed=transposed
    )


def _transpose_shape(shape: torch.Size) -> list[int]:
    return [shape[1], shape[0]] + list(shape[2:])


def _conv_backward_flop_count(
    inputs: tuple[Union[torch.Tensor, bool, tuple[bool]]], outputs: tuple[torch.Tensor]
) -> int:
    grad_out_shape, x_shape, w_shape = (
        T.cast(torch.Tensor, i).shape for i in inputs[:3]
    )
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count: int = 0

    if T.cast(tuple[bool], output_mask)[0]:
        grad_input_shape = outputs[0].shape

        flop_count = flop_count + _conv_flop_count(
            grad_out_shape, w_shape, grad_input_shape, not fwd_transposed
        )
    if T.cast(tuple[bool], output_mask)[1]:
        grad_weight_shape = outputs[1].shape

        flop_count += _conv_flop_count(
            list(_transpose_shape(x_shape)),
            list(grad_out_shape),
            list(grad_weight_shape),
            T.cast(bool, fwd_transposed),
        )

    return flop_count


flop_mapping: T.Mapping[
    T.Callable[..., T.Any], T.Callable[[tuple[T.Any, ...], tuple[T.Any, ...]], int]
] = {
    aten.mm: _matmul_flop_count,
    aten.matmul: _matmul_flop_count,
    aten.addmm: _addmm_flop_count,
    aten.bmm: _bmm_flop_count,
    aten.convolution: _conv_flop_count,
    aten._convolution: _conv_flop_count,
    aten.convolution_backward: _conv_backward_flop_count,
    # Add their default to make sure they can be mapped.
    aten.mm.default: _matmul_flop_count,
    aten.matmul.default: _matmul_flop_count,
    aten.addmm.default: _addmm_flop_count,
    aten.bmm.default: _bmm_flop_count,
    aten.convolution.default: _conv_flop_count,
    aten._convolution.default: _conv_flop_count,
    aten.convolution_backward.default: _conv_backward_flop_count,
}


def _normalize_tuple[_T](x: _T) -> tuple[_T]:
    if not isinstance(x, tuple):
        return (x,)
    return x


class FlopTensorDispatchMode(TorchDispatchMode):
    """
    A context manager to measure flops of a module. Requires PyTorch 1.13+.

    Examples::

        >>> import copy
        >>> import torch
        >>> import torchvision.models as models
        >>> from torchtnt.utils.flops import FlopTensorDispatchMode

        >>> module = models.resnet18()
        >>> module_input = torch.randn(1, 3, 224, 224)
        >>> with FlopTensorDispatchMode(module) as ftdm:
        >>>     # count forward flops
        >>>     res = module(module_input).mean()
        >>>     flops_forward = copy.deepcopy(ftdm.flop_counts)

        >>>     # reset count before counting backward flops
        >>>     ftdm.reset()
        >>>     res.backward()
        >>>     flops_backward = copy.deepcopy(ftdm.flop_counts)

    """

    def __init__(self, module: torch.nn.Module) -> None:
        """
        Initializes a FlopTensorDispatchMode context manager object.

        Parameters
        ----------
        module
            The module to count flops on.
        """
        self._all_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._instrument_module(module, "")

        self.flop_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._parents: list[str] = [""]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook_handle in self._all_hooks:
            hook_handle.remove()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
    ) -> PyTree:
        rs = func(*args, **kwargs)
        outs = _normalize_tuple(rs)

        if func in flop_mapping:
            flop_count = flop_mapping[func](args, outs)
            for par in self._parents:
                self.flop_counts[par][func.__name__] += flop_count
        else:
            logging.debug(f"{func} is not yet supported in FLOPs calculation.")

        return rs

    def _create_backwards_push(self, name: str) -> T.Callable[..., T.Any]:
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
                )
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                parents = self._parents
                parents.append(name)
                return grad_outs

        # But this class can't be lifted out of the function as it is a static class
        # using a function parameter.
        return PushState.apply

    def _create_backwards_pop(self, name: str) -> T.Callable[..., T.Any]:
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
                )
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                parents = self._parents
                assert parents[-1] == name
                parents.pop()
                return grad_outs

        # But this class can't be lifted out of the function as it is a static class
        # using a function parameter.
        return PopState.apply

    def _enter_module(self, name: str) -> T.Callable[..., T.Any]:
        def f(module: torch.nn.Module, inputs: tuple[T.Any, ...]):
            parents = self._parents
            parents.append(name)
            inputs = _normalize_tuple(inputs)
            out = self._create_backwards_pop(name)(*inputs)
            return out

        return f

    def _exit_module(self, name: str) -> T.Callable[..., T.Any]:
        def f(
            module: torch.nn.Module,
            inputs: tuple[T.Any, ...],
            outputs: tuple[T.Any, ...],
        ):
            parents = self._parents
            assert parents[-1] == name
            parents.pop()
            outputs = _normalize_tuple(outputs)
            return self._create_backwards_push(name)(*outputs)

        return f

    def _instrument_module(
        self,
        mod: torch.nn.Module,
        par_name: str,
    ) -> None:
        for name, module in dict(mod.named_children()).items():
            formatted_name = name
            if par_name != "":
                formatted_name = f"{par_name}.{name}"
            self._all_hooks.append(
                module.register_forward_pre_hook(self._enter_module(formatted_name))
            )
            self._all_hooks.append(
                module.register_forward_hook(self._exit_module(formatted_name))
            )
            self._instrument_module(module, formatted_name)

    def reset(self) -> None:
        """
        Resets current flop count.
        """
        self._parents = [""]
        self.flop_counts.clear()
