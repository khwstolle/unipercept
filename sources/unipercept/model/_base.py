"""
Defines the interface for a perception model.
"""

import abc
import copy
import typing as T

import torch
from tensordict import LazyStackedTensorDict, TensorDictBase
from torch import Tensor, nn

from laco import apply_overrides, instantiate
from unipercept.log import logger
from unipercept.types import Device, Pathable

from ._checkpoint import load_checkpoint, read_checkpoint
from ._io import InputData

__all__ = [
    "ModelInput",
    "ModelOutput",
    "ModelBase",
    "ModelFactory",
]


#########################
# BASE CLASS FOR MODELS #
#########################

ModelInput = InputData | TensorDictBase | dict[str, Tensor]


class ModelOutput(T.NamedTuple):
    """
    The output of a model.
    """

    losses: TensorDictBase | dict[str, Tensor] | None
    results: (
        list[TensorDictBase]
        | list[dict[str, Tensor]]
        | LazyStackedTensorDict
        | TensorDictBase
        | None
    )


class ModelBase(nn.Module):
    """
    Defines the interface for a perception model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def select_inputs(self, data: ModelInput, **kwargs) -> tuple[T.Any, ...]:
        msg = f"Method `select_inputs` must be implemented for {cls.__name__}"
        raise NotImplementedError(msg)

    def predict(
        self, data: ModelInput, **kwargs
    ) -> list[TensorDictBase] | LazyStackedTensorDict | TensorDictBase | None:
        self = self.eval()
        with torch.inference_mode():
            inputs = self.select_inputs(data, **kwargs)
            return self(*inputs).results

    def losses(self, data: ModelInput, **kwargs) -> dict[str, Tensor]:
        self = self.train()
        inputs = self.select_inputs(data, **kwargs)
        return self(*inputs).losses

    if T.TYPE_CHECKING:

        def __call__(self, *args: Tensor) -> ModelOutput: ...


class ModelFactory:
    def __init__(
        self,
        model_config,
        weights: Pathable | None = None,
        freeze_weights: bool = False,
        compile: bool | dict[str, T.Any] = False,
    ):
        self.model_config = model_config
        self.weights = weights or None
        self.freeze_weights = freeze_weights

        if isinstance(compile, bool):
            self.compile = {} if compile else None
        else:
            self.compile = compile

    def __call__(
        self,
        *,
        weights: Pathable | None = None,
        overrides: T.Sequence[str] | T.Mapping[str, T.Any] | None = None,
        device: Device | str | None = None,
    ) -> ModelBase:
        """
        TODO interface not clearly defined yet
        """

        # Configuration
        model_config = copy.deepcopy(self.model_config)
        if overrides is not None:
            if isinstance(overrides, T.Mapping):
                overrides_list = [f"{k}={v}" for k, v in overrides.items()]
            else:
                overrides_list = list(overrides)
            logger.info(
                "Model factory: config %s",
                ", ".join(overrides_list) if len(overrides_list) > 0 else "(none)",
            )
            model_config = apply_overrides(model_config, overrides_list)
        else:
            logger.info("Model factory: config has no overrides")

        # Instantiate model
        model = T.cast(ModelBase, instantiate(self.model_config))

        # Compile if options (kwargs to torch.compile) are set
        if self.compile is not None:
            logger.info("Model factory: compile options: %s", self.compile)
            model = torch.compile(model, **self.compile)
        else:
            logger.info("Model factory: compile disabled")

        # Load weights
        if weights is None:
            weights = self.weights
        if weights is not None:
            logger.info("Model factory: using weights from %s", weights)

            load_kwargs = {}
            if device is not None:
                load_kwargs["device"] = device
            load_checkpoint(weights, model, **load_kwargs)
        else:
            logger.info("Model factory: using random initialization")

        # Freeze weights from the **initialization** checkpoint if requested
        if self.freeze_weights:
            freeze_keys = read_checkpoint(weights).keys()
            counter = 0
            for name, param in model.named_parameters():
                if name in freeze_keys:
                    param.requires_grad = False
                    logger.debug("Freezing parameter: %s (imported)", name)
                    counter += 1
            logger.info("Model factory: frozen %d parameters (imported)", counter)

        if device is not None:
            model = model.to(device=device)

        return model
