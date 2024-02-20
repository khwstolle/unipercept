import enum as E
import os
import typing as T

import torch
from torch import nn

from unipercept import file_io
from unipercept.log import logger
from unipercept.types import Pathable, StateDict, Tensor

__all__ = ["read_checkpoint", "load_checkpoint", "MissingWeightsAction"]


def read_checkpoint(state: Pathable, device: str = "cpu") -> dict[str, Tensor]:
    """
    Read a state dict from a path.
    """

    state_path = file_io.Path(state)
    if state_path.is_dir():
        state_path = state_path / "model.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)

    match state_path.suffix:
        case ".pth":
            # Load PyTorch pickled checkpoint
            state_dict = torch.load(state_path, map_location=device)
            if isinstance(state_dict, nn.Module):
                # If the checkpoint is a nn.Module, extract its state_dict
                state_dict = state_dict.state_dict()
            elif not isinstance(state_dict, dict):
                pass  # OK
            else:
                msg = f"Expected a state_dict or a nn.Module, got {type(state_dict)}"
                raise TypeError(msg)
        case ".safetensors":
            # Load SafeTensors checkpoint
            import safetensors.torch as st

            state_dict = st.load_file(state_path, device=device)
        case _:
            msg = (
                f"Checkpoint file must be a .pth or .safetensors file, got {state_path}"
            )
            raise ValueError(msg)

    return state_dict


class MissingWeightsAction(E.StrEnum):
    r"""
    Actions to take when loading a checkpoint with missing or unexpected weights.
    """

    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


def load_checkpoint(
    state: Pathable | StateDict,
    target: nn.Module,
    *,
    on_missing: MissingWeightsAction = MissingWeightsAction.WARN,
    on_unexpected: MissingWeightsAction = MissingWeightsAction.WARN,
    **kwargs,
) -> None:
    """
    Load a checkpoint into a model from a file or a dictionary. The model is modified in-place.

    Parameters
    ----------
    state
        Path to the checkpoint file (.pth/.safetensors file) or a dictionary containing the model state.
    target
        The model to load the state into.
    raise_missing
        Whether to raise a warning if weights are missing in the checkpoint. Defaults to `True`.
    raise_unexpected
        Whether to raise a warning if unexpected weights are found in the checkpoint. Defaults to `False`.
    **kwargs
        Additional keyword arguments to pass to the checkpoint reader.
    """

    # Check remote
    # if isinstance(state, str) and state.startswith(WANDB_RUN_PREFIX):
    #     state = _read_model_wandb(state)

    # Check argument type
    if isinstance(state, str | os.PathLike):
        state_dict = read_checkpoint(state, **kwargs)
    else:
        assert isinstance(state, T.Mapping), type(state)
        state_dict = state

    missing, unexpected = target.load_state_dict(state_dict, strict=False)

    # Error reporting - when an exception must be raised, we ensure that both the
    # missing and unexpected weights are printed to the debugging log before crashing.
    def _handle_problematic_weights(
        action: MissingWeightsAction,
        problem_msg: str,
        problem_keys: list[str],
        display_keys: int = 30,
    ):
        problem_count = len(problem_keys)
        if problem_count == 0 or action == MissingWeightsAction.IGNORE:
            return ""
        problem_keys = sorted(problem_keys)
        if problem_count > display_keys:
            problem_keys = problem_keys[:display_keys] + [
                f"... and {len(problem_keys) - display_keys} more",
            ]
        msg = (
            problem_msg
            + f" (total {problem_count}) in checkpoint: \n- "
            + "\n- ".join(problem_keys)
        )
        logger.warning(msg, stacklevel=0)
        return msg

    msg_missing = _handle_problematic_weights(
        on_missing,
        "missing weights",
        missing,
    )
    msg_unexpect = _handle_problematic_weights(
        on_unexpected,
        "unexpected weights",
        unexpected,
    )

    if msg_missing and on_missing == MissingWeightsAction.RAISE:
        raise ValueError(msg_missing)
    if msg_unexpect and on_unexpected == MissingWeightsAction.RAISE:
        raise ValueError(msg_unexpect)
