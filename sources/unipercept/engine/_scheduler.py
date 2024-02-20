"""Build a ``timm.scheduler.Scheduler`` from a dictionary of hyperparameters."""

from __future__ import annotations

import enum
import functools
import typing as T
import warnings

from timm.scheduler.scheduler import Scheduler
from torch.optim import Optimizer

from ._types import Interval

__all__ = ["SchedType", "create_scheduler", "SchedulerFactory"]

SchedulerAndNum: T.TypeAlias = tuple[Scheduler, int]

MIN_LR_RECOMMENDED = 1e-7
MIN_LR_FEASIBLE = 1e-9


class SchedType(enum.StrEnum):
    COSINE = enum.auto()
    TANH = enum.auto()
    STEP = enum.auto()
    MULTISTEP = enum.auto()
    PLATEAU = enum.auto()
    POLY = enum.auto()


class SchedulerFactory:
    _partial: T.Final

    def __init__(
        self,
        scd: str | SchedType = SchedType.POLY,
        **kwargs,
    ):
        if isinstance(scd, type) and issubclass(scd, Scheduler):
            self._partial = functools.partial(scd, **kwargs)  # type: ignore
        elif isinstance(scd, str) or isinstance(scd, SchedType):
            self._partial = functools.partial(create_scheduler, scd, **kwargs)
        else:
            raise TypeError(f"Invalid scheduler type: {type(scd)}")

    def __call__(
        self, optimizer: Optimizer, epochs: float, updates_per_epoch: int
    ) -> SchedulerAndNum:
        return self._partial(optimizer, epochs, updates_per_epoch)


def create_scheduler(
    scd,
    optimizer: Optimizer,
    epochs: int,
    updates_per_epoch: int,
    /,
    *,
    decay_interval: Interval | None = None,
    decay_milestones: T.Sequence[Interval] | None = None,
    cooldown_interval: Interval | None = None,
    patience_interval: Interval | None = None,
    decay_rate: float = 0.1,
    min_lr: float | None = None,
    warmup_lr: float | None = None,
    warmup_interval: Interval | None = None,
    warmup_prefix: bool = True,
    noise: float | T.Sequence[float] | None = None,
    noise_pct: float = 0.67,
    noise_std: float = 1.0,
    noise_seed: int = 42,
    cycle_mul: float = 1.0,
    cycle_decay: float = 0.1,
    cycle_limit: int = 1,
    k_decay: float = 1.0,
    plateau_mode: str = "max",
    step_on_epochs: bool = False,
    sanity_check: bool = True,
    **kwargs,
) -> SchedulerAndNum:
    # Check base learning rate from optimizer and perform sanity checks
    base_lr = optimizer.param_groups[0]["lr"]
    if sanity_check:
        if base_lr < MIN_LR_RECOMMENDED:
            warnings.warn(
                "Optimizer lr is very low, consider setting warmup_lr to a higher value."
            )
        elif base_lr < MIN_LR_FEASIBLE:
            msg = f"Base learning rate {base_lr} is not feasible."
            raise ValueError(msg)

        # Handle default arguments
        if warmup_lr is None:
            warmup_lr = min(MIN_LR_RECOMMENDED, base_lr * 1e-3)
            warmup_lr = max(MIN_LR_FEASIBLE, warmup_lr)
        if min_lr is None:
            min_lr = warmup_lr
        elif warmup_lr < min_lr:
            msg = f"Minimum learning rate {min_lr=} is less than {warmup_lr=}."
            raise ValueError(msg)
    else:
        if min_lr is None:
            msg = "Sanity check is disabled, please provide minimum learning rate."
            raise ValueError(msg)
        if warmup_lr is None:
            warmup_lr = min_lr

    # Handle interval arguments
    initial_interval = Interval(epochs, "epochs")
    if warmup_interval is None:
        warmup_interval = Interval(0, "steps")
    if decay_interval is None:
        decay_interval = Interval(0, "steps")
    if cooldown_interval is None:
        cooldown_interval = warmup_interval
    if patience_interval is None:
        patience_interval = Interval(0, "steps")

    # Calculate total epochs
    if not step_on_epochs:
        warmup_t = warmup_interval.get_steps(updates_per_epoch)
        t_initial = initial_interval.get_steps(updates_per_epoch)
        decay_t = decay_interval.get_steps(updates_per_epoch)

        if decay_milestones is not None:
            decay_ts = [d.get_steps(updates_per_epoch) for d in decay_milestones]
        else:
            decay_ts = []
        cooldown_t = cooldown_interval.get_steps(updates_per_epoch)
    else:
        warmup_t = round(warmup_interval.get_epochs(updates_per_epoch))
        t_initial = round(initial_interval.get_epochs(updates_per_epoch))
        decay_t = round(decay_interval.get_epochs(updates_per_epoch))

        if decay_milestones is not None:
            decay_ts = [
                round(d.get_epochs(updates_per_epoch)) for d in decay_milestones
            ]
        else:
            decay_ts = []
        cooldown_t = round(cooldown_interval.get_epochs(updates_per_epoch))

    # Setup warmup args
    warmup_args = {
        "warmup_lr_init": warmup_lr,
        "warmup_t": warmup_t,
        "warmup_prefix": warmup_prefix,
    }

    # Setup noise args for supporting schedulers
    if noise is not None:
        if isinstance(noise, (list, tuple)):
            noise_range = [n * t_initial for n in noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        elif isinstance(noise, float):
            noise_range = noise * t_initial
        else:
            raise TypeError(f"Invalid noise type: {type(noise)}")
    else:
        noise_range = None
    noise_args = {
        "noise_range_t": noise_range,
        "noise_pct": noise_pct,
        "noise_std": noise_std,
        "noise_seed": noise_seed,
    }

    # Setup cycle args for supporting schedulers
    cycle_args = {
        "cycle_mul": cycle_mul,
        "cycle_decay": cycle_decay,
        "cycle_limit": cycle_limit,
    }

    # Setup scheduler
    lr_scheduler = None
    match SchedType(scd):
        case SchedType.COSINE:
            from timm.scheduler import CosineLRScheduler

            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=t_initial,
                lr_min=min_lr,
                t_in_epochs=step_on_epochs,
                **cycle_args,
                **warmup_args,
                **noise_args,
                k_decay=k_decay,
            )
        case SchedType.TANH:
            from timm.scheduler import TanhLRScheduler

            lr_scheduler = TanhLRScheduler(
                optimizer,
                t_initial=t_initial,
                lr_min=min_lr,
                t_in_epochs=step_on_epochs,
                **cycle_args,
                **warmup_args,
                **noise_args,
            )
        case SchedType.STEP:
            from timm.scheduler import StepLRScheduler

            lr_scheduler = StepLRScheduler(
                optimizer,
                decay_t=decay_t,
                decay_rate=decay_rate,
                t_in_epochs=step_on_epochs,
                **warmup_args,
                **noise_args,
            )
        case SchedType.MULTISTEP:
            from timm.scheduler import MultiStepLRScheduler

            lr_scheduler = MultiStepLRScheduler(
                optimizer,
                decay_t=list(decay_ts),
                decay_rate=decay_rate,
                t_in_epochs=step_on_epochs,
                **warmup_args,
                **noise_args,
            )
        case SchedType.PLATEAU:
            from timm.scheduler import PlateauLRScheduler

            assert step_on_epochs, "Plateau LR only supports step per epoch."
            warmup_args.pop("warmup_prefix", False)
            lr_scheduler = PlateauLRScheduler(
                optimizer,
                decay_rate=decay_rate,
                patience_t=round(patience_interval.get_epochs(updates_per_epoch)),
                cooldown_t=0,
                **warmup_args,
                lr_min=min_lr,  # type: ignore
                mode=plateau_mode,
                **noise_args,
            )
        case SchedType.POLY:
            from timm.scheduler import PolyLRScheduler

            lr_scheduler = PolyLRScheduler(
                optimizer,
                power=decay_rate,  # overloading 'decay_rate' as polynomial power
                t_initial=t_initial,
                lr_min=min_lr,
                t_in_epochs=step_on_epochs,
                k_decay=k_decay,
                **cycle_args,
                **warmup_args,
                **noise_args,
            )
        case _:
            raise ValueError(f"Unknown scheduler: {scd}")

    if hasattr(lr_scheduler, "get_cycle_length"):
        # For cycle based schedulers (cosine, tanh, poly) recalculate total epochs w/ cycles & cooldown
        t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
        if step_on_epochs:
            epochs = t_with_cycles_and_cooldown
        else:
            epochs = t_with_cycles_and_cooldown / updates_per_epoch

    return lr_scheduler, float(epochs)
