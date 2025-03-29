"""Simple CLI extension for loading configuration files."""

import argparse
import enum
import os
import typing as T

import expath
import pandas as pd
import torch
from bullet import Bullet
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

from unipercept._api_config import read_config
from unipercept.log import create_table, logger

__all__ = ["add_config_args", "ConfigFileContentType"]

NONINTERACTIVE: T.Final = False
BULLET_STYLES: T.Final = {"bullet": " >", "margin": 2, "pad_right": 2}

if T.TYPE_CHECKING:

    class ConfigFileContentType(DictConfig):
        @property
        def ENGINE(self) -> DictConfig:
            return self.engine

        @property
        def MODEL(self) -> DictConfig:
            return self.model

else:
    ConfigFileContentType: T.TypeAlias = DictConfig


class ConfigSource(enum.StrEnum):
    TEMPLATES = enum.auto()
    CHECKPOINTS = enum.auto()


# --------------------------------------- #
# Config loading and modification actions #
# --------------------------------------- #


def _load_config_to_namespace(
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    dest: str,
    cfg_path: str,
) -> DictConfig:
    cfg = read_config(cfg_path)
    assert cfg is not None
    cfg["CONFIG_BASE"] = cfg_path

    setattr(namespace, dest, cfg)

    return cfg


class ConfigLoad(argparse.Action):
    def __init__(self, option_strings, dest: str, **kwargs):
        super().__init__(option_strings, dest, type=str, nargs=1, **kwargs)

    @T.override
    def __call__(
        self, parser, namespace, name: str | None | T.Iterable[str], option_string=None
    ):
        if not isinstance(name, str) and isinstance(name, T.Iterable):
            name = list(name)
            if len(name) > 1:
                parser.exit(
                    message=(
                        f"Only one configuration file can be specified! Found {name}\n"
                    ),
                    status=1,
                )
                return
            name = None if len(name) == 0 else str(name[0])
        if name is None:
            if NONINTERACTIVE:
                parser.exit(message="No configuration file specified!\n", status=1)
                return
            name = self.interactive()
        assert isinstance(name, str), type(name)

        if getattr(namespace, self.dest, None) is not None:
            parser.exit(
                message=(
                    "Configuration already loaded. Ensure the `--config` flag is "
                    "passed *before* any overrides are defined. "
                    "Multiple configurations are currently not supported."
                ),
                status=1,
            )
            return

        _load_config_to_namespace(parser, namespace, self.dest, name)

    def interactive(self) -> list[str]:
        """Interactively build the ``--config`` arguments."""
        values = [self.interactive_select(), *self.interactive_override()]
        return values

    @staticmethod
    def interactive_select(configs_root="configs://") -> str:
        print(
            "No configuration file specified (--config <path> [config.key=value ...])."
        )

        # Prompt 1: Where to look for configurations?
        try:
            action = Bullet(
                prompt="Select a configuration source:",
                choices=[v.value for v in ConfigSource],
                **BULLET_STYLES,
            )  # type: ignore
        except KeyboardInterrupt:
            print("Received interrupt singal. Exiting.")
            exit(1)
            return None
        choice = action.launch()  # type: ignore
        choice = ConfigSource(choice)

        match choice:
            case ConfigSource.TEMPLATES:
                configs_root = expath.locate("configs://")
            case ConfigSource.CHECKPOINTS:
                configs_root = expath.locate("//unipercept/output/")
            case _:
                msg = f"Invalid choice: {action}"
                raise ValueError(msg)

        configs_root = configs_root.expanduser().resolve()
        config_candidates = configs_root.glob("**/*")
        config_candidates = list(
            filter(
                lambda f: f.is_file()
                and not f.name.startswith("_")
                and f.suffix in (".py", ".yaml"),
                config_candidates,
            )
        )
        config_candidates.sort()

        if len(config_candidates) == 0:
            print(f"No configuration files found in {str(configs_root)}.")
            exit(1)
            return None
        print(
            f"Found {len(config_candidates)} configuration files in {str(configs_root)}."
        )

        # Prompt 2: Which configuration file to use?
        choices = [str(p.relative_to(configs_root)) for p in config_candidates]
        try:
            action = Bullet(
                prompt="Select a configuration file:", choices=choices, **BULLET_STYLES
            )  # type: ignore
        except KeyboardInterrupt:
            print("Received interrupt singal. Exiting.")
            exit(1)
            return None
        choice = action.launch()  # type: ignore
        choice = str(configs_root / choice)

        print(f"Using configuration file: {choice}\n")

        return choice


class ConfigOverrides(argparse.Action):
    r"""
    This action is used to apply configuration overrides to the configuration
    using Hydra overrides syntax. The overrides are applied after the configuration
    has been loaded.
    """

    target: T.Final[str]  # target in the namespace where the config lives

    def __init__(
        self,
        option_strings,
        dest: str,
        target: str,
        first_maybe_file: bool = True,
        **kwargs,
    ):
        super().__init__(option_strings, dest, type=str, **kwargs)

        self.first_maybe_file = first_maybe_file
        self.target = target
        self.overrides_parser = OverridesParser.create()

    @T.override
    def __call__(self, parser, namespace, values, option_string=None):
        match values:
            case str():
                overrides = [values]
            case None:
                overrides = []
            case _:
                overrides = list(values)

        if len(overrides) == 0:
            logger.debug("No overrides to apply to '%s'. Skipping.", self.target)
            return  # No overrides to apply
        cfg = getattr(namespace, self.target)
        if cfg is None:
            if not self.first_maybe_file:
                parser.exit(
                    message=(
                        "The argument parser found configuration overrides, but "
                        f"no configuration file was provided. Found: {overrides}"
                    ),
                    status=1,
                )
                return
            cfg_path, *overrides = overrides
            if not cfg_path.endswith(".py") and not cfg_path.endswith(".yaml"):
                parser.exit(
                    message=(
                        "When no `--config` flag is explicitly set, the first "
                        "remaining argument should be a configuration file, but "
                        f"found '{cfg_path}' instead."
                    ),
                    status=1,
                )
                return

            logger.info(
                "No configuration was explicitly loaded via `--config <path>` . "
                "Attempting to load the configuration file from the first remaining "
                "argument: %s",
                cfg_path,
            )
            cfg = _load_config_to_namespace(parser, namespace, self.target, cfg_path)
        assert cfg is not None

        # Read the current overrides to a variable to ensure it cannot be tampered
        # with by overrides. This restriction helps ensure reproducible settings.
        current_overrides = cfg.get("CONFIG_OVERRIDES", [])

        # Apply the overrides
        self.apply_overrides(cfg, overrides)

        # Update the overrides list in the configuration
        cfg["CONFIG_OVERRIDES"] = current_overrides + overrides

    @staticmethod
    def safe_update(cfg, key: str, value: T.Any):
        r"""
        Update a key in an OmegaConf configuration, but ensure that the path to the key
        is configurable. This is to prevent accidental overwriting of non-configurable
        values.
        """
        parts = key.split(".")
        for idx in range(1, len(parts)):
            prefix = ".".join(parts[:idx])
            v = OmegaConf.select(cfg, prefix, default=None)
            if v is None:
                break
            if not OmegaConf.is_config(v):
                msg = (
                    f"Trying to update key {key}, but {prefix} ({type(v)}) is not "
                    "configurable."
                )
                raise KeyError(msg)
        OmegaConf.update(cfg, key, value, merge=True)

    def apply_overrides(self, cfg: DictConfig, overrides: list[str]):
        r"""
        Apply a list of Hydra overrides to the configuration.
        """

        # Apply each override sequentially
        applied = []
        for ov in self.overrides_parser.parse_overrides(overrides):
            key = ov.key_or_group
            value = ov.value()
            if value == "None":
                value = None
            self.safe_update(cfg, key, value)
            applied.append({"Key": key, "Value": value, "Type": type(value).__name__})

        # Log the applied overrides
        if len(applied) > 0:
            logger.info(
                "Configuration overrides applied from CLI:\n%s",
                create_table(pd.DataFrame.from_records(applied), format="wide"),
            )
        return cfg


# ------------- #
# Config macros #
# ------------- #


class ConfigMode(argparse.Action):
    r"""
    Configuration modes are patches that can be applied to the configuration
    after it has been loaded.

    These are used to apply a batch of changes to the
    configuration that would usually be applied together, e.g. to enable debugging mode
    requires several settings to increase verbosity and reproducability.
    """

    target: T.Final[str]

    def __init__(self, option_strings, dest, *, target: str, **kwargs):
        super().__init__(option_strings, dest, type=None, nargs=0, **kwargs)

        self.target = target

    @T.override
    def __call__(self, parser, namespace, values, option_string=None):
        cfg = getattr(namespace, self.target)

        if cfg is None:
            msg = "Cannot apply patch when configuration is not loaded!"
            raise RuntimeError(msg)

        self.apply_patch(cfg)

    def apply_patch(self, cfg: ConfigFileContentType):
        raise NotImplementedError()

    FLAGS: T.ClassVar[tuple[str, ...] | None] = None
    HELP: T.ClassVar[str | None] = None

    def __init_subclass__(cls, *, flags: str | T.Iterable[str], help: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.FLAGS = (flags,) if isinstance(flags, str) else tuple(flags)
        cls.HELP = help

    @classmethod
    def apply_parser(
        cls, parser: argparse.ArgumentParser | argparse._ArgumentGroup, target: str
    ):
        if cls.FLAGS is None:
            msg = "No flags specified for the configuration patch!"
            raise ValueError(msg)
        parser.add_argument(
            *cls.FLAGS,
            action=cls,
            help=cls.HELP,
            target=target,
        )


class ConfigDebugMode(
    ConfigMode, flags="--debug", help="enable debug mode and increase verbosity"
):
    @T.override
    def apply_patch(self, cfg):
        from logging import DEBUG

        from unipercept.engine.debug import DebugMode

        logger.info("Applying debug mode to the configuration.")

        os.environ["TORCH_DEBUG"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["PYTORCH_DEBUG"] = "1"
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["UP_LOGS_LEVEL"] = "DEBUG"

        torch._logging.set_logs(dynamo=DEBUG)
        torch._dynamo.reset()
        torch._dynamo.config.verbose = True

        cfg.ENGINE.params.debug = DebugMode.UNDERFLOW_OVERFLOW


class ConfigDetectionAnomaliesMode(
    ConfigMode,
    flags="--detect-anamolies",
    help=("enable anamoly detection"),
):
    @T.override
    def apply_patch(self, cfg):
        torch.autograd.set_detect_anomaly(True)


class ConfigDeterministicMode(
    ConfigMode,
    flags="--deterministic",
    help="enable deterministic mode",
):
    @T.override
    def apply_patch(self, cfg):
        cfg.ENGINE.params.full_determinism = True


class ConfigDisableTrackers(
    ConfigMode,
    flags="--disable-trackers",
    help="patches the configuration to disable all experiment trackers",
):
    @T.override
    def apply_patch(self, cfg):
        cfg.ENGINE.params.trackers = []


# ---------------- #
# Config arguments #
# ---------------- #


def add_config_args(
    parser: argparse.ArgumentParser,
    *,
    options: T.Sequence[str] = ("--config", "-c"),
    dest="config",
    **kwargs,
) -> None:
    r"""
    Adds a configuration file argument to a given parser.
    """

    group_config = parser.add_argument_group("configuration")
    group_config.add_argument(
        *options,
        dest=dest,
        action=ConfigLoad,
        metavar="CONFIG",
        help="configuration file path (.config.py or .yaml)",
        **kwargs,
    )
    group_config.add_argument(
        "overrides",
        action=ConfigOverrides,
        metavar="K=V",
        help="configuration overrides (key=value)",
        nargs=argparse.REMAINDER,
        target=dest,
    )

    group_patches = parser.add_argument_group("patches")
    ConfigDebugMode.apply_parser(group_patches, target=dest)
    ConfigDisableTrackers.apply_parser(group_patches, target=dest)
