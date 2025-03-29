import argparse
import sys
import typing as T

import laco

from unipercept.cli._command import command
from unipercept.cli._config import add_config_args

from ._utils import create_subtemplate

__all__ = []

Subcommand = create_subtemplate()

command(name="config", help="configuration management")(Subcommand)


class CatSubcommand(Subcommand, name="cat"):
    """
    Check the dataset for integrity. This will check that the dataset has a
    valid manifest and that all files in the manifest are present/loadable.
    """

    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--format",
            default="yaml",
            help="output format",
            choices=["yaml", "python", "json"],
        )
        prs.add_argument("--key", default="config", help="key to output")

        add_config_args(prs)

    @staticmethod
    @T.override
    def main(args: argparse.Namespace):
        from omegaconf import OmegaConf

        args_dict = vars(args)
        args_dict.pop("func")

        fmt = args_dict.pop("format")
        out = args_dict.get(args_dict.pop("key"))
        if fmt == "json":
            import json

            args_dict["config"] = OmegaConf.to_object(args.config)
            res = json.dumps(out, indent=4, ensure_ascii=False)
            print(res, file=sys.stdout, flush=True)
        elif fmt == "yaml":
            config_yaml = laco.dump(args.config)
            print(config_yaml, file=sys.stdout, flush=True)
        else:
            print(f"Unknown format: {fmt}", file=sys.stderr)


if __name__ == "__main__":
    command.root("echo")
