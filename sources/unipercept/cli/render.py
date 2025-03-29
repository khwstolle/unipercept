"""
Commands that create renders.
"""

import argparse
import typing as T

import expath
import render
import tensors

from unipercept.cli._command import command, logger
from unipercept.data.sets import catalog
from unipercept.utils.seed import set_seed

from ._utils import create_subtemplate

__all__ = []


Subcommand = create_subtemplate()

command(name="render", help="rendering tools")(Subcommand)
set_seed()


def _add_info_args(prs: argparse.ArgumentParser):
    prs.add_argument("--info", "-i", type=catalog.get_info, help="dataset info key")


def _add_path_args(prs: argparse.ArgumentParser, formats: T.Iterable[str]):
    prs.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="overwrite existing files",
    )
    prs.add_argument(
        "--source",
        "-s",
        type=str,
        default="auto",
        help="file format to read from",
        choices=list(formats),
    )
    prs.add_argument(
        "--statistics",
        "-v",
        action="store_true",
        help="show statistics of the tensor",
    )
    prs.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="do not write any files",
    )
    prs.add_argument(
        "--show",
        action="store_true",
        help="show the rendered image in the terminal (if supported)",
    )
    prs.add_argument(
        "inputs",
        type=expath.locate,
        nargs="+",
        help="target files",
    )


class DepthSubcommand(Subcommand, name="depth"):
    """
    Render a depth map.
    """

    @staticmethod
    def setup(prs: argparse.ArgumentParser):
        _add_info_args(prs)
        _add_path_args(prs, tensors.DepthFormat)

    @staticmethod
    def render(
        input_file: expath.locate,
        *,
        overwrite: bool,
        dry_run: bool,
        show: bool,
        summarize: bool,
        load_kwargs: dict[str, T.Any] | None = None,
        render_kwargs: dict[str, T.Any] | None = None,
    ):
        logger.info(f"Rendering {input_file}")
        if not input_file.exists():
            logger.error(f"File {input_file} does not exist.")
            return

        if render_kwargs is None:
            render_kwargs = {}
        if load_kwargs is None:
            load_kwargs = {}

        tensor = tensors.load_depth(input_file, **load_kwargs)
        result = render.draw_image_depth(tensor, **render_kwargs)

        if not dry_run:
            output_file = input_file.with_suffix(".render.png")
            if output_file.exists() and not overwrite:
                logger.warning(f"Skipping {output_file} (already exists)")
            else:
                result.save(output_file)

        if show:
            render.terminal.show(result)
        if summarize:
            stats = T.OrderedDict()
            stats["dtype"] = tensor.dtype
            stats["shape"] = tensor.shape
            stats["min"] = f"{tensor.min().item():.4f}"
            stats["max"] = f"{tensor.max().item():.4f}"
            stats["mean"] = f"{tensor.mean().item():.4f}"
            stats["std"] = f"{tensor.std().item():.4e}"

            logger.info(", ".join(f"{k}: {v}" for k, v in stats.items()))

    @classmethod
    @T.override
    def main(cls, args: argparse.Namespace):
        for input_file in args.inputs:
            cls.render(
                input_file,
                overwrite=args.overwrite,
                summarize=args.statistics,
                dry_run=args.dry_run,
                show=args.show,
                load_kwargs={
                    "format": args.source,
                },
                render_kwargs={
                    "info": args.info,
                },
            )


if __name__ == "__main__":
    command.root("render")
