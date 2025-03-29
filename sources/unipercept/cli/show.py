"""
Show an item in the terminal.

The type of the item is determined by the file extension and a combination of heuristics.
The item is rendered shown in the terminal using the appropriate viewer.
"""

from __future__ import annotations

import argparse
from sys import stdout

import expath
import render
from PIL import Image as pil_image

from ._command import command


@command("show")
def show(p: argparse.ArgumentParser):
    p.add_argument("path", type=str, nargs="+", help="path to the showable object")

    def main(args):
        for path_str in args.path:
            path = expath.locate(path_str)
            if str(path) != path_str:
                print(f"{path_str} -> {str(path)}", file=stdout, flush=True)
            else:
                print(f"{path_str}", file=stdout, flush=True)

            if not path.is_file():
                print(f"Skipped {path_str} (not a file)", file=stdout, flush=True)
                continue

            # Handle images
            _show_image(path)

    return main


def _show_image(path: expath.locate):
    img = pil_image.open(path)
    render.terminal.show(img)


if __name__ == "__main__":
    command.root(__file__)
