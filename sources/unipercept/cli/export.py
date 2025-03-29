"""
Export CLI
==========

This module provides the CLI for exporting models, e.g. turning a config into a 
target format like torch FX code, ONNX, or PyTorch's native export format.
"""


import argparse
import os

import torch.fx
import torch.compiler

import unipercept as up
from unipercept.cli._command import command, logger
"

import argparse
import dataclasses as D
import random
import sys
import typing as T

import pandas as pd
import torch
from tabulate import tabulate_formats
from tqdm import tqdm

import expath, render
from unipercept.cli._command import command, logger
from unipercept.data.sets import PerceptionDataset, catalog
from unipercept.data.sets._metadata import Metadata
from unipercept.log import create_table
from unipercept.model import InputData
from unipercept.utils.inspect import generate_path
from unipercept.utils.seed import set_seed

from ._utils import create_subtemplate

__all__ = []


Subcommand = create_subtemplate()

command(name="datasets", help="dataset operations")(Subcommand)
set_seed()

def _default_args(prs: argparse.ArgumentParser):
    mode = prs.add_mutually_exclusive_group()
    mode.add_argument("--training", "-T", action="store_true", help="export training")
    mode.add_argument(
        "--inference", "-I", action="store_true", help="export inference", default=True
    )

    prs.add_argument("--output", "-o", type=expath.locate, help="output file (dry run if not set)")

def _load_model(args) -> torch.nn.Module:
    if args.output is None:
        logger.info("Dry run")
        return

    if os.path.exists(args.output):
        logger.warning("Output file exists, overwriting")

    torch._dynamo.reset()
    return up.create_model(args.config)

class TorchFXSubcommand(Subcommand):
    r"""
    Use ``torch.fx`` to trace and export a model as a symbolic graph represented
    using Python code.
    """
    @classmethod
    def init(cls, prs: argparse.ArgumentParser):
        _default_args(prs)

    @classmethod
    def main(cls, args):
        model = _load_model(args)

        gm = torch.fx.symbolic_trace(model)
        gm.print_readable()
        gm.graph.print_tabular()




class ONNXSubcommand(Subcommand):
    r"""
    Use ``torch.onnx`` to trace and export a model as an ONNX graph.
    """
    @classmethod
    def init(cls, prs: argparse.ArgumentParser):
        _default_args(prs)

    @classmethod
    def main(cls, args):
        dataloader, info = up.create_dataset(args.config, return_loader=False)
        inputs = next(dataloader)

        inputs = tuple(model.select_inputs(inputs, torch.device("cuda")))
        model = model.cuda()

        adapter = up.model.ModelAdapter(model, inputs)
        exp = torch.onnx.export(
            adapter, adapter.flattened_inputs, args.output, verbose=True
        )

        print(exp)


class ONNXSubcommand(Subcommand):
    r"""
    Use ``torch.onnx`` to trace and export a model as an ONNX graph.
    """
    @classmethod
    def init(cls, prs: argparse.ArgumentParser):
        _default_args(prs)

    @classmethod
    def main(cls, args):
        dataloader, info = up.create_dataset(args.config, return_loader=False)
        inputs = next(dataloader
        dataloader, info = up.create_dataset(args.config, return_loader=False)
        inputs = next(dataloader)

        inputs = tuple(model.select_inputs(inputs, torch.device("cuda")))
        model = model.cuda()

        try:
            exp = torch.export.export(model, inputs)
        except Exception as e:
            print(e)
            exp = torch.export.export(model, inputs, strict=False)

        print(exp)

if __name__ == "__main__":
    command.root("export")
