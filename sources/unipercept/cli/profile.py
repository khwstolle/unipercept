"""
Model profiling entry point.
"""

import argparse
import inspect
import pathlib
import sys
import typing as T
from collections import defaultdict

import create_dataset
import create_engine
import create_model
import expath
import pandas as pd
import regex as re
import torch
import torch.autograd
import torch.autograd.profiler
import torch.utils.data
from omegaconf.errors import ConfigAttributeError
from tensordict import TensorDictBase
from torch import nn
from tqdm import tqdm

from unipercept.cli._command import command, logger
from unipercept.cli._config import add_config_args
from unipercept.log import create_table, logger
from unipercept.model import ModelBase
from unipercept.utils.seed import set_seed
from unipercept.utils.time import get_timestamp

from ._utils import create_subtemplate

__all__ = []


Subcommand = command(name="profile", description=__doc__, help="profiling tools")(
    create_subtemplate()
)

set_seed()


def _add_loader_type_args(
    subparser: argparse.ArgumentParser,
) -> None:
    """ """
    subparser.add_argument(
        "--loader",
        "-l",
        type=str,
        default=None,
        help="evaluation suite key or stage number to use for profiling",
    )
    subparser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=3,
        help="number of iterations to profile",
    )
    subparser.add_argument(
        "--sort-by",
        type=str,
        default="self_cpu_memory_usage",
        help="sort by this column when showing output in console",
    )
    subparser.add_argument("--weights", "-w", default=None, type=str)

    mode = subparser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--training", "-T", action="store_true", help="profile training")
    mode.add_argument(
        "--inference", "-I", default=True, action="store_true", help="profile inference"
    )

    # subparser.add_argument("path", nargs="*", type=str)


def _find_session_path(config: T.Any) -> pathlib.Path:
    """
    Find the path to the session file.
    """

    proj_name = config.ENGINE.params.project_name

    try:
        path = expath.locate(f"//unipercept/output/{proj_name}/{config.session_id}/profile")
    except (KeyError, ConfigAttributeError):
        path = expath.locate(f"//unipercept/output/profile/{proj_name}/{get_timestamp()}/profile")
        logger.warning("No session file found in config, using default path: %s", path)

    path.mkdir(exist_ok=True, parents=True)

    return path


class ParamCountSubcommand(Subcommand, name="params"):
    type ParameterCounts = dict[str, int]

    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        add_config_args(prs)
        prs.add_argument(
            "--max-depth",
            "-d",
            type=int,
            default=None,
            help="maximum amount of submodules to display",
        )
        prs.add_argument(
            "--human-readable",
            "-H",
            action="store_true",
            default=False,
            help="use human-readable formatting for outputs",
        )
        prs.add_argument(
            "--match",
            "-m",
            type=re.compile,
            default=None,
            help="filter parameters by (group) name",
        )
        prs.add_argument(
            "--prefix",
            "-p",
            type=str,
            default="",
            help="filter paramters by a prefix path",
        )
        prs.add_argument(
            "--leaves", "-L", action="store_true", help="only show leaf parameters"
        )

    @classmethod
    def main(cls, args: argparse.Namespace):
        model = create_model(args.config)

        logger.info("Analysing model parameters...")
        table, total = cls.summarize_params(
            model,
            max_depth=args.max_depth,
            human_readable=args.human_readable,
            match=args.match,
            prefix=args.prefix,
            leaves_only=args.leaves,
        )
        table = pd.DataFrame(table)
        print(create_table(table, format="wide"), file=sys.stdout, flush=True)
        print(f"Total parameters: {total:,}", file=sys.stdout, flush=True)

    @staticmethod
    def count_params(source: T.Iterable[tuple[str, nn.Parameter]]) -> ParameterCounts:
        """
        Counts the number of parameters in a model.

        Returns
        -------
        dict[str, int]:
            Mapping of parameter names to the number of elements in the parameter.
            The key of the root is an empty string.
        """
        counts = defaultdict(int)
        for name, prm in source:
            size = prm.numel()
            name_parts = name.split(".")
            for k in range(len(name_parts) + 1):
                prefix = ".".join(name_parts[:k])
                counts[prefix] += size
        return dict(counts)

    @classmethod
    def summarize_params(
        cls,
        model: nn.Module,
        *,
        max_depth: int | None = None,
        human_readable: bool = True,
        match: re.Pattern | None = None,
        prefix: str = "",
        leaves_only: bool = False,
    ) -> tuple[dict[str, list], int]:
        # Ensure prefixes comprise only whole modules or are empty
        if prefix != "" and not prefix.endswith("."):
            prefix += "."

        # Create a list of parameters and their names, filtering by the prefix
        params = [
            (name[len(prefix) :], param)
            for name, param in model.named_parameters()
            if name.startswith(prefix)
        ]
        if len(params) == 0:
            logger.error("No parameters found.")
            return {}

        # Sort the parameters by their name
        params = sorted(params, key=lambda x: x[0])

        # Gather data to summarize
        counts = cls.count_params(params)
        param_shape: dict[str, tuple[int, ...]] = {k: tuple(v.shape) for k, v in params}

        # Create a table as output
        class ParamStats(T.NamedTuple):
            depth: int
            parameter: str
            numel: int
            shape: tuple[int, ...] | None = None

        stats: list[ParamStats] = []

        def _format_size(x: float) -> str:
            if not human_readable:
                return str(x)
            METRIC_SEP: T.Final[float] = 1e3

            # Handle special case for small sizes, where spaces are padded for alignment
            x = float(x)
            if x < METRIC_SEP:
                return f"{int(x):d}" + (" " * 5)

            # Format the size in a human-readable format
            for unit in ("k", "M", "G"):  # noqa: B007
                x /= METRIC_SEP
                if x < METRIC_SEP:
                    break
            return f"{x:.2f} {unit}"

        def _format_shape(s: tuple[int, ...] | None) -> str:
            if s is None:
                return ""
            if not human_readable:
                return str(s)
            return "( " + " × ".join(str(x) for x in s) + " )"

        def _fill_recursive(
            lvl: int,
            path: str,
        ) -> None:
            if max_depth is not None and lvl >= max_depth:
                logger.debug("Reached maximum depth of %d", max_depth)
                return
            for name, v in counts.items():
                if name.count(".") == lvl and name.startswith(path):
                    if (not match or match.search(name) is not None) and (
                        not leaves_only or name in param_shape
                    ):
                        stats.append(
                            ParamStats(
                                depth=lvl,
                                parameter=prefix + name,
                                numel=v,
                                shape=param_shape.get(name),
                            )
                        )
                    if name not in params:  # is a submodule group
                        _fill_recursive(lvl + 1, name + ".")

        # Fill the root group (model or submodule found by prefix)
        total_params = counts.pop("")
        logger.info("Total parameters: %s", _format_size(total_params))

        if not leaves_only:
            stats.append(
                ParamStats(
                    depth=0,
                    parameter=prefix.strip(".") or "model",
                    numel=total_params,
                    shape=None,
                )
            )

        # Fill the table recursively
        _fill_recursive(0, "")

        if len(stats) == 0:
            logger.error("No parameters found.")
            return {}

        # Create the table
        columns = list(ParamStats._fields)

        if human_readable:
            rows = []
            for stat in stats:
                rows.append(
                    [
                        stat.depth,
                        stat.parameter,
                        _format_size(stat.numel),
                        _format_shape(stat.shape),
                    ]
                )

            # Add a column for the how much each row contributes to the total
            columns.append("contrib")
            for row, stat in zip(rows, stats, strict=True):
                row.append(f"{100 * stat.numel / total_params:.2f} %")
        else:
            rows = map(tuple, stats)

        # Return the results as a Pandas DataFrame
        return {
            col: [row[i] for row in rows] for i, col in enumerate(columns)
        }, total_params


class FlopCountSubcommand(Subcommand, name="flops"):
    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        _add_loader_type_args(prs)

    @classmethod
    def main(cls, args: argparse.Namespace):
        logger.info("Analysing model FLOPs...")
        cls.count_flops(
            model,
            loader,
            device,
            backend="pytorch",
            verbose=False,
            print_per_layer_stat=True,
        )

    @staticmethod
    def count_flops(
        model: ModelBase,
        loader: torch.utils.data.DataLoader,
        device: torch.types.Device,
        backend="pytorch",
        verbose=False,
        print_per_layer_stat=True,
    ) -> None:
        from ptflops import get_model_complexity_info

        # from fvcore.nn import FlopCountAnalysis
        # inputs = next(iter(loader))
        # inputs = inputs.to(device)
        # model_adapter = ModelAdapter(model, inputs, allow_non_tensor=True)
        # flops = FlopCountAnalysis(model_adapter, inputs=model_adapter.flattened_inputs)
        # logger.info("Running FLOP analysis...")
        # logger.info("Total FLOPs:\n%s", flops.total())

        model = model.to(device)

        # Get a single batch of data to use as a template
        inputs = next(iter(loader))[:1].to(device)
        inputs_shape = tuple(inputs.captures.images.shape)
        inputs = model.select_inputs(inputs, device)

        # Deteraine the forward arguments such that we can provide inputs as keywords
        forward_args = inspect.signature(model.forward).parameters

        def randomize(value):
            try:
                return torch.rand_like(value)
            except RuntimeError:
                return torch.rand_like(value.float()).to(value.dtype)

        def inputs_constructor(size: tuple[int, int]) -> dict[str, T.Any]:
            res = {}
            for param, value in zip(forward_args, inputs, strict=False):
                if isinstance(value, torch.Tensor):
                    res[param] = randomize(value)
                elif isinstance(value, TensorDictBase):
                    res[param] = value.apply(randomize)
                elif value is None:
                    res[param] = None
                elif hasattr(value, "clone"):
                    res[param] = value.clone()
                else:
                    res[param] = value

            return res

        verbose = False

        with device:
            macs, params = get_model_complexity_info(
                model,
                inputs_shape,
                as_strings=True,
                input_constructor=inputs_constructor,
                backend=backend,
                print_per_layer_stat=print_per_layer_stat,
                verbose=verbose,
            )
            if macs is not None:
                print("{:<30}  {:<8}".format("Computational complexity: ", macs))
            if params is not None:
                print("{:<30}  {:<8}".format("Number of parameters: ", params))


class MemorySubcommand(Subcommand, name="memory"):
    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        _add_loader_type_args(prs)

    @classmethod
    def main(cls, args: argparse.Namespace):
        logger.info("Analysing model memory...")
        cls.analyse_memory(
            model,
            loader,
            handler,
            iterations=args.iterations,
            path_export=path_export,
        )

    @staticmethod
    def _analyse_memory(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        handler,
        *,
        iterations: int,
        path_export: expath.locate,
    ) -> None:
        model = model.cuda()

        logger.info("Warming up...")
        loader_iter = iter(loader)
        for _ in range(11):
            data = next(loader_iter).cuda()
            handler(model, data)

        logger.info("Recording %d iterations...", iterations)
        torch.cuda.memory._record_memory_history()
        for _ in range(iterations):
            data = next(loader_iter).cuda()
            handler(model, data)

        path_snapshot = path_export / "cuda_snapshot.pkl"
        torch.cuda.memory._dump_snapshot(str(path_snapshot))

        logger.info(
            "Upload the snapshot to https://pytorch.org/memory_viz using local path: %s",
            path_snapshot,
        )


class TraceSubcommand(Subcommand, name="trace"):
    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        _add_loader_type_args(prs)

    @classmethod
    def main(cls, args: argparse.Namespace):
        logger.info("Analysing model trace...")
        cls.analyse_trace(
            model,
            loader,
            handler,
            iterations=args.iterations,
            path_export=path_export,
        )

    @staticmethod
    def trace(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        handler,
        *,
        iterations: int,
        path_export: expath.locate,
    ) -> None:
        logger.info("Profiling model")

        loader_iter = iter(loader)

        with torch.profiler.profile(
            profile_memory=True,
            with_flops=True,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(path_export)),
        ) as prof:
            for _ in tqdm(iterable=range(iterations)):
                data = next(loader_iter)
                handler(model, data)
                prof.step()
        assert prof is not None

        # Print the results
        logger.info(
            "Key averages sorted by `self_cuda_time_total`:\n\n%s\n",
            prof.key_averages().table(sort_by=args.sort_by, row_limit=-1),
        )

        with open(path_export / "key_averages.txt", "w") as f:
            f.write(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )
        prof.export_stacks(str(path_export / "stacks.txt"))

        logger.info("Exporting chrome trace file...")
        prof.export_chrome_trace(str(path_export / "trace.json"))

        logger.debug("Finished profiling session.")


def _main(args):
    config: T.Any = args.config
    path_export = _find_session_path(config)

    logger.info("Saving results to %s", path_export)

    engine = create_engine(config)
    model = create_model(config, state=args.weights)
    model.to(engine.xlr.device)

    if args.parameter_count:
        _analyse_params(model)

    # Exit early when no further profiling is requested
    if not any([args.flops, args.memory, args.trace]):
        exit(0)

    # Prepare model and loader
    if args.training:
        logger.info("Profiling in TRAINING mode")
        handler = engine.run_training_step
        model.train()

        i_stage = int(args.loader) if args.loader is not None else 0
        logger.info("Preparing dataset for stage %d (--loader)", i_stage)

        loader, info = create_dataset(config, variant=i_stage, training=True)
    elif args.inference:
        logger.info("Profiling in EVALUATION mode")
        handler = engine.run_inference_step
        model.eval()

        logger.info("Preparing dataset for evaluation suite %s (--loader)", args.loader)
        loader, info = create_dataset(config, variant=args.loader, batch_size=1)
    else:
        logger.error(
            "Unknown mode; provide either the `--training` or `--inference` flag"
        )
        exit(1)


if __name__ == "__main__":
    command.root("profile")
