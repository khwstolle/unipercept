"""Evaluation entry point."""

from __future__ import annotations

import argparse
import os

import unipercept as up
from unipercept.cli._command import command
from unipercept.cli._config import ConfigFileContentType as config_t

_logger = up.log.get_logger()


KEY_SESSION_ID = "session_id"


@command(help="evaluate a trained model", description=__doc__)
@command.with_config
def evaluate(p: argparse.ArgumentParser):
    p.add_argument("--no-jit", action="store_true", help="disable JIT compilation")
    p.add_argument(
        "--weights",
        "-w",
        type=str,
        help=(
            "Path to model weights, ignoring the `WEIGHTS` field in the config file."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=up.expath.locate,
        help=(
            "Path to output directory. If not provided, the scratch directory is used."
        ),
    )
    p.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Skip inference and only run evaluation on existing model outputs. "
            "Requires the `output` argument to be provided and the model inference to "
            "be run previously to generate the outputs."
        ),
    )
    p.add_argument(
        "--suite",
        "-S",
        nargs="+",
        type=str,
        help=(
            "Evaluation suite(s) to run. If not provided, all enabled suites defined "
            "in the config are run."
        ),
    )

    return _main


def _step(args) -> config_t:
    if args.no_jit:
        _logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"

    up.state.barrier()  # Ensure the config file is not created before all processes validate its existence
    return args.config


def _main(args):
    config = _step(args)
    engine = up.create_engine(config)
    if args.skip_inference:
        if not args.output or not args.output.is_dir():
            msg = (
                "The `output` argument must be provided and point to an existing "
                "directory if `skip_inference` is enabled."
            )
            raise ValueError(msg)
        model_factory = None
    else:
        model_factory = up.create_model_factory(config, weights=args.weights or None)

    suites = args.suite if args.suite is not None and len(args.suite) > 0 else None
    try:
        results = engine.run_evaluation(
            model_factory,
            suites=suites,
            path=up.expath.locate(args.output) if args.output is not None else None,
        )
        _logger.info(
            "Evaluation results: \n%s", up.log.create_table(results, format="long")
        )
    except KeyboardInterrupt:
        _logger.info("Evaluation interrupted")


if __name__ == "__main__":
    command.root("evaluate")
