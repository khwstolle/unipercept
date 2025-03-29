r"""Path handlers
------------------

Used to define the paths to the cache, output, and datasets directories,
see `pyproject.toml`
"""

from functools import partial

from expath.handlers.env import EnvironPathHandler

build_cache_path_handler = partial(
    EnvironPathHandler,
    "//unipercept/cache",
    "UP_CACHE",
    "UNIPERCEPT_CACHE",
    default="~/.cache/unipercept",
)
build_datasets_path_handler = partial(
    EnvironPathHandler,
    "//unipercept/datasets",
    "UP_DATASETS",
    "UNIPERCEPT_DATASETS",
    default="~/datasets",
)
build_output_path_handler = partial(
    EnvironPathHandler,
    "//unipercept/output",
    "UP_OUTPUT",
    "UNIPERCEPT_OUTPUT",
    default="./output",
)
build_scratch_path_handler = partial(
    EnvironPathHandler,
    "//unipercept/scratch/",
    "UP_SCRATCH",
    "UNIPERCEPT_SCRATCH",
    default=None,
)
