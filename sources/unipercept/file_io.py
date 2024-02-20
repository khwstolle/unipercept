"""
Implements a path manager using ``iopath``.
"""

import os
import pathlib
import typing as T

from iopath.common.file_io import (
    HTTPURLHandler,
    OneDrivePathHandler,
    PathHandler,
    PathManagerFactory,
)

from .types import Pathable
from .utils.iopath_handlers import (
    EnvironPathHandler,
    MetadataPathHandler,
    WandBArtifactHandler,
)

_manager: T.Final = PathManagerFactory.get(defaults_setup=False)


class Path(type(pathlib.Path()) if not T.TYPE_CHECKING else pathlib.Path):
    """
    Extends `pathlib.Path` to work with `iopath.common.file_io.PathManager`.
    """

    def __new__(
        cls, path: str | os.PathLike | pathlib.Path, *args, force: bool = False
    ) -> pathlib.Path:
        if isinstance(path, pathlib.Path | pathlib.WindowsPath | pathlib.PosixPath):
            return path
        if isinstance(path, str):
            path = _manager.get_local_path(path, force=force)
        return pathlib.Path(path, *args)

    # def __getattr__(self, name: str) -> T.Any:
    #     """
    #     Forward all other attribute accesses to the underlying `pathlib.Path` object.
    #     """
    #     return getattr(Path, name)


#############
# Utilities #
#############


def join(base: str | Path, *other: str | Path) -> str:
    """
    Joins paths using the path manager.

    Parameters
    ----------
    *paths : str
        The paths to join.

    Returns
    -------
    str
        The joined path.

    """
    base = str(base)
    return os.path.join(base, *map(str, other))


#################
# Path handlers #
#################
# Register handlers with the manager object
for h in (
    OneDrivePathHandler(),
    HTTPURLHandler(),
    WandBArtifactHandler(),
    MetadataPathHandler("configs://", "unipercept.configs"),
    EnvironPathHandler(
        "//datasets/",
        "UP_DATASETS",
        "UNIPERCEPT_DATASETS",
        "UNICORE_DATASETS",
        "DETECTRON2_DATASETS",
        "D2_DATASETS",
        default="~/datasets",
    ),
    EnvironPathHandler(
        "//cache/",
        "UP_CACHE",
        "UNIPERCEPT_CACHE",
        "UNICORE_CACHE",
        default="~/.cache/unipercept",
    ),
    EnvironPathHandler(
        "//output/",
        "UP_OUTPUT",
        "UNIPERCEPT_OUTPUT",
        "UNICORE_OUTPUT",
        default="./output",
    ),
    EnvironPathHandler(
        "//scratch/",
        "UP_SCRATCH",
        "UNIPERCEPT_SCRATCH",
        "UNICORE_SCRATCH",
        default=None,
    ),
):
    _manager.register_handler(h, allow_override=False)

_exports: frozenset[str] = frozenset(
    fn_name for fn_name in dir(_manager) if not fn_name.startswith("_")
)


def get_local_path(path: Pathable, force: bool = False, **kwargs: T.Any) -> str:
    return _manager.get_local_path(str(path), force=force, **kwargs)


def __getattr__(name: str):
    if name in _exports:
        return getattr(_manager, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    global _exports
    return __all__ + list(_exports)


if T.TYPE_CHECKING:

    def opent(
        path: str, mode: str = "r", buffering: int = 32, **kwargs: T.Any
    ) -> T.Iterable[T.Any]: ...

    def open(
        path: str, mode: str = "r", buffering: int = -1, **kwargs: T.Any
    ) -> T.IO[str] | T.IO[bytes]: ...

    def opena(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        callback_after_file_close: T.Callable[[None], None] | None = None,
        **kwargs: T.Any,
    ) -> T.IO[str] | T.IO[bytes]: ...

    def async_join(*paths: str, **kwargs: T.Any) -> bool: ...

    def async_close(**kwargs: T.Any) -> bool: ...

    def copy(
        src_path: str, dst_path: str, overwrite: bool = False, **kwargs: T.Any
    ) -> bool: ...

    def mv(src_path: str, dst_path: str, **kwargs: T.Any) -> bool: ...

    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs: T.Any
    ) -> None: ...

    def exists(path: str, **kwargs: T.Any) -> bool: ...

    def isfile(path: str, **kwargs: T.Any) -> bool: ...

    def isdir(path: str, **kwargs: T.Any) -> bool: ...

    def ls(path: str, **kwargs: T.Any) -> list[str]: ...

    def mkdirs(path: str, **kwargs: T.Any) -> None: ...

    def rm(path: str, **kwargs: T.Any) -> None: ...

    def symlink(src_path: str, dst_path: str, **kwargs: T.Any) -> bool: ...

    def set_cwd(path: str | None, **kwargs: T.Any) -> bool: ...

    def register_handler(handler: PathHandler, allow_override: bool = True) -> None: ...

    def set_strict_kwargs_checking(enable: bool) -> None: ...

    def set_logging(enable_logging=True) -> None: ...
