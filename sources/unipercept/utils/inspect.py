from __future__ import annotations

import os
import sys
import types
import typing as T


def calling_module_name(
    frames: int = 0, left: int | None = None, strict: bool = True
) -> str:
    """
    Return the name of the module that called the current function from the perspective
    of the function that calls this function.

    Parameters
    ----------
    frames
        The number of frames to go back in the stack offset by 2, by default 0.
    left
        The number nested modules to return, e.g. when the module name is `a.b.c` and `left=1` the return value would
        be `a.b`. By default `None`, which returns the full module name.
    strict
        If the module is not found, raise an error.
        If `False`, return the name of the module that called the current function is returned.

    Returns
    -------
    The name of the module that called the current function.

    """
    import inspect

    frm = inspect.stack()[2 + frames]
    mod = inspect.getmodule(frm[0])
    if mod is None:
        if strict:
            msg = f"Could not find module for {frm.filename}, which was called from {frm.function}"
            raise ModuleNotFoundError(msg)
        return frm.filename.split(".")[0]

    name = mod.__name__
    if left is not None:
        name = ".".join(name.split(".")[: left + 1])

    return name


def caller_identity() -> tuple[str, tuple[str, int, str]]:  # type: ignore
    """
    Identify the caller by a name and hashable key.

    Returns
    -------
    tuple[str,str]
        Module name of the caller and a hashable key
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", __file__) not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "<main>"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def generate_path(obj: T.Any) -> str:
    """
    Inverse of ``locate()``. Generates the fully qualified name of an object.
    Handles cases where the object is not directly importable, e.g. due to
    nested classes or private modules.

    The generated path is simplified by removing redundant module parts, e.g.
    ``module.submodule._impl.class`` may become ``module.submodule.class`` if
    the later also resolves to the same object.

    Bound methods are supported by inspecting the ``__self__`` attribute.

    Parameters
    ----------
    obj
        The object to generate the path for.

    Returns
    -------
    str
        The fully qualified name of the object.
    """

    def __check(path: str, obj: T.Any) -> bool:
        # Check if the path resolves to the same object
        try:
            check_ok = locate_object(path) is obj
        except ImportError:
            check_ok = False
        return check_ok

    try:
        self = obj.__self__
    except AttributeError:
        self = None

    if self is not None:
        self_path = generate_path(self)
        return f"{self_path}.{obj.__name__}"

    module, qualname = obj.__module__, obj.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        path = f"{prefix}.{qualname}"
        if __check(path, obj):
            return path

    # Default to the full path plus qualname
    path = f"{module}.{qualname}"
    if not __check(path, obj):
        msg = f"Cannot generate path for object {obj}!"
        raise ImportError(msg)

    return path


def locate_object(path: str) -> T.Any:
    """
    Dynamically locates and returns an object by its fully qualified name.

    Parameters
    ----------
    name (str):
        The fully qualified name of the object to locate.

    Returns
    -------
    Any:
        The located object.

    Raises
    ------
    ImportError
        If the object cannot be located.
    """
    import importlib
    import pydoc

    obj = pydoc.locate(path)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {path}!") from e
        else:
            obj = _locate(path)  # it raises if fails
    if path == "":
        raise ImportError("Empty path")
    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = importlib.import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, types.ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = importlib.import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj
