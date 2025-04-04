""" """

import contextlib
import functools
import gc
import typing as T

import torch

CUDA_OOM_ERROR: T.Final[str] = "CUDA out of memory. "


@contextlib.contextmanager
def ignore_cuda_oom():
    """
    A context which ignores CUDA OOM errors.
    """
    try:
        yield
    except RuntimeError as e:
        if CUDA_OOM_ERROR in str(e):
            pass
        else:
            raise


@T.overload
def retry_if_cuda_oom[_C: T.Callable](__fn: _C, /, *, cpu_fallback: bool) -> _C: ...


@T.overload
def retry_if_cuda_oom[_C: T.Callable](
    __fn: _C | None = None, /, *, cpu_fallback: bool = ...
) -> T.Callable[[_C], _C]: ...


def retry_if_cuda_oom[_C: T.Callable](
    __fn: _C | None = None,
    /,
    cpu_fallback: bool = True,
) -> _C | T.Callable[[_C], _C]:
    """
    Decorator that retries a function if it raises OOM errors on the GPU.

    Parameters
    ----------
    fn : Callable | None
        The function to decorate.
    cpu_fallback : bool
        If True, the function will be called on the CPU if it fails on the GPU attempts.

    Returns
    -------
    Callable
        The decorated function.
    """

    if __fn is None:
        return lambda fn: retry_if_cuda_oom(fn, cpu_fallback=cpu_fallback)

    @functools.wraps(__fn)
    def fn_with_retry(*args, **kwargs):
        if torch.compiler.is_compiling():
            return __fn(*args, **kwargs)
        with ignore_cuda_oom():
            return __fn(*args, **kwargs)
        torch.cuda.empty_cache()
        if cpu_fallback:
            with ignore_cuda_oom():
                return __fn(*args, **kwargs)
            cpu_args = [_maybe_to_cpu(x) for x in args]
            cpu_kwargs = {k: _maybe_to_cpu(v) for k, v in kwargs.items()}
            return __fn(*cpu_args, **cpu_kwargs)
        return __fn(*args, **kwargs)

    return fn_with_retry


def _maybe_to_cpu[_T](x: _T) -> _T:
    try:
        like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")  # type: ignore[attr-defined]
    except AttributeError:
        like_gpu_tensor = False
    if like_gpu_tensor:
        return x.to(device="cpu")  # type: ignore[attr-defined]
    return x


def release_memory():
    """
    Attempt to free memory from the GPU, XPU, NPU, or MPS.
    """
    gc.collect()
    # torch.xpu.empty_cache()
    # torch.mlu.empty_cache()
    # torch.npu.empty_cache()
    # torch.mps.empty_cache()
    torch.cuda.empty_cache()


_OOM_EXCEPTION_PATTERNS = [
    "CUDA out of memory.",
    "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
    "DefaultCPUAllocator: can't allocate memory",
]


def check_oom_exception(e: Exception) -> bool:
    """
    Checks whether an exception is CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory.
    """
    if isinstance(e, RuntimeError) and len(e.args) == 1:
        return any(err in e.args[0] for err in _OOM_EXCEPTION_PATTERNS)
    return False


@T.overload
def find_executable[**_P, _R](
    __fn: T.Callable[T.Concatenate[int, _P], _R], /, *, max_iter: int
) -> T.Callable[_P, _R]: ...
@T.overload
def find_executable[**_P, _R](
    __fn: None = None, /, *, max_iter: int
) -> T.Callable[[T.Callable[T.Concatenate[int, _P], _R]], T.Callable[_P, _R]]: ...


def find_executable[**_P, _R](
    __fn: T.Callable[T.Concatenate[int, _P], _R] | None = None, /, *, max_iter: int = 10
) -> (
    T.Callable[_P, _R]
    | T.Callable[[T.Callable[T.Concatenate[int, _P], _R]], T.Callable[_P, _R]]
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions
    related to out-of-memory or CUDNN, the function will be retried with the first
    integer argument increased by one.

    This will continue until the function executes successfully or the user interrupts
    the process.

    Parameters
    ----------
    fn :
        The function to decorate. If not provided, the decorator will return a partial
        function that can be called with the function to decorate.
    max_iterations : int | None
        The maximum number of iterations to attempt before raising an error.

    Returns
    -------
    Callable[[int, ...], ...] :
        The decorated function or a partial function that can be called with the function to
        decorate.

    Raises
    ------
    StopIteration :
        If the maximum number of iterations is reached.

    ```
    """
    if __fn is None:
        return lambda fn: find_executable(fn, max_iter=max_iter)

    assert max_iter is None or max_iter >= 1, f"{max_iter=} <= 1"

    n = 0

    @functools.wraps(__fn)
    def _fn_wrap(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        nonlocal n
        release_memory()
        while True:
            if n >= max_iter:
                msg = f"Max iterations ({max_iter}) reached, unable to execute {__fn}"
                raise StopIteration(msg)
            try:
                return __fn(n, *args, **kwargs)
            except Exception as e:
                if check_oom_exception(e):
                    n += 1
                    release_memory()
                    continue
                raise

    return _fn_wrap
