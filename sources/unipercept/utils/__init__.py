"""
A collection of reusable utilities for UniPercept
"""

from unipercept.utils.module import lazy_module_factory

__all__ = []
__lazy__ = [
    "abbreviate",
    "box",
    "catalog",
    "check",
    "cuda",
    "datainfo",
    "dataset",
    "decorators",
    "descriptors",
    "dicttools",
    "flops",
    "formatter",
    "frozendict",
    "function",
    "generics",
    "image",
    "inspect",
    "iopath_handlers",
    "mask",
    "matchable",
    "memory",
    "missing",
    "module",
    "pickle",
    "quantile",
    "registry",
    "seed",
    "signal",
    "status",
    "status",
    "string",
    "tensor",
    "tensorclass",
    "time",
    "ulid",
]
__getattr__, __dir__ = lazy_module_factory(__name__, __lazy__)

del lazy_module_factory
