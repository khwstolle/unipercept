"""
This module implements building blocks for building neural networks in PyTorch.
"""

from unipercept.utils.module import lazy_module_factory

__lazy__ = ["huggingface", "slurm_integration", "wandb_integration"]
__all__ = []
__getattr__, __dir__ = lazy_module_factory(__name__, __lazy__)

del lazy_module_factory
