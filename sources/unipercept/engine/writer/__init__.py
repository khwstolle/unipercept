r"""
This submodule contains writer classes for storing the data that comes out of a model.

Important considerations when selecting the type of writer to use are:
    - The amount of data that must be saved
    - The available memory and storage on the machine
    - The amount of concurrent processes that run the model
    - The computational resources available
"""

# from . import memmap
from ._base import *
