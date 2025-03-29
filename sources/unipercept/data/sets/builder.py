r"""
Datasets
========

This module provides classes for building datasets from various sources. We define a
:class:`Dataset` as a collection of data that can be accessed by a :class:`DataLoader`.

A dataset is composed of the following components:

- A manifest of the contents
- A metadata object describing the dataset
- A unique identifier for a specific instance of the dataset
"""

import abc
from dataclasses import field
from typing import Protocol, TypedDict

import expath


class DatasetMetadata:
    r"""
    Represents a metadata object for a dataset.
    """

    version: str
    config: dict[str, str] = field(default_factory=dict)


class DatasetManifest:
    r"""
    Represents a manifest object for a dataset.
    """

    version: str


class DatasetConfig(TypedDict):
    r"""
    Represents a configuration object for a dataset.
    """


class DatasetProtocol[Manifest: DatasetManifest, Metadata: DatasetMetadata, Item](
    Protocol
):
    r"""
    Represents a dataset that can be initialized from a manifest and meta-data.
    """

    @property
    def version(self) -> str:
        r"""
        The version of the dataset handler.
        """
        from unipercept import __version__ as version

        return version

    def __init__(self, *sources: Manifest, metadata: Metadata): ...

    def __getitem__(self, key: int) -> Item: ...

    def __len__(self) -> int: ...

    def __iter__(self): ...


class DatasetBuilder[
    Config: DatasetConfig,
    Manifest: DatasetManifest,
    Metadata: DatasetMetadata,
](abc.ABC):
    r"""
    Defines a builder class for creating instances of a dataset.
    """

    __slots__ = ("config",)

    config: Config

    def __init__(self, name: str, *variant: str, config: Config):
        self.name = name
        self.variant = variant
        self.config = config

    @abc.abstractmethod
    def prepare(self, *, interactive: bool, **kwargs) -> None:
        r"""
        Perform basic checks to ensure the dataset can be built,
        this is usually ran before any io-bound operations or
        downloads are performed.
        """

    @abc.abstractmethod
    def build(self, *, to: expath.locate, **kwargs) -> tuple[Manifest, Metadata]:
        r"""
        Build the dataset and write the manifest and metadata to disk.
        """

    def __call__(
        self,
        *,
        interactive: bool = False,
        to: expath.PathType = "//unipercept/cachedatasets/{key}",
    ) -> tuple[Manifest, Metadata]:
        path_dataset = expath.locate(to)

        self.prepare(interactive=interactive)
        self.build(to=expath.locate(to))
