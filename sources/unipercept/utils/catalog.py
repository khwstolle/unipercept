"""
Implements a simple data manager for registering datasets and their info functions.
"""

import typing as T
from functools import partial
from importlib.metadata import entry_points

import regex as re

from unipercept.utils.registry import IndexedRegistry

__all__ = ["Catalog", "CatalogFromPackageMetadata"]


_DEFAULT_ID_PATTERN: T.Final[re.Pattern] = re.compile(r"^[a-z\d\-]+$")
_DEFAULT_ID_SEPARATOR: T.Final[str] = "/"


class Catalog[_D_co, _I_co]:
    """
    Data manager for registering datasets and their info functions.
    """

    __slots__ = (
        "_variant_sep",
        "_id_regex",
        "_require_info",
        "_info_registry",
        "_data_registry",
    )

    def __init__(
        self,
        *,
        id_pattern: re.Pattern[str] = _DEFAULT_ID_PATTERN,
        variant_separator: str = _DEFAULT_ID_SEPARATOR,
        require_info: bool = True,
        data_abc: type[_D_co] = object,
        info_abc: type[_I_co] = object,
    ):
        """
        Parameters
        ----------
        id_pattern : re.Pattern
            The pattern to use for validating dataset IDs.
        variant_separator : str
            The separator to use for separating dataset IDs from variant IDs.
        """
        self._require_info: T.Final[bool] = require_info
        self._variant_sep: T.Final[str] = variant_separator
        self._id_regex: T.Final[re.Pattern[str]] = id_pattern

        self._info_registry = T.cast(
            IndexedRegistry[T.Callable[..., _I_co], str],
            IndexedRegistry(
                infer_key=self.parse_key, check=partial(issubclass, info_abc)
            ),
        )
        self._data_registry = T.cast(
            IndexedRegistry[type[_D_co], str],
            IndexedRegistry(
                infer_key=self.parse_key, check=partial(issubclass, data_abc)
            ),
        )

    def parse_key(self, key: str | type[_D_co]) -> str:
        """
        Convert a string or class to a canonical ID.

        Parameters
        ----------
        other : Union[str, type]
            The string or class to convert.

        Returns
        -------
        str
            The canonical ID.
        """

        if not isinstance(key, str):
            if hasattr(key, "__name__"):
                name = key.__name__.lower()  # type: ignore
            else:
                msg = f"Cannot convert {key} to a canonical ID, as it has no name."
                raise ValueError(msg)
        else:
            name = key.lower()
        match = self._id_regex.search(name)
        if not match:
            raise ValueError(f"{key} ({name}) does not match {self._id_regex.pattern}")
        return match.group()

    def split_query(self, query: str) -> tuple[str, frozenset[str]]:
        """
        Split a query into a dataset ID and a variant ID.
        """
        if self._variant_sep not in query:
            return query, []
        key, *variant = query.split(self._variant_sep, maxsplit=1)
        return key, variant

    # ------------- #
    # MAIN ELEMENTS #
    # ------------- #

    def register(
        self, id: str | None = None, *, info: T.Callable[..., _I_co] | None = None
    ) -> T.Callable[[type[_D_co]], type[_D_co]]:
        """
        Register a dataset.

        Parameters
        ----------
        id : str | None
            The ID to register the dataset with. If None, the dataset class name will
            be used (flattened and converted to snake_case).
        """

        if info is None and self._require_info:
            msg = "No info function provided, but required."
            raise ValueError(msg)

        def wrapped(ds: type[_D_co]) -> type[_D_co]:
            key = id or self.parse_key(ds)
            self._data_registry[key] = ds
            if callable(info):
                self._info_registry[key] = info
            elif self._require_info:
                msg = f"Invalid info function: {info}"
                raise TypeError(msg)

            return ds

        return wrapped

    def get_data(self, query: str) -> type[_D_co]:
        """
        Return the dataset class for the given dataset ID.
        """
        return self._data_registry[query]

    def keys_data(self) -> frozenset[str]:
        return frozenset(self._data_registry.keys())

    # ---- #
    # Info #
    # ---- #

    def register_info[**_P](
        self,
        key: str,
        /,
    ) -> T.Callable[[T.Callable[_P, _I_co]], T.Callable[_P, _I_co]]:
        """
        Register a dataset.

        Parameters
        ----------
        id : Optional[str]
            The ID to register the dataset with. If None, the dataset class name will
            be canonicalized using ``canonicalize_id``.
        """

        def wrapped(info: T.Callable[_P, _I_co]) -> T.Callable[_P, _I_co]:
            self._info_registry[key] = info

            return info

        return wrapped

    def get_info(self, query: str) -> _I_co:
        """
        Return the info for the given dataset ID.
        """
        _id, variant = self.split_query(query)
        return self._info_registry[_id](*variant)

    def get_info_at(self, query: str, key: str) -> T.Any:
        """
        Return the info for the given dataset ID.
        """
        _id, variant = self.split_query(query)
        return self._info_registry[_id](*variant)[key]  # type: ignore

    def keys_info(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return frozenset(self._info_registry.keys())

    # ----------- #
    # Generic API #
    # ----------- #

    @T.overload
    def get(self, query: str, *, info: T.Literal[False]) -> type[_D_co]: ...

    @T.overload
    def get(self, query: str, *, info: T.Literal[True]) -> _I_co: ...

    def get(self, query: str, *, info: bool = False) -> type[_D_co] | _I_co:
        return self.get_info(query) if info else self.get_data(query)

    def __getitem__(self, query: str) -> type[_D_co]:
        return self.get(query, info=False)

    def __getitems__(self, queries: frozenset[str]) -> list[type[_D_co]]:
        return [self.get(query, info=False) for query in queries]

    def keys(self, *, info: bool = False) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return self.keys_info() if info else self.keys_data()

    def __iter__(self):
        return iter(self.keys())


class CatalogFromPackageMetadata[_D_co, _I_co](Catalog[_D_co, _I_co]):
    """
    Variant of :class:`DataManager` that reads registered items from the metadata
    registered through ``importlib.metadata``.

    Notes
    -----
    This comes with the restriction that each registered ID can only reference both
    a dataset and an info function.
    """

    __slots__ = ("group",)

    def __init__(
        self,
        *,
        group: str,
        **kwargs,
    ):
        """
        Parameters
        ----------
        group : str
            The metadata group to read from.
        **kwargs
            See :class:`DataManager`.
        """
        super().__init__(**kwargs)

        self.group = group

    def entrypoint_names(self) -> frozenset[str]:
        """
        Returns a list of all registered keys from ``importlib.metadata`` with
        ``self.group``.
        """
        return frozenset(entry_points(group=self.group).names)

    def load_entrypoint(self, key: str) -> type[_D_co]:
        """
        Return the entrypoint for the given dataset ID.
        """
        try:
            return entry_points(group=self.group)[key].load()
        except KeyError as err:
            msg = (
                f"Could not find entrypoint for dataset ID {key=}. "
                f"Choose from: {self.entrypoint_names()}"
            )
            raise KeyError(msg) from err

    @T.override
    def get_data(self, query: str) -> type[_D_co]:
        """
        Return the dataset class for the given dataset ID.
        """
        try:
            return self._data_registry[query]
        except KeyError:
            return self.load_entrypoint(query)

    @T.override
    def keys_data(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        reg_ds = set(super().keys_data())
        meta_ds = set(self.entrypoint_names())

        return frozenset(reg_ds | meta_ds)

    def _maybe_load_entrypoint(self, query: str, *, raises: bool = True) -> None:
        _id, _ = self.split_query(query)
        if _id in super().keys_info():
            return
        if _id not in self.entrypoint_names():
            return
        ds = self.load_entrypoint(_id)  # this should trigger registration
        if _id not in super().keys_info() and raises:
            msg = (
                f"Could not find info for dataset ID {_id=}. "
                f"While {_id=} is a valid entrypoint, loading it did not yield a "
                f"registered info function. Found entrypoint: {ds}"
            )
            raise KeyError(msg)

    @T.override
    def get_info(self, query: str) -> _I_co:
        """
        Return the info for the given dataset ID.
        """
        self._maybe_load_entrypoint(query)

        _id, variant = self.split_query(query)
        return self._info_registry[_id](*variant)

    @T.override
    def get_info_at(self, query: str, key: str) -> T.Any:
        """
        Return the info for the given dataset ID.
        """
        self._maybe_load_entrypoint(query)
        return super().get_info_at(query, key)

    @T.override
    def keys_info(self) -> frozenset[str]:
        """
        Return a frozenset of all registered dataset IDs.
        """
        return super().keys_info() | self.entrypoint_names()
