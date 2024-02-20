"""
Implements a simple registry with type checks and key canonicalization.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Final, override

__all__ = ["IndexedRegistry", "AnonymousRegistry"]


class IndexedRegistry[_T, _I]:
    __slots__ = ("_memory", "_infer_fn", "_check_fn", "_exist_ok")

    _memory: Final[OrderedDict[str, _T]]
    _infer_fn: Final[Callable[[str | _I], str] | None]
    _check_fn: Final[Callable[[Any], bool] | None]
    _exist_ok: Final[bool]

    def __init__(
        self,
        *,
        exist_ok: bool = False,
        check: Callable[[_T], bool] | None = None,
        infer_key: Callable[[str | _I], str] | None = None,
    ) -> None:
        self._memory = OrderedDict()
        self._infer_fn = infer_key
        self._check_fn = check
        self._exist_ok = exist_ok

    @property
    def exist_ok(self) -> bool:
        return self._exist_ok

    def check(self, obj: _T, /, *, raises: bool = False) -> bool:
        if self._check_fn is not None:
            valid = self._check_fn(obj)
            if not valid and raises:
                msg = f"Object {obj} is not valid (check failed)."
                raise ValueError(msg)
            return valid
        return True

    def infer_key(self, obj: str | _I, /) -> str:
        if self._infer_fn is not None:
            return self._infer_fn(obj)
        if isinstance(obj, str):
            return obj
        msg = f"Cannot infer ID for object {obj} ({type(obj)})."
        raise ValueError(msg)

    def keys(self):
        return self._memory.keys()

    def values(self):
        return self._memory.values()

    def items(self):
        return self._memory.items()

    def register(self, key: str, /) -> Callable[[_T], _T]:
        key = self.infer_key(key)

        def decorator(value: _T) -> _T:
            self[key] = value
            return value

        return decorator

    def __getitem__(self, __key: str, /) -> _T:
        return self._memory[self.infer_key(__key)]

    def __setitem__(self, __key: str, value: _T, /) -> None:
        self._memory[self.infer_key(__key)] = value

    def __delitem__(self, __key: str, /) -> None:
        del self._memory[self.infer_key(__key)]

    def __contains__(self, __key: str, /) -> bool:
        return self.infer_key(__key) in self._memory

    def __iter__(self) -> Iterator[str]:
        yield from self._memory.keys()

    def __len__(self) -> int:
        return len(self._memory)


class AnonymousRegistry[_T]:
    """
    Simple registry class to keep track of registered unnamed (anonymous) objects.
    """

    __slots__ = ["_collection", "_check_fn", "_exist_ok"]
    _collection: Final[set[_T]]
    _check_fn: Final[Callable[[Any], bool] | None]
    _exist_ok: Final[bool]

    def __init__(
        self,
        *,
        check: Callable[[Any], bool] | None = None,
        exist_ok: bool = False,
    ) -> None:
        self._collection = set()
        self._check_fn = check
        self._exist_ok = exist_ok

    @property
    def exist_ok(self) -> bool:
        return self._exist_ok

    def items(self) -> frozenset[_T]:
        return frozenset(self._collection)

    def check(self, item: Any, /, *, raises: bool = False) -> bool:
        r"""
        Checks if an item is valid.
        """
        if self._check_fn is not None:
            valid = self._check_fn(item)
            if not valid and raises:
                msg = f"Item {item} is not valid (check failed)."
                raise ValueError(msg)
            return valid
        return True

    def add(
        self, item: _T, /, *, skip_exist: bool = False, skip_check: bool = False
    ) -> None:
        r"""
        Adds an item to the registry.
        """
        if not skip_check:
            self.check(item, raises=True)

        if item in self:
            if skip_exist:
                return
            if not self._exist_ok:
                msg = f"Item {item} already exists in the registry."
                raise ValueError(msg)

        self._collection.add(item)

    def register[_I: _T](
        self,
        /,
        *,
        skip_exist: bool = False,
        skip_check: bool = False,
    ) -> Callable[[_I], _I]:
        r"""
        Decorator to register an item, see :meth:`add`.
        """
        def decorator(item: _I) -> _I:
            self.add(item, skip_exist=skip_exist, skip_check=skip_check)
            return item

        return decorator

    def __contains__(self, item: Any) -> bool:
        return item in self._collection

    def __iter__(self) -> Iterator[_T]:
        return iter(self._collection)

    def __len__(self) -> int:
        return len(self._collection)

    @override
    def __hash__(self) -> int:
        return hash(self._collection)

    @override
    def __dir__(self) -> list[str]:
        return list(map(str, self._collection))

    @override
    def __repr__(self) -> str:
        items = ", ".join(map(repr, self))
        return f"<{self.__class__.__name__}>{{{items}}}"

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {len(self)} items"
