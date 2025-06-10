from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Sequence, TypeVar

T = TypeVar("T")


class MergeStrategy(ABC, Generic[T]):
    """Base strategy for merging results from parallel nodes."""

    @abstractmethod
    def merge(self, results: Sequence[T]) -> T:
        """Merge a sequence of results into a single value."""
        raise NotImplementedError


class ConcatMerge(MergeStrategy[str]):
    """Concatenate string results."""

    def merge(self, results: Sequence[str]) -> str:
        return "".join(results)


class ListMerge(MergeStrategy[T]):
    """Return results as a list."""

    def merge(self, results: Sequence[T]) -> List[T]:
        return list(results)


class DictUpdateMerge(MergeStrategy[Dict[str, Any]]):
    """Merge dictionaries where later values overwrite earlier ones."""

    def merge(self, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for item in results:
            merged.update(item)
        return merged


__all__ = [
    "MergeStrategy",
    "ConcatMerge",
    "ListMerge",
    "DictUpdateMerge",
]

