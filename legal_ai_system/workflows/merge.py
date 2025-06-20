from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MergeStrategy(ABC):
    """Base interface for merging parallel node results."""

    @abstractmethod
    def merge(self, results: List[Any]) -> Any:
        """Merge a list of results into a single value."""
        raise NotImplementedError


class FirstResultMerge(MergeStrategy):
    """Return only the first result."""

    def merge(self, results: List[Any]) -> Any:
        return results[0] if results else None


class ListMerge(MergeStrategy):
    """Return all results as a list."""

    def merge(self, results: List[Any]) -> List[Any]:
        return list(results)


class DictMerge(MergeStrategy):
    """Merge dictionaries, later values overwrite earlier ones."""

    def merge(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for item in results:
            merged.update(item)
        return merged


# Default merge strategies mapping.
#
# These provide convenient ready-to-use instances for common merge
# behaviors.  They are exposed at the package level for easy reuse by
# workflow builders and other components.
DEFAULT_MERGE_STRATEGIES: Dict[str, MergeStrategy] = {
    "first": FirstResultMerge(),
    "list": ListMerge(),
    "dict": DictMerge(),
}


__all__ = [
    "MergeStrategy",
    "FirstResultMerge",
    "ListMerge",
    "DictMerge",
    "DEFAULT_MERGE_STRATEGIES",
]
