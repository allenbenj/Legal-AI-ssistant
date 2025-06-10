"""Merge strategies for results from parallel workflow branches."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .types import MergeStrategy


class ConcatMerge(MergeStrategy[List[Any]]):
    """Merge lists by concatenation."""

    async def merge(self, results: Sequence[List[Any]]) -> List[Any]:
        merged: List[Any] = []
        for result in results:
            merged.extend(result)
        return merged


class DictUpdateMerge(MergeStrategy[Dict[str, Any]]):
    """Merge dictionaries using ``dict.update`` semantics."""

    async def merge(
        self, results: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for result in results:
            merged.update(result)
        return merged
