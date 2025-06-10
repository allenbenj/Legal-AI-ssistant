"""Merge strategies for combining parallel results."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, TypeVar

from .types import MergeStrategy

T = TypeVar("T")


class ConcatMerge(MergeStrategy[List[T]]):
    """Concatenate lists from parallel executions."""

    async def merge(self, results: Sequence[List[T]]) -> List[T]:
        merged: List[T] = []
        for item in results:
            merged.extend(item)
        return merged


class DictUpdateMerge(MergeStrategy[Dict[str, Any]]):
    """Merge dictionaries by updating keys."""

    async def merge(self, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for d in results:
            merged.update(d)
        return merged
