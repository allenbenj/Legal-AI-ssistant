from __future__ import annotations

"""Shared helpers for Memory Brain functionality."""

import asyncio
from typing import Any, Dict, List

from ..tools.contradiction_detector import ContradictionDetector, MemoryEntry
from ..services.memory_service import memory_manager_context

__all__ = [
    "load_memory_entries",
    "persist_statement",
    "run_contradiction_check",
    "MemoryBrainCore",
]


def load_memory_entries() -> List[MemoryEntry]:
    """Return stored statement entries from :class:`UnifiedMemoryManager`."""

    async def _load() -> List[MemoryEntry]:
        async with memory_manager_context() as manager:
            entries = await manager.get_context_window("memory_brain")
            return [
                MemoryEntry(
                    speaker=e.get("metadata", {}).get("speaker", ""),
                    statement=e.get("content", ""),
                    source=e.get("metadata", {}).get("source", ""),
                )
                for e in entries
                if e.get("entry_type") == "statement"
            ]

    try:
        return asyncio.run(_load())
    except Exception:  # pragma: no cover - initialization can fail offline
        return []


def persist_statement(entry: MemoryEntry) -> None:
    """Persist a statement to the :class:`UnifiedMemoryManager`."""

    async def _store() -> None:
        async with memory_manager_context() as manager:
            await manager.add_context_window_entry(
                session_id="memory_brain",
                entry_type="statement",
                content=entry.statement,
                metadata={"speaker": entry.speaker, "source": entry.source},
            )

    try:
        asyncio.run(_store())
    except Exception:  # pragma: no cover - storage errors are non fatal
        pass


def run_contradiction_check(
    memory_entries: List[MemoryEntry],
    speaker: str,
    statement: str,
    source: str = "",
) -> Dict[str, Any]:
    """Check a statement for contradictions against stored entries."""

    detector = ContradictionDetector(memory_entries)
    return detector.check(speaker, statement, source)


class MemoryBrainCore:
    """Convenience wrapper maintaining memory state."""

    def __init__(self) -> None:
        self.memory_entries: List[MemoryEntry] = []

    def load_entries(self) -> None:
        self.memory_entries = load_memory_entries()

    def add_statement(self, entry: MemoryEntry) -> None:
        self.memory_entries.append(entry)
        persist_statement(entry)

    def check(self, speaker: str, statement: str, source: str = "") -> Dict[str, Any]:
        return run_contradiction_check(self.memory_entries, speaker, statement, source)
