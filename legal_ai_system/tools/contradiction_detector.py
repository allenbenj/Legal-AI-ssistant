from __future__ import annotations

"""Simple contradiction detection tool."""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MemoryEntry:
    """Simple container for a statement made by a speaker."""

    speaker: str
    statement: str
    source: str = ""


class ContradictionDetector:
    """Placeholder contradiction detection using naive heuristics."""

    def __init__(self, memory_entries: List[MemoryEntry] | None = None) -> None:
        self.memory_entries = memory_entries or []

    def check(self, speaker: str, statement: str, source: str = "") -> Dict[str, Any]:
        """Return a dummy contradiction result."""
        contradictions = []
        lower_stmt = statement.lower()
        for entry in self.memory_entries:
            content = entry.statement.lower()
            if speaker == entry.speaker and content and content != lower_stmt:
                if (
                    "not" in lower_stmt and lower_stmt.replace("not ", "") in content
                ) or ("not" in content and content.replace("not ", "") in lower_stmt):
                    contradictions.append(
                        {
                            "original": {
                                "speaker": entry.speaker,
                                "statement": entry.statement,
                                "source": entry.source,
                            },
                            "conflict": statement,
                        }
                    )
        return {
            "contradictions": contradictions,
            "count": len(contradictions),
            "source": source,
        }
