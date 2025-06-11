from __future__ import annotations

"""Simplified knowledge graph reasoning components."""

from dataclasses import dataclass
from typing import Any, List


Entity = Any


@dataclass
class ConnectedEntities:
    """Represents an entity and its connected entities."""

    entity_id: str
    connected: List[Entity]


@dataclass
class CaseEntities:
    """Entities associated with a particular case."""

    case_id: str
    entities: List[Entity]


@dataclass
class PathResult:
    """A path between two entities within the graph."""

    path: List[Entity]


class KnowledgeGraphReasoningAgent:
    """Placeholder agent for reasoning over a knowledge graph."""

    async def find_path(self, start: Entity, end: Entity) -> PathResult:
        """Return a minimal two step path."""
        return PathResult(path=[start, end])

