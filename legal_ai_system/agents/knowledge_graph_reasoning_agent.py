from __future__ import annotations

"""Simple reasoning utilities over the knowledge graph."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List

from ..services.knowledge_graph_manager import Entity, EntityType, Relationship, RelationshipType


@dataclass
class ConnectedEntities:
    entity_id: str
    connected: List[Entity]


@dataclass
class CaseEntities:
    case_id: str
    entities: List[Entity]


@dataclass
class PathResult:
    source_id: str
    target_id: str
    path: List[Entity]


class KnowledgeGraphReasoningAgent:
    """Lightweight reasoning agent operating on a knowledge graph manager."""

    def __init__(self, kg_manager: Any) -> None:
        self.kg = kg_manager

    async def get_connected_entities(self, entity_id: str) -> ConnectedEntities:
        nodes = await self.kg.find_connected_entities(entity_id)
        return ConnectedEntities(entity_id=entity_id, connected=nodes)

    async def get_case_entities(self, case_id: str) -> CaseEntities:
        nodes = await self.kg.find_connected_entities(case_id)
        return CaseEntities(case_id=case_id, entities=nodes)

    async def shortest_path(self, source_id: str, target_id: str) -> PathResult:
        """Return a simple breadth-first search path between two entities."""
        queue = deque([[source_id]])
        visited = {source_id}
        while queue:
            path = queue.popleft()
            last = path[-1]
            if last == target_id:
                entities = [await self.kg.get_entity(eid) for eid in path]
                return PathResult(source_id, target_id, entities)
            for ent in await self.kg.find_connected_entities(last):
                if ent.id not in visited:
                    visited.add(ent.id)
                    queue.append(path + [ent.id])
        return PathResult(source_id, target_id, [])


__all__ = [
    "KnowledgeGraphReasoningAgent",
    "ConnectedEntities",
    "CaseEntities",
    "PathResult",
]
