from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..services.knowledge_graph_manager import Entity, RelationshipType


@dataclass
class ConnectedEntities:
    """Simple container for connected entities."""

    entity_id: str
    connected: List[Entity]


@dataclass
class CaseEntities:
    """Entities linked to a case."""

    case_id: str
    entities: List[Entity]


@dataclass
class PathResult:
    """Result of a shortest path query."""

    start_id: str
    end_id: str
    path: List[Entity]


class KnowledgeGraphReasoningAgent:
    """Minimal reasoning utilities over a knowledge graph manager."""

    def __init__(self, kg_manager) -> None:
        self.kg = kg_manager

    async def get_connected_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_depth: int = 2,
    ) -> ConnectedEntities:
        connected = await self.kg.find_connected_entities(
            entity_id, relationship_types=relationship_types, max_depth=max_depth
        )
        return ConnectedEntities(entity_id=entity_id, connected=connected)

    async def get_case_entities(self, case_id: str) -> CaseEntities:
        connected = await self.kg.find_connected_entities(
            case_id,
            relationship_types=[RelationshipType.INVOLVES],
            max_depth=1,
        )
        return CaseEntities(case_id=case_id, entities=connected)

    async def shortest_path(
        self,
        start_id: str,
        end_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_depth: int = 3,
    ) -> PathResult:
        # Simple breadth-first search using knowledge graph manager's data
        visited = {start_id}
        queue = [(start_id, [start_id])]
        while queue:
            current, path = queue.pop(0)
            if current == end_id:
                entities = [await self.kg.get_entity(eid) for eid in path]
                return PathResult(start_id=start_id, end_id=end_id, path=entities)
            neighbors = await self.kg.find_connected_entities(
                current, relationship_types=relationship_types, max_depth=1
            )
            for entity in neighbors:
                if entity.id not in visited and len(path) < max_depth + 1:
                    visited.add(entity.id)
                    queue.append((entity.id, path + [entity.id]))
        return PathResult(start_id=start_id, end_id=end_id, path=[])
