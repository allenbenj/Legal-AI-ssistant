from dataclasses import dataclass
from typing import List, Optional

from ..services.knowledge_graph_manager import (
    KnowledgeGraphManager,
    Entity,
    RelationshipType,
)


@dataclass
class ConnectedEntities:
    """Result for connected entities query."""

    entity: Entity
    connected: List[Entity]


@dataclass
class CaseEntities:
    """Entities involved in a case."""

    case: Entity
    entities: List[Entity]


@dataclass
class PathResult:
    """Shortest path between two entities."""

    source: Entity
    target: Entity
    path: List[Entity]


class KnowledgeGraphReasoningAgent:
    """Lightweight reasoning agent over the knowledge graph."""

    def __init__(self, kg_manager: KnowledgeGraphManager) -> None:
        self.kg_manager = kg_manager

    async def get_connected_entities(self, entity_id: str) -> ConnectedEntities:
        entity = await self.kg_manager.get_entity(entity_id)
        connected = await self.kg_manager.find_connected_entities(entity_id)
        return ConnectedEntities(entity=entity, connected=connected)

    async def get_case_entities(self, case_id: str) -> CaseEntities:
        case = await self.kg_manager.get_entity(case_id)
        involved: List[Entity] = []
        for rel in self.kg_manager.relationships.values():
            if rel.source_entity_id == case_id and rel.type == RelationshipType.INVOLVES:
                ent = await self.kg_manager.get_entity(rel.target_entity_id)
                if ent:
                    involved.append(ent)
            elif rel.target_entity_id == case_id and rel.type == RelationshipType.INVOLVES:
                ent = await self.kg_manager.get_entity(rel.source_entity_id)
                if ent:
                    involved.append(ent)
        return CaseEntities(case=case, entities=involved)

    async def shortest_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> Optional[PathResult]:
        if source_id == target_id:
            ent = await self.kg_manager.get_entity(source_id)
            return PathResult(source=ent, target=ent, path=[ent])

        from collections import deque

        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if current == target_id:
                entities = [await self.kg_manager.get_entity(eid) for eid in path]
                return PathResult(source=entities[0], target=entities[-1], path=entities)
            for rel in self.kg_manager.relationships.values():
                next_id = None
                if rel.source_entity_id == current:
                    next_id = rel.target_entity_id
                elif rel.target_entity_id == current:
                    next_id = rel.source_entity_id
                if next_id and next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        return None
