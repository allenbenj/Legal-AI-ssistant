from __future__ import annotations

"""Reasoning utilities for querying the knowledge graph."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ..services.knowledge_graph_manager import (
    KnowledgeGraphManager,
    Entity,
    RelationshipType,
)


@dataclass
class ConnectedEntities:
    """Result wrapper for entities connected to a given entity."""

    entity_id: str
    connected: List[Entity]


@dataclass
class CaseEntities:
    """Entities related to a specific legal case."""

    case_id: str
    entities: List[Entity]


@dataclass
class PathResult:
    """Result from a shortest path query."""

    start_id: str
    end_id: str
    path: List[Entity]


class KnowledgeGraphReasoningAgent:
    """Lightweight agent exposing simple graph reasoning helpers."""

    def __init__(self, graph: KnowledgeGraphManager) -> None:
        self.graph = graph

    async def get_connected_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_depth: int = 2,
    ) -> ConnectedEntities:
        """Return entities connected to ``entity_id`` via the given relationships."""
        connected = await self.graph.find_connected_entities(
            entity_id, relationship_types, max_depth
        )
        return ConnectedEntities(entity_id=entity_id, connected=connected)

    async def get_case_entities(self, case_id: str) -> CaseEntities:
        """Return entities connected to a case."""
        result = await self.get_connected_entities(case_id)
        return CaseEntities(case_id=case_id, entities=result.connected)

    async def shortest_path(
        self,
        start_id: str,
        end_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_depth: int = 5,
    ) -> PathResult:
        """Compute a simple breadth-first shortest path between two entities."""
        # Basic BFS using relationships stored in the manager
        queue: List[List[str]] = [[start_id]]
        visited = {start_id}
        relationships = getattr(self.graph, "relationships", {})

        while queue:
            path = queue.pop(0)
            current = path[-1]
            if current == end_id:
                entities = [await self.graph.get_entity(eid) for eid in path]
                return PathResult(start_id=start_id, end_id=end_id, path=entities)

            if len(path) > max_depth:
                continue

            for rel in relationships.values():
                if relationship_types and rel.type not in relationship_types:
                    continue
                neighbor = None
                if rel.source_entity_id == current:
                    neighbor = rel.target_entity_id
                elif rel.target_entity_id == current:
                    neighbor = rel.source_entity_id
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return PathResult(start_id=start_id, end_id=end_id, path=[])


__all__ = [
    "KnowledgeGraphReasoningAgent",
    "ConnectedEntities",
    "CaseEntities",
    "PathResult",
]
