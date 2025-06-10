from __future__ import annotations

"""Minimal reasoning utilities over the in-memory knowledge graph."""

from dataclasses import dataclass
from typing import List, Any

from ..services.knowledge_graph_manager import Entity


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
    path: List[Entity]


class KnowledgeGraphReasoningAgent:
    """Simple reasoning helper operating on a provided graph manager."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    async def get_connected_entities(self, entity_id: str) -> ConnectedEntities:
        connected = await self.graph.find_connected_entities(entity_id)
        return ConnectedEntities(entity_id=entity_id, connected=connected)

    async def get_case_entities(self, case_id: str) -> CaseEntities:
        entities = await self.graph.find_connected_entities(case_id)
        return CaseEntities(case_id=case_id, entities=entities)

    async def shortest_path(self, start_id: str, end_id: str) -> PathResult:
        from collections import deque

        queue = deque([[start_id]])
        visited = {start_id}

        while queue:
            path = queue.popleft()
            last = path[-1]
            if last == end_id:
                entities = [await self.graph.get_entity(pid) for pid in path]
                return PathResult(path=entities)
            neighbors = [
                rel.target_entity_id if rel.source_entity_id == last else rel.source_entity_id
                for rel in getattr(self.graph, "relationships", {}).values()
                if rel.source_entity_id == last or rel.target_entity_id == last
            ]
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    queue.append(path + [n])
        return PathResult(path=[])


__all__ = [
    "KnowledgeGraphReasoningAgent",
    "ConnectedEntities",
    "CaseEntities",
    "PathResult",
]
