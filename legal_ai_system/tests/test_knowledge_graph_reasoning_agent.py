import asyncio
from typing import Dict

import pytest

try:
    from legal_ai_system.agents.knowledge_graph_reasoning_agent import (
        KnowledgeGraphReasoningAgent,
        ConnectedEntities,
        CaseEntities,
        PathResult,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    pytest.skip("Reasoning agent not available", allow_module_level=True)
from legal_ai_system.services.knowledge_graph_manager import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
)


class DummyKG:
    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

    async def get_entity(self, entity_id: str):
        return self.entities.get(entity_id)

    async def find_connected_entities(self, entity_id: str, relationship_types=None, max_depth=2):
        connected = set()
        for rel in self.relationships.values():
            if rel.source_entity_id == entity_id:
                connected.add(rel.target_entity_id)
            elif rel.target_entity_id == entity_id:
                connected.add(rel.source_entity_id)
        return [self.entities[eid] for eid in connected]


def build_dummy_graph() -> DummyKG:
    kg = DummyKG()
    e1 = Entity(id="p1", type=EntityType.PERSON, name="Alice")
    e2 = Entity(id="c1", type=EntityType.CASE, name="CaseA")
    e3 = Entity(id="o1", type=EntityType.ORGANIZATION, name="ACME")
    kg.entities = {e.id: e for e in [e1, e2, e3]}
    r1 = Relationship(
        id="r1",
        source_entity_id="p1",
        target_entity_id="c1",
        type=RelationshipType.INVOLVES,
    )
    r2 = Relationship(
        id="r2",
        source_entity_id="c1",
        target_entity_id="o1",
        type=RelationshipType.INVOLVES,
    )
    kg.relationships = {r1.id: r1, r2.id: r2}
    return kg


@pytest.mark.asyncio
async def test_get_connected_entities():
    kg = build_dummy_graph()
    agent = KnowledgeGraphReasoningAgent(kg)
    result = await agent.get_connected_entities("c1")
    assert isinstance(result, ConnectedEntities)
    names = {e.name for e in result.connected}
    assert names == {"Alice", "ACME"}


@pytest.mark.asyncio
async def test_get_case_entities():
    kg = build_dummy_graph()
    agent = KnowledgeGraphReasoningAgent(kg)
    result = await agent.get_case_entities("c1")
    assert isinstance(result, CaseEntities)
    names = {e.name for e in result.entities}
    assert names == {"Alice", "ACME"}


@pytest.mark.asyncio
async def test_shortest_path():
    kg = build_dummy_graph()
    agent = KnowledgeGraphReasoningAgent(kg)
    result = await agent.shortest_path("p1", "o1")
    assert isinstance(result, PathResult)
    path_names = [e.name for e in result.path]
    assert path_names == ["Alice", "CaseA", "ACME"]
