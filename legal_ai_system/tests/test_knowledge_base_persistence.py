import asyncio
import types
import pytest
from types import SimpleNamespace

from legal_ai_system.agents.knowledge_base_agent import KnowledgeBaseAgent
from legal_ai_system.services.knowledge_graph_manager import KnowledgeGraphManager, Entity, EntityType

class DummyContainer:
    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg_manager = kg_manager
    def get_service(self, name: str):
        assert name == "knowledge_graph_manager"
        return self.kg_manager

@pytest.mark.asyncio
async def test_entities_persisted(mocker):
    kg_manager = mocker.AsyncMock(spec=KnowledgeGraphManager)
    kg_manager.create_entity.return_value = Entity(id="e1", type=EntityType.PERSON, name="Alice")
    container = DummyContainer(kg_manager)
    agent = KnowledgeBaseAgent(container)
    task_data = {"entities": [{"name": "Alice", "entity_type": "person"}]}
    await agent._process_task(task_data, {"document_id": "doc1"})
    kg_manager.create_entity.assert_called_once()
