import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub heavy dependency pydantic before importing package modules
sys.modules.setdefault("pydantic", ModuleType("pydantic")).BaseModel = object
fastapi_mod = sys.modules.setdefault("fastapi", ModuleType("fastapi"))
fastapi_mod.WebSocket = object
fastapi_mod.WebSocketDisconnect = Exception

from legal_ai_system.workflows.nodes.human_review_node import HumanReviewNode
from legal_ai_system.workflows.nodes.progress_tracking_node import ProgressTrackingNode
from legal_ai_system.utils.reviewable_memory import ReviewPriority


@pytest.mark.asyncio
async def test_human_review_node(mocker):
    review_memory = AsyncMock()
    review_memory.process_extraction_result.return_value = {"findings_added": 1}
    item = MagicMock()
    item.to_dict.return_value = {"id": "1"}
    review_memory.get_pending_reviews_async.side_effect = [[item], []]
    node = HumanReviewNode(review_memory)
    extraction = SimpleNamespace(document_id="doc1")
    result = await node(extraction)
    review_memory.process_extraction_result.assert_called_once_with(extraction, "doc1")
    review_memory.get_pending_reviews_async.assert_any_call(priority=ReviewPriority.CRITICAL)
    review_memory.get_pending_reviews_async.assert_any_call(priority=ReviewPriority.HIGH)
    assert result["high_risk_findings"] == [{"id": "1"}]


@pytest.mark.asyncio
async def test_progress_tracking_node():
    manager = AsyncMock()
    node = ProgressTrackingNode(manager, topic="test")
    data = {"document_id": "doc1", "progress": 0.5}
    result = await node(data)
    manager.broadcast.assert_called_once_with("test", {"type": "progress_update", **data})
    assert result == data
