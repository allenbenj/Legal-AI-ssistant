import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock, MagicMock
from enum import Enum

import pytest

# Stub pydantic only if missing to avoid interfering with other tests
import importlib.util
if 'pydantic' not in sys.modules and importlib.util.find_spec('pydantic') is None:
    pydantic_stub = ModuleType('pydantic')
    pydantic_stub.BaseModel = object
    sys.modules['pydantic'] = pydantic_stub

if 'numpy' not in sys.modules:
    np_stub = ModuleType('numpy')
    np_stub.array = lambda *a, **k: a
    class _FakeNdarray(list):
        ndim = 1
    np_stub.ndarray = _FakeNdarray
    sys.modules['numpy'] = np_stub

if 'fastapi' not in sys.modules:
    fastapi_stub = ModuleType('fastapi')
    fastapi_stub.WebSocket = object
    fastapi_stub.WebSocketDisconnect = Exception
    sys.modules['fastapi'] = fastapi_stub

for name in [
    'sklearn',
    'sklearn.metrics',
    'sklearn.metrics.pairwise',
    'sklearn.tree',
]:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)
sys.modules['sklearn.metrics.pairwise'].cosine_similarity = lambda *a, **k: []
sys.modules['sklearn.tree'].DecisionTreeClassifier = object

if 'legal_ai_system.utils.reviewable_memory' not in sys.modules:
    rm_stub = ModuleType('legal_ai_system.utils.reviewable_memory')
    ReviewPriority = Enum('ReviewPriority', 'LOW MEDIUM HIGH CRITICAL')
    class ReviewableMemory: ...
    rm_stub.ReviewPriority = ReviewPriority
    rm_stub.ReviewableMemory = ReviewableMemory
    sys.modules['legal_ai_system.utils.reviewable_memory'] = rm_stub

from legal_ai_system.workflows.nodes.human_review_node import HumanReviewNode
from legal_ai_system.workflows.nodes.progress_tracking_node import ProgressTrackingNode
from legal_ai_system.utils.reviewable_memory import ReviewPriority


@pytest.mark.asyncio
async def test_human_review_node_returns_list_and_count(mocker):
    mem = AsyncMock()
    mem.process_extraction_result.return_value = {}
    item1 = MagicMock(); item1.to_dict.return_value = {"id": "a"}
    item2 = MagicMock(); item2.to_dict.return_value = {"id": "b"}
    mem.get_pending_reviews_async.side_effect = [[item1], [item2]]
    node = HumanReviewNode(mem)

    extraction = SimpleNamespace(document_id="doc")
    result_obj = await node(extraction)
    assert result_obj["high_risk_findings"] == [{"id": "a"}, {"id": "b"}]
    mem.process_extraction_result.assert_called_with(extraction, "doc")
    mem.get_pending_reviews_async.assert_any_call(priority=ReviewPriority.CRITICAL)
    mem.get_pending_reviews_async.assert_any_call(priority=ReviewPriority.HIGH)

    mem.get_pending_reviews_async.side_effect = [[item1], [item2]]
    dict_input = {"ontology_result": extraction, "document_id": "doc"}
    result_dict = await node(dict_input)
    assert result_dict["high_risk_findings"] == 2


@pytest.mark.asyncio
async def test_progress_tracking_node_message_types(mocker):
    manager = AsyncMock()
    node = ProgressTrackingNode(manager, topic="topic")

    update = {"document_id": "1", "progress": 0.1}
    await node(update)
    manager.broadcast.assert_awaited_with("topic", {"type": "progress_update", **update})
    manager.broadcast.reset_mock()

    update2 = {"message": "work", "progress": 0.2}
    await node(update2)
    manager.broadcast.assert_awaited_with("topic", {"type": "progress", **update2})
