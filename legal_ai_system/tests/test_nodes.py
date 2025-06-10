import types
from types import SimpleNamespace

import pytest

from legal_ai_system.workflows.nodes import HumanReviewNode, ProgressTrackingNode


@pytest.mark.asyncio
async def test_human_review_node_queues_high_risk(mocker):
    memory = mocker.AsyncMock()
    node = HumanReviewNode(memory)
    extraction = SimpleNamespace(
        entities=[SimpleNamespace(entity_type="VIOLATION", source_text_snippet="brady violation")],
        relationships=[],
        extraction_metadata={},
    )
    data = {"ontology_result": extraction, "document_id": "doc1"}

    result = await node(data)

    memory.process_extraction_result.assert_awaited_once()
    args = memory.process_extraction_result.await_args.args
    assert args[0] is extraction
    assert args[1] == "doc1"
    assert result["high_risk_findings"] == 1


@pytest.mark.asyncio
async def test_progress_tracking_node_broadcasts(mocker):
    manager = mocker.AsyncMock()
    node = ProgressTrackingNode(manager, topic="test")

    update = {"message": "processing", "progress": 0.5}
    result = await node(update)

    manager.broadcast.assert_awaited_once_with("test", {"type": "progress", **update})
    assert result == update
