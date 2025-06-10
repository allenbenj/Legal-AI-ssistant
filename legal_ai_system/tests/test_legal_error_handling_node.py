import pytest

from legal_ai_system.workflows.nodes.legal_error_handling_node import LegalErrorHandlingNode
from legal_ai_system.workflows import LegalWorkflowBuilder


@pytest.mark.asyncio
async def test_node_fallback_and_escalation():
    async def primary(data):
        return {"confidence": 0.5, "value": "primary"}

    async def ensemble(data):
        return {"confidence": 0.9, "value": "ensemble"}

    escalations = {}

    async def escalate(data, exc):
        escalations["called"] = True

    node = LegalErrorHandlingNode(primary, ensemble, escalate, threshold=0.8)

    result = await node({})
    assert result["value"] == "ensemble"
    # Ensemble confidence is above threshold so escalation should not occur
    assert "called" not in escalations


@pytest.mark.asyncio
async def test_workflow_integration_error_path():
    async def failing(data):
        raise RuntimeError("boom")

    async def ensemble(data):
        return {"confidence": 0.4}

    escalations = {}

    async def escalate(data, exc):
        escalations["error"] = str(exc)

    builder = LegalWorkflowBuilder()
    builder.add_step(lambda x: x)
    handler = LegalErrorHandlingNode(failing, ensemble, escalate, threshold=0.8)
    builder.set_error_handler(handler)

    result = await builder.run({})
    assert result["escalated"] is True
    assert "error" in escalations
