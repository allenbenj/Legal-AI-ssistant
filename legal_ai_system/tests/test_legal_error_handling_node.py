import pytest

from legal_ai_system.workflows.nodes.legal_error_handling_node import (
    LegalErrorHandlingNode,
)
from legal_ai_system.workflows import LegalWorkflowBuilder


@pytest.mark.asyncio
async def test_primary_success() -> None:
    async def primary(x):
        return {"result": x, "confidence": 0.9}

    async def ensemble(x):
        return {"result": x, "confidence": 0.1}

    node = LegalErrorHandlingNode(primary, ensemble)
    out = await node("x")
    assert out["result"] == "x"


@pytest.mark.asyncio
async def test_fallback_success() -> None:
    async def primary(x):
        return {"result": x, "confidence": 0.4}

    async def ensemble(x):
        return {"result": x + "e", "confidence": 0.85}

    node = LegalErrorHandlingNode(primary, ensemble)
    out = await node("a")
    assert out["result"] == "ae"


@pytest.mark.asyncio
async def test_escalation_on_failure() -> None:
    calls = {}

    async def primary(x):
        raise RuntimeError("fail")

    async def ensemble(x):
        return {"result": x, "confidence": 0.3}

    async def escalate(info):
        calls["called"] = True
        return "escalated"

    node = LegalErrorHandlingNode(primary, ensemble, escalate=escalate)
    out = await node("z")
    assert out == "escalated"
    assert calls.get("called")


@pytest.mark.asyncio
async def test_builder_integration() -> None:
    async def step(x):
        return {"result": x, "confidence": 1.0}

    async def noop(x):
        return x

    handler = LegalErrorHandlingNode(step, noop)
    builder = LegalWorkflowBuilder()
    builder.add_step(step)
    builder.set_error_handler(handler)
    result = await builder.run("ok")
    assert result["confidence"] == 1.0
