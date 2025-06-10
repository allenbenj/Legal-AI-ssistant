import asyncio
from typing import Any

import pytest

from legal_ai_system.utils.workflow_builder import WorkflowBuilder


@pytest.mark.asyncio
async def test_node_registration_and_execution() -> None:
    builder = WorkflowBuilder()

    async def start(x: int) -> int:
        return x + 1

    builder.register_node("start", start)
    builder.set_entry_point("start")

    result = await builder.run(1)
    assert result == 2


@pytest.mark.asyncio
async def test_conditional_branching() -> None:
    builder = WorkflowBuilder()

    async def start(x: int) -> int:
        return x

    async def pos(x: int) -> int:
        return x + 1

    async def neg(x: int) -> int:
        return x - 1

    builder.register_node("start", start)
    builder.register_node("pos", pos)
    builder.register_node("neg", neg)
    builder.set_entry_point("start")
    builder.add_edge("start", "pos", condition=lambda v: v > 0)
    builder.add_edge("start", "neg", condition=lambda v: v <= 0)

    result = await builder.run(5)
    assert result == 6


@pytest.mark.asyncio
async def test_parallel_execution_and_merge() -> None:
    builder = WorkflowBuilder()
    calls: dict[str, int] = {"a": 0, "b": 0}

    async def start(x: int) -> int:
        return x

    async def node_a(x: int) -> int:
        calls["a"] += 1
        await asyncio.sleep(0.01)
        return x + 1

    async def node_b(x: int) -> int:
        calls["b"] += 1
        await asyncio.sleep(0.01)
        return x + 2

    async def merge(results: list[int]) -> int:
        return sum(results)

    builder.register_node("start", start)
    builder.register_node("a", node_a)
    builder.register_node("b", node_b)
    builder.register_node("merge", merge)

    builder.set_entry_point("start")
    builder.add_parallel("start", ["a", "b"], "merge")

    result = await builder.run(0)
    assert result == 3
    assert calls == {"a": 1, "b": 1}
