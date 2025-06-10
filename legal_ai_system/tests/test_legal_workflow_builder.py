import pytest

from legal_ai_system.workflows import LegalWorkflowBuilder, ListMerge


@pytest.mark.asyncio
async def test_parallel_processing_merge_list() -> None:
    async def node_a(x: str) -> str:
        return x + "a"

    async def node_b(x: str) -> str:
        return x + "b"

    builder = LegalWorkflowBuilder()

    async def start(x: str) -> str:
        return x

    builder.add_step(start)
    builder.add_parallel_processing([node_a, node_b], merge_strategy=ListMerge())

    result = await builder.run("x")
    assert result == ["xa", "xb"]
