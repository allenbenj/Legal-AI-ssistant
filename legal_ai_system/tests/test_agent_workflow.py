import asyncio
from typing import AsyncIterator, List

import pytest
from pydantic import BaseModel

from legal_ai_system.core.agent_types import AgentCapability
from legal_ai_system.workflows import AgentWorkflow


class TextIn(BaseModel):
    text: str


class TextOut(BaseModel):
    text: str


class UpperAgent(AgentCapability[TextIn, TextOut]):
    async def process_batch(self, inputs: List[TextIn]) -> List[TextOut]:
        return [TextOut(text=i.text.upper()) for i in inputs]

    async def process_stream(
        self, input_stream: AsyncIterator[TextIn]
    ) -> AsyncIterator[TextOut]:
        async for item in input_stream:
            yield TextOut(text=item.text.upper())

    def supports_capability(self, capability: str) -> bool:  # pragma: no cover
        return capability == "text"


class SuffixAgent(AgentCapability[TextOut, TextOut]):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    async def process_batch(self, inputs: List[TextOut]) -> List[TextOut]:
        return [TextOut(text=i.text + self.suffix) for i in inputs]

    async def process_stream(
        self, input_stream: AsyncIterator[TextOut]
    ) -> AsyncIterator[TextOut]:
        async for item in input_stream:
            yield TextOut(text=item.text + self.suffix)

    def supports_capability(self, capability: str) -> bool:  # pragma: no cover
        return capability == "text"


@pytest.mark.asyncio
async def test_workflow_batch() -> None:
    wf = AgentWorkflow([UpperAgent(), SuffixAgent("!")])
    result = await wf.process_batch([TextIn(text="hello")])
    assert result[0].text == "HELLO!"


@pytest.mark.asyncio
async def test_workflow_stream() -> None:
    wf = AgentWorkflow([UpperAgent(), SuffixAgent("!")])

    async def gen() -> AsyncIterator[TextIn]:
        yield TextIn(text="hello")

    out = []
    async for item in wf.process_stream(gen()):
        out.append(item.text)

    assert out == ["HELLO!"]
