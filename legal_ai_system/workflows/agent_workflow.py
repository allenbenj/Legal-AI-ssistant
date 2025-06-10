from __future__ import annotations

from typing import Any, AsyncIterator, Iterable, List, Generic

from ..core.agent_types import AgentCapability, T_Input, T_Output


class AgentWorkflow(Generic[T_Input, T_Output]):
    """Composable workflow that chains multiple agents."""

    def __init__(
        self, agents: Iterable[AgentCapability[Any, Any]] | None = None
    ) -> None:
        self.agents: List[AgentCapability[Any, Any]] = list(agents or [])

    def add_agent(self, agent: AgentCapability[Any, Any]) -> None:
        """Append an agent to the workflow."""

        self.agents.append(agent)

    async def process_batch(self, inputs: List[T_Input]) -> List[T_Output]:
        """Process a batch of inputs sequentially through all agents."""

        data: Any = inputs
        for agent in self.agents:
            data = await agent.process_batch(data)  # type: ignore[arg-type]
        return data  # type: ignore[return-value]

    async def process_stream(
        self, input_stream: AsyncIterator[T_Input]
    ) -> AsyncIterator[T_Output]:
        """Process a stream of inputs through the chain of agents."""

        stream: AsyncIterator[Any] = input_stream
        for agent in self.agents:
            stream = agent.process_stream(stream)  # type: ignore[assignment]
        async for item in stream:
            yield item  # type: ignore[misc]

    def supports_capability(self, capability: str) -> bool:
        """Check if all agents in the workflow support a given capability."""

        return all(agent.supports_capability(capability) for agent in self.agents)


__all__ = ["AgentWorkflow"]
