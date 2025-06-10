from __future__ import annotations

from enum import Enum
from typing import AsyncIterator, Generic, List, Protocol, TypeVar

from pydantic import BaseModel

T_Input = TypeVar("T_Input", bound=BaseModel)
T_Output = TypeVar("T_Output", bound=BaseModel)
T_Config = TypeVar("T_Config", bound=BaseModel)


class ProcessingStrategy(str, Enum):
    """Strategies supported by processing agents."""

    HYBRID = "hybrid"  # Pattern + LLM
    LLM_ONLY = "llm_only"
    PATTERN_ONLY = "pattern_only"
    ENSEMBLE = "ensemble"


class AgentCapability(Protocol[T_Input, T_Output]):
    """Protocol defining the core capabilities of an agent."""

    async def process_batch(self, inputs: List[T_Input]) -> List[T_Output]:
        """Process a batch of inputs and return a list of outputs."""

    async def process_stream(
        self, input_stream: AsyncIterator[T_Input]
    ) -> AsyncIterator[T_Output]:
        """Process a stream of inputs yielding outputs as they become available."""

    def supports_capability(self, capability: str) -> bool:
        """Return True if the agent supports the named capability."""


__all__ = [
    "ProcessingStrategy",
    "AgentCapability",
    "T_Input",
    "T_Output",
    "T_Config",
]
