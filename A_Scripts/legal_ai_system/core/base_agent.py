import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class AgentResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

class BaseAgent(ABC):
    """Simplified base agent used for tests."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    async def _process_task(self, data: Any) -> Any:
        pass

    async def process(self, data: Any) -> AgentResult:
        try:
            result = await self._process_task(data)
            return AgentResult(True, result)
        except Exception as e:
            self.logger.exception("Agent error")
            return AgentResult(False, error=str(e))
