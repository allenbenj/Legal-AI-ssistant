from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List


class LegalWorkflowBuilder:
    """Simple asynchronous workflow runner."""

    def __init__(self) -> None:
        self._steps: List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = []

    def add_step(self, step: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        """Append an asynchronous step to the workflow."""

        self._steps.append(step)

    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow sequentially."""

        state = initial_state
        for step in self._steps:
            state = await step(state)
        return state


__all__ = ["LegalWorkflowBuilder"]
