"""Simplified access to the generic :class:`LegalWorkflowBuilder`."""

from __future__ import annotations


    pass

class LegalWorkflowBuilder:
    """Minimal async workflow builder with optional parallel steps."""

    def __init__(self) -> None:
        self._steps: List[Tuple[str, Any]] = []

    def add_step(self, func: Callable[[Any], Awaitable[Any]]) -> None:
        """Add a sequential processing step."""
        self._steps.append(("step", func))

    def add_parallel_processing(
        self,
        funcs: List[Callable[[Any], Awaitable[Any]]],
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Add a set of functions to run in parallel and merge their results."""
        merge = merge_strategy or DEFAULT_MERGE_STRATEGIES["list"]
        self._steps.append(("parallel", funcs, merge))

    async def run(self, data: Any) -> Any:
        """Execute the configured workflow."""
        result: Any = data
        for item in self._steps:
            if item[0] == "step":
                _, func = item
                result = await func(result)
            else:
                _, funcs, merge = item
                results = await asyncio.gather(*(f(result) for f in funcs))
                result = merge.merge(list(results))
        return result


__all__ = ["LegalWorkflowBuilder"]
