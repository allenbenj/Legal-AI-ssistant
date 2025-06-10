"""Setup a basic LangGraph document processing workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List

try:  # pragma: no cover - optional dependency
    from langgraph.graph import StateGraph, END
except Exception:  # ImportError or other issues if langgraph not installed

    class StateGraph:  # pragma: no cover - simple placeholder
        """Fallback ``StateGraph`` when ``langgraph`` is unavailable."""

        def __init__(self) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        # provide stub methods so static analysis tools see expected attributes
        def add_node(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def set_entry_point(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_edge(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_parallel_nodes(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_conditional_edges(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LangGraph is required to build workflows")

    END = "END"

if TYPE_CHECKING:  # pragma: no cover - hint for type checkers
    from langgraph.graph import StateGraph as _RealStateGraph
    from langgraph.graph import END as _RealEND

from ..agents.agent_nodes import AnalysisNode, SummaryNode


def build_graph(topic: str) -> StateGraph:
    """Build a LangGraph pipeline demonstrating advanced features."""

    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    graph.add_node("summary", SummaryNode())
    graph.add_node("echo", lambda x: x)
    graph.add_node("combine", lambda items: " ".join(items))
    graph.add_node("final", lambda x: x)

    graph.set_entry_point("analysis")

    # Execute analysis result through summary and echo in parallel then combine
    graph.add_parallel_nodes("analysis", ["summary", "echo"], "combine")

    # Route based on length of combined output
    graph.add_conditional_edges(
        "combine",
        [
            (lambda v: len(v) > 200, "summary"),
            (lambda v: len(v) <= 200, "final"),
        ],
    )

    graph.add_edge("summary", "final")
    graph.add_edge("final", END)

    return graph


__all__ = ["build_graph"]
