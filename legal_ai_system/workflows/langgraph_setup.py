"""Setup a basic LangGraph document processing workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _uppercase(text: str) -> str:
    return text.upper()


def _merge_text(results: list[str]) -> str:
    return "\n".join(results)


def build_graph(topic: str) -> StateGraph:
    """Build a simple LangGraph pipeline for a given topic."""
    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    graph.add_node("summary", SummaryNode())
    graph.add_node("noop", lambda x: x)
    graph.add_node("uppercase", _uppercase)
    graph.add_node("merge", _merge_text)

    graph.set_entry_point("analysis")
    graph.add_conditional_edges(
        "analysis",
        [
            (lambda t: "skip" in t.lower(), "noop"),
            (lambda t: "skip" not in t.lower(), "summary"),
        ],
    )
    graph.add_parallel_nodes("summary", ["uppercase", "noop"], "merge")
    graph.add_edge("noop", END)
    graph.add_edge("merge", END)

    return graph


__all__ = ["build_graph"]
