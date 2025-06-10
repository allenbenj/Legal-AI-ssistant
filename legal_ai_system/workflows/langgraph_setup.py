"""Setup a basic LangGraph document processing workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List

try:  # pragma: no cover - optional dependency
    from langgraph.graph import StateGraph, END, BaseNode
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
    class BaseNode:
        pass

if TYPE_CHECKING:  # pragma: no cover - hint for type checkers
    from langgraph.graph import StateGraph as _RealStateGraph
    from langgraph.graph import END as _RealEND

from ..agents.agent_nodes import AnalysisNode, SummaryNode
from .nodes import HumanReviewNode, ProgressTrackingNode


try:  # pragma: no cover - optional dependency at runtime
    from ..utils.reviewable_memory import ReviewableMemory
except Exception:  # pragma: no cover - during tests
    ReviewableMemory = Any  # type: ignore




    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))

    graph.add_node("summary", SummaryNode())
    graph.add_node("echo", lambda x: x)
    graph.add_node("combine", lambda items: " ".join(items))
    graph.add_node("final", lambda x: x)



    return graph


def build_advanced_legal_workflow(topic: str) -> StateGraph:
    """Build an expanded LangGraph workflow including citation validation."""
    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    graph.add_node("citations", CitationCheckNode())
    graph.add_node("summary", SummaryNode())

    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "citations")
    graph.add_edge("citations", "summary")
    graph.add_edge("summary", END)

    return graph


__all__ = ["build_graph", "build_advanced_legal_workflow", "CitationCheckNode"]
