from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover - fallback when langgraph not installed

    class StateGraph:
        """Fallback StateGraph when langgraph is unavailable."""

        def __init__(self) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_node(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def set_entry_point(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_edge(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LangGraph is required to build workflows")

    END = "END"

if TYPE_CHECKING:  # pragma: no cover - for type checkers
    from langgraph.graph import StateGraph as _RealStateGraph
    from langgraph.graph import END as _RealEND

from ..agents.agent_nodes import AnalysisNode, SummaryNode


def build_advanced_legal_workflow(topic: str) -> StateGraph:
    """Construct an advanced workflow graph for the given topic."""

    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    graph.add_node("analysis_refine", AnalysisNode(topic))
    graph.add_node("summary", SummaryNode())

    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "analysis_refine")
    graph.add_edge("analysis_refine", "summary")
    graph.add_edge("summary", END)

    return graph


__all__ = ["build_advanced_legal_workflow"]
