from __future__ import annotations

from ..langgraph_setup import END, StateGraph
from .document_classification_node import DocumentClassificationNode
from .analysis_paths import (
    ContractAnalysisPath,
    LitigationAnalysisPath,
    RegulatoryAnalysisPath,
    EvidenceAnalysisPath,
)


def build_advanced_legal_workflow() -> StateGraph:
    """Build a routed workflow based on document classification."""
    graph = StateGraph()

    classifier = DocumentClassificationNode()
    graph.add_node("classify_document", classifier)

    contract = ContractAnalysisPath()
    litigation = LitigationAnalysisPath()
    regulatory = RegulatoryAnalysisPath()
    evidence = EvidenceAnalysisPath()

    contract.build(graph)
    litigation.build(graph)
    regulatory.build(graph)
    evidence.build(graph)

    graph.set_entry_point("classify_document")
    graph.add_conditional_edges(
        "classify_document",
        {
            "contract": contract.entry_point,
            "litigation": litigation.entry_point,
            "regulatory": regulatory.entry_point,
            "evidence": evidence.entry_point,
        },
    )

    return graph


__all__ = ["build_advanced_legal_workflow"]
