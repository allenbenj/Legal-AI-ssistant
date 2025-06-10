from __future__ import annotations

from ...utils.workflow_builder import WorkflowBuilder

from .document_classification_node import DocumentClassificationNode
from .analysis_paths import (
    ContractAnalysisPath,
    LitigationAnalysisPath,
    RegulatoryAnalysisPath,
    EvidenceAnalysisPath,
)


def add_conditional_edges(
    builder: WorkflowBuilder, start: str, mapping: dict[str, callable]
) -> None:
    """Helper to add multiple conditional edges from ``start``."""
    for target, condition in mapping.items():
        builder.add_edge(start, target, condition=condition)


def build_advanced_legal_workflow() -> WorkflowBuilder:
    """Create a workflow that routes documents based on classification."""
    builder = WorkflowBuilder()

    classifier_node = DocumentClassificationNode()

    async def classify_document(text: str):
        return classifier_node(text)

    builder.register_node("classify_document", classify_document)
    builder.set_entry_point("classify_document")

    contract_path = ContractAnalysisPath()
    litigation_path = LitigationAnalysisPath()
    regulatory_path = RegulatoryAnalysisPath()
    evidence_path = EvidenceAnalysisPath()

    contract_path.build(builder)
    litigation_path.build(builder)
    regulatory_path.build(builder)
    evidence_path.build(builder)

    add_conditional_edges(
        builder,
        "classify_document",
        {
            ContractAnalysisPath.ENTRY_NODE: lambda r: isinstance(r, dict) and r.get("primary_type") == "contract",
            LitigationAnalysisPath.ENTRY_NODE: lambda r: isinstance(r, dict) and r.get("primary_type") == "court_filing",
            RegulatoryAnalysisPath.ENTRY_NODE: lambda r: isinstance(r, dict) and r.get("primary_type") == "statute",
            EvidenceAnalysisPath.ENTRY_NODE: lambda r: isinstance(r, dict) and r.get("primary_type") not in {"contract", "court_filing", "statute"},
        },
    )

    return builder


__all__ = ["build_advanced_legal_workflow", "add_conditional_edges"]
