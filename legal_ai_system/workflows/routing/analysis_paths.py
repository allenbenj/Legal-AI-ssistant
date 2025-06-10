from __future__ import annotations

from typing import Any

from ..langgraph_setup import END


class ContractAnalysisPath:
    """Subgraph for contract document analysis."""

    entry_point = "contract_analysis"

    def build(self, graph: Any) -> None:
        def analyze(document: str) -> str:
            return "contract_result"

        graph.add_node(self.entry_point, analyze)
        graph.add_edge(self.entry_point, END)


class LitigationAnalysisPath:
    """Subgraph for litigation document analysis."""

    entry_point = "litigation_analysis"

    def build(self, graph: Any) -> None:
        def analyze(document: str) -> str:
            return "litigation_result"

        graph.add_node(self.entry_point, analyze)
        graph.add_edge(self.entry_point, END)


class RegulatoryAnalysisPath:
    """Subgraph for regulatory document analysis."""

    entry_point = "regulatory_analysis"

    def build(self, graph: Any) -> None:
        def analyze(document: str) -> str:
            return "regulatory_result"

        graph.add_node(self.entry_point, analyze)
        graph.add_edge(self.entry_point, END)


class EvidenceAnalysisPath:
    """Subgraph for evidence document analysis."""

    entry_point = "evidence_analysis"

    def build(self, graph: Any) -> None:
        def analyze(document: str) -> str:
            return "evidence_result"

        graph.add_node(self.entry_point, analyze)
        graph.add_edge(self.entry_point, END)


__all__ = [
    "ContractAnalysisPath",
    "LitigationAnalysisPath",
    "RegulatoryAnalysisPath",
    "EvidenceAnalysisPath",
]
