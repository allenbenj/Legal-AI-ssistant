from __future__ import annotations

from ...utils.workflow_builder import WorkflowBuilder


class ContractAnalysisPath:
    """Subgraph for contract-specific analysis."""

    ENTRY_NODE = "contract_analysis"

    def build(self, builder: WorkflowBuilder) -> None:
        async def contract_analysis(text: str) -> str:
            return f"Contract analysis of {text}"

        async def contract_summary(text: str) -> str:
            return f"Contract summary: {text}"

        builder.register_node(self.ENTRY_NODE, contract_analysis)
        builder.register_node("contract_summary", contract_summary)
        builder.add_edge(self.ENTRY_NODE, "contract_summary")


class LitigationAnalysisPath:
    """Subgraph for litigation-related documents."""

    ENTRY_NODE = "litigation_analysis"

    def build(self, builder: WorkflowBuilder) -> None:
        async def litigation_analysis(text: str) -> str:
            return f"Litigation analysis of {text}"

        async def litigation_summary(text: str) -> str:
            return f"Litigation summary: {text}"

        builder.register_node(self.ENTRY_NODE, litigation_analysis)
        builder.register_node("litigation_summary", litigation_summary)
        builder.add_edge(self.ENTRY_NODE, "litigation_summary")


class RegulatoryAnalysisPath:
    """Subgraph for statutory or regulatory documents."""

    ENTRY_NODE = "regulatory_analysis"

    def build(self, builder: WorkflowBuilder) -> None:
        async def regulatory_analysis(text: str) -> str:
            return f"Regulatory analysis of {text}"

        async def regulatory_summary(text: str) -> str:
            return f"Regulatory summary: {text}"

        builder.register_node(self.ENTRY_NODE, regulatory_analysis)
        builder.register_node("regulatory_summary", regulatory_summary)
        builder.add_edge(self.ENTRY_NODE, "regulatory_summary")


class EvidenceAnalysisPath:
    """Fallback path for miscellaneous evidence documents."""

    ENTRY_NODE = "evidence_analysis"

    def build(self, builder: WorkflowBuilder) -> None:
        async def evidence_analysis(text: str) -> str:
            return f"Evidence analysis of {text}"

        async def evidence_summary(text: str) -> str:
            return f"Evidence summary: {text}"

        builder.register_node(self.ENTRY_NODE, evidence_analysis)
        builder.register_node("evidence_summary", evidence_summary)
        builder.add_edge(self.ENTRY_NODE, "evidence_summary")


__all__ = [
    "ContractAnalysisPath",
    "LitigationAnalysisPath",
    "RegulatoryAnalysisPath",
    "EvidenceAnalysisPath",
]
