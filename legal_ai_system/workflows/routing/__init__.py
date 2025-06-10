"""Routing workflow components for advanced legal analysis."""

from .document_classification_node import DocumentClassificationNode
from .analysis_paths import (
    ContractAnalysisPath,
    LitigationAnalysisPath,
    RegulatoryAnalysisPath,
    EvidenceAnalysisPath,
)
from .advanced_workflow import build_advanced_legal_workflow

__all__ = [
    "DocumentClassificationNode",
    "ContractAnalysisPath",
    "LitigationAnalysisPath",
    "RegulatoryAnalysisPath",
    "EvidenceAnalysisPath",
    "build_advanced_legal_workflow",
]
