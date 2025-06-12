"""Agent package initialization with logical categories."""

from .semantic_analysis_agent import SemanticAnalysisAgent
from .structural_analysis_agent import StructuralAnalysisAgent
from .citation_analysis_agent import CitationAnalysisAgent
from .legal_analysis_agent import LegalAnalysisAgent
from .document_processor_agent import DocumentProcessorAgent
from .document_processor_agent_v2 import DocumentProcessorAgentV2
from .document_rewriter_agent import DocumentRewriterAgent
from .text_correction_agent import TextCorrectionAgent
from .knowledge_base_agent import KnowledgeBaseAgent
from .knowledge_graph_reasoning_agent import KnowledgeGraphReasoningAgent
from .legal_reasoning_engine import LegalReasoningEngine
from .graph_inference_agent import GraphInferenceAgent
from .precedent_matching_agent import PrecedentMatchingAgent
from .entity_extraction_agent import StreamlinedEntityExtractionAgent
from .ontology_extraction_agent import OntologyExtractionAgent
from .auto_tagging_agent import AutoTaggingAgent
from .violation_detector_agent import ViolationDetectorAgent
from .note_taking_agent import NoteTakingAgent

# Mapping of agent categories to lists of agent classes for easier introspection
# and registration. Importing here keeps all agents discoverable from a single
# location without altering the existing dynamic loader in the service container.
AGENT_CATEGORIES = {
    "analysis": [
        SemanticAnalysisAgent,
        StructuralAnalysisAgent,
        CitationAnalysisAgent,
        LegalAnalysisAgent,
    ],
    "document_processing": [
        DocumentProcessorAgent,
        DocumentProcessorAgentV2,
        DocumentRewriterAgent,
        TextCorrectionAgent,
    ],
    "knowledge_reasoning": [
        KnowledgeBaseAgent,
        KnowledgeGraphReasoningAgent,
        LegalReasoningEngine,
        GraphInferenceAgent,
        PrecedentMatchingAgent,
    ],
    "extraction_classification": [
        StreamlinedEntityExtractionAgent,
        OntologyExtractionAgent,
        AutoTaggingAgent,
        ViolationDetectorAgent,
    ],
    "utility": [
        NoteTakingAgent,
    ],
}

__all__ = ["AGENT_CATEGORIES"]
