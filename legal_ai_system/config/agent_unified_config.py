"""Compatibility layer exposing unified agent configuration helpers."""

# Import names from the core module so they can be re-exported for backwards
# compatibility.  These imports appear unused in this module, but they are
# referenced via ``__all__`` below.  Linters such as PyDev may still flag them
# as unused, so we explicitly ignore that warning.
from ..core.agent_unified_config import (
    AgentConfigHelper,  # noqa: F401
    configure_all_agents_unified,  # noqa: F401
    create_agent_memory_mixin,  # noqa: F401
    get_agent_configuration_status,  # noqa: F401
    setup_agents_example,  # noqa: F401
    validate_agent_setup,  # noqa: F401
)

from ..agents.document_processor_agent import DocumentProcessorAgent
from ..agents.document_processor_agent_v2 import DocumentProcessorAgentV2
from ..agents.document_rewriter_agent import DocumentRewriterAgent
from ..agents.ontology_extraction_agent import OntologyExtractionAgent
from ..utils.hybrid_extractor import HybridLegalExtractor
from ..services.realtime_graph_manager import RealTimeGraphManager
from ..core.optimized_vector_store import OptimizedVectorStore
from ..utils.reviewable_memory import ReviewableMemory
from typing import Dict, Type, Any

# Registry of agent and utility classes exposed for dynamic configuration
AGENT_CLASS_REGISTRY: Dict[str, Type[Any]] = {
    "DocumentProcessorAgent": DocumentProcessorAgent,
    "DocumentProcessorAgentV2": DocumentProcessorAgentV2,
    "DocumentRewriterAgent": DocumentRewriterAgent,
    "OntologyExtractionAgent": OntologyExtractionAgent,
    "HybridLegalExtractor": HybridLegalExtractor,
    "RealTimeGraphManager": RealTimeGraphManager,
    "OptimizedVectorStore": OptimizedVectorStore,
    "ReviewableMemory": ReviewableMemory,
}

__all__ = [
    "AgentConfigHelper",
    "configure_all_agents_unified",
    "create_agent_memory_mixin",
    "get_agent_configuration_status",
    "setup_agents_example",
    "validate_agent_setup",
    "AGENT_CLASS_REGISTRY",

]
