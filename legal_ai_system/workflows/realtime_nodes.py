from __future__ import annotations

import time
from typing import Any, Dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Avoid circular import at runtime
    from ..services.realtime_analysis_workflow import RealTimeAnalysisWorkflow


class LegalWorkflowNode:
    """Base interface for workflow processing nodes."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DocumentProcessingNode(LegalWorkflowNode):
    """Run the :class:`DocumentProcessorAgent` and store the result."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Document processing", 0.1)
        phase_start = time.time()

        dp_result = await workflow.document_processor.process(
            state["document_path"], metadata=state.get("metadata", {})
        )
        document_result = getattr(dp_result, "data", dp_result)

        if not document_result or not workflow._is_processing_successful(document_result):
            raise ValueError("Document processing failed")

        state["document_result"] = document_result
        state.setdefault("processing_times", {})["document_processing"] = time.time() - phase_start
        return state


class DocumentRewritingNode(LegalWorkflowNode):
    """Clean up extracted text using the :class:`DocumentRewriterAgent`."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Document rewriting", 0.2)
        phase_start = time.time()

        document_text = workflow._extract_text_from_result(state["document_result"])
        rewrite_res = await workflow.document_rewriter.rewrite_text(
            document_text, {"document_id": state["document_id"]}
        )

        state["rewrite_result"] = rewrite_res
        state["document_text"] = rewrite_res.corrected_text
        state.setdefault("processing_times", {})["document_rewrite"] = time.time() - phase_start
        return state


# Backwards compatibility alias
DocumentRewriteNode = DocumentRewritingNode


class HybridExtractionNode(LegalWorkflowNode):
    """Perform hybrid entity extraction."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Hybrid entity extraction", 0.3)
        phase_start = time.time()

        legal_doc = workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],
            text_override=state.get("document_text"),
        )
        hyb_res = await workflow.hybrid_extractor.extract_from_document(legal_doc)

        state["hybrid_result"] = hyb_res
        state.setdefault("processing_times", {})["hybrid_extraction"] = time.time() - phase_start
        return state


class OntologyExtractionNode(LegalWorkflowNode):
    """Run ontology extraction on the legal document."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Ontology extraction", 0.5)
        phase_start = time.time()

        legal_doc = workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],
            text_override=state.get("document_text"),
        )
        ont_res = await workflow.ontology_extractor.process(legal_doc)

        state["ontology_result"] = ont_res
        state.setdefault("processing_times", {})["ontology_extraction"] = time.time() - phase_start
        return state


class GraphBuildingNode(LegalWorkflowNode):
    """Update the knowledge graph with extracted entities."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Building knowledge graph", 0.7)
        phase_start = time.time()

        graph_updates = await workflow._process_entities_realtime(
            state["hybrid_result"], state["ontology_result"], state["document_id"]
        )

        state["graph_updates"] = graph_updates
        state.setdefault("processing_times", {})["graph_update"] = time.time() - phase_start
        return state


class VectorStoreUpdateNode(LegalWorkflowNode):
    """Persist embeddings from the extraction results."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Updating vector store", 0.8)
        phase_start = time.time()

        vector_updates = await workflow._update_vector_store_realtime(
            state["hybrid_result"], state.get("document_text", ""), state["document_id"]
        )

        state["vector_updates"] = vector_updates
        state.setdefault("processing_times", {})["vector_store_update"] = time.time() - phase_start
        return state


class MemoryIntegrationNode(LegalWorkflowNode):
    """Store results in the reviewable memory system."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Memory integration", 0.9)
        phase_start = time.time()

        memory_updates = await workflow._integrate_with_memory(
            state["hybrid_result"], state["ontology_result"], state["document_path"]
        )

        state["memory_updates"] = memory_updates
        state.setdefault("processing_times", {})["memory_integration"] = time.time() - phase_start
        return state


class ValidationNode(LegalWorkflowNode):
    """Validate extractions and compute confidence scores."""

    async def __call__(self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]) -> Dict[str, Any]:
        await workflow._notify_progress("Validation", 1.0)
        phase_start = time.time()

        validation = await workflow._validate_extraction_quality(
            state["hybrid_result"], state["ontology_result"], state.get("graph_updates", {})
        )
        scores = workflow._calculate_confidence_scores(
            state["hybrid_result"], state["ontology_result"], validation
        )
        sync_status = await workflow._get_sync_status()

        state["validation_results"] = validation
        state["confidence_scores"] = scores
        state["sync_status"] = sync_status
        state.setdefault("processing_times", {})["validation"] = time.time() - phase_start
        return state


__all__ = [
    "LegalWorkflowNode",
    "DocumentProcessingNode",
    "DocumentRewriteNode",
    "HybridExtractionNode",
    "OntologyExtractionNode",
    "GraphBuildingNode",
    "VectorStoreUpdateNode",
    "MemoryIntegrationNode",
    "ValidationNode",
]
