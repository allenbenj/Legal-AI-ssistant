"""Workflow nodes for real-time analysis pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict

from ..services.realtime_analysis_workflow import RealTimeAnalysisWorkflow


class LegalWorkflowNode:
    """Base interface for workflow nodes."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError


class DocumentProcessingNode(LegalWorkflowNode):
    """Run document processing phase."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Document processing", 0.1)
        phase_start = time.time()

        dp_result = await workflow.document_processor.process(
            state["document_path"], metadata=state.get("metadata", {})
        )
        document_result = dp_result.data if dp_result else None
        state.setdefault("processing_times", {})[
            "document_processing"
        ] = time.time() - phase_start
        state["document_result"] = document_result

        if not document_result or not workflow._is_processing_successful(
            document_result
        ):
            raise ValueError("Document processing failed")
        return state


class DocumentRewriteNode(LegalWorkflowNode):
    """Rewrite processed document text."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Document rewriting", 0.2)
        phase_start = time.time()

        document_text = workflow._extract_text_from_result(state["document_result"])
        rewrite_res = await workflow.document_rewriter.rewrite_text(
            document_text, {"document_id": state["document_id"]}
        )
        state.setdefault("processing_times", {})[
            "document_rewriting"
        ] = time.time() - phase_start
        state["rewrite_result"] = rewrite_res
        state["document_text"] = rewrite_res.corrected_text
        return state


class HybridExtractionNode(LegalWorkflowNode):
    """Perform hybrid entity extraction."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Hybrid entity extraction", 0.3)
        phase_start = time.time()

        legal_doc = workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],
            text_override=state.get("document_text"),
        )
        hyb_res = await workflow.hybrid_extractor.extract_from_document(legal_doc)
        state.setdefault("processing_times", {})[
            "hybrid_extraction"
        ] = time.time() - phase_start
        state["hybrid_result"] = hyb_res
        return state


class OntologyExtractionNode(LegalWorkflowNode):
    """Extract entities using ontology extractor."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Ontology extraction", 0.5)
        phase_start = time.time()

        legal_doc = workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],
            text_override=state.get("document_text"),
        )
        ontology_result = await workflow.ontology_extractor.process(legal_doc)
        state.setdefault("processing_times", {})[
            "ontology_extraction"
        ] = time.time() - phase_start
        state["ontology_result"] = ontology_result
        return state


class GraphUpdateNode(LegalWorkflowNode):
    """Update knowledge graph in real-time."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Building knowledge graph", 0.7)
        phase_start = time.time()

        graph_updates = await workflow._process_entities_realtime(
            state["hybrid_result"], state["ontology_result"], state["document_id"]
        )
        state.setdefault("processing_times", {})[
            "graph_building"
        ] = time.time() - phase_start
        state["graph_updates"] = graph_updates
        return state


class VectorStoreUpdateNode(LegalWorkflowNode):
    """Update vector store with extracted vectors."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Updating vector store", 0.8)
        phase_start = time.time()

        vector_updates = await workflow._update_vector_store_realtime(
            state["hybrid_result"], state.get("document_text", ""), state["document_id"]
        )
        state.setdefault("processing_times", {})[
            "vector_updates"
        ] = time.time() - phase_start
        state["vector_updates"] = vector_updates
        return state


class MemoryIntegrationNode(LegalWorkflowNode):
    """Integrate results with reviewable memory."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Memory integration", 0.9)
        phase_start = time.time()

        memory_updates = await workflow._integrate_with_memory(
            state["hybrid_result"], state["ontology_result"], state["document_path"]
        )
        state.setdefault("processing_times", {})[
            "memory_integration"
        ] = time.time() - phase_start
        state["memory_updates"] = memory_updates
        return state


class ValidationNode(LegalWorkflowNode):
    """Validate extraction quality and compute confidence."""

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress(
            "Validation and quality assessment", 0.95
        )
        phase_start = time.time()

        validation_results = await workflow._validate_extraction_quality(
            state["hybrid_result"], state["ontology_result"], state["graph_updates"]
        )
        confidence_scores = workflow._calculate_confidence_scores(
            state["hybrid_result"], state["ontology_result"], validation_results
        )
        state.setdefault("processing_times", {})["validation"] = (
            time.time() - phase_start
        )
        state["validation_results"] = validation_results
        state["confidence_scores"] = confidence_scores
        return state


__all__ = [
    "LegalWorkflowNode",
    "DocumentProcessingNode",
    "DocumentRewriteNode",
    "HybridExtractionNode",
    "OntologyExtractionNode",
    "GraphUpdateNode",
    "VectorStoreUpdateNode",
    "MemoryIntegrationNode",
    "ValidationNode",
]
