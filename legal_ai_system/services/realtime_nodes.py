from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from .realtime_analysis_workflow import RealTimeAnalysisResult


class BaseRealtimeNode:
    """Base node that exposes the workflow instance."""

    def __init__(self, workflow: "RealTimeAnalysisWorkflow") -> None:
        self.workflow = workflow


class DocumentProcessingNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Document processing", 0.1)
        phase_start = time.time()
        dp_result = await self.workflow.document_processor.process(
            state["document_path"], metadata=state.get("metadata", {})
        )
        document_result = dp_result.data if dp_result else None
        state["document_result"] = document_result
        state["processing_times"]["document_processing"] = time.time() - phase_start
        if not document_result or not self.workflow._is_processing_successful(document_result):
            raise ValueError("Document processing failed")
        state["document_text"] = self.workflow._extract_text_from_result(document_result)
        return state


class DocumentRewritingNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Document rewriting", 0.2)
        phase_start = time.time()
        rewrite_res = await self.workflow.document_rewriter.rewrite_text(
            state["document_text"], {"document_id": state["document_id"]}
        )
        state["document_text"] = rewrite_res.corrected_text
        state["processing_times"]["document_rewriting"] = time.time() - phase_start
        return state


class HybridExtractionNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Hybrid entity extraction", 0.3)
        phase_start = time.time()
        legal_doc = self.workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],
            text_override=state["document_text"],
        )
        state["hybrid_result"] = await self.workflow.hybrid_extractor.extract_from_document(legal_doc)
        state["processing_times"]["hybrid_extraction"] = time.time() - phase_start
        return state


class OntologyExtractionNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Ontology extraction", 0.5)
        phase_start = time.time()
        legal_doc = self.workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],
            text_override=state["document_text"],
        )
        state["ontology_result"] = await self.workflow.ontology_extractor.process(legal_doc)
        state["processing_times"]["ontology_extraction"] = time.time() - phase_start
        return state


class GraphUpdateNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Building knowledge graph", 0.7)
        phase_start = time.time()
        state["graph_updates"] = await self.workflow._process_entities_realtime(
            state["hybrid_result"], state["ontology_result"], state["document_id"]
        )
        state["processing_times"]["graph_building"] = time.time() - phase_start
        return state


class VectorStoreUpdateNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Updating vector store", 0.8)
        phase_start = time.time()
        state["vector_updates"] = await self.workflow._update_vector_store_realtime(
            state["hybrid_result"], state["document_text"], state["document_id"]
        )
        state["processing_times"]["vector_updates"] = time.time() - phase_start
        return state


class MemoryIntegrationNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Memory integration", 0.9)
        phase_start = time.time()
        state["memory_updates"] = await self.workflow._integrate_with_memory(
            state["hybrid_result"], state["ontology_result"], state["document_path"]
        )
        state["processing_times"]["memory_integration"] = time.time() - phase_start
        return state


class ValidationNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await self.workflow._notify_progress("Validation and quality assessment", 0.95)
        phase_start = time.time()
        val = await self.workflow._validate_extraction_quality(
            state["hybrid_result"], state["ontology_result"], state["graph_updates"]
        )
        conf = self.workflow._calculate_confidence_scores(
            state["hybrid_result"], state["ontology_result"], val
        )
        state["validation_results"] = val
        state["confidence_scores"] = conf
        state["processing_times"]["validation"] = time.time() - phase_start
        return state


class ResultNode(BaseRealtimeNode):
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        total_time = time.time() - state["start_time"]
        result = RealTimeAnalysisResult(
            document_path=state["document_path"],
            document_id=state["document_id"],
            document_processing=state["document_result"],
            ontology_extraction=state["ontology_result"],
            hybrid_extraction=state["hybrid_result"],
            graph_updates=state["graph_updates"],
            vector_updates=state["vector_updates"],
            memory_updates=state["memory_updates"],
            processing_times=state["processing_times"],
            total_processing_time=total_time,
            confidence_scores=state["confidence_scores"],
            validation_results=state["validation_results"],
            sync_status=await self.workflow._get_sync_status(),
        )
        state["result"] = result
        await self.workflow._update_performance_stats(result)
        if self.workflow.documents_processed % self.workflow.auto_optimization_threshold == 0:
            asyncio.create_task(self.workflow._auto_optimize_system())
        await self.workflow._notify_progress("Analysis complete", 1.0)
        await self.workflow._notify_update("document_processed", result.to_dict())
        self.workflow.logger.info(
            f"Real-time analysis completed in {total_time:.2f}s"
        )
        return state


__all__ = [
    "DocumentProcessingNode",
    "DocumentRewritingNode",
    "HybridExtractionNode",
    "OntologyExtractionNode",
    "GraphUpdateNode",
    "VectorStoreUpdateNode",
    "MemoryIntegrationNode",
    "ValidationNode",
    "ResultNode",
]
