from __future__ import annotations

import time
from typing import Any, Dict


class BaseRealtimeNode:
    """Base class for realtime workflow nodes."""

    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DocumentProcessingNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Document processing", 0.1)
        phase_start = time.time()
        dp_result = await wf.document_processor.process(context["document_path"], metadata=context.get("kwargs", {}))
        document_result = dp_result.data if dp_result else None
        context["document_processing"] = document_result
        context["processing_times"]["document_processing"] = time.time() - phase_start
        if not document_result or not wf._is_processing_successful(document_result):
            raise ValueError("Document processing failed")
        return context


class DocumentRewritingNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Document rewriting", 0.2)
        phase_start = time.time()
        document_text = wf._extract_text_from_result(context["document_processing"])
        rewrite_res = await wf.document_rewriter.rewrite_text(document_text, {"document_id": context["document_id"]})
        context["rewritten_text"] = rewrite_res.corrected_text
        context["processing_times"]["document_rewriting"] = time.time() - phase_start
        return context


class HybridExtractionNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Hybrid entity extraction", 0.3)
        phase_start = time.time()
        legal_doc = wf._create_legal_document(context["document_processing"], context["document_path"], context["document_id"], text_override=context.get("rewritten_text"))
        hyb_res = await wf.hybrid_extractor.extract_from_document(legal_doc)
        context["hybrid_extraction"] = hyb_res
        context["processing_times"]["hybrid_extraction"] = time.time() - phase_start
        return context


class OntologyExtractionNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Ontology extraction", 0.5)
        phase_start = time.time()
        legal_doc = wf._create_legal_document(context["document_processing"], context["document_path"], context["document_id"], text_override=context.get("rewritten_text"))
        ontology_result = await wf.ontology_extractor.process(legal_doc)
        context["ontology_extraction"] = ontology_result
        context["processing_times"]["ontology_extraction"] = time.time() - phase_start
        return context


class GraphBuildingNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Building knowledge graph", 0.7)
        phase_start = time.time()
        graph_updates = await wf._process_entities_realtime(context["hybrid_extraction"], context["ontology_extraction"], context["document_id"])
        context["graph_updates"] = graph_updates
        context["processing_times"]["graph_building"] = time.time() - phase_start
        return context


class VectorStoreUpdateNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Updating vector store", 0.8)
        phase_start = time.time()
        vec_updates = await wf._update_vector_store_realtime(context["hybrid_extraction"], context.get("rewritten_text", ""), context["document_id"])
        context["vector_updates"] = vec_updates
        context["processing_times"]["vector_updates"] = time.time() - phase_start
        return context


class MemoryIntegrationNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Memory integration", 0.9)
        phase_start = time.time()
        mem_updates = await wf._integrate_with_memory(context["hybrid_extraction"], context["ontology_extraction"], context["document_path"])
        context["memory_updates"] = mem_updates
        context["processing_times"]["memory_integration"] = time.time() - phase_start
        return context


class ValidationNode(BaseRealtimeNode):
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wf = context["workflow"]
        await wf._notify_progress("Validation and quality assessment", 0.95)
        phase_start = time.time()
        validation_results = await wf._validate_extraction_quality(context["hybrid_extraction"], context["ontology_extraction"], context["graph_updates"])
        confidence_scores = wf._calculate_confidence_scores(context["hybrid_extraction"], context["ontology_extraction"], validation_results)
        context["validation_results"] = validation_results
        context["confidence_scores"] = confidence_scores
        context["processing_times"]["validation"] = time.time() - phase_start
        return context

__all__ = [
    "BaseRealtimeNode",
    "DocumentProcessingNode",
    "DocumentRewritingNode",
    "HybridExtractionNode",
    "OntologyExtractionNode",
    "GraphBuildingNode",
    "VectorStoreUpdateNode",
    "MemoryIntegrationNode",
    "ValidationNode",
]
