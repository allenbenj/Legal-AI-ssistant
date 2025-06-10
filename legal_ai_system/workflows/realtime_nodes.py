

from __future__ import annotations

import time

from typing import Any, Dict

from ..services.realtime_analysis_workflow import RealTimeAnalysisWorkflow


class LegalWorkflowNode:


    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError




    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Document processing", 0.1)
        phase_start = time.time()

        dp_result = await workflow.document_processor.process(
            state["document_path"], metadata=state.get("metadata", {})
        )
        document_result = dp_result.data if dp_result else None


        if not document_result or not workflow._is_processing_successful(
            document_result
        ):
            raise ValueError("Document processing failed")


    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Document rewriting", 0.2)
        phase_start = time.time()

        document_text = workflow._extract_text_from_result(state["document_result"])
        rewrite_res = await workflow.document_rewriter.rewrite_text(
            document_text, {"document_id": state["document_id"]}
        )

        state["rewrite_result"] = rewrite_res
        state["document_text"] = rewrite_res.corrected_text
        return state




    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Hybrid entity extraction", 0.3)
        phase_start = time.time()

        legal_doc = workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],

        state["hybrid_result"] = hyb_res
        return state




    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Ontology extraction", 0.5)
        phase_start = time.time()

        legal_doc = workflow._create_legal_document(
            state["document_result"],
            state["document_path"],
            state["document_id"],

    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Building knowledge graph", 0.7)
        phase_start = time.time()



    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Updating vector store", 0.8)
        phase_start = time.time()



    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        await workflow._notify_progress("Memory integration", 0.9)
        phase_start = time.time()



    async def __call__(
        self, workflow: RealTimeAnalysisWorkflow, state: Dict[str, Any]
    ) -> Dict[str, Any]:

        )
        state.setdefault("processing_times", {})["validation"] = (
            time.time() - phase_start
        )

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
