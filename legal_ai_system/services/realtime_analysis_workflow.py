"""
Real-Time Legal Document Analysis Workflow.

This module orchestrates the complete real-time analysis pipeline combining
document processing, hybrid extraction, knowledge graph building, and
agent memory integration with user feedback loops.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .workflow_config import WorkflowConfig

from ..agents.document_processor_agent import DocumentProcessorAgent
from ..agents.document_rewriter_agent import DocumentRewriterAgent
from ..agents.ontology_extraction_agent import OntologyExtractionAgent
from ..core.optimized_vector_store import OptimizedVectorStore
from ..utils.hybrid_extractor import HybridLegalExtractor
from ..utils.reviewable_memory import (
    ReviewableMemory,
    ReviewDecision,
    ReviewStatus,
)
from .realtime_graph_manager import RealTimeGraphManager
from ..config.workflow_config import WorkflowConfig
from ..config.agent_unified_config import AGENT_CLASS_MAP


@dataclass
class RealTimeAnalysisResult:
    """Complete result from real-time analysis workflow."""

    document_path: str
    document_id: str

    # Processing results
    document_processing: Any
    ontology_extraction: Any
    hybrid_extraction: Any

    # Graph and memory updates
    graph_updates: Dict[str, Any]
    vector_updates: Dict[str, Any]
    memory_updates: Dict[str, Any]

    # Performance metrics
    processing_times: Dict[str, float]
    total_processing_time: float

    # Quality metrics
    confidence_scores: Dict[str, float]
    validation_results: Dict[str, Any]

    # Real-time sync status
    sync_status: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_path": self.document_path,
            "document_id": self.document_id,
            "processing_times": self.processing_times,
            "total_processing_time": self.total_processing_time,
            "confidence_scores": self.confidence_scores,
            "validation_results": self.validation_results,
            "sync_status": self.sync_status,
            "graph_updates": self.graph_updates,
            "vector_updates": self.vector_updates,
            "memory_updates": self.memory_updates,
        }


class RealTimeAnalysisWorkflow:
    """
    Master workflow for real-time legal document analysis.

    Features:
    - Real-time document processing and entity extraction
    - Hybrid NER+LLM extraction with validation
    - Automatic knowledge graph building and synchronization
    - Vector store optimization with intelligent caching
    - Agent memory integration with user feedback loops
    - Performance monitoring and optimization
    """


        self.logger = services.logger


        # Performance tracking
        self.documents_processed = 0
        self.processing_times: List[float] = []
        self.performance_stats = {}

        # Callbacks for real-time updates
        self.progress_callbacks: List[Callable] = []
        self.update_callbacks: List[Callable] = []

        # User feedback integration
        self.feedback_callback: Optional[Callable] = None
        self.pending_feedback: Dict[str, Any] = {}

        # Synchronization
        self.processing_lock = asyncio.Semaphore(self.max_concurrent_documents)
        self.optimization_lock = asyncio.Lock()

    def _apply_config(self) -> None:
        """Apply configuration dictionary to instance attributes."""
        cfg = self.config
        self.enable_real_time_sync = cfg.get("enable_real_time_sync", True)
        self.confidence_threshold = cfg.get("confidence_threshold", 0.75)
        self.enable_user_feedback = cfg.get("enable_user_feedback", True)
        self.parallel_processing = cfg.get("parallel_processing", True)
        self.max_concurrent_documents = cfg.get("max_concurrent_documents", 3)
        self.performance_monitoring = cfg.get("performance_monitoring", True)
        self.auto_optimization_threshold = cfg.get("auto_optimization_threshold", 100)

        self.processing_lock = asyncio.Semaphore(self.max_concurrent_documents)

    def update_config(self, **new_config: Any) -> None:
        """Update workflow configuration in-place."""
        self.config.update(new_config)
        self._apply_config()

    async def initialize(self):
        """Initialize the real-time analysis workflow."""
        self.logger.info("Initializing real-time analysis workflow...")

        # Initialize all components
        await self.hybrid_extractor.initialize()
        await self.graph_manager.initialize_service()
        await self.vector_store.initialize()
        await self.reviewable_memory.initialize()

        # Register callbacks between components

        self.logger.info("Real-time analysis workflow initialized")

    async def process_document_realtime(
        self, document_path: str, **kwargs
    ) -> RealTimeAnalysisResult:
        """
        Process a document through the complete real-time analysis pipeline.

        Args:
            document_path: Path to document to process
            **kwargs: Additional processing options

        Returns:
            RealTimeAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        document_id = f"doc_{hash(document_path) % 100000}_{int(time.time())}"

        async with self.processing_lock:
            try:
                self.logger.info(f"Starting real-time analysis for: {document_path}")
                processing_times = {}

                # Phase 1: Document Processing
                await self._notify_progress("Document processing", 0.1)
                phase_start = time.time()

                dp_result = await self.document_processor.process(
                    document_path, metadata=kwargs
                )
                document_result = dp_result.data if dp_result else None
                processing_times["document_processing"] = time.time() - phase_start

                if not document_result or not self._is_processing_successful(
                    document_result
                ):
                    raise ValueError("Document processing failed")

                # Phase 2: Document Rewriting
                await self._notify_progress("Document rewriting", 0.2)
                phase_start = time.time()

                document_text = self._extract_text_from_result(document_result)
                rewrite_res = await self.document_rewriter.rewrite_text(
                    document_text, {"document_id": document_id}
                )
                processing_times["document_rewriting"] = time.time() - phase_start

                # Phase 3: Hybrid Extraction (NER + LLM)
                await self._notify_progress("Hybrid entity extraction", 0.3)
                phase_start = time.time()

                document_text = rewrite_res.corrected_text

                legal_doc_for_hybrid = self._create_legal_document(
                    document_result,
                    document_path,
                    document_id,
                    text_override=document_text,
                )
                hyb_res = await self.hybrid_extractor.extract_from_document(
                    legal_doc_for_hybrid
                )
                hybrid_result = hyb_res
                processing_times["hybrid_extraction"] = time.time() - phase_start

                # Phase 3: Ontology Extraction (for compatibility)
                await self._notify_progress("Ontology extraction", 0.5)
                phase_start = time.time()

                legal_document = self._create_legal_document(
                    document_result,
                    document_path,
                    document_id,
                    text_override=document_text,
                )
                ontology_result = await self.ontology_extractor.process(legal_document)
                processing_times["ontology_extraction"] = time.time() - phase_start

                # Phase 4: Real-Time Graph Building
                await self._notify_progress("Building knowledge graph", 0.7)
                phase_start = time.time()

                graph_updates = await self._process_entities_realtime(
                    hybrid_result, ontology_result, document_id
                )
                processing_times["graph_building"] = time.time() - phase_start

                # Phase 5: Vector Store Updates
                await self._notify_progress("Updating vector store", 0.8)
                phase_start = time.time()

                vector_updates = await self._update_vector_store_realtime(
                    hybrid_result, document_text, document_id
                )
                processing_times["vector_updates"] = time.time() - phase_start

                # Phase 6: Memory Integration with User Feedback
                await self._notify_progress("Memory integration", 0.9)
                phase_start = time.time()

                memory_updates = await self._integrate_with_memory(
                    hybrid_result, ontology_result, document_path
                )
                processing_times["memory_integration"] = time.time() - phase_start

                # Phase 7: Validation and Quality Assessment
                await self._notify_progress("Validation and quality assessment", 0.95)
                phase_start = time.time()

                validation_results = await self._validate_extraction_quality(
                    hybrid_result, ontology_result, graph_updates
                )
                confidence_scores = self._calculate_confidence_scores(
                    hybrid_result, ontology_result, validation_results
                )
                processing_times["validation"] = time.time() - phase_start

                # Calculate total processing time
                total_time = time.time() - start_time

                # Create comprehensive result
                result = RealTimeAnalysisResult(
                    document_path=document_path,
                    document_id=document_id,
                    document_processing=document_result,
                    ontology_extraction=ontology_result,
                    hybrid_extraction=hybrid_result,
                    graph_updates=graph_updates,
                    vector_updates=vector_updates,
                    memory_updates=memory_updates,
                    processing_times=processing_times,
                    total_processing_time=total_time,
                    confidence_scores=confidence_scores,
                    validation_results=validation_results,
                    sync_status=await self._get_sync_status(),
                )

                # Update performance tracking
                await self._update_performance_stats(result)

                # Trigger auto-optimization if needed
                if self.documents_processed % self.auto_optimization_threshold == 0:
                    asyncio.create_task(self._auto_optimize_system())

                await self._notify_progress("Analysis complete", 1.0)
                await self._notify_update("document_processed", result.to_dict())

                self.logger.info(f"Real-time analysis completed in {total_time:.2f}s")
                return result

            except Exception as e:
                self.logger.error(f"Real-time analysis failed for {document_path}: {e}")
                raise

    async def _process_entities_realtime(
        self, hybrid_result, ontology_result, document_id: str
    ) -> Dict[str, Any]:
        """Process entities in real-time and update knowledge graph."""
        graph_updates = {
            "nodes_created": 0,
            "nodes_updated": 0,
            "edges_created": 0,
            "edges_updated": 0,
            "hybrid_entities_processed": 0,
            "ontology_entities_processed": 0,
        }

        try:
            # Process hybrid extraction entities
            for entity in hybrid_result.validated_entities:
                if entity.confidence >= self.confidence_threshold:
                    node_id = await self.graph_manager.process_entity_realtime(
                        self._convert_to_extracted_entity(entity),
                        document_id,
                        {"extraction_method": "hybrid"},
                    )

                    if node_id:
                        graph_updates["nodes_created"] += 1
                        graph_updates["hybrid_entities_processed"] += 1

            # Process ontology extraction entities
            for entity in ontology_result.entities:
                if entity.confidence >= self.confidence_threshold:
                    node_id = await self.graph_manager.process_entity_realtime(
                        entity, document_id, {"extraction_method": "ontology"}
                    )

                    if node_id:
                        graph_updates["nodes_created"] += 1
                        graph_updates["ontology_entities_processed"] += 1

            # Process relationships
            for relationship in ontology_result.relationships:
                if relationship.confidence >= self.confidence_threshold:
                    edge_id = await self.graph_manager.process_relationship_realtime(
                        relationship, document_id, {"extraction_method": "ontology"}
                    )

                    if edge_id:
                        graph_updates["edges_created"] += 1

        except Exception as e:
            self.logger.error(f"Real-time graph processing failed: {e}")

        return graph_updates

    async def _update_vector_store_realtime(
        self, hybrid_result, document_text: str, document_id: str
    ) -> Dict[str, Any]:
        """Update vector store with extracted entities in real-time."""
        vector_updates = {"vectors_added": 0, "processing_time": 0.0}

        try:
            start_time = time.time()

            # Add document-level vector
            await self.vector_store.add_vector_async(
                content_to_embed=document_text[:1000],  # Limit size
                document_id_ref=document_id,
                index_target="document",
                confidence_score=0.9,
                source_file=hybrid_result.document_id,
                custom_metadata={
                    "extraction_timestamp": datetime.now().isoformat()
                },
            )
            vector_updates["vectors_added"] += 1

            # Add entity vectors
            for entity in hybrid_result.validated_entities:
                if entity.confidence >= self.confidence_threshold:
                    await self.vector_store.add_vector_async(
                        content_to_embed=entity.entity_text,
                        document_id_ref=document_id,
                        index_target="entity",
                        vector_id_override=f"{entity.consensus_type}_{hash(entity.entity_text) % 10000}",
                        confidence_score=entity.confidence,
                        source_file=document_id,
                        custom_metadata={
                            "extraction_method": "hybrid",
                            "discrepancy": entity.discrepancy,
                        },
                    )
                    vector_updates["vectors_added"] += 1

            # Add targeted extraction results
            for extraction_type, results in hybrid_result.targeted_extractions.items():
                for result in results:
                    if result.get("confidence", 0) >= self.confidence_threshold:
                        await self.vector_store.add_vector_async(
                            content_to_embed=result.get("description", ""),
                            document_id_ref=document_id,
                            index_target="entity",
                            vector_id_override=f"{extraction_type}_{hash(str(result)) % 10000}",
                            confidence_score=result.get("confidence", 0.8),
                            source_file=document_id,
                            custom_metadata={
                                "targeted_extraction": True,
                                "extraction_type": extraction_type,
                            },
                        )
                        vector_updates["vectors_added"] += 1

            vector_updates["processing_time"] = time.time() - start_time

        except Exception as e:
            self.logger.error(f"Vector store update failed: {e}")

        return vector_updates

    async def _integrate_with_memory(
        self, hybrid_result, ontology_result, document_path: str
    ) -> Dict[str, Any]:
        """Integrate results with reviewable memory system."""
        memory_updates = {
            "items_added_to_review": 0,
            "auto_approved": 0,
            "flagged_for_review": 0,
        }

        try:
            # Process ontology results through reviewable memory
            review_stats = await self.reviewable_memory.process_extraction_result(
                ontology_result, document_path
            )

            memory_updates.update(review_stats)

            # Process high-priority hybrid extractions
            for extraction_type, results in hybrid_result.targeted_extractions.items():
                if extraction_type in [
                    "brady_violations",
                    "prosecutorial_misconduct",
                    "witness_tampering",
                ]:
                    # These require immediate review
                    for _ in results:
                        # Create review item for critical findings
                        # This would integrate with the reviewable memory system
                        memory_updates["flagged_for_review"] += 1

        except Exception as e:
            self.logger.error(f"Memory integration failed: {e}")
        finally:
            # Always fetch latest pending reviews for GUI updates
            try:
                pending = await self.reviewable_memory.get_pending_reviews_async(
                    limit=20
                )
                await self._notify_update(
                    "pending_reviews", [item.to_dict() for item in pending]
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch pending reviews: {e}")

        return memory_updates

    async def _validate_extraction_quality(
        self, hybrid_result, ontology_result, _graph_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the quality of extractions and cross-reference results."""
        validation = {
            "hybrid_validation_score": 0.0,
            "ontology_validation_score": 0.0,
            "cross_validation_score": 0.0,
            "discrepancy_count": 0,
            "consensus_entities": 0,
        }

        try:
            # Calculate hybrid validation score
            if hybrid_result.validated_entities:
                discrepancies = sum(
                    1 for e in hybrid_result.validated_entities if e.discrepancy
                )
                validation["discrepancy_count"] = discrepancies
                validation["consensus_entities"] = (
                    len(hybrid_result.validated_entities) - discrepancies
                )
                validation["hybrid_validation_score"] = 1.0 - (
                    discrepancies / len(hybrid_result.validated_entities)
                )

            # Calculate ontology validation score
            if ontology_result.entities:
                high_confidence_entities = sum(
                    1 for e in ontology_result.entities if e.confidence >= 0.8
                )
                validation["ontology_validation_score"] = (
                    high_confidence_entities / len(ontology_result.entities)
                )

            # Cross-validation between hybrid and ontology
            common_entities = self._find_common_entities(hybrid_result, ontology_result)
            if common_entities:
                validation["cross_validation_score"] = len(common_entities) / max(
                    len(hybrid_result.validated_entities),
                    len(ontology_result.entities),
                    1,
                )

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")

        return validation

    def _find_common_entities(self, hybrid_result, ontology_result) -> List[str]:
        """Find entities that appear in both hybrid and ontology results."""
        hybrid_texts = {e.entity_text.lower() for e in hybrid_result.validated_entities}
        ontology_texts = {e.source_text.lower() for e in ontology_result.entities}

        return list(hybrid_texts.intersection(ontology_texts))

    def _calculate_confidence_scores(
        self, hybrid_result, ontology_result, validation_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive confidence scores."""
        scores = {}

        # Overall confidence based on validation
        scores["overall"] = (
            validation_results.get("hybrid_validation_score", 0) * 0.4
            + validation_results.get("ontology_validation_score", 0) * 0.3
            + validation_results.get("cross_validation_score", 0) * 0.3
        )

        # Entity-level confidence
        if hybrid_result.validated_entities:
            scores["entities"] = sum(
                e.confidence for e in hybrid_result.validated_entities
            ) / len(hybrid_result.validated_entities)

        # Relationship confidence
        if ontology_result.relationships:
            scores["relationships"] = sum(
                r.confidence for r in ontology_result.relationships
            ) / len(ontology_result.relationships)

        # Targeted extraction confidence
        targeted_confidences = []
        for results in hybrid_result.targeted_extractions.values():
            for result in results:
                if "confidence" in result:
                    targeted_confidences.append(result["confidence"])

        if targeted_confidences:
            scores["targeted_extractions"] = sum(targeted_confidences) / len(
                targeted_confidences
            )

        return scores

    async def _get_sync_status(self) -> Dict[str, str]:
        """Get synchronization status across all components."""
        return {
            "graph_sync": "synced" if self.enable_real_time_sync else "disabled",
            "vector_sync": "synced",
            "memory_sync": "synced",
        }

    async def _update_performance_stats(self, result: RealTimeAnalysisResult):
        """Update performance statistics."""
        self.documents_processed += 1
        self.processing_times.append(result.total_processing_time)

        # Keep only recent performance data
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]

        self.performance_stats = {
            "documents_processed": self.documents_processed,
            "avg_processing_time": sum(self.processing_times)
            / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "last_update": datetime.now().isoformat(),
        }

    async def _auto_optimize_system(self):
        """Automatically optimize system performance."""
        async with self.optimization_lock:
            try:
                self.logger.info("Starting auto-optimization...")

                # Optimize vector store
                vector_optimization = await self.vector_store.optimize_performance()

                self.logger.info(
                    f"Auto-optimization completed: vector={vector_optimization['optimization_completed']}"
                )

            except Exception as e:
                self.logger.error(f"Auto-optimization failed: {e}")

    # Utility methods
    def _is_processing_successful(self, result) -> bool:
        """Check if document processing was successful."""
        return result is not None and (
            (hasattr(result, "success") and result.success)
            or (hasattr(result, "content") and result.content)
            or (hasattr(result, "text") and result.text)
        )

    def _extract_text_from_result(self, result) -> str:
        """Extract text content from processing result."""
        if hasattr(result, "content") and result.content:
            return result.content
        elif hasattr(result, "text") and result.text:
            return result.text
        elif hasattr(result, "extracted_data"):
            return str(result.extracted_data)
        else:
            return ""

    def _create_legal_document(
        self,
        result,
        document_path: str,
        document_id: str,
        text_override: Optional[str] = None,
    ):
        """Create legal document object for ontology extraction."""
        from ..core.models import LegalDocument

        return LegalDocument(
            id=document_id,
            file_path=Path(document_path),
            content=text_override or self._extract_text_from_result(result),
            metadata={"processing_result": result},
        )

    def _convert_to_extracted_entity(self, validation_result):
        """Convert validation result to extracted entity format."""
        from ..agents.ontology_extraction_agent import ExtractedEntity

        return ExtractedEntity(
            entity_type=validation_result.consensus_type,
            entity_id=f"{validation_result.consensus_type}_{hash(validation_result.entity_text) % 10000}",
            attributes={"name": validation_result.entity_text},
            confidence=validation_result.confidence,
            source_text_snippet=validation_result.entity_text,
            span=(0, len(validation_result.entity_text)),
        )

    # Callback management
    def register_progress_callback(self, callback: Callable):
        """Register callback for progress updates."""
        self.progress_callbacks.append(callback)

    def register_update_callback(self, callback: Callable):
        """Register callback for real-time updates."""
        self.update_callbacks.append(callback)

    def set_feedback_callback(self, callback: Callable):
        """Set callback for user feedback integration."""
        self.feedback_callback = callback

    async def _notify_progress(self, message: str, progress: float):
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                await callback(message, progress)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")

    async def _notify_update(self, event_type: str, data: Dict[str, Any]):
        """Notify update callbacks."""
        for callback in self.update_callbacks:
            try:
                await callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Update callback failed: {e}")

    async def _on_graph_update(self, event_type: str, data: Dict[str, Any]):
        """Handle graph update events."""
        await self._notify_update(f"graph_{event_type}", data)

    async def fetch_pending_reviews(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return pending review items for the GUI."""
        try:
            items = await self.reviewable_memory.get_pending_reviews_async(limit=limit)
            return [item.to_dict() for item in items]
        except Exception as e:
            self.logger.error(f"Failed to fetch pending reviews: {e}")
            return []

    async def submit_review_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Submit user review decision to the reviewable memory."""
        if not feedback:
            return False
        try:
            decision = ReviewDecision(
                item_id=feedback["item_id"],
                decision=ReviewStatus(feedback.get("decision", "approved")),
                reviewer_id=feedback.get("reviewer_id", "gui"),
                modified_content=feedback.get("modified_content"),
                reviewer_notes=feedback.get("reviewer_notes", ""),
                confidence_override=feedback.get("confidence_override"),
            )
            success = await self.reviewable_memory.submit_review_decision_async(decision)
            if success:
                await self._notify_update(
                    "review_decision",
                    {"item_id": decision.item_id, "decision": decision.decision.value},
                )
            return success
        except Exception as e:
            self.logger.error(f"Review feedback processing failed: {e}")
            return False

    # System management
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        graph_stats = await self.graph_manager.get_realtime_stats()
        vector_stats = await self.vector_store.get_service_status()
        memory_stats = await self.reviewable_memory.get_review_stats_async()
        extraction_stats = {}

        return {
            "workflow_stats": self.performance_stats,
            "graph_stats": graph_stats,
            "vector_stats": vector_stats,
            "memory_stats": memory_stats,
            "extraction_stats": extraction_stats,
            "system_health": {
                "components_initialized": True,
                "real_time_sync_enabled": self.enable_real_time_sync,
                "documents_processed": self.documents_processed,
            },
        }

    async def force_system_sync(self) -> Dict[str, Any]:
        """Force synchronization across all components."""
        sync_results = {}

        try:
            # Vector store optimization
            sync_results["vector_store"] = (
                await self.vector_store.optimize_performance()
            )

            # Get memory stats (no force sync needed)
            sync_results["memory"] = (
                await self.reviewable_memory.get_review_stats_async()
            )

        except Exception as e:
            self.logger.error(f"System sync failed: {e}")
            sync_results["error"] = str(e)

        return sync_results

    async def close(self):
        """Close the real-time analysis workflow."""
        # Close all components
        await self.hybrid_extractor.close()
        await self.graph_manager.close()
        await self.vector_store.close()
        await self.reviewable_memory.close()

        self.logger.info("Real-time analysis workflow closed")
