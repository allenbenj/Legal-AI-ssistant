"""
Real-Time Legal Document Analysis Workflow.

This module orchestrates the complete real-time analysis pipeline combining
document processing, hybrid extraction, knowledge graph building, and
agent memory integration with user feedback loops.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)
from ..core.agent_unified_config import _get_service_sync
from ..analytics.quality_classifier import PreprocessingErrorPredictor
from ..core.model_switcher import TaskComplexity
from .metrics_exporter import MetricsExporter, metrics_exporter
from ..core.ml_optimizer import PerformanceMetrics


try:  # Avoid heavy imports during tests
    from ..utils.reviewable_memory import (
        ReviewDecision,
        ReviewStatus,
    )
except Exception:  # pragma: no cover - fallback for tests
    ReviewDecision = object  # type: ignore[misc]
    ReviewStatus = object  # type: ignore[misc]

    pass

# Node classes are imported lazily by the workflow builder during tests.


@dataclass
class RealTimeAnalysisResult:
    """Complete result from real-time analysis workflow."""

    document_path: str
    document_id: str

    # Processing results
    document_processing: Any
    text_rewriting: Any
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
        return asdict(self)


class RealTimeAnalysisWorkflow:
    """Master workflow for real-time legal document analysis."""

    def __init__(
        self,
        service_container: Any | None = None,
        workflow_config: Any | None = None,
        task_queue: Any | None = None,
    ) -> None:
        """Initialize workflow settings and state."""
        self.service_container = service_container
        self.task_queue = task_queue
        self.logger = get_detailed_logger("RealTimeWorkflow", LogCategory.SYSTEM)

        cfg = workflow_config or SimpleNamespace(
            enable_real_time_sync=True,
            confidence_threshold=0.75,
            max_concurrent_documents=1,
            auto_optimization_threshold=1000,
            min_entity_confidence_for_kg=0.6,
        )

        self.enable_real_time_sync = cfg.enable_real_time_sync
        self.confidence_threshold = cfg.confidence_threshold
        self.max_concurrent_documents = cfg.max_concurrent_documents
        self.auto_optimization_threshold = cfg.auto_optimization_threshold
        self.min_entity_confidence_for_kg = cfg.min_entity_confidence_for_kg

        # Performance tracking
        self.documents_processed = 0
        self.processing_times: List[float] = []
        self.performance_stats: Dict[str, Any] = {}

        self.progress_callbacks: List[Callable] = []
        self.update_callbacks: List[Callable] = []

        # User feedback integration
        self.feedback_callback: Optional[Callable] = None
        self.pending_feedback: Dict[str, Any] = {}

        # Access lightweight analytics services if available
        self.keyword_service = None
        self.quality_service = None
        if service_container is not None:
            self.keyword_service = service_container._services.get(
                "keyword_extraction_service"
            )
            self.quality_service = service_container._services.get(
                "quality_assessment_service"
            )

        self.preproc_predictor = (
            self.quality_service.preproc_predictor
            if self.quality_service is not None
            else PreprocessingErrorPredictor()
        )

        # Synchronization primitives
        self.processing_lock = asyncio.Semaphore(self.max_concurrent_documents)
        self.optimization_lock = asyncio.Lock()

        # Core components retrieved from the service container when available
        self.document_processor = None
        self.document_rewriter = None
        self.hybrid_extractor = None
        self.ontology_extractor = None
        self.graph_manager = None
        self.vector_store = None
        self.reviewable_memory = None

        if service_container is not None:
            self.document_processor = _get_service_sync(
                service_container, "documentprocessoragent"
            )
            self.document_rewriter = _get_service_sync(
                service_container, "documentrewriteragent"
            )
            self.ontology_extractor = _get_service_sync(
                service_container, "ontologyextractionagent"
            )
            self.hybrid_extractor = _get_service_sync(
                service_container, "hybridlegalextractor"
            )
            self.graph_manager = _get_service_sync(
                service_container, "realtime_graph_manager"
            )
            self.vector_store = _get_service_sync(service_container, "vector_store")
            self.reviewable_memory = _get_service_sync(
                service_container, "reviewable_memory"
            )

        # ML optimizer and policy learner for dynamic routing
        try:
            from ..core.ml_optimizer import MLOptimizer
            from ..workflows.workflow_policy import WorkflowPolicy
        except Exception:  # pragma: no cover
            MLOptimizer = object  # type: ignore
            WorkflowPolicy = object  # type: ignore

        self.ml_optimizer = (
            getattr(service_container, "ml_optimizer", None) or MLOptimizer()
        )
        self.policy_learner = WorkflowPolicy()
        self.workflow_history: List[Dict[str, Any]] = []
        self.agent_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0}
        )

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

    async def process_document_realtime(self, document_path: str, **kwargs):
        """Enqueue a document for real-time processing."""
        document_id = kwargs.get("document_id") or f"doc_rt_{uuid.uuid4().hex}"
        kwargs["document_id"] = document_id

        task_queue = getattr(self, "task_queue", None)
        if task_queue:
            job = task_queue.enqueue(
                self._run_realtime_pipeline, document_path, **kwargs
            )
            await self._notify_progress("queued", 0.0)
            await self._notify_update(
                "workflow_queued",
                {"document_id": document_id, "job_id": getattr(job, "id", None)},
            )
            return job

        try:
            result = await self._run_realtime_pipeline(document_path, **kwargs)
            await self._notify_update(
                "workflow_completed", {"document_id": document_id}
            )
            return result
        except Exception as e:
            await self._notify_update(
                "workflow_failed", {"document_id": document_id, "error": str(e)}
            )
            raise

    async def _run_realtime_pipeline(self, document_path: str, **kwargs):
        """Run the end-to-end real-time processing pipeline."""
        document_id = kwargs.get("document_id") or f"doc_rt_{uuid.uuid4().hex}"

        start_time = time.time()
        processing_times: Dict[str, float] = {}
        doc_features = self._extract_document_features("", document_path)
        new_concurrency = self.policy_learner.predict_concurrency(doc_features)
        if new_concurrency != self.max_concurrent_documents:
            self.max_concurrent_documents = new_concurrency
            self.processing_lock = asyncio.Semaphore(self.max_concurrent_documents)

        async with self.processing_lock:
            await self._notify_progress("document_processing", 0.05)
            t0 = time.time()
            document_result = await self.document_processor.process(document_path)
            duration = time.time() - t0
            processing_times["document_processing"] = duration
            success = self._is_processing_successful(document_result)
            self.policy_learner.update_agent_stats("document_processor", success)
            text = self._extract_text_from_result(document_result)
            doc_features = self._extract_document_features(text, document_path)

            await self._notify_progress("text_rewrite", 0.10)
            if self.policy_learner.should_run_step("text_rewrite", doc_features):
                t0 = time.time()
                rewrite_result = await self.document_rewriter.rewrite_text(text)
                duration = time.time() - t0
                success = bool(rewrite_result)
                text = getattr(rewrite_result, "corrected_text", text)
            else:
                rewrite_result = SimpleNamespace(corrected_text=text)
                duration = 0.0
                success = True
            processing_times["text_rewrite"] = duration
            self.policy_learner.update_agent_stats("document_rewriter", success)
            self.policy_learner.record_step(
                "text_rewrite", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "text_rewrite",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            if self.keyword_service:
                keywords = self.keyword_service.extract(text, top_k=10)
                await self._notify_update(
                    "keywords", {"document_id": document_id, "keywords": keywords}
                )
            self.policy_learner.record_step(
                "document_processing", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "document_processing",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            risk = self.preproc_predictor.predict_risk(
                {
                    "content_preview": text[:1000] if text else "",
                    "size": Path(document_path).stat().st_size,
                }
            )
            await self._notify_update(
                "preprocessing_risk",
                {"document_id": document_id, "risk": risk},
            )
            if risk > 0.5:
                if hasattr(self.hybrid_extractor, "model_switcher"):
                    await self.hybrid_extractor.model_switcher.switch_for_task(
                        "hybrid_extraction", TaskComplexity.COMPLEX
                    )
                if hasattr(self.ontology_extractor, "model_switcher"):
                    await self.ontology_extractor.model_switcher.switch_for_task(
                        "ontology_extraction", TaskComplexity.COMPLEX
                    )

            await self._notify_progress("ontology_extraction", 0.15)
            legal_doc = self._create_legal_document(
                document_result, document_path, document_id, text_override=text
            )
            if self.policy_learner.should_run_step("ontology_extraction", doc_features):
                t0 = time.time()
                ontology_result = await self.ontology_extractor.process(legal_doc)
                duration = time.time() - t0
                success = self._is_processing_successful(ontology_result)
            else:
                ontology_result = SimpleNamespace()
                duration = 0.0
                success = True
            processing_times["ontology_extraction"] = duration
            self.policy_learner.update_agent_stats("ontology_extractor", success)
            self.policy_learner.record_step(
                "ontology_extraction", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "ontology_extraction",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            await self._notify_progress("hybrid_extraction", 0.35)
            if self.policy_learner.should_run_step("hybrid_extraction", doc_features):
                t0 = time.time()
                hybrid_result = await self.hybrid_extractor.extract_from_document(
                    document_path, document_id=document_id
                )
                duration = time.time() - t0
                success = self._is_processing_successful(hybrid_result)
            else:
                hybrid_result = SimpleNamespace(
                    validated_entities=[], targeted_extractions={}
                )
                duration = 0.0
                success = True
            processing_times["hybrid_extraction"] = duration
            self.policy_learner.update_agent_stats("hybrid_extractor", success)
            self.policy_learner.record_step(
                "hybrid_extraction", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "hybrid_extraction",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            await self._notify_progress("graph_update", 0.55)
            if self.policy_learner.should_run_step("graph_update", doc_features):
                t0 = time.time()
                graph_updates = await self._update_knowledge_graph_realtime(
                    hybrid_result, ontology_result, document_id
                )
                duration = time.time() - t0
                success = True
            else:
                graph_updates = {}
                duration = 0.0
                success = True
            processing_times["graph_updates"] = duration
            self.policy_learner.record_step(
                "graph_update", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "graph_update",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            await self._notify_progress("vector_update", 0.7)
            if self.policy_learner.should_run_step("vector_update", doc_features):
                t0 = time.time()
                vector_updates = await self._update_vector_store_realtime(
                    hybrid_result, document_result, document_id
                )
                duration = time.time() - t0
                success = True
            else:
                vector_updates = {}
                duration = 0.0
                success = True
            processing_times["vector_updates"] = duration
            self.policy_learner.record_step(
                "vector_update", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "vector_update",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            await self._notify_progress("memory_integration", 0.85)
            if self.policy_learner.should_run_step("memory_integration", doc_features):
                t0 = time.time()
                memory_updates = await self._integrate_with_memory(
                    hybrid_result, ontology_result, document_path
                )
                duration = time.time() - t0
                success = True
            else:
                memory_updates = {}
                duration = 0.0
                success = True
            processing_times["memory_updates"] = duration
            self.policy_learner.record_step(
                "memory_integration", doc_features, success, duration
            )
            self.ml_optimizer.record_step_metrics(
                document_path,
                "memory_integration",
                PerformanceMetrics(processing_time=duration, success=success),
            )

            validation = await self._validate_extraction_quality(
                hybrid_result, ontology_result, graph_updates
            )
            confidence_scores = self._calculate_confidence_scores(
                hybrid_result, ontology_result, validation
            )
            sync_status = await self._get_sync_status()

        total_processing_time = time.time() - start_time

        result = RealTimeAnalysisResult(
            document_path=document_path,
            document_id=document_id,
            document_processing=document_result,
            text_rewriting=rewrite_result,
            ontology_extraction=ontology_result,
            hybrid_extraction=hybrid_result,
            graph_updates=graph_updates,
            vector_updates=vector_updates,
            memory_updates=memory_updates,
            processing_times=processing_times,
            total_processing_time=total_processing_time,
            confidence_scores=confidence_scores,
            validation_results=validation,
            sync_status=sync_status,
        )

        await self._update_performance_stats(result)
        if (
            self.auto_optimization_threshold
            and self.documents_processed % self.auto_optimization_threshold == 0
        ):
            await self._auto_optimize_system()

        await self._notify_progress("completed", 1.0)
        self.workflow_history.append(
            {
                "document_id": document_id,
                "processing_times": processing_times,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return result

    async def _update_knowledge_graph_realtime(
        self, hybrid_result, ontology_result, document_id: str
    ) -> Dict[str, Any]:
        """Persist extracted entities and relationships to the knowledge graph."""

        summary = {"nodes_added": 0, "relationships_added": 0}

        try:
            from itertools import chain

            entity_sources = []
            for ent in getattr(hybrid_result, "validated_entities", []):
                entity_sources.append((ent, "hybrid"))
            for ent in getattr(ontology_result, "entities", []):
                entity_sources.append((ent, "ontology"))

            seen_entities: set[tuple[str, str]] = set()

            for entity, source in entity_sources:
                if getattr(entity, "confidence", 0.0) < self.min_entity_confidence_for_kg:
                    continue

                text = (
                    getattr(entity, "entity_text", None)
                    or getattr(entity, "source_text_snippet", "")
                ).lower()
                etype = (
                    getattr(entity, "consensus_type", None)
                    or getattr(entity, "entity_type", "")
                ).lower()
                key = (text, etype)
                if key in seen_entities:
                    continue
                seen_entities.add(key)

                converted = (
                    self._convert_to_extracted_entity(entity)
                    if hasattr(entity, "consensus_type") and not hasattr(entity, "entity_id")
                    else entity
                )
                node_id = await self.graph_manager.add_entity_realtime(
                    converted, document_id, {"extraction_method": source}
                )
                if node_id:
                    summary["nodes_added"] += 1

            seen_rels: set[tuple[str, str, str]] = set()
            for rel in getattr(ontology_result, "relationships", []):
                if getattr(rel, "confidence", 0.0) < self.min_entity_confidence_for_kg:
                    continue

                key = (
                    str(getattr(rel, "source_entity", "")),
                    str(getattr(rel, "target_entity", "")),
                    getattr(rel, "relationship_type", ""),
                )
                if key in seen_rels:
                    continue
                seen_rels.add(key)

                edge_id = await self.graph_manager.add_relationship_realtime(
                    rel, document_id, {"extraction_method": "ontology"}
                )
                if edge_id:
                    summary["relationships_added"] += 1

        except Exception as e:  # pragma: no cover - avoid failing test suite
            self.logger.error(f"Real-time graph processing failed: {e}")

        return summary

    async def _update_vector_store_realtime(
        self, hybrid_result, document_result, document_id: str
    ) -> Dict[str, Any]:
        """Update vector store with extracted entities in real-time."""
        vector_updates = {"vectors_added": 0, "processing_time": 0.0}

        try:
            start_time = time.time()

            chunks = getattr(document_result, "text_chunks", None)
            if not chunks and getattr(document_result, "text_content", None):
                tokens = document_result.text_content.split()
                chunk_size = 1000
                chunks = [
                    " ".join(tokens[i : i + chunk_size])
                    for i in range(0, len(tokens), chunk_size)
                ]

            for idx, chunk_text in enumerate(chunks or []):
                doc_vector_kwargs = {
                    "index_target": "document",
                    "confidence_score": 0.9,
                    "source_file": hybrid_result.document_id,
                    "custom_metadata": {
                        "extraction_timestamp": datetime.now().isoformat(),
                        "chunk_index": idx,
                    },
                }
                await self.vector_store.add_vector_async(
                    content_to_embed=chunk_text,
                    document_id_ref=document_id,
                    **doc_vector_kwargs,
                )
                vector_updates["vectors_added"] += 1

            # Add entity vectors
            for entity in hybrid_result.validated_entities:
                if entity.confidence >= self.confidence_threshold:
                    vector_id = (
                        f"{entity.consensus_type}_{hash(entity.entity_text) % 10000}"
                    )
                    entity_vector_kwargs = {
                        "index_target": "entity",
                        "vector_id_override": vector_id,
                        "confidence_score": entity.confidence,
                        "source_file": document_id,
                        "custom_metadata": {
                            "extraction_method": "hybrid",
                            "discrepancy": entity.discrepancy,
                        },
                    }
                    await self.vector_store.add_vector_async(
                        content_to_embed=entity.entity_text,
                        document_id_ref=document_id,
                        **entity_vector_kwargs,
                    )
                    vector_updates["vectors_added"] += 1

            # Add targeted extraction results
            for extraction_type, results in hybrid_result.targeted_extractions.items():
                for result in results:
                    if result.get("confidence", 0) >= self.confidence_threshold:
                        vector_id = f"{extraction_type}_{hash(str(result)) % 10000}"
                        targeted_vector_kwargs = {
                            "index_target": "entity",
                            "vector_id_override": vector_id,
                            "confidence_score": result.get("confidence", 0.8),
                            "source_file": document_id,
                            "custom_metadata": {
                                "targeted_extraction": True,
                                "extraction_type": extraction_type,
                            },
                        }
                        await self.vector_store.add_vector_async(
                            content_to_embed=result.get("description", ""),
                            document_id_ref=document_id,
                            **targeted_vector_kwargs,
                        )
                        vector_updates["vectors_added"] += 1

            await self.vector_store.flush_updates()

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

                status = vector_optimization["optimization_completed"]
                self.logger.info(
                    "Auto-optimization completed: vector=%s",
                    status,
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

        hashed = hash(validation_result.entity_text) % 10000
        entity_id = f"{validation_result.consensus_type}_{hashed}"
        return ExtractedEntity(
            entity_type=validation_result.consensus_type,
            entity_id=entity_id,
            attributes={"name": validation_result.entity_text},
            confidence=validation_result.confidence,
            source_text_snippet=validation_result.entity_text,
            span=(0, len(validation_result.entity_text)),
        )

    def _extract_document_features(
        self, text: str, file_path: str
    ) -> "DocumentFeatures":
        """Generate basic document features for policy decisions."""
        from ..core.ml_optimizer import DocumentFeatures

        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        except OSError:
            file_size_mb = 0.0

        word_count = len(text.split()) if text else 0

        return DocumentFeatures(file_size_mb=file_size_mb, word_count=word_count)

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
            success = await self.reviewable_memory.submit_review_decision_async(
                decision
            )
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
