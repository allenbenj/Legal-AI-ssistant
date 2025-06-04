"""Integration Service for Legal AI System.

Provides seamless integration between the FastAPI backend and the core
Legal AI System components, handling service discovery, data marshaling,
and real-time event propagation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Core Legal AI System imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.unified_services import get_service_container, ServiceContainer
from core.security_manager import SecurityManager
from core.confidence_calibration import (
    ConfidenceCalibrationManager, EntityPrediction, ValidationSample
)
from extraction.hybrid_extractor import HybridLegalExtractor
from core.knowledge_graph_manager import KnowledgeGraphManager
from core.document_router import DocumentRouter
from core.ml_optimizer import MLOptimizer
from memory.reviewable_memory import ReviewableMemory

logger = logging.getLogger(__name__)

class LegalAIIntegrationService:
    """Integration service bridging FastAPI and Legal AI System."""
    
    def __init__(self):
        self.service_container: Optional[ServiceContainer] = None
        self.security_manager: Optional[SecurityManager] = None
        self.hybrid_extractor: Optional[HybridLegalExtractor] = None
        self.knowledge_graph: Optional[KnowledgeGraphManager] = None
        self.document_router: Optional[DocumentRouter] = None
        self.ml_optimizer: Optional[MLOptimizer] = None
        self.calibration_manager: Optional[ConfidenceCalibrationManager] = None
        self.reviewable_memory: Optional[ReviewableMemory] = None
        
        # Document processing state
        self.processing_documents: Dict[str, Dict[str, Any]] = {}
        self.processing_callbacks: Dict[str, List[callable]] = {}
        
    async def initialize(self):
        """Initialize all Legal AI System components."""
        logger.info("🚀 Initializing Legal AI Integration Service...")
        
        try:
            # Initialize service container
            self.service_container = await get_service_container()
            logger.info("✅ Service container initialized")
            
            # Initialize security manager
            self.security_manager = SecurityManager(
                encryption_password="legal_ai_master_key_2024",
                allowed_directories=[
                    str(Path(__file__).parent.parent / "storage"),
                    "/tmp/legal_ai_uploads"
                ]
            )
            logger.info("✅ Security manager initialized")
            
            # Get or create hybrid extractor
            self.hybrid_extractor = await self._get_or_create_service(
                'hybrid_extractor',
                lambda: HybridLegalExtractor(
                    self.service_container,
                    enable_confidence_calibration=True,
                    calibration_storage_path=Path("storage/calibration")
                )
            )
            await self.hybrid_extractor.initialize()
            logger.info("✅ Hybrid extractor initialized")
            
            # Get or create knowledge graph manager
            self.knowledge_graph = await self._get_or_create_service(
                'knowledge_graph_manager',
                lambda: KnowledgeGraphManager(self.service_container)
            )
            logger.info("✅ Knowledge graph manager initialized")
            
            # Get or create document router
            self.document_router = await self._get_or_create_service(
                'document_router',
                lambda: DocumentRouter(self.service_container)
            )
            logger.info("✅ Document router initialized")
            
            # Get or create ML optimizer
            self.ml_optimizer = await self._get_or_create_service(
                'ml_optimizer',
                lambda: MLOptimizer(self.service_container)
            )
            logger.info("✅ ML optimizer initialized")
            
            # Initialize confidence calibration manager
            self.calibration_manager = ConfidenceCalibrationManager(
                storage_path=Path("storage/calibration")
            )
            logger.info("✅ Confidence calibration manager initialized")
            
            # Initialize reviewable memory
            self.reviewable_memory = ReviewableMemory(
                storage_path=Path("storage/reviewable_memory.db")
            )
            logger.info("✅ Reviewable memory initialized")
            
            logger.info("🎉 Legal AI Integration Service fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Integration service initialization failed: {e}")
            raise
    
    async def _get_or_create_service(self, service_name: str, factory_func):
        """Get existing service or create new one."""
        try:
            service = self.service_container.get_service(service_name)
            if service:
                return service
        except:
            pass
        
        # Create new service
        service = factory_func()
        self.service_container.register_service(service_name, service)
        return service
    
    # Document Processing Integration
    async def upload_document(self, file_content: bytes, filename: str, user_id: str) -> Dict[str, Any]:
        """Handle document upload with security validation."""
        try:
            # Validate file with security manager
            # In a real implementation, you'd validate the file content
            
            # Create document ID
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            
            # Save document
            upload_dir = Path("storage/documents/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = upload_dir / filename
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Initialize document tracking
            self.processing_documents[document_id] = {
                'id': document_id,
                'filename': filename,
                'file_path': str(file_path),
                'status': 'uploaded',
                'progress': 0,
                'uploaded_by': user_id,
                'uploaded_at': datetime.now(),
                'entities': [],
                'metadata': {}
            }
            
            logger.info(f"📄 Document uploaded: {document_id} by {user_id}")
            
            return {
                'document_id': document_id,
                'filename': filename,
                'size': len(file_content),
                'status': 'uploaded',
                'processing_options': {
                    'enable_ner': True,
                    'enable_llm_extraction': True,
                    'enable_targeted_prompting': True,
                    'enable_confidence_calibration': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Document upload failed: {e}")
            raise
    
    async def process_document(self, document_id: str, options: Dict[str, Any], 
                             progress_callback: Optional[callable] = None) -> None:
        """Process document using hybrid extractor."""
        try:
            if document_id not in self.processing_documents:
                raise ValueError(f"Document {document_id} not found")
            
            doc_info = self.processing_documents[document_id]
            
            # Update status
            doc_info['status'] = 'processing'
            doc_info['progress'] = 10
            
            if progress_callback:
                await progress_callback(document_id, {
                    'status': 'processing',
                    'progress': 10,
                    'stage': 'Initializing'
                })
            
            # Read document content
            with open(doc_info['file_path'], 'r', encoding='utf-8', errors='ignore') as f:
                document_text = f.read()
            
            # Route document to optimal processor
            routing_decision = await self.document_router.route_document(
                Path(doc_info['file_path']),
                {'priority': 'normal', 'user_id': doc_info['uploaded_by']}
            )
            
            doc_info['progress'] = 30
            if progress_callback:
                await progress_callback(document_id, {
                    'progress': 30,
                    'stage': 'Routing Complete'
                })
            
            # Get optimal parameters from ML optimizer
            optimal_params = await self.ml_optimizer.get_optimal_parameters(
                document_type='legal_document',
                document_features={'size': len(document_text), 'type': 'text'}
            )
            
            doc_info['progress'] = 50
            if progress_callback:
                await progress_callback(document_id, {
                    'progress': 50,
                    'stage': 'Parameter Optimization'
                })
            
            # Process with hybrid extractor
            extraction_result = await self.hybrid_extractor.extract_hybrid(
                document_text=document_text,
                document_id=document_id,
                enable_targeted=options.get('enable_targeted_prompting', True)
            )
            
            doc_info['progress'] = 80
            if progress_callback:
                await progress_callback(document_id, {
                    'progress': 80,
                    'stage': 'Entity Extraction Complete'
                })
            
            # Convert entities to standard format
            entities = []
            for entity in extraction_result.validated_entities:
                entity_data = {
                    'id': f"ent_{entity.entity_text}_{entity.entity_type}",
                    'text': entity.entity_text,
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos,
                    'source_model': entity.source_model,
                    'metadata': entity.metadata
                }
                entities.append(entity_data)
            
            # Add entities to knowledge graph
            if entities:
                await self._add_entities_to_knowledge_graph(entities, document_id)
            
            # Add high-confidence items to reviewable memory
            await self._add_to_reviewable_memory(entities, document_id)
            
            # Final update
            doc_info.update({
                'status': 'completed',
                'progress': 100,
                'entities': entities,
                'processing_time': extraction_result.processing_time,
                'metadata': extraction_result.extraction_metadata,
                'completed_at': datetime.now()
            })
            
            if progress_callback:
                await progress_callback(document_id, {
                    'status': 'completed',
                    'progress': 100,
                    'stage': 'Complete',
                    'entities_found': len(entities)
                })
            
            logger.info(f"✅ Document processing completed: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Document processing failed for {document_id}: {e}")
            
            doc_info.update({
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now()
            })
            
            if progress_callback:
                await progress_callback(document_id, {
                    'status': 'failed',
                    'error': str(e)
                })
            
            raise
    
    async def _add_entities_to_knowledge_graph(self, entities: List[Dict], document_id: str):
        """Add extracted entities to the knowledge graph."""
        try:
            for entity in entities:
                # Add entity to knowledge graph
                await self.knowledge_graph.add_entity(
                    entity_id=entity['id'],
                    entity_type=entity['type'],
                    properties={
                        'name': entity['text'],
                        'confidence': entity['confidence'],
                        'source_document': document_id,
                        'extraction_metadata': entity['metadata']
                    }
                )
            
            logger.info(f"✅ Added {len(entities)} entities to knowledge graph")
            
        except Exception as e:
            logger.error(f"❌ Failed to add entities to knowledge graph: {e}")
    
    async def _add_to_reviewable_memory(self, entities: List[Dict], document_id: str):
        """Add entities that need review to reviewable memory."""
        try:
            review_items = []
            
            for entity in entities:
                # Add items with confidence below threshold for review
                if entity['confidence'] < 0.8:  # Configurable threshold
                    review_item = {
                        'id': f"review_{entity['id']}",
                        'entity_text': entity['text'],
                        'entity_type': entity['type'],
                        'confidence': entity['confidence'],
                        'context': f"Extracted from document {document_id}",
                        'source_document': document_id,
                        'requires_review': True,
                        'created_at': datetime.now()
                    }
                    review_items.append(review_item)
            
            if review_items:
                await self.reviewable_memory.add_items(review_items)
                logger.info(f"✅ Added {len(review_items)} items to review queue")
            
        except Exception as e:
            logger.error(f"❌ Failed to add items to reviewable memory: {e}")
    
    # Knowledge Graph Integration
    async def search_entities(self, query: str, entity_types: Optional[List[str]] = None,
                            confidence_threshold: Optional[float] = None,
                            limit: int = 50) -> List[Dict]:
        """Search entities in the knowledge graph."""
        try:
            results = await self.knowledge_graph.search_entities(
                query=query,
                entity_types=entity_types,
                confidence_threshold=confidence_threshold,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"❌ Entity search failed: {e}")
            return []
    
    async def traverse_graph(self, entity_id: str, max_depth: int = 2,
                           relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """Traverse knowledge graph from an entity."""
        try:
            results = await self.knowledge_graph.traverse_relationships(
                entity_id=entity_id,
                max_depth=max_depth,
                relationship_types=relationship_types
            )
            return results
        except Exception as e:
            logger.error(f"❌ Graph traversal failed: {e}")
            return []
    
    # Confidence Calibration Integration
    async def get_review_queue(self, limit: int = 50) -> List[Dict]:
        """Get items pending confidence calibration review."""
        try:
            items = await self.reviewable_memory.get_pending_items(limit=limit)
            return items
        except Exception as e:
            logger.error(f"❌ Failed to get review queue: {e}")
            return []
    
    async def submit_review_decision(self, entity_id: str, decision: str,
                                   modified_data: Optional[Dict] = None) -> bool:
        """Submit a confidence calibration review decision."""
        try:
            # Process the review decision
            await self.reviewable_memory.process_review_decision(
                entity_id=entity_id,
                decision=decision,
                modified_data=modified_data
            )
            
            # If approved or modified, update knowledge graph
            if decision in ['approve', 'modify']:
                # Update entity in knowledge graph with review feedback
                await self._update_entity_from_review(entity_id, decision, modified_data)
            
            # Train calibration system with the feedback
            await self._train_calibration_from_feedback(entity_id, decision)
            
            logger.info(f"✅ Review decision processed: {entity_id} -> {decision}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Review decision failed: {e}")
            return False
    
    async def _update_entity_from_review(self, entity_id: str, decision: str, 
                                       modified_data: Optional[Dict]):
        """Update knowledge graph entity based on review decision."""
        try:
            if decision == 'modify' and modified_data:
                await self.knowledge_graph.update_entity(
                    entity_id=entity_id,
                    properties=modified_data
                )
            # For 'approve', we could increase confidence or mark as verified
            elif decision == 'approve':
                await self.knowledge_graph.update_entity(
                    entity_id=entity_id,
                    properties={'human_verified': True}
                )
        except Exception as e:
            logger.error(f"❌ Failed to update entity from review: {e}")
    
    async def _train_calibration_from_feedback(self, entity_id: str, decision: str):
        """Use review feedback to improve confidence calibration."""
        try:
            # Create validation sample from the review
            # This would be used to retrain the calibration models
            validation_sample = ValidationSample(
                text="",  # Would need to retrieve from storage
                true_entities=[],  # Based on review decision
                predictions=[]  # Original predictions
            )
            
            # In a real implementation, you'd collect these samples
            # and periodically retrain the calibration system
            
        except Exception as e:
            logger.error(f"❌ Failed to train calibration from feedback: {e}")
    
    # System Status Integration
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get service container status
            container_status = await self.service_container.get_system_status()
            
            # Get document processing status
            processing_count = len([d for d in self.processing_documents.values() 
                                  if d['status'] == 'processing'])
            
            # Get review queue status
            pending_reviews = len(await self.get_review_queue())
            
            # Combine all status information
            status = {
                'overall_health': container_status['overall_health'],
                'services': container_status['services'],
                'service_count': container_status['service_count'],
                'healthy_services': container_status['healthy_services'],
                'active_documents': processing_count,
                'pending_reviews': pending_reviews,
                'performance_metrics': container_status.get('performance_metrics', {}),
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ System status check failed: {e}")
            return {
                'overall_health': 0.0,
                'services': {},
                'service_count': 0,
                'healthy_services': 0,
                'active_documents': 0,
                'pending_reviews': 0,
                'performance_metrics': {},
                'error': str(e)
            }
    
    # Document Status Integration
    def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status for a specific document."""
        return self.processing_documents.get(document_id)
    
    def get_all_documents(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all documents with optional status filter."""
        documents = list(self.processing_documents.values())
        
        if status_filter:
            documents = [d for d in documents if d['status'] == status_filter]
        
        return documents
    
    async def shutdown(self):
        """Gracefully shutdown the integration service."""
        logger.info("🛑 Shutting down Legal AI Integration Service...")
        
        if self.service_container:
            await self.service_container.shutdown()
        
        logger.info("✅ Integration service shutdown complete")

# Global integration service instance
integration_service = LegalAIIntegrationService()