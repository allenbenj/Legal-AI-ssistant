"""
Integration workflow for combining DocumentProcessor and OntologyExtraction agents.

This module orchestrates the flow from document processing to ontology-driven
entity and relationship extraction, creating a complete legal document analysis pipeline.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..core.base_agent import BaseAgent
from ..core.types import LegalDocument, ProcessingResult
from ..agents.document_processor import DocumentProcessorAgent
from ..agents.ontology_extraction import OntologyExtractionAgent, OntologyExtractionResult


@dataclass
class IntegratedAnalysisResult:
    """Complete analysis result combining document processing and ontology extraction."""
    document_id: str
    document_processing: ProcessingResult
    ontology_extraction: OntologyExtractionResult
    integration_metadata: Dict[str, Any]
    total_processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'document_processing': self.document_processing.to_dict() if hasattr(self.document_processing, 'to_dict') else str(self.document_processing),
            'ontology_extraction': self.ontology_extraction.to_dict(),
            'integration_metadata': self.integration_metadata,
            'total_processing_time': self.total_processing_time
        }


class DocumentOntologyWorkflow:
    """
    Workflow that integrates document processing with ontology extraction.
    
    This workflow:
    1. Processes documents using DocumentProcessorAgent
    2. Extracts structured legal information using OntologyExtractionAgent  
    3. Combines results into a comprehensive legal document analysis
    """
    
    def __init__(self, services, **config):
        self.services = services
        self.config = config
        self.logger = services.logger
        
        # Initialize agents
        self.document_processor = DocumentProcessorAgent(services, **config)
        self.ontology_extractor = OntologyExtractionAgent(services, **config)
        
        # Workflow configuration
        self.process_in_parallel = config.get('process_in_parallel', False)
        self.skip_ontology_for_failed_processing = config.get('skip_ontology_for_failed_processing', True)
        self.max_retries = config.get('max_retries', 2)
    
    async def process_document(self, document_path: str, **kwargs) -> IntegratedAnalysisResult:
        """
        Process a document through the complete analysis pipeline.
        
        Args:
            document_path: Path to the document to process
            **kwargs: Additional processing options
            
        Returns:
            IntegratedAnalysisResult with complete analysis
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Document Processing
            self.logger.info(f"Starting document processing for {document_path}")
            
            processing_result = await self._process_document_with_retry(document_path, **kwargs)
            
            if not processing_result or not self._is_processing_successful(processing_result):
                if self.skip_ontology_for_failed_processing:
                    return self._create_failed_result(document_path, start_time, processing_result)
                
            # Step 2: Create LegalDocument object for ontology extraction
            legal_document = self._create_legal_document_from_processing(processing_result, document_path)
            
            # Step 3: Ontology Extraction
            self.logger.info(f"Starting ontology extraction for {document_path}")
            
            ontology_result = await self._extract_ontology_with_retry(legal_document)
            
            # Step 4: Combine results
            total_time = (datetime.now() - start_time).total_seconds()
            
            return IntegratedAnalysisResult(
                document_id=legal_document.id,
                document_processing=processing_result,
                ontology_extraction=ontology_result,
                integration_metadata={
                    'workflow_version': '1.0',
                    'processing_time_breakdown': {
                        'document_processing': getattr(processing_result, 'processing_time', 0),
                        'ontology_extraction': ontology_result.processing_time,
                        'integration_overhead': total_time - (
                            getattr(processing_result, 'processing_time', 0) + ontology_result.processing_time
                        )
                    },
                    'document_path': document_path,
                    'configuration': {
                        'parallel_processing': self.process_in_parallel,
                        'skip_failed': self.skip_ontology_for_failed_processing
                    }
                },
                total_processing_time=total_time
            )
            
        except Exception as e:
            self.logger.error(f"Workflow failed for {document_path}: {e}")
            return self._create_error_result(document_path, start_time, str(e))
    
    async def process_multiple_documents(self, document_paths: List[str], **kwargs) -> List[IntegratedAnalysisResult]:
        """
        Process multiple documents through the analysis pipeline.
        
        Args:
            document_paths: List of document paths to process
            **kwargs: Additional processing options
            
        Returns:
            List of IntegratedAnalysisResults
        """
        self.logger.info(f"Processing {len(document_paths)} documents")
        
        if self.process_in_parallel:
            # Process documents in parallel
            tasks = [self.process_document(path, **kwargs) for path in document_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to process {document_paths[i]}: {result}")
                    processed_results.append(
                        self._create_error_result(document_paths[i], datetime.now(), str(result))
                    )
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # Process documents sequentially
            results = []
            for path in document_paths:
                try:
                    result = await self.process_document(path, **kwargs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {path}: {e}")
                    results.append(self._create_error_result(path, datetime.now(), str(e)))
            
            return results
    
    async def _process_document_with_retry(self, document_path: str, **kwargs) -> Optional[ProcessingResult]:
        """Process document with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                return await self.document_processor.process_file(document_path, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Document processing failed after {self.max_retries + 1} attempts: {e}")
                    raise
                else:
                    self.logger.warning(f"Document processing attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
        
        return None
    
    async def _extract_ontology_with_retry(self, legal_document: LegalDocument) -> OntologyExtractionResult:
        """Extract ontology with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                return await self.ontology_extractor.process_document(legal_document)
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Ontology extraction failed after {self.max_retries + 1} attempts: {e}")
                    raise
                else:
                    self.logger.warning(f"Ontology extraction attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(1)
        
        return self.ontology_extractor._create_empty_result(legal_document.id, datetime.now())
    
    def _is_processing_successful(self, result: ProcessingResult) -> bool:
        """Check if document processing was successful."""
        if not result:
            return False
        
        # Check for specific success indicators
        if hasattr(result, 'success') and not result.success:
            return False
        
        if hasattr(result, 'content') and result.content:
            return True
        
        if hasattr(result, 'text') and result.text:
            return True
        
        # If we have extracted data, consider it successful
        if hasattr(result, 'extracted_data') and result.extracted_data:
            return True
        
        return False
    
    def _create_legal_document_from_processing(self, processing_result: ProcessingResult, document_path: str) -> LegalDocument:
        """Create a LegalDocument object from processing results."""
        document_id = f"doc_{hash(document_path) % 10000}"
        
        # Extract content based on processing result structure
        content = ""
        if hasattr(processing_result, 'content') and processing_result.content:
            content = processing_result.content
        elif hasattr(processing_result, 'text') and processing_result.text:
            content = processing_result.text
        elif hasattr(processing_result, 'extracted_data') and processing_result.extracted_data:
            # Handle structured data results
            if isinstance(processing_result.extracted_data, dict):
                content = str(processing_result.extracted_data)
            else:
                content = processing_result.extracted_data
        
        # Create LegalDocument
        legal_document = LegalDocument(
            id=document_id,
            file_path=document_path,
            content=content,
            metadata={
                'original_processing_result': processing_result,
                'created_from_workflow': True
            }
        )
        
        # Add processed content if available
        if hasattr(processing_result, 'processed_content'):
            legal_document.processed_content = processing_result.processed_content
        
        return legal_document
    
    def _create_failed_result(self, document_path: str, start_time: datetime, processing_result: Optional[ProcessingResult]) -> IntegratedAnalysisResult:
        """Create result for failed document processing."""
        document_id = f"failed_doc_{hash(document_path) % 10000}"
        
        # Create empty ontology result
        empty_ontology = OntologyExtractionResult(
            document_id=document_id,
            entities=[],
            relationships=[],
            extraction_metadata={'error': 'Skipped due to failed document processing'},
            processing_time=0.0,
            confidence_scores={'overall': 0.0}
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return IntegratedAnalysisResult(
            document_id=document_id,
            document_processing=processing_result,
            ontology_extraction=empty_ontology,
            integration_metadata={
                'status': 'failed',
                'error': 'Document processing failed',
                'document_path': document_path
            },
            total_processing_time=total_time
        )
    
    def _create_error_result(self, document_path: str, start_time: datetime, error_message: str) -> IntegratedAnalysisResult:
        """Create result for workflow errors."""
        document_id = f"error_doc_{hash(document_path) % 10000}"
        
        empty_ontology = OntologyExtractionResult(
            document_id=document_id,
            entities=[],
            relationships=[],
            extraction_metadata={'error': error_message},
            processing_time=0.0,
            confidence_scores={'overall': 0.0}
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return IntegratedAnalysisResult(
            document_id=document_id,
            document_processing=None,
            ontology_extraction=empty_ontology,
            integration_metadata={
                'status': 'error',
                'error': error_message,
                'document_path': document_path
            },
            total_processing_time=total_time
        )
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of the integrated workflow."""
        doc_processor_status = await self.document_processor.get_health_status()
        ontology_extractor_status = await self.ontology_extractor.get_health_status()
        
        return {
            'workflow_name': 'DocumentOntologyWorkflow',
            'status': 'healthy',
            'configuration': {
                'parallel_processing': self.process_in_parallel,
                'skip_failed_processing': self.skip_ontology_for_failed_processing,
                'max_retries': self.max_retries
            },
            'agents': {
                'document_processor': doc_processor_status,
                'ontology_extractor': ontology_extractor_status
            }
        }