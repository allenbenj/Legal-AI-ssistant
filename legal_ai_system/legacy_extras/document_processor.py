"""
Document Processor Agent - Restored Full Functionality
Now uses the complete document processor with PDF, DOCX, OCR, and Excel support
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_agent import BaseAgent
from .document_processor_full import DocumentProcessorAgent as FullDocumentProcessor


class DocumentProcessorAgent(BaseAgent):
    """
    Document Processor Agent - Full-featured document processing
    
    Restored to use the complete document processor with all format support:
    - PDF processing with PyMuPDF
    - DOCX/DOC processing 
    - OCR for images and scanned documents
    - Excel/CSV structured data
    - PowerPoint reference extraction
    - HTML and Markdown processing
    """
    
    def __init__(self, service_container):
        super().__init__(service_container, "DocumentProcessor")
        
        # Use the full-featured processor with service container
        self.processor = FullDocumentProcessor(service_container)
        
        # Get manager instances for enhanced functionality
        self.config_manager = self.get_config_manager()
        self.security_manager = self.get_security_manager()
        self.memory_manager = self.get_memory_manager()
        self.knowledge_graph_manager = self.get_knowledge_graph_manager()
        
        logger.info("DocumentProcessor initialized with full functionality restored and manager integration")
    
    async def _process_task(self, task_data, metadata):
        """
        Process document using full-featured processor with security validation.
        Supports all major document formats.
        """
        # Enhanced processing with security validation if available
        if self.security_manager and isinstance(task_data, dict) and 'content' in task_data:
            try:
                # Use security manager for secure processing if we have content and path
                if 'document_path' in task_data and 'user_session' in metadata:
                    secure_result = self.security_manager.process_document_securely(
                        content=task_data['content'],
                        user_session=metadata['user_session'],
                        document_path=task_data['document_path']
                    )
                    
                    # Update task_data with secure content
                    task_data['content'] = secure_result['content']
                    metadata['pii_detected'] = secure_result.get('pii_detected', {})
                    metadata['secure_processing'] = True
                    
                    logger.info(f"Document processed securely by {self.name}", extra={
                        'pii_types_detected': list(secure_result.get('pii_detected', {}).keys()),
                        'processed_by': secure_result.get('processed_by')
                    })
                    
            except Exception as e:
                logger.warning(f"Security processing failed, falling back to standard processing: {e}")
                metadata['secure_processing'] = False
        
        # Delegate to full processor
        result = await self.processor._process_task(task_data, metadata)
        
        # Store result in memory manager if available
        if self.memory_manager and isinstance(result, dict):
            try:
                await self.memory_manager.store_processing_result(
                    agent_name=self.name,
                    task_data=task_data,
                    result=result,
                    metadata=metadata
                )
                logger.debug(f"Processing result stored in memory manager by {self.name}")
            except Exception as e:
                logger.warning(f"Failed to store result in memory manager: {e}")
        
        return result
    
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return await self.processor.get_supported_formats()
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        return await self.processor.get_processing_stats()


# Maintain compatibility exports
__all__ = ["DocumentProcessorAgent"]