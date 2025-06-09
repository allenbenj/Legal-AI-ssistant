"""
Document Processor - Clean Implementation (No GUI Dependencies)
==============================================================
Professional document processing with xAI/Grok integration and comprehensive logging.
Streamlit-compatible, no PyQt6 dependencies.
"""

import sys
import os
import json
import time
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Core imports only - NO GUI dependencies
try:
    from ..core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
except ImportError:
    # Fallback for direct execution
    import logging
    def get_detailed_logger(name, category=None):
        return logging.getLogger(name)
    def detailed_log_function(category):
        def decorator(func):
            return func
        return decorator
    class LogCategory:
        DOCUMENT = "document"
        LLM = "llm"
        API = "api"

# Initialize loggers
main_logger = get_detailed_logger("Document_Processor_Clean", LogCategory.DOCUMENT)
file_logger = get_detailed_logger("File_Processing", LogCategory.DOCUMENT)
llm_logger = get_detailed_logger("LLM_Processing", LogCategory.LLM)
api_logger = get_detailed_logger("API_Calls", LogCategory.API)

@dataclass
class ProcessingResult:
    """Clean processing result without GUI dependencies"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    processing_time: float = 0.0

class CleanDocumentProcessor:
    """Document processor without any GUI dependencies"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "grok-3-mini"):
        """Initialize processor with API configuration"""
        main_logger.info("Initializing Clean Document Processor")
        
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        self.model = model
        self.supported_formats = ['.txt', '.pdf', '.docx', '.doc', '.md', '.csv', '.xlsx', '.xls']
        
        main_logger.info("Processor initialized", parameters={
            'model': self.model,
            'has_api_key': bool(self.api_key),
            'supported_formats': self.supported_formats
        })
    
    @detailed_log_function(LogCategory.DOCUMENT)
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a file and return clean results"""
        start_time = time.time()
        file_path = Path(file_path)
        
        main_logger.info(f"Processing file: {file_path}")
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            main_logger.error(error_msg)
            return ProcessingResult(
                success=False,
                content="",
                metadata={},
                error=error_msg
            )
        
        # Determine file type and process accordingly
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                result = self._process_text_file(file_path)
            elif file_extension == '.pdf':
                result = self._process_pdf_file(file_path)
            elif file_extension in ['.docx', '.doc']:
                result = self._process_word_file(file_path)
            elif file_extension == '.md':
                result = self._process_markdown_file(file_path)
            elif file_extension in ['.csv']:
                result = self._process_csv_file(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                result = self._process_excel_file(file_path)
            else:
                result = self._process_generic_file(file_path)
            
            processing_time = time.time() - start_time
            
            main_logger.info("File processing completed", parameters={
                'processing_time': processing_time,
                'content_length': len(result.get('content', '')),
                'success': True
            })
            
            return ProcessingResult(
                success=True,
                content=result.get('content', ''),
                metadata=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {file_path}: {str(e)}"
            main_logger.error(error_msg, exception=e)
            
            return ProcessingResult(
                success=False,
                content="",
                metadata={'file_path': str(file_path), 'file_extension': file_extension},
                error=error_msg,
                processing_time=processing_time
            )
    
    def _process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Process plain text files"""
        file_logger.trace(f"Reading text file: {file_path}")
        
        # Try multiple encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                file_logger.info(f"Text file read successfully with {encoding} encoding")
                return {
                    'content': content,
                    'encoding': encoding,
                    'file_size': file_path.stat().st_size,
                    'line_count': content.count('\n') + 1
                }
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read as binary and convert
        with open(file_path, 'rb') as f:
            raw_content = f.read()
        
        content = raw_content.decode('utf-8', errors='replace')
        file_logger.warning("Used fallback encoding with error replacement")
        
        return {
            'content': content,
            'encoding': 'utf-8_with_errors',
            'file_size': len(raw_content)
        }
    
    def _process_pdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF files"""
        file_logger.trace(f"Reading PDF file: {file_path}")
        
        try:
            import PyPDF2
            file_logger.trace("PyPDF2 library loaded successfully")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        file_logger.trace(f"Page {page_num + 1}: {len(page_text)} characters")
                        text_content.append(page_text)
                
                content = "\n".join(text_content)
                
                file_logger.info("PDF processed successfully", parameters={
                    'page_count': len(pdf_reader.pages),
                    'extracted_pages': len(text_content),
                    'total_content_length': len(content)
                })
                
                return {
                    'content': content,
                    'encoding': 'pdf_extracted',
                    'page_count': len(pdf_reader.pages),
                    'extraction_method': 'PyPDF2'
                }
                
        except ImportError:
            file_logger.warning("PyPDF2 not available, returning basic PDF info")
            return {
                'content': f"PDF document: {file_path.name} (PyPDF2 required for text extraction)",
                'encoding': 'pdf_metadata_only',
                'extraction_method': 'metadata_only'
            }
        except Exception as e:
            file_logger.error(f"Failed to extract PDF content: {file_path}", exception=e)
            return {
                'content': f"PDF document: {file_path.name} (extraction failed)",
                'encoding': 'pdf_error',
                'error': str(e)
            }
    
    def _process_word_file(self, file_path: Path) -> Dict[str, Any]:
        """Process Word documents"""
        file_logger.trace(f"Reading Word document: {file_path}")
        
        try:
            from docx import Document
            file_logger.trace("python-docx library loaded successfully")
            
            doc = Document(file_path)
            
            paragraphs = []
            for para_num, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text
                if para_text.strip():
                    file_logger.trace(f"Paragraph {para_num + 1}: {len(para_text)} characters")
                    paragraphs.append(para_text)
            
            content = "\n".join(paragraphs)
            
            file_logger.info("Word document processed successfully", parameters={
                'paragraph_count': len(paragraphs),
                'total_content_length': len(content)
            })
            
            return {
                'content': content,
                'encoding': 'docx_extracted',
                'paragraph_count': len(paragraphs),
                'extraction_method': 'python-docx'
            }
            
        except ImportError:
            file_logger.warning("python-docx not available")
            return {
                'content': f"Word document: {file_path.name} (python-docx required)",
                'encoding': 'docx_metadata_only',
                'extraction_method': 'metadata_only'
            }
        except Exception as e:
            file_logger.error(f"Failed to extract Word content: {file_path}", exception=e)
            return {
                'content': f"Word document: {file_path.name} (extraction failed)",
                'encoding': 'docx_error',
                'error': str(e)
            }
    
    def _process_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Process Markdown files"""
        return self._process_text_file(file_path)
    
    def _process_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV files"""
        file_logger.trace(f"Reading CSV file: {file_path}")
        
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # Create summary content
            content = f"CSV File: {file_path.name}\n"
            content += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            content += f"Column Names: {', '.join(df.columns)}\n\n"
            content += "First 5 rows:\n"
            content += df.head().to_string()
            
            if len(df) > 5:
                content += f"\n\n... (showing 5 of {len(df)} rows)"
            
            return {
                'content': content,
                'encoding': 'csv_processed',
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns)
            }
            
        except ImportError:
            return self._process_text_file(file_path)
        except Exception as e:
            file_logger.error(f"Failed to process CSV: {file_path}", exception=e)
            return self._process_text_file(file_path)
    
    def _process_excel_file(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel files"""
        file_logger.trace(f"Reading Excel file: {file_path}")
        
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            
            content = f"Excel File: {file_path.name}\n"
            content += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            content += f"Column Names: {', '.join(df.columns)}\n\n"
            content += "First 5 rows:\n"
            content += df.head().to_string()
            
            if len(df) > 5:
                content += f"\n\n... (showing 5 of {len(df)} rows)"
            
            return {
                'content': content,
                'encoding': 'excel_processed',
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns)
            }
            
        except ImportError:
            file_logger.warning("pandas not available for Excel processing")
            return {
                'content': f"Excel file: {file_path.name} (pandas required for processing)",
                'encoding': 'excel_metadata_only'
            }
        except Exception as e:
            file_logger.error(f"Failed to process Excel: {file_path}", exception=e)
            return {
                'content': f"Excel file: {file_path.name} (processing failed)",
                'encoding': 'excel_error',
                'error': str(e)
            }
    
    def _process_generic_file(self, file_path: Path) -> Dict[str, Any]:
        """Process unsupported file types"""
        return {
            'content': f"Unsupported file type: {file_path.name}",
            'encoding': 'unsupported',
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix
        }
    
    @detailed_log_function(LogCategory.LLM)
    def analyze_with_llm(self, content: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze content with LLM (no GUI dependencies)"""
        if not self.api_key:
            return {
                'success': False,
                'error': 'No API key configured',
                'analysis': None
            }
        
        prompt = self._build_analysis_prompt(content, analysis_type)
        
        try:
            import openai
            
            # Set up OpenAI client for xAI
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            
            llm_logger.info("LLM analysis completed successfully")
            
            return {
                'success': True,
                'analysis': analysis,
                'model': self.model,
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            llm_logger.error("LLM analysis failed", exception=e)
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }
    
    def _build_analysis_prompt(self, content: str, analysis_type: str) -> str:
        """Build analysis prompt based on type"""
        base_prompt = f"Analyze the following document content:\n\n{content}\n\n"
        
        if analysis_type == "legal":
            return base_prompt + "Provide a legal analysis focusing on key legal concepts, potential violations, and relevant citations."
        elif analysis_type == "summary":
            return base_prompt + "Provide a concise summary of the main points and key information."
        elif analysis_type == "entities":
            return base_prompt + "Extract key entities including people, organizations, dates, and legal references."
        else:
            return base_prompt + "Provide a general analysis of the content, highlighting important information and insights."


# CLI interface for standalone usage
def main():
    """CLI interface for clean document processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Document Processor (No GUI)")
    parser.add_argument("file_path", help="Path to document to process")
    parser.add_argument("--api-key", help="xAI API key")
    parser.add_argument("--model", default="grok-3-mini", help="Model to use")
    parser.add_argument("--analyze", action="store_true", help="Perform LLM analysis")
    parser.add_argument("--analysis-type", default="general", help="Type of analysis")
    
    args = parser.parse_args()
    
    processor = CleanDocumentProcessor(api_key=args.api_key, model=args.model)
    
    # Process file
    result = processor.process_file(args.file_path)
    
    print(f"Processing {'succeeded' if result.success else 'failed'}")
    print(f"Content length: {len(result.content)}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    # Perform analysis if requested
    if args.analyze and result.success:
        analysis_result = processor.analyze_with_llm(result.content, args.analysis_type)
        if analysis_result['success']:
            print("\nAnalysis:")
            print(analysis_result['analysis'])
        else:
            print(f"Analysis failed: {analysis_result['error']}")


if __name__ == "__main__":
    main()