"""
Shared Components for Document Processing
Phase 1: Quick Win - Extract duplicated code from agents
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
import functools
from collections import deque

# === 1. DEPENDENCY MANAGEMENT ===

class DependencyManager:
    """Centralized dependency management for optional libraries"""
    
    _dependencies = {}
    _checked = False
    
    DEPENDENCY_MAP = {
        'pymupdf': {'import_name': 'fitz', 'package': 'PyMuPDF', 'formats': ['pdf']},
        'docx': {'import_name': 'docx', 'package': 'python-docx', 'formats': ['docx', 'doc']},
        'pytesseract': {'import_name': 'pytesseract', 'package': 'pytesseract', 'formats': ['image']},
        'PIL': {'import_name': 'PIL', 'package': 'Pillow', 'formats': ['jpg', 'png', 'gif']},
        'pandas': {'import_name': 'pandas', 'package': 'pandas', 'formats': ['xlsx', 'xls', 'csv']},
        'openpyxl': {'import_name': 'openpyxl', 'package': 'openpyxl', 'formats': ['xlsx']},
        'pptx': {'import_name': 'pptx', 'package': 'python-pptx', 'formats': ['pptx', 'ppt']},
        'markdown': {'import_name': 'markdown', 'package': 'markdown', 'formats': ['md']},
        'bs4': {'import_name': 'bs4', 'package': 'beautifulsoup4', 'formats': ['html']}
    }
    
    @classmethod
    def check_dependencies(cls) -> Dict[str, bool]:
        """Check all optional dependencies once at startup"""
        if cls._checked:
            return cls._dependencies
        
        logger = logging.getLogger(__name__)
        
        for name, info in cls.DEPENDENCY_MAP.items():
            try:
                __import__(info['import_name'])
                cls._dependencies[name] = True
                logger.debug(f"✓ {info['package']} available")
            except ImportError:
                cls._dependencies[name] = False
                logger.warning(f"✗ {info['package']} not available - {info['formats']} support disabled")
        
        cls._checked = True
        return cls._dependencies
    
    @classmethod
    def is_available(cls, dependency: str) -> bool:
        """Check if a specific dependency is available"""
        if not cls._checked:
            cls.check_dependencies()
        return cls._dependencies.get(dependency, False)
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of all supported file formats"""
        if not cls._checked:
            cls.check_dependencies()
        
        supported = ['txt', 'json']  # Always supported
        for dep_name, info in cls.DEPENDENCY_MAP.items():
            if cls._dependencies.get(dep_name, False):
                supported.extend(info['formats'])
        
        return sorted(list(set(supported)))


# === 2. SHARED CHUNKING LOGIC ===

@dataclass
class DocumentChunk:
    """Standardized document chunk"""
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    metadata: Dict[str, Any] = None


class DocumentChunker:
    """Shared chunking logic for both processors"""
    
    def __init__(self, chunk_size: int = 3000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_chunks(self, content: str) -> List[DocumentChunk]:
        """Create overlapping chunks with metadata"""
        if len(content) <= self.chunk_size:
            return [DocumentChunk(
                content=content,
                start_index=0,
                end_index=len(content),
                chunk_index=0,
                metadata={"is_single_chunk": True}
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Avoid cutting words in the middle
            if end < len(content):
                last_space = content.rfind(' ', start, end)
                if last_space > start + self.chunk_size - 100:
                    end = last_space
            
            chunk = DocumentChunk(
                content=content[start:end],
                start_index=start,
                end_index=end,
                chunk_index=chunk_index,
                metadata={"total_length": len(content)}
            )
            chunks.append(chunk)
            
            if end >= len(content):
                break
                
            start = end - self.overlap
            chunk_index += 1
        
        return chunks


# === 3. LEGAL DOCUMENT CLASSIFICATION ===

class LegalDocumentClassifier:
    """Shared legal document classification logic"""
    
    LEGAL_INDICATORS = {
        'motion': ['motion to', 'motion for', 'motion that', 'movant'],
        'complaint': ['complaint', 'plaintiff', 'defendant', 'cause of action'],
        'affidavit': ['affidavit', 'affiant', 'sworn statement', 'under oath'],
        'deposition': ['deposition', 'examination under oath', 'q:', 'a:', 'sworn testimony'],
        'court_order': ['order', 'it is ordered', 'court orders', 'hereby ordered'],
        'warrant': ['warrant', 'search warrant', 'arrest warrant', 'probable cause'],
        'brief': ['brief', 'argument', 'legal memorandum', 'statement of facts'],
        'contract': ['agreement', 'contract', 'party of the first part', 'whereas'],
        'statute': ['statute', 'section', 'subsection', 'chapter', 'code'],
        'constitution': ['constitution', 'amendment', 'article', 'bill of rights'],
        'evidence_log': ['evidence', 'exhibit', 'chain of custody', 'collected'],
        'witness_statement': ['witness', 'statement', 'testified', 'observed']
    }
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify legal document type with confidence scores"""
        if not text or len(text.strip()) < 10:
            return {"type": "unknown", "confidence": 0.0, "is_legal": False}
        
        text_lower = text.lower()
        scores = {}
        
        for doc_type, indicators in self.LEGAL_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        if not scores:
            return {"type": "unknown", "confidence": 0.0, "is_legal": False}
        
        # Find the best match
        primary_type = max(scores.items(), key=lambda x: x[1])
        total_indicators = sum(scores.values())
        
        return {
            "type": primary_type[0],
            "confidence": primary_type[1] / len(self.LEGAL_INDICATORS[primary_type[0]]),
            "scores": scores,
            "is_legal": total_indicators >= 2,  # At least 2 indicators
            "total_indicators": total_indicators
        }


# === 4. PERFORMANCE METRICS ===

class PerformanceMetrics:
    """Simple performance tracking"""
    
    def __init__(self):
        self.metrics = deque(maxlen=1000)  # Keep last 1000 measurements
    
    def record(self, operation: str, duration: float, success: bool, **metadata):
        """Record a performance measurement"""
        self.metrics.append({
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
            **metadata
        })
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        filtered = [m for m in self.metrics if not operation or m["operation"] == operation]
        
        if not filtered:
            return {"count": 0}
        
        durations = [m["duration"] for m in filtered]
        successes = [m for m in filtered if m["success"]]
        
        return {
            "count": len(filtered),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "success_rate": len(successes) / len(filtered),
            "total_time": sum(durations)
        }


# Global performance tracker
performance_tracker = PerformanceMetrics()


def measure_performance(operation_name: str = None):
    """Decorator to automatically measure function performance"""
    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            success = False
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                raise
            finally:
                duration = time.perf_counter() - start_time
                performance_tracker.record(op_name, duration, success)
                
                logger = logging.getLogger(func.__module__)
                status = "SUCCESS" if success else "FAILED"
                logger.debug(f"PERF: {op_name} {status} in {duration:.3f}s")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            success = False
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                raise
            finally:
                duration = time.perf_counter() - start_time
                performance_tracker.record(op_name, duration, success)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# === 5. BASIC CACHING ===

class ProcessingCache:
    """Simple file-hash based caching for immediate wins"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("./storage/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # In-memory cache for current session
        self.max_memory_items = 100
    
    def _get_cache_key(self, file_path: Path, processing_type: str) -> str:
        """Generate cache key from file path and processing type"""
        try:
            file_stat = file_path.stat()
            content = f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}:{processing_type}"
            return hashlib.sha256(content.encode()).hexdigest()
        except (OSError, IOError):
            # Fallback for files that can't be stat'd
            return hashlib.sha256(f"{file_path}:{processing_type}".encode()).hexdigest()
    
    def get(self, file_path: Path, processing_type: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        cache_key = self._get_cache_key(file_path, processing_type)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    
                # Add to memory cache
                self._add_to_memory_cache(cache_key, result)
                return result
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
        
        return None
    
    def set(self, file_path: Path, processing_type: str, result: Dict[str, Any]):
        """Cache processing result"""
        cache_key = self._get_cache_key(file_path, processing_type)
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, result)
        
        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, default=str, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def _add_to_memory_cache(self, key: str, result: Dict[str, Any]):
        """Add to memory cache with simple FIFO eviction"""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = result
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        disk_files = list(self.cache_dir.glob("*.json")) if self.cache_dir.exists() else []
        return {
            "memory_items": len(self.memory_cache),
            "disk_items": len(disk_files),
            "cache_dir": str(self.cache_dir)
        }


# Global instances
dependency_manager = DependencyManager()
document_chunker = DocumentChunker()
document_classifier = LegalDocumentClassifier()
processing_cache = ProcessingCache()

# Export everything
__all__ = [
    'DependencyManager', 'DocumentChunker', 'LegalDocumentClassifier', 
    'DocumentChunk', 'ProcessingCache', 'PerformanceMetrics',
    'measure_performance', 'performance_tracker',
    'dependency_manager', 'document_chunker', 'document_classifier', 'processing_cache'
]