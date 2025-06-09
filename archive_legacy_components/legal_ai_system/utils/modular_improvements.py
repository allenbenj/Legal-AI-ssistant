"""
Modular Improvements for Document Processing System

These improvements can be implemented incrementally while maintaining compatibility.
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

# === 1. Dependency Management ===

class DependencyManager:
    """Centralized dependency management for optional libraries"""
    
    _dependencies = {}
    _checked = False
    
    @classmethod
    def check_dependencies(cls):
        """Check all optional dependencies once"""
        if cls._checked:
            return cls._dependencies
        
        dependencies = {
            'pymupdf': {'import': 'fitz', 'package': 'PyMuPDF', 'formats': ['pdf']},
            'docx': {'import': 'docx', 'package': 'python-docx', 'formats': ['docx', 'doc']},
            'pytesseract': {'import': 'pytesseract', 'package': 'pytesseract', 'formats': ['image', 'ocr']},
            'PIL': {'import': 'PIL', 'package': 'Pillow', 'formats': ['image']},
            'pandas': {'import': 'pandas', 'package': 'pandas', 'formats': ['xlsx', 'xls', 'csv']},
            'openpyxl': {'import': 'openpyxl', 'package': 'openpyxl', 'formats': ['xlsx']},
            'pptx': {'import': 'pptx', 'package': 'python-pptx', 'formats': ['pptx', 'ppt']},
            'markdown': {'import': 'markdown', 'package': 'markdown', 'formats': ['md']},
            'bs4': {'import': 'bs4', 'package': 'beautifulsoup4', 'formats': ['html', 'md']}
        }
        
        for name, info in dependencies.items():
            try:
                __import__(info['import'])
                cls._dependencies[name] = True
            except ImportError:
                cls._dependencies[name] = False
                logging.warning(f"{info['package']} not available - {info['formats']} support disabled")
        
        cls._checked = True
        return cls._dependencies
    
    @classmethod
    def is_available(cls, dependency: str) -> bool:
        """Check if a specific dependency is available"""
        if not cls._checked:
            cls.check_dependencies()
        return cls._dependencies.get(dependency, False)
    
    @classmethod
    def get_handler_for_format(cls, file_format: str) -> Optional[str]:
        """Get available handler for a file format"""
        format_handlers = {
            'pdf': 'pymupdf',
            'docx': 'docx',
            'doc': 'docx',
            'xlsx': 'pandas',
            'xls': 'pandas',
            'csv': 'pandas',
            'pptx': 'pptx',
            'ppt': 'pptx',
            'md': 'markdown',
            'html': 'bs4'
        }
        
        handler = format_handlers.get(file_format.lower())
        if handler and cls.is_available(handler):
            return handler
        return None

# === 2. File Handler Registry ===

class FileHandlerRegistry:
    """Registry pattern for file handlers with dynamic loading"""
    
    def __init__(self):
        self._handlers = {}
        self._strategies = {}
        self.dep_manager = DependencyManager()
    
    def register_handler(self, extensions: List[str], handler_class: type, 
                        strategy: str, dependencies: List[str] = None):
        """Register a file handler"""
        # Check dependencies
        if dependencies:
            for dep in dependencies:
                if not self.dep_manager.is_available(dep):
                    logging.warning(f"Handler for {extensions} not registered - missing {dep}")
                    return
        
        for ext in extensions:
            self._handlers[ext] = handler_class
            self._strategies[ext] = strategy
    
    def get_handler(self, file_path: Path) -> Optional[object]:
        """Get handler instance for file"""
        ext = file_path.suffix.lower()
        handler_class = self._handlers.get(ext)
        
        if handler_class:
            return handler_class()
        
        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            # Handle by MIME type
            pass
        
        return None
    
    def get_strategy(self, file_path: Path) -> str:
        """Get processing strategy for file"""
        ext = file_path.suffix.lower()
        return self._strategies.get(ext, ProcessingStrategy.REFERENCE_ONLY)

# === 3. Async Task Queue ===

class ProcessingQueue:
    """Async task queue for managing processing workload"""
    
    def __init__(self, max_workers: int = 5):
        self.queue = asyncio.Queue()
        self.workers = []
        self.max_workers = max_workers
        self.results = {}
        self._running = False
    
    async def start(self):
        """Start worker tasks"""
        self._running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
    
    async def stop(self):
        """Stop all workers"""
        self._running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    async def _worker(self, name: str):
        """Worker coroutine"""
        while self._running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                result = await self._process_task(task)
                self.results[task['id']] = result
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Worker {name} error: {e}")
    
    async def _process_task(self, task: Dict[str, Any]) -> Any:
        """Process a single task"""
        processor = task['processor']
        data = task['data']
        options = task.get('options', {})
        
        return await processor.process(data, options)
    
    async def add_task(self, task_id: str, processor: Any, data: Any, 
                      options: Dict[str, Any] = None) -> str:
        """Add task to queue"""
        task = {
            'id': task_id,
            'processor': processor,
            'data': data,
            'options': options or {}
        }
        
        await self.queue.put(task)
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result for task ID"""
        start_time = asyncio.get_event_loop().time()
        
        while task_id not in self.results:
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
        
        return self.results.pop(task_id)

# === 4. Caching Layer ===

class ProcessingCache:
    """Cache for processed documents to avoid reprocessing"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}  # In-memory cache for current session
        self.max_memory_items = 100
    
    def _get_cache_key(self, file_path: Path, processing_type: str) -> str:
        """Generate cache key from file path and processing type"""
        file_stat = file_path.stat()
        content = f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}:{processing_type}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get(self, file_path: Path, processing_type: str) -> Optional[ProcessingResult]:
        """Get cached result if available"""
        cache_key = self._get_cache_key(file_path, processing_type)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    result = ProcessingResult(**data)
                    
                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, result)
                    
                    return result
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
        
        return None
    
    async def set(self, file_path: Path, processing_type: str, result: ProcessingResult):
        """Cache processing result"""
        cache_key = self._get_cache_key(file_path, processing_type)
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, result)
        
        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result.__dict__, f, default=str)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def _add_to_memory_cache(self, key: str, result: ProcessingResult):
        """Add to memory cache with LRU eviction"""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = result

# === 5. Enhanced Error Handling ===

class ProcessingError(Exception):
    """Base exception for processing errors"""
    def __init__(self, message: str, error_type: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}

class RetryableError(ProcessingError):
    """Error that can be retried"""
    def __init__(self, message: str, retry_after: float = 1.0, **kwargs):
        super().__init__(message, "retryable", kwargs)
        self.retry_after = retry_after

class ErrorHandler:
    """Centralized error handling with retry logic"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except RetryableError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = e.retry_after * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    continue
            except Exception as e:
                # Non-retryable error
                raise ProcessingError(
                    f"Processing failed: {str(e)}",
                    "non_retryable",
                    {"original_error": str(e), "type": type(e).__name__}
                )
        
        raise last_error

# === 6. Metrics and Monitoring ===

class ProcessingMetrics:
    """Collect and report processing metrics"""
    
    def __init__(self):
        self.metrics = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "errors_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "by_format": {},
            "by_strategy": {}
        }
    
    def record_processing(self, file_format: str, strategy: str, 
                         processing_time: float, success: bool):
        """Record processing metrics"""
        self.metrics["documents_processed"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        if not success:
            self.metrics["errors_count"] += 1
        
        # By format
        if file_format not in self.metrics["by_format"]:
            self.metrics["by_format"][file_format] = {"count": 0, "time": 0.0}
        self.metrics["by_format"][file_format]["count"] += 1
        self.metrics["by_format"][file_format]["time"] += processing_time
        
        # By strategy
        if strategy not in self.metrics["by_strategy"]:
            self.metrics["by_strategy"][strategy] = {"count": 0, "time": 0.0}
        self.metrics["by_strategy"][strategy]["count"] += 1
        self.metrics["by_strategy"][strategy]["time"] += processing_time
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics["cache_misses"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        total_cached = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_rate = self.metrics["cache_hits"] / total_cached if total_cached > 0 else 0
        
        return {
            **self.metrics,
            "avg_processing_time": (
                self.metrics["total_processing_time"] / self.metrics["documents_processed"]
                if self.metrics["documents_processed"] > 0 else 0
            ),
            "success_rate": (
                (self.metrics["documents_processed"] - self.metrics["errors_count"]) / 
                self.metrics["documents_processed"]
                if self.metrics["documents_processed"] > 0 else 0
            ),
            "cache_hit_rate": cache_rate
        }

# === 7. Configuration Management ===

class ProcessingConfig:
    """Centralized configuration management"""
    
    DEFAULT_CONFIG = {
        "file_processing": {
            "max_file_size_mb": 100,
            "chunk_size": 3000,
            "chunk_overlap": 200,
            "timeout": 600,
            "max_retries": 3,
            "ocr_enabled": True,
            "ocr_language": "eng",
            "extract_tables": True,
            "preserve_structure": True
        },
        "entity_extraction": {
            "min_confidence": 0.7,
            "max_entities_per_chunk": 50,
            "chunk_size": 3000,
            "temperature": 0.1,
            "validation_enabled": True
        },
        "pipeline": {
            "max_workers": 5,
            "cache_enabled": True,
            "cache_dir": "./cache",
            "metrics_enabled": True
        }
    }
    
    def __init__(self, config_file: Path = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: Path):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self._deep_merge(self.config, user_config)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
    
    def _deep_merge(self, base: dict, update: dict):
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path (e.g., 'file_processing.chunk_size')"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
