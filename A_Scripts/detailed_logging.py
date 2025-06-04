"""
DETAILED Logging Infrastructure for Legal AI System
==================================================
Comprehensive logging system with detailed tracking of every operation,
function call, decision point, and system state change.
"""

import logging
import sys
import os
import json
import time
import traceback
import functools
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class LogLevel(Enum):
    """Enhanced log levels for detailed tracking"""
    TRACE = 5      # Most detailed - every operation
    DEBUG = 10     # Function calls and parameters
    INFO = 20      # Normal operation flow
    WARNING = 30   # Recoverable issues
    ERROR = 40     # Error conditions
    CRITICAL = 50  # System failure

class LogCategory(Enum):
    """Log categories for filtering and analysis"""
    SYSTEM = "SYSTEM"
    GUI = "GUI"
    AGENT = "AGENT"
    WORKFLOW = "WORKFLOW"
    DOCUMENT = "DOCUMENT"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    VECTOR_STORE = "VECTOR_STORE"
    LLM = "LLM"
    DATABASE = "DATABASE"
    FILE_IO = "FILE_IO"
    VALIDATION = "VALIDATION"
    ERROR_HANDLING = "ERROR_HANDLING"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    API = "API"

@dataclass
class DetailedLogEntry:
    """Comprehensive log entry with all context"""
    timestamp: str
    level: str
    category: str
    component: str
    function: str
    message: str
    parameters: Dict[str, Any] = None
    result: Any = None
    execution_time: float = None
    thread_id: int = None
    call_stack: List[str] = None
    system_state: Dict[str, Any] = None
    error_details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class DetailedLogger:
    """Enhanced logger with comprehensive tracking capabilities"""
    
    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(name)
        self.entries: List[DetailedLogEntry] = []
        self._lock = threading.Lock()
        
        # Add TRACE level
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        
        # Configure logger
        self._configure_logger()
    
    def _configure_logger(self):
        """Configure the underlying logger with multiple handlers"""
        self.logger.setLevel(LogLevel.TRACE.value)
        
        # Console handler with color coding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for all logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"detailed_{self.name.lower()}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LogLevel.TRACE.value)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # JSON handler for structured logs
        json_file = LOGS_DIR / f"structured_{self.name.lower()}_{timestamp}.json"
        json_handler = JSONHandler(json_file)
        json_handler.setLevel(LogLevel.TRACE.value)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(json_handler)
    
    def _create_log_entry(self, level: LogLevel, message: str, **kwargs) -> DetailedLogEntry:
        """Create a detailed log entry with full context"""
        # Get caller information
        frame = sys._getframe(3)  # Skip internal calls
        function_name = frame.f_code.co_name
        
        # Build call stack
        call_stack = []
        current_frame = frame
        for _ in range(5):  # Limit stack depth
            if current_frame:
                call_stack.append(f"{current_frame.f_code.co_filename}:{current_frame.f_code.co_name}:{current_frame.f_lineno}")
                current_frame = current_frame.f_back
            else:
                break
        
        entry = DetailedLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.name,
            category=self.category.value,
            component=self.name,
            function=function_name,
            message=message,
            thread_id=threading.get_ident(),
            call_stack=call_stack,
            **kwargs
        )
        
        with self._lock:
            self.entries.append(entry)
        
        return entry
    
    def trace(self, message: str, **kwargs):
        """Most detailed logging - every operation"""
        entry = self._create_log_entry(LogLevel.TRACE, message, **kwargs)
        self.logger.log(LogLevel.TRACE.value, f"[{self.category.value}] {message}")
        return entry
    
    def debug(self, message: str, **kwargs):
        """Debug level - function calls and parameters"""
        entry = self._create_log_entry(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(f"[{self.category.value}] {message}")
        return entry
    
    def info(self, message: str, **kwargs):
        """Info level - normal operation flow"""
        entry = self._create_log_entry(LogLevel.INFO, message, **kwargs)
        self.logger.info(f"[{self.category.value}] {message}")
        return entry
    
    def warning(self, message: str, **kwargs):
        """Warning level - recoverable issues"""
        entry = self._create_log_entry(LogLevel.WARNING, message, **kwargs)
        self.logger.warning(f"[{self.category.value}] {message}")
        return entry
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Error level - error conditions"""
        error_details = {}
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        entry = self._create_log_entry(LogLevel.ERROR, message, error_details=error_details, **kwargs)
        self.logger.error(f"[{self.category.value}] {message}")
        return entry
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """Critical level - system failure"""
        error_details = {}
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        entry = self._create_log_entry(LogLevel.CRITICAL, message, error_details=error_details, **kwargs)
        self.logger.critical(f"[{self.category.value}] {message}")
        return entry
    
    def function_call(self, func_name: str, parameters: Dict[str, Any] = None, **kwargs):
        """Log function entry with parameters"""
        param_str = json.dumps(parameters, default=str) if parameters else "None"
        message = f"FUNCTION_ENTRY: {func_name}() called with parameters: {param_str}"
        return self.trace(message, parameters=parameters, **kwargs)
    
    def function_result(self, func_name: str, result: Any = None, execution_time: float = None, **kwargs):
        """Log function exit with result and timing"""
        result_str = json.dumps(result, default=str) if result is not None else "None"
        time_str = f" [took {execution_time:.4f}s]" if execution_time else ""
        message = f"FUNCTION_EXIT: {func_name}() returned: {result_str}{time_str}"
        return self.trace(message, result=result, execution_time=execution_time, **kwargs)
    
    def state_change(self, component: str, old_state: Any, new_state: Any, reason: str = "", **kwargs):
        """Log system state changes"""
        message = f"STATE_CHANGE: {component} changed from {old_state} to {new_state}"
        if reason:
            message += f" - Reason: {reason}"
        return self.info(message, 
                        system_state={'old_state': old_state, 'new_state': new_state, 'reason': reason},
                        **kwargs)
    
    def decision_point(self, decision: str, factors: Dict[str, Any], result: str, **kwargs):
        """Log decision points with reasoning"""
        message = f"DECISION: {decision} -> {result}"
        factors_str = json.dumps(factors, default=str)
        message += f" (factors: {factors_str})"
        return self.info(message, parameters=factors, result=result, **kwargs)
    
    def performance_metric(self, operation: str, duration: float, additional_metrics: Dict[str, Any] = None, **kwargs):
        """Log performance measurements"""
        message = f"PERFORMANCE: {operation} took {duration:.4f}s"
        if additional_metrics:
            metrics_str = json.dumps(additional_metrics, default=str)
            message += f" (metrics: {metrics_str})"
        return self.debug(message, execution_time=duration, parameters=additional_metrics, **kwargs)
    
    def export_logs(self, filepath: Path = None) -> Path:
        """Export all log entries to JSON file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = LOGS_DIR / f"exported_logs_{self.name.lower()}_{timestamp}.json"
        
        with self._lock:
            log_data = {
                'logger_name': self.name,
                'category': self.category.value,
                'export_timestamp': datetime.now().isoformat(),
                'total_entries': len(self.entries),
                'entries': [entry.to_dict() for entry in self.entries]
            }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.info(f"Exported {len(self.entries)} log entries to {filepath}")
        return filepath

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'TRACE': '\033[90m',     # Dark gray
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class JSONHandler(logging.Handler):
    """Custom handler for structured JSON logging"""
    
    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        self._lock = threading.Lock()
    
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'thread_id': record.thread,
                'process_id': record.process
            }
            
            with self._lock:
                with open(self.filepath, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
        except Exception:
            self.handleError(record)

def detailed_log_function(category: LogCategory = LogCategory.SYSTEM):
    """Decorator for automatic detailed function logging"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create logger for this function
            logger = DetailedLogger(f"{func.__module__}.{func.__name__}", category)
            
            # Log function entry
            parameters = {
                'args': [str(arg) for arg in args],
                'kwargs': {k: str(v) for k, v in kwargs.items()}
            }
            logger.function_call(func.__name__, parameters)
            
            start_time = time.time()
            try:
                # Execute function
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful completion
                logger.function_result(func.__name__, result, execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.4f}s", exception=e)
                raise
        
        return wrapper
    return decorator

# Global logger registry
_loggers: Dict[str, DetailedLogger] = {}
_lock = threading.Lock()

def get_detailed_logger(name: str, category: LogCategory = LogCategory.SYSTEM) -> DetailedLogger:
    """Get or create a detailed logger instance"""
    with _lock:
        if name not in _loggers:
            _loggers[name] = DetailedLogger(name, category)
        return _loggers[name]

def export_all_logs() -> Path:
    """Export all logger data to a comprehensive report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = LOGS_DIR / f"comprehensive_log_export_{timestamp}.json"
    
    all_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_loggers': len(_loggers),
        'loggers': {}
    }
    
    with _lock:
        for name, logger in _loggers.items():
            with logger._lock:
                all_data['loggers'][name] = {
                    'name': logger.name,
                    'category': logger.category.value,
                    'total_entries': len(logger.entries),
                    'entries': [entry.to_dict() for entry in logger.entries]
                }
    
    with open(export_file, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    
    return export_file

# Example usage and testing
if __name__ == "__main__":
    # Test the detailed logging system
    logger = get_detailed_logger("TEST_COMPONENT", LogCategory.SYSTEM)
    
    logger.info("Testing detailed logging system")
    logger.trace("This is a trace message with detailed context")
    logger.debug("Debug message with parameters", parameters={'test': 'value'})
    
    # Test function logging
    @detailed_log_function(LogCategory.DOCUMENT)
    def test_function(param1: str, param2: int = 10):
        logger = get_detailed_logger("test_function", LogCategory.DOCUMENT)
        logger.info(f"Processing {param1} with value {param2}")
        return f"result_{param1}_{param2}"
    
    result = test_function("test", param2=20)
    
    # Test decision logging
    logger.decision_point(
        "File Processing Method",
        {'file_size': 1024, 'file_type': 'pdf', 'complexity': 'medium'},
        "Use advanced PDF processor"
    )
    
    # Test state change logging
    logger.state_change("DocumentProcessor", "idle", "processing", "User initiated processing")
    
    # Export logs
    export_file = logger.export_logs()
    print(f"Detailed logs exported to: {export_file}")