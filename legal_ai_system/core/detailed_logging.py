# legal_ai_system/core/detailed_logging.py
"""
DETAILED Logging Infrastructure for Legal AI System
==================================================
Comprehensive logging system with detailed tracking of every operation,
function call, decision point, and system state change.
"""

import functools
import json
import logging
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Standard attributes found on ``logging.LogRecord`` instances.
# Used to filter them out when capturing custom ``extra`` fields.
_STANDARD_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}

# Define LOGS_DIR relative to this file's location (core/) then up to legal_ai_system/logs
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class LogLevel(Enum):
    """Enhanced log levels for detailed tracking"""

    TRACE = 5  # Most detailed - every operation
    DEBUG = 10  # Function calls and parameters
    INFO = 20  # Normal operation flow
    WARNING = 30  # Recoverable issues
    ERROR = 40  # Error conditions
    CRITICAL = 50  # System failure


class LogCategory(Enum):
    """Log categories for filtering and analysis"""

    SYSTEM = "SYSTEM"
    GUI = "GUI"
    AGENT = "AGENT"
    WORKFLOW = "WORKFLOW"
    DOCUMENT = "DOCUMENT"
    DOCUMENT_PROCESSING = "DOCUMENT_PROCESSING"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    VECTOR_STORE = "VECTOR_STORE"
    VECTOR_STORE_EMBEDDING = "VECTOR_STORE_EMBEDDING"
    VECTOR_STORE_DB = "VECTOR_STORE_DB"
    VECTOR_STORE_SEARCH = "VECTOR_STORE_SEARCH"
    LLM = "LLM"
    DATABASE = "DATABASE"
    FILE_IO = "FILE_IO"
    VALIDATION = "VALIDATION"
    ERROR_HANDLING = "ERROR_HANDLING"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    API = "API"
    CONFIG = "CONFIG"  # Added for ConfigurationManager


@dataclass
class DetailedLogEntry:
    """Comprehensive log entry with all context"""

    timestamp: str
    level: str
    category: str
    component: str
    function: str
    message: str
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    execution_time: Optional[float] = None
    thread_id: Optional[int] = None
    call_stack: Optional[List[str]] = None
    system_state: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class DetailedLogger:
    """Enhanced logger with comprehensive tracking capabilities"""

    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(name)
        self.entries: List[DetailedLogEntry] = (
            []
        )  # In-memory store, consider if this should be optional or managed
        self._lock = threading.RLock()

        # Add TRACE level if not already added by another instance
        if logging.getLevelName("TRACE") == "Level 5":  # Check if already added
            logging.addLevelName(LogLevel.TRACE.value, "TRACE")

            # Custom log method for TRACE level
            def trace_method(self_logger, message, *args, **kwargs):
                if self_logger.isEnabledFor(LogLevel.TRACE.value):
                    self_logger._log(LogLevel.TRACE.value, message, args, **kwargs)

            logging.Logger.trace = trace_method  # type: ignore

        # Configure logger
        if not self.logger.handlers:  # Configure only if no handlers are present
            self._configure_logger()

    def getChild(self, suffix: str) -> "DetailedLogger":
        """Return a child logger with the same category."""
        child_name = f"{self.name}.{suffix}"
        return get_detailed_logger(child_name, self.category)

    def _configure_logger(self):
        """Configure the underlying logger with multiple handlers"""
        self.logger.setLevel(LogLevel.TRACE.value)  # Set level for the logger instance

        # Console handler with color coding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            logging.INFO
        )  # Console shows INFO and above by default
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # File handler for all logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = (
            f"detailed_{self.name.lower().replace('.', '_')}_{timestamp}.log"
        )
        log_file = LOGS_DIR / log_file_name
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(
            LogLevel.TRACE.value
        )  # File logs everything from TRACE up
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # JSON handler for structured logs
        json_file_name = f"structured_{self.name.lower().replace('.', '_')}_{timestamp}.jsonl"  # Use .jsonl for line-delimited JSON
        json_file = LOGS_DIR / json_file_name
        json_handler = JSONHandler(json_file)
        json_handler.setLevel(
            LogLevel.TRACE.value
        )  # JSON logs everything from TRACE up

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)

    def _create_log_entry(
        self, level: LogLevel, message: str, **kwargs
    ) -> DetailedLogEntry:
        """Create a detailed log entry with full context"""
        # Get caller information
        frame = None
        try:
            frame = sys._getframe(
                3
            )  # Skip internal calls (_create_log_entry, trace/debug/etc., public method)
            function_name = frame.f_code.co_name
            filename = Path(frame.f_code.co_filename).name  # Get just the filename
        except Exception:  # Fallback if frame inspection fails
            function_name = "unknown_function"
            filename = "unknown_file"

        # Build call stack
        call_stack = []
        current_frame = frame
        if current_frame:  # Only build stack if frame is available
            for _ in range(5):  # Limit stack depth
                if current_frame:
                    call_stack.append(
                        f"{Path(current_frame.f_code.co_filename).name}:{current_frame.f_code.co_name}:{current_frame.f_lineno}"
                    )
                    current_frame = current_frame.f_back
                else:
                    break

        # Prepare parameters, handling potential un-JSON-serializable items by converting to string
        parameters_to_log = {}
        if "parameters" in kwargs and kwargs["parameters"] is not None:
            for k, v in kwargs["parameters"].items():
                try:
                    json.dumps(v)  # Test serializability
                    parameters_to_log[k] = v
                except TypeError:
                    parameters_to_log[k] = str(v)

        entry_kwargs = {k: v for k, v in kwargs.items() if k != "parameters"}
        if parameters_to_log:
            entry_kwargs["parameters"] = parameters_to_log

        entry = DetailedLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.name,
            category=self.category.value,
            component=f"{filename}:{self.name}",  # Include filename for clarity
            function=function_name,
            message=message,
            thread_id=threading.get_ident(),
            call_stack=call_stack,
            **entry_kwargs,  # Use filtered kwargs
        )

        # Consider if self.entries is truly needed or if file/JSON logs are sufficient.
        # For long-running apps, this list can grow very large.
        # with self._lock:
        #     self.entries.append(entry)

        return entry

    def _build_extra(
        self, entry: DetailedLogEntry, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Construct extra dictionary for LogRecord."""
        base_extra = {
            "category_val": entry.category,
            "parameters_val": entry.parameters,
            "result_val": entry.result,
            "execution_time_val": entry.execution_time,
            "error_details_val": entry.error_details,
        }
        if extra:
            base_extra.update(extra)
        return base_extra

    def trace(
        self,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Most detailed logging - every operation"""
        kwargs["parameters"] = parameters
        entry = self._create_log_entry(LogLevel.TRACE, message, **kwargs)
        self.logger.log(
            LogLevel.TRACE.value,
            f"[{self.category.value}] {message} {json.dumps(entry.parameters) if entry.parameters else ''}",
            extra=self._build_extra(entry, extra),
        )
        return entry

    def debug(
        self,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Debug level - function calls and parameters"""
        kwargs["parameters"] = parameters
        entry = self._create_log_entry(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(
            f"[{self.category.value}] {message} {json.dumps(entry.parameters) if entry.parameters else ''}",
            extra=self._build_extra(entry, extra),
        )
        return entry

    def info(
        self,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Info level - normal operation flow"""
        kwargs["parameters"] = parameters
        entry = self._create_log_entry(LogLevel.INFO, message, **kwargs)
        self.logger.info(
            f"[{self.category.value}] {message} {json.dumps(entry.parameters) if entry.parameters else ''}",
            extra=self._build_extra(entry, extra),
        )
        return entry

    def warning(
        self,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Warning level - recoverable issues"""
        kwargs["parameters"] = parameters
        if exception:
            kwargs["error_details"] = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
            }
        entry = self._create_log_entry(LogLevel.WARNING, message, **kwargs)
        self.logger.warning(
            f"[{self.category.value}] {message} {json.dumps(entry.parameters) if entry.parameters else ''}",
            exc_info=exception is not None,
            extra=self._build_extra(entry, extra),
        )
        return entry

    def error(
        self,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Error level - error conditions"""
        kwargs["parameters"] = parameters
        error_details_payload = {}
        if exception:
            error_details_payload = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc(),
            }
        kwargs["error_details"] = error_details_payload

        entry = self._create_log_entry(LogLevel.ERROR, message, **kwargs)
        self.logger.error(
            f"[{self.category.value}] {message} {json.dumps(entry.parameters) if entry.parameters else ''}",
            exc_info=exception is not None,
            extra=self._build_extra(entry, extra),
        )
        return entry

    def critical(
        self,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        *,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Critical level - system failure"""
        kwargs["parameters"] = parameters
        error_details_payload = {}
        if exception:
            error_details_payload = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc(),
            }
        kwargs["error_details"] = error_details_payload

        entry = self._create_log_entry(LogLevel.CRITICAL, message, **kwargs)
        self.logger.critical(
            f"[{self.category.value}] {message} {json.dumps(entry.parameters) if entry.parameters else ''}",
            exc_info=exception is not None,
            extra=self._build_extra(entry, extra),
        )
        return entry

    def function_call(
        self, func_name: str, parameters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> DetailedLogEntry:
        """Log function entry with parameters"""
        # param_str = json.dumps(parameters, default=str) if parameters else "None" # Already handled by self.trace
        message = f"FUNCTION_ENTRY: {func_name}()"
        return self.trace(message, parameters=parameters, **kwargs)

    def function_result(
        self,
        func_name: str,
        result: Any = None,
        execution_time: Optional[float] = None,
        **kwargs,
    ) -> DetailedLogEntry:
        """Log function exit with result and timing"""
        # result_str = json.dumps(result, default=str) if result is not None else "None" # Handled by self.trace
        time_str = f" [took {execution_time:.4f}s]" if execution_time else ""
        message = f"FUNCTION_EXIT: {func_name}(){time_str}"
        return self.trace(
            message, result=result, execution_time=execution_time, **kwargs
        )

    def state_change(
        self, component: str, old_state: Any, new_state: Any, reason: str = "", **kwargs
    ):
        """Log system state changes"""
        message = f"STATE_CHANGE: {component} changed state."
        # if reason: # Redundant with parameters
        #     message += f" - Reason: {reason}"
        return self.info(
            message,
            parameters={
                "component": component,
                "old_state": str(old_state),
                "new_state": str(new_state),
                "reason": reason,
            },  # Ensure states are strings
            **kwargs,
        )

    def decision_point(
        self, decision_name: str, factors: Dict[str, Any], outcome: str, **kwargs
    ):  # Renamed decision to decision_name, result to outcome
        """Log decision points with reasoning"""
        message = f"DECISION: {decision_name} -> {outcome}"
        # factors_str = json.dumps(factors, default=str) # Handled by self.info
        return self.info(
            message,
            parameters={
                "decision_name": decision_name,
                "factors": factors,
                "outcome": outcome,
            },
            **kwargs,
        )

    def performance_metric(
        self,
        operation: str,
        duration: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DetailedLogEntry:
        """Log performance measurements"""
        message = f"PERFORMANCE: {operation} took {duration:.4f}s"
        # if additional_metrics: # Handled by self.debug
        #     metrics_str = json.dumps(additional_metrics, default=str)
        #     message += f" (metrics: {metrics_str})"
        return self.debug(
            message, execution_time=duration, parameters=additional_metrics, **kwargs
        )

    def export_logs(self, filepath: Optional[Path] = None) -> Path:
        """Export all log entries to JSON file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = (
                LOGS_DIR
                / f"exported_logs_{self.name.lower().replace('.', '_')}_{timestamp}.jsonl"
            )

        # This method implies self.entries is maintained, which might be memory intensive.
        # If logs are primarily written to files, this method would need to read from those files.
        # For now, assuming self.entries is used for short-term in-memory logging if desired.
        # If self.entries is disabled, this method should be adapted or removed.

        # Placeholder if self.entries is not used:
        if not hasattr(self, "entries") or not self.entries:
            self.warning(
                "In-memory log entry list is not maintained or is empty. Cannot export from memory.",
                parameters={"filepath": str(filepath)},
            )
            # Create an empty export or signal that logs are in files.
            Path(filepath).touch()  # Create an empty file.
            return filepath

        with self._lock:
            log_data = {
                "logger_name": self.name,
                "category": self.category.value,
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(self.entries),
                "entries": [entry.to_dict() for entry in self.entries],
            }

        with open(filepath, "w") as f:
            for entry in log_data["entries"]:  # Write line-delimited JSON
                json.dump(entry, f, default=str)
                f.write("\n")

        self.info(f"Exported {len(self.entries)} log entries to {filepath}")
        return filepath


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""

    COLORS = {
        "TRACE": "\033[90m",  # Dark gray
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        # Ensurelevelname is a string before modification
        levelname_str = str(record.levelname)
        record.levelname = f"{log_color}{levelname_str}{self.COLORS['RESET']}"
        return super().format(record)


class JSONHandler(logging.Handler):
    """Custom handler for structured JSON logging.

    Any values passed via ``extra`` on the logging call are automatically
    appended to the JSON output. This keeps ``DetailedLogger`` flexible without
    requiring changes to every handler when new context fields are needed.
    """

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        self._lock = threading.RLock()  # Use RLock for reentrant lock

    def emit(self, record: logging.LogRecord):  # Added type hint
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "message": record.getMessage(),  # Use getMessage() to handle args
                "thread_id": record.thread,
                "process_id": record.process,
            }

            # Include any custom attributes added via ``extra``
            for key, value in record.__dict__.items():
                if key not in _STANDARD_LOG_RECORD_ATTRS and key not in log_entry:
                    log_entry[key] = value

            with self._lock:
                with open(
                    self.filepath, "a", encoding="utf-8"
                ) as f:  # Specify encoding
                    f.write(
                        json.dumps(log_entry, default=str) + "\n"
                    )  # Use default=str for non-serializable
        except Exception:
            self.handleError(record)


def detailed_log_function(category: LogCategory = LogCategory.SYSTEM):
    """Decorator for automatic detailed function logging"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create logger for this function's module
            logger_name = func.__module__
            # Check if the function is a method of a class
            if args and hasattr(args[0], "__class__"):
                class_name = args[0].__class__.__name__
                logger_name = f"{func.__module__}.{class_name}"

            logger = get_detailed_logger(logger_name, category)

            # Log function entry
            parameters_to_log = {}
            try:
                # Log positional arguments
                arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                parameters_to_log.update(
                    {name: str(val) for name, val in zip(arg_names, args)}
                )
                # Log keyword arguments
                parameters_to_log.update({k: str(v) for k, v in kwargs.items()})
            except Exception:  # Fallback if inspection fails
                parameters_to_log = {
                    "args": [str(a) for a in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }

            logger.function_call(func.__name__, parameters=parameters_to_log)

            start_time = time.perf_counter()  # Use perf_counter for more precision
            try:
                # Execute function
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time

                # Log successful completion
                logger.function_result(
                    func.__name__,
                    result=str(result)[:500],
                    execution_time=execution_time,
                )  # Truncate long results
                return result

            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {execution_time:.4f}s",
                    exception=e,
                    parameters={
                        "function_name": func.__name__,
                        "execution_time": execution_time,
                    },
                )
                raise

        return wrapper

    return decorator


# Global logger registry
_loggers: Dict[str, DetailedLogger] = {}
_registry_lock = (
    threading.RLock()
)  # Renamed from _lock to avoid conflict with DetailedLogger._lock


def get_detailed_logger(
    name: str, category: LogCategory = LogCategory.SYSTEM
) -> DetailedLogger:
    """Get or create a detailed logger instance"""
    with _registry_lock:
        if name not in _loggers:
            _loggers[name] = DetailedLogger(name, category)
        # Ensure category is updated if logger exists but category is different
        elif _loggers[name].category != category:
            _loggers[name].category = category
        return _loggers[name]


def export_all_logs() -> Path:
    """Export all logger data to a comprehensive report.
    This method is illustrative. In a real system, logs are continuously written to files.
    This might be used for a snapshot or if in-memory logging (self.entries) was enabled.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = LOGS_DIR / f"comprehensive_log_export_{timestamp}.jsonl"

    # This function implies reading from the `self.entries` list of each logger.
    # If that list is not being populated (to save memory), this function needs
    # to be re-thought. It might involve consolidating the .jsonl files.
    # For now, assuming `self.entries` might hold some recent logs or is used in specific contexts.

    all_entries_data = []
    with _registry_lock:
        for logger_instance in _loggers.values():
            with logger_instance._lock:  # Accessing internal lock of DetailedLogger
                # This assumes DetailedLogger instances store entries in self.entries
                # If not, this part will be empty.
                all_entries_data.extend(
                    [entry.to_dict() for entry in logger_instance.entries]
                )

    if not all_entries_data:
        get_detailed_logger("LogExporter").warning(
            "No in-memory log entries found to export. Log files are in the logs/ directory.",
            parameters={"export_file": str(export_file)},
        )
        export_file.touch()  # Create an empty file to signify attempt
        return export_file

    # Sort all entries by timestamp
    all_entries_data.sort(key=lambda x: x["timestamp"])

    with open(export_file, "w", encoding="utf-8") as f:
        for entry_dict in all_entries_data:
            json.dump(entry_dict, f, default=str)
            f.write("\n")

    get_detailed_logger("LogExporter").info(
        f"Exported {len(all_entries_data)} log entries to {export_file}",
        parameters={"total_loggers": len(_loggers)},
    )

    return export_file
