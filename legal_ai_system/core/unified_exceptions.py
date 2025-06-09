# legal_ai_system/core/unified_exceptions.py
"""
Unified Exception Handling System with DETAILED Logging
======================================================
Comprehensive exception hierarchy with detailed logging, error recovery,
and forensic tracking for the Legal AI System.
"""

import sys
import traceback
import time
import json
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os # For process_id
import threading # For thread_id

# Import detailed logging system
try:
    from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function, DetailedLogger
except ImportError: # Fallback for direct execution or testing
    # Mock logger if detailed_logging is not available in this context
    class MockDetailedLogger:
        def __init__(self, name, category=None): self.name = name
        def info(self, *args, **kwargs): print(f"INFO: {args}")
        def error(self, *args, **kwargs): print(f"ERROR: {args}")
        def warning(self, *args, **kwargs): print(f"WARNING: {args}")
        def trace(self, *args, **kwargs): print(f"TRACE: {args}")
        def critical(self, *args, **kwargs): print(f"CRITICAL: {args}")

    def get_detailed_logger(name, category=None) -> MockDetailedLogger: # type: ignore
        return MockDetailedLogger(name, category)

    def detailed_log_function(category): # type: ignore
        def decorator(func):
            return func
        return decorator
    
    class LogCategory: # type: ignore
        ERROR_HANDLING = "ERROR_HANDLING"
        SECURITY = "SECURITY"
        SYSTEM = "SYSTEM" # Added for default

# Initialize specialized loggers for error handling
error_logger: DetailedLogger = get_detailed_logger("ErrorHandler", LogCategory.ERROR_HANDLING) # type: ignore
recovery_logger: DetailedLogger = get_detailed_logger("ErrorRecovery", LogCategory.ERROR_HANDLING) # type: ignore
forensics_logger: DetailedLogger = get_detailed_logger("ErrorForensics", LogCategory.ERROR_HANDLING) # type: ignore
security_error_logger: DetailedLogger = get_detailed_logger("SecurityErrors", LogCategory.SECURITY) # type: ignore


class ErrorSeverity(Enum):
    """Error severity levels with detailed classification"""
    TRACE = 1           # Minor issues, debugging information
    INFO = 2            # Informational errors, user feedback
    WARNING = 3         # Recoverable issues, degraded functionality
    ERROR = 4           # Significant errors, partial functionality loss
    CRITICAL = 5        # Critical errors, major functionality loss
    FATAL = 6           # System failure, complete shutdown required

class ErrorCategory(Enum):
    """Error categories for classification and handling"""
    SYSTEM = "system"                    # System-level errors
    CONFIGURATION = "configuration"      # Configuration and settings errors
    DOCUMENT = "document"               # Document processing errors
    VECTOR_STORE = "vector_store"       # Vector storage and search errors
    KNOWLEDGE_GRAPH = "knowledge_graph" # Knowledge graph errors
    AGENT = "agent"                     # AI agent execution errors
    LLM = "llm"                        # LLM provider and API errors
    GUI = "gui"                        # User interface errors
    DATABASE = "database"              # Database connection and query errors
    FILE_IO = "file_io"                # File system and I/O errors
    NETWORK = "network"                # Network and API communication errors
    VALIDATION = "validation"          # Data validation errors
    SECURITY = "security"              # Security and authentication errors
    PERFORMANCE = "performance"        # Performance and resource errors
    WORKFLOW = "workflow"              # Workflow orchestration errors

class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies with detailed implementation"""
    NONE = "none"                      # No recovery possible
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"              # Use alternative method
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality
    USER_INTERVENTION = "user_intervention"        # Require user action
    SYSTEM_RESTART = "system_restart"             # Restart component/system
    DATA_RECOVERY = "data_recovery"               # Attempt data recovery

@dataclass
class ErrorContext:
    """Comprehensive error context with forensic information"""
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = "unknown_component"
    function: str = "unknown_function"
    operation: str = "unknown_operation"
    parameters: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict) # Consider what system state is safe/useful to log
    call_stack: List[str] = field(default_factory=list)
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    memory_usage_mb: float = 0.0 # Corrected name
    cpu_usage_percent: float = 0.0 # Corrected name
    user_context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    request_id: Optional[str] = None

class LegalAIException(Exception):
    """
    Base exception class for Legal AI System with comprehensive logging
    and error context tracking.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        recovery_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.NONE,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None
    ):
        """Initialize exception with comprehensive error information"""
        super().__init__(message)
        
        self.message = message
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        # Error code generation needs category and severity before context is fully captured if context is None
        self.error_code = error_code or self._generate_error_code(category, severity)
        self.context = context or self._capture_context() # Capture context after other fields are set
        self.cause = cause
        self.user_message = user_message or self._generate_user_message()
        self.technical_details = technical_details or {}
        
        # Forensic information
        self.exception_id = f"{self.category.value}_{int(time.time())}_{id(self)}"
        self.stack_trace = traceback.format_exc() if cause else traceback.format_stack()[-3] # Get relevant part of stack
        self.creation_time = datetime.now()
        
        # Log the exception creation
        self._log_exception_creation()
    
    def _generate_error_code(self, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Generate standardized error code"""
        # Use provided category and severity as self.category/severity might not be set yet if context is None
        return f"{category.value.upper()}_{severity.name}_{int(time.time())}"

    def _capture_context(self) -> ErrorContext:
        """Capture comprehensive error context"""
        # Default values
        component_val = "unknown_component"
        function_val = "unknown_function"
        call_stack_val = []
        
        try:
            # Get current frame information
            # The frame depth needs to be carefully managed.
            # If _capture_context is called directly by __init__, frame(1) is __init__.
            # If called from a helper within __init__, it changes.
            # A robust way is to pass the frame or inspect the stack more carefully.
            # For simplicity, let's assume it's called from __init__.
            frame = sys._getframe(1) # Current frame is _capture_context
            if frame.f_back: # __init__ frame
                frame = frame.f_back
                if frame.f_back: # Caller of __init__
                    frame = frame.f_back 
                    component_val = Path(frame.f_code.co_filename).name
                    function_val = frame.f_code.co_name

            # Build call stack
            current_frame = frame
            for _ in range(10):  # Limit stack depth
                if current_frame:
                    call_stack_val.append(
                        f"{Path(current_frame.f_code.co_filename).name}:{current_frame.f_code.co_name}:{current_frame.f_lineno}"
                    )
                    current_frame = current_frame.f_back
                else:
                    break
        except Exception:
            # If frame inspection fails, stick to defaults
            pass

        # Get system information (optional, can be slow or permission-denied)
        memory_usage_val = 0.0
        cpu_usage_val = 0.0
        try:
            import psutil # Import locally to make it optional
            process = psutil.Process(os.getpid())
            memory_usage_val = process.memory_info().rss / (1024 * 1024)  # MB
            cpu_usage_val = process.cpu_percent(interval=None) # Non-blocking CPU usage
        except ImportError:
            error_logger.warning("psutil not installed, cannot get detailed memory/CPU usage for errors.")
        except Exception as e:
            error_logger.warning(f"Failed to get psutil info: {e}")


        return ErrorContext(
            component=component_val,
            function=function_val,
            call_stack=call_stack_val,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            memory_usage_mb=memory_usage_val, # Corrected name
            cpu_usage_percent=cpu_usage_val # Corrected name
        )
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message"""
        if self.severity in [ErrorSeverity.TRACE, ErrorSeverity.INFO]:
            return f"Information: {self.message}"
        elif self.severity == ErrorSeverity.WARNING:
            return f"Warning: {self.message}"
        elif self.severity == ErrorSeverity.ERROR:
            return f"An error occurred: {self.message}"
        elif self.severity == ErrorSeverity.CRITICAL:
            return f"A critical error occurred: {self.message}. Please contact support if the issue persists."
        else:  # FATAL
            return f"A fatal system error occurred: {self.message}. The application may need to be restarted."
    
    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def _log_exception_creation(self):
        """Log exception creation with comprehensive details"""
        log_params = {
            'exception_id': self.exception_id,
            'error_code': self.error_code,
            'severity': self.severity.name,
            'category': self.category.value,
            'recovery_strategy': self.recovery_strategy.value,
            'component': self.context.component,
            'function': self.context.function,
            'memory_usage_mb': self.context.memory_usage_mb,
            'cpu_usage_percent': self.context.cpu_usage_percent
        }
        # Use the global error_logger instance
        error_logger.error(f"Exception Created: {self.__class__.__name__} - {self.message}", 
                           parameters=log_params,
                           exception=self.cause if self.cause else self) # Log the cause if available
        
        # Log forensic information
        forensics_logger.info(f"Exception Forensics: {self.exception_id}",
                             parameters={
                                 'call_stack': self.context.call_stack,
                                 'system_state': self.context.system_state, # Be careful about logging sensitive state
                                 'technical_details': self.technical_details,
                                 'full_stack_trace': self.stack_trace # Log full trace here
                             })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            'exception_id': self.exception_id,
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.name,
            'category': self.category.value,
            'recovery_strategy': self.recovery_strategy.value,
            'error_code': self.error_code,
            'user_message': self.user_message,
            'creation_time': self.creation_time.isoformat(),
            'context': {
                'component': self.context.component,
                'function': self.context.function,
                'operation': self.context.operation,
                'parameters': {k: str(v)[:200] for k,v in self.context.parameters.items()}, # Truncate params
                'thread_id': self.context.thread_id,
                'process_id': self.context.process_id,
                'memory_usage_mb': self.context.memory_usage_mb,
                'cpu_usage_percent': self.context.cpu_usage_percent,
                'call_stack': self.context.call_stack
            },
            'technical_details': self.technical_details,
            'cause': str(self.cause) if self.cause else None,
            'stack_trace_summary': self.stack_trace.splitlines()[-3:] # Summary of stack trace
        }

# Specialized Exception Classes

class ConfigurationError(LegalAIException):
    """Configuration and settings related errors"""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs): # Optional config_key
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.USER_INTERVENTION)
        
        if config_key:
            kwargs.setdefault('technical_details', {}).update({'config_key': config_key})
        
        super().__init__(message, **kwargs)

class DocumentProcessingError(LegalAIException):
    """Document processing and analysis errors"""
    def __init__(self, message: str, document_id: Optional[str] = None, file_path: Optional[Union[str, Path]] = None, **kwargs): # Optional params
        kwargs.setdefault('category', ErrorCategory.DOCUMENT)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if document_id:
            technical_details['document_id'] = document_id
        if file_path:
            technical_details['file_path'] = str(file_path) # Ensure path is string
        
        super().__init__(message, **kwargs)

class VectorStoreError(LegalAIException):
    """Vector storage and similarity search errors"""
    def __init__(self, message: str, index_type: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VECTOR_STORE)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.FALLBACK)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if index_type:
            technical_details['index_type'] = index_type
        if operation:
            technical_details['operation'] = operation
        
        super().__init__(message, **kwargs)

class KnowledgeGraphError(LegalAIException):
    """Knowledge graph operations errors"""
    def __init__(self, message: str, graph_operation: Optional[str] = None, entity_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.KNOWLEDGE_GRAPH)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.GRACEFUL_DEGRADATION)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if graph_operation:
            technical_details['graph_operation'] = graph_operation
        if entity_id:
            technical_details['entity_id'] = entity_id
        
        super().__init__(message, **kwargs)

class AgentExecutionError(LegalAIException):
    """AI agent execution and workflow errors"""
    def __init__(self, message: str, agent_name: Optional[str] = None, task_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AGENT)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if agent_name:
            technical_details['agent_name'] = agent_name
        if task_id:
            technical_details['task_id'] = task_id
        
        super().__init__(message, **kwargs)

# Alias for backward compatibility

AgentError = AgentExecutionError

class AgentProcessingError(AgentExecutionError):
    """Errors occurring during agent processing steps."""

    def __init__(self, message: str, underlying_exception: Optional[Exception] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.AGENT)
        kwargs.setdefault("recovery_strategy", ErrorRecoveryStrategy.RETRY)
        if underlying_exception is not None:
            kwargs.setdefault("technical_details", {})["cause"] = str(underlying_exception)
        super().__init__(message, **kwargs)
        self.cause = underlying_exception

class SystemInitializationError(LegalAIException):
    """System initialization and startup errors"""
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.SYSTEM_RESTART)
        
        if component:
            kwargs.setdefault('technical_details', {})['component'] = component
        
        super().__init__(message, **kwargs)

class MemoryManagerError(LegalAIException):
    """Memory management and persistence errors"""
    def __init__(self, message: str, memory_type: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATABASE)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        
        if memory_type:
            kwargs.setdefault('technical_details', {})['memory_type'] = memory_type
        
        super().__init__(message, **kwargs)

class WorkflowExecutionError(LegalAIException):
    """Workflow orchestration and execution errors"""
    def __init__(self, message: str, workflow_name: Optional[str] = None, step: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.WORKFLOW)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if workflow_name:
            technical_details['workflow_name'] = workflow_name
        if step:
            technical_details['workflow_step'] = step
        
        super().__init__(message, **kwargs)

class LLMProviderError(LegalAIException):
    """LLM provider and API communication errors"""
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, api_response: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.LLM)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.FALLBACK)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if provider:
            technical_details['provider'] = provider
        if model:
            technical_details['model'] = model
        if api_response:
            technical_details['api_response'] = api_response[:500] # Truncate long API responses
        
        super().__init__(message, **kwargs)

class GUIError(LegalAIException):
    """User interface and interaction errors"""
    def __init__(self, message: str, component: Optional[str] = None, user_action: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.GUI)
        kwargs.setdefault('severity', ErrorSeverity.WARNING) # Usually not critical
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.GRACEFUL_DEGRADATION)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if component:
            technical_details['gui_component'] = component
        if user_action:
            technical_details['user_action'] = user_action
        
        super().__init__(message, **kwargs)

class DatabaseError(LegalAIException):
    """Database connection and operation errors"""
    def __init__(self, message: str, database_type: Optional[str] = None, query: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATABASE)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if database_type:
            technical_details['database_type'] = database_type
        if query:
            technical_details['query_summary'] = query[:200] # Log summary, not full query for security
        
        super().__init__(message, **kwargs)

class FileIOError(LegalAIException):
    """File system and I/O operation errors"""
    def __init__(self, message: str, file_path: Optional[Union[str,Path]] = None, operation: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.FILE_IO)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.RETRY)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if file_path:
            technical_details['file_path'] = str(file_path)
        if operation:
            technical_details['file_operation'] = operation
        
        super().__init__(message, **kwargs)

class SecurityError(LegalAIException):
    """Security, authentication, and authorization errors"""
    def __init__(self, message: str, security_context: Optional[str] = None, user_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.USER_INTERVENTION)
        
        tech_details = kwargs.setdefault('technical_details', {})
        if security_context:
            tech_details['security_context'] = security_context
        if user_id:
            tech_details['user_id'] = user_id

        super().__init__(message, **kwargs)
        
        # Log security events separately using the dedicated security logger
        security_error_logger.critical(f"Security Event: {message}", parameters={
            'exception_id': self.exception_id,
            'security_context': security_context,
            'user_id': user_id,
            'component': self.context.component,
            'function': self.context.function
        })

class ValidationError(LegalAIException):
    """Data validation and schema errors"""
    def __init__(self, message: str, field_name: Optional[str] = None, expected_type: Optional[str] = None, actual_value: Any = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.WARNING) # Usually a data issue, not system critical
        kwargs.setdefault('recovery_strategy', ErrorRecoveryStrategy.USER_INTERVENTION)
        
        technical_details = kwargs.setdefault('technical_details', {})
        if field_name:
            technical_details['field_name'] = field_name
        if expected_type:
            technical_details['expected_type'] = expected_type
        if actual_value is not None:
            technical_details['actual_value_summary'] = str(actual_value)[:200] # Truncate for security/log size
        
        super().__init__(message, **kwargs)

class ErrorHandler:
    """
    Centralized error handling system with recovery strategies,
    forensic logging, and user notification management.
    """
    
    def __init__(self):
        """Initialize error handler with comprehensive tracking"""
        error_logger.info("=== INITIALIZING ERROR HANDLER SYSTEM ===")
        
        self.error_history: List[LegalAIException] = [] # Consider capping size or periodic cleanup
        self.recovery_attempts: Dict[str, int] = {}
        self.error_patterns: Dict[str, int] = {} # Key: pattern_key, Value: count
        self.error_statistics = {
            'total_errors': 0,
            'by_severity': {severity.name: 0 for severity in ErrorSeverity},
            'by_category': {category.value: 0 for category in ErrorCategory},
            'by_recovery_strategy': {strategy.value: 0 for strategy in ErrorRecoveryStrategy} # Corrected key
        }
        
        error_logger.info("Error handler initialized")
    
    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def handle_exception(
        self,
        exception: Union[Exception, LegalAIException],
        context_override: Optional[ErrorContext] = None, # Renamed context to context_override
        user_notification: bool = True,
        attempt_recovery: bool = True
    ) -> bool:
        """
        Handle exception with comprehensive logging, recovery attempts,
        and user notification management.
        """
        error_logger.info(f"Handling exception: {type(exception).__name__} - {str(exception)[:100]}...") # Log snippet
        
        # Convert to LegalAIException if needed
        if not isinstance(exception, LegalAIException):
            # If context_override is provided, use it. Otherwise, _capture_context will be called inside LegalAIException.
            legal_exception = LegalAIException(
                message=str(exception),
                cause=exception,
                context=context_override # Pass context_override here
            )
        else:
            legal_exception = exception
            if context_override: # If it's already LegalAIException but we want to override context
                legal_exception.context = context_override
        
        # Update statistics
        self._update_error_statistics(legal_exception)
        
        # Add to history (consider limiting history size)
        if len(self.error_history) > 1000: # Example limit
            self.error_history.pop(0)
        self.error_history.append(legal_exception)
        
        # Detect error patterns
        self._detect_error_patterns(legal_exception)
        
        # Attempt recovery if requested
        recovery_success = False
        if attempt_recovery and legal_exception.recovery_strategy != ErrorRecoveryStrategy.NONE:
            recovery_success = self._attempt_recovery(legal_exception)
        
        # Log comprehensive error information (already logged at LegalAIException creation)
        # error_logger.error(f"Exception Handled: {legal_exception.exception_id}",
        #                   parameters={
        #                       'recovery_attempted': attempt_recovery,
        #                       'recovery_success': recovery_success,
        #                       'user_notification_enabled': user_notification, # Corrected key
        #                       'is_pattern_error': self._is_pattern_error(legal_exception) # Corrected key
        #                   },
        #                   exception=legal_exception) # Log the full exception object
        
        # Notify user if requested
        if user_notification:
            self._notify_user(legal_exception)
        
        return recovery_success
    
    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def _update_error_statistics(self, exception: LegalAIException):
        """Update comprehensive error statistics"""
        self.error_statistics['total_errors'] += 1
        self.error_statistics['by_severity'][exception.severity.name] += 1
        self.error_statistics['by_category'][exception.category.value] += 1
        self.error_statistics['by_recovery_strategy'][exception.recovery_strategy.value] += 1 # Corrected key
        
        error_logger.trace("Error statistics updated", parameters=self.error_statistics)
    
    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def _detect_error_patterns(self, exception: LegalAIException):
        """Detect recurring error patterns for proactive handling"""
        # Create a more robust pattern key
        pattern_key = f"{exception.category.value}|{exception.context.component}|{exception.context.function}|{exception.message[:50]}"
        
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
        
        if self.error_patterns[pattern_key] >= 3: # Threshold for pattern detection
            error_logger.warning(f"Recurring Error Pattern Detected: {pattern_key}",
                               parameters={
                                   'pattern_count': self.error_patterns[pattern_key],
                                   'pattern_key': pattern_key,
                                   'severity': exception.severity.name
                               })
    
    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def _attempt_recovery(self, exception: LegalAIException) -> bool:
        """Attempt error recovery based on strategy"""
        recovery_logger.info(f"Attempting recovery for {exception.exception_id} using {exception.recovery_strategy.value}")
        
        recovery_key = f"{exception.category.value}_{exception.recovery_strategy.value}"
        self.recovery_attempts[recovery_key] = self.recovery_attempts.get(recovery_key, 0) + 1
        
        success = False
        try:
            if exception.recovery_strategy == ErrorRecoveryStrategy.RETRY:
                success = self._retry_operation(exception)
            elif exception.recovery_strategy == ErrorRecoveryStrategy.FALLBACK:
                success = self._fallback_operation(exception)
            elif exception.recovery_strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._graceful_degradation(exception)
            elif exception.recovery_strategy == ErrorRecoveryStrategy.DATA_RECOVERY:
                success = self._data_recovery(exception)
            # USER_INTERVENTION and SYSTEM_RESTART are typically handled outside this direct attempt
            elif exception.recovery_strategy in [ErrorRecoveryStrategy.USER_INTERVENTION, ErrorRecoveryStrategy.SYSTEM_RESTART]:
                 recovery_logger.info(f"Recovery strategy {exception.recovery_strategy.value} requires external action for {exception.exception_id}")
                 return False # Not automatically recoverable here
            else: # ErrorRecoveryStrategy.NONE or unknown
                recovery_logger.info(f"No automatic recovery action for strategy {exception.recovery_strategy.value} of {exception.exception_id}")
                return False
            
            recovery_logger.info(f"Recovery attempt for {exception.exception_id} {'succeeded' if success else 'failed'}")
            return success
        
        except Exception as recovery_error:
            recovery_logger.error(f"Recovery attempt itself failed for {exception.exception_id}",
                                exception=recovery_error)
            return False
    
    def _retry_operation(self, exception: LegalAIException) -> bool:
        """Implement retry recovery strategy"""
        recovery_logger.trace(f"Implementing retry strategy for {exception.exception_id} (placeholder)")
        # Actual retry logic would be in the calling code, orchestrated by this handler's decision.
        # This function's role is to decide IF a retry is appropriate.
        # For now, assume retry is possible if this strategy is chosen.
        return True # Signifies that a retry can be attempted by the caller
    
    def _fallback_operation(self, exception: LegalAIException) -> bool:
        """Implement fallback recovery strategy"""
        recovery_logger.trace(f"Implementing fallback strategy for {exception.exception_id} (placeholder)")
        # Actual fallback logic would be in the calling code.
        return True # Signifies that a fallback can be attempted
    
    def _graceful_degradation(self, exception: LegalAIException) -> bool:
        """Implement graceful degradation strategy"""
        recovery_logger.trace(f"Implementing graceful degradation for {exception.exception_id} (placeholder)")
        # Logic to switch to a simpler mode or disable a feature.
        return True # Signifies degradation was applied
    
    def _data_recovery(self, exception: LegalAIException) -> bool:
        """Implement data recovery strategy"""
        recovery_logger.trace(f"Implementing data recovery for {exception.exception_id} (placeholder)")
        # Attempt to restore from backup or repair corrupted data.
        return False # Data recovery is complex and often not fully automatic

    def _is_pattern_error(self, exception: LegalAIException) -> bool:
        """Check if exception is part of a detected pattern"""
        pattern_key = f"{exception.category.value}|{exception.context.component}|{exception.context.function}|{exception.message[:50]}"
        return self.error_patterns.get(pattern_key, 0) >= 3
    
    def _notify_user(self, exception: LegalAIException):
        """Notify user about the error (implementation depends on GUI/API system)"""
        # This should integrate with the FastAPI backend to send a user-friendly message
        # or log it in a way the user/admin can see.
        user_msg = exception.user_message
        error_logger.info(f"User Notification Triggered for {exception.exception_id}: {user_msg}", 
                          parameters={'severity': exception.severity.name})
        # Example: If there's a WebSocket manager:
        # if websocket_manager and exception.severity >= ErrorSeverity.WARNING:
        #     websocket_manager.broadcast_error_notification(user_msg, exception.severity.name)
        pass
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        return {
            'report_generated_at': datetime.now().isoformat(), # Corrected key
            'statistics': self.error_statistics,
            'detected_patterns': {k:v for k,v in self.error_patterns.items() if v >=3}, # Corrected key
            'recovery_attempts_summary': self.recovery_attempts, # Corrected key
            'recent_errors_summary': [ # Corrected key
                {
                    'exception_id': e.exception_id,
                    'severity': e.severity.name,
                    'category': e.category.value,
                    'message_summary': e.message[:100], # Corrected key
                    'timestamp': e.creation_time.isoformat()
                }
                for e in self.error_history[-10:]  # Last 10 errors summary
            ]
        }

# Global error handler instance
_error_handler_instance: Optional[ErrorHandler] = None # Renamed from _error_handler
_handler_lock = threading.RLock() # Lock for initializing the handler

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance (thread-safe singleton)"""
    global _error_handler_instance
    if _error_handler_instance is None:
        with _handler_lock:
            if _error_handler_instance is None: # Double-check locking
                _error_handler_instance = ErrorHandler()
    return _error_handler_instance

def handle_error( # Public API function
    exception: Union[Exception, LegalAIException],
    context_override: Optional[ErrorContext] = None, # Renamed context to context_override
    user_notification: bool = True,
    attempt_recovery: bool = True
) -> bool:
    """Convenience function for handling errors using the global handler."""
    handler = get_error_handler()
    return handler.handle_exception(
        exception, context_override, user_notification, attempt_recovery
    )

# Decorator for automatic error handling
def with_error_handling( # Public API decorator
    recovery_strategy_override: ErrorRecoveryStrategy = ErrorRecoveryStrategy.NONE, # Renamed
    user_notification_override: bool = True, # Renamed
    category_override: ErrorCategory = ErrorCategory.SYSTEM # Renamed
):
    """Decorator for automatic error handling with detailed logging and overrides."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LegalAIException as e:
                # If already a LegalAIException, handle it directly.
                # We might want to update its category/recovery if overrides are provided.
                if category_override != ErrorCategory.SYSTEM: # Check if override is different from default
                    e.category = category_override
                if recovery_strategy_override != ErrorRecoveryStrategy.NONE:
                    e.recovery_strategy = recovery_strategy_override
                
                handle_error(e, user_notification=user_notification_override)
                # Re-raise the original, possibly modified, LegalAIException
                raise
            except Exception as e:
                # Capture context at the point of the original exception
                # For non-LegalAIException, we create a new one, so context capture is important here.
                # We can try to infer component and function from the func being decorated.
                err_ctx = ErrorContext(
                    component=Path(func.__code__.co_filename).name,
                    function=func.__name__
                )
                # Add more details to err_ctx if possible, e.g., args

                legal_exception = LegalAIException(
                    message=f"Unhandled error in {func.__name__}: {str(e)}",
                    category=category_override,
                    recovery_strategy=recovery_strategy_override,
                    cause=e,
                    context=err_ctx
                )
                handle_error(legal_exception, user_notification=user_notification_override)
                # Re-raise the new LegalAIException
                raise legal_exception
        
        return wrapper
    return decorator

if __name__ == "__main__":
    pass