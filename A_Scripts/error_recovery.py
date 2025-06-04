"""
Basic Error Recovery System - Phase 2
Intelligent error handling with retry logic and fallback strategies
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class ErrorType(Enum):
    """Classification of error types for recovery strategies"""
    FILE_CORRUPTED = "file_corrupted"
    LLM_TIMEOUT = "llm_timeout"
    MEMORY_ERROR = "memory_error"
    PARSING_ERROR = "parsing_error"
    NETWORK_ERROR = "network_error"
    DEPENDENCY_MISSING = "dependency_missing"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    error_type: ErrorType
    strategy: str
    success: bool
    duration: float
    error_message: Optional[str] = None


class ErrorRecovery:
    """
    Basic error recovery with multiple strategies.
    
    Features:
    - Automatic error classification
    - Multiple recovery strategies per error type
    - Exponential backoff for retries
    - Recovery history tracking
    - Fallback mechanisms
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.recovery_history = []
        self.logger = logging.getLogger(__name__)
        
        # Define recovery strategies for each error type
        self.recovery_strategies = {
            ErrorType.FILE_CORRUPTED: [
                self._try_alternative_reader,
                self._try_raw_text_extraction,
                self._try_partial_recovery
            ],
            ErrorType.LLM_TIMEOUT: [
                self._reduce_chunk_size,
                self._switch_to_faster_model,
                self._use_simpler_prompt
            ],
            ErrorType.MEMORY_ERROR: [
                self._reduce_batch_size,
                self._enable_streaming,
                self._clear_cache_and_retry
            ],
            ErrorType.PARSING_ERROR: [
                self._try_different_encoding,
                self._try_fallback_parser,
                self._extract_raw_content
            ],
            ErrorType.NETWORK_ERROR: [
                self._retry_with_backoff,
                self._switch_provider,
                self._use_local_fallback
            ],
            ErrorType.DEPENDENCY_MISSING: [
                self._suggest_installation,
                self._use_alternative_library,
                self._fallback_to_basic_processing
            ]
        }
    
    async def recover_with_retry(self, 
                               func: Callable, 
                               *args, 
                               context: Dict[str, Any] = None, 
                               **kwargs) -> Any:
        """
        Execute function with automatic error recovery and retries.
        
        Args:
            func: Function to execute
            *args: Function arguments
            context: Additional context for recovery
            **kwargs: Function keyword arguments
        
        Returns:
            Function result or raises exception if all recovery attempts fail
        """
        context = context or {}
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                
                # Success - record if this was a retry
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Try recovery strategies if not the last attempt
                if attempt < self.max_retries - 1:
                    recovery_success = await self._attempt_recovery(
                        error_type, e, context, attempt
                    )
                    
                    if recovery_success:
                        # Update context with recovery modifications
                        if "modified_args" in context:
                            args = context["modified_args"]
                        if "modified_kwargs" in context:
                            kwargs.update(context["modified_kwargs"])
                        
                        # Wait before retry with exponential backoff
                        delay = self.base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                
                # If we're here, recovery failed or this is the last attempt
                break
        
        # All attempts failed
        self.logger.error(f"All {self.max_retries} attempts failed. Last error: {str(last_error)}")
        raise last_error
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for appropriate recovery strategy"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # File-related errors
        if any(keyword in error_str for keyword in ["corrupted", "invalid file", "bad file"]):
            return ErrorType.FILE_CORRUPTED
        
        # Timeout errors
        if any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return ErrorType.LLM_TIMEOUT
        
        # Memory errors
        if any(keyword in error_str for keyword in ["memory", "out of memory", "memoryerror"]):
            return ErrorType.MEMORY_ERROR
        
        # Parsing errors
        if any(keyword in error_str for keyword in ["parsing", "decode", "encoding", "invalid format"]):
            return ErrorType.PARSING_ERROR
        
        # Network errors
        if any(keyword in error_str for keyword in ["network", "connection", "http", "ssl"]):
            return ErrorType.NETWORK_ERROR
        
        # Import/dependency errors
        if "importerror" in error_type_name or "modulenotfounderror" in error_type_name:
            return ErrorType.DEPENDENCY_MISSING
        
        return ErrorType.UNKNOWN
    
    async def _attempt_recovery(self, 
                              error_type: ErrorType, 
                              error: Exception, 
                              context: Dict[str, Any], 
                              attempt: int) -> bool:
        """
        Attempt recovery using appropriate strategies for error type.
        
        Returns:
            True if recovery was attempted and might help, False otherwise
        """
        strategies = self.recovery_strategies.get(error_type, [])
        
        if not strategies:
            self.logger.warning(f"No recovery strategies for error type: {error_type}")
            return False
        
        # Try the first available strategy that hasn't been tried yet
        strategy_index = min(attempt, len(strategies) - 1)
        strategy = strategies[strategy_index]
        
        try:
            start_time = time.time()
            success = await strategy(error, context)
            duration = time.time() - start_time
            
            # Record recovery attempt
            recovery_record = RecoveryAttempt(
                error_type=error_type,
                strategy=strategy.__name__,
                success=success,
                duration=duration,
                error_message=str(error) if not success else None
            )
            self.recovery_history.append(recovery_record)
            
            if success:
                self.logger.info(f"Recovery strategy '{strategy.__name__}' succeeded")
            else:
                self.logger.warning(f"Recovery strategy '{strategy.__name__}' failed")
            
            return success
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy '{strategy.__name__}' threw exception: {str(recovery_error)}")
            return False
    
    # Recovery strategy implementations
    
    async def _try_alternative_reader(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try alternative file reading methods"""
        self.logger.info("Attempting alternative file reader")
        # This would implement fallback readers
        context["use_alternative_reader"] = True
        return True
    
    async def _try_raw_text_extraction(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try basic text extraction as fallback"""
        self.logger.info("Attempting raw text extraction")
        context["use_raw_extraction"] = True
        return True
    
    async def _try_partial_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try to recover partial content"""
        self.logger.info("Attempting partial content recovery")
        context["allow_partial_content"] = True
        return True
    
    async def _reduce_chunk_size(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Reduce chunk size for LLM processing"""
        current_size = context.get("chunk_size", 3000)
        new_size = max(500, current_size // 2)
        
        if new_size < 500:
            return False  # Too small to be useful
        
        self.logger.info(f"Reducing chunk size from {current_size} to {new_size}")
        context["chunk_size"] = new_size
        
        # Update kwargs if present
        if "modified_kwargs" not in context:
            context["modified_kwargs"] = {}
        context["modified_kwargs"]["chunk_size"] = new_size
        
        return True
    
    async def _switch_to_faster_model(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Switch to a faster model for processing"""
        self.logger.info("Switching to faster model")
        context["use_faster_model"] = True
        return True
    
    async def _use_simpler_prompt(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Use a simpler prompt to reduce processing load"""
        self.logger.info("Using simpler prompt")
        context["use_simple_prompt"] = True
        return True
    
    async def _reduce_batch_size(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Reduce batch size to use less memory"""
        current_batch = context.get("batch_size", 10)
        new_batch = max(1, current_batch // 2)
        
        self.logger.info(f"Reducing batch size from {current_batch} to {new_batch}")
        context["batch_size"] = new_batch
        return True
    
    async def _enable_streaming(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Enable streaming to reduce memory usage"""
        self.logger.info("Enabling streaming mode")
        context["enable_streaming"] = True
        return True
    
    async def _clear_cache_and_retry(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Clear cache to free memory"""
        self.logger.info("Clearing cache to free memory")
        context["clear_cache"] = True
        return True
    
    async def _try_different_encoding(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try different text encoding"""
        encodings = context.get("tried_encodings", [])
        available_encodings = ["utf-8", "latin-1", "cp1252", "utf-16"]
        
        for encoding in available_encodings:
            if encoding not in encodings:
                self.logger.info(f"Trying encoding: {encoding}")
                context["encoding"] = encoding
                context.setdefault("tried_encodings", []).append(encoding)
                return True
        
        return False  # No more encodings to try
    
    async def _try_fallback_parser(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try fallback parser"""
        self.logger.info("Using fallback parser")
        context["use_fallback_parser"] = True
        return True
    
    async def _extract_raw_content(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Extract raw content without parsing"""
        self.logger.info("Extracting raw content")
        context["extract_raw_only"] = True
        return True
    
    async def _retry_with_backoff(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Simple retry with backoff (already handled by main retry loop)"""
        return True
    
    async def _switch_provider(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Switch to alternative provider"""
        self.logger.info("Switching to alternative provider")
        context["switch_provider"] = True
        return True
    
    async def _use_local_fallback(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Use local processing as fallback"""
        self.logger.info("Using local fallback processing")
        context["use_local_fallback"] = True
        return True
    
    async def _suggest_installation(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Suggest installing missing dependency"""
        self.logger.warning(f"Missing dependency: {str(error)}")
        context["suggest_install"] = str(error)
        return False  # Can't automatically fix this
    
    async def _use_alternative_library(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Use alternative library"""
        self.logger.info("Using alternative library")
        context["use_alternative_lib"] = True
        return True
    
    async def _fallback_to_basic_processing(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Fallback to basic processing without optional dependencies"""
        self.logger.info("Falling back to basic processing")
        context["basic_processing_only"] = True
        return True
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {"total_attempts": 0}
        
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r.success)
        
        error_type_counts = {}
        strategy_success_rates = {}
        
        for record in self.recovery_history:
            # Count by error type
            error_type = record.error_type.value
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            # Track strategy success rates
            strategy = record.strategy
            if strategy not in strategy_success_rates:
                strategy_success_rates[strategy] = {"attempts": 0, "successes": 0}
            
            strategy_success_rates[strategy]["attempts"] += 1
            if record.success:
                strategy_success_rates[strategy]["successes"] += 1
        
        # Calculate success rates
        for strategy_data in strategy_success_rates.values():
            attempts = strategy_data["attempts"]
            successes = strategy_data["successes"]
            strategy_data["success_rate"] = successes / attempts if attempts > 0 else 0
        
        return {
            "total_attempts": total_attempts,
            "successful_recoveries": successful_recoveries,
            "overall_success_rate": successful_recoveries / total_attempts,
            "error_type_distribution": error_type_counts,
            "strategy_performance": strategy_success_rates
        }


# Global error recovery instance
error_recovery = ErrorRecovery()

# Export
__all__ = ["ErrorRecovery", "ErrorType", "RecoveryAttempt", "error_recovery"]