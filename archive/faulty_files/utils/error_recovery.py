# legal_ai_system/utils/error_recovery.py
"""
Error Recovery System - Phase 2
Intelligent error handling with retry logic and fallback strategies for various operations.
"""

import asyncio
# import logging # Replaced by detailed_logging
from typing import Any, Dict, List, Optional, Callable, Coroutine # Added Coroutine
from dataclasses import dataclass, field # Added field
from enum import Enum
import time

# Use detailed_logging
from ..core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Initialize logger for this module
error_recovery_logger = get_detailed_logger("ErrorRecoveryUtil", LogCategory.ERROR_HANDLING)


class ErrorType(Enum):
    """Classification of error types for recovery strategies."""
    FILE_CORRUPTED = "file_corrupted"
    LLM_TIMEOUT = "llm_timeout"
    LLM_API_ERROR = "llm_api_error" # More specific than just timeout
    MEMORY_ALLOCATION_ERROR = "memory_allocation_error" # Renamed
    PARSING_ERROR = "parsing_error"
    NETWORK_CONNECTIVITY_ERROR = "network_connectivity_error" # Renamed
    DEPENDENCY_MISSING = "dependency_missing"
    DATABASE_TRANSACTION_ERROR = "database_transaction_error" # Added
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded" # Added
    UNKNOWN_SYSTEM_ERROR = "unknown_system_error" # Renamed

@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    error_type: ErrorType
    strategy_used: str # Renamed from strategy
    success: bool
    duration_sec: float # Renamed
    attempt_number: int # Added
    error_message: Optional[str] = None
    context_modifications: Optional[Dict[str, Any]] = None # What context was changed

class ErrorRecovery:
    """
    Provides mechanisms for recovering from common operational errors.
    """
    
    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def __init__(self, max_retries_default: int = 3, base_delay_sec_default: float = 1.0): # Renamed params
        error_recovery_logger.info("Initializing ErrorRecovery utility.")
        self.max_retries_default = max_retries_default
        self.base_delay_sec_default = base_delay_sec_default
        self.recovery_attempt_history: List[RecoveryAttempt] = [] # Renamed, consider capping size

        # Define recovery strategies for each error type
        # Each strategy function should return True if recovery action was taken (and caller should retry),
        # False if no recovery action applicable or recovery failed.
        # They can modify the 'context' dict to influence retries.
        self.error_type_strategies: Dict[ErrorType, List[Callable[[Exception, Dict[str, Any]], Coroutine[Any, Any, bool]]]] = { # Type hint
            ErrorType.FILE_CORRUPTED: [
                self._strategy_try_alternative_reader, self._strategy_try_raw_extraction
            ],
            ErrorType.LLM_TIMEOUT: [
                self._strategy_reduce_llm_request_size, self._strategy_switch_to_faster_llm_model
            ],
            ErrorType.LLM_API_ERROR: [ # Different from timeout, might be server error
                self._strategy_retry_with_backoff, self._strategy_switch_llm_provider_or_model
            ],
            ErrorType.MEMORY_ALLOCATION_ERROR: [
                self._strategy_reduce_batch_or_chunk_size, self._strategy_trigger_gc_and_retry
            ],
            ErrorType.PARSING_ERROR: [
                self._strategy_try_different_encoding, self._strategy_use_fallback_parser
            ],
            ErrorType.NETWORK_CONNECTIVITY_ERROR: [
                self._strategy_retry_with_backoff # Network errors are often transient
            ],
            ErrorType.DATABASE_TRANSACTION_ERROR: [
                self._strategy_retry_transaction # DB deadlocks or transient issues
            ],
            ErrorType.RATE_LIMIT_EXCEEDED: [
                self._strategy_wait_for_rate_limit_reset # Wait longer
            ],
            # DEPENDENCY_MISSING usually cannot be recovered at runtime automatically
            ErrorType.DEPENDENCY_MISSING: [self._strategy_log_dependency_issue],
            ErrorType.UNKNOWN_SYSTEM_ERROR: [self._strategy_log_and_no_retry] # Default for unknown
        }
        error_recovery_logger.info("ErrorRecovery initialized.", 
                                 parameters={'max_retries': self.max_retries_default, 'base_delay': self.base_delay_sec_default})

    @detailed_log_function(LogCategory.ERROR_HANDLING)
    async def attempt_recovery_async(self, # Renamed from recover_with_retry
                               func_to_recover: Callable[..., Coroutine[Any, Any, Any]], # Renamed, type hint
                               *args: Any, 
                               error_context: Optional[Dict[str, Any]] = None, # Renamed
                               max_retries_override: Optional[int] = None, # Renamed
                               **kwargs: Any) -> Any:
        """
        Executes an async function with automatic error recovery and retries.
        """
        error_recovery_logger.info(f"Attempting recovery for function: {func_to_recover.__name__}")
        current_error_context = error_context or {}
        retries = max_retries_override if max_retries_override is not None else self.max_retries_default
        last_exception_caught: Optional[Exception] = None # Renamed

        for attempt_num in range(retries + 1): # Allow initial attempt + number of retries
            current_error_context['current_attempt'] = attempt_num + 1 # For strategies to know
            try:
                # Pass modified context (args/kwargs) to the function if strategies changed them
                modified_args = current_error_context.get("modified_args", args)
                modified_kwargs = current_error_context.get("modified_kwargs", kwargs)
                
                result = await func_to_recover(*modified_args, **modified_kwargs)
                
                if attempt_num > 0: # Log if a retry was successful
                    error_recovery_logger.info(f"Function {func_to_recover.__name__} succeeded on attempt {attempt_num + 1}.")
                return result
                
            except Exception as e:
                last_exception_caught = e
                error_type_classified = self._classify_error_type(e) # Renamed
                
                error_recovery_logger.warning(f"Attempt {attempt_num + 1} for {func_to_recover.__name__} failed.", 
                                             parameters={'error_type': error_type_classified.value}, exception=e)
                
                if attempt_num < retries:
                    recovery_strategy_applied = await self._apply_recovery_strategies(
                        error_type_classified, e, current_error_context, attempt_num
                    )
                    
                    if recovery_strategy_applied: # If a strategy modified context and suggests retry
                        delay_sec = self.base_delay_sec_default * (2 ** attempt_num) # Exponential backoff
                        error_recovery_logger.info(f"Retrying {func_to_recover.__name__} in {delay_sec:.2f}s after recovery strategy.",
                                                 parameters={'strategy_applied': recovery_strategy_applied})
                        await asyncio.sleep(delay_sec)
                        continue # Retry the loop
                    else: # No applicable/successful recovery strategy, or strategy advises not to retry
                        error_recovery_logger.warning(f"No successful recovery strategy applied for {func_to_recover.__name__} on attempt {attempt_num + 1}. Failing.")
                        break # Break loop and re-raise
                else: # Max retries reached
                    error_recovery_logger.error(f"Max retries ({retries}) reached for {func_to_recover.__name__}. Failing.")
                    break # Break loop
        
        # If loop finished due to exhausted retries or failed recovery
        error_recovery_logger.error(f"All recovery attempts for {func_to_recover.__name__} failed.", 
                                   parameters={'last_error': str(last_exception_caught)})
        if last_exception_caught:
            raise last_exception_caught # Re-raise the last caught exception
        else: # Should not happen if loop broke due to error
            raise RuntimeError(f"Error recovery failed for {func_to_recover.__name__} without a specific exception.")


    def _classify_error_type(self, error: Exception) -> ErrorType: # Renamed
        """Classifies an exception into an ErrorType for strategy selection."""
        # This can be made more sophisticated, checking specific exception types
        error_str_lower = str(error).lower()
        error_class_name_lower = type(error).__name__.lower()

        if isinstance(error, (FileNotFoundError, PermissionError)): return ErrorType.FILE_CORRUPTED # Or FILE_ACCESS_ERROR
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)): return ErrorType.LLM_TIMEOUT # Or GENERIC_TIMEOUT
        if "timeout" in error_str_lower: return ErrorType.LLM_TIMEOUT

        if isinstance(error, MemoryError): return ErrorType.MEMORY_ALLOCATION_ERROR
        if isinstance(error, (json.JSONDecodeError, SyntaxError)): return ErrorType.PARSING_ERROR # SyntaxError for bad JSON too
        
        # For network/DB, check for specific library exceptions if possible
        # Example: if isinstance(error, (requests.exceptions.ConnectionError, asyncpg.exceptions.InterfaceError)):
        if any(kw in error_str_lower for kw in ["connection refused", "host not found", "network is unreachable"]):
            return ErrorType.NETWORK_CONNECTIVITY_ERROR
        if "deadlock" in error_str_lower or "transaction aborted" in error_str_lower:
            return ErrorType.DATABASE_TRANSACTION_ERROR
        if "rate limit" in error_str_lower or "quota exceeded" in error_str_lower:
            return ErrorType.RATE_LIMIT_EXCEEDED

        if isinstance(error, ImportError): return ErrorType.DEPENDENCY_MISSING
        
        # Check for LLM specific API errors (e.g. based on status codes if it's an HTTPError from a library)
        # if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        #     if error.response.status_code in [500, 502, 503, 504]: return ErrorType.LLM_API_ERROR
        #     if error.response.status_code == 429: return ErrorType.RATE_LIMIT_EXCEEDED

        error_recovery_logger.debug("Error classified.", parameters={'error_class': error_class_name_lower, 'error_str': error_str_lower, 'classified_as': ErrorType.UNKNOWN_SYSTEM_ERROR.value})
        return ErrorType.UNKNOWN_SYSTEM_ERROR

    async def _apply_recovery_strategies(self, error_type: ErrorType, error_obj: Exception, # Renamed params
                                       current_context: Dict[str, Any], attempt_num: int) -> bool:
        """Applies suitable recovery strategies for the given error type."""
        strategies_to_try = self.error_type_strategies.get(error_type, [])
        if not strategies_to_try:
            error_recovery_logger.debug(f"No defined recovery strategies for error type.", parameters={'error_type': error_type.value})
            return False

        # Try strategies in order. Some strategies might be mutually exclusive or progressive.
        # This simple version tries the strategy corresponding to the attempt number, if available.
        strategy_idx_to_try = attempt_num % len(strategies_to_try) # Cycle through strategies on retries
        selected_strategy_func = strategies_to_try[strategy_idx_to_try]
        
        strategy_name = selected_strategy_func.__name__
        error_recovery_logger.info(f"Attempting recovery strategy.", 
                                 parameters={'error_type': error_type.value, 'strategy': strategy_name, 'attempt': attempt_num + 1})
        
        start_time = time.perf_counter()
        try:
            # Strategy functions are async and modify current_context if they change params for retry
            strategy_succeeded = await selected_strategy_func(error_obj, current_context)
            duration_sec = time.perf_counter() - start_time
            
            recovery_attempt_obj = RecoveryAttempt( # Renamed
                error_type=error_type, strategy_used=strategy_name, success=strategy_succeeded,
                duration_sec=duration_sec, attempt_number=attempt_num + 1,
                error_message=str(error_obj) if not strategy_succeeded else None,
                context_modifications=current_context.get("last_strategy_modifications") # Strategy should populate this
            )
            self.recovery_attempt_history.append(recovery_attempt_obj)
            
            if strategy_succeeded:
                error_recovery_logger.info(f"Recovery strategy '{strategy_name}' applied successfully.", parameters={'duration_sec': duration_sec})
                return True # Indicates a retry with potentially modified context is warranted
            else:
                error_recovery_logger.warning(f"Recovery strategy '{strategy_name}' did not lead to a retryable state.")
                return False
        except Exception as recovery_exec_err:
            error_recovery_logger.error(f"Recovery strategy '{strategy_name}' itself raised an exception.", exception=recovery_exec_err)
            return False # Strategy failed

    # --- Placeholder Recovery Strategy Implementations ---
    # These should modify `context` if they change parameters for the next retry.
    # They return True if a retry is sensible after the strategy, False otherwise.

    async def _strategy_try_alternative_reader(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Try alternative file reader.")
        context["use_alternative_reader_flag"] = True # Example modification
        context["last_strategy_modifications"] = {"use_alternative_reader_flag": True}
        return True

    async def _strategy_try_raw_extraction(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Try raw text extraction.")
        context["use_raw_extraction_flag"] = True
        context["last_strategy_modifications"] = {"use_raw_extraction_flag": True}
        return True
        
    async def _strategy_reduce_llm_request_size(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Reduce LLM request size (e.g. chunk size).")
        current_chunk_size = context.get("modified_kwargs", {}).get("chunk_size", context.get("initial_chunk_size", 3000))
        new_chunk_size = max(500, int(current_chunk_size * 0.75))
        if new_chunk_size == current_chunk_size and new_chunk_size == 500: return False # Cannot reduce further

        context.setdefault("modified_kwargs", {})["chunk_size"] = new_chunk_size
        context["last_strategy_modifications"] = {"chunk_size": new_chunk_size}
        error_recovery_logger.info(f"Reduced chunk size to {new_chunk_size}")
        return True

    async def _strategy_switch_to_faster_llm_model(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Switch to a faster/smaller LLM model.")
        # This requires ModelSwitcher integration or knowledge of available models.
        # context.setdefault("modified_kwargs", {})["model_name"] = "faster-model-xyz"
        context["last_strategy_modifications"] = {"switched_to_faster_model": True} # Placeholder
        return True # Assume switch is possible

    async def _strategy_retry_with_backoff(self, error: Exception, context: Dict[str, Any]) -> bool:
        # The main retry loop already handles backoff. This strategy just confirms a retry is okay.
        error_recovery_logger.info("Applying strategy: Retry with backoff (handled by main loop).")
        context["last_strategy_modifications"] = {"retry_confirmed": True}
        return True

    # ... Implement other strategy placeholders similarly ...
    async def _strategy_reduce_batch_or_chunk_size(self, error: Exception, context: Dict[str, Any]) -> bool:
        return await self._strategy_reduce_llm_request_size(error, context) # Reuse logic for now

    async def _strategy_trigger_gc_and_retry(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Triggering GC and retrying.")
        import gc
        gc.collect()
        context["last_strategy_modifications"] = {"gc_triggered": True}
        return True

    async def _strategy_try_different_encoding(self, error: Exception, context: Dict[str, Any]) -> bool:
        # This would require the original function to accept an encoding parameter
        # and for this context to track tried encodings.
        tried_encodings = context.get("tried_encodings_list", [])
        possible_encodings = ["utf-8", "latin-1", "cp1252"]
        next_encoding_to_try = next((enc for enc in possible_encodings if enc not in tried_encodings), None)
        if next_encoding_to_try:
            context.setdefault("modified_kwargs", {})["encoding_override"] = next_encoding_to_try
            tried_encodings.append(next_encoding_to_try)
            context["tried_encodings_list"] = tried_encodings
            context["last_strategy_modifications"] = {"encoding_override": next_encoding_to_try}
            error_recovery_logger.info(f"Applying strategy: Trying encoding {next_encoding_to_try}.")
            return True
        error_recovery_logger.info("Applying strategy: No more encodings to try.")
        return False

    async def _strategy_use_fallback_parser(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Use fallback parser.")
        context.setdefault("modified_kwargs", {})["use_fallback_parser_flag"] = True
        context["last_strategy_modifications"] = {"use_fallback_parser_flag": True}
        return True

    async def _strategy_switch_llm_provider_or_model(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Switch LLM provider or model.")
        # Requires integration with LLMManager/ModelSwitcher
        # context.setdefault("modified_kwargs", {})["use_fallback_llm_provider"] = True
        context["last_strategy_modifications"] = {"switched_llm_provider": True} # Placeholder
        return True

    async def _strategy_retry_transaction(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.info("Applying strategy: Retry database transaction (handled by main loop).")
        context["last_strategy_modifications"] = {"db_retry_confirmed": True}
        return True

    async def _strategy_wait_for_rate_limit_reset(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Extract 'Retry-After' header if possible from error, or use a default long wait
        retry_after_sec = 60.0 # Default wait for rate limits
        error_str = str(error).lower()
        match = re.search(r'retry after (\d+)', error_str)
        if match: retry_after_sec = float(match.group(1))
        
        error_recovery_logger.info(f"Applying strategy: Waiting for rate limit reset.", parameters={'wait_sec': retry_after_sec})
        await asyncio.sleep(retry_after_sec)
        context["last_strategy_modifications"] = {"waited_for_rate_limit": retry_after_sec}
        return True # Retry after waiting

    async def _strategy_log_dependency_issue(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.critical(f"Dependency Missing: {str(error)}. This usually requires manual intervention (installation).")
        context["last_strategy_modifications"] = {"dependency_issue_logged": str(error)}
        return False # Cannot auto-recover

    async def _strategy_log_and_no_retry(self, error: Exception, context: Dict[str, Any]) -> bool:
        error_recovery_logger.error(f"Unknown error encountered. No automatic retry strategy.", parameters={'error': str(error)})
        context["last_strategy_modifications"] = {"unknown_error_no_retry": True}
        return False # Do not retry unknown errors by default

    @detailed_log_function(LogCategory.ERROR_HANDLING)
    def get_recovery_statistics_summary(self) -> Dict[str, Any]: # Renamed
        """Get summary statistics about recovery attempts."""
        if not self.recovery_attempt_history:
            return {"total_recovery_attempts": 0}
        
        total = len(self.recovery_attempt_history)
        successful = sum(1 for r in self.recovery_attempt_history if r.success)
        
        by_error_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0})
        by_strategy: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0})

        for rec in self.recovery_attempt_history:
            by_error_type[rec.error_type.value]["attempts"] += 1
            by_strategy[rec.strategy_used]["attempts"] += 1
            if rec.success:
                by_error_type[rec.error_type.value]["successes"] += 1
                by_strategy[rec.strategy_used]["successes"] += 1
        
        # Calculate success rates
        for data in by_error_type.values(): data["success_rate"] = (data["successes"] / data["attempts"]) if data["attempts"] > 0 else 0
        for data in by_strategy.values(): data["success_rate"] = (data["successes"] / data["attempts"]) if data["attempts"] > 0 else 0
            
        summary = {
            "total_recovery_attempts": total,
            "successful_recovery_attempts": successful,
            "overall_recovery_success_rate": (successful / total) if total > 0 else 0,
            "attempts_by_error_type": dict(by_error_type),
            "performance_by_strategy": dict(by_strategy)
        }
        error_recovery_logger.info("Error recovery statistics summary generated.", parameters={'total_attempts': total})
        return summary

# Global error recovery instance (singleton pattern)
_error_recovery_instance: Optional[ErrorRecovery] = None
_instance_lock = threading.Lock()

def get_error_recovery_instance(max_retries: int = 3, base_delay_sec: float = 1.0) -> ErrorRecovery:
    """Get the global singleton instance of ErrorRecovery."""
    global _error_recovery_instance
    if _error_recovery_instance is None:
        with _instance_lock: # Thread-safe initialization
            if _error_recovery_instance is None:
                _error_recovery_instance = ErrorRecovery(max_retries_default=max_retries, base_delay_sec_default=base_delay_sec)
    return _error_recovery_instance

# Export for easy use
__all__ = ["ErrorRecovery", "ErrorType", "RecoveryAttempt", "get_error_recovery_instance"]