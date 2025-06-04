# legal_ai_system/core/base_agent.py
"""Legal AI System - Base Agent Framework.

This module provides the foundational classes and patterns for all agents
in the Legal AI System. All specialized agents inherit from BaseAgent to
ensure consistent behavior, error handling, and performance tracking.
"""

import asyncio
import time # Replaced logging with detailed_logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use absolute imports with fallback
try:
    from legal_ai_system.core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
    from legal_ai_system.config.constants import Constants
    from legal_ai_system.core.unified_exceptions import AgentError
except ImportError:
    # Fallback for relative imports
    try:
        from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
        from ..config.constants import Constants
        from .unified_exceptions import AgentError
    except ImportError:
        # Final fallback
        import logging
        class LogCategory:
            AGENT = "AGENT"
        def get_detailed_logger(name, category):
            return logging.getLogger(name)
        def detailed_log_function(category):
            def decorator(func):
                return func
            return decorator
        class Constants:
            class Version:
                APP_VERSION = "1.0.0"
            class Time:
                DEFAULT_SERVICE_TIMEOUT_SECONDS = 300
                DEFAULT_RETRY_DELAY_SECONDS = 1.0
            class Performance:
                MAX_RETRY_ATTEMPTS = 3
        class AgentError(Exception):
            pass


# Get a logger for this module
base_agent_logger = get_detailed_logger("BaseAgent", LogCategory.AGENT)


class AgentStatus(Enum):
    """Agent execution status enumeration."""
    IDLE = "idle"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels for agent queue management."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

T = TypeVar('T')

@dataclass
class AgentResult(Generic[T]):
    """Standardized result format for all agent operations."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    agent_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat()
        }

# AgentError is now imported from unified_exceptions.py

@dataclass
class AgentTask:
    """Task container for agent processing queue."""
    id: str
    data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None

class BaseAgent(ABC):
    """Abstract base class for all agents in the Legal AI System."""
    
    def __init__(self, service_container: Optional[Any] = None, name: Optional[str] = None, agent_type: Optional[str] = "generic") -> None: # Added agent_type
        """Initialize agent with service container for dependency injection."""
        self.service_container = service_container # Keep this for service access
        self.services = service_container # For backward compatibility if some old code uses it
        
        self.name = name or self.__class__.__name__
        self.agent_type = agent_type # Store agent type
        self.status = AgentStatus.IDLE
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.current_task: Optional[AgentTask] = None
        
        # Logger specific to this agent instance
        self.logger = get_detailed_logger(f"Agent_{self.name}", LogCategory.AGENT)

        # Performance tracking
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'last_execution': None,
            'last_error_timestamp': None,
        }
        
        self.config = self._load_config()
        
        self._processing = False
        self._shutdown_event = asyncio.Event() # For graceful shutdown
        
        self.logger.info(f"Initialized agent", parameters={'agent_name': self.name, 'agent_type': self.agent_type})
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent-specific configuration."""
        default_config = {
            'timeout': Constants.Time.DEFAULT_SERVICE_TIMEOUT_SECONDS,
            'max_retries': Constants.Performance.MAX_RETRY_ATTEMPTS,
            'retry_delay': Constants.Time.DEFAULT_RETRY_DELAY_SECONDS
        }
        
        agent_specific_config = {}
        if self.service_container and hasattr(self.service_container, 'get_service'):
            try:
                config_manager = self.service_container.get_service("configuration_manager") # Corrected name
                if config_manager:
                    # Attempt to load config for this specific agent, e.g., from a section 'agents.<agent_name>'
                    agent_specific_config = config_manager.get(f"agents.{self.name.lower()}", {})
                    # Also load general processing config
                    processing_config = config_manager.get_processing_config() # Assuming this method exists
                    default_config.update({
                        key: processing_config[key] for key in [
                            'max_concurrent_documents', 'batch_size', 
                            'enable_auto_tagging', 'auto_tag_confidence_threshold'
                        ] if key in processing_config
                    })
                    self.logger.info(f"Loaded configuration from ConfigurationManager", parameters={'agent_name': self.name})
            except Exception as e:
                self.logger.warning(f"Failed to load config from ConfigurationManager. Using defaults.", 
                                   parameters={'agent_name': self.name, 'error': str(e)})
        
        # Merge default, processing, and agent-specific configs
        final_config = {**default_config, **agent_specific_config}
        return final_config

    # =================== SERVICE CONTAINER CONVENIENCE METHODS ===================
    # These are good, assuming service_container.get_service(service_name) is the standard.
    
    def _get_service(self, service_name: str) -> Optional[Any]:
        """Helper to safely get a service from the container."""
        if self.service_container and hasattr(self.service_container, 'get_service'):
            try:
                return self.service_container.get_service(service_name)
            except Exception as e: # Catch specific "service not found" if possible
                self.logger.warning(f"Service '{service_name}' not found or failed to load.", 
                                   parameters={'error': str(e)})
        return None

    def get_config_manager(self): return self._get_service("configuration_manager")
    def get_security_manager(self): return self._get_service("security_manager")
    def get_embedding_manager(self): return self._get_service("embedding_manager")
    def get_knowledge_graph_manager(self): return self._get_service("knowledge_graph_manager")
    def get_vector_store_manager(self): return self._get_service("vector_store_manager")
    def get_memory_manager(self): return self._get_service("unified_memory_manager")
    def get_llm_manager(self): return self._get_service("llm_manager") # Added LLM Manager

    # =================== AGENT PROCESSING METHODS ===================
    
    @abstractmethod
    async def _process_task(self, task_data: Any, metadata: Dict[str, Any]) -> Any:
        """Process a single task - must be implemented by subclasses."""
        pass
    
    @detailed_log_function(LogCategory.AGENT)
    async def process(self, data: Any, priority: TaskPriority = TaskPriority.NORMAL, 
                     metadata: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> AgentResult: # metadata optional
        """Process data with the agent."""
        start_time = time.perf_counter() # Use perf_counter for more precision
        task_id = f"{self.name}_{int(start_time * 1000)}" # More unique task ID
        
        effective_metadata = metadata or {}
        effective_timeout = timeout or self.config.get('timeout', Constants.Time.DEFAULT_SERVICE_TIMEOUT_SECONDS)

        self.logger.info(f"Task started", parameters={'task_id': task_id, 'priority': priority.name, 'timeout': effective_timeout})

        try:
            self.status = AgentStatus.PROCESSING
            self.stats['total_tasks'] += 1
            
            task = AgentTask(
                id=task_id,
                data=data,
                priority=priority,
                metadata=effective_metadata,
                timeout=effective_timeout
            )
            self.current_task = task
            
            result_data = await asyncio.wait_for(
                self._process_task_with_retries(task),
                timeout=task.timeout
            )
            
            execution_time = time.perf_counter() - start_time
            
            self.stats['successful_tasks'] += 1
            self._update_avg_execution_time(execution_time)
            self.stats['last_execution'] = datetime.now().isoformat()
            
            agent_result = AgentResult(
                success=True,
                data=result_data,
                metadata=task.metadata,
                execution_time=execution_time,
                agent_name=self.name
            )
            
            self.status = AgentStatus.COMPLETED
            self.logger.info(f"Task completed successfully", 
                            parameters={'task_id': task_id, 'execution_time_sec': execution_time})
            return agent_result
            
        except asyncio.TimeoutError:
            execution_time = time.perf_counter() - start_time
            error_msg = f"Task {task_id} timed out after {effective_timeout}s"
            self.logger.error(error_msg, parameters={'task_id': task_id, 'timeout': effective_timeout})
            self.stats['failed_tasks'] += 1
            self.stats['last_error_timestamp'] = datetime.now().isoformat()
            self.status = AgentStatus.ERROR
            
            return AgentResult(
                success=False,
                error=error_msg,
                metadata=effective_metadata,
                execution_time=execution_time,
                agent_name=self.name
            )
            
        except AgentError as ae: # Catch specific agent errors
            execution_time = time.perf_counter() - start_time
            self.logger.error(f"Agent error in task {task_id}: {ae.message}", 
                             parameters={'task_id': task_id, 'error_details': ae.details, 'agent_name_in_error': ae.agent_name}, 
                             exception=ae)
            self.stats['failed_tasks'] += 1
            self.stats['last_error_timestamp'] = datetime.now().isoformat()
            self.status = AgentStatus.ERROR
            return AgentResult(success=False, error=ae.message, metadata=effective_metadata, execution_time=execution_time, agent_name=self.name)

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_msg = f"Unhandled error in task {task_id}: {str(e)}"
            self.logger.error(error_msg, parameters={'task_id': task_id}, exception=e) # Log with exception info
            self.stats['failed_tasks'] += 1
            self.stats['last_error_timestamp'] = datetime.now().isoformat()
            self.status = AgentStatus.ERROR
            
            return AgentResult(
                success=False,
                error=error_msg,
                metadata=effective_metadata,
                execution_time=execution_time,
                agent_name=self.name
            )
            
        finally:
            self.current_task = None
            if self.status != AgentStatus.ERROR: # Don't reset to IDLE if an error occurred and wasn't handled
                self.status = AgentStatus.IDLE
    
    async def _process_task_with_retries(self, task: AgentTask) -> Any:
        """Process task with retry logic."""
        max_retries = self.config.get('max_retries', Constants.Performance.MAX_RETRY_ATTEMPTS)
        retry_delay_base = self.config.get('retry_delay', Constants.Time.DEFAULT_RETRY_DELAY_SECONDS)
        
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{max_retries + 1} for task {task.id}", 
                                 parameters={'task_id': task.id, 'attempt': attempt + 1})
                return await self._process_task(task.data, task.metadata)
                
            except AgentError as ae: # Catch specific AgentError to avoid retrying unrecoverable agent logic issues
                self.logger.warning(f"Agent error on attempt {attempt + 1} for task {task.id} (will not retry): {ae.message}",
                                   parameters={'task_id': task.id, 'attempt': attempt + 1})
                last_exception = ae
                break # Do not retry AgentErrors unless they are marked as retryable by their type

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} for task {task.id} failed: {str(e)}",
                                   parameters={'task_id': task.id, 'attempt': attempt + 1})
                
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    retry_delay = (retry_delay_base * (Constants.Performance.EXPONENTIAL_BACKOFF_MULTIPLIER ** attempt)) + (os.urandom(1)[0] / 255.0) # type: ignore
                    self.logger.info(f"Retrying task {task.id} in {retry_delay:.2f}s...",
                                    parameters={'task_id': task.id, 'delay_sec': retry_delay})
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"Task {task.id} failed after {max_retries + 1} attempts.",
                                     parameters={'task_id': task.id})
        
        if last_exception is not None:
            raise AgentError(f"All retry attempts failed for task {task.id}. Last error: {str(last_exception)}", 
                             self.name, details={'original_exception': str(last_exception)}) from last_exception
        return None # Should not be reached if an exception occurred
    
    def _update_avg_execution_time(self, execution_time: float) -> None:
        """Update average execution time using Welford's algorithm for numerical stability."""
        total_successful = self.stats['successful_tasks']
        if total_successful == 0: # Should be 1 if we just completed one
             self.stats['avg_execution_time'] = 0.0 # Avoid division by zero if logic changes
             return

        if total_successful == 1:
            self.stats['avg_execution_time'] = execution_time
        else:
            # Welford's online algorithm for mean
            old_avg = self.stats['avg_execution_time']
            self.stats['avg_execution_time'] = old_avg + (execution_time - old_avg) / total_successful
    
    # Queue-based processing methods
    async def add_task_to_queue(self, data: Any, priority: TaskPriority = TaskPriority.NORMAL, # Renamed for clarity
                               metadata: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> str:
        """Add task to agent's processing queue."""
        task_id = f"{self.name}_q_{int(time.time() * 1000)}" # More unique task ID
        
        task = AgentTask(
            id=task_id,
            data=data,
            priority=priority, # Priority can be used by a PriorityQueue if implemented
            metadata=metadata or {},
            timeout=timeout or self.config.get('timeout', Constants.Time.DEFAULT_SERVICE_TIMEOUT_SECONDS)
        )
        
        await self.task_queue.put(task)
        self.logger.debug(f"Added task to queue", parameters={'task_id': task_id, 'agent_name': self.name, 'queue_size': self.task_queue.qsize()})
        return task_id
    
    async def start_processing_loop(self) -> None: # Renamed for clarity
        """Start background task processing loop."""
        if self._processing:
            self.logger.warning(f"Background processing loop already running.", parameters={'agent_name': self.name})
            return
        
        self._processing = True
        self._shutdown_event.clear()
        self.logger.info(f"Starting background processing loop.", parameters={'agent_name': self.name})
        
        while self._processing and not self._shutdown_event.is_set():
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0) # Wait 1s for a task
                
                self.logger.debug(f"Processing task from queue", parameters={'task_id': task.id, 'agent_name': self.name})
                agent_result = await self.process( # Call the main process method
                    task.data, 
                    task.priority, 
                    task.metadata,
                    task.timeout
                )
                
                self.task_queue.task_done()
                
                if agent_result.success:
                    self.logger.debug(f"Queued task completed successfully", parameters={'task_id': task.id, 'agent_name': self.name})
                else:
                    self.logger.error(f"Queued task failed", parameters={'task_id': task.id, 'agent_name': self.name, 'error': agent_result.error})
                
            except asyncio.TimeoutError:
                # No task in queue, continue loop if still processing
                continue
            except asyncio.CancelledError:
                self.logger.info("Processing loop cancelled.", parameters={'agent_name': self.name})
                break
            except Exception as e:
                self.logger.error(f"Error in background processing loop", parameters={'agent_name': self.name}, exception=e)
                await asyncio.sleep(self.config.get('retry_delay', 1.0)) # Wait before trying to get next task

        self._processing = False
        self.logger.info(f"Background processing loop stopped.", parameters={'agent_name': self.name})

    async def stop_processing_loop(self) -> None: # Renamed for clarity
        """Stop background task processing loop."""
        self.logger.info(f"Stopping background processing loop...", parameters={'agent_name': self.name})
        self._processing = False # Signal loop to stop
        self._shutdown_event.set() # Signal loop to stop (if waiting on queue)

        # Optionally, wait for current task to finish if one is running
        # This requires more complex state management (e.g. self.current_processing_task_future)

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self.logger.info(f"Shutting down agent...", parameters={'agent_name': self.name})
        await self.stop_processing_loop()
        
        # Wait for the processing loop to actually exit if it was running
        # This might require joining a task if start_processing_loop was run as a task.
        # For now, just setting flags.

        # Clear queue (optional: process remaining tasks or save them)
        while not self.task_queue.empty():
            task = await self.task_queue.get()
            self.logger.warning(f"Discarding queued task during shutdown", parameters={'task_id': task.id, 'agent_name': self.name})
            self.task_queue.task_done()
        
        self.logger.info(f"Agent shutdown complete.", parameters={'agent_name': self.name})
    
    # Status and monitoring methods
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and basic stats."""
        return {
            'name': self.name,
            'type': self.agent_type,
            'status': self.status.value,
            'current_task_id': self.current_task.id if self.current_task else None,
            'queue_size': self.task_queue.qsize(),
            'config': self.config, # Be cautious about exposing sensitive config details
            'stats': self.stats # Performance stats
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        total_tasks = self.stats['total_tasks']
        success_rate = (self.stats['successful_tasks'] / total_tasks) if total_tasks > 0 else 0.0
        
        return {
            'agent_name': self.name,
            'agent_type': self.agent_type,
            'total_tasks_processed': total_tasks,
            'successful_tasks': self.stats['successful_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'success_rate': success_rate,
            'average_execution_time_sec': self.stats['avg_execution_time'],
            'last_successful_execution': self.stats['last_execution'],
            'last_error_timestamp': self.stats['last_error_timestamp']
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the agent."""
        # Basic health check, can be extended by subclasses
        is_healthy = self.status != AgentStatus.ERROR and not self._shutdown_event.is_set()
        
        # Check dependencies (e.g., LLM manager)
        llm_manager = self.get_llm_manager()
        llm_status = "not_configured"
        if llm_manager and hasattr(llm_manager, 'health_check'):
            llm_health = await llm_manager.health_check()
            llm_status = llm_health.get("status", "unknown")
            if llm_status != "healthy":
                is_healthy = False # Degrade health if a critical dependency is unhealthy
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "current_operational_status": self.status.value,
            "queue_size": self.task_queue.qsize(),
            "dependencies_status": {
                "llm_manager": llm_status
            },
            "timestamp": datetime.now().isoformat()
        }
    
    # Utility methods for subclasses
    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """Utility method to call LLM through service container's LLMManager."""
        llm_manager = self.get_llm_manager()
        if not llm_manager:
            raise AgentError("LLMManager service not available.", self.name)
        
        try:
            # Assuming llm_manager has an async `complete` method
            response_obj = await llm_manager.complete(prompt, **kwargs) # Expects LLMResponse object
            return response_obj.content 
        except Exception as e:
            raise AgentError(f"LLM call failed: {str(e)}", self.name, details={'original_exception': str(e)}) from e
    
    def _validate_input(self, data: Any, required_fields: Optional[List[str]] = None) -> None:
        """Basic input validation."""
        if data is None:
            raise AgentError("Input data cannot be None.", self.name, details={'validation_error': 'data_is_none'})
        
        if required_fields and isinstance(data, dict):
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise AgentError(f"Missing required fields: {', '.join(missing_fields)}.", 
                                 self.name, details={'missing_fields': missing_fields})
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.agent_type}', status='{self.status.value}')"