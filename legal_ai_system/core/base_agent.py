"""Legal AI System - Base Agent Framework.

This module provides the foundational classes and patterns for all agents
in the Legal AI System. All specialized agents inherit from BaseAgent to
ensure consistent behavior, error handling, and performance tracking.

The base agent framework includes:
- Async task processing with timeouts and retries
- Standardized result formats with metadata
- Performance tracking and statistics
- Service container integration for dependency injection
- Comprehensive error handling and logging

Typical usage example:
    >>> class MyAgent(BaseAgent):
    ...     async def _process_task(self, task_data, metadata):
    ...         return {"result": "processed"}
    >>>
    >>> agent = MyAgent(service_container)
    >>> result = await agent.process({"input": "data"})
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.constants import Constants
from ..core.detailed_logging import get_detailed_logger, LogCategory

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status enumeration.

    Defines the possible states an agent can be in during its lifecycle.
    Used for monitoring and coordination between agents.
    """

    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels for agent queue management.

    Defines priority ordering for task processing. Higher priority
    tasks are processed before lower priority tasks.
    """

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


T = TypeVar("T")


@dataclass
class AgentResult(Generic[T]):
    """Standardized result format for all agent operations.

    Provides consistent structure for agent results including success status,
    data payload, error information, and execution metadata. Generic type T
    allows for type-safe result data.

    Attributes:
        success: Whether the operation completed successfully.
        data: The result data, type varies by agent and operation.
        error: Error message if operation failed, None if successful.
        metadata: Additional metadata about the operation.
        execution_time: Time taken to complete the operation in seconds.
        agent_name: Name of the agent that produced this result.
        timestamp: When the result was created.

    Example:
        >>> result = AgentResult[str](
        ...     success=True,
        ...     data="processed text",
        ...     agent_name="TextProcessor"
        ... )
        >>> print(result.to_dict())
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    agent_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format.

        Returns:
            Dictionary representation of the result with all fields.
            Timestamp is converted to ISO format string.
        """
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
        }


class AgentError(Exception):
    """Base exception for agent operations.

    Custom exception class for agent-specific errors that includes
    additional context like agent name and error details.

    Attributes:
        agent_name: Name of the agent that raised the error.
        details: Additional error context and debugging information.
    """

    def __init__(
        self,
        message: str,
        agent_name: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AgentError with context information.

        Args:
            message: Human-readable error description.
            agent_name: Name of the agent that encountered the error.
            details: Additional error context for debugging.
        """
        super().__init__(message)
        self.agent_name = agent_name
        self.details = details or {}


@dataclass
class AgentTask:
    """Task container for agent processing queue.

    Represents a unit of work to be processed by an agent, including
    the data to process, priority level, and execution constraints.

    Attributes:
        id: Unique identifier for the task.
        data: The payload data to be processed.
        priority: Processing priority level.
        metadata: Additional task metadata and context.
        created_at: When the task was created.
        timeout: Optional timeout in seconds for task processing.
    """

    id: str
    data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the Legal AI System.

    Provides comprehensive foundation for building specialized agents with
    consistent patterns for async processing, error handling, performance
    tracking, and service integration.

    Key features:
    - Async task processing with timeouts and retries
    - Standardized error handling and result formats
    - Performance statistics and monitoring
    - Service container integration for dependency injection
    - Task queuing with priority support
    - Comprehensive logging and debugging support

    Attributes:
        name: Human-readable name for the agent.
        status: Current execution status of the agent.
        task_queue: Async queue for pending tasks.
        current_task: Currently executing task, if any.
        stats: Performance statistics dictionary.
        config: Agent configuration loaded from ConfigurationManager.
        service_container: Service container for dependency access.

    Example:
        >>> class DocumentAgent(BaseAgent):
        ...     async def _process_task(self, data, metadata):
        ...         return {"processed": True}
        >>>
        >>> agent = DocumentAgent(service_container, "DocProcessor")
        >>> result = await agent.process({"text": "content"})
    """

    def __init__(
        self,
        service_container=None,
        name: Optional[str] = None,
        agent_type: str = "base",
    ) -> None:
        """Initialize agent with service container for dependency injection.

        Args:
            service_container: Service container providing access to managers
                and other system components. Can be None for testing.
            name: Human-readable name for the agent. Defaults to class name
                if not provided.
        """
        # Support both old 'services' parameter and new service_container approach
        if hasattr(service_container, "get_service"):
            self.service_container = service_container
            self.services = service_container  # Backward compatibility
        else:
            # Legacy support - treat as old services object
            self.services = service_container
            self.service_container = None

        self.name = name or self.__class__.__name__
        self.agent_type = agent_type
        # Detailed logger for the agent instance
        self.logger = get_detailed_logger(self.name, LogCategory.AGENT)
        self.status = AgentStatus.IDLE
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.current_task: Optional[AgentTask] = None

        # Performance tracking
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0,
            "last_execution": None,
        }

        # Configuration - now uses ConfigurationManager if available
        self.config = self._load_config()

        # Task processing flag
        self._processing = False
        self._shutdown = False

        self.logger.info("Initialized agent.")

    def _load_config(self) -> Dict[str, Any]:
        """Load agent-specific configuration from ConfigurationManager if available"""
        default_config = {
            "timeout": Constants.Time.DEFAULT_SERVICE_TIMEOUT_SECONDS,
            "max_retries": Constants.Performance.MAX_RETRY_ATTEMPTS,
            "retry_delay": Constants.Time.DEFAULT_RETRY_DELAY_SECONDS,
        }

        # Use ConfigurationManager if available
        if self.service_container:
            try:
                config_manager = self.service_container.get_service("config_manager")
                if config_manager:
                    # Get processing config that applies to agents
                    processing_config = config_manager.get_processing_config()

                    # Update with relevant settings
                    default_config.update(
                        {
                            "max_concurrent_documents": processing_config.get(
                                "max_concurrent_documents",
                                Constants.Performance.MAX_CONCURRENT_DOCUMENTS,
                            ),
                            "batch_size": processing_config.get(
                                "batch_size", Constants.Performance.DEFAULT_BATCH_SIZE
                            ),
                            "enable_auto_tagging": processing_config.get(
                                "enable_auto_tagging", True
                            ),
                            "auto_tag_confidence_threshold": processing_config.get(
                                "auto_tag_confidence_threshold",
                                Constants.Document.AUTO_TAG_CONFIDENCE_THRESHOLD,
                            ),
                        }
                    )

                    logger.info(
                        f"Loaded configuration from ConfigurationManager for {self.name}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load config from ConfigurationManager: {e}")

        return default_config

    # =================== SERVICE CONTAINER CONVENIENCE METHODS ===================

    def get_config_manager(self):
        """Get ConfigurationManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("config_manager")
            except Exception as e:
                logger.warning(f"Failed to get config_manager: {e}")
        return None

    def get_security_manager(self):
        """Get SecurityManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("security_manager")
            except Exception as e:
                logger.warning(f"Failed to get security_manager: {e}")
        return None

    def get_llm_manager(self):
        """Get LLMManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("llm_manager")
            except Exception as e:
                logger.warning(f"Failed to get llm_manager: {e}")
        return None

    def get_embedding_manager(self):
        """Get EmbeddingManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("embedding_manager")
            except Exception as e:
                logger.warning(f"Failed to get embedding_manager: {e}")
        return None

    def get_knowledge_graph_manager(self):
        """Get KnowledgeGraphManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("knowledge_graph_manager")
            except Exception as e:
                logger.warning(f"Failed to get knowledge_graph_manager: {e}")
        return None

    def get_vector_store_manager(self):
        """Get VectorStoreManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("vector_store_manager")
            except Exception as e:
                logger.warning(f"Failed to get vector_store_manager: {e}")
        return None

    def get_memory_manager(self):
        """Get UnifiedMemoryManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("unified_memory_manager")
            except Exception as e:
                logger.warning(f"Failed to get unified_memory_manager: {e}")
        return None

    def get_workflow_state_manager(self):
        """Get WorkflowStateManager from service container"""
        if self.service_container:
            try:
                return self.service_container.get_service("workflow_state_manager")
            except Exception as e:
                logger.warning(f"Failed to get workflow_state_manager: {e}")
        return None

    def _get_service(self, name: str) -> Any:
        """Generic helper to retrieve a service by name."""
        if self.service_container:
            try:
                return self.service_container.get_service(name)
            except Exception as e:
                self.logger.warning(
                    f"Service '{name}' unavailable", parameters={"error": str(e)}
                )
        return None

    # =================== AGENT PROCESSING METHODS ===================

    @abstractmethod
    async def _process_task(self, task_data: Any, metadata: Dict[str, Any]) -> Any:
        """
        Process a single task - must be implemented by subclasses

        Args:
            task_data: The data to process
            metadata: Additional metadata for the task

        Returns:
            The processed result

        Raises:
            AgentError: If processing fails
        """
        pass

    async def process(
        self,
        data: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AgentResult:
        """
        Process data with the agent

        Args:
            data: Data to process
            priority: Task priority
            metadata: Additional metadata
            timeout: Task timeout in seconds

        Returns:
            AgentResult with processing results
        """
        start_time = time.time()
        task_id = f"{self.name}_{int(start_time)}"

        try:
            self.status = AgentStatus.PROCESSING
            self.stats["total_tasks"] += 1

            # Create task
            task = AgentTask(
                id=task_id,
                data=data,
                priority=priority,
                metadata=metadata or {},
                timeout=timeout or self.config["timeout"],
            )

            self.current_task = task

            # Process the task
            result_data = await asyncio.wait_for(
                self._process_task_with_retries(task), timeout=task.timeout
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Update stats
            self.stats["successful_tasks"] += 1
            self._update_avg_execution_time(execution_time)
            self.stats["last_execution"] = datetime.now()

            # Create successful result
            result = AgentResult(
                success=True,
                data=result_data,
                metadata=task.metadata,
                execution_time=execution_time,
                agent_name=self.name,
            )

            self.status = AgentStatus.COMPLETED
            logger.info(
                f"Agent {self.name} completed task {task_id} in {execution_time:.2f}s"
            )

            return result

        except asyncio.TimeoutError:
            error_msg = (
                f"Task {task_id} timed out after {timeout or self.config['timeout']}s"
            )
            logger.error(f"Agent {self.name}: {error_msg}")
            self.stats["failed_tasks"] += 1
            self.status = AgentStatus.ERROR

            return AgentResult(
                success=False,
                error=error_msg,
                metadata=metadata or {},
                execution_time=time.time() - start_time,
                agent_name=self.name,
            )

        except Exception as e:
            error_msg = f"Task {task_id} failed: {str(e)}"
            logger.error(f"Agent {self.name}: {error_msg}", exc_info=True)
            self.stats["failed_tasks"] += 1
            self.status = AgentStatus.ERROR

            return AgentResult(
                success=False,
                error=error_msg,
                metadata=metadata or {},
                execution_time=time.time() - start_time,
                agent_name=self.name,
            )

        finally:
            self.current_task = None
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.IDLE

    async def _process_task_with_retries(self, task: AgentTask) -> Any:
        """Process task with retry logic"""
        max_retries = self.config["max_retries"]
        retry_delay = self.config["retry_delay"]

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self._process_task(task.data, task.metadata)

            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    logger.warning(
                        f"Agent {self.name} attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(
                        retry_delay * (attempt + 1)
                    )  # Exponential backoff
                else:
                    logger.error(
                        f"Agent {self.name} failed after {max_retries + 1} attempts"
                    )
                    raise AgentError(
                        f"All retry attempts failed. Last error: {e}", self.name
                    )

        # Should never reach here, but just in case
        if last_error is None:
            raise AgentError("Unknown processing error", self.name)
        raise last_error

    def _update_avg_execution_time(self, execution_time: float) -> None:
        """Update average execution time"""
        total_successful = self.stats["successful_tasks"]
        if total_successful == 1:
            self.stats["avg_execution_time"] = execution_time
        else:
            # Running average
            current_avg = self.stats["avg_execution_time"]
            self.stats["avg_execution_time"] = (
                (current_avg * (total_successful - 1)) + execution_time
            ) / total_successful

    # Queue-based processing methods
    async def add_task(
        self,
        data: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add task to processing queue"""
        task_id = f"{self.name}_{int(time.time())}"
        task = AgentTask(
            id=task_id, data=data, priority=priority, metadata=metadata or {}
        )

        await self.task_queue.put(task)
        logger.debug(f"Added task {task_id} to {self.name} queue")

        return task_id

    async def start_processing(self) -> None:
        """Start background task processing"""
        if self._processing:
            logger.warning(f"Agent {self.name} is already processing")
            return

        self._processing = True
        logger.info(f"Started background processing for agent {self.name}")

        while self._processing and not self._shutdown:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Process the task
                result = await self.process(
                    task.data, task.priority, task.metadata, task.timeout
                )

                # Mark task as done
                self.task_queue.task_done()

                # Log result
                if result.success:
                    logger.debug(f"Completed queued task {task.id}")
                else:
                    logger.error(f"Failed queued task {task.id}: {result.error}")

            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in background processing for {self.name}: {e}")

    async def stop_processing(self) -> None:
        """Stop background task processing"""
        self._processing = False
        logger.info(f"Stopped background processing for agent {self.name}")

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        self._shutdown = True
        await self.stop_processing()

        # Wait for current task to complete
        if self.current_task:
            logger.info(f"Waiting for current task to complete in {self.name}")
            # Give it a moment to finish
            await asyncio.sleep(1.0)

        logger.info(f"Agent {self.name} shut down")

    # Status and monitoring methods
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "status": self.status.value,
            "current_task": self.current_task.id if self.current_task else None,
            "queue_size": self.task_queue.qsize(),
            "stats": self.stats.copy(),
            "config": self.config.copy(),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        total_tasks = self.stats["total_tasks"]
        if total_tasks == 0:
            success_rate = 0.0
        else:
            success_rate = self.stats["successful_tasks"] / total_tasks

        return {
            "agent_name": self.name,
            "total_tasks": total_tasks,
            "successful_tasks": self.stats["successful_tasks"],
            "failed_tasks": self.stats["failed_tasks"],
            "success_rate": success_rate,
            "avg_execution_time": self.stats["avg_execution_time"],
            "last_execution": self.stats["last_execution"],
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Try to access services
            if self.services and hasattr(self.services, "llm_manager"):
                llm_health = await self.services.llm_manager.health_check()
            else:
                llm_health = {"status": "unknown"}

            return {
                "status": (
                    "healthy" if self.status != AgentStatus.ERROR else "unhealthy"
                ),
                "agent_name": self.name,
                "current_status": self.status.value,
                "queue_size": self.task_queue.qsize(),
                "services_available": bool(self.services),
                "llm_status": llm_health.get("status", "unknown"),
            }

        except Exception as e:
            return {"status": "unhealthy", "agent_name": self.name, "error": str(e)}

    # Utility methods for subclasses
    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """Utility method to call LLM through services"""
        if not self.services or not hasattr(self.services, "llm_manager"):
            raise AgentError("LLM service not available", self.name)

        try:
            response = await self.services.llm_manager.complete(prompt, **kwargs)
            return response.content
        except Exception as e:
            raise AgentError(f"LLM call failed: {e}", self.name)

    def _validate_input(
        self, data: Any, required_fields: Optional[List[str]] = None
    ) -> None:
        """Validate input data"""
        if data is None:
            raise AgentError("Input data cannot be None", self.name)

        if required_fields and isinstance(data, dict):
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise AgentError(
                    f"Missing required fields: {missing_fields}", self.name
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"
