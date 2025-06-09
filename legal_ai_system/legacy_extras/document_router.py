"""
Document Router - Intelligent Routing & Load Balancing
====================================================
Intelligent routing based on document characteristics with load balancing
and performance optimization for the Legal AI System.
"""

import asyncio
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

# Import detailed logging
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Initialize loggers
router_logger = get_detailed_logger("Document_Router", LogCategory.SYSTEM)
load_logger = get_detailed_logger("Load_Balancing", LogCategory.PERFORMANCE)
routing_logger = get_detailed_logger("Routing_Decisions", LogCategory.SYSTEM)

class DocumentSize(Enum):
    """Document size categories for routing decisions."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"

class DocumentComplexity(Enum):
    """Document complexity levels for processing strategy."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

class ProcessorType(Enum):
    """Available processor types with different capabilities."""
    FAST = "fast"           # Quick processing for simple documents
    STANDARD = "standard"   # Balanced processing
    HEAVY = "heavy"         # Deep analysis for complex documents
    SPECIALIZED = "specialized"  # Legal-specific processors

class Priority(Enum):
    """Processing priority levels."""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BATCH = "batch"

@dataclass
class ProcessorCapabilities:
    """Processor capabilities and limitations."""
    max_file_size_mb: float
    supported_formats: List[str]
    max_concurrent_jobs: int
    avg_processing_time_per_mb: float
    specializations: List[str]
    quality_score: float = 1.0

@dataclass
class LoadMetrics:
    """Current load metrics for a processor."""
    current_jobs: int = 0
    total_processed: int = 0
    avg_processing_time: float = 0.0
    success_rate: float = 1.0
    last_job_time: Optional[datetime] = None
    queue_length: int = 0

@dataclass
class RoutingDecision:
    """Details of a routing decision for analysis."""
    document_path: str
    document_size: DocumentSize
    complexity: DocumentComplexity
    priority: Priority
    selected_processor: str
    processor_type: ProcessorType
    routing_reason: str
    estimated_processing_time: float
    load_factor: float
    timestamp: datetime

class LoadTracker:
    """Tracks processor loads and performance metrics."""
    
    @detailed_log_function(LogCategory.PERFORMANCE)
    def __init__(self):
        """Initialize load tracking system."""
        self.processor_loads: Dict[str, LoadMetrics] = {}
        self.processor_queues: Dict[str, asyncio.Queue] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = asyncio.Lock()
        
        load_logger.info("Load tracker initialized")
    
    @detailed_log_function(LogCategory.PERFORMANCE)
    async def register_processor(self, processor_id: str, capabilities: ProcessorCapabilities):
        """Register a new processor for load tracking."""
        async with self._lock:
            self.processor_loads[processor_id] = LoadMetrics()
            self.processor_queues[processor_id] = asyncio.Queue(maxsize=capabilities.max_concurrent_jobs * 2)
            
            load_logger.info(f"Processor registered: {processor_id}", parameters={
                'max_concurrent_jobs': capabilities.max_concurrent_jobs,
                'supported_formats': capabilities.supported_formats,
                'specializations': capabilities.specializations
            })
    
    @detailed_log_function(LogCategory.PERFORMANCE)
    async def start_job(self, processor_id: str, document_path: str, estimated_time: float):
        """Record job start for load tracking."""
        async with self._lock:
            if processor_id in self.processor_loads:
                metrics = self.processor_loads[processor_id]
                metrics.current_jobs += 1
                metrics.last_job_time = datetime.now()
                
                # Add to queue
                if processor_id in self.processor_queues:
                    await self.processor_queues[processor_id].put({
                        'document': document_path,
                        'start_time': time.time(),
                        'estimated_time': estimated_time
                    })
                
                load_logger.trace(f"Job started on {processor_id}", parameters={
                    'document': document_path,
                    'current_jobs': metrics.current_jobs,
                    'estimated_time': estimated_time
                })
    
    @detailed_log_function(LogCategory.PERFORMANCE)
    async def complete_job(self, processor_id: str, document_path: str, success: bool, actual_time: float):
        """Record job completion for performance tracking."""
        async with self._lock:
            if processor_id in self.processor_loads:
                metrics = self.processor_loads[processor_id]
                metrics.current_jobs = max(0, metrics.current_jobs - 1)
                metrics.total_processed += 1
                
                # Update average processing time
                if metrics.avg_processing_time == 0:
                    metrics.avg_processing_time = actual_time
                else:
                    metrics.avg_processing_time = (metrics.avg_processing_time + actual_time) / 2
                
                # Update success rate
                if success:
                    metrics.success_rate = (metrics.success_rate * (metrics.total_processed - 1) + 1.0) / metrics.total_processed
                else:
                    metrics.success_rate = (metrics.success_rate * (metrics.total_processed - 1)) / metrics.total_processed
                
                # Store in performance history
                self.performance_history[processor_id].append({
                    'document': document_path,
                    'success': success,
                    'processing_time': actual_time,
                    'timestamp': datetime.now()
                })
                
                load_logger.info(f"Job completed on {processor_id}", parameters={
                    'document': document_path,
                    'success': success,
                    'actual_time': actual_time,
                    'current_jobs': metrics.current_jobs,
                    'success_rate': metrics.success_rate
                })
    
    def get_load_factor(self, processor_id: str) -> float:
        """Calculate current load factor for processor (0.0 = idle, 1.0 = full)."""
        if processor_id not in self.processor_loads:
            return 0.0
        
        metrics = self.processor_loads[processor_id]
        if processor_id not in self.processor_queues:
            return 0.0
        
        queue = self.processor_queues[processor_id]
        max_jobs = queue.maxsize // 2  # Half of queue size is considered "full load"
        
        load_factor = metrics.current_jobs / max(1, max_jobs)
        return min(1.0, load_factor)
    
    def get_processor_score(self, processor_id: str) -> float:
        """Get overall processor performance score."""
        if processor_id not in self.processor_loads:
            return 0.0
        
        metrics = self.processor_loads[processor_id]
        load_factor = self.get_load_factor(processor_id)
        
        # Score based on success rate, inverse load factor, and processing speed
        speed_score = 1.0 / max(0.1, metrics.avg_processing_time) if metrics.avg_processing_time > 0 else 1.0
        load_score = 1.0 - load_factor
        success_score = metrics.success_rate
        
        overall_score = (success_score * 0.4) + (load_score * 0.4) + (min(1.0, speed_score) * 0.2)
        return overall_score

class DocumentRouter:
    """
    Intelligent document routing with load balancing and performance optimization.
    
    Features:
    - Size-based routing (small/medium/large/xlarge documents)
    - Complexity assessment for optimal processor selection
    - Priority-based SLA management
    - Load balancing across processor pools
    - Performance tracking and optimization
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self, service_config: Optional[Dict[str, Any]] = None):
        """Initialize document router with intelligent routing capabilities."""
        router_logger.info("=== INITIALIZING DOCUMENT ROUTER ===")
        
        self.config = service_config or {}
        self.load_tracker = LoadTracker()
        self.processor_pools: Dict[ProcessorType, List[str]] = defaultdict(list)
        self.processor_capabilities: Dict[str, ProcessorCapabilities] = {}
        self.routing_rules = self._initialize_routing_rules()
        self.routing_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.total_routed = 0
        self.routing_times: deque = deque(maxlen=1000)
        
        router_logger.info("Document router initialization complete", parameters={
            'routing_rules_count': len(self.routing_rules),
            'processor_pools': list(self.processor_pools.keys())
        })
    
    @detailed_log_function(LogCategory.SYSTEM)
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Initialize routing rules based on document characteristics."""
        routing_logger.info("Initializing routing rules")
        
        rules = {
            "size_based": {
                "small": {
                    "max_size_mb": 1.0,
                    "preferred_processors": ["fast", "standard"],
                    "max_processing_time": 30
                },
                "medium": {
                    "max_size_mb": 10.0,
                    "preferred_processors": ["standard", "heavy"],
                    "max_processing_time": 120
                },
                "large": {
                    "max_size_mb": 100.0,
                    "preferred_processors": ["heavy", "specialized"],
                    "max_processing_time": 600
                },
                "xlarge": {
                    "max_size_mb": float('inf'),
                    "preferred_processors": ["specialized"],
                    "max_processing_time": 3600
                }
            },
            "complexity_based": {
                "simple": {
                    "indicators": ["memo", "letter", "email", "note"],
                    "preferred_processor": "fast",
                    "confidence_threshold": 0.6
                },
                "moderate": {
                    "indicators": ["contract", "agreement", "policy", "report"],
                    "preferred_processor": "standard",
                    "confidence_threshold": 0.7
                },
                "complex": {
                    "indicators": ["brief", "statute", "regulation", "case"],
                    "preferred_processor": "heavy",
                    "confidence_threshold": 0.8
                },
                "highly_complex": {
                    "indicators": ["constitutional", "appellate", "supreme_court"],
                    "preferred_processor": "specialized",
                    "confidence_threshold": 0.9
                }
            },
            "priority_based": {
                "urgent": {"sla_minutes": 5, "dedicated_workers": 2, "bypass_queue": True},
                "high": {"sla_minutes": 15, "dedicated_workers": 1, "bypass_queue": False},
                "normal": {"sla_minutes": 60, "dedicated_workers": 0, "bypass_queue": False},
                "low": {"sla_minutes": 240, "dedicated_workers": 0, "bypass_queue": False},
                "batch": {"sla_minutes": 1440, "dedicated_workers": 0, "bypass_queue": False}
            }
        }
        
        routing_logger.info("Routing rules initialized", parameters={
            'size_categories': len(rules['size_based']),
            'complexity_levels': len(rules['complexity_based']),
            'priority_levels': len(rules['priority_based'])
        })
        
        return rules
    
    @detailed_log_function(LogCategory.SYSTEM)
    async def register_processor(self, processor_id: str, processor_type: ProcessorType, 
                                capabilities: ProcessorCapabilities):
        """Register a processor with specific capabilities."""
        router_logger.info(f"Registering processor: {processor_id}", parameters={
            'processor_type': processor_type.value,
            'max_file_size_mb': capabilities.max_file_size_mb,
            'supported_formats': capabilities.supported_formats
        })
        
        # Store capabilities
        self.processor_capabilities[processor_id] = capabilities
        
        # Add to appropriate processor pool
        self.processor_pools[processor_type].append(processor_id)
        
        # Register with load tracker
        await self.load_tracker.register_processor(processor_id, capabilities)
        
        router_logger.info(f"Processor {processor_id} registered successfully")
    
    @detailed_log_function(LogCategory.SYSTEM)
    async def route_document(self, document_path: Path, metadata: Dict[str, Any]) -> Tuple[str, RoutingDecision]:
        """Route document to optimal processor based on characteristics."""
        start_time = time.time()
        
        routing_logger.info(f"Routing document: {document_path.name}")
        
        # Analyze document characteristics
        size_category = self._categorize_by_size(document_path)
        complexity = await self._assess_complexity(document_path, metadata)
        priority = Priority(metadata.get("priority", "normal"))
        
        # Select optimal processor
        processor_id, processor_type, routing_reason = await self._select_optimal_processor(
            size_category, complexity, priority, document_path
        )
        
        # Estimate processing time
        estimated_time = self._estimate_processing_time(document_path, processor_id, complexity)
        
        # Get load factor
        load_factor = self.load_tracker.get_load_factor(processor_id)
        
        # Create routing decision
        decision = RoutingDecision(
            document_path=str(document_path),
            document_size=size_category,
            complexity=complexity,
            priority=priority,
            selected_processor=processor_id,
            processor_type=processor_type,
            routing_reason=routing_reason,
            estimated_processing_time=estimated_time,
            load_factor=load_factor,
            timestamp=datetime.now()
        )
        
        # Record routing decision
        self.routing_history.append(decision)
        self.total_routed += 1
        
        routing_time = time.time() - start_time
        self.routing_times.append(routing_time)
        
        # Start job tracking
        await self.load_tracker.start_job(processor_id, str(document_path), estimated_time)
        
        routing_logger.info(f"Document routed to {processor_id}", parameters={
            'processor_type': processor_type.value,
            'size_category': size_category.value,
            'complexity': complexity.value,
            'priority': priority.value,
            'estimated_time': estimated_time,
            'load_factor': load_factor,
            'routing_time': routing_time,
            'routing_reason': routing_reason
        })
        
        return processor_id, decision
    
    def _categorize_by_size(self, document_path: Path) -> DocumentSize:
        """Categorize document by file size."""
        try:
            size_mb = document_path.stat().st_size / (1024 * 1024)
            
            if size_mb <= 1.0:
                return DocumentSize.SMALL
            elif size_mb <= 10.0:
                return DocumentSize.MEDIUM
            elif size_mb <= 100.0:
                return DocumentSize.LARGE
            else:
                return DocumentSize.XLARGE
                
        except Exception as e:
            router_logger.warning(f"Could not determine size for {document_path}", exception=e)
            return DocumentSize.MEDIUM  # Default
    
    async def _assess_complexity(self, document_path: Path, metadata: Dict[str, Any]) -> DocumentComplexity:
        """Assess document complexity based on content and metadata."""
        # Check metadata for complexity hints
        if "document_type" in metadata:
            doc_type = metadata["document_type"].lower()
            
            # Check against complexity indicators
            for complexity_str, rules in self.routing_rules["complexity_based"].items():
                if any(indicator in doc_type for indicator in rules["indicators"]):
                    complexity = DocumentComplexity(complexity_str)
                    routing_logger.trace(f"Complexity assessed from metadata: {complexity.value}")
                    return complexity
        
        # Check filename for complexity hints
        filename_lower = document_path.name.lower()
        for complexity_str, rules in self.routing_rules["complexity_based"].items():
            if any(indicator in filename_lower for indicator in rules["indicators"]):
                complexity = DocumentComplexity(complexity_str)
                routing_logger.trace(f"Complexity assessed from filename: {complexity.value}")
                return complexity
        
        # Default based on file type
        if document_path.suffix.lower() in ['.txt', '.md', '.email']:
            return DocumentComplexity.SIMPLE
        elif document_path.suffix.lower() in ['.pdf', '.docx', '.doc']:
            return DocumentComplexity.MODERATE
        else:
            return DocumentComplexity.MODERATE
    
    async def _select_optimal_processor(self, size: DocumentSize, complexity: DocumentComplexity, 
                                       priority: Priority, document_path: Path) -> Tuple[str, ProcessorType, str]:
        """Select the optimal processor based on all factors."""
        
        # Get preferred processor types for this size
        size_rules = self.routing_rules["size_based"][size.value]
        preferred_types = size_rules["preferred_processors"]
        
        # Get preferred type for complexity
        complexity_rules = self.routing_rules["complexity_based"][complexity.value]
        complexity_preferred = complexity_rules["preferred_processor"]
        
        # Combine preferences (complexity takes precedence)
        if complexity_preferred in preferred_types:
            candidate_types = [complexity_preferred]
        else:
            candidate_types = preferred_types
        
        # Find available processors of preferred types
        available_processors = []
        for proc_type_str in candidate_types:
            # Convert string to ProcessorType enum
            try:
                proc_type = ProcessorType(proc_type_str)
            except ValueError:
                continue
                
            for processor_id in self.processor_pools.get(proc_type, []):
                # Check if processor can handle this document
                capabilities = self.processor_capabilities[processor_id]
                file_size_mb = document_path.stat().st_size / (1024 * 1024)
                
                if (file_size_mb <= capabilities.max_file_size_mb and
                    document_path.suffix.lower() in [fmt.lower() for fmt in capabilities.supported_formats]):
                    available_processors.append((processor_id, proc_type))
        
        if not available_processors:
            raise ValueError(f"No available processors for document: {document_path}")
        
        # Select best processor based on current load and performance
        best_processor = None
        best_score = -1
        
        for processor_id, proc_type in available_processors:
            score = self.load_tracker.get_processor_score(processor_id)
            
            # Priority bonus
            if priority == Priority.URGENT:
                score += 0.2
            elif priority == Priority.HIGH:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_processor = (processor_id, proc_type)
        
        if not best_processor:
            # Fallback to first available
            best_processor = available_processors[0]
        
        processor_id, processor_type = best_processor
        
        # Generate routing reason
        routing_reason = f"Selected {processor_type.value} processor for {complexity.value} {size.value} document (score: {best_score:.2f})"
        
        return processor_id, processor_type, routing_reason
    
    def _estimate_processing_time(self, document_path: Path, processor_id: str, 
                                 complexity: DocumentComplexity) -> float:
        """Estimate processing time for document."""
        if processor_id not in self.processor_capabilities:
            return 60.0  # Default 1 minute
        
        capabilities = self.processor_capabilities[processor_id]
        file_size_mb = document_path.stat().st_size / (1024 * 1024)
        
        # Base time from processor capabilities
        base_time = file_size_mb * capabilities.avg_processing_time_per_mb
        
        # Complexity multiplier
        complexity_multipliers = {
            DocumentComplexity.SIMPLE: 0.5,
            DocumentComplexity.MODERATE: 1.0,
            DocumentComplexity.COMPLEX: 2.0,
            DocumentComplexity.HIGHLY_COMPLEX: 4.0
        }
        
        estimated_time = base_time * complexity_multipliers.get(complexity, 1.0)
        
        # Add some buffer
        return max(5.0, estimated_time * 1.2)
    
    @detailed_log_function(LogCategory.PERFORMANCE)
    async def complete_job(self, processor_id: str, document_path: str, success: bool, actual_time: float):
        """Record job completion for performance tracking."""
        await self.load_tracker.complete_job(processor_id, document_path, success, actual_time)
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        stats = {
            'total_routed': self.total_routed,
            'avg_routing_time': sum(self.routing_times) / len(self.routing_times) if self.routing_times else 0,
            'processor_pools': {
                proc_type.value: len(processors) 
                for proc_type, processors in self.processor_pools.items()
            },
            'load_factors': {
                processor_id: self.load_tracker.get_load_factor(processor_id)
                for processor_id in self.processor_capabilities.keys()
            },
            'performance_scores': {
                processor_id: self.load_tracker.get_processor_score(processor_id)
                for processor_id in self.processor_capabilities.keys()
            }
        }
        
        # Recent routing decisions
        if self.routing_history:
            recent_decisions = list(self.routing_history)[-10:]
            stats['recent_decisions'] = [
                {
                    'document': Path(d.document_path).name,
                    'processor': d.selected_processor,
                    'processor_type': d.processor_type.value,
                    'complexity': d.complexity.value,
                    'estimated_time': d.estimated_processing_time,
                    'routing_reason': d.routing_reason
                }
                for d in recent_decisions
            ]
        
        router_logger.info("Routing statistics generated", parameters=stats)
        return stats
    
    async def initialize(self):
        """Async initialization for service container compatibility."""
        router_logger.info("Document router async initialization complete")
        return self
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for service container monitoring."""
        return {
            'healthy': True,
            'total_processors': sum(len(processors) for processors in self.processor_pools.values()),
            'total_routed': self.total_routed,
            'avg_routing_time': sum(self.routing_times) / len(self.routing_times) if self.routing_times else 0,
            'processor_pools': {
                proc_type.value: len(processors) 
                for proc_type, processors in self.processor_pools.items()
            }
        }

# Service container factory function
def create_document_router(config: Optional[Dict[str, Any]] = None) -> DocumentRouter:
    """Factory function for service container integration."""
    return DocumentRouter(service_config=config or {})