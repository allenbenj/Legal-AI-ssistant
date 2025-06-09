"""
ML Optimizer - Machine Learning Parameter Optimization
=====================================================
ML-powered optimization for processing parameters with intelligent
parameter tuning and document similarity analysis.
"""

import json
from pathlib import Path

# Import numpy with fallback
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    # Fallback numpy-like functionality for basic operations
    class FakeLinalg:
        """Minimal linalg module with norm function."""

        @staticmethod
        def norm(vec):
            if isinstance(vec, list):
                return (sum(x * x for x in vec)) ** 0.5
            return 1.0

    class FakeNumPy:
        """Simplified NumPy replacement used when real NumPy is unavailable."""

        linalg = FakeLinalg()

        @staticmethod
        def array(data):
            return data

        @staticmethod
        def dot(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return sum(x * y for x, y in zip(a, b))
            return 0

        @staticmethod
        def var(data):
            if not data:
                return 0.0
            mean = sum(data) / len(data)
            return sum((x - mean) ** 2 for x in data) / len(data)

    np = FakeNumPy()
    HAS_NUMPY = False
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import sqlite3
import threading
import hashlib

# Import detailed logging
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Initialize loggers
ml_logger = get_detailed_logger("ML_Optimizer", LogCategory.SYSTEM)
performance_logger = get_detailed_logger("ML_Performance", LogCategory.PERFORMANCE)
optimization_logger = get_detailed_logger("Parameter_Optimization", LogCategory.SYSTEM)


class OptimizationObjective(Enum):
    """Optimization objectives for parameter tuning."""

    SPEED = "speed"  # Minimize processing time
    ACCURACY = "accuracy"  # Maximize extraction accuracy
    COST = "cost"  # Minimize API costs
    BALANCED = "balanced"  # Balance speed, accuracy, and cost
    MEMORY = "memory"  # Minimize memory usage


class DocumentCategory(Enum):
    """Document categories for targeted optimization."""

    LEGAL_BRIEF = "legal_brief"
    CONTRACT = "contract"
    STATUTE = "statute"
    CASE_LAW = "case_law"
    REGULATION = "regulation"
    MEMO = "memo"
    EMAIL = "email"
    GENERIC = "generic"


@dataclass
class ProcessingParameters:
    """Processing parameters that can be optimized."""

    chunk_size: int = 3000
    chunk_overlap: int = 200
    temperature: float = 0.1
    max_tokens: int = 2000
    confidence_threshold: float = 0.7
    model_name: str = "gpt-4"
    batch_size: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    use_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingParameters":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization analysis."""

    processing_time: float
    accuracy_score: float = 0.0
    f1_score: float = 0.0
    entities_extracted: int = 0
    api_cost: float = 0.0
    memory_usage_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def get_composite_score(self, objective: OptimizationObjective) -> float:
        """Calculate composite score based on optimization objective."""
        if objective == OptimizationObjective.SPEED:
            return 1.0 / max(0.1, self.processing_time)
        elif objective == OptimizationObjective.ACCURACY:
            return (self.accuracy_score + self.f1_score) / 2
        elif objective == OptimizationObjective.COST:
            return 1.0 / max(0.01, self.api_cost)
        elif objective == OptimizationObjective.MEMORY:
            return 1.0 / max(1.0, self.memory_usage_mb)
        elif objective == OptimizationObjective.BALANCED:
            speed_score = 1.0 / max(0.1, self.processing_time)
            accuracy_score = (self.accuracy_score + self.f1_score) / 2
            cost_score = 1.0 / max(0.01, self.api_cost)
            return speed_score * 0.4 + accuracy_score * 0.4 + cost_score * 0.2
        else:
            return 0.0


@dataclass
class DocumentFeatures:
    """Document features for similarity analysis."""

    file_size_mb: float
    page_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    table_count: int = 0
    image_count: int = 0
    complexity_score: float = 0.0
    language: str = "en"
    has_legal_terminology: bool = False

    def to_vector(self):
        """Convert features to vector for similarity calculation."""
        vector_data = [
            self.file_size_mb,
            self.page_count,
            self.word_count / 1000.0,  # Normalize
            self.sentence_count / 100.0,  # Normalize
            self.paragraph_count / 10.0,  # Normalize
            self.table_count,
            self.image_count,
            self.complexity_score,
            1.0 if self.has_legal_terminology else 0.0,
        ]
        return np.array(vector_data)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    optimized_parameters: ProcessingParameters
    expected_improvement: float
    confidence: float
    optimization_reason: str
    based_on_samples: int


class MLOptimizer:
    """
    Machine learning optimizer for processing parameters.

    Features:
    - Performance history tracking with SQLite storage
    - Document similarity analysis using feature vectors
    - Parameter optimization using statistical analysis
    - Multi-objective optimization (speed, accuracy, cost, etc.)
    - Automated parameter suggestions with confidence scores
    """

    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(
        self,
        storage_dir: str = "./storage/databases",
        service_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ML optimizer with performance tracking."""
        ml_logger.info("=== INITIALIZING ML OPTIMIZER ===")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.config = service_config or {}
        self.db_path = self.storage_dir / "ml_optimizer.db"

        # Initialize database
        self._init_database()

        # In-memory caches
        self.performance_history: deque = deque(maxlen=10000)
        self.document_features_cache: Dict[str, DocumentFeatures] = {}
        self.optimization_cache: Dict[str, OptimizationResult] = {}

        # Analysis settings
        self.min_samples_for_optimization = self.config.get("min_samples", 50)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.max_optimization_age_hours = self.config.get(
            "max_optimization_age_hours", 24
        )

        # Load recent performance data
        self._load_recent_performance()

        # Threading
        self._lock = threading.RLock()

        ml_logger.info(
            "ML optimizer initialization complete",
            parameters={
                "db_path": str(self.db_path),
                "min_samples": self.min_samples_for_optimization,
                "similarity_threshold": self.similarity_threshold,
                "performance_history_size": len(self.performance_history),
            },
        )

    @detailed_log_function(LogCategory.SYSTEM)
    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS performance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_path TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_perf_type ON performance_records(document_type);
                CREATE INDEX IF NOT EXISTS idx_perf_hash ON performance_records(document_hash);
                CREATE INDEX IF NOT EXISTS idx_perf_score ON performance_records(composite_score);
                CREATE INDEX IF NOT EXISTS idx_perf_created ON performance_records(created_at);
                
                CREATE TABLE IF NOT EXISTS optimization_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    parameters_json TEXT NOT NULL,
                    expected_improvement REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    samples_count INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_cache_key ON optimization_cache(cache_key);
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON optimization_cache(expires_at);
            """
            )

        ml_logger.info("ML optimizer database initialized")

    @detailed_log_function(LogCategory.SYSTEM)
    def _load_recent_performance(self):
        """Load recent performance data into memory cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT document_path, document_type, parameters_json, metrics_json, 
                           features_json, objective, composite_score, created_at
                    FROM performance_records 
                    WHERE created_at > datetime('now', '-7 days')
                    ORDER BY created_at DESC
                    LIMIT 5000
                """
                )

                for row in cursor.fetchall():
                    record = {
                        "document_path": row[0],
                        "document_type": row[1],
                        "parameters": json.loads(row[2]),
                        "metrics": json.loads(row[3]),
                        "features": json.loads(row[4]),
                        "objective": row[5],
                        "composite_score": row[6],
                        "timestamp": datetime.fromisoformat(row[7]),
                    }
                    self.performance_history.append(record)

            ml_logger.info(
                f"Loaded {len(self.performance_history)} recent performance records"
            )

        except Exception as e:
            ml_logger.error("Failed to load recent performance data", exception=e)

    @detailed_log_function(LogCategory.PERFORMANCE)
    def record_performance(
        self,
        document_path: str,
        document_type: DocumentCategory,
        parameters: ProcessingParameters,
        metrics: PerformanceMetrics,
        features: DocumentFeatures,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
    ):
        """Record processing performance for ML training."""

        performance_logger.info(f"Recording performance for {document_path}")

        try:
            with self._lock:
                # Calculate document hash for similarity tracking
                doc_hash = self._calculate_document_hash(document_path, features)

                # Calculate composite score
                composite_score = metrics.get_composite_score(objective)

                # Create record
                record = {
                    "document_path": document_path,
                    "document_type": document_type.value,
                    "document_hash": doc_hash,
                    "parameters": parameters.to_dict(),
                    "metrics": asdict(metrics),
                    "features": asdict(features),
                    "objective": objective.value,
                    "composite_score": composite_score,
                    "timestamp": datetime.now(),
                }

                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO performance_records 
                        (document_path, document_type, document_hash, parameters_json, 
                         metrics_json, features_json, objective, composite_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            document_path,
                            document_type.value,
                            doc_hash,
                            json.dumps(parameters.to_dict()),
                            json.dumps(asdict(metrics)),
                            json.dumps(asdict(features)),
                            objective.value,
                            composite_score,
                        ),
                    )

                # Add to memory cache
                self.performance_history.append(record)

                # Cache document features
                self.document_features_cache[doc_hash] = features

                # Invalidate optimization cache for this document type
                cache_key = f"{document_type.value}_{objective.value}"
                if cache_key in self.optimization_cache:
                    del self.optimization_cache[cache_key]

                performance_logger.info(
                    f"Performance recorded",
                    parameters={
                        "document_type": document_type.value,
                        "composite_score": composite_score,
                        "processing_time": metrics.processing_time,
                        "accuracy_score": metrics.accuracy_score,
                    },
                )

        except Exception as e:
            performance_logger.error(
                f"Failed to record performance for {document_path}", exception=e
            )

    @detailed_log_function(LogCategory.SYSTEM)
    async def get_optimal_parameters(
        self,
        document_type: DocumentCategory,
        document_features: DocumentFeatures,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        force_refresh: bool = False,
    ) -> OptimizationResult:
        """Get ML-optimized processing parameters for document."""

        optimization_logger.info(
            f"Getting optimal parameters for {document_type.value}"
        )

        # Check cache first
        cache_key = self._generate_cache_key(
            document_type, document_features, objective
        )

        if not force_refresh and cache_key in self.optimization_cache:
            cached_result = self.optimization_cache[cache_key]
            optimization_logger.info("Returning cached optimization result")
            return cached_result

        # Check database cache
        if not force_refresh:
            db_result = self._get_cached_optimization(cache_key)
            if db_result:
                self.optimization_cache[cache_key] = db_result
                optimization_logger.info(
                    "Returning database cached optimization result"
                )
                return db_result

        # Find similar documents
        similar_docs = self._find_similar_documents(
            document_type, document_features, objective
        )

        if len(similar_docs) < self.min_samples_for_optimization:
            optimization_logger.warning(
                f"Insufficient samples for optimization: {len(similar_docs)} < {self.min_samples_for_optimization}"
            )
            return self._get_default_optimization_result(document_type, objective)

        # Analyze best performing parameters
        optimal_params, improvement, confidence, reason = self._optimize_parameters(
            similar_docs, objective
        )

        # Create optimization result
        result = OptimizationResult(
            optimized_parameters=optimal_params,
            expected_improvement=improvement,
            confidence=confidence,
            optimization_reason=reason,
            based_on_samples=len(similar_docs),
        )

        # Cache result
        self.optimization_cache[cache_key] = result
        self._cache_optimization_result(cache_key, result)

        optimization_logger.info(
            f"Optimization complete",
            parameters={
                "document_type": document_type.value,
                "samples_used": len(similar_docs),
                "expected_improvement": improvement,
                "confidence": confidence,
            },
        )

        return result

    def _calculate_document_hash(
        self, document_path: str, features: DocumentFeatures
    ) -> str:
        """Calculate hash for document similarity tracking."""
        content = f"{Path(document_path).name}_{features.file_size_mb}_{features.word_count}_{features.complexity_score}"
        return hashlib.md5(content.encode()).hexdigest()

    def _find_similar_documents(
        self,
        document_type: DocumentCategory,
        document_features: DocumentFeatures,
        objective: OptimizationObjective,
    ) -> List[Dict[str, Any]]:
        """Find similar documents based on features and type."""

        target_vector = document_features.to_vector()
        similar_docs = []

        for record in self.performance_history:
            # Filter by document type and objective
            if (
                record["document_type"] != document_type.value
                or record["objective"] != objective.value
            ):
                continue

            # Calculate similarity
            record_features = DocumentFeatures(**record["features"])
            record_vector = record_features.to_vector()

            # Cosine similarity
            similarity = self._cosine_similarity(target_vector, record_vector)

            if similarity >= self.similarity_threshold:
                record["similarity"] = similarity
                similar_docs.append(record)

        # Sort by similarity and composite score
        similar_docs.sort(
            key=lambda x: (x["similarity"], x["composite_score"]), reverse=True
        )

        optimization_logger.trace(
            f"Found {len(similar_docs)} similar documents",
            parameters={
                "document_type": document_type.value,
                "similarity_threshold": self.similarity_threshold,
            },
        )

        return similar_docs

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception:
            return 0.0

    def _optimize_parameters(
        self,
        similar_docs: List[Dict[str, Any]],
        objective: OptimizationObjective,
    ) -> Tuple[ProcessingParameters, float, float, str]:
        """Optimize parameters based on similar documents performance."""

        _ = objective  # reserved for future objective-specific logic

        # Sort by performance score
        sorted_docs = sorted(
            similar_docs, key=lambda x: x["composite_score"], reverse=True
        )

        # Take top 25% performers
        top_performers = sorted_docs[: max(1, len(sorted_docs) // 4)]

        # Calculate average parameters from top performers
        param_sums = defaultdict(float)
        param_counts = defaultdict(int)

        for doc in top_performers:
            params = doc["parameters"]
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    param_sums[key] += value
                    param_counts[key] += 1

        # Calculate optimized parameters
        optimized_params_dict = {}
        for key in param_sums:
            if param_counts[key] > 0:
                optimized_params_dict[key] = param_sums[key] / param_counts[key]

        # Handle non-numeric parameters (use mode)
        string_params = defaultdict(list)
        for doc in top_performers:
            params = doc["parameters"]
            for key, value in params.items():
                if isinstance(value, (str, bool)):
                    string_params[key].append(value)

        for key, values in string_params.items():
            if values:
                # Use most common value
                optimized_params_dict[key] = max(set(values), key=values.count)

        # Create ProcessingParameters object
        optimal_params = ProcessingParameters()
        for key, value in optimized_params_dict.items():
            if hasattr(optimal_params, key):
                setattr(optimal_params, key, value)

        # Calculate expected improvement
        top_score = top_performers[0]["composite_score"]
        avg_score = sum(doc["composite_score"] for doc in similar_docs) / len(
            similar_docs
        )
        expected_improvement = float((top_score - avg_score) / max(0.01, avg_score))

        # Calculate confidence based on sample size and score variance
        score_variance = float(np.var([doc["composite_score"] for doc in similar_docs]))
        sample_confidence = min(1.0, len(similar_docs) / 100.0)
        variance_confidence = 1.0 / (1.0 + score_variance)
        confidence = (sample_confidence + variance_confidence) / 2

        reason = f"Optimized based on top {len(top_performers)} performers from {len(similar_docs)} similar documents"

        return optimal_params, expected_improvement, confidence, reason

    def _generate_cache_key(
        self,
        document_type: DocumentCategory,
        document_features: DocumentFeatures,
        objective: OptimizationObjective,
    ) -> str:
        """Generate cache key for optimization results."""
        feature_hash = hashlib.md5(
            str(document_features.to_vector()).encode()
        ).hexdigest()[:8]
        return f"{document_type.value}_{objective.value}_{feature_hash}"

    def _get_cached_optimization(self, cache_key: str) -> Optional[OptimizationResult]:
        """Get optimization result from database cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT parameters_json, expected_improvement, confidence, reason, samples_count
                    FROM optimization_cache 
                    WHERE cache_key = ? AND expires_at > datetime('now')
                """,
                    (cache_key,),
                )

                row = cursor.fetchone()
                if row:
                    return OptimizationResult(
                        optimized_parameters=ProcessingParameters.from_dict(
                            json.loads(row[0])
                        ),
                        expected_improvement=row[1],
                        confidence=row[2],
                        optimization_reason=row[3],
                        based_on_samples=row[4],
                    )
        except Exception as e:
            optimization_logger.error("Failed to get cached optimization", exception=e)

        return None

    def _cache_optimization_result(self, cache_key: str, result: OptimizationResult):
        """Cache optimization result in database."""
        try:
            expires_at = datetime.now() + timedelta(
                hours=self.max_optimization_age_hours
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO optimization_cache 
                    (cache_key, parameters_json, expected_improvement, confidence, reason, samples_count, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        cache_key,
                        json.dumps(result.optimized_parameters.to_dict()),
                        result.expected_improvement,
                        result.confidence,
                        result.optimization_reason,
                        result.based_on_samples,
                        expires_at.isoformat(),
                    ),
                )
        except Exception as e:
            optimization_logger.error(
                "Failed to cache optimization result", exception=e
            )

    def _get_default_optimization_result(
        self, document_type: DocumentCategory, objective: OptimizationObjective
    ) -> OptimizationResult:
        """Get default optimization result when insufficient data."""
        default_params = ProcessingParameters()

        # Adjust defaults based on document type
        if document_type == DocumentCategory.LEGAL_BRIEF:
            default_params.chunk_size = 4000
            default_params.confidence_threshold = 0.8
        elif document_type == DocumentCategory.CONTRACT:
            default_params.chunk_size = 3500
            default_params.confidence_threshold = 0.75
        elif document_type == DocumentCategory.MEMO:
            default_params.chunk_size = 2000
            default_params.temperature = 0.2

        # Adjust for objective
        if objective == OptimizationObjective.SPEED:
            default_params.chunk_size = 2000
            default_params.max_tokens = 1500
        elif objective == OptimizationObjective.ACCURACY:
            default_params.chunk_size = 4000
            default_params.confidence_threshold = 0.9

        return OptimizationResult(
            optimized_parameters=default_params,
            expected_improvement=0.0,
            confidence=0.1,
            optimization_reason="Default parameters - insufficient training data",
            based_on_samples=0,
        )

    @detailed_log_function(LogCategory.SYSTEM)
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "total_performance_records": len(self.performance_history),
            "cache_size": len(self.optimization_cache),
            "document_types": {},
            "objectives": {},
            "avg_composite_scores": {},
            "parameter_distributions": {},
        }

        # Analyze by document type
        type_counts = defaultdict(int)
        objective_counts = defaultdict(int)
        type_scores = defaultdict(list)

        for record in self.performance_history:
            doc_type = record["document_type"]
            objective = record["objective"]
            score = record["composite_score"]

            type_counts[doc_type] += 1
            objective_counts[objective] += 1
            type_scores[doc_type].append(score)

        stats["document_types"] = dict(type_counts)
        stats["objectives"] = dict(objective_counts)
        stats["avg_composite_scores"] = {
            doc_type: sum(scores) / len(scores) if scores else 0
            for doc_type, scores in type_scores.items()
        }

        # Parameter analysis
        if self.performance_history:
            recent_params = [
                record["parameters"] for record in list(self.performance_history)[-100:]
            ]

            for param_name in ["chunk_size", "temperature", "confidence_threshold"]:
                values = [
                    p.get(param_name)
                    for p in recent_params
                    if p.get(param_name) is not None
                ]
                if values:
                    stats["parameter_distributions"][param_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "samples": len(values),
                    }

        ml_logger.info("Optimization statistics generated", parameters=stats)
        return stats

    async def initialize(self):
        """Async initialization for service container compatibility."""
        ml_logger.info("ML optimizer async initialization complete")
        return self

    def health_check(self) -> Dict[str, Any]:
        """Health check for service container monitoring."""
        return {
            "healthy": True,
            "performance_records": len(self.performance_history),
            "optimization_cache_size": len(self.optimization_cache),
            "database_path": str(self.db_path),
            "min_samples_threshold": self.min_samples_for_optimization,
        }


# Service container factory function
def create_ml_optimizer(config: Optional[Dict[str, Any]] = None) -> MLOptimizer:
    """Factory function for service container integration."""
    return MLOptimizer(service_config=config or {})
