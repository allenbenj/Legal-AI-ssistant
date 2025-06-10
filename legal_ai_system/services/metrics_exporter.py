from __future__ import annotations

from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, start_http_server

from ..core.detailed_logging import get_detailed_logger, LogCategory


class MetricsExporter:
    """Expose Prometheus metrics for workflow monitoring."""

    def __init__(self, port: int = 8001) -> None:
        self.logger = get_detailed_logger("MetricsExporter", LogCategory.SYSTEM)
        start_http_server(port)
        self.logger.info("Prometheus metrics server started", parameters={"port": port})

        self.workflow_execution_seconds = Histogram(
            "workflow_execution_seconds",
            "Time spent executing workflows in seconds",
        )
        self.workflow_errors_total = Counter(
            "workflow_errors_total",
            "Total number of workflow errors",
        )

        # Knowledge graph query metrics
        self.kg_queries_total = Counter(
            "knowledge_graph_queries_total",
            "Total number of knowledge graph queries",
        )
        self.kg_query_cache_hits = Counter(
            "knowledge_graph_query_cache_hits_total",
            "Total knowledge graph query cache hits",
        )

        # Vector store performance metrics
        self.vector_add_seconds = Histogram(
            "vector_store_add_seconds",
            "Time spent adding vectors to the store",
        )
        self.vector_search_seconds = Histogram(
            "vector_store_search_seconds",
            "Time spent searching the vector store",
        )

        # Connection pool gauges
        self.pg_pool_in_use = Gauge(
            "pg_pool_in_use_connections",
            "Number of PostgreSQL connections currently in use",
        )
        self.pg_pool_free = Gauge(
            "pg_pool_free_connections",
            "Number of free PostgreSQL connections",
        )
        self.redis_pool_in_use = Gauge(
            "redis_pool_in_use_connections",
            "Approximate number of Redis connections in use",
        )

    def observe_workflow_time(self, duration: float) -> None:
        """Record workflow execution duration."""
        self.workflow_execution_seconds.observe(duration)

    def inc_workflow_error(self) -> None:
        """Increment the workflow error counter."""
        self.workflow_errors_total.inc()

    # --- Knowledge graph metrics ---

    def inc_kg_query(self, cache_hit: bool = False) -> None:
        """Increment knowledge graph query counters."""
        self.kg_queries_total.inc()
        if cache_hit:
            self.kg_query_cache_hits.inc()

    # --- Vector store metrics ---

    def observe_vector_add(self, duration: float) -> None:
        """Record duration of vector add operation."""
        self.vector_add_seconds.observe(duration)

    def observe_vector_search(self, duration: float) -> None:
        """Record duration of vector search operation."""
        self.vector_search_seconds.observe(duration)

    # --- Connection pool metrics ---

    def update_pool_metrics(
        self, pg_in_use: int = 0, pg_free: int = 0, redis_in_use: int = 0
    ) -> None:
        """Update gauges for connection pool utilization."""
        self.pg_pool_in_use.set(pg_in_use)
        self.pg_pool_free.set(pg_free)
        self.redis_pool_in_use.set(redis_in_use)

    def snapshot(self) -> dict:
        """Return current metric values for health endpoints."""
        return {
            "kg_queries_total": self.kg_queries_total._value.get(),
            "kg_query_cache_hits": self.kg_query_cache_hits._value.get(),
            "vector_add_seconds_sum": self.vector_add_seconds._sum.get(),
            "vector_search_seconds_sum": self.vector_search_seconds._sum.get(),
            "pg_pool_in_use": self.pg_pool_in_use._value.get(),
            "pg_pool_free": self.pg_pool_free._value.get(),
            "redis_pool_in_use": self.redis_pool_in_use._value.get(),
        }


metrics_exporter: Optional[MetricsExporter] = None


def init_metrics_exporter(port: int = 8001) -> MetricsExporter:
    """Create and return the singleton metrics exporter."""
    global metrics_exporter
    if metrics_exporter is None:
        metrics_exporter = MetricsExporter(port=port)
    return metrics_exporter


__all__ = ["MetricsExporter", "init_metrics_exporter", "metrics_exporter"]
