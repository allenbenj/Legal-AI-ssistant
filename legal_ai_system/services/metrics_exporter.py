from __future__ import annotations

from typing import Optional

from prometheus_client import Counter, Histogram, start_http_server

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

    def observe_workflow_time(self, duration: float) -> None:
        """Record workflow execution duration."""
        self.workflow_execution_seconds.observe(duration)

    def inc_workflow_error(self) -> None:
        """Increment the workflow error counter."""
        self.workflow_errors_total.inc()


metrics_exporter: Optional[MetricsExporter] = None


def init_metrics_exporter(port: int = 8001) -> MetricsExporter:
    """Create and return the singleton metrics exporter."""
    global metrics_exporter
    if metrics_exporter is None:
        metrics_exporter = MetricsExporter(port=port)
    return metrics_exporter


__all__ = ["MetricsExporter", "init_metrics_exporter", "metrics_exporter"]
