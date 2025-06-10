# Monitoring and Metrics

This project exposes Prometheus metrics and structured logs to help monitor
workflow performance.

## Metrics Exporter

`MetricsExporter` located in `services/metrics_exporter.py` starts a Prometheus
HTTP server when the service container is created. By default the exporter
listens on port **8001**.

Metrics available:

- `workflow_execution_seconds` – histogram of workflow execution times.
- `workflow_errors_total` – counter of workflow processing errors.

Access the metrics endpoint at `http://localhost:8001/metrics` after starting the
application.

## Structured Logging

`init_logging` now writes JSON log lines alongside the regular log file. The
structured log file uses the `.jsonl` extension and is created in the
`legal_ai_system/logs` directory.

These logs include extra fields such as the logger name and function. They can be
shipped to monitoring systems for further analysis.
