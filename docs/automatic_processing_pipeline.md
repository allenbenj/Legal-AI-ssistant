# Automatic Processing Pipeline

The automatic processing pipeline monitors configured directories for new
files and forwards them to the `LegalAIIntegrationService` for analysis.
It provides several features:

- **Auto-trigger** – newly detected documents are sent to the workflow
  without manual intervention.
- **Configurable** – a list of enabled agents can be supplied when
  creating the pipeline so only selected components run automatically.
- **Priority system** – files are queued using an `asyncio.PriorityQueue`
  so high priority tasks run first.
- **Batch processing** – multiple documents can be processed concurrently
  with the `max_concurrent` setting.

This service is implemented in
`legal_ai_system/services/automatic_processing.py`.
