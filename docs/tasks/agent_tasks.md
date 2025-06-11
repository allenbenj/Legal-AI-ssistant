# Agent Task List

This document consolidates outstanding tasks for the agents within the Legal AI System. Each item captures a specific area for improvement or new functionality. The tasks are numbered to reflect their order in the original plan.

1. **Audit asyncio.to_thread usage** – Refactor all instances to use fully asynchronous libraries (e.g., `aiofiles`, `aiodocument`). Remove thread‑pool overhead in `document_processor_agent.py` and related utilities.
2. **Expand async streaming** in `RealTimeAnalysisWorkflow` – Persist each progress update to disk and provide a WebSocket API for real-time status updates.
3. **Integrate Legal‑BERT and RoBERTa‑Legal** – Modify `OntologyExtractionAgent` so it can dynamically load both models and use fallback logic.
4. **Review CacheManager** – Add Redis-based caching with disk persistence and create regression tests verifying cache behavior.
5. **Implement legal citation extraction** – Extract case law, statutes, and regulations, providing confidence scores and linking each citation to the ontology.
6. **Expand the ontology** – Ensure briefs, motions, discovery materials, and depositions flow through all workflows, with confidence and statistical significance tracking.
7. **Remove concurrency bottlenecks** – Enable simultaneous document processing across all workflows and confirm asynchronous operations wherever beneficial.
8. **Optimize FAISS indexing** – Add periodic maintenance hooks and benchmark query performance before and after optimization.
9. **Persist data to disk** – Ensure the knowledge graph, vector store, and reviewable memory store all persist data, avoiding in-memory-only storage during long operations.
10. **Add comprehensive pytest coverage** – Provide tests for the new functionality and update documentation with installation and usage details.
