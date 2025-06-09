# Cleanup and Refactoring Plan

Based on the documentation and file layout, the following steps are recommended to clean up the project.

## 1. Establish a Clear Package Structure

Move Python modules into a standard package hierarchy:

```
legal_ai_system/
├── core/            # configuration_manager.py, embedding_manager.py, service_container.py, security_manager.py
├── agents/          # each agent in its own subdirectory
├── workflows/       # realtime_analysis_workflow.py, ontology_integration.py
├── gui/             # main_gui.py, unified_gui.py, streamlit_app.py
├── api/             # main.py (FastAPI) and related routers
├── knowledge/       # knowledge_graph_manager.py, vector_store.py
├── memory/          # unified_memory_manager.py, claude_memory_store.py
└── utils/           # helper modules, error_recovery.py
```

Remove obsolete or duplicate scripts (e.g., `Criminal_Law_Parser_0.py`, old backups) after migrating any useful code into the new structure.

## 2. Consolidate Document Processing

Combine the multiple document processor implementations (`document_processor.py`, `document_processor_full.py`, `document_processor_clean.py`) into a single well-tested module under `agents/document_processor/`. The module should expose an async interface and avoid GUI dependencies.

## 3. Unify Vector Store and Knowledge Graph

Use `enhanced_vector_store.py` as the base for a new `knowledge/vector_store.py`. Integrate with Neo4j via `knowledge_graph_manager.py`. Ensure embedding operations are centralized in `embedding_manager.py`.

## 4. Refine Memory Management

Integrate functionality from `claude_memory_store.py` into `unified_memory_manager.py`. Follow the architecture outlined in `dependency_visualizations/memory_management.md`. Provide a clear API for session persistence and retrieval.

## 5. Streamline Configuration and Logging

Move `settings.py`, `constants.py`, and `grok_config.py` into a `config/` package. Ensure all modules import configuration via `ConfigurationManager`. Use the structured logging facilities from `detailed_logging.py` across the project.

## 6. Frontend Separation

The React frontend files mentioned in `refactoring_plan.txt` should reside in a separate `frontend/` directory. Keep backend code independent of frontend build tools.

## 7. Documentation and Diagrams

Collect Markdown documents and diagrams into a `docs/` directory. Keep only the most relevant diagrams (e.g., `ultimate_drawio_masterpiece.xml`, `memory_management_masterpiece.xml`) and remove duplicates. Link these assets in the README.

## 8. Testing and CI

Organize tests under `tests/` using pytest. Implement unit tests for each agent and service. Configure a CI workflow (GitHub Actions) to run `pytest` and basic linting (`flake8`, `black --check`).

## 9. Dependency Management

Maintain a single `requirements.txt` at the project root. Consider using `pip-tools` for pinned versions. Provide a `setup_environment.sh` script to install dependencies and download required spaCy models.

## 10. Version Control Hygiene

Remove generated files and personal data. Add `.gitignore` rules for environment folders, logs, and compiled artifacts. Ensure sensitive information such as API keys is loaded from environment variables instead of being hardcoded.

---

Following this plan will reduce clutter, improve maintainability, and make it easier for contributors to navigate the codebase.
