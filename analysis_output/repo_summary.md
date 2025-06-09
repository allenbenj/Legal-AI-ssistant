# Repository Summary

This repository contains a large collection of Python scripts and documentation for a **Legal AI System**. The main directory is `A_Scripts/` which includes the following notable components:

- **FastAPI Backend (`main.py`)** – Implements REST, GraphQL, and WebSocket endpoints.
- **Streamlit GUI (`main_gui.py`, `unified_gui.py`, `streamlit_app.py`)** – Provides a multi‑tab user interface for document processing, knowledge graph visualization, and system configuration.
- **Agents** – Multiple agent files implementing document processing, entity extraction, violation detection, and more. Agents inherit from `base_agent.py` which defines the common async framework.
- **Core Utilities** – Modules like `configuration_manager.py`, `embedding_manager.py`, and `service_container.py` manage configuration, embeddings, and service dependencies.
- **Memory Management** – `unified_memory_manager.py`, `claude_memory_store.py`, and extensive documentation in the `dependency_visualizations/` folder outline a sophisticated memory subsystem.
- **Documentation** – Markdown files such as `README.md`, `VISUALIZATION_STRATEGY.md`, `XAI_SETUP_GUIDE.md`, and `CLAUDE.md` describe system architecture, visualization strategies, and setup instructions.
- **Refactoring Plan** – `refactoring_plan.txt` details an intended reorganization into a clean package structure with directories like `core/`, `agents/`, `services/`, and `workflows/`.

The repository also contains many experimental or legacy scripts (e.g., `Criminal_Law_Parser_0.py`, `minimal_api.py`), PDF diagrams, and test files.
