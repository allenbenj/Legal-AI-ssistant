# Legal AI System

This repository contains a modular legal analysis platform. Source files live inside the `legal_ai_system` package. Key subdirectories:

- **core** – shared components such as the base agent, configuration utilities and vector store implementations.
- **agents** – specialized agents for tasks like document processing, citation analysis and violation detection.
- **services** – service classes for memory management, knowledge graphs and real-time workflows.
- **utils** – active helper modules for parsing and ontology management.
- **scripts** – entry points and helper scripts for running the system.
- **tests** – unit tests.
- **archive_legacy_components** – deprecated modules and scripts kept for historical reference.



All required Python dependencies are defined in `pyproject.toml` and mirrored in
`package.json` for compatibility with tooling that expects a Node-style manifest.

The GUI includes a Violation Review tab backed by a lightweight SQLite database
(`violations.db`). This store is initialized automatically by the service
container and can be inspected with the tests in `tests/test_violation_review.py`.

Run `python legal_ai_system/scripts/main.py` to launch the API server.
For the graphical interface start `python legal_ai_system/scripts/streamlit_app.py` or simply `python -m legal_ai_system`.
