# Legal AI System

This repository contains a modular legal analysis platform. Source files live inside the `legal_ai_system` package. Key subdirectories:

- **core** – shared components such as the base agent, configuration utilities and vector store implementations.
- **agents** – specialized agents for tasks like document processing, citation analysis and violation detection.
- **services** – service classes for memory management, knowledge graphs and real-time workflows.
- **utils** – helper modules for parsing, ontology management and other utilities.
- **scripts** – entry points and helper scripts for running the system.
- **docs** – documentation and planning notes.
- **visualizations** – architecture diagrams and interactive network files.
- **tests** – unit tests.

Extra reference materials and archived scripts are listed in
`legal_ai_system/docs/legacy_files.md`. Files moved out of the main package are
stored under `legal_ai_system/legacy_extras/` for historical reference.

All required Python dependencies are defined in `pyproject.toml` and mirrored in
`package.json` for compatibility with tooling that expects a Node-style manifest.

The GUI includes a Violation Review tab backed by a lightweight SQLite database
(`violations.db`). This store is initialized automatically by the service
container and can be inspected with the tests in `tests/test_violation_review.py`.

Run `python legal_ai_system/scripts/main.py` to launch the API server.
For the graphical interface start `python legal_ai_system/scripts/streamlit_app.py` or simply `python -m legal_ai_system`.
