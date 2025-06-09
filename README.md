# Legal AI System

This repository contains a modular legal analysis platform. Source files live inside the `legal_ai_system` package. Key subdirectories:

- **core** – shared components such as the base agent, configuration utilities and vector store implementations.
- **agents** – specialized agents for tasks like document processing, citation analysis and violation detection.
- **services** – service classes for memory management, knowledge graphs and real-time workflows.
- **utils** – helper modules for parsing, ontology management and other utilities.
- **scripts** – entry points and helper scripts for running the system.
- **tests** – unit tests.
- **archive_legacy_components** – deprecated modules and scripts kept for historical reference.

Extra reference materials and historical components reside in the
`archive_legacy_components` directory. The folder includes its own
`__init__.py` so these modules can be imported if ever needed. Quality tools
ignore this directory via settings in `pyproject.toml`.

Run `python legal_ai_system/scripts/main.py` to start the demo application.
