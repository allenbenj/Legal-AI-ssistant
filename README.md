# Legal AI System

This repository contains a modular legal analysis platform. Source files live inside the `legal_ai_system` package. Key subdirectories:

- **core** – shared components such as the base agent, configuration utilities and vector store implementations.
- **agents** – specialized agents for tasks like document processing, citation analysis and violation detection.
- **services** – service classes for memory management, knowledge graphs and real-time workflows.
- **utils** – active helper modules for parsing and ontology management.
- **scripts** – entry points and helper scripts for running the system.
- **archive_legacy_components** – deprecated modules and backup scripts retained for reference.
- **visualizations** – architecture diagrams and interactive network files.
- **tests** – unit tests.


Deprecated utilities and backup scripts now live under the `archive_legacy_components` folder at the project root. Several unused modules from `legal_ai_system/utils` have been relocated there to keep the active codebase lean. See `archive_legacy_components/ARCHIVED_UTILS.md` for a list of these modules.

Run `python legal_ai_system/scripts/main.py` to start the demo application.
