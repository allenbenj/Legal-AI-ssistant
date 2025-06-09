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
- **extraneous_files** – archive for unused scripts, environment helpers, and other extras. Move outdated files here for reference. See `extraneous_files/ARCHIVE.md` for the list of archived items.

Older reference materials previously found in `legal_ai_system/docs` were
removed from the repository. If you need to review any of those documents,
check the `extraneous_files` archive for legacy notes and scripts.

Run `python legal_ai_system/scripts/main.py` to start the demo application.

