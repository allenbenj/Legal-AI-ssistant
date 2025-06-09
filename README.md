# Legal AI System

This repository contains a modular legal analysis platform. Source files live
inside the `legal_ai_system` package. Key subdirectories:

- **core** – shared components such as the base agent, configuration utilities
  and vector store implementations.
- **agents** – specialized agents for tasks like document processing, citation
  analysis and violation detection.
- **services** – service classes for memory management, knowledge graphs and
  real-time workflows.
- **utils** – helper modules for parsing, ontology management and other
  utilities.
- **scripts** – entry points and helper scripts for running the system.
- **tests** – unit tests.

Legacy reference materials are sometimes placed under
`legal_ai_system/docs/legacy_files.md` if present.

## Installation

1. Create a Python virtual environment (optional but recommended).
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Additional development dependencies are defined in `pyproject.toml`.

## Usage

Run the main demo application:

```bash
python legal_ai_system/scripts/main.py
```

For a lightweight API, start `minimal_api.py` which exposes a FastAPI server.
Once running, interactive API documentation is available at
`http://localhost:8000/docs`.
