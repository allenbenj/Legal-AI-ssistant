# Legal AI System

This repository contains the core components for the Legal AI assistant. To run the code you need Python 3.9 or later.

## Quick Start

To install all required and optional dependencies in one step, run the unified setup script:

```bash
python legal_ai_system/scripts/install_all_dependencies.py
```

This script creates a `.venv` virtual environment, installs every Python package (including extras like audio transcription and database utilities), and installs the Node packages for the React frontend.

If you prefer manual installation:
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install project dependencies:
   ```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # development and testing tools
```
   The requirements file includes database drivers such as **asyncpg** for PostgreSQL and **aioredis** for Redis. Installing `requirements-dev.txt` pulls in optional tools like `pytest` and linters.
3. Install Node packages for the React frontend:
   ```bash
   (cd frontend && npm install)
   ```
4. Build the frontend for production (optional when serving via FastAPI):
   ```bash
   (cd frontend && npm run build)
   ```
   This uses `vite.config.ts` which outputs to `frontend/dist`. FastAPI looks for
   static files in `LegalAISettings.frontend_dist_path`, which defaults to the same directory,
   so a production build will be served automatically when present.

If you plan to use the optional **LexPredict** pipelines, also install `lexnlp`:
```bash
pip install lexnlp
```

### Optional Dependencies

The system can optionally transcribe audio and perform speaker diarization. To
enable these features, install additional libraries:

```bash
pip install ffmpeg-python openai-whisper whisperx pdfplumber pyannote.audio
```

When adding new agents, run the dependency check script to verify that all
optional modules are installed:

```bash
python legal_ai_system/scripts/check_optional_dependencies.py
```

To experiment with advanced workflow orchestration using the
[`langgraph`](https://pypi.org/project/langgraph/) engine install it
separately:

```bash
pip install langgraph==0.0.20
```

The [advanced LangGraph guide](legal_ai_system/docs/advanced_langgraph.md) explains how this
optional dependency enables document classification routing, specialized
subgraphs, and real-time progress updates over WebSocket. It also shows a
`CaseWorkflowState` example for passing state between nodes.

If Pylance or other type checkers report assignment errors for ``BaseNode``
after installing ``langgraph``, run the helper script below. It scans all Python
files and rewrites optional imports so that the fallback ``BaseNode`` is always
aliased consistently:

```bash
python legal_ai_system/scripts/fix_langgraph_typing.py
```

For more detailed instructions see [ENV_SETUP.md](legal_ai_system/docs/ENV_SETUP.md).

### PostgreSQL Setup

Run the initialization script to create the PostgreSQL database before applying migrations:

```bash
python legal_ai_system/scripts/init_postgres_db.py
python legal_ai_system/scripts/migrate_database.py
```

The script reads the `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
`POSTGRES_DB` environment variables to determine the connection details. Set `POSTGRES_ADMIN_DB` if
you need to use an administrative database other than `postgres`.

The older `setup_environment_task.py` script can also be used to create the vir
tual environment and run the tests:
```bash
python legal_ai_system/scripts/setup_environment_task.py
```
For a faster workflow, you can run the helper script below. It automatically
creates `.venv` if needed, installs all development dependencies, and invokes
`pytest`:

```bash
./scripts/run_tests.sh
```

## Running Tests

Before invoking `pytest`, install the development dependencies:

```bash
pip install -e .[dev]
```

Running tests with coverage is enabled by default via `pytest-cov`. After
installing the development dependencies simply run `pytest`:

```bash
pytest
```

Alternatively, run the installation helper:

```bash
python legal_ai_system/scripts/install_all_dependencies.py
```

Missing packages such as `pytest-mock` will cause test failures.

See [docs/test_setup.md](legal_ai_system/docs/test_setup.md) for more information.

## Using the GUIs

The repository provides three optional interfaces. Install the dependencies and
choose the one that suits your workflow:

1. **Streamlit Dashboard** – run `python -m legal_ai_system` after installing
   `requirements.txt` (and optionally `requirements-dev.txt`). This launches a browser-based dashboard connected to the
   backend. Development builds are supported with the React frontend.
   Use `npm run build` to generate `frontend/dist`, which FastAPI serves
   automatically when present.

2. **Memory Brain Demo** – run
   `python -m legal_ai_system.gui.scripts.memory_brain_gui` to launch a focused
   Streamlit interface for testing the memory management features.

### PyQt6 Interface

Install `PyQt6` and launch the integrated desktop GUI:

```bash
pip install PyQt6

```

This interface can open local documents and run the default analysis workflow
without a browser. Under the hood the GUI starts a **BackendBridge** which
initialises the asynchronous service container and forwards all actions to the
`LegalAIIntegrationService`. Progress updates are emitted back to the widgets in
real time.

```python
from legal_ai_system.gui import IntegratedMainWindow
window = IntegratedMainWindow()
window.show()
```

The bridge ensures the desktop app communicates with the same backend services
as the API and other frontends. It is primarily a demo and lacks the advanced
features of the Streamlit and React frontends.

Detailed instructions are available in [docs/gui_setup.md](legal_ai_system/docs/gui_setup.md).

### Extraction Options

The ontology extraction agent supports multiple NER backends. Enable them in
`legal_ai_system/config/defaults.yaml`:

- `enable_spacy_ner`: load a spaCy pipeline specified by `spacy_ner_model`.
- `enable_legal_bert`: use a HuggingFace Legal‑BERT model defined by
  `legal_bert_model_name`.
- `enable_regex_extraction`: apply regex patterns from the configuration files.

All enabled methods are combined with confidence weighting during extraction.


### Start Task


```
### Example: Build a Workflow

```python
# Create the default workflow and process a PDF
workflow = builder.build()
result = workflow.run("sample.pdf")
print(result)
```

You can customize the workflow builder to enable or disable specific agents.
See the documents in the `legal_ai_system/docs/` folder for architecture details and advanced
usage. The [Integration Guide](legal_ai_system/docs/integration_plan.md) summarises the
five-phase integration plan, WebSocket patterns and deployment tips and
includes sections on security, testing, success metrics and troubleshooting.

The audit report in
[legal_ai_system/docs/file_audit.md](legal_ai_system/docs/file_audit.md)
captures the current folder layout, dependency findings, and a cleanup plan.
Refer to it when making structural changes or removing deprecated modules.

If PostgreSQL becomes unavailable, the application enters a degraded mode.
Refer to [legal_ai_system/docs/postgres_downtime_impact.md](legal_ai_system/docs/postgres_downtime_impact.md)
for a detailed list of features that are disabled or limited during downtime.
