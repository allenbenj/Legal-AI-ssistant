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
   ```
   The requirements file includes database drivers such as **asyncpg** for PostgreSQL and **aioredis** for Redis. If you see errors like `Import "asyncpg" could not be resolved` or `Import "aioredis" could not be resolved`, ensure the dependencies are installed in the active environment.
3. Install Node packages for the React frontend:
   ```bash
   npm install
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

To experiment with advanced workflow orchestration using the
[`langgraph`](https://pypi.org/project/langgraph/) engine install it
separately:

```bash
pip install langgraph==0.0.20
```

See [advanced_langgraph.md](docs/advanced_langgraph.md) for details on how the
package integrates with document routing and WebSocket progress updates.

For more detailed instructions see [ENV_SETUP.md](ENV_SETUP.md).

The older `setup_environment_task.py` script can also be used to create the vir
tual environment and run the tests:
```bash
python legal_ai_system/scripts/setup_environment_task.py
```


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
See the documents in the `docs/` folder for architecture details and advanced
usage. The [Integration Guide](docs/integration_plan.md) summarises the
five-phase integration plan, WebSocket patterns and deployment tips and
includes sections on security, testing, success metrics and troubleshooting.
For LangGraph specific routing examples see [advanced_langgraph.md](docs/advanced_langgraph.md).

## Database Setup

The application uses SQLite by default, created under `legal_ai_system/storage`.
To use PostgreSQL or another database, update `sqlite_path` or the relevant
connection URLs in `config/defaults.yaml`.  Migrations are handled
automatically on startup.

## Backup and Failover

Vector store indexes are periodically optimized and saved to the `backups/`
folder under the configured vector storage path.  Set the intervals via
`vector_store_optimization_interval` and `vector_store_backup_interval` in the
configuration file or environment variables.  When running multiple vector store
instances, list their addresses in `vector_store_hosts` and choose a
`load_balancing_strategy` such as `round_robin` to distribute requests.
If a node becomes unavailable the service container will route traffic to the
remaining healthy instances.
