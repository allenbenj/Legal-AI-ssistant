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

## Environment Variables

Several services expect credentials or URLs to be defined in environment variables:

- `OPENAI_API_KEY` or `XAI_API_KEY` – API key for your selected language model provider.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` – connection settings for the Neo4j database.
- `REDIS_URL_CACHE` – Redis connection string used for caching.
- `LEGAL_AI_ENCRYPTION_PASSWORD_SECRET` – master password used by the `SecurityManager`.

Additional optional variables are documented in `legal_ai_system/core/settings.py` if you need to override default paths or tuning parameters.

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
