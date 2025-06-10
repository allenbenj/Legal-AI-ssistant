# Legal AI System

This repository contains the core components for the Legal AI assistant. To run the code you need Python 3.9 or later.

## Quick Start

1. (Recommended) Create and activate a virtual environment:
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

For more detailed instructions see [ENV_SETUP.md](ENV_SETUP.md).

## Running the FastAPI Server

The backend API is implemented with FastAPI. You can start it directly
using `uvicorn` or by executing the main script:

```bash
# Option 1: using uvicorn
uvicorn legal_ai_system.scripts.main:app --reload

# Option 2: run the script directly
python -m legal_ai_system.scripts.main
```

Optional environment variables control the host and port:

- `LEGAL_AI_API_HOST` – interface to bind to (default `0.0.0.0`)
- `LEGAL_AI_API_PORT` – port number (default `8000`)
- `FRONTEND_DIST_DIR` – directory of the built React app if you wish the
  API to serve the static files.

## Building the React Frontend

The React GUI lives under `legal_ai_system/frontend/`. To create a build
install Node packages and run the build script:

```bash
npm install
npm run build
```

The output will be written to `legal_ai_system/frontend/dist/` by default.
Set `API_BASE_URL` (e.g. `http://localhost:8000`) so the frontend knows
how to contact the backend when compiled.

## Launching the Streamlit or Unified GUI

Two options are available for the interactive interface:

```bash
# Streamlit dashboard
streamlit run legal_ai_system/gui/streamlit_app.py

# Unified launcher (uses __main__)
python -m legal_ai_system
```

The GUI expects the backend API URL in `API_BASE_URL` and will use the
built frontend files from `FRONTEND_DIST_DIR` when served through
FastAPI.
