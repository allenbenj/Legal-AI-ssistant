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

## Building the Frontend

The React GUI lives under `frontend/`. To build the static assets you need Node
and npm installed (Node 18 or newer works well).

```bash
cd frontend
npm install
npm run build
```

The build command creates a `dist` directory containing the optimized
application bundle.
