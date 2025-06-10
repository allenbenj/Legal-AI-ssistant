# Environment Setup Guide

This document summarizes steps to create an isolated Python environment and install project dependencies.

## Python Version
Ensure Python 3.9 or later is installed.

```
$ python3 --version
Python 3.12.10
```

## Virtual Environment
Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` inside the environment:

```bash
pip install --upgrade pip
```

## Installing Dependencies
Install all required packages from the project requirements file:

```bash
pip install -r requirements.txt
```

If you plan to use LangGraph-based workflows, install the optional dependency:

```bash
pip install langgraph
```

Additional database utilities:

```bash
pip install sqlalchemy lancedb
```

## Verification
After installation, verify critical imports:

```bash
python - <<'PY'
import fastapi, uvicorn, streamlit, pydantic
import openai, sentence_transformers
import neo4j, sqlalchemy, lancedb
import asyncpg, aioredis
import torch
print('Environment ready')
PY
```

If the script prints `Environment ready` without errors, the setup was successful.
