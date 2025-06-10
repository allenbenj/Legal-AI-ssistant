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
   The requirements file includes **asyncpg**, which is required for PostgreSQL connections. If you see an error such as `Import "asyncpg" could not be resolved`, make sure the dependency is installed in the active environment.

For more detailed instructions see [ENV_SETUP.md](ENV_SETUP.md).
