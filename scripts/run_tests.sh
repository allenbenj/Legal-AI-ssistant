#!/usr/bin/env bash
# Simple helper to create a virtual environment, install dev dependencies,
# and execute the pytest suite.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
cd "$REPO_ROOT"

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]

pytest "$@"
