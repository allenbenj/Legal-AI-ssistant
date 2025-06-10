# Testing Setup

This project uses **pytest** for running automated tests. To run the tests, first install the required dependencies and then execute `pytest` from the repository root.

## Required Python Packages

The core dependencies for running the tests are listed below. They can be installed using `pip install -r requirements.txt` for the runtime libraries and `pip install -e .[dev]` when using Poetry.

- `pytest` and `pytest-asyncio` – main test framework with asyncio support
- `pytest-mock` – utilities for mocking during tests
- `pytest-cov` – required because coverage reporting is enabled by default
- `typer` – required to test the CLI script
- `streamlit` – used by the GUI script (tests mock this module)

Additional packages from `requirements.txt` are needed for the application itself (FastAPI, pydantic, etc.).

Running the tests without these dependencies will result in import errors. For
example, missing `pytest-mock` will cause fixture failures.

## Running the Tests

Ensure the development dependencies are installed before executing `pytest`. Run
`pip install -e .[dev]` or use `python legal_ai_system/scripts/install_all_dependencies.py`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pytest
```

The command above sets up a virtual environment, installs dependencies, and executes the test suite.
