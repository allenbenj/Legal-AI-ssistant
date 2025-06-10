# Testing Setup

This project uses **pytest** for running automated tests. To run the tests, first install the required dependencies and then execute `pytest` from the repository root.

## Required Python Packages

The core dependencies for running the tests are listed below. They can be installed using `pip install -r requirements.txt` for the runtime libraries and `pip install -e .[dev]` when using Poetry.

- `pytest` and `pytest-asyncio` – main test framework with asyncio support
- `pytest-mock` – utilities for mocking during tests
- `typer` – required to test the CLI script
- `streamlit` – used by the GUI script (tests mock this module)

Additional packages from `requirements.txt` are needed for the application itself (FastAPI, pydantic, etc.).

## Running the Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-mock typer streamlit
pytest
```

The command above sets up a virtual environment, installs dependencies, and executes the test suite.
