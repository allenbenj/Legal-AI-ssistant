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

## Running the Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pytest
```

The commands above create a virtual environment, install all development
dependencies, and execute the test suite. To automate these steps, you can run
the helper script:

```bash
./scripts/run_tests.sh
```

`run_tests.sh` ensures `.venv` exists, installs the required packages, and then
invokes `pytest`.
