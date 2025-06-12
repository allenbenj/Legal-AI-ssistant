# Testing Setup

This project uses **nose2** for running automated tests built on the standard `unittest` framework. Before running `nose2`, install the development dependencies with `pip install -e .[dev]` or `python legal_ai_system/scripts/install_all_dependencies.py`. Missing packages such as **pytest-mock** will cause test failures.

## Required Python Packages

The core dependencies for running the tests are listed below. They can be installed using `pip install -r requirements.txt` for the runtime libraries and `pip install -r requirements-dev.txt` or `pip install -e .[dev]` when using Poetry.

- `nose2` and `pytest-asyncio` – test runner with asyncio support
- `pytest-mock` – utilities for mocking during tests
- `pytest-cov` – required because coverage reporting is enabled by default
- `typer` – required to test the CLI script
- `streamlit` – used by the GUI script (tests mock this module)
- `numpy` – used by vector store operations
- `PyYAML` – configuration parsing in multiple modules
- `joblib` – saving and loading ML models

Additional packages from `requirements.txt` are needed for the application itself (FastAPI, pydantic, etc.). Development tools come from `requirements-dev.txt`.

Running the tests without these dependencies will result in import errors. For
example, missing `pytest-mock` will cause fixture failures.

## Running the Tests

Ensure the development dependencies are installed before executing `nose2`. Run
`pip install -e .[dev]` or use `python legal_ai_system/scripts/install_all_dependencies.py`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
nose2
```

Missing packages such as `pytest-mock` will cause failures if the development dependencies are not installed.
The commands above create a virtual environment, install all development
dependencies, and execute the test suite. To automate these steps, you can run
the helper script:

```bash
./scripts/run_tests.sh
```

`run_tests.sh` ensures `.venv` exists, installs the required packages, and then
invokes `nose2`.
