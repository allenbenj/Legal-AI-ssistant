# Developer Onboarding Guide

Welcome to the Legal AI System project! This guide helps new contributors set up
 their environment and explains the typical development workflow.

## Prerequisites
- Python 3.9 or later
- Node.js 18+
- Git

## Environment Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Legal-AI-ssistant
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   You can install everything with the convenience script:
   ```bash
   python legal_ai_system/scripts/install_all_dependencies.py
   ```
   This script installs Python packages and Node packages for the React frontend.
   If you prefer manual installation run:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   (cd frontend && npm install)
   ```
4. **Optional extras**
   Additional libraries for audio transcription and advanced parsing are listed in
   [ENV_SETUP.md](ENV_SETUP.md). Install them as needed.

## Running Tests
Tests use **nose2** on top of the standard `unittest` framework. After
installing the `dev` dependencies run `nose2` directly:
```bash
pip install -e .[dev]
nose2
```
See [docs/test_setup.md](test_setup.md) for details.

## Development Workflow
1. Create a feature branch and make changes.
2. Ensure code conforms to formatting rules:
   ```bash
   black .
   isort .
   ```
3. Run `nose2` to verify that all tests pass.
4. Commit your work and open a pull request.

## Additional Resources
- [System Layout](system_layout.md) – overview of services and agents
- [API Endpoints](api_endpoints.md) – REST API documentation
- [Integration Guide](integration_plan.md) – deployment and WebSocket usage

## Understanding the Architecture
Sequence diagrams illustrating how services initialize and how workflows
execute can be found under [docs/diagrams](diagrams/). Reviewing these diagrams
along with [system_layout.md](system_layout.md) provides a good overview of the
project structure.

Happy coding!
