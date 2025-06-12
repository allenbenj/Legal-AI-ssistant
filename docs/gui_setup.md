# GUI Setup Guide

This guide explains how to launch the optional graphical interfaces included with the Legal AI System.

## Streamlit Dashboard

The Streamlit GUI is the default entry point when running `python -m legal_ai_system`.
Install the Python dependencies and then launch the interface:

```bash
pip install -r requirements.txt  # includes Streamlit
pip install -r requirements-dev.txt
python -m legal_ai_system
```

Alternatively you can run it directly with Streamlit:

```bash
streamlit run legal_ai_system/gui/streamlit_app.py
```

The app connects to the FastAPI backend to display workflow progress and system metrics.
Ensure the backend is running or configured in `config/settings.py`.

## PyQt6 Demo

A lightweight PyQt6 GUI is available for quickly testing the LangGraph workflow.
It requires the `PyQt6` package which is now listed in `requirements.txt`:

```bash
pip install PyQt6
```

This GUI lets you open a document and run the default analysis graph locally.

See the [PyQt6 Interface](../README.md#pyqt6-interface) section in the main
README for a summary of its capabilities and limitations.


## React Frontend

The repository also includes a React interface under `frontend/`.
Install the Node dependencies and start the development server:

```bash
cd frontend
npm install
npm run dev
```

For production builds use `npm run build`, which outputs files to `frontend/dist`.
The FastAPI server defined in `legal_ai_system/scripts/main.py` will automatically
serve these static assets when the directory exists.

---
For more details on backend configuration see [ENV_SETUP.md](ENV_SETUP.md).
