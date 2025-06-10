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

If you plan to use the optional **LexPredict** pipelines, also install `lexnlp`:
```bash
pip install lexnlp
```

### Optional Dependencies

The system can optionally transcribe audio and perform speaker diarization. To
enable these features, install additional libraries:

```bash
pip install ffmpeg-python openai-whisper whisperx pdfplumber pyannote.audio
```

For more detailed instructions see [ENV_SETUP.md](ENV_SETUP.md).

Alternatively, run the helper script to automate the setup and validation:
```bash
python legal_ai_system/scripts/setup_environment_task.py
```
### Example: Build a Workflow



workflow = builder.build()
result = workflow.run("sample.pdf")
print(result)
```
