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
python scripts/setup_environment_task.py
```


npm install
npm run build
```


## Document Processor V2

`DocumentProcessorAgentV2` outputs a `MultiModalDocument` that unifies text, images, tables and optional audio or video data. The agent also offers helpers for specialized legal workflows:

- `process_video_depositions(path)` &ndash; returns a `DepositionTranscript` with timestamped segments.
- `process_legal_forms(path)` &ndash; extracts form fields into a `StructuredForm`.
- `process_contract_redlines(path)` &ndash; summarizes revisions into a `RedlineAnalysis`.

### Example

```python
import asyncio
from pathlib import Path
from legal_ai_system.agents.document_processor_agent_v2 import DocumentProcessorAgentV2

async def main() -> None:
    agent = DocumentProcessorAgentV2()
    transcript = await agent.process_video_depositions(Path("deposition.mp4"))
    form_data = await agent.process_legal_forms(Path("form.pdf"))
    redline = await agent.process_contract_redlines(Path("contract.docx"))
    print(transcript, form_data, redline)

asyncio.run(main())
```

