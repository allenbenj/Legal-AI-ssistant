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

For more detailed instructions see [ENV_SETUP.md](ENV_SETUP.md).

Alternatively, run the helper script to automate the setup and validation:
```bash
python scripts/setup_environment_task.py
```

```bash
npm install
npm run build
```

## Example: Building a Workflow

Below is a minimal example using `LegalWorkflowBuilder` to create a typed
workflow composed of two agents.  Each agent defines its input and output
models so the builder can verify compatibility.

```python
from legal_ai_system.workflows.builder import LegalWorkflowBuilder
from legal_ai_system.agents.entity_extraction_agent import EntityExtractionAgent
from legal_ai_system.agents.text_correction_agent import TextCorrectionAgent
from pydantic import BaseModel
import asyncio


class TextIn(BaseModel):
    text: str


class TextOut(BaseModel):
    text: str


async def main() -> None:
    builder = LegalWorkflowBuilder[TextIn, TextOut]()
    builder.add_agent(EntityExtractionAgent())
    builder.add_agent(TextCorrectionAgent())
    workflow = builder.build()
    result = await workflow.process_batch([TextIn(text="Example")])
    print(result[0].text)


if __name__ == "__main__":
    asyncio.run(main())
```


