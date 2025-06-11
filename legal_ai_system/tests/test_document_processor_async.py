import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from legal_ai_system.agents.document_processor_agent import DocumentProcessorAgent
except Exception as exc:  # pragma: no cover - skip if deps missing
    pytest.skip(f"DocumentProcessorAgent unavailable: {exc}", allow_module_level=True)


@pytest.mark.asyncio
async def test_concurrent_txt_processing(tmp_path, monkeypatch):
    files = []
    for i in range(5):
        p = tmp_path / f"file_{i}.txt"
        p.write_text(f"hello {i}")
        files.append(p)

    agent = DocumentProcessorAgent(SimpleNamespace())

    async def slow_txt(file_path: Path, options):
        await asyncio.sleep(0.2)
        return {"text_content": file_path.read_text(), "extracted_metadata": {}, "processing_notes": []}

    monkeypatch.setattr(agent, "_process_txt_async", slow_txt)

    start = time.perf_counter()
    results = await asyncio.gather(*(agent._process_task(p, {}) for p in files))
    duration = time.perf_counter() - start

    assert len(results) == 5
    assert duration < 0.6
