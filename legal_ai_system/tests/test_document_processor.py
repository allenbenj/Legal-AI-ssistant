from pathlib import Path
import email.message
import json
import zipfile

import pytest

import yaml  # type: ignore

from legal_ai_system.agents.document_processor_agent import DocumentProcessorAgent


@pytest.mark.asyncio
async def test_process_eml(tmp_path: Path) -> None:
    msg = email.message.EmailMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Hello"
    msg.set_content("Greetings!")

    eml_path = tmp_path / "sample.eml"
    eml_path.write_bytes(msg.as_bytes())

    agent = DocumentProcessorAgent(None)
    result = await agent.process(eml_path)
    assert result.data["extracted_metadata"]["subject"] == "Hello"
    assert "Greetings" in result.data["text_content"]


@pytest.mark.asyncio
async def test_process_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "foo")
        zf.writestr("b.txt", "bar")

    agent = DocumentProcessorAgent(None)
    result = await agent.process(zip_path)
    files = result.data["extracted_metadata"]["contained_files"]
    assert len(files) == 2


@pytest.mark.asyncio
async def test_process_json_and_yaml(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json.dump({"a": 1}, json_path.open("w"))

    yaml_path = tmp_path / "data.yaml"
    yaml.safe_dump({"b": 2}, yaml_path.open("w"))

    agent = DocumentProcessorAgent(None)
    res_json = await agent.process(json_path)
    assert '"a": 1' in res_json.data["text_content"]

    res_yaml = await agent.process(yaml_path)
    assert "b: 2" in res_yaml.data["text_content"]

