import sys
from types import ModuleType

# Stub heavy optional dependencies
for name in ["fitz", "pytesseract", "PIL", "PIL.Image"]:
    sys.modules.setdefault(name, ModuleType(name))

import pytest
from legal_ai_system.workflows.routing import (
    build_advanced_legal_workflow,
    DocumentClassificationNode,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "classification,path_keyword",
    [
        ({"primary_type": "contract"}, "Contract summary"),
        ({"primary_type": "court_filing"}, "Litigation summary"),
        ({"primary_type": "statute"}, "Regulatory summary"),
        ({"primary_type": "other"}, "Evidence summary"),
    ],
)
async def test_routing(monkeypatch, classification, path_keyword):
    def fake_classify(self, text):
        return classification

    monkeypatch.setattr(DocumentClassificationNode, "__call__", fake_classify)

    wf = build_advanced_legal_workflow()
    result = await wf.run("doc")
    assert path_keyword in result
