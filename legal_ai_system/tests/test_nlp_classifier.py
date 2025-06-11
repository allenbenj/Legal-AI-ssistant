import importlib
import sys
from types import ModuleType

import pytest

from legal_ai_system.utils.document_utils import LegalDocumentClassifier
from legal_ai_system.utils import nlp_classifier as nlp_mod


class FakePipeline:
    def __call__(self, text, labels):
        return {"labels": ["contract"], "scores": [0.95]}


def test_nlp_classifier_uses_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setattr(
        sys.modules["transformers"], "pipeline", lambda *a, **kw: FakePipeline()
    )
    importlib.reload(nlp_mod)
    clf = nlp_mod.NLPDocumentClassifier()
    result = clf.classify("This is a contract.")
    assert result["used_ml_model"] is True
    assert result["primary_type"] == "contract"


def test_nlp_classifier_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))

    def raise_error(*args, **kwargs):
        raise RuntimeError("load fail")

    monkeypatch.setattr(sys.modules["transformers"], "pipeline", raise_error)
    importlib.reload(nlp_mod)
    clf = nlp_mod.NLPDocumentClassifier()
    text = "The parties agree to the contract terms."
    result = clf.classify(text)
    fallback = LegalDocumentClassifier().classify(text)
    fallback["used_ml_model"] = False
    assert result == fallback


@pytest.mark.asyncio
async def test_document_processor_integration(monkeypatch, tmp_path):
    # stub heavy dependencies to import agent
    for name in [
        "fitz",
        "pytesseract",
        "PIL",
        "pandas",
        "pptx",
        "pptx.exc",
        "markdown",
        "bs4",
        "striprtf",
        "striprtf.striprtf",
        "legal_ai_system.services.service_container",
    ]:
        if name not in sys.modules:
            mod = ModuleType(name)
            if name == "PIL":
                class Image:  # type: ignore
                    pass
                class UnidentifiedImageError(Exception):
                    pass
                mod.Image = Image
                mod.UnidentifiedImageError = UnidentifiedImageError
            if name == "pptx":
                class Presentation:  # type: ignore
                    pass
                mod.Presentation = Presentation
            if name == "pptx.exc":
                class PackageNotFoundError(Exception):
                    pass
                mod.PackageNotFoundError = PackageNotFoundError
            if name == "striprtf.striprtf":
                mod.rtf_to_text = lambda x: ""
            sys.modules[name] = mod

    from legal_ai_system.agents import document_processor_agent as dp_mod

    monkeypatch.setattr(
        dp_mod, "NLPDocumentClassifier", lambda: FakePipelineClassifier(), False
    )
    importlib.reload(dp_mod)
    agent = dp_mod.DocumentProcessorAgent(None)

    txt = tmp_path / "sample.txt"
    txt.write_text("This is a contract.")
    result = await agent._process_task(txt, {})
    assert result["classification_details"]["primary_type"] == "contract"
    assert result["classification_details"]["used_ml_model"]


class FakePipelineClassifier(nlp_mod.NLPDocumentClassifier):
    def __init__(self):
        self.labels = ["contract"]
        self._pipeline = lambda text, labels: {"labels": ["contract"], "scores": [1.0]}
        self._fallback = LegalDocumentClassifier()

    def classify(self, text: str, filename: str | None = None):
        return {
            "is_legal_document": True,
            "primary_type": "contract",
            "primary_score": 1.0,
            "filename": filename,
            "used_ml_model": True,
        }
