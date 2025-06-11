import os
from pathlib import Path

import pytest

import sys
import types

sys.modules.setdefault("fitz", types.ModuleType("fitz"))
sys.modules.setdefault("pytesseract", types.ModuleType("pytesseract"))
from legal_ai_system.utils import document_utils
from legal_ai_system.utils.ocr_decision import predict_ocr_settings

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional
    Image = None  # type: ignore


@pytest.mark.unit
def test_adaptive_chunker_learns_size():
    text = "Section 1\nThis is a short paragraph.\n\nSection 2\n" + "A" * 500
    chunker = document_utils.AdaptiveDocumentChunker()
    chunks = chunker.chunk_text(text)
    assert chunker.chunk_size >= 1000
    assert chunks


@pytest.mark.unit
@pytest.mark.skipif(Image is None, reason="Pillow not available")
def test_predict_ocr_settings_image(tmp_path):
    img_path = tmp_path / "img.png"
    img = Image.new("RGB", (20, 20), color=(255, 255, 255))
    img.save(img_path)
    result = predict_ocr_settings(img_path)
    assert result["ocr_needed"] is True
    assert result["dpi"] == 300 or result["dpi"] >= 72
