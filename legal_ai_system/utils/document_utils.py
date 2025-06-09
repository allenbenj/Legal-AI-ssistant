"""Utilities for document discovery and text extraction with optional OCR."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def discover_documents(
    directory: Path, extensions: Iterable[str] = (".pdf", ".txt", ".png", ".jpg")
) -> List[Path]:
    """Recursively discover files under *directory* with given extensions."""
    docs: List[Path] = []
    for ext in extensions:
        docs.extend(directory.rglob(f"*{ext}"))
    return docs


def extract_text(path: Path, ocr: bool = True) -> str:
    """Return plain text from *path* using PDF parsing or OCR when required."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        text: List[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and ocr:
        image = Image.open(path)
        return pytesseract.image_to_string(image)

    return path.read_text(errors="ignore")


__all__ = ["discover_documents", "extract_text"]
