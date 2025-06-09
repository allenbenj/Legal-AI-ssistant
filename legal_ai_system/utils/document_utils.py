"""Utilities for document discovery and text extraction with optional OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

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
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text.append(page.get_text("text"))
        return "\n".join(text)

    if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and ocr:
        image = Image.open(path)
        return pytesseract.image_to_string(image)

    return path.read_text(errors="ignore")


__all__ = ["discover_documents", "extract_text"]


class DocumentChunker:
    """Simple text chunker used by agents for splitting large documents."""

    def __init__(self, chunk_size: int = 4000, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        step = max(1, self.chunk_size - self.overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]


class LegalDocumentClassifier:
    """Very lightweight keyword based classifier for legal documents."""

    KEYWORD_SETS = {
        "contract": ["agreement", "party", "contract"],
        "court_filing": ["plaintiff", "defendant", "court"],
        "statute": ["section", "subsection", "act"],
    }

    def classify(self, text: str, filename: str | None = None) -> Dict[str, Any]:
        lowered = text.lower()
        best_type = "unknown"
        best_score = 0.0
        for doc_type, keywords in self.KEYWORD_SETS.items():
            hits = sum(1 for kw in keywords if kw in lowered)
            score = hits / len(keywords)
            if score > best_score:
                best_type = doc_type
                best_score = score
        return {
            "is_legal_document": best_score > 0,
            "primary_type": best_type,
            "primary_score": best_score,
            "filename": filename,
        }


__all__ = [
    "discover_documents",
    "extract_text",
    "DocumentChunker",
    "LegalDocumentClassifier",
]
