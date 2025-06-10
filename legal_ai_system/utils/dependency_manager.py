"""Simple dependency checking utilities for optional packages."""

from __future__ import annotations

from typing import Dict


class DependencyManager:
    """Check for optional third-party dependencies at runtime."""

    DEPENDENCY_MAP: Dict[str, Dict[str, str]] = {
        "pymupdf": {"import": "fitz", "package": "PyMuPDF"},
        "docx": {"import": "docx", "package": "python-docx"},
        "pytesseract": {"import": "pytesseract", "package": "pytesseract"},
        "PIL": {"import": "PIL", "package": "Pillow"},
        "pandas": {"import": "pandas", "package": "pandas"},
        "openpyxl": {"import": "openpyxl", "package": "openpyxl"},
        "pptx": {"import": "pptx", "package": "python-pptx"},
        "markdown": {"import": "markdown", "package": "markdown"},
        "bs4": {"import": "bs4", "package": "beautifulsoup4"},
        "striprtf": {"import": "striprtf", "package": "striprtf"},
        "pdfplumber": {"import": "pdfplumber", "package": "pdfplumber"},
        "ffmpeg": {"import": "ffmpeg", "package": "ffmpeg-python"},
        "openai-whisper": {"import": "whisper", "package": "openai-whisper"},
        "whisperx": {"import": "whisperx", "package": "whisperx"},
        "pyannote.audio": {"import": "pyannote.audio", "package": "pyannote.audio"},
    }

    def __init__(self) -> None:
        self._available: Dict[str, bool] = {}

    def check_dependencies(self) -> Dict[str, bool]:
        """Populate the internal availability map."""
        for name, info in self.DEPENDENCY_MAP.items():
            try:
                __import__(info["import"])
                self._available[name] = True
            except Exception:
                self._available[name] = False
        return self._available

    def is_available(self, name: str) -> bool:
        """Return True if *name* dependency was successfully imported."""
        if not self._available:
            self.check_dependencies()
        return self._available.get(name, False)


__all__ = ["DependencyManager"]
