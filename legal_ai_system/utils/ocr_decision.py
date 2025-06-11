from __future__ import annotations

"""Heuristics for determining if OCR is needed and recommended settings."""

from pathlib import Path
from typing import Dict, Any

try:
    from langdetect import detect, LangDetectException
except Exception:  # pragma: no cover - optional dependency
    detect = None
    LangDetectException = Exception  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore


def predict_ocr_settings(
    path: Path, default_lang: str = "eng", default_dpi: int = 300
) -> Dict[str, Any]:
    """Return OCR recommendation for *path*.

    The result dictionary contains ``ocr_needed``, ``language`` and ``dpi`` keys.
    ``language`` is derived using ``langdetect`` when possible.
    ``dpi`` is estimated from image metadata when available.
    """

    result = {"ocr_needed": False, "language": default_lang, "dpi": default_dpi}

    if not path.exists():
        return result

    ext = path.suffix.lower()

    # Image types always need OCR
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        result["ocr_needed"] = True
        if Image:
            try:
                with Image.open(path) as img:
                    dpi_info = img.info.get("dpi")
                    if dpi_info:
                        result["dpi"] = int(max(dpi_info))
            except Exception:
                pass
        return result

    # Basic PDF check using PyMuPDF if available
    if ext == ".pdf" and fitz:
        try:
            with fitz.open(path) as doc:
                if doc.page_count > 0:
                    first_page = doc.load_page(0)
                    text = first_page.get_text("text").strip()
                    if not text:
                        result["ocr_needed"] = True
                    else:
                        if detect:
                            try:
                                result["language"] = detect(text)
                            except LangDetectException:
                                pass
        except Exception:
            pass
        return result

    # For plain text files try to detect language
    if detect and path.stat().st_size < 5_000_000:
        try:
            sample = path.read_text(errors="ignore")[:1000]
            if sample.strip():
                result["language"] = detect(sample)
        except Exception:
            pass

    return result


__all__ = ["predict_ocr_settings"]
