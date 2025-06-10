from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import difflib


def transcribe_audio(path: Path) -> str:
    """Return a dummy transcript for the given audio file."""
    if not path.exists():
        raise FileNotFoundError(path)
    # In a real system we'd invoke an ASR engine. For tests we return a
    # deterministic transcript based on file name.
    return "Test audio transcript"


def process_video_deposition(path: Path) -> List[Dict[str, str]]:
    """Return dummy speaker-labelled text for a deposition video."""
    if not path.exists():
        raise FileNotFoundError(path)
    # Real implementation would perform speaker diarization + transcription.
    return [
        {"speaker": "Attorney", "text": "Please state your name."},
        {"speaker": "Witness", "text": "John Doe."},
    ]


def extract_form_fields(path: Path) -> Dict[str, str]:
    """Parse simple key/value pairs from a text form."""
    if not path.exists():
        raise FileNotFoundError(path)
    fields: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def analyze_redline(original: str, revised: str) -> Dict[str, List[str]]:
    """Return inserted and deleted lines between two versions."""
    diff = list(difflib.ndiff(original.splitlines(), revised.splitlines()))
    insertions = [line[2:] for line in diff if line.startswith("+ ")]
    deletions = [line[2:] for line in diff if line.startswith("- ")]
    return {"insertions": insertions, "deletions": deletions}

__all__ = [
    "transcribe_audio",
    "process_video_deposition",
    "extract_form_fields",
    "analyze_redline",
]
