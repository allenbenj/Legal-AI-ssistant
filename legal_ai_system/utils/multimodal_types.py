"""Data structures for handling multimodal document content."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProcessedImage:
    """Image extracted from a document with optional OCR text."""

    file_path: str
    caption: Optional[str] = None
    ocr_text: Optional[str] = None


@dataclass
class StructuredTable:
    """Tabular data parsed from a document."""

    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None


@dataclass
class ProcessedChart:
    """Chart or plot extracted from a document."""

    chart_type: str
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class AudioSegment:
    """Segment of audio with a transcript."""

    start_ms: int
    end_ms: int
    transcript: str


@dataclass
class VideoFrame:
    """Key video frame with description."""

    timestamp_ms: int
    image_path: str
    description: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Basic metadata describing a document."""

    title: Optional[str] = None
    author: Optional[str] = None
    created: Optional[str] = None
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalDocument:
    """Unified container for multimodal document content."""

    text_content: str
    images: List[ProcessedImage] = field(default_factory=list)
    tables: List[StructuredTable] = field(default_factory=list)
    charts: List[ProcessedChart] = field(default_factory=list)
    audio_transcripts: Optional[List[AudioSegment]] = None
    video_keyframes: Optional[List[VideoFrame]] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


__all__ = [
    "ProcessedImage",
    "StructuredTable",
    "ProcessedChart",
    "AudioSegment",
    "VideoFrame",
    "DocumentMetadata",
    "MultiModalDocument",
]
