"""Optional pydantic models for workflow nodes."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class NodeInput(BaseModel):
    """Base model for node inputs."""

    data: Any


class NodeOutput(BaseModel):
    """Base model for node outputs."""

    result: Any
