"""Pydantic models for workflow inputs and outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NodeInput(BaseModel):
    """Base model for inputs to workflow nodes."""

    payload: Any = Field(..., description="Input payload")


class NodeOutput(BaseModel):
    """Base model for outputs from workflow nodes."""

    result: Any = Field(..., description="Output result")
