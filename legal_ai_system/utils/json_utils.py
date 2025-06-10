"""Utility helpers for working with JSON data returned by LLMs."""

from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """Extract JSON content from an LLM response string.

    Parameters
    ----------
    text:
        The raw text returned by the language model. It may contain markdown
        code fences around the JSON payload or extra text before/after the JSON
        object.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON dictionary or an empty dictionary if extraction fails.
    """
    try:
        json_content = text
        if "```json" in text:
            json_content = text.split("```json", 1)[1].split("```", 1)[0]
        elif (
            "```" in text
            and text.strip().startswith("```")
            and text.strip().endswith("```")
        ):
            json_content = text.strip()[3:-3]
        return json.loads(json_content.strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


__all__ = ["extract_json_from_llm_response"]
