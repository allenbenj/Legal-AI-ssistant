"""Document Rewriter Agent.

This agent performs lightweight text rewriting on extracted document text
prior to further analysis. It focuses on spelling correction using the
``pyspellchecker`` package so downstream agents operate on cleaner input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pyspellchecker import SpellChecker

from ..core.base_agent import BaseAgent


@dataclass
class DocumentRewriteResult:
    """Result returned from :class:`DocumentRewriterAgent`."""

    corrected_text: str
    corrections: List[Dict[str, str]] = field(default_factory=list)
    original_length: int = 0
    corrected_length: int = 0
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentRewriterAgent(BaseAgent):
    """Agent that performs basic spelling corrections on text."""

    def __init__(self, service_container: Any, **config: Any) -> None:
        super().__init__(service_container, name="DocumentRewriterAgent", agent_type="rewrite")

        self.spell_checker = SpellChecker(language=config.get("language", "en"))
        self.config.update(config)

        self.logger.info("DocumentRewriterAgent initialized", parameters={"language": self.spell_checker.language})

    async def rewrite_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentRewriteResult:
        """Public helper that rewrites text and returns :class:`DocumentRewriteResult`."""

        result = await self.process(text, metadata=metadata)
        if result.success and isinstance(result.data, dict):
            return DocumentRewriteResult(**result.data)
        return DocumentRewriteResult(corrected_text=text, corrections=[], original_length=len(text), corrected_length=len(text))

    async def _process_task(self, task_data: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)

        text = task_data or ""
        corrections: List[Dict[str, str]] = []
        corrected_tokens: List[str] = []

        # Simple tokenisation preserving punctuation
        tokens = re.findall(r"\w+|\W+", text, re.UNICODE)

        for token in tokens:
            if token.isalpha():
                lower = token.lower()
                if lower in self.spell_checker:  # word is correct
                    corrected_tokens.append(token)
                    continue

                suggestion = self.spell_checker.correction(lower)
                if suggestion and suggestion != lower:
                    corrected_word = suggestion
                    # Preserve capitalisation if token was capitalised
                    if token[0].isupper():
                        corrected_word = suggestion.capitalize()

                    corrections.append({"original": token, "corrected": corrected_word})
                    corrected_tokens.append(corrected_word)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)

        corrected_text = "".join(corrected_tokens)
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return {
            "corrected_text": corrected_text,
            "corrections": corrections,
            "original_length": len(text),
            "corrected_length": len(corrected_text),
            "processing_time_sec": round(processing_time, 3),
        }

