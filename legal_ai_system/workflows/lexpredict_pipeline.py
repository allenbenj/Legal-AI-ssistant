"""LexPredict pipeline wrappers.

This module provides optional wrappers around the ``lexnlp`` package from
LexPredict. When ``lexnlp`` is installed, users may create pipelines using
predefined options from ``LEXPREDICT_STORE_OPTIONS``. Each pipeline groups a
set of LexNLP extractors for quick text analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    import lexnlp.extract.en.dates as lex_dates
    import lexnlp.extract.en.amounts as lex_amounts
    import lexnlp.extract.en.citations as lex_citations
    import lexnlp.extract.en.courts as lex_courts

    LEXNLP_AVAILABLE = True
except Exception:  # pragma: no cover - lexnlp missing
    LEXNLP_AVAILABLE = False


if LEXNLP_AVAILABLE:
    # Available pipeline configurations mapping name -> components
    LEXPREDICT_STORE_OPTIONS: Dict[str, List[str]] = {
        "basic_info": ["dates", "amounts"],
        "contract_entities": ["dates", "amounts", "citations"],
        "case_citations": ["dates", "citations", "courts"],
    }

    def list_available_pipelines() -> List[str]:
        """Return names of available LexPredict pipelines."""

        return list(LEXPREDICT_STORE_OPTIONS.keys())

    class LexPredictPipeline:
        """Simple wrapper executing a set of LexNLP extractors."""

        def __init__(self, pipeline_type: str = "basic_info") -> None:
            if pipeline_type not in LEXPREDICT_STORE_OPTIONS:
                raise ValueError(
                    f"Unknown pipeline type '{pipeline_type}'. Available: "
                    f"{list(LEXPREDICT_STORE_OPTIONS.keys())}"
                )
            self.pipeline_type = pipeline_type
            self.components = LEXPREDICT_STORE_OPTIONS[pipeline_type]

        def run(self, text: str) -> Dict[str, Any]:
            """Execute the pipeline on the provided text."""

            results: Dict[str, Any] = {}
            if "dates" in self.components:
                results["dates"] = list(lex_dates.get_raw_dates(text))
            if "amounts" in self.components:
                results["amounts"] = list(lex_amounts.get_amounts(text))
            if "citations" in self.components:
                results["citations"] = list(lex_citations.get_citations(text))
            if "courts" in self.components:
                results["courts"] = list(lex_courts.get_courts(text))
            return results

    __all__ = ["LexPredictPipeline", "list_available_pipelines"]
else:

    class LexPredictPipeline:
        """Placeholder pipeline when ``lexnlp`` is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "LexPredict's lexnlp package is required to use LexPredictPipeline"
            )

    def list_available_pipelines() -> List[str]:  # pragma: no cover
        raise RuntimeError("lexnlp is required to list LexPredict pipelines")

    __all__ = ["LexPredictPipeline", "list_available_pipelines"]
