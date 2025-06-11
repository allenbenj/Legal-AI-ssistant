from __future__ import annotations

"""Agent for matching legal precedents using specialized language models."""

from typing import Any, Dict, List

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from ..core.base_agent import BaseAgent
from ..core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function


class PrecedentMatchingAgent(BaseAgent):
    """Identify similar case law using embeddings and vector search."""

    @detailed_log_function(LogCategory.AGENT)
    def __init__(self, service_container: Any, **config: Any) -> None:
        super().__init__(service_container, name="PrecedentMatchingAgent")
        self.config = config
        self.logger = get_detailed_logger(self.__class__.__name__, LogCategory.AGENT)

        # Load legal‑domain language models
        self.legal_bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.contract_bert = AutoModel.from_pretrained("pile-of-law/legalbert-large-1.7M-2")
        self.case_law_bert = AutoModel.from_pretrained("law-ai/InCaseLawBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # Siamese model for similarity scoring
        self.case_similarity_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

        # Optional services
        self.vector_store = self._get_service("vector_store")
        self.reasoning_chain_model = self._get_service("reasoning_chain_model")

    @detailed_log_function(LogCategory.AGENT)
    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Any:
        case_facts = task_data.get("case_facts", "")
        top_k = int(task_data.get("top_k", 5))
        results = await self.match_precedents(case_facts, top_k=top_k)
        return {"matches": results}

    async def match_precedents(self, case_facts: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return similar cases from the vector store with optional reasoning score."""
        if not case_facts:
            return []

        query_embedding = self.case_similarity_model.encode(case_facts, convert_to_tensor=False)
        matches: List[Dict[str, Any]] = []

        if self.vector_store:
            search_results = await self.vector_store.search_similar_async(
                case_facts,
                index_target="document",
                top_k=top_k,
            )
        else:
            search_results = []

        for sr in search_results:
            reasoning_score = 0.0
            if self.reasoning_chain_model and hasattr(sr.metadata, "facts"):
                try:
                    reasoning_score = self.reasoning_chain_model.compare_reasoning(case_facts, sr.metadata.facts)
                except Exception:
                    pass
            matches.append(
                {
                    "case_id": sr.document_id,
                    "similarity": sr.similarity_score,
                    "reasoning_score": reasoning_score,
                }
            )
        return matches

__all__ = ["PrecedentMatchingAgent"]
