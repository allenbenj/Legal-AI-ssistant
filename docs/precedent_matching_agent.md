# PrecedentMatchingAgent

The `PrecedentMatchingAgent` identifies relevant case law by comparing the facts
of a matter with existing documents in the vector store. It leverages several
legal-domain language models and a Siamese similarity model for robust matching.

## Features

- **Legal Language Models** – Loads Legal‑BERT variants for contracts and case
  law.
- **Siamese Similarity** – Uses a sentence-transformer model to encode the query
  and stored cases for comparison.
- **Vector Store Integration** – Retrieves nearest neighbours from the existing
  FAISS index via `search_similar_async`.
- **Reasoning Score (optional)** – When a `reasoning_chain_model` service is
  available, each candidate case can be ranked by a custom reasoning similarity
  metric.

## Example Usage

```python
from legal_ai_system.agents.precedent_matching_agent import PrecedentMatchingAgent

services = ServiceContainer()
matcher = PrecedentMatchingAgent(services)
results = await matcher.match_precedents("facts of the case", top_k=3)
for item in results:
    print(item)
```

The returned items contain the `case_id`, similarity score and optional
reasoning score.
