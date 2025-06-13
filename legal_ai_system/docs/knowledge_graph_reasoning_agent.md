# KnowledgeGraphReasoningAgent

The `KnowledgeGraphReasoningAgent` builds on the knowledge graph managed by
`KnowledgeGraphManager` to derive new insights and answer complex legal queries.

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_depth` | Maximum hop depth when traversing the graph for reasoning. | `3` |
| `confidence_threshold` | Minimum edge confidence required to include inferred relationships. | `0.5` |
| `enable_link_prediction` | If `true`, enables probabilistic link prediction to discover implicit edges. | `true` |
| `return_explanation` | When set, the agent returns the reasoning path for each inference. | `false` |

## Example Usage

```python
from legal_ai_system.agents.knowledge_graph_reasoning_agent import KnowledgeGraphReasoningAgent

services = ServiceContainer()
reasoner = KnowledgeGraphReasoningAgent(services, max_depth=2)
result = await reasoner.infer("What precedent supports argument X?")
```

The result contains any discovered nodes, relationships, and optional explanatory
paths if `return_explanation` is enabled.
