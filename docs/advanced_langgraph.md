# Advanced LangGraph Workflows

This page describes how to enable and use the optional `langgraph` package for more complex document processing flows. LangGraph allows the system to route documents through specialized subgraphs based on classification results and exposes real‑time progress events over WebSocket.

## Installation

`langgraph` is not required for basic features. Install it only when you need the enhanced workflow engine:

```bash
pip install langgraph==0.0.20
```

If you use Poetry:

```bash
poetry add langgraph@0.0.20
```

After installation the default stubs under the `langgraph/` folder will automatically defer to the real package when available.

## Document Classification Routing

When enabled, uploaded documents are first classified. Each category (e.g. contract, brief, correspondence) has its own LangGraph subgraph optimised for that type. The orchestrator chooses the subgraph based on the classification result and then executes the appropriate nodes.

```
                     ┌─────────────┐
input document ───► │ classifier  │
                     └────┬────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
      contract         litigation      general
       subgraph        subgraph        subgraph
```

Each subgraph can use different agents or processing steps. See [`WorkflowOrchestrator`](../legal_ai_system/services/workflow_orchestrator.py) for how the routing logic is applied.

## Monitoring via WebSocket

During execution each node emits progress updates using the existing `ConnectionManager`. Clients subscribe to a document‑specific topic and receive events such as `node_started`, `node_finished`, and custom status messages. This mirrors the behaviour shown in the [Integration Guide](integration_plan.md) for real‑time metrics.

## CaseWorkflowState Example

`CaseWorkflowState` is a simple pydantic model capturing the current node, accumulated results and any classification tags. When using LangGraph you pass this state object through the graph so each node can update it:

```python
from pydantic import BaseModel
from langgraph import StateGraph

class CaseWorkflowState(BaseModel):
    current_node: str
    document_id: str
    tags: list[str] = []
    data: dict = {}

async def classify(state: CaseWorkflowState) -> CaseWorkflowState:
    state.tags.append("brief")
    state.current_node = "classify"
    return state

workflow = StateGraph()
workflow.add_node("classify", classify)
workflow.set_entry_point("classify")
workflow.add_edge("classify", langgraph.END)
result = workflow.run(CaseWorkflowState(document_id="123"))
```

This pattern allows every stage to augment the state and makes the entire workflow easily inspectable.

