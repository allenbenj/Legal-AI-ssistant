# Shared Memory Usage

The unified memory manager now exposes helper methods for storing data that is accessible to all agents within a session.

Use `store_shared_memory` and `retrieve_shared_memory` from `AgentConfigHelper` or the memory mixin to exchange information between agents.

```python
helper.store_shared_memory(key="summary", value={"text": "..."}, session_id=session_id)
other = helper.retrieve_shared_memory(key="summary", session_id=session_id)
```

Agents mixing in `AgentMemoryMixin` gain `store_shared_memory()` and `retrieve_shared_memory()` methods to work with shared session knowledge.
