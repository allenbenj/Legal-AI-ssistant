# Code Cleanup Backlog

The following modules or folders either duplicate functionality that now lives elsewhere or provide stub implementations. They should be cleaned up once the system is fully integrated with the real services.

| Module/Folder | Description | Action |
|---------------|-------------|--------|
| `legal_ai_network` | Stubbed network classes for the GUI. | **Replace** with real network service |
| `legal_ai_database` | Stub database and cache managers. | **Replace** with production database layer |
| `aioredis` | Thin alias to `redis.asyncio` to avoid optional dependency. | **Remove** after adopting official client |
| `langgraph` | Local stub for optional `langgraph` library. | **Replace** with real library when available |
| `legal_ai_system/gui/streamlit_app.py` | Older Streamlit-based interface that duplicates the PyQt6 desktop features. | **Remove** once the PyQt6 GUI is stable |
| `legal_ai_system/agents/legal_reasoning_engine.py` | Placeholder agent for reasoning over a knowledge graph. | **Replace** with full implementation |
| `legal_ai_system/services/realtime_nodes.py` | Placeholder classes for workflow nodes. | **Replace** with fully implemented nodes |

Entries like the deprecated `integrated_gui.py` script have already been removed as noted in `docs/legacy/removed_backends.md`.

