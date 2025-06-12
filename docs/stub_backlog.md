# Stub Backlog

This backlog tracks all modules currently marked with `AGENT_STUB` comments. Each entry records the
file path, a short description of its intended functionality, the date it was flagged, and the
current status or plan for implementation. Update this list whenever a stub is added or removed.

| File Path | Description | Date Flagged | Status/Plan |
|-----------|-------------|--------------|-------------|
| `legal_ai_system/legal_ai_network/__init__.py` | Networking stubs for the integrated GUI. | 2025-06-12 | Replace with asynchronous API client and WebSocket implementation. |
| `legal_ai_system/legal_ai_database/__init__.py` | Simplified database utilities and preferences storage. | 2025-06-12 | Implement production database layer with caching and persistence. |
| `legal_ai_system/legal_ai_desktop/__init__.py` | Minimal PyQt6 desktop UI components for early prototypes. | 2025-06-12 | Remove once features are migrated to the main GUI. |
| `legal_ai_system/legal_ai_widgets/__init__.py` | Demo widget collection for prototype UI elements. | 2025-06-12 | Integrate useful widgets into `gui/widgets` and delete the rest. |
| `legal_ai_system/legal_ai_charts/__init__.py` | Thin wrapper re-exporting chart widgets. | 2025-06-12 | Replace with direct imports after chart modules are finalized. |

