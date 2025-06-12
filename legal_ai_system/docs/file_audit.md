# File Audit

This document inventories the current contents of each consolidated top-level folder under `legal_ai_system` and summarizes dependency analysis results.

## Inventory
```
## aioredis
legal_ai_system/aioredis/__init__.py

## config
legal_ai_system/config/__init__.py
legal_ai_system/config/agent_unified_config.py
legal_ai_system/config/agent_unified_config.pyi
legal_ai_system/config/constants.py
legal_ai_system/config/defaults.yaml
legal_ai_system/config/entity_patterns.yaml
legal_ai_system/config/relationship_patterns.yaml
legal_ai_system/config/settings.py
legal_ai_system/config/workflow_config.py

## docs
legal_ai_system/docs/ENV_SETUP.md
legal_ai_system/docs/advanced_langgraph.md
legal_ai_system/docs/api_endpoints.md
legal_ai_system/docs/application_assessment.md
legal_ai_system/docs/cleanup_backlog.md
legal_ai_system/docs/design-system.md
legal_ai_system/docs/developer_onboarding.md
legal_ai_system/docs/diagrams/realtime_analysis_workflow_sequence.md
legal_ai_system/docs/diagrams/realtime_analysis_workflow_sequence.puml
legal_ai_system/docs/diagrams/realtime_analysis_workflow_sequence.svg
legal_ai_system/docs/diagrams/service_container_initialization.md
legal_ai_system/docs/diagrams/service_container_initialization.puml
legal_ai_system/docs/diagrams/service_container_initialization.svg
legal_ai_system/docs/folder_cleanup_log.md
legal_ai_system/docs/gui_setup.md
legal_ai_system/docs/integration_plan.md
legal_ai_system/docs/knowledge_graph_reasoning_agent.md
legal_ai_system/docs/legacy/removed_backends.md
legal_ai_system/docs/legacy/violation_review_agent_original.md
legal_ai_system/docs/monitoring.md
legal_ai_system/docs/precedent_matching_agent.md
legal_ai_system/docs/shared_memory.md
legal_ai_system/docs/system_layout.md
legal_ai_system/docs/tasks/agent_tasks.md
legal_ai_system/docs/test_setup.md

## frontend
legal_ai_system/frontend/index.html
legal_ai_system/frontend/package-lock.json
legal_ai_system/frontend/package.json
legal_ai_system/frontend/public/data/sample_documents.json
legal_ai_system/frontend/public/data/sample_memory_entries.json
legal_ai_system/frontend/src/api/client.ts
legal_ai_system/frontend/src/apiClient.ts
legal_ai_system/frontend/src/components/AsyncErrorBoundary.tsx
legal_ai_system/frontend/src/components/DocumentProcessing.tsx
legal_ai_system/frontend/src/components/ErrorBoundary.tsx
legal_ai_system/frontend/src/components/Login.tsx
legal_ai_system/frontend/src/components/ProgressiveLoader.tsx
legal_ai_system/frontend/src/components/ReviewQueueWindow.tsx
legal_ai_system/frontend/src/components/StatusDashboard.tsx
legal_ai_system/frontend/src/components/WorkflowDesigner.tsx
legal_ai_system/frontend/src/components/skeletons/Skeleton.tsx
legal_ai_system/frontend/src/components/skeletons/TableSkeleton.tsx
legal_ai_system/frontend/src/contexts/AuthContext.tsx
legal_ai_system/frontend/src/design-system/components/Button.tsx
legal_ai_system/frontend/src/design-system/components/Card.tsx
legal_ai_system/frontend/src/design-system/components/Grid.tsx
legal_ai_system/frontend/src/design-system/components/Input.tsx
legal_ai_system/frontend/src/design-system/index.ts
legal_ai_system/frontend/src/design-system/tokens.ts
legal_ai_system/frontend/src/hooks/useDocumentUpdates.ts
legal_ai_system/frontend/src/hooks/useLoadingState.ts
legal_ai_system/frontend/src/hooks/useMetrics.ts
legal_ai_system/frontend/src/hooks/useRealtimeSystemStatus.ts
legal_ai_system/frontend/src/hooks/useWebSocket.ts
legal_ai_system/frontend/src/index.tsx
legal_ai_system/frontend/src/legal-ai-gui.tsx
legal_ai_system/frontend/src/types/index.ts
legal_ai_system/frontend/src/types/review.ts
legal_ai_system/frontend/src/types/status.ts
legal_ai_system/frontend/src/types/violation.ts
legal_ai_system/frontend/src/types/workflow.ts
legal_ai_system/frontend/tsconfig.json
legal_ai_system/frontend/vite.config.ts

## langgraph
legal_ai_system/langgraph/__init__.py
legal_ai_system/langgraph/graph.py

## legal_ai_charts
legal_ai_system/legal_ai_charts/__init__.py

## legal_ai_database
legal_ai_system/legal_ai_database/__init__.py

## legal_ai_desktop
legal_ai_system/legal_ai_desktop/__init__.py

## legal_ai_network
legal_ai_system/legal_ai_network/__init__.py

## legal_ai_widgets
legal_ai_system/legal_ai_widgets/__init__.py

## scripts
legal_ai_system/scripts/check_optional_dependencies.py
legal_ai_system/scripts/fix_langgraph_typing.py
legal_ai_system/scripts/install_all_dependencies.py
legal_ai_system/scripts/migrate_database.py
legal_ai_system/scripts/run_tests.sh
legal_ai_system/scripts/run_tool_cli.py
legal_ai_system/scripts/setup_environment_task.py
legal_ai_system/scripts/start_linkage_check.py

```

## Dependency Analysis

### Python (pydeps & vulture)

- pydeps was executed to generate an SVG dependency graph for the entire package using:
  `python -m pydeps --noshow -o legal_ai_system.svg legal_ai_system`
- vulture identified several unused imports and variables. Example output:
```
_RealBaseNode  # unused import (legal_ai_system/agents/agent_nodes.py:20)
original_doc_content  # unused variable (legal_ai_system/agents/structural_analysis_agent.py:289)
min_similarity  # unused variable (legal_ai_system/core/enhanced_vector_store.py:689)
max_depth  # unused variable (legal_ai_system/tests/test_knowledge_graph_reasoning_agent.py:31)
relationship_types  # unused variable (legal_ai_system/tests/test_knowledge_graph_reasoning_agent.py:31)
_RealStateGraph  # unused import (legal_ai_system/workflows/advanced_langgraph.py:30)
_RealEND  # unused import (legal_ai_system/workflows/advanced_langgraph.py:31)
_RealStateGraph  # unused import (legal_ai_system/workflows/langgraph_setup.py:34)
SystemInitializationError  # unused import (legal_ai_system/services/service_container.py:39)
SystemInitializationError  # unused import (legal_ai_system/services/service_container.py:53)
SystemInitializationError  # unused import (legal_ai_system/services/service_container.py:67)
```

### TypeScript (depcruise)

- Dependency Cruiser was run with `--no-config` on `frontend/src`.
- src/apiClient.ts was reported as an orphan module with no dependents.
```
14:      "source": "src/apiClient.ts",
```

### Frontend Component Usage Audit

Using `grep`, the following components were not imported anywhere in `frontend/src`:

- `components/MetricsChart.tsx`
- `components/ReviewQueue.tsx`
- `components/SystemHealth.tsx`
- `components/skeletons/CardSkeleton.tsx`
- `components/skeletons/DashboardSkeleton.tsx`
- `design-system/components/Alert.tsx`

These unused components were removed from the source tree to reduce maintenance
overhead. If future features require them, they can be restored from version
history.

## Candidates for Cleanup

The following packages are mostly stubs and were previously flagged in `cleanup_backlog.md`:

- legal_ai_desktop
- legal_ai_widgets
- legal_ai_charts
- legal_ai_network
- legal_ai_database
- aioredis
- langgraph

These modules contain minimal code and are primarily re-export or placeholder implementations. They can likely be removed or consolidated once the PyQt6 GUI (`legal_ai_system/gui/legal_ai_pyqt6_integrated.py`) is fully adopted.

## Documentation Review

The following documentation files reference modules or behaviour that no longer
match the current codebase:

- `gui_setup.md` – states that running `python -m legal_ai_system` launches the
  Streamlit dashboard. The entry point now starts the PyQt6 GUI located at
  `gui/legal_ai_pyqt6_integrated.py`. Update the instructions accordingly.
- `api_endpoints.md` – describes REST endpoints for a FastAPI backend, but the
  repository does not include the corresponding FastAPI application. Consider
  removing this file or replacing it with up‑to‑date API documentation when the
  backend is implemented.
- `legacy/violation_review_agent_original.md` – contains archived code for a
  deleted agent. Keep only if historical reference is required; otherwise it can
  be removed.
- `legacy/removed_backends.md` – documents removed FastAPI servers. This file is
  redundant once those services are fully retired.

Please update or prune these documents during the next cleanup cycle.
