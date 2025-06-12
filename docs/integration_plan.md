# Integration Guide

This guide outlines the five-phase plan for integrating the Legal AI System, common WebSocket usage patterns, deployment tips, and operational best practices. Additional sections cover security considerations, testing suggestions, success metrics, and troubleshooting advice.

## Integration Roadmap

1. **Environment Preparation** – Install Python and Node dependencies as described in [ENV_SETUP.md](ENV_SETUP.md). Verify that packages such as `fastapi`, `uvicorn`, and database drivers install correctly.
2. **Service Container Configuration** – Use the `create_service_container` function to wire services together in the correct order. The initialization order is summarised in [system_layout.md](system_layout.md).
3. **API and WebSocket Endpoints** – Expose REST endpoints from the integrated application and set up the `ConnectionManager` for real-time updates. React hooks from `frontend/src/hooks` subscribe to updates during document processing.
4. **Deployment** – Run the FastAPI app with Uvicorn or inside Docker. Ensure environment variables for database connections and secret keys are provided. Build the frontend once and serve the static files from `frontend/dist`.
5. **Monitoring and Optimization** – Start the `RealtimePublisher` to broadcast system metrics and log performance. Tune database connections and vector stores based on load patterns.

6. **Enhanced LangGraph Workflow** – Install the optional `langgraph` dependency, then follow [advanced_langgraph.md](advanced_langgraph.md) for setup instructions, classification routing, WebSocket progress monitoring, and the `CaseWorkflowState` example.

## WebSocket Patterns

The `ConnectionManager` manages connections, subscriptions, and topic-based broadcasts:

```python
class ConnectionManager:
    """Manage WebSocket connections and topic subscriptions."""
    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
```
【F:legal_ai_system/api/websocket_manager.py†L31-L38】

Clients connect and subscribe to specific topics:

```python
async def subscribe(self, client_id: str, topic: str) -> None:
    self.subscriptions[client_id].add(topic)
    self.topic_subscribers[topic].add(client_id)
    await self.send_personal_message({"type": "subscription_ack", "topic": topic}, client_id)
```
【F:legal_ai_system/api/websocket_manager.py†L76-L83】

The backend broadcasts system status periodically using `RealtimePublisher`:

```python
class RealtimePublisher:
    """Publish system metrics periodically over WebSocket."""
    def start_system_monitoring(self, interval: float = 1.0) -> None:
        self._task = asyncio.create_task(self._monitor_loop(interval))
```
【F:legal_ai_system/services/realtime_publisher.py†L29-L37】【F:legal_ai_system/services/realtime_publisher.py†L42-L48】

## Deployment Tips

- Build the frontend once with `npm run build`. FastAPI serves the contents of `frontend/dist` automatically when present.
- Store secrets and database credentials as environment variables to avoid committing them to source control.
- When deploying with Docker, map persistent volumes for uploads and ensure port mappings for both HTTP and WebSocket endpoints.

## Security

Follow the repository [Security Policy](../SECURITY.md) and keep dependencies up to date. Use a secrets manager for credentials and restrict network access to databases whenever possible.

## Testing

Run the automated tests with `pytest` after installing development dependencies. See [test_setup.md](test_setup.md) for details. Mock heavy optional dependencies to keep the suite fast.

## Success Metrics

Track system performance and user satisfaction:

- Average document processing time.
- Error rate during workflow execution.
- WebSocket latency for real-time updates.
- User feedback scores or support tickets.

## Troubleshooting

**Import paths** – When importing design-system components outside of `frontend/src`, use a relative path:

```tsx
import { Button } from '../../frontend/src/design-system';
```
【F:docs/design-system.md†L34-L39】

**Service container initialization** – Ensure services are registered in the order expected by `create_service_container`:

1. `ConfigurationManager`
2. `PersistenceManager` and `UserRepository`
3. `SecurityManager` and authentication services
4. `LLMManager`, `ModelSwitcher`, and `EmbeddingManager`
5. `KnowledgeGraphManager`, `VectorStore`, then `RealTimeGraphManager`
6. `UnifiedMemoryManager` and `ReviewableMemory`
7. `ViolationReviewDB`
8. `RealTimeAnalysisWorkflow` and agent factories
9. LangGraph node factories and `WorkflowOrchestrator`
10. Finally call `initialize_all_services`
【F:docs/system_layout.md†L141-L154】

If initialization fails, check for missing dependencies or incorrect configuration keys in the service container.
