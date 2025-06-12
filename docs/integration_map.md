# Integration Map

This document summarizes how the main components of the Legal AI System fit together.
It complements `legal_ai_system/docs/system_layout.md` but focuses on the high level
relationships between services, agents, workflows and the GUI/API entry points.

## Core Layers

1. **Service Container** – created by `create_service_container()`.
   It registers and initializes all services and agents.  Services include
   persistence, vector stores, analytics, and workflow orchestration.
2. **Agents** – dynamically registered from `legal_ai_system/agents` via
   the service container.  Each agent can be fetched by name from the container
   and used within workflows.
3. **Workflow Engine** – `WorkflowOrchestrator` obtains agents and managers
   from the container to run document processing pipelines such as
   `RealTimeAnalysisWorkflow`.
4. **API & GUI** – the FastAPI backend and the PyQt6 GUI use the
   `LegalAIIntegrationService` which itself relies on the container.  The
   GUI connects through `BackendBridge` so user actions trigger backend workflows.
5. **Analytics & Quality** – `KeywordExtractionService` and
   `QualityAssessmentService` provide metrics for dashboards and automated
   quality checks.

## Data Flow

1. Documents are uploaded via the API or GUI.
2. The integration service saves the file using the persistence layer and
   triggers a workflow through the orchestrator.
3. Workflows invoke agents in sequence; each agent may read or update
   data in the vector store, memory manager or knowledge graph services.
4. Progress updates are emitted back through the integration service to
   the frontend (GUI or REST clients).

This map should help new developers understand which modules interact and
where to extend the system for new services or agents.
