# Legal AI System - Dependency Flow Diagrams

## Core Service Container Flow

```mermaid
graph TD
    A[unified_services.py] --> B[ServiceContainer]
    B --> C[get_service_container()]
    C --> D[40+ Files Import This]
    
    D --> E[api/main.py]
    D --> F[agents/*.py]
    D --> G[core/*.py]
    D --> H[workflows/*.py]
    
    B --> I[LazyServiceProxy]
    I --> J[Resolves Circular Dependencies]
    
    B --> K[ServiceState Enum]
    K --> L[INITIALIZING]
    K --> M[RUNNING] 
    K --> N[SHUTTING_DOWN]
    K --> O[SHUTDOWN]
```

## Agent Inheritance Hierarchy

```mermaid
graph TD
    A[base_agent.py - BaseAgent] --> B[document_processor.py]
    A --> C[auto_tagging.py] 
    A --> D[citation_analysis.py]
    A --> E[entity_extraction.py]
    A --> F[legal_analysis.py]
    A --> G[note_taking.py]
    A --> H[ontology_extraction.py]
    A --> I[semantic_analysis.py]
    A --> J[structural_analysis.py]
    A --> K[text_correction.py]
    A --> L[violation_detector.py]
    A --> M[knowledge_base_agent.py]
    
    N[unified_services.py] --> A
    O[detailed_logging.py] --> A
    P[llm_providers.py] --> A
```

## LLM Provider Chain

```mermaid
graph LR
    A[llm_providers.py] --> B[BaseLLMProvider]
    B --> C[OllamaProvider]
    B --> D[OpenAIProvider] 
    B --> E[XAIProvider]
    
    F[LLMManager] --> B
    F --> G[provider_fallback()]
    G --> H[Primary: xAI/Grok]
    H --> I[Fallback: Ollama]
    
    J[model_switcher.py] --> F
    J --> K[GrokModelSwitcher]
    K --> L[grok-3-mini]
    K --> M[grok-3-reasoning]
    K --> N[grok-2-1212]
```

## Memory and Knowledge Management

```mermaid
graph TD
    A[unified_memory_manager.py] --> B[MemoryManager]
    B --> C[SessionMemory]
    B --> D[EntityMemory]
    B --> E[DocumentMemory]
    
    F[knowledge_graph_enhanced.py] --> G[EnhancedKnowledgeGraph]
    G --> H[Neo4j Integration]
    G --> I[Entity Deduplication]
    G --> J[Relationship Extraction]
    
    K[vector_store_enhanced.py] --> L[EnhancedVectorStore]
    L --> M[FAISS Backend]
    L --> N[LanceDB Backend]
    L --> O[Hybrid Storage]
    
    B --> G
    B --> L
```

## API and Frontend Integration

```mermaid
graph LR
    A[React Frontend] --> B[enhanced-legal-ai-gui2.tsx]
    B --> C[API_BASE_URL: :8000/api/v1]
    B --> D[GRAPHQL_URL: :8000/graphql]
    B --> E[WS_URL: ws://:8000/ws]
    
    F[api/main.py] --> G[FastAPI Backend]
    G --> H[REST Endpoints]
    G --> I[GraphQL Schema]
    G --> J[WebSocket Handler]
    
    K[integration_service.py] --> G
    L[unified_services.py] --> G
    M[security_manager.py] --> G
```

## Workflow Orchestration

```mermaid
graph TD
    A[ultimate_orchestrator.py] --> B[UltimateOrchestrator]
    B --> C[workflow_state_manager.py]
    C --> D[WorkflowStateManager]
    
    E[realtime_analysis_workflow.py] --> F[RealTimeAnalysisWorkflow]
    F --> G[Document Processing]
    F --> H[Entity Extraction]
    F --> I[Knowledge Graph Update]
    F --> J[Vector Store Update]
    
    B --> F
    B --> K[ontology_integration.py]
    K --> L[OntologyIntegrationWorkflow]
```

## Error Handling and Recovery

```mermaid
graph TD
    A[error_recovery.py] --> B[ErrorRecoveryManager]
    B --> C[CircuitBreaker]
    B --> D[RetryManager]
    B --> E[FallbackManager]
    
    F[unified_exceptions.py] --> G[LegalAIException]
    G --> H[ServiceException]
    G --> I[LLMException]
    G --> J[VectorStoreException]
    G --> K[KnowledgeGraphException]
    
    L[detailed_logging.py] --> M[get_detailed_logger()]
    M --> N[66 Files Use This]
```

## Security and Configuration

```mermaid
graph LR
    A[security_manager.py] --> B[SecurityManager]
    B --> C[Authentication]
    B --> D[Authorization]
    B --> E[Input Validation]
    
    F[configuration_manager.py] --> G[ConfigurationManager]
    G --> H[Environment Variables]
    G --> I[Settings Validation]
    G --> J[Dynamic Updates]
    
    K[settings.py] --> L[Pydantic Models]
    L --> M[Database Config]
    L --> N[LLM Config] 
    L --> O[Storage Config]
```