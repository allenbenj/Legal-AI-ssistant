# System Layout

This document summarizes the major agents and services within the **Legal AI System** and how they interact.  The architecture centers on a dependency injection container that wires together agents, managers, and workflows.  Each component registers itself with the container so that initialization order, configuration, and shutdown are handled consistently.

## Core Architecture

- **ServiceContainer** – manages creation, initialization and shutdown of all services and agents.  It provides dependency injection throughout the system.  Source: `services/service_container.py` lines 1‑6 show its purpose.  The container registers factories for each component, initializes them asynchronously, and exposes lifecycle hooks so resources are cleaned up at shutdown.
- **IntegrationService** – acts as a facade between external APIs and internal agents or workflows.  It retrieves services from the container and provides high‑level operations.  Defined in `services/integration_service.py` lines 1‑9.  The service is the typical entry point for the API layer and orchestrates workflows such as document upload and real-time analysis.
- **BaseAgent** – abstract base class providing asynchronous task processing, error handling and service container access for all agents.  It defines standardized result formats and integrates detailed logging.  See `core/base_agent.py` lines 1‑12.
- **RealTimeAnalysisWorkflow** – orchestrates the main document analysis pipeline combining document processing, ontology extraction and knowledge graph updates.  Document processing components are instantiated in its constructor.  The workflow coordinates updates to the knowledge graph and vector store while tracking performance statistics.

## Agents

The project defines several specialized agents under `legal_ai_system/agents`:

- **DocumentProcessorAgent** – extracts text and metadata from a variety of file types.  See `document_processor_agent.py` lines 1‑8.
- Handles PDFs, DOCX, HTML, and image formats using optional dependencies for parsing.  Returns structured content for downstream agents.
- **DocumentProcessorAgentV2** – advanced processor returning a `MultiModalDocument`. See `document_processor_agent_v2.py` lines 25-132.
- Supports `process_video_depositions`, `process_legal_forms`, and `process_contract_redlines` for specialized legal content.
- **DocumentRewriterAgent** – performs lightweight spelling correction on extracted text.  See `document_rewriter_agent.py` lines 1‑7.
- Uses `pyspellchecker` to clean common OCR mistakes before analysis.
- **OntologyExtractionAgent** – extracts legal entities and relationships using ontology‑driven patterns.  See `ontology_extraction_agent.py` lines 1‑7.
- Integrates with the knowledge graph to ensure entities are linked consistently.
- **EntityExtractionAgent** – streamlined entity extraction for legal documents.  See `entity_extraction_agent.py` lines 1‑7.
- Focused on speed and identifies names of parties, statutes, and case citations.
- **SemanticAnalysisAgent** – summarization and legal topic identification.  See `semantic_analysis_agent.py` lines 1‑8.
- Utilizes large language models for document summaries and topic classification.
- **StructuralAnalysisAgent** – identifies IRAC components and structural elements in documents.  See `structural_analysis_agent.py` lines 1‑8.
- Extracts Issue, Rule, Application, and Conclusion sections with structural cues.
- **CitationAnalysisAgent** – detects and classifies legal citations.  See `citation_analysis_agent.py` lines 1‑7.
- Records citation references so they can be cross-linked in the graph.
- **TextCorrectionAgent** – performs grammar and style correction.  See `text_correction_agent.py` lines 1‑8.
- Applies legal writing style guides and enforces consistent formatting.
- **AutoTaggingAgent** – automatic tagging and classification of documents.  See `auto_tagging_agent.py` lines 1‑6.
- Learns from user-provided examples and stores tags with the vector store.
- Generated tags are persisted to the vector store metadata with the originating document ID.
- **KnowledgeBaseAgent** – resolves entities and structures data for the knowledge base.  See `knowledge_base_agent.py` lines 1‑8.
- Persists resolved entities and relationships to the `KnowledgeGraphManager` using a circuit breaker for reliability.
- **KnowledgeGraphReasoningAgent** – infers new relationships by traversing the knowledge graph.  See `knowledge_graph_reasoning_agent.py` lines 1‑8.
- Provides multi-hop reasoning over entities and can answer cross-document legal queries.
- **LegalAnalysisAgent** – performs IRAC analysis with contradiction detection.  See `legal_analysis_agent.py` lines 1‑9.
- Runs deep reasoning and validates legal logic using LLM-based checks.
- **NoteTakingAgent** – generates notes with legal context awareness.  See `note_taking_agent.py` lines 1‑5.
- Summarizes key points and links them back to the originating document.
- **ViolationDetectorAgent** – identifies potential legal violations.  See `violation_detector_agent.py` lines 1‑6.
- Evaluates potential compliance or ethics issues for later review.
- **LegalAuditAgent / EthicsReviewAgent / LEOConductAgent** – lightweight review agents for GUI violation review.  See `legal_agents.py` lines 1‑32.
- Provide quick client-side checks within the GUI when deeper analysis is not required.


## Services

Key services registered with the container include:

- **SecurityManager** – authentication, authorization and PII detection.  Defined in `services/security_manager.py` lines 1‑11.
- Provides encryption utilities and enforces secure access to all agents.
- **MemoryManager** – manages persistent context and agent memory.  Defined in `services/memory_manager.py` lines 1‑9.
- Stores conversation history and workflow results in SQLite or Redis backends.
- **KnowledgeGraphManager** – handles graph storage and entity/relationship management.  Defined in `services/knowledge_graph_manager.py` lines 1‑9.
- Interfaces with Neo4j and manages schema evolution for legal entities.
- **RealTimeGraphManager** – synchronizes the semantic graph with vector store updates.  Defined in `services/realtime_graph_manager.py` lines 1‑9.
- Keeps embeddings and graph entries consistent during real-time analysis.
- **DatabaseManager** – database utilities and setup (not shown in excerpt).
- **VectorStoreManager**, **EmbeddingManager**, **UnifiedMemoryManager** – additional managers located in `core` modules.
- Provide embedding generation, vector search, and unified memory abstraction.

## Workflows and Connections

`RealTimeAnalysisWorkflow` assembles several agents and managers to process a document.  Its constructor creates a `DocumentProcessorAgent`, `DocumentRewriterAgent`, `OntologyExtractionAgent`, `HybridLegalExtractor`, `RealTimeGraphManager`, `EnhancedVectorStore` and `ReviewableMemory`.  Source: `services/realtime_analysis_workflow.py` lines 92‑109.  The workflow coordinates extraction, enrichment, and persistence in a single asynchronous pipeline.

During processing:

1. `DocumentProcessorAgent` extracts text and metadata.
2. `DocumentRewriterAgent` cleans the text.
3. `OntologyExtractionAgent` and `HybridLegalExtractor` identify entities and relationships.
4. `RealTimeGraphManager` updates the knowledge graph and vector store.
5. `ReviewableMemory` stores processed data for later review.

`IntegrationService` exposes methods such as `handle_document_upload` to initiate the workflow and to interact with `SecurityManager` and other services.  Each step saves intermediate results through the `MemoryManager` so users can review and retry processing.

LangGraph based workflows use `AnalysisNode` and `SummaryNode` (see `agents/agent_nodes.py`) which internally retrieve the integration service via the service container to run analysis or summarization.  This allows complex graphs of tasks to be executed with consistent resource management.

### Typed Workflow Engine

Overall, the ServiceContainer acts as the hub connecting agents and services, enabling workflows like `RealTimeAnalysisWorkflow` and integration via APIs or GUI components.

### Service Container Initialization

The factory `create_service_container` builds the container in a defined order so that each component can rely on the services created before it.  The key steps are:

1. **ConfigurationManager** – loaded first so later services can read application settings.
2. **PersistenceManager** – establishes database and Redis connections.  `UserRepository` and security components depend on it.
3. **UserRepository** and **SecurityManager** – authentication services are configured using persistence.
4. **LLMManager** and **ModelSwitcher** – provide language model access for downstream agents.
5. **EmbeddingManager** – used by the vector store for semantic search.
6. **KnowledgeGraphManager** and **VectorStore** – storage managers required by the graph layer.
7. **RealTimeGraphManager** – synchronizes updates between the graph and vector store.
8. **UnifiedMemoryManager** and **ReviewableMemory** – hold workflow state and review items.
9. **ViolationReviewDB** – persistence for compliance decisions.
10. **RealTimeAnalysisWorkflow** – created after graph and memory managers are ready.
11. **Agent factories** and LangGraph node classes – registered for workflow use.
12. **WorkflowOrchestrator** – retrieved from the container after the workflow is registered.
13. Finally all services are initialized via `initialize_all_services`.

This ordering ensures that persistence and graph managers exist before they are consumed by higher level components like the workflow and orchestrator.

Workflow orchestrators obtain the `RealTimeAnalysisWorkflow` instance from the service container when executing a document pipeline.  This reuse avoids creating duplicate workflows and ensures updates are coordinated through the same persistence and graph managers.

### `handle_document_upload` response

The `IntegrationService.handle_document_upload` method returns a small JSON
object acknowledging that processing has started.  The fields are:

- `document_id` – the identifier stored in the `MemoryManager`
- `filename` – the stored filename on disk
- `size_bytes` – byte size of the uploaded file
- `status` – initial status is `loaded` once a document is uploaded. It will
  progress through `preprocessing`, `processing`, `in_review`, and finally
  `completed` or `error`.
- `message` – human readable confirmation

If the `SecurityManager` or workflow orchestrator are unavailable the method
raises `ServiceLayerError`.

### Document Status Lifecycle

Documents move through several states as they are processed:

1. **loaded** – successfully uploaded and awaiting preprocessing.
2. **preprocessing** – text and metadata extraction is in progress.
3. **processing** – analysis agents are running on the document.
4. **in_review** – human review is required for some extractions.
5. **completed** – processing finished successfully and the document is searchable.
6. **error** – an unrecoverable issue occurred during processing.

Tracking these states helps the UI report progress and highlight any processing issues.
