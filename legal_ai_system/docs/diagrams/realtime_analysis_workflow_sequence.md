# RealTimeAnalysisWorkflow Execution Sequence

```mermaid
sequenceDiagram
    participant User
    participant Integration as IntegrationService
    participant Workflow as RealTimeAnalysisWorkflow
    participant Processor as DocumentProcessorAgent
    participant Rewriter as DocumentRewriterAgent
    participant Ontology as OntologyExtractionAgent
    participant Graph as RealTimeGraphManager
    participant Memory as ReviewableMemory

    User->>Integration: handle_document_upload(doc)
    Integration->>Workflow: run(doc)
    Workflow->>Processor: extract text
    Processor-->>Workflow: content
    Workflow->>Rewriter: clean text
    Rewriter-->>Workflow: cleaned
    Workflow->>Ontology: extract entities
    Ontology-->>Workflow: entities
    Workflow->>Graph: update graph
    Graph-->>Workflow: confirmed
    Workflow->>Memory: store results
    Memory-->>Workflow: stored
    Integration<<--Workflow: return summary
```
