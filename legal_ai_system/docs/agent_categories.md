# Agent Categories

The Legal AI System organizes its agents into logical categories. This overview helps developers quickly locate and register the correct components when extending workflows or the GUI.

## Analysis Agents
- `SemanticAnalysisAgent`
- `StructuralAnalysisAgent`
- `CitationAnalysisAgent`
- `LegalAnalysisAgent`

## Document Processing
- `DocumentProcessorAgent`
- `DocumentProcessorAgentV2`
- `DocumentRewriterAgent`
- `TextCorrectionAgent`

## Knowledge & Reasoning
- `KnowledgeBaseAgent`
- `KnowledgeGraphReasoningAgent`
- `LegalReasoningEngine`
- `GraphInferenceAgent`
- `PrecedentMatchingAgent`

## Extraction & Classification
- `StreamlinedEntityExtractionAgent`
- `OntologyExtractionAgent`
- `AutoTaggingAgent`
- `ViolationDetectorAgent`

## Utility
- `NoteTakingAgent`

These groupings match the `AGENT_CATEGORIES` mapping defined in `legal_ai_system/agents/__init__.py` and make it easier to understand how the system's capabilities are structured.
