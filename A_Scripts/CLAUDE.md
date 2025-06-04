# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This repository contains a comprehensive **Legal AI Assistant System** - an advanced document processing and analysis platform specifically designed for legal professionals. The system combines multiple AI agents, vector storage, knowledge graphs, and intelligent automation to provide sophisticated legal document analysis, violation detection, and case management capabilities.

### Core Architecture

The system follows a **microservices architecture** with clear separation of concerns:

- **Core Services Layer**: Centralized service container with dependency injection
- **Multi-LLM Support**: Ollama (local), OpenAI, and xAI providers with automatic fallback
- **Vector Storage**: Hybrid FAISS + LanceDB for high-performance similarity search
- **Knowledge Graph**: Neo4j integration for complex legal entity relationships
- **Intelligent Agents**: Specialized AI agents for different legal analysis tasks
- **Memory Management**: Context-aware session management across multiple files/tasks
- **Auto-Processing**: File watching with automatic document ingestion and tagging

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv legal_ai_env
source legal_ai_env/bin/activate  # Linux/Mac
# or
legal_ai_env\Scripts\activate     # Windows

# Install dependencies (includes spaCy, FAISS, Neo4j drivers)
pip install -r requirements.txt

# Install legal-specific NLP models
python -m spacy download en_core_web_lg
pip install https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and Neo4j configuration
```

### Real-Time System Commands
```bash
# Analyze single document with real-time graph building
python -m legal_ai_system.cli analyze-document document.pdf --targeted --verbose

# Build knowledge graph from multiple documents
python -m legal_ai_system.cli build-knowledge-graph documents/*.pdf --parallel

# Force graph synchronization with Neo4j
python -m legal_ai_system.cli sync-graph --force

# Optimize vector store for faster searches
python -m legal_ai_system.cli optimize-vector-store --force-rebuild

# Get comprehensive system status
python -m legal_ai_system.cli system-status

# Configure confidence thresholds
python -m legal_ai_system.cli configure-thresholds --auto-approve 0.9 --review 0.7

# Force system-wide synchronization
python -m legal_ai_system.cli force-system-sync
```

### Legacy Commands
```bash
# Start main GUI application
python -m legal_ai_system.main

# Run basic CLI interface
python -m legal_ai_system.cli

# Start development server with hot reload
python -m legal_ai_system.dev_server --reload
python -m legal_ai_system.main --config production
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agents/        # Agent tests
pytest tests/test_core/          # Core system tests
pytest tests/test_integration/   # Integration tests

# Run tests with coverage
pytest --cov=legal_ai_system --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run single test file
pytest tests/test_llm_providers.py -v
```

### Database Management
```bash
# Initialize databases
python -m legal_ai_system.utils.init_db

# Run migrations
python -m legal_ai_system.utils.migrate

# Reset vector store
python -m legal_ai_system.utils.reset_vectors

# Backup data
python -m legal_ai_system.utils.backup --output backups/

# Restore from backup
python -m legal_ai_system.utils.restore --input backups/backup_20241201.zip
```

### Development Tools
```bash
# Type checking
mypy legal_ai_system/

# Code formatting
black legal_ai_system/
isort legal_ai_system/

# Linting
pylint legal_ai_system/
flake8 legal_ai_system/

# Security scanning
bandit -r legal_ai_system/

# Dependency updates
pip-compile requirements.in
pip-compile dev-requirements.in
```

## System Architecture

### Directory Structure
```
legal_ai_system/
├── core/                    # Core system components
│   ├── services.py         # Service container and DI
│   ├── llm_providers.py    # Multi-LLM provider system
│   ├── vector_stores.py    # Vector storage management
│   ├── memory_manager.py   # Session and context management
│   └── knowledge_graph.py  # Neo4j integration
├── agents/                 # Intelligent AI agents
│   ├── base_agent.py      # Base agent framework
│   ├── document_processor.py
│   ├── auto_tagging.py    # Learning-based auto-tagging
│   ├── violation_detector.py
│   ├── legal_analyzer.py
│   └── note_taking.py     # Context-aware note system
├── gui/                   # PyQt6 user interface
│   ├── main_window.py     # Main application window
│   ├── panels/            # UI panels
│   └── widgets/           # Custom widgets
├── storage/               # Data storage
│   ├── databases/         # SQLite databases
│   ├── vectors/           # Vector stores (FAISS/LanceDB)
│   └── documents/         # Document storage
├── config/                # Configuration management
│   ├── settings.py        # Pydantic settings
│   ├── prompts/           # LLM prompts
│   └── schemas/           # Data schemas
├── utils/                 # Utility modules
│   ├── file_watcher.py    # Auto file processing
│   ├── embedding_manager.py
│   └── legal_utils.py     # Legal-specific utilities
└── tests/                 # Test suite
    ├── unit/              # Unit tests
    ├── integration/       # Integration tests
    └── performance/       # Performance tests
```

### Service Architecture

The system uses a **Service Container** pattern for dependency injection and lifecycle management:

```python
# Initialize services
from legal_ai_system.core.services import initialize_services
services = await initialize_services()

# Access services
llm_manager = services.llm_manager
vector_manager = services.vector_manager
agent = services.get_agent('document_processor')
```

### Agent System

The system uses specialized AI agents with real-time knowledge graph building and vector optimization:

#### Core Agents

- **DocumentProcessorAgent**: Comprehensive document processing with intelligent file type handling
  - Full Processing: PDF, DOCX, TXT, MD, HTML, RTF
  - Structured Data: XLSX, XLS, CSV with database schema generation
  - Reference-Only: PPTX, PPT, Images with OCR preview

- **OntologyExtractionAgent**: Ontology-driven legal entity and relationship extraction
  - Extracts 20+ legal entity types (Person, Case, Evidence, Judge, etc.)
  - Identifies legal relationships (Filed_By, Supports, Refutes, Presided_By, etc.)
  - Hybrid pattern matching + LLM validation approach

- **HybridLegalExtractor**: Advanced NER+LLM hybrid extraction system
  - **Multi-Model NER**: spaCy, Flair, Blackstone for legal text
  - **Targeted Extractions**: Brady violations, prosecutorial misconduct, witness tampering
  - **Cross-Validation**: NER vs LLM output validation with discrepancy resolution

#### Real-Time Infrastructure

- **RealTimeGraphManager**: High-performance knowledge graph with automatic Neo4j sync
  - Auto-sync at high confidence levels with transaction safety
  - Entity deduplication and intelligent merging
  - Usage-based caching and performance optimization

- **OptimizedVectorStore**: ANN-indexed vector store with intelligent caching
  - FAISS integration for fast similarity search
  - Smart query caching with TTL and usage-based eviction
  - Pre-indexing of frequently accessed entities

- **ReviewableMemory**: Human-in-the-loop validation system
  - Staging area for extracted entities before permanent storage
  - Configurable confidence thresholds for auto-approve/review/reject
  - Priority assignment for critical legal findings

#### Master Workflow

- **RealTimeAnalysisWorkflow**: Complete real-time document analysis pipeline
  - Document processing → Hybrid extraction → Graph building → Vector updates → Memory integration
  - Real-time synchronization with performance monitoring and auto-optimization

#### CLI Management

- **SystemCommands**: Command-line interface for system management
  - `sync-graph`, `build-knowledge-graph`, `optimize-vector-store`
  - `analyze-document`, `system-status`, `configure-thresholds`

#### Planned Agents

- **AutoTaggingAgent**: Learns from user behavior to auto-tag documents
- **ViolationDetectorAgent**: Identifies legal violations and misconduct
- **LegalAnalyzerAgent**: Performs sophisticated legal analysis
- **NoteTakingAgent**: Context-aware note-taking with suggestions

### Memory & Context Management

The system maintains intelligent context across sessions:

- **Session Memory**: Per-topic/project memory with SQLite storage
- **Entity Memory**: Tracks legal entities (people, cases, statutes) across sessions  
- **Context Window**: Intelligent selection of relevant context for LLM calls
- **Auto-Summarization**: Compresses long documents for continued inclusion

## Configuration

### Environment Variables

Key environment variables (see `config/settings.py` for full list):

```bash
# LLM Configuration
LLM_PROVIDER=ollama                    # Primary LLM provider
LLM_MODEL=llama3.2                     # Primary model
FALLBACK_PROVIDER=xai                  # Fallback provider
OLLAMA_HOST=http://localhost:11434     # Ollama server
XAI_API_KEY=your_xai_key_here         # xAI API key

# Vector Storage
VECTOR_STORE_TYPE=hybrid               # faiss, lance, or hybrid
LANCE_DB_PATH=./storage/vectors/lancedb

# Features
ENABLE_AUTO_TAGGING=true               # Enable learning-based tagging
ENABLE_FILE_WATCHING=true              # Auto-process new files
WATCH_DIRECTORIES=["./storage/documents/inbox"]

# Database
NEO4J_URI=bolt://localhost:7687        # Neo4j connection
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### LLM Provider Setup

#### xAI/Grok Models (Primary - Recommended for legal analysis)
```bash
# Set API key for Grok models (primary provider)
export XAI_API_KEY="your_xai_api_key"
export LLM_PROVIDER="xai"

# Available Models:
# grok-3-mini (default) - Fast, efficient legal analysis
export XAI_MODEL="grok-3-mini"

# grok-3-reasoning - Complex legal reasoning with step-by-step analysis
export XAI_MODEL="grok-3-reasoning"

# grok-2-1212 - Balanced performance and reasoning
export XAI_MODEL="grok-2-1212"

# The system automatically switches between models based on task complexity:
# - Simple tasks (citations, extractions) → grok-3-mini
# - Complex analysis (constitutional review, violations) → grok-3-reasoning
# - Users can manually switch models at any time
```

#### Ollama (Fallback - Recommended for privacy)
```bash
# Install Ollama (fallback for sensitive documents)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

### Database Setup

#### Neo4j (Optional but recommended)
```bash
# Using Docker
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    --env NEO4J_AUTH=neo4j/CaseDBMS \
    neo4j:latest
```

## Development Patterns

### Adding New Agents

1. **Create agent class** inheriting from `BaseAgent`:
```python
from legal_ai_system.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    async def process(self, input_data: Any) -> Any:
        # Agent logic here
        pass
```

2. **Register in services**:
```python
# In core/services.py
self._agents['my_custom'] = MyCustomAgent(self)
```

3. **Add tests**:
```python
# tests/test_agents/test_my_custom_agent.py
async def test_my_custom_agent():
    agent = MyCustomAgent(mock_services)
    result = await agent.process(test_data)
    assert result is not None
```

### Adding LLM Providers

1. **Implement provider class**:
```python
class MyLLMProvider(BaseLLMProvider):
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        pass
```

2. **Register in LLMManager**:
```python
# Update _create_provider method
if config.provider == "my_provider":
    return MyLLMProvider(config)
```

### Working with Vector Stores

```python
# Add documents
await services.vector_manager.add_documents([
    {
        'id': 'doc_1',
        'text': 'Document content...',
        'metadata': {'type': 'legal_brief', 'jurisdiction': 'federal'}
    }
])

# Search with filters
results = await services.vector_manager.search(
    query="constitutional violations",
    k=10,
    filters={'jurisdiction': 'federal'}
)
```

### Dynamic Model Switching

The system supports runtime switching between Grok models:

```python
from legal_ai_system.core.model_switcher import GrokModelSwitcher, TaskComplexity

# Initialize model switcher
switcher = GrokModelSwitcher(services.llm_manager, api_key)

# Switch to reasoning model for complex analysis
result = switcher.switch_to_model("grok-3-reasoning", "Complex constitutional analysis")

# Auto-switch based on task type
result = switcher.switch_for_task("violation_detection", TaskComplexity.COMPLEX)

# Get optimized prompt for current model
prompt = switcher.get_optimized_prompt_for_current_model(
    "legal_analysis",
    document_text="..."
)

# Check available models
models = switcher.get_available_models()
```

#### Model Selection Guidelines

- **grok-3-mini**: Fast analysis, citations, document classification
- **grok-3-reasoning**: Constitutional analysis, complex violations, multi-step reasoning
- **grok-2-1212**: Balanced choice for most legal tasks

## Testing Patterns

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Use `pytest` fixtures for setup

### Integration Tests  
- Test component interactions
- Use test databases/containers
- Verify end-to-end workflows

### Performance Tests
- Benchmark document processing speeds
- Test with large document sets
- Monitor memory usage

## Security Considerations

- **API Keys**: Store in environment variables, never commit to code
- **Local Processing**: Use Ollama for sensitive documents
- **Data Encryption**: Enable for production deployments
- **Input Validation**: Validate all user inputs and file uploads
- **Rate Limiting**: Configure appropriate limits for API calls

## Troubleshooting

### Common Issues

**LLM Provider Failures**:
- Check API keys and network connectivity
- Verify Ollama is running (for local models)
- Check rate limits and quotas

**Vector Store Issues**:
- Ensure sufficient disk space
- Check file permissions on storage directories
- Verify embedding model availability

**Database Connection Issues**:
- Confirm Neo4j is running and accessible
- Check connection credentials
- Verify network firewall settings

**Memory Issues**:
- Monitor context window sizes
- Enable auto-summarization
- Adjust batch sizes for processing

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export AGENT_DEBUG=true
```

Save intermediate results:
```bash
export SAVE_INTERMEDIATE=true
```

## Performance Optimization

- **Vector Search**: Use FAISS for high-speed similarity search
- **Batch Processing**: Process documents in batches
- **Caching**: Enable embedding cache for repeated operations
- **Memory Management**: Use auto-summarization for large contexts
- **Async Operations**: Leverage async/await throughout the system

## Deployment

### Development
```bash
python -m legal_ai_system.main --env development
```

### Production
```bash
# Set production environment
export APP_ENV=production
export DEBUG=false

# Run with production settings
python -m legal_ai_system.main --config production

# Or use Docker
docker-compose up -d
```

This system represents a sophisticated legal AI platform with enterprise-grade architecture, comprehensive testing, and production-ready deployment capabilities.