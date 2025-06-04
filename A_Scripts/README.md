# Legal AI System - Unified GUI

A comprehensive web-based interface for the Legal AI System providing document processing, memory management, violation review, knowledge graph visualization, and system administration.

## Features

### 🎯 Six Integrated Tabs

1. **📊 Analysis Dashboard**
   - High-level system overview and metrics
   - Document processing statistics
   - Violation and compliance trends
   - Performance analytics

2. **📄 Document Processor**
   - File upload with validation (PDF, DOCX, TXT, MD)
   - Configurable processing options (NER, LLM extraction, classification)
   - Real-time processing status monitoring
   - Model selection and API key management

3. **🧠 Memory Brain**
   - AI memory entry management and visualization
   - Memory type filtering (FACT, RULE, PRECEDENT, ENTITY)
   - Access tracking and confidence analysis
   - Memory association mapping

4. **⚠️ Violation Review**
   - Detected violation management
   - Severity-based filtering and sorting
   - Bulk approval/rejection workflows
   - Compliance tracking and reporting

5. **🕸️ Knowledge Graph**
   - Interactive knowledge graph visualization
   - Entity and relationship management
   - Graph querying and exploration
   - Export capabilities (JSON, CSV)

6. **⚙️ Settings & Logs**
   - Application configuration management
   - System monitoring and health checks
   - Comprehensive logging and audit trails
   - User preference management

### 🤖 XAI/Grok Integration

- **Direct API Integration**: Connect directly to xAI's Grok models
- **Multiple Models**: Support for Grok-3-Mini, Grok-3-Reasoning, Grok-2-1212
- **Legal Optimization**: Specialized prompts for legal document analysis
- **Analysis Types**: Legal analysis, violation detection, entity extraction, summarization
- **Token Tracking**: Monitor usage and costs
- **Model Comparison**: Compare results across different Grok variants

## Architecture

### 🏗️ Component Structure

```
gui/
├── __init__.py              # Module initialization
├── main_gui.py              # Enhanced main application
├── unified_gui.py           # Six tab implementations
├── shared_components.py     # Reusable UI components
├── database_manager.py      # SQLite database integration
├── streamlit_app.py         # Legacy Streamlit app
├── test_gui_basic.py        # Basic validation tests
└── README.md               # This documentation
```

### 🔧 Key Components

- **EnhancedGUIApplication**: Main application class with database integration
- **DatabaseManager**: SQLite-based data persistence
- **APIClient**: Backend communication handler
- **XAIGrokClient**: Direct integration with XAI/Grok models
- **XAIIntegratedGUI**: Enhanced GUI with XAI capabilities
- **SessionManager**: Streamlit session state management
- **ErrorHandler**: Centralized error handling and user notifications
- **DataVisualization**: Shared plotting and chart components
- **MockDataGenerator**: Test data generation for development

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

```bash
pip install streamlit pandas plotly requests
```

### Optional Dependencies

```bash
pip install networkx  # For advanced graph visualization
pip install pydantic  # For enhanced data validation
```

## Usage

### Starting the GUI

```bash
# From the legal_ai_system directory
streamlit run gui/main_gui.py

# Or from the project root
streamlit run legal_ai_system/gui/main_gui.py
```

The GUI will be available at `http://localhost:8501`

### Configuration

The system uses SQLite for data persistence. The database file `legal_ai_gui.db` will be created automatically in the current directory.

### Sample Data

The system automatically loads sample data for demonstration:
- Sample violation records
- Sample memory entries
- Mock analytics data

## Database Schema

### Tables

- **violations**: Detected compliance violations
- **memory_entries**: AI memory storage
- **graph_nodes**: Knowledge graph entities
- **graph_edges**: Knowledge graph relationships
- **documents**: Document processing records
- **system_logs**: Application audit trail

### Data Models

```python
@dataclass
class ViolationRecord:
    id: str
    document_id: str
    violation_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    status: str    # PENDING, REVIEWED, APPROVED, REJECTED
    description: str
    confidence: float
    detected_time: datetime
    reviewed_by: Optional[str] = None
    review_time: Optional[datetime] = None
    comments: Optional[str] = None

@dataclass
class MemoryRecord:
    id: str
    memory_type: str  # FACT, RULE, PRECEDENT, ENTITY
    content: str
    confidence: float
    source_document: str
    created_time: datetime
    last_accessed: datetime
    access_count: int
    tags: Optional[str] = None
    metadata: Optional[str] = None
```

## API Integration

The GUI communicates with the Legal AI System backend via REST API:

- **Health Check**: `GET /api/v1/system/health`
- **Document Upload**: `POST /api/v1/documents/upload`
- **Document Processing**: `POST /api/v1/documents/{id}/process`
- **Memory Management**: `GET/POST /api/v1/memory`
- **Violation Management**: `GET/POST /api/v1/violations`
- **Knowledge Graph**: `GET /api/v1/knowledge-graph`

## Development

### Testing

```bash
# Run basic validation tests
python gui/test_gui_basic.py

# Run comprehensive tests (requires dependencies)
python gui/test_gui_system.py
```

### Code Quality

- **Total Lines**: 3,300+ lines of code
- **Classes**: 20+ well-structured classes
- **Functions**: 75+ modular functions
- **Test Coverage**: Comprehensive validation suite

### Adding New Features

1. **New Tab**: Extend `unified_gui.py` with a new tab class
2. **Database Integration**: Add new tables/models to `database_manager.py`
3. **Shared Components**: Add reusable components to `shared_components.py`
4. **API Endpoints**: Extend `APIClient` class for new backend integration

## Performance

### Optimization Features

- **Database Indexing**: Optimized queries for large datasets
- **Session Caching**: Efficient Streamlit session management
- **Lazy Loading**: On-demand data loading for better performance
- **Modular Architecture**: Clean separation of concerns

### Scalability

- **SQLite Database**: Supports thousands of records efficiently
- **Modular Design**: Easy to extend with new features
- **Error Handling**: Robust error recovery and user feedback
- **Logging**: Comprehensive audit trail and debugging

## Security

### Data Protection

- **Input Validation**: Comprehensive file and data validation
- **SQL Injection Prevention**: Parameterized queries
- **Session Security**: Secure session state management
- **Error Handling**: Safe error messages without information leakage

### Access Control

- **User Tracking**: Session-based user identification
- **Audit Trail**: Complete action logging
- **Data Isolation**: Proper data segregation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Database Issues**: Check file permissions and disk space
3. **Backend Connection**: Verify API endpoint configuration
4. **Performance**: Monitor database size and optimize queries

### Logging

System logs are available in:
- **Database**: `system_logs` table
- **Console**: Streamlit console output
- **File**: Application log files (if configured)

## Contributing

### Code Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features

### Development Workflow

1. Create feature branch
2. Implement changes with tests
3. Validate with test suite
4. Submit pull request

## License

Part of the Legal AI System project. See main project license for details.

## Support

For issues and feature requests, please refer to the main Legal AI System documentation and issue tracker.

---

*Legal AI System GUI v2.1.0 - Professional Legal AI Assistant Interface*