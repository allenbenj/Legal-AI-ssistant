# Legal AI System - Professional Edition

Enterprise-grade Legal Document Analysis Platform with AI-powered processing, knowledge graph generation, and multiple interface options.

## 🏛️ System Overview

The Legal AI System is a comprehensive microservices-based platform designed for legal professionals, featuring:

- **Multi-Agent AI Processing**: Specialized agents for document analysis, violation detection, and legal research
- **Real-time Knowledge Graph**: Dynamic legal entity and relationship extraction with Neo4j integration  
- **Advanced Vector Search**: FAISS-powered similarity search with intelligent caching
- **Multiple Interfaces**: React frontend with real-time features + Streamlit interface for development
- **FastAPI Backend**: RESTful API with GraphQL and WebSocket support
- **Comprehensive Integration**: Frontend-backend separation with modern web technologies

## 🚀 Quick Start

### Installation

```bash
# Navigate to the legal_ai_system directory
cd legal_ai_system

# Install Python dependencies
pip install -r requirements.txt

# Install React frontend dependencies
cd my-legal-tech-gui
npm install
cd ..
```

### Launch Options

#### Option 1: React Frontend + FastAPI Backend (Recommended)

```bash
# Terminal 1 - Start FastAPI backend
python api/main.py
# Backend runs at http://127.0.0.1:8000

# Terminal 2 - Start React frontend
cd my-legal-tech-gui
npm run dev
# Frontend runs at http://localhost:5173
```

#### Option 2: Streamlit Development Interface

```bash
# Direct execution
python main.py
# Opens Streamlit at http://localhost:8501
```

#### Option 3: Quick Testing Server

```bash
# Simple FastAPI server for testing
python quick_start.py
# Runs at http://localhost:8000
```

## 🎯 Key Features

### AI Agents
- **Document Processor**: Comprehensive PDF, DOCX, Excel analysis
- **Legal Analyzer**: Constitutional and statutory analysis
- **Violation Detector**: Identifies legal violations and misconduct
- **Citation Analyzer**: Legal citation extraction and validation
- **Entity Extractor**: Legal entity identification and relationship mapping

### Technical Architecture
- **Service Container**: Dependency injection with lifecycle management
- **Multi-LLM Support**: xAI/Grok (primary), OpenAI (backup), Ollama (local)
- **Vector Storage**: Hybrid FAISS + LanceDB for high-performance search
- **Knowledge Graph**: Real-time Neo4j integration with auto-sync
- **Detailed Logging**: Complete operation tracking with multiple log files

### API Configuration
The system supports multiple LLM providers:
- **xAI/Grok** (Recommended): `grok-3-mini`, `grok-3-reasoning`, `grok-2-1212`
- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`
- **Ollama**: Local models for privacy-sensitive documents

## 📋 Interface Overview

### React Frontend (Production Interface)

The modern React interface (`my-legal-tech-gui`) provides:

1. **Interactive Dashboard**: Real-time system metrics and performance charts
2. **Document Processing**: Advanced file upload with drag-and-drop support
3. **Knowledge Graph Visualization**: Interactive graph exploration with entity details
4. **Confidence Calibration**: Human-in-the-loop validation interface  
5. **Agent Management**: Monitor and control AI agent operations
6. **Real-time Updates**: WebSocket integration for live progress tracking
7. **Authentication**: JWT-based user management
8. **Debug Panel**: Comprehensive logging and debugging tools

### Streamlit Interface (Development)

The Streamlit interface provides:

1. **API Configuration**: Easy setup for xAI and OpenAI keys
2. **Agent Selection**: Choose from specialized legal AI agents  
3. **File Upload**: Support for PDF, DOCX, Excel, CSV, and text files
4. **Real-time Processing**: Step-by-step operation visibility
5. **Live Logging**: View detailed logs in real-time
6. **Results Display**: Comprehensive analysis results with structured output

## 🔧 Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# LLM Configuration
XAI_API_KEY=your_xai_key_here
OPENAI_API_KEY=your_openai_key_here

# Vector Storage
VECTOR_STORE_TYPE=hybrid
LANCE_DB_PATH=./storage/vectors/lancedb

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Logging
LOG_LEVEL=TRACE
ENABLE_DETAILED_LOGGING=true
```

### Directory Structure

```
legal_ai_system/
├── main.py                 # Streamlit entry point
├── requirements.txt        # Python dependencies
├── api/                    # FastAPI backend
│   ├── main.py            # Primary API server
│   ├── integration_service.py
│   └── requirements.txt
├── my-legal-tech-gui/     # React frontend
│   ├── src/
│   │   ├── enhanced-legal-ai-gui2.tsx  # Main component
│   │   ├── main.tsx       # Entry point
│   │   └── index.css      # Tailwind CSS
│   ├── package.json       # Node dependencies
│   └── vite.config.ts     # Vite configuration
├── core/                  # Core system components
│   ├── unified_services.py # Service container
│   ├── llm_providers.py   # Multi-LLM support
│   ├── vector_store_enhanced.py
│   └── knowledge_graph_enhanced.py
├── agents/                # AI agents (15 specialized agents)
│   ├── base_agent.py      # Agent framework
│   ├── document_processor.py
│   ├── legal_analysis.py
│   └── violation_detector.py
├── workflows/             # Orchestration
├── storage/               # Data storage
├── utils/                 # Utilities
├── memory/                # Memory management
├── quick_start.py         # Simple testing server
└── docs/                  # Documentation
```

## 📊 Usage Examples

### React Frontend Workflow

#### Document Analysis
1. Start backend: `python api/main.py`
2. Start frontend: `cd my-legal-tech-gui && npm run dev`
3. Open `http://localhost:5173` in browser
4. Navigate to "Document Processing" section
5. Drag and drop legal documents or click to upload
6. Monitor real-time processing with progress bars
7. View results in dashboard with extracted entities

#### Knowledge Graph Exploration
1. Access "Knowledge Graph" section in React interface
2. Search for entities using the search bar
3. Click nodes to explore relationships
4. Use interactive graph visualization to navigate legal connections
5. Export graph data or visualizations

#### Confidence Calibration
1. Navigate to "Confidence Calibration" section
2. Review extracted entities requiring human validation
3. Approve, modify, or reject AI extractions
4. Submit feedback to improve system accuracy

### Streamlit Development Workflow

#### Quick Analysis
1. Launch: `python main.py`
2. Configure API keys in interface
3. Select specialized legal agent
4. Upload document and view step-by-step processing
5. Review detailed logs and results

## 🔌 API Integration

### FastAPI Backend Endpoints

The system provides comprehensive API endpoints:

#### REST API (`http://127.0.0.1:8000/api/v1`)
- **POST `/auth/token`** - JWT authentication (mock mode)
- **POST `/documents/upload`** - File upload and processing
- **GET `/system/health`** - System status and metrics
- **POST `/calibration/review`** - Submit confidence calibration feedback

#### GraphQL (`http://127.0.0.1:8000/graphql`)
- **Query `searchEntities`** - Complex entity search with filters
- **Query `traverseGraph`** - Knowledge graph traversal and exploration
- **Query `systemStatus`** - Real-time system monitoring

#### WebSocket (`ws://127.0.0.1:8000/ws/{user_id}`)
- Real-time document processing updates
- System status notifications
- Live confidence calibration feedback
- Agent status monitoring

### Frontend-Backend Integration

The React frontend integrates seamlessly with the backend through:
- **State Management**: Zustand with real-time updates
- **Data Fetching**: React Query with caching and error handling
- **Real-time Communication**: WebSocket connections for live updates
- **Authentication**: JWT token management with localStorage

## 🔍 Logging and Monitoring

The system provides comprehensive logging:

- **Main Log**: General system operations
- **API Log**: LLM API calls and responses
- **Results Log**: Processing results and analysis
- **Error Log**: Detailed error tracking and recovery

All logs are timestamped and include detailed context for debugging and audit purposes.

## 🛠️ Troubleshooting

### Common Issues

**Python Dependencies**: If Python dependencies fail to install:
```bash
pip install -r requirements.txt --upgrade
```

**Node Dependencies**: If React frontend dependencies fail:
```bash
cd my-legal-tech-gui
npm install --force
# or
yarn install
```

**Port Conflicts**: 
- Frontend (port 5173): Vite will automatically find next available port
- Backend (port 8000): 
  ```bash
  # Change port in api/main.py or use
  uvicorn api.main:app --port 8001
  ```
- Streamlit (port 8501):
  ```bash
  streamlit run main.py --server.port 8502
  ```

**API Keys**: Ensure your API keys are valid and have sufficient credits.

**CORS Issues**: If frontend can't connect to backend, ensure CORS is enabled in `api/main.py`.

**WebSocket Connection**: If real-time updates aren't working, check WebSocket connection in browser dev tools.

**Memory Issues**: For large documents, monitor system memory usage and consider batch processing.

## 🏗️ Architecture Notes

This system implements enterprise software architecture best practices:

- **Frontend-Backend Separation**: React frontend with FastAPI backend
- **Single Responsibility**: Each component has a clear, focused purpose
- **Dependency Injection**: Centralized service management via unified services
- **Real-time Communication**: WebSocket integration for live updates
- **Modern Web Standards**: REST API, GraphQL, and JWT authentication
- **Comprehensive Error Handling**: Graceful failure recovery throughout
- **Modular Design**: Easy to extend and maintain

### Recent System Streamlining (2024)

The system has been comprehensively cleaned and optimized:
- ✅ **2-4GB space savings** through intelligent archival
- ✅ **No duplicate functionality** - consolidated implementations
- ✅ **Clear module boundaries** - production vs development separation  
- ✅ **Professional organization** - clean directory structure
- ✅ **Enhanced integration** - seamless frontend-backend communication
- ✅ **Multiple interface options** - React (production) + Streamlit (development)

### Technology Stack

**Frontend**: React 18 + TypeScript + Tailwind CSS + Zustand + React Query  
**Backend**: FastAPI + Python + GraphQL + WebSocket  
**AI/ML**: Multi-LLM support (xAI/Grok, OpenAI, Ollama)  
**Data**: Neo4j + FAISS + Vector Storage + Knowledge Graphs  
**Infrastructure**: Service Container + Dependency Injection + Comprehensive Logging

## 📝 License

Professional Legal AI System - Enterprise Edition