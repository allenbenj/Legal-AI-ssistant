#!/bin/bash

# Legal AI System - Environment Setup and Dependency Verification Script
# This script sets up the complete development environment and verifies all dependencies

set -e

echo "🔧 Legal AI System - Environment Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="legal_ai_env"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check version
check_version() {
    local tool=$1
    local min_version=$2
    local current_version=$3
    
    if [[ $(printf '%s\n' "$min_version" "$current_version" | sort -V | head -n1) = "$min_version" ]]; then
        print_success "$tool version $current_version (>= $min_version)"
        return 0
    else
        print_error "$tool version $current_version is below minimum required version $min_version"
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if command_exists "apt-get"; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y \
            python3 python3-pip python3-venv \
            nodejs npm \
            curl wget git \
            build-essential \
            pkg-config \
            libffi-dev \
            libssl-dev \
            default-jre \
            graphviz
    elif command_exists "yum"; then
        # CentOS/RHEL
        sudo yum install -y \
            python3 python3-pip \
            nodejs npm \
            curl wget git \
            gcc gcc-c++ make \
            pkg-config \
            libffi-devel \
            openssl-devel \
            java-11-openjdk \
            graphviz
    elif command_exists "brew"; then
        # macOS
        brew install \
            python@3.11 \
            node \
            git \
            graphviz \
            openjdk@11
    else
        print_warning "Unknown package manager. Please install dependencies manually:"
        echo "  - Python 3.8+"
        echo "  - Node.js 16+"
        echo "  - Git"
        echo "  - Build tools (gcc, make)"
        echo "  - Java 11+ (for Neo4j)"
        echo "  - Graphviz"
    fi
}

# Function to setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_PATH" ]]; then
        python3 -m venv "$VENV_PATH"
        print_success "Virtual environment created at $VENV_PATH"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found, creating minimal requirements..."
        
        # Create basic requirements.txt
        cat > requirements.txt << EOF
# Core framework dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4

# GraphQL and WebSocket
strawberry-graphql[fastapi]>=0.214.0
websockets>=12.0

# Database and storage
neo4j>=5.14.0
faiss-cpu>=1.7.4
lancedb>=0.3.0
sqlite3

# AI and ML libraries
openai>=1.3.0
anthropic>=0.7.0
ollama>=0.1.7
sentence-transformers>=2.2.0
torch>=2.1.0
transformers>=4.35.0
spacy>=3.7.0
flair>=0.13.0

# Legal NLP
https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz

# Scientific computing
numpy>=1.24.0
pandas>=2.1.0
scikit-learn>=1.3.0
scipy>=1.11.0

# GUI and visualization
PyQt6>=6.6.0
matplotlib>=3.7.0
plotly>=5.17.0

# Utilities
python-dotenv>=1.0.0
requests>=2.31.0
aiohttp>=3.9.0
httpx>=0.25.0
pyyaml>=6.0.1
rich>=13.7.0
click>=8.1.0
tqdm>=4.66.0

# Development tools
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.10.0
isort>=5.12.0
mypy>=1.7.0
pylint>=3.0.0
EOF
        
        pip install -r requirements.txt
        print_success "Basic Python environment set up"
    fi
    
    # Install spaCy models
    print_status "Installing spaCy language models..."
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_lg
    print_success "spaCy models installed"
}

# Function to setup Node.js environment
setup_node_env() {
    print_status "Setting up Node.js environment..."
    
    cd "$PROJECT_ROOT/my-legal-tech-gui"
    
    if [[ ! -f "package.json" ]]; then
        print_warning "package.json not found, creating React app..."
        npx create-react-app . --template typescript
    fi
    
    # Ensure all dependencies are in package.json
    print_status "Installing Node.js dependencies..."
    
    # Install core dependencies
    npm install \
        zustand \
        @tanstack/react-query \
        graphql-request \
        react-use-websocket \
        recharts \
        react-force-graph-2d \
        lucide-react \
        @types/node \
        @types/react \
        @types/react-dom
    
    # Install development dependencies
    npm install --save-dev \
        @types/d3 \
        @typescript-eslint/eslint-plugin \
        @typescript-eslint/parser \
        eslint-plugin-react-hooks
    
    print_success "Node.js dependencies installed"
}

# Function to setup Docker environment
setup_docker() {
    print_status "Checking Docker setup..."
    
    if command_exists "docker"; then
        print_success "Docker is available"
        
        # Pull required Docker images
        docker pull neo4j:latest
        docker pull python:3.11-slim
        
        print_success "Docker images pulled"
    else
        print_warning "Docker not found. Please install Docker for container support:"
        echo "  • Ubuntu/Debian: https://docs.docker.com/engine/install/ubuntu/"
        echo "  • macOS: https://docs.docker.com/desktop/install/mac-install/"
        echo "  • Windows: https://docs.docker.com/desktop/install/windows-install/"
    fi
}

# Function to setup optional tools
setup_optional_tools() {
    print_status "Setting up optional tools..."
    
    # Ollama installation
    if ! command_exists "ollama"; then
        print_status "Installing Ollama..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://ollama.ai/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            print_warning "Please install Ollama manually from: https://ollama.ai/download"
        else
            print_warning "Please install Ollama manually from: https://ollama.ai/download"
        fi
    else
        print_success "Ollama is already installed"
    fi
    
    # Neo4j Desktop
    if ! command_exists "neo4j"; then
        print_warning "Neo4j not found. Consider installing Neo4j Desktop:"
        echo "  • Download from: https://neo4j.com/download/"
        echo "  • Or use Docker: docker run -p7474:7474 -p7687:7687 neo4j:latest"
    else
        print_success "Neo4j is available"
    fi
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    cd "$PROJECT_ROOT"
    
    # Create required directories
    mkdir -p storage/{databases,vectors,documents/{inbox,processed,failed},neo4j/{data,logs}}
    mkdir -p logs/{development,production}
    mkdir -p pids
    mkdir -p config/{prompts,schemas}
    
    # Create .env template if it doesn't exist
    if [[ ! -f ".env" ]]; then
        cat > .env.example << EOF
# Legal AI System Configuration

# API Keys
XAI_API_KEY=your_xai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# LLM Configuration
LLM_PROVIDER=xai
LLM_MODEL=grok-3-mini
FALLBACK_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=CaseDBMS

# Vector Store Configuration
VECTOR_STORE_TYPE=hybrid
LANCE_DB_PATH=./storage/vectors/lancedb

# Application Settings
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Security
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Feature Flags
ENABLE_AUTO_TAGGING=true
ENABLE_FILE_WATCHING=true
ENABLE_KNOWLEDGE_GRAPH=true

# File Processing
WATCH_DIRECTORIES=["./storage/documents/inbox"]
MAX_FILE_SIZE_MB=100
SUPPORTED_FILE_TYPES=["pdf","docx","txt","md","html"]

# Performance
MAX_CONCURRENT_DOCUMENTS=5
EMBEDDING_BATCH_SIZE=32
VECTOR_SEARCH_K=10

# Confidence Thresholds
AUTO_APPROVE_THRESHOLD=0.9
REVIEW_THRESHOLD=0.7
REJECT_THRESHOLD=0.3
EOF
        
        cp .env.example .env
        print_success "Environment configuration created"
        print_warning "Please edit .env with your actual API keys and configuration"
    fi
    
    print_success "Directory structure created"
}

# Main setup function
main() {
    echo "This script will set up the complete Legal AI System development environment."
    echo "The following will be installed/configured:"
    echo "  • Python virtual environment with all dependencies"
    echo "  • Node.js environment for React frontend"
    echo "  • Optional tools (Ollama, Neo4j, Docker)"
    echo "  • Directory structure and configuration files"
    echo ""
    
    read -p "Continue with setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Check system requirements
    print_status "Checking system requirements..."
    
    # Check Python
    if command_exists "python3"; then
        python_version=$(python3 --version | cut -d' ' -f2)
        check_version "Python" "3.8.0" "$python_version"
    else
        print_error "Python 3 not found"
        install_system_deps
    fi
    
    # Check Node.js
    if command_exists "node"; then
        node_version=$(node --version | sed 's/v//')
        check_version "Node.js" "16.0.0" "$node_version"
    else
        print_error "Node.js not found"
        install_system_deps
    fi
    
    # Check Git
    if command_exists "git"; then
        print_success "Git is available"
    else
        print_error "Git not found"
        install_system_deps
    fi
    
    # Run setup steps
    create_directories
    setup_python_env
    setup_node_env
    setup_docker
    setup_optional_tools
    
    # Final instructions
    echo ""
    echo "🎉 Legal AI System Environment Setup Complete!"
    echo "=============================================="
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Edit .env file with your API keys:"
    echo "     nano $PROJECT_ROOT/.env"
    echo ""
    echo "  2. Activate the virtual environment:"
    echo "     source $VENV_PATH/bin/activate"
    echo ""
    echo "  3. Start the development environment:"
    echo "     $PROJECT_ROOT/scripts/start_development.sh"
    echo ""
    echo "📚 Documentation:"
    echo "  • System Overview: $PROJECT_ROOT/docs/SYSTEM_OVERVIEW.md"
    echo "  • API Documentation: http://localhost:8000/docs (after starting)"
    echo "  • Frontend: http://localhost:3000 (after starting)"
    echo ""
    echo "🔧 Management Scripts:"
    echo "  • Development: $PROJECT_ROOT/scripts/start_development.sh"
    echo "  • Production:  $PROJECT_ROOT/scripts/start_production.sh"
    echo "  • This setup:  $PROJECT_ROOT/scripts/setup_environment.sh"
    echo ""
    echo "⚠️  Remember to:"
    echo "  • Set your API keys in .env"
    echo "  • Install Neo4j or use Docker for knowledge graph features"
    echo "  • Install Ollama for local LLM processing"
    echo ""
}

# Run main function
main "$@"