#!/bin/bash

# Legal AI System - Development Environment Startup Script
# This script coordinates the startup of all system components in development mode

set -e

echo "🚀 Starting Legal AI System Development Environment..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT"
FRONTEND_DIR="$PROJECT_ROOT/my-legal-tech-gui"
API_DIR="$PROJECT_ROOT/api"
LOG_DIR="$PROJECT_ROOT/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

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

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 1
    else
        return 0
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down services..."
    
    # Kill background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Kill specific processes if they exist
    pkill -f "uvicorn.*legal_ai_system.api.main" 2>/dev/null || true
    pkill -f "npm.*start" 2>/dev/null || true
    pkill -f "neo4j" 2>/dev/null || true
    
    print_success "Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Consider activating one:"
    echo "  python -m venv legal_ai_env"
    echo "  source legal_ai_env/bin/activate"
fi

# Check required ports
REQUIRED_PORTS=(8000 3000 7474 7687 11434)
for port in "${REQUIRED_PORTS[@]}"; do
    if ! check_port $port; then
        print_error "Port $port is already in use. Please free it before starting."
        case $port in
            8000) echo "  This is typically the FastAPI backend port" ;;
            3000) echo "  This is typically the React frontend port" ;;
            7474) echo "  This is the Neo4j HTTP port" ;;
            7687) echo "  This is the Neo4j Bolt port" ;;
            11434) echo "  This is the Ollama server port" ;;
        esac
        exit 1
    fi
done

print_success "All required ports are available"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js to run the frontend."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm to run the frontend."
    exit 1
fi

# Step 1: Start Neo4j (if not already running)
print_status "Starting Neo4j database..."
if ! pgrep -f "neo4j" > /dev/null; then
    if command -v neo4j &> /dev/null; then
        neo4j start > "$LOG_DIR/neo4j.log" 2>&1 &
        sleep 5
    elif command -v docker &> /dev/null; then
        print_status "Starting Neo4j via Docker..."
        docker run --name legal-ai-neo4j \
            -p7474:7474 -p7687:7687 \
            -d \
            -v "$PROJECT_ROOT/storage/neo4j/data:/data" \
            -v "$PROJECT_ROOT/storage/neo4j/logs:/logs" \
            --env NEO4J_AUTH=neo4j/CaseDBMS \
            neo4j:latest > "$LOG_DIR/neo4j-docker.log" 2>&1 || true
        sleep 10
    else
        print_warning "Neo4j not found. Knowledge graph features may be limited."
    fi
else
    print_success "Neo4j is already running"
fi

# Step 2: Start Ollama (if available)
print_status "Checking Ollama availability..."
if command -v ollama &> /dev/null; then
    if ! pgrep -f "ollama" > /dev/null; then
        print_status "Starting Ollama server..."
        ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        sleep 3
        
        # Pull required models if not present
        print_status "Ensuring required models are available..."
        ollama list | grep -q "llama3.2" || ollama pull llama3.2 &
        ollama list | grep -q "nomic-embed-text" || ollama pull nomic-embed-text &
    else
        print_success "Ollama is already running"
    fi
else
    print_warning "Ollama not found. Will use API-based LLM providers only."
fi

# Step 3: Install backend dependencies
print_status "Installing/updating backend dependencies..."
cd "$PROJECT_ROOT"
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt > "$LOG_DIR/pip-install.log" 2>&1
    print_success "Backend dependencies installed"
else
    print_warning "requirements.txt not found, skipping backend dependency installation"
fi

# Step 4: Install frontend dependencies
print_status "Installing/updating frontend dependencies..."
cd "$FRONTEND_DIR"
if [[ -f "package.json" ]]; then
    npm install > "$LOG_DIR/npm-install.log" 2>&1
    print_success "Frontend dependencies installed"
else
    print_error "Frontend package.json not found"
    exit 1
fi

# Step 5: Initialize databases and vector stores
print_status "Initializing storage systems..."
cd "$PROJECT_ROOT"
if [[ -f "legal_ai_system/utils/init_db.py" ]]; then
    python -m legal_ai_system.utils.init_db > "$LOG_DIR/init-db.log" 2>&1 || true
    print_success "Databases initialized"
fi

# Step 6: Start FastAPI backend
print_status "Starting FastAPI backend server..."
cd "$API_DIR"
uvicorn legal_ai_system.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info \
    > "$LOG_DIR/fastapi.log" 2>&1 &

BACKEND_PID=$!
print_success "FastAPI backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
if wait_for_service "http://localhost:8000/health" "FastAPI backend"; then
    print_success "Backend health check passed"
else
    print_error "Backend failed to start properly"
    exit 1
fi

# Step 7: Start React frontend
print_status "Starting React frontend development server..."
cd "$FRONTEND_DIR"
npm start > "$LOG_DIR/react.log" 2>&1 &

FRONTEND_PID=$!
print_success "React frontend started (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
if wait_for_service "http://localhost:3000" "React frontend"; then
    print_success "Frontend is ready"
else
    print_error "Frontend failed to start properly"
    exit 1
fi

# Step 8: Display startup summary
echo ""
echo "🎉 Legal AI System Development Environment Ready!"
echo "=================================================="
echo ""
echo "📊 Service Status:"
echo "  • FastAPI Backend:  http://localhost:8000"
echo "  • GraphQL Endpoint: http://localhost:8000/graphql"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • React Frontend:   http://localhost:3000"
echo "  • Neo4j Browser:    http://localhost:7474 (neo4j/CaseDBMS)"
echo ""
echo "📝 Log Files:"
echo "  • Backend:  $LOG_DIR/fastapi.log"
echo "  • Frontend: $LOG_DIR/react.log"
echo "  • Neo4j:    $LOG_DIR/neo4j.log"
echo "  • Ollama:   $LOG_DIR/ollama.log"
echo ""
echo "🔧 Development Commands:"
echo "  • View backend logs:  tail -f $LOG_DIR/fastapi.log"
echo "  • View frontend logs: tail -f $LOG_DIR/react.log"
echo "  • GraphQL playground: open http://localhost:8000/graphql"
echo "  • Stop all services:  Ctrl+C"
echo ""
echo "⚠️  Environment Variables:"
echo "  • Set XAI_API_KEY for Grok models"
echo "  • Set OPENAI_API_KEY for OpenAI models"
echo "  • Check .env file for other configurations"
echo ""

# Keep script running and monitor services
print_status "Monitoring services... Press Ctrl+C to stop all services"

while true; do
    sleep 10
    
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process died unexpectedly!"
        break
    fi
    
    # Check if frontend is still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend process died unexpectedly!"
        break
    fi
    
    # Optional: Add health checks here
done