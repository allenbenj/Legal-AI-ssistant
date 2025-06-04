#!/bin/bash

# Legal AI System - Production Environment Startup Script
# This script starts the Legal AI System in production mode with proper monitoring

set -e

echo "🚀 Starting Legal AI System Production Environment..."
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/production"
PID_DIR="$PROJECT_ROOT/pids"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

# Function to check if service is running
is_service_running() {
    local pid_file=$1
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

# Function to start service with monitoring
start_service() {
    local service_name=$1
    local command=$2
    local pid_file="$PID_DIR/${service_name}.pid"
    local log_file="$LOG_DIR/${service_name}.log"
    
    if is_service_running "$pid_file"; then
        print_warning "$service_name is already running"
        return 0
    fi
    
    print_status "Starting $service_name..."
    
    # Start service in background and capture PID
    nohup bash -c "$command" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    # Wait a moment and check if process is still running
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        print_success "$service_name started (PID: $pid)"
        return 0
    else
        print_error "$service_name failed to start"
        rm -f "$pid_file"
        return 1
    fi
}

# Function to stop service
stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if is_service_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        print_status "Stopping $service_name (PID: $pid)..."
        
        # Try graceful shutdown first
        kill -TERM "$pid" 2>/dev/null
        
        # Wait up to 10 seconds for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            print_warning "Force killing $service_name..."
            kill -KILL "$pid" 2>/dev/null
        fi
        
        rm -f "$pid_file"
        print_success "$service_name stopped"
    else
        print_warning "$service_name is not running"
    fi
}

# Function to get service status
service_status() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if is_service_running "$pid_file"; then
        local pid=$(cat "$pid_file")
        echo -e "${GREEN}✓${NC} $service_name (PID: $pid)"
    else
        echo -e "${RED}✗${NC} $service_name (not running)"
    fi
}

# Function to cleanup on exit
cleanup() {
    print_status "Received shutdown signal, stopping all services..."
    stop_service "fastapi"
    stop_service "neo4j"
    stop_service "ollama"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check environment
print_status "Checking production environment..."

# Ensure we're in production mode
export APP_ENV=production
export DEBUG=false

# Check required environment variables
required_vars=("XAI_API_KEY" "NEO4J_PASSWORD")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    print_error "Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please set these variables before starting production:"
    echo "  export XAI_API_KEY='your_api_key'"
    echo "  export NEO4J_PASSWORD='your_password'"
    exit 1
fi

# Check for virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "No virtual environment detected. Production requires a virtual environment."
    echo "Please activate your virtual environment before starting production."
    exit 1
fi

print_success "Environment checks passed"

# Handle command line arguments
case "${1:-start}" in
    "start")
        print_status "Starting all services..."
        
        # Start Neo4j
        if command -v neo4j &> /dev/null; then
            start_service "neo4j" "neo4j console"
        elif command -v docker &> /dev/null; then
            start_service "neo4j" "docker run --name legal-ai-neo4j-prod -p7474:7474 -p7687:7687 -v $PROJECT_ROOT/storage/neo4j/data:/data --env NEO4J_AUTH=neo4j/$NEO4J_PASSWORD neo4j:latest"
        else
            print_warning "Neo4j not available, knowledge graph features will be limited"
        fi
        
        # Start Ollama (if available)
        if command -v ollama &> /dev/null; then
            start_service "ollama" "ollama serve"
        else
            print_warning "Ollama not available, will use API providers only"
        fi
        
        # Start FastAPI with production settings
        cd "$PROJECT_ROOT"
        start_service "fastapi" "uvicorn legal_ai_system.api.main:app --host 0.0.0.0 --port 8000 --workers 4 --access-log --log-level warning"
        
        # Wait for services to be ready
        sleep 5
        
        print_success "Production environment started!"
        echo ""
        echo "📊 Service Status:"
        service_status "fastapi"
        service_status "neo4j"
        service_status "ollama"
        echo ""
        echo "🌐 Endpoints:"
        echo "  • API Server:       http://localhost:8000"
        echo "  • API Documentation: http://localhost:8000/docs"
        echo "  • GraphQL:          http://localhost:8000/graphql"
        echo "  • Health Check:     http://localhost:8000/health"
        echo ""
        echo "📝 Logs:"
        echo "  • FastAPI:  $LOG_DIR/fastapi.log"
        echo "  • Neo4j:    $LOG_DIR/neo4j.log"
        echo "  • Ollama:   $LOG_DIR/ollama.log"
        echo ""
        echo "💡 Management Commands:"
        echo "  • Check status:  $0 status"
        echo "  • Stop services: $0 stop"
        echo "  • Restart:       $0 restart"
        echo "  • View logs:     $0 logs [service]"
        ;;
        
    "stop")
        print_status "Stopping all services..."
        stop_service "fastapi"
        stop_service "neo4j"
        stop_service "ollama"
        print_success "All services stopped"
        ;;
        
    "restart")
        print_status "Restarting all services..."
        "$0" stop
        sleep 2
        "$0" start
        ;;
        
    "status")
        echo "🔍 Service Status:"
        service_status "fastapi"
        service_status "neo4j"
        service_status "ollama"
        ;;
        
    "logs")
        service_name=${2:-"fastapi"}
        log_file="$LOG_DIR/${service_name}.log"
        if [[ -f "$log_file" ]]; then
            tail -f "$log_file"
        else
            print_error "Log file not found: $log_file"
            echo "Available logs:"
            ls -la "$LOG_DIR/"
        fi
        ;;
        
    "health")
        print_status "Checking service health..."
        
        # Check FastAPI health
        if curl -s "http://localhost:8000/health" > /dev/null; then
            print_success "FastAPI is healthy"
        else
            print_error "FastAPI health check failed"
        fi
        
        # Check Neo4j health
        if curl -s "http://localhost:7474" > /dev/null; then
            print_success "Neo4j is healthy"
        else
            print_error "Neo4j health check failed"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service]|health}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status"
        echo "  logs     - Follow logs (specify service: fastapi, neo4j, ollama)"
        echo "  health   - Check service health"
        exit 1
        ;;
esac