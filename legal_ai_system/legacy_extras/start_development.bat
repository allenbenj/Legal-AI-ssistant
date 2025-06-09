@echo off
setlocal EnableDelayedExpansion

REM Legal AI System - Development Environment Startup Script (Windows)
REM This script coordinates the startup of all system components in development mode

echo 🚀 Starting Legal AI System Development Environment...
echo ==================================================

REM Configuration
set "PROJECT_ROOT=%~dp0.."
set "FRONTEND_DIR=%PROJECT_ROOT%\my-legal-tech-gui"
set "API_DIR=%PROJECT_ROOT%\api"
set "LOG_DIR=%PROJECT_ROOT%\logs"

REM Create logs directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Function to print timestamped output
:print_status
echo [%time%] %~1
goto :eof

:print_success
echo [%time%] ✓ %~1
goto :eof

:print_error
echo [%time%] ✗ %~1
goto :eof

:print_warning
echo [%time%] ⚠ %~1
goto :eof

REM Check prerequisites
call :print_status "Checking prerequisites..."

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is not installed or not in PATH"
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Node.js is not installed or not in PATH"
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    call :print_error "npm is not installed or not in PATH"
    pause
    exit /b 1
)

call :print_success "All required tools are available"

REM Check for virtual environment
if "%VIRTUAL_ENV%"=="" (
    call :print_warning "No virtual environment detected. Consider activating one:"
    echo   python -m venv legal_ai_env
    echo   legal_ai_env\Scripts\activate
)

REM Step 1: Check/Start Neo4j (Docker)
call :print_status "Checking Neo4j availability..."
docker --version >nul 2>&1
if not errorlevel 1 (
    call :print_status "Starting Neo4j via Docker..."
    docker ps | findstr "legal-ai-neo4j" >nul 2>&1
    if errorlevel 1 (
        docker run --name legal-ai-neo4j ^
            -p7474:7474 -p7687:7687 ^
            -d ^
            -v "%PROJECT_ROOT%\storage\neo4j\data:/data" ^
            -v "%PROJECT_ROOT%\storage\neo4j\logs:/logs" ^
            --env NEO4J_AUTH=neo4j/CaseDBMS ^
            neo4j:latest >"%LOG_DIR%\neo4j-docker.log" 2>&1
        timeout /t 10 /nobreak >nul
        call :print_success "Neo4j started via Docker"
    ) else (
        call :print_success "Neo4j is already running"
    )
) else (
    call :print_warning "Docker not found. Neo4j features may be limited."
)

REM Step 2: Check Ollama
call :print_status "Checking Ollama availability..."
ollama --version >nul 2>&1
if not errorlevel 1 (
    REM Check if Ollama is running
    tasklist | findstr "ollama" >nul 2>&1
    if errorlevel 1 (
        call :print_status "Starting Ollama server..."
        start /B ollama serve >"%LOG_DIR%\ollama.log" 2>&1
        timeout /t 3 /nobreak >nul
        
        call :print_status "Ensuring required models are available..."
        start /B ollama pull llama3.2
        start /B ollama pull nomic-embed-text
    ) else (
        call :print_success "Ollama is already running"
    )
) else (
    call :print_warning "Ollama not found. Will use API-based LLM providers only."
)

REM Step 3: Install backend dependencies
call :print_status "Installing/updating backend dependencies..."
cd /d "%PROJECT_ROOT%"
if exist "requirements.txt" (
    pip install -r requirements.txt >"%LOG_DIR%\pip-install.log" 2>&1
    call :print_success "Backend dependencies installed"
) else (
    call :print_warning "requirements.txt not found, skipping backend dependency installation"
)

REM Step 4: Install frontend dependencies
call :print_status "Installing/updating frontend dependencies..."
cd /d "%FRONTEND_DIR%"
if exist "package.json" (
    npm install >"%LOG_DIR%\npm-install.log" 2>&1
    call :print_success "Frontend dependencies installed"
) else (
    call :print_error "Frontend package.json not found"
    pause
    exit /b 1
)

REM Step 5: Initialize databases
call :print_status "Initializing storage systems..."
cd /d "%PROJECT_ROOT%"
if exist "legal_ai_system\utils\init_db.py" (
    python -m legal_ai_system.utils.init_db >"%LOG_DIR%\init-db.log" 2>&1
    call :print_success "Databases initialized"
)

REM Step 6: Start FastAPI backend
call :print_status "Starting FastAPI backend server..."
cd /d "%API_DIR%"
start "FastAPI Backend" /B uvicorn legal_ai_system.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info >"%LOG_DIR%\fastapi.log" 2>&1

REM Wait for backend to be ready
call :print_status "Waiting for backend to be ready..."
timeout /t 10 /nobreak >nul

REM Step 7: Start React frontend
call :print_status "Starting React frontend development server..."
cd /d "%FRONTEND_DIR%"
start "React Frontend" /B npm start >"%LOG_DIR%\react.log" 2>&1

REM Wait for frontend to be ready
call :print_status "Waiting for frontend to be ready..."
timeout /t 15 /nobreak >nul

REM Step 8: Display startup summary
echo.
echo 🎉 Legal AI System Development Environment Ready!
echo ==================================================
echo.
echo 📊 Service Status:
echo   • FastAPI Backend:  http://localhost:8000
echo   • GraphQL Endpoint: http://localhost:8000/graphql
echo   • API Documentation: http://localhost:8000/docs
echo   • React Frontend:   http://localhost:3000
echo   • Neo4j Browser:    http://localhost:7474 (neo4j/CaseDBMS)
echo.
echo 📝 Log Files:
echo   • Backend:  %LOG_DIR%\fastapi.log
echo   • Frontend: %LOG_DIR%\react.log
echo   • Neo4j:    %LOG_DIR%\neo4j-docker.log
echo   • Ollama:   %LOG_DIR%\ollama.log
echo.
echo 🔧 Development Commands:
echo   • View backend logs:  type "%LOG_DIR%\fastapi.log"
echo   • View frontend logs: type "%LOG_DIR%\react.log"
echo   • GraphQL playground: http://localhost:8000/graphql
echo   • Stop services: Close this window or Ctrl+C
echo.
echo ⚠️  Environment Variables:
echo   • Set XAI_API_KEY for Grok models
echo   • Set OPENAI_API_KEY for OpenAI models
echo   • Check .env file for other configurations
echo.

REM Open browser windows
call :print_status "Opening browser windows..."
start http://localhost:8000/docs
timeout /t 3 /nobreak >nul
start http://localhost:3000

echo Press any key to stop all services...
pause >nul

REM Cleanup - Kill started processes
call :print_status "Shutting down services..."
taskkill /F /FI "WINDOWTITLE eq FastAPI Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq React Frontend*" >nul 2>&1
docker stop legal-ai-neo4j >nul 2>&1

call :print_success "Services stopped"
echo Development environment shut down.
pause