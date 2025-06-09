@echo off
echo 🔧 Setting up Legal AI System Environment...

REM Navigate to project directory
cd /d "E:\core_system\legal_ai_system"

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install fastapi uvicorn python-multipart pydantic strawberry-graphql websockets

REM Try to install from requirements.txt if it exists
if exist "requirements.txt" (
    echo Installing from requirements.txt...
    pip install -r requirements.txt
)

REM Start the backend server
echo 🚀 Starting Legal AI System Backend...
echo Backend will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

pause