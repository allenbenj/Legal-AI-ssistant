#!/usr/bin/env python3
"""
Startup script that fixes Python path issues for the Legal AI System.
This ensures the system can find both the installed packages and local modules.
"""

import sys
import os
from pathlib import Path

# Get the current directory (Legal AI System root)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"🔧 Project Root: {PROJECT_ROOT}")
print(f"🐍 Python Executable: {sys.executable}")
print(f"📦 Python Path: {sys.path[:3]}...")  # Show first 3 entries

# Set environment variables for the project
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)

# Try to import and start the API
try:
    print("🚀 Starting Legal AI System API...")
    
    # Import uvicorn
    import uvicorn  # type: ignore
    
    # Import our API app using absolute package path
    # The API implementation resides in legal_ai_system.scripts.main
    # rather than the older api.main location referenced by legacy scripts.
    from legal_ai_system.scripts.main import app
    
    print("✅ Imports successful, starting server...")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)],
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"🔍 Available packages:")
    
    # Show installed packages
    import subprocess
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        print(result.stdout)
    except Exception:
        print("Could not list packages")
        
except Exception as e:
    print(f"❌ Startup Error: {e}")
    import traceback
    traceback.print_exc()