#!/usr/bin/env python3
"""
Simple test script to check if the Legal AI System can start.
Run this to diagnose startup issues.
"""

import sys
import os
from pathlib import Path

print("🔍 Legal AI System Startup Diagnostics")
print("=" * 50)

# Check Python version
print(f"✓ Python Version: {sys.version}")
print(f"✓ Python Path: {sys.executable}")

# Check current directory
current_dir = Path.cwd()
print(f"✓ Current Directory: {current_dir}")

# Check if we're in the right directory
legal_ai_files = ['api', 'core', 'agents', 'my-legal-tech-gui']
missing_files = []
for file in legal_ai_files:
    if not (current_dir / file).exists():
        missing_files.append(file)

if missing_files:
    print(f"❌ Missing directories: {missing_files}")
    print("❌ You may not be in the correct directory")
else:
    print("✓ All required directories found")

# Test importing key modules
print("\n🧪 Testing Module Imports:")
try:
    import fastapi
    print(f"✓ FastAPI: {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI not found: {e}")

try:
    import uvicorn
    print(f"✓ Uvicorn: {uvicorn.__version__}")
except ImportError as e:
    print(f"❌ Uvicorn not found: {e}")

try:
    from pathlib import Path
    print("✓ Pathlib available")
except ImportError as e:
    print(f"❌ Pathlib not found: {e}")

# Check if port 8000 is available
import socket
def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

print(f"\n🌐 Network Checks:")
print(f"✓ Port 8000 available: {check_port(8000)}")
print(f"✓ Port 3000 available: {check_port(3000)}")

# Test basic API import
print(f"\n🚀 Testing API Import:")
try:
    sys.path.append(str(current_dir))
    from api.main import app
    print("✓ API module imports successfully")
    print("✓ FastAPI app created")
except Exception as e:
    print(f"❌ API import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Diagnostics complete!")
print("\nIf you see ❌ errors above, those need to be fixed first.")
print("If all ✓ checks pass, try running:")
print("  python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")