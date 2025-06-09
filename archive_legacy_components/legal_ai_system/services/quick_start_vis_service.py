#!/usr/bin/env python3
"""
Quick start script that bypasses complex imports and starts a simple server.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("🚀 Quick Start - Legal AI System")
print("=" * 40)

try:
    # Try direct import
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    
    # Create a simple FastAPI app
    app = FastAPI(title="Legal AI System", version="1.0.0")
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root() -> dict:
        return {"message": "Legal AI System API", "status": "running"}
    
    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy", "api": "running"}
    
    @app.get("/api/v1/status")
    async def api_status() -> dict:
        return {
            "api_version": "1.0.0",
            "system": "Legal AI System",
            "status": "operational"
        }
    
    print("✅ FastAPI app created successfully")
    print("🌐 Starting server on http://localhost:8000")
    print("📖 API docs will be at http://localhost:8000/docs")
    print("⚡ Health check: http://localhost:8000/health")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Please install required packages:")
    print("   pip install fastapi uvicorn")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()