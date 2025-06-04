#!/usr/bin/env python3
"""
Minimal Legal AI System API for testing the frontend interface.
This bypasses complex dependencies to get the system running quickly.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import json
import time

# Create FastAPI app
app = FastAPI(
    title="Legal AI System - Minimal API",
    description="Simplified API for testing the frontend interface",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for testing
mock_documents = []
mock_entities = []
mock_graph_data = {
    "nodes": [
        {"id": "person_1", "label": "John Doe", "type": "PERSON", "confidence": 0.95},
        {"id": "case_1", "label": "Case #2024-001", "type": "CASE", "confidence": 0.92},
        {"id": "court_1", "label": "Superior Court", "type": "COURT", "confidence": 0.88}
    ],
    "links": [
        {"source": "person_1", "target": "case_1", "type": "DEFENDANT_IN"},
        {"source": "case_1", "target": "court_1", "type": "FILED_IN"}
    ]
}

# Request models
class DocumentUpload(BaseModel):
    enable_ner: bool = True
    enable_llm_extraction: bool = True
    enable_confidence_calibration: bool = True
    confidence_threshold: float = 0.7

# Routes
@app.get("/")
async def root():
    return {"message": "Legal AI System - Minimal API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "api": "minimal_mode"}

@app.get("/api/v1/documents")
async def get_documents():
    return {"documents": mock_documents}

@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # Simulate document processing
    document_id = f"doc_{int(time.time())}"
    mock_documents.append({
        "id": document_id,
        "filename": file.filename,
        "status": "processing",
        "uploaded_at": time.time()
    })
    
    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": "Document processing started"
    }

@app.get("/api/v1/knowledge-graph/entities")
async def get_entities():
    return {"entities": mock_graph_data["nodes"]}

@app.get("/api/v1/knowledge-graph/data")
async def get_graph_data():
    return mock_graph_data

@app.get("/api/v1/system/status")
async def get_system_status():
    return {
        "overall_health": "healthy",
        "services": {
            "api": "running",
            "frontend": "available",
            "processing": "ready"
        },
        "performance_metrics": {
            "documents_processed": len(mock_documents),
            "entities_extracted": len(mock_entities),
            "average_confidence": 0.85
        }
    }

@app.post("/api/v1/auth/token")
async def login():
    # Mock authentication
    return {
        "access_token": "mock_token_12345",
        "token_type": "bearer",
        "user": {
            "id": "user_1",
            "username": "demo_user",
            "email": "demo@legal-ai.com",
            "access_level": "admin"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Minimal Legal AI System API")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend should connect to: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)