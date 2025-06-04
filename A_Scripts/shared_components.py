"""
Shared Components for Legal AI System GUI
Provides reusable components for data fetching, visualization, and error handling
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class APIClient:
    """Handles all API communications with the Legal AI System backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self._health_status = None  # Cache health status to avoid repeated checks
        
    def check_health(self) -> Dict[str, Any]:
        """Check system health status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/system/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            # Gracefully handle when backend server is not running
            logger.warning(f"Backend server not available at {self.base_url}")
            return {"status": "BACKEND_UNAVAILABLE", "message": "Backend server is not running"}
        except requests.exceptions.Timeout:
            logger.warning(f"Health check timeout for {self.base_url}")
            return {"status": "TIMEOUT", "message": "Health check timed out"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "ERROR", "message": str(e)}
    
    def upload_document(self, file_data: bytes, filename: str, file_type: str) -> Dict[str, Any]:
        """Upload document to the system"""
        try:
            files = {'file': (filename, file_data, file_type)}
            response = self.session.post(f"{self.base_url}/api/v1/documents/upload", files=files)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return {"status": "ERROR", "message": str(e)}
    
    def process_document(self, document_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process document with specified options"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/documents/{document_id}/process",
                json={"processing_options": options}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {"status": "ERROR", "message": str(e)}
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get document processing status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/documents/{document_id}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get document status failed: {e}")
            return {"status": "ERROR", "message": str(e)}
    
    def get_memory_data(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve memory data with optional filters"""
        try:
            params = filters or {}
            response = self.session.get(f"{self.base_url}/api/v1/memory", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get memory data failed: {e}")
            return []
    
    def get_violations(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve violations with optional filters"""
        try:
            params = filters or {}
            response = self.session.get(f"{self.base_url}/api/v1/violations", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get violations failed: {e}")
            return []
    
    def get_knowledge_graph_data(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve knowledge graph data"""
        try:
            params = {"query": query} if query else {}
            response = self.session.get(f"{self.base_url}/api/v1/knowledge-graph", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get knowledge graph failed: {e}")
            return {"nodes": [], "edges": []}

class ErrorHandler:
    """Centralized error handling and user notifications"""
    
    @staticmethod
    def display_error(message: str, details: Optional[str] = None):
        """Display error message to user"""
        st.error(f"⚠️ {message}")
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def display_warning(message: str):
        """Display warning message to user"""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def display_success(message: str):
        """Display success message to user"""
        st.success(f"✅ {message}")
    
    @staticmethod
    def display_info(message: str):
        """Display info message to user"""
        st.info(f"ℹ️ {message}")

class DataVisualization:
    """Shared visualization components"""
    
    @staticmethod
    def create_metrics_dashboard(metrics: Dict[str, Any]) -> None:
        """Create metrics dashboard with key statistics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Documents Processed",
                value=metrics.get("documents_processed", 0),
                delta=metrics.get("documents_delta", 0)
            )
        
        with col2:
            st.metric(
                label="Active Workflows",
                value=metrics.get("active_workflows", 0),
                delta=metrics.get("workflows_delta", 0)
            )
        
        with col3:
            st.metric(
                label="Pending Reviews",
                value=metrics.get("pending_reviews", 0),
                delta=metrics.get("reviews_delta", 0)
            )
        
        with col4:
            st.metric(
                label="System Health",
                value=metrics.get("system_health", "Unknown"),
                delta=None
            )
    
    @staticmethod
    def create_timeline_chart(data: List[Dict], title: str = "Activity Timeline") -> go.Figure:
        """Create timeline chart for activities"""
        if not data:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        df = pd.DataFrame(data)
        fig = px.timeline(
            df,
            x_start="start_time",
            x_end="end_time",
            y="activity",
            color="status",
            title=title
        )
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_entity_distribution_chart(entities: List[Dict]) -> go.Figure:
        """Create entity type distribution chart"""
        if not entities:
            return go.Figure().add_annotation(text="No entities found", showarrow=False)
        
        entity_counts = {}
        for entity in entities:
            entity_type = entity.get("type", "Unknown")
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        fig = px.pie(
            values=list(entity_counts.values()),
            names=list(entity_counts.keys()),
            title="Entity Type Distribution"
        )
        return fig
    
    @staticmethod
    def create_confidence_histogram(data: List[Dict], field: str = "confidence") -> go.Figure:
        """Create confidence score histogram"""
        if not data:
            return go.Figure().add_annotation(text="No confidence data available", showarrow=False)
        
        confidence_scores = [item.get(field, 0) for item in data if item.get(field) is not None]
        
        fig = px.histogram(
            x=confidence_scores,
            title="Confidence Score Distribution",
            labels={"x": "Confidence Score", "y": "Count"}
        )
        fig.update_layout(height=300)
        return fig

class SessionManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def init_session_state():
        """Initialize session state variables"""
        if 'api_client' not in st.session_state:
            st.session_state.api_client = APIClient()
        
        if 'current_document_id' not in st.session_state:
            st.session_state.current_document_id = None
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        
        if 'selected_violations' not in st.session_state:
            st.session_state.selected_violations = []
        
        if 'memory_filters' not in st.session_state:
            st.session_state.memory_filters = {}
        
        if 'graph_query' not in st.session_state:
            st.session_state.graph_query = ""
    
    @staticmethod
    def get_api_client() -> APIClient:
        """Get API client from session state"""
        return st.session_state.api_client
    
    @staticmethod
    def set_current_document(document_id: str):
        """Set current document ID"""
        st.session_state.current_document_id = document_id
    
    @staticmethod
    def get_current_document() -> Optional[str]:
        """Get current document ID"""
        return st.session_state.current_document_id

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_file_upload(file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if file is None:
            return False, "No file uploaded"
        
        # Check file size (max 50MB)
        if file.size > 50 * 1024 * 1024:
            return False, "File size exceeds 50MB limit"
        
        # Check file type
        allowed_types = ['pdf', 'docx', 'txt', 'md', 'doc']
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            return False, f"File type '{file_extension}' not supported. Allowed types: {allowed_types}"
        
        return True, "File validation passed"
    
    @staticmethod
    def validate_processing_options(options: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate processing options"""
        required_fields = ['enable_ner', 'enable_llm_extraction']
        for field in required_fields:
            if field not in options:
                return False, f"Missing required field: {field}"
        
        if 'confidence_threshold' in options:
            threshold = options['confidence_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                return False, "Confidence threshold must be between 0 and 1"
        
        return True, "Processing options valid"

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def create_status_indicator(status: str) -> str:
        """Create status indicator with appropriate color"""
        status_colors = {
            "HEALTHY": "🟢",
            "WARNING": "🟡",
            "ERROR": "🔴",
            "PROCESSING": "🟡",
            "COMPLETED": "🟢",
            "FAILED": "🔴",
            "PENDING": "⚪"
        }
        return f"{status_colors.get(status, '⚪')} {status}"
    
    @staticmethod
    def create_filter_sidebar(filters: Dict[str, Any], filter_options: Dict[str, List]) -> Dict[str, Any]:
        """Create filter sidebar for data filtering"""
        st.sidebar.subheader("Filters")
        
        updated_filters = {}
        for filter_name, options in filter_options.items():
            if filter_name == "date_range":
                start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=30))
                end_date = st.sidebar.date_input("End Date", value=datetime.now())
                updated_filters["start_date"] = start_date.isoformat()
                updated_filters["end_date"] = end_date.isoformat()
            elif filter_name == "status":
                selected_status = st.sidebar.multiselect("Status", options, default=filters.get("status", []))
                updated_filters["status"] = selected_status
            elif filter_name == "type":
                selected_types = st.sidebar.multiselect("Type", options, default=filters.get("type", []))
                updated_filters["type"] = selected_types
            elif filter_name == "confidence":
                min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, filters.get("min_confidence", 0.0))
                updated_filters["min_confidence"] = min_confidence
        
        return updated_filters
    
    @staticmethod
    def create_data_table(data: List[Dict], columns: List[str], key: str = "data_table") -> pd.DataFrame:
        """Create interactive data table"""
        if not data:
            st.write("No data available")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if columns:
            df = df[columns] if all(col in df.columns for col in columns) else df
        
        return st.dataframe(df, key=key, use_container_width=True)
    
    @staticmethod
    def create_progress_bar(current: int, total: int, label: str = "Progress") -> None:
        """Create progress bar with label"""
        if total > 0:
            progress = current / total
            st.progress(progress, text=f"{label}: {current}/{total} ({progress:.1%})")
        else:
            st.progress(0, text=f"{label}: No items")

# Production data handlers - no mock data