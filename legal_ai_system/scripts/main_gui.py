"""
Main GUI Application Entry Point
Unified Legal AI System Interface with full database integration
"""

import sys
import os
from pathlib import Path
import uuid
import sentry_sdk
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

# Import shared components
from shared_components import (
    APIClient, ErrorHandler, DataVisualization, SessionManager,
    DataValidator, UIComponents)
# Add the parent directory to sys.path to resolve imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# MUST be first Streamlit command - configure page
st.set_page_config(
    page_title="Legal AI System - Enhanced GUI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)
sentry_sdk.init(
    dsn="https://2fbad862414aad747dba577c60110470@o4509439121489920.ingest.us.sentry.io/4509439123587072",
    # Add data like request headers and IP for users, if applicable;
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
  traces_sample_rate=1.0,
)
# Fix for Streamlit execution - use absolute imports
try:
    from legal_ai_system.config.constants import Constants
except ImportError:
    # Fallback for when package structure isn't available
    class Constants:
        class Version:
            APP_VERSION = "2.1.0"

# Import our components
from shared_components import SessionManager, ErrorHandler, UIComponents
from database_manager import DatabaseManager, ViolationRecord, MemoryRecord
from ..core.unified_services import get_service_container, register_core_services
from ..services.violation_review import ViolationReviewEntry, ActorInfo
from unified_gui import (
    DocumentProcessorTab, MemoryBrainTab, ViolationReviewTab,
    KnowledgeGraphTab, SettingsLogsTab, AnalysisDashboardTab
)
from xai_integration import XAIIntegratedGUI, XAIGrokClient, check_xai_setup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGUIApplication:
    """Enhanced GUI Application with full database integration"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        register_core_services()
        container = get_service_container()
        try:
            self.violation_review_manager = container.get_service("violation_review_manager")
        except Exception:
            self.violation_review_manager = None
        self._initialize_session_state()
        self._initialize_sample_data()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        SessionManager.init_session_state()
        
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = self.db_manager
        
        if 'current_user' not in st.session_state:
            st.session_state.current_user = "demo_user"
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
    
    def _initialize_sample_data(self):
        """Initialize sample data for demonstration"""
        if 'sample_data_loaded' not in st.session_state:
            self._load_sample_violations()
            self._load_sample_memory()
            st.session_state.sample_data_loaded = True
    
    def _load_sample_violations(self):
        """Load sample violation data"""
        sample_violations = [
            ViolationRecord(
                id="viol_001",
                document_id="doc_001",
                violation_type="Data Privacy",
                severity="HIGH",
                status="PENDING",
                description="Personal data exposed without proper consent mechanism",
                confidence=0.89,
                detected_time=datetime.now()
            ),
            ViolationRecord(
                id="viol_002",
                document_id="doc_002",
                violation_type="Contract Breach",
                severity="CRITICAL",
                status="REVIEWED",
                description="Contract terms contradicting regulatory requirements",
                confidence=0.95,
                detected_time=datetime.now(),
                reviewed_by="legal_team",
                review_time=datetime.now()
            ),
            ViolationRecord(
                id="viol_003",
                document_id="doc_001",
                violation_type="Compliance",
                severity="MEDIUM",
                status="APPROVED",
                description="Missing required disclosure statements",
                confidence=0.76,
                detected_time=datetime.now(),
                reviewed_by="compliance_officer",
                review_time=datetime.now(),
                comments="Verified and approved for remediation"
            )
        ]

        for violation in sample_violations:
            self.db_manager.save_violation(violation)
            if self.violation_review_manager:
                entry = ViolationReviewEntry(
                    case_id="demo_case",
                    violation_type=violation.violation_type,
                    statute="",
                    jurisdiction="",
                    actor=ActorInfo(name="demo", role="system"),
                    harms=[violation.description],
                    created_by="GUI",
                )
                self.violation_review_manager.insert_violation(entry)
    
    def _load_sample_memory(self):
        """Load sample memory data"""
        sample_memories = [
            MemoryRecord(
                id="mem_001",
                memory_type="FACT",
                content="Company XYZ operates under GDPR jurisdiction",
                confidence=0.92,
                source_document="doc_001",
                created_time=datetime.now(),
                last_accessed=datetime.now(),
                access_count=5,
                tags='["gdpr", "jurisdiction", "company"]',
                metadata='{"importance": "high", "verified": true}'
            ),
            MemoryRecord(
                id="mem_002",
                memory_type="RULE",
                content="All personal data processing requires explicit consent",
                confidence=0.98,
                source_document="regulation_doc",
                created_time=datetime.now(),
                last_accessed=datetime.now(),
                access_count=12,
                tags='["consent", "personal_data", "processing"]',
                metadata='{"regulation": "GDPR Article 6", "mandatory": true}'
            )
        ]
        
        for memory in sample_memories:
            self.db_manager.save_memory_entry(memory)
    
    def log_user_action(self, action: str, details: str = ""):
        """Log user actions to database"""
        self.db_manager.log_system_event(
            level="INFO",
            component="GUI",
            message=f"User action: {action}",
            user_id=st.session_state.current_user,
            session_id=st.session_state.session_id,
            metadata={"action": action, "details": details}
        )
    
    def render_sidebar(self):
        """Render sidebar navigation and status"""
        st.sidebar.title("⚖️ Legal AI System")
        st.sidebar.markdown("Professional Legal AI Assistant")
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        
        # Get real analytics from database
        analytics = self.db_manager.get_analytics_data()
        
        api_status = SessionManager.get_api_client().check_health()
        status_indicator = UIComponents.create_status_indicator(api_status.get("status", "ERROR"))
        st.sidebar.markdown(f"**Backend:** {status_indicator}")
        
        # XAI status
        xai_status = check_xai_setup()
        xai_indicator = UIComponents.create_status_indicator("HEALTHY" if xai_status["ready"] else "WARNING")
        st.sidebar.markdown(f"**XAI/Grok:** {xai_indicator}")
        
        if not xai_status["api_key_configured"]:
            st.sidebar.caption("⚠️ XAI API key not configured")
        
        # Real quick stats from database
        st.sidebar.markdown("**Quick Stats:**")
        st.sidebar.write(f"• Documents: {analytics.get('documents', {}).get('total', 0)}")
        st.sidebar.write(f"• Violations: {analytics.get('violations', {}).get('total', 0)}")
        st.sidebar.write(f"• Memory: {analytics.get('memory', {}).get('total', 0)} entries")
        st.sidebar.write(f"• Graph: {analytics.get('knowledge_graph', {}).get('nodes', 0)} nodes")
        
        # User info
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**User:** {st.session_state.current_user}")
        st.sidebar.markdown(f"**Session:** {st.session_state.session_id[:8]}...")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("*Legal AI System v2.1.0*")
    
    def render_main_content(self, selected_tab: str):
        """Render main content area based on selected tab"""
        
        # Enhanced tab classes with database integration
        class EnhancedViolationReviewTab(ViolationReviewTab):
            @staticmethod
            def render():
                st.header("⚠️ Violation Review")
                st.markdown("Review detected violations, manage approval workflows, and track compliance issues.")
                
                db_manager = st.session_state.db_manager
                
                # Get real violation data from database
                violations_data = db_manager.get_violations()
                
                if not violations_data:
                    st.info("No violations found in the database. Sample data will be loaded automatically.")
                    return
                
                # Violation metrics
                st.subheader("Violation Overview")
                
                total_violations = len(violations_data)
                pending_reviews = len([v for v in violations_data if v.status == "PENDING"])
                critical_violations = len([v for v in violations_data if v.severity == "CRITICAL"])
                avg_confidence = sum(v.confidence for v in violations_data) / len(violations_data) if violations_data else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Violations", total_violations)
                col2.metric("Pending Reviews", pending_reviews)
                col3.metric("Critical Issues", critical_violations)
                col4.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                # Filters
                st.subheader("Filter Violations")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    severity_filter = st.multiselect(
                        "Severity",
                        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                        default=["HIGH", "CRITICAL"]
                    )
                
                with col2:
                    status_filter = st.multiselect(
                        "Status",
                        ["PENDING", "REVIEWED", "APPROVED", "REJECTED"],
                        default=["PENDING", "REVIEWED"]
                    )
                
                with col3:
                    available_types = list(set(v.violation_type for v in violations_data))
                    type_filter = st.multiselect(
                        "Violation Type",
                        available_types,
                        default=available_types
                    )
                
                with col4:
                    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
                
                # Apply filters
                filters = {
                    "severity": severity_filter,
                    "status": status_filter,
                    "violation_type": type_filter,
                    "min_confidence": min_confidence
                }
                
                filtered_violations = db_manager.get_violations(filters)
                
                # Violations table
                st.subheader(f"Violations ({len(filtered_violations)} found)")
                
                if filtered_violations:
                    # Convert to display format
                    display_data = []
                    for v in filtered_violations:
                        display_data.append({
                            "ID": v.id,
                            "Type": v.violation_type,
                            "Severity": v.severity,
                            "Status": v.status,
                            "Confidence": f"{v.confidence:.2f}",
                            "Description": v.description[:50] + "..." if len(v.description) > 50 else v.description,
                            "Document": v.document_id
                        })
                    
                    # Display interactive table
                    selected_rows = st.dataframe(
                        display_data,
                        use_container_width=True,
                        hide_index=True,
                        selection_mode="multi",
                        key="violations_table"
                    )
                    
                    # Bulk actions
                    st.subheader("Violation Actions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Approve Selected"):
                            # Get selected violation IDs
                            if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
                                selected_violations = [filtered_violations[i] for i in selected_rows.selection.rows]
                                for violation in selected_violations:
                                    db_manager.update_violation_status(
                                        violation.id, 
                                        "APPROVED", 
                                        st.session_state.current_user,
                                        "Bulk approved via GUI"
                                    )
                                ErrorHandler.display_success(f"Approved {len(selected_violations)} violations")
                                st.rerun()
                    
                    with col2:
                        if st.button("❌ Reject Selected"):
                            if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
                                selected_violations = [filtered_violations[i] for i in selected_rows.selection.rows]
                                for violation in selected_violations:
                                    db_manager.update_violation_status(
                                        violation.id, 
                                        "REJECTED", 
                                        st.session_state.current_user,
                                        "Bulk rejected via GUI"
                                    )
                                ErrorHandler.display_success(f"Rejected {len(selected_violations)} violations")
                                st.rerun()
                    
                    with col3:
                        comment = st.text_input("Add Comment", placeholder="Enter review comment...")
                        if st.button("💬 Add Comment") and comment:
                            if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
                                selected_violations = [filtered_violations[i] for i in selected_rows.selection.rows]
                                for violation in selected_violations:
                                    db_manager.update_violation_status(
                                        violation.id, 
                                        violation.status, 
                                        st.session_state.current_user,
                                        comment
                                    )
                                ErrorHandler.display_success(f"Added comment to {len(selected_violations)} violations")
                                st.rerun()
                
                else:
                    st.info("No violations match the current filters")
        
        class EnhancedMemoryBrainTab(MemoryBrainTab):
            @staticmethod
            def render():
                st.header("🧠 Memory Brain")
                st.markdown("Manage AI memory, visualize memory associations, and analyze memory usage patterns.")
                
                db_manager = st.session_state.db_manager
                
                # Get real memory data from database
                memory_entries = db_manager.get_memory_entries()
                
                if not memory_entries:
                    st.info("No memory entries found. Sample data will be loaded automatically.")
                    return
                
                # Memory overview metrics
                st.subheader("Memory Overview")
                
                total_entries = len(memory_entries)
                type_counts = {}
                for entry in memory_entries:
                    type_counts[entry.memory_type] = type_counts.get(entry.memory_type, 0) + 1
                
                avg_confidence = sum(e.confidence for e in memory_entries) / len(memory_entries) if memory_entries else 0
                total_access = sum(e.access_count for e in memory_entries)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Entries", total_entries)
                col2.metric("Avg Confidence", f"{avg_confidence:.2f}")
                col3.metric("Total Accesses", total_access)
                col4.metric("Active Types", len(type_counts))
                
                # Memory filters
                st.subheader("Memory Filters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    available_types = list(set(e.memory_type for e in memory_entries))
                    memory_type_filter = st.multiselect(
                        "Memory Type",
                        available_types,
                        default=available_types
                    )
                
                with col2:
                    confidence_filter = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
                
                with col3:
                    search_content = st.text_input("Search Content", placeholder="Search in memory content...")
                
                # Apply filters
                filters = {
                    "memory_type": memory_type_filter,
                    "min_confidence": confidence_filter
                }
                if search_content:
                    filters["search_content"] = search_content
                
                filtered_entries = db_manager.get_memory_entries(filters)
                
                # Memory data table
                st.subheader("Memory Entries")
                
                if filtered_entries:
                    # Convert to display format
                    display_data = []
                    for entry in filtered_entries:
                        display_data.append({
                            "ID": entry.id,
                            "Type": entry.memory_type,
                            "Content": entry.content[:100] + "..." if len(entry.content) > 100 else entry.content,
                            "Confidence": f"{entry.confidence:.2f}",
                            "Source": entry.source_document,
                            "Access Count": entry.access_count,
                            "Last Accessed": entry.last_accessed.strftime("%Y-%m-%d %H:%M")
                        })
                    
                    # Interactive table with memory access tracking
                    selected_memory = st.dataframe(
                        display_data,
                        use_container_width=True,
                        hide_index=True,
                        selection_mode="single",
                        key="memory_table"
                    )
                    
                    # Update access count when memory is selected
                    if hasattr(selected_memory, 'selection') and selected_memory.selection.rows:
                        selected_idx = selected_memory.selection.rows[0]
                        selected_entry = filtered_entries[selected_idx]
                        
                        # Update access count
                        db_manager.update_memory_access(selected_entry.id)
                        
                        # Show detailed view
                        st.subheader("Memory Details")
                        st.write(f"**ID:** {selected_entry.id}")
                        st.write(f"**Type:** {selected_entry.memory_type}")
                        st.write(f"**Confidence:** {selected_entry.confidence:.3f}")
                        st.write(f"**Source:** {selected_entry.source_document}")
                        st.write(f"**Created:** {selected_entry.created_time}")
                        st.write(f"**Content:**")
                        st.text_area("", value=selected_entry.content, height=100, disabled=True)
                        
                        if selected_entry.tags:
                            st.write(f"**Tags:** {selected_entry.tags}")
                        if selected_entry.metadata:
                            st.write(f"**Metadata:** {selected_entry.metadata}")
                
                else:
                    st.info("No memory entries match the current filters")
        
        # Route to appropriate enhanced tab
        if selected_tab == "violations":
            EnhancedViolationReviewTab.render()
        elif selected_tab == "memory":
            EnhancedMemoryBrainTab.render()
        elif selected_tab == "dashboard":
            AnalysisDashboardTab.render()
        elif selected_tab == "processor":
            DocumentProcessorTab.render()
        elif selected_tab == "xai_processor":
            XAIIntegratedGUI.render_xai_document_processor()
        elif selected_tab == "graph":
            KnowledgeGraphTab.render()
        elif selected_tab == "settings":
            SettingsLogsTab.render()
    
    def run(self):
        """Main application entry point"""
        
        # Render sidebar
        self.render_sidebar()
        
        # Navigation tabs
        tab_options = {
            "📊 Analysis Dashboard": "dashboard",
            "📄 Document Processor": "processor", 
            "🤖 XAI Document Processor": "xai_processor",
            "🧠 Memory Brain": "memory",
            "⚠️ Violation Review": "violations",
            "🕸️ Knowledge Graph": "graph",
            "⚙️ Settings & Logs": "settings"
        }
        
        selected_tab_name = st.sidebar.radio("Navigation", list(tab_options.keys()))
        selected_tab = tab_options[selected_tab_name]
        
        # Log tab navigation
        self.log_user_action(f"Navigated to {selected_tab_name}")
        
        # Render main content
        self.render_main_content(selected_tab)

def main():
    """Application entry point"""
    try:
        app = EnhancedGUIApplication()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application startup failed: {e}")

if __name__ == "__main__":
    main()
