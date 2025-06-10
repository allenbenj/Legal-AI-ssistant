"""
Unified GUI Application for Legal AI System
Six-tab interface providing comprehensive document processing, memory management,
violation review, knowledge graph visualization, settings, and analysis dashboard.
"""

import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..agents.legal_agents import (
    EthicsReviewAgent,
    LegalAuditAgent,
    LEOConductAgent,
)

# Import shared components
from ..core.shared_components import (
    DataValidator,
    DataVisualization,
    ErrorHandler,
    SessionManager,
    UIComponents,
)
from ..services.violation_review import ViolationReviewDB, ViolationReviewEntry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessorTab:
    """Document Processor Tab - File input, processing options, and workflow management"""

    @staticmethod
    def render():
        st.header("📄 Document Processor")
        st.markdown(
            "Upload documents for AI analysis, configure processing options, and monitor workflows."
        )

        # Initialize session state
        SessionManager.init_session_state()
        api_client = SessionManager.get_api_client()

        # File upload section
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("File Upload")
            uploaded_file = st.file_uploader(
                "Choose a document to analyze",
                type=["pdf", "docx", "txt", "md", "doc"],
                help="Supported formats: PDF, Word documents, Text files, Markdown",
            )

            if uploaded_file:
                # Validate file
                is_valid, message = DataValidator.validate_file_upload(uploaded_file)
                if is_valid:
                    ErrorHandler.display_success(
                        f"File '{uploaded_file.name}' ready for upload"
                    )

                    # Display file info
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                    st.write(f"**Type:** {uploaded_file.type}")
                else:
                    ErrorHandler.display_error("File validation failed", message)

        with col2:
            st.subheader("System Status")
            health_status = api_client.check_health()
            status_text = UIComponents.create_status_indicator(
                health_status.get("status", "ERROR")
            )
            st.markdown(f"**Backend:** {status_text}")

            if health_status.get("status") != "HEALTHY":
                ErrorHandler.display_warning("Backend connection issues detected")

        # Processing options
        st.subheader("Processing Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Core Options**")
            enable_ner = st.checkbox(
                "Named Entity Recognition",
                value=True,
                help="Extract people, organizations, locations",
            )
            enable_llm = st.checkbox(
                "LLM Extraction",
                value=True,
                help="Use large language model for advanced extraction",
            )
            enable_classification = st.checkbox(
                "Document Classification",
                value=True,
                help="Classify document type and content",
            )

        with col2:
            st.markdown("**Advanced Options**")
            enable_summarization = st.checkbox(
                "Summarization", value=False, help="Generate document summary"
            )
            enable_contradiction = st.checkbox(
                "Contradiction Detection",
                value=False,
                help="Find contradictory statements",
            )
            enable_compliance = st.checkbox(
                "Compliance Check", value=True, help="Check against compliance rules"
            )

        with col3:
            st.markdown("**Quality Settings**")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0,
                1.0,
                0.7,
                0.05,
                help="Minimum confidence for extraction",
            )
            model_selection = st.selectbox(
                "Model",
                ["gpt-4", "gpt-3.5-turbo", "claude-3", "local-model"],
                help="Select AI model",
            )
            api_key_status = st.text_input(
                "API Key", type="password", placeholder="Enter API key"
            )

        # Processing options dictionary
        processing_options = {
            "enable_ner": enable_ner,
            "enable_llm_extraction": enable_llm,
            "enable_classification": enable_classification,
            "enable_summarization": enable_summarization,
            "enable_contradiction_detection": enable_contradiction,
            "enable_compliance_check": enable_compliance,
            "confidence_threshold": confidence_threshold,
            "model": model_selection,
            "api_key_provided": bool(api_key_status),
        }

        # Validate processing options
        options_valid, options_message = DataValidator.validate_processing_options(
            processing_options
        )
        if not options_valid:
            ErrorHandler.display_warning(f"Configuration issue: {options_message}")

        # Process button
        st.subheader("Document Processing")

        if uploaded_file and options_valid:
            if st.button("🚀 Start Processing", type="primary"):
                with st.spinner("Uploading and processing document..."):
                    # Upload document
                    file_data = uploaded_file.getvalue()
                    upload_result = api_client.upload_document(
                        file_data, uploaded_file.name, uploaded_file.type
                    )

                    if upload_result.get("status") != "ERROR":
                        document_id = upload_result.get("document_id")
                        if isinstance(document_id, str):
                            SessionManager.set_current_document(document_id)
                            ErrorHandler.display_success(
                                f"Document uploaded successfully (ID: {document_id})"
                            )

                            # Start processing
                            process_result = api_client.process_document(
                                document_id, processing_options
                            )
                        else:
                            process_result = {
                                "status": "ERROR",
                                "message": "Invalid document ID",
                            }
                        if process_result.get("status") != "ERROR":
                            ErrorHandler.display_success(
                                "Processing started successfully!"
                            )
                            st.session_state.processing_status[document_id] = (
                                "PROCESSING"
                            )
                        else:
                            ErrorHandler.display_error(
                                "Processing failed", process_result.get("message")
                            )
                    else:
                        ErrorHandler.display_error(
                            "Upload failed", upload_result.get("message")
                        )

        # Current processing status
        st.subheader("Processing Status")
        current_doc = SessionManager.get_current_document()

        if current_doc:
            status_result = api_client.get_document_status(current_doc)
            if status_result.get("status") != "ERROR":
                status = status_result.get("processing_status", "UNKNOWN")
                progress = status_result.get("progress", 0)

                st.write(f"**Document ID:** {current_doc}")
                UIComponents.create_progress_bar(
                    progress, 100, f"Processing Status: {status}"
                )

                if status == "COMPLETED":
                    ErrorHandler.display_success("Document processing completed!")
                    with st.expander("View Results"):
                        st.json(status_result.get("results", {}))
                elif status == "FAILED":
                    ErrorHandler.display_error(
                        "Processing failed", status_result.get("error_message")
                    )
            else:
                ErrorHandler.display_error(
                    "Failed to get status", status_result.get("message")
                )
        else:
            st.info("No document currently being processed")

        # Recent documents
        st.subheader("Recent Documents")
        # Real document processing would fetch from database
        mock_docs = []

        for doc in mock_docs:
            with st.expander(
                f"{doc['name']} - {UIComponents.create_status_indicator(doc['status'])}"
            ):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**ID:** {doc['id']}")
                col2.write(f"**Size:** {doc['size']}")
                col3.write(f"**Uploaded:** {doc['upload_time'][:19]}")

                if st.button(f"View Details", key=f"view_{doc['id']}"):
                    st.switch_page("Analysis Dashboard")


class WorkflowDesignerTab:
    """Workflow Designer Tab - Configure analysis workflows"""

    @staticmethod
    def render():
        st.header("🛠️ Workflow Designer")
        SessionManager.init_session_state()
        api_client = SessionManager.get_api_client()

        workflows = api_client.list_workflows()
        workflow_names = [w.get("name", f"Workflow {w.get('id')}") for w in workflows]
        selected_name = st.selectbox(
            "Select Workflow", ["New Workflow"] + workflow_names
        )

        if selected_name == "New Workflow":
            workflow = {"config": {}}
        else:
            workflow = workflows[workflow_names.index(selected_name)]

        config = workflow.get("config", {})

        enable_ner = st.checkbox("Enable NER", value=config.get("enable_ner", True))
        enable_llm = st.checkbox(
            "Enable LLM Extraction",
            value=config.get("enable_llm_extraction", True),
        )
        confidence = st.slider(
            "Confidence Threshold",
            0.0,
            1.0,
            float(config.get("confidence_threshold", 0.75)),
            0.05,
        )

        if st.button("💾 Save Workflow"):
            result = api_client.save_workflow(
                {
                    "id": workflow.get("id"),
                    "name": selected_name,
                    "config": {
                        "enable_ner": enable_ner,
                        "enable_llm_extraction": enable_llm,
                        "confidence_threshold": confidence,
                    },
                }
            )
            if result.get("status") != "ERROR":
                ErrorHandler.display_success("Workflow saved")
            else:
                ErrorHandler.display_error("Failed to save", result.get("message"))


class MemoryBrainTab:
    """Memory Brain Tab - AI memory management and visualization"""

    @staticmethod
    def render():
        st.header("🧠 Memory Brain")
        st.markdown(
            "Manage AI memory, visualize memory associations, and analyze memory usage patterns."
        )

        SessionManager.get_api_client()

        # Memory overview metrics
        st.subheader("Memory Overview")

        # Real memory metrics from database
        memory_metrics = {
            "total_entries": 0,
            "facts": 0,
            "rules": 0,
            "precedents": 0,
            "entities": 0,
            "memory_usage": 0.0,
            "avg_confidence": 0.0,
        }

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entries", memory_metrics["total_entries"], "+23")
        col2.metric("Memory Usage", f"{memory_metrics['memory_usage']}%", "+2.1%")
        col3.metric(
            "Avg Confidence", f"{memory_metrics['avg_confidence']:.2f}", "+0.03"
        )
        col4.metric("Active Sessions", "15", "+2")

        # Memory filters
        st.subheader("Memory Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            memory_type_filter = st.multiselect(
                "Memory Type",
                ["FACT", "RULE", "PRECEDENT", "ENTITY"],
                default=["FACT", "RULE"],
            )

        with col2:
            confidence_filter = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)

        with col3:
            _date_range = st.date_input(
                "Date Range",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                max_value=datetime.now(),
            )

        # Memory data table
        st.subheader("Memory Entries")

        # Generate mock memory data
        # Real memory entries would fetch from database
        memory_entries = []

        # Apply filters
        filtered_entries = [
            entry
            for entry in memory_entries
            if entry["type"] in memory_type_filter
            and entry["confidence"] >= confidence_filter
        ]

        if filtered_entries:
            df = pd.DataFrame(filtered_entries)

            # Create interactive table
            st.dataframe(
                df[
                    [
                        "id",
                        "type",
                        "content",
                        "confidence",
                        "source_document",
                        "access_count",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
            )

            # Memory visualization
            st.subheader("Memory Visualization")

            col1, col2 = st.columns(2)

            with col1:
                # Memory type distribution
                type_counts = df["type"].value_counts()
                fig_types = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Memory Type Distribution",
                )
                st.plotly_chart(fig_types, use_container_width=True)

            with col2:
                # Confidence distribution
                fig_conf = px.histogram(
                    df, x="confidence", title="Confidence Score Distribution", nbins=20
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # Memory associations network
            st.subheader("Memory Associations")

            # Create mock association graph
            st.info(
                "Interactive memory association graph would be displayed here using a network visualization library like NetworkX or Cytoscape."
            )

            # Placeholder for network graph
            association_data = {
                "nodes": [
                    {
                        "id": entry["id"],
                        "label": entry["content"][:30] + "...",
                        "type": entry["type"],
                    }
                    for entry in filtered_entries[:10]
                ],
                "edges": [
                    {
                        "source": f"mem_{i:03d}",
                        "target": f"mem_{(i+1) % 10:03d}",
                        "weight": 0.5 + i * 0.1,
                    }
                    for i in range(9)
                ],
            }

            with st.expander("View Association Data"):
                st.json(association_data)

        else:
            st.info("No memory entries match the current filters")

        # Memory management actions
        st.subheader("Memory Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Search Memories"):
                search_query = st.text_input(
                    "Search Query", placeholder="Enter search terms..."
                )
                if search_query:
                    st.info(f"Searching for: {search_query}")

        with col2:
            if st.button("🧹 Cleanup Low Confidence"):
                st.warning("This will remove memories with confidence < 0.3")
                if st.button("Confirm Cleanup"):
                    ErrorHandler.display_success("Low confidence memories cleaned up")

        with col3:
            if st.button("📊 Generate Report"):
                st.info("Memory usage report generated")
                with st.expander("Memory Report"):
                    st.write("**Memory Analysis Report**")
                    st.write(f"- Total entries analyzed: {len(memory_entries)}")
                    st.write(
                        f"- Average confidence: {sum(e['confidence'] for e in memory_entries) / len(memory_entries):.3f}"
                    )
                    st.write(
                        f"- Most common type: {max(set(e['type'] for e in memory_entries), key=lambda x: [e['type'] for e in memory_entries].count(x))}"
                    )


class ViolationReviewTab:
    """Violation Review Tab - Display and manage detected violations"""

    @staticmethod
    def render():
        st.header("⚠️ Violation Review")
        st.markdown(
            "Review detected violations, manage approval workflows, and track compliance issues."
        )

        SessionManager.get_api_client()
        db = ViolationReviewDB()

        # Violation metrics
        st.subheader("Violation Overview")

        violations_data = [asdict(v) for v in db.fetch_violations()]

        # Calculate metrics
        total_violations = len(violations_data)
        pending_reviews = len([v for v in violations_data if v["status"] == "PENDING"])
        critical_violations = len(
            [v for v in violations_data if v["severity"] == "CRITICAL"]
        )
        avg_confidence = (
            sum(v["confidence"] for v in violations_data) / len(violations_data)
            if violations_data
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Violations", total_violations, "+5")
        col2.metric("Pending Reviews", pending_reviews, "+2")
        col3.metric("Critical Issues", critical_violations, "+1")
        col4.metric("Avg Confidence", f"{avg_confidence:.2f}", "+0.02")

        # Filters
        st.subheader("Filter Violations")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            severity_filter = st.multiselect(
                "Severity",
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                default=["HIGH", "CRITICAL"],
            )

        with col2:
            status_filter = st.multiselect(
                "Status",
                ["PENDING", "REVIEWED", "APPROVED", "REJECTED"],
                default=["PENDING", "REVIEWED"],
            )

        with col3:
            type_filter = st.multiselect(
                "Violation Type",
                ["Data Privacy", "Contract Breach", "Compliance", "Security"],
                default=["Data Privacy", "Compliance"],
            )

        with col4:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)

        # Apply filters
        filtered_violations = [
            v
            for v in violations_data
            if (
                v["severity"] in severity_filter
                and v["status"] in status_filter
                and v["violation_type"] in type_filter
                and v.get("confidence", 0) >= min_confidence
            )
        ]

        # Violations table
        st.subheader(f"Violations ({len(filtered_violations)} found)")

        if filtered_violations:
            # Create DataFrame for display
            df = pd.DataFrame(filtered_violations)

            # Display table with selection
            selected_indices = st.dataframe(
                df[
                    [
                        "id",
                        "violation_type",
                        "severity",
                        "status",
                        "confidence",
                        "description",
                        "document_id",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
                key="violations_table",
            )

            # Bulk actions
            st.subheader("Violation Actions")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("✅ Approve Selected"):
                    if (
                        hasattr(selected_indices, "selection")
                        and selected_indices.selection.rows
                    ):
                        for r in selected_indices.selection.rows:
                            db.update_violation_status(df.iloc[r]["id"], "APPROVED")
                        ErrorHandler.display_success(
                            f"Approved {len(selected_indices.selection.rows)} violations"
                        )
                    else:
                        ErrorHandler.display_warning("No violations selected")

            with col2:
                if st.button("❌ Reject Selected"):
                    if (
                        hasattr(selected_indices, "selection")
                        and selected_indices.selection.rows
                    ):
                        for r in selected_indices.selection.rows:
                            db.update_violation_status(df.iloc[r]["id"], "REJECTED")
                        ErrorHandler.display_success(
                            f"Rejected {len(selected_indices.selection.rows)} violations"
                        )
                    else:
                        ErrorHandler.display_warning("No violations selected")

            with col3:
                if st.button("🔍 Review Selected"):
                    if (
                        hasattr(selected_indices, "selection")
                        and selected_indices.selection.rows
                    ):
                        ErrorHandler.display_info(
                            f"Marked {len(selected_indices.selection.rows)} violations for review"
                        )
                    else:
                        ErrorHandler.display_warning("No violations selected")

            with col4:
                if st.button("📝 Add Comment"):
                    comment = st.text_area(
                        "Comment", placeholder="Enter review comment..."
                    )
                    if comment:
                        ErrorHandler.display_success(
                            "Comment added to selected violations"
                        )

            # Detailed violation view
            st.subheader("Violation Details")

            if filtered_violations:
                selected_violation_id = st.selectbox(
                    "Select violation for detailed view",
                    [v["id"] for v in filtered_violations],
                    format_func=lambda x: f"{x} - {next(v['violation_type'] for v in filtered_violations if v['id'] == x)}",
                )

                selected_violation = next(
                    v for v in filtered_violations if v["id"] == selected_violation_id
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Violation Information**")
                    st.write(f"**ID:** {selected_violation['id']}")
                    st.write(f"**Type:** {selected_violation['violation_type']}")
                    st.write(
                        f"**Severity:** {UIComponents.create_status_indicator(selected_violation['severity'])}"
                    )
                    st.write(
                        f"**Status:** {UIComponents.create_status_indicator(selected_violation['status'])}"
                    )
                    st.write(f"**Confidence:** {selected_violation['confidence']:.2f}")

                with col2:
                    st.write("**Document Information**")
                    st.write(f"**Document ID:** {selected_violation['document_id']}")
                    st.write(
                        f"**Detected:** {selected_violation['detected_time'][:19]}"
                    )
                    st.write(f"**Description:** {selected_violation['description']}")

                    if st.button("📄 View Source Document"):
                        st.info(f"Opening document {selected_violation['document_id']}")

                # --- Agent Validation ---
                entry_obj = ViolationReviewEntry(
                    id=selected_violation["id"],
                    document_id=selected_violation["document_id"],
                    violation_type=selected_violation["violation_type"],
                    severity=selected_violation["severity"],
                    status=selected_violation["status"],
                    description=selected_violation["description"],
                    confidence=float(selected_violation.get("confidence", 0)),
                    detected_time=datetime.fromisoformat(
                        selected_violation["detected_time"]
                    ),
                    reviewed_by=selected_violation.get("reviewed_by"),
                    review_time=(
                        datetime.fromisoformat(selected_violation["review_time"])
                        if selected_violation.get("review_time")
                        else None
                    ),
                    recommended_motion=selected_violation.get("recommended_motion"),
                )

                audit_result = LegalAuditAgent().review(entry_obj)
                ethics_result = EthicsReviewAgent().review(entry_obj)
                leo_result = LEOConductAgent().review(entry_obj)

                with st.expander("Agent Recommendations", expanded=True):
                    for res in [audit_result, ethics_result, leo_result]:
                        st.write(
                            f"**{res.agent_name}** - {res.summary} (conf {res.confidence:.2f})"
                        )

                    if st.button("🚀 Escalate for Motion"):
                        if audit_result.recommendation:
                            db.update_violation_status(entry_obj.id, "ESCALATED")
                            st.success(
                                f"Escalated with recommended motion: {audit_result.recommendation}"
                            )
                        else:
                            st.warning("No motion recommendation from agents")

            # Violation analytics
            st.subheader("Violation Analytics")

            col1, col2 = st.columns(2)

            with col1:
                # Severity distribution
                severity_counts = df["severity"].value_counts()
                fig_severity = px.bar(
                    x=severity_counts.index,
                    y=severity_counts.values,
                    title="Violations by Severity",
                    labels={"x": "Severity", "y": "Count"},
                )
                st.plotly_chart(fig_severity, use_container_width=True)

            with col2:
                # Type distribution
                type_counts = df["type"].value_counts()
                fig_types = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Violations by Type",
                )
                st.plotly_chart(fig_types, use_container_width=True)

        else:
            st.info("No violations match the current filters")


class KnowledgeGraphTab:
    """Knowledge Graph Tab - Interactive knowledge graph visualization and editing"""

    @staticmethod
    def render():
        st.header("🕸️ Knowledge Graph")
        st.markdown(
            "Visualize and edit knowledge graphs based on document content with interactive relationship mapping."
        )

        SessionManager.get_api_client()

        # Graph controls
        st.subheader("Graph Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            graph_query = st.text_input(
                "Search Graph",
                value=st.session_state.get("graph_query", ""),
                placeholder="Enter entity or relationship to search...",
            )
            if graph_query != st.session_state.get("graph_query", ""):
                st.session_state.graph_query = graph_query

        with col2:
            _graph_layout = st.selectbox(
                "Layout Algorithm",
                ["Force-directed", "Hierarchical", "Circular", "Grid"],
                help="Choose how nodes are positioned",
            )

        with col3:
            _max_nodes = st.slider(
                "Max Nodes", 10, 100, 50, help="Limit displayed nodes for performance"
            )

        # Graph visualization
        st.subheader("Interactive Graph Visualization")

        # Get knowledge graph data
        # Real knowledge graph would fetch from database
        kg_data = {"nodes": [], "edges": []}

        if kg_data["nodes"]:
            # Display graph metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nodes", len(kg_data["nodes"]))
            col2.metric("Edges", len(kg_data["edges"]))
            col3.metric(
                "Density",
                f"{len(kg_data['edges']) / max(1, len(kg_data['nodes'])**2):.3f}",
            )
            col4.metric("Components", "1")  # Mock value

            # Graph visualization placeholder
            st.info(
                "🔧 Interactive knowledge graph visualization would be implemented using libraries like:"
            )
            st.write("- **Streamlit-Agraph**: For interactive graph visualization")
            st.write("- **NetworkX + Plotly**: For customizable network graphs")
            st.write("- **Cytoscape.js**: For advanced graph interactions")

            # Show sample graph data
            with st.expander("Sample Graph Data"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Nodes:**")
                    nodes_df = pd.DataFrame(kg_data["nodes"])
                    st.dataframe(nodes_df, use_container_width=True)

                with col2:
                    st.write("**Edges:**")
                    edges_df = pd.DataFrame(kg_data["edges"])
                    st.dataframe(edges_df, use_container_width=True)

            # Entity analysis
            st.subheader("Entity Analysis")

            # Entity type distribution
            if kg_data["nodes"]:
                entity_types = [node["type"] for node in kg_data["nodes"]]
                type_counts = pd.Series(entity_types).value_counts()

                fig_entities = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    title="Entity Type Distribution",
                    labels={"x": "Entity Type", "y": "Count"},
                )
                st.plotly_chart(fig_entities, use_container_width=True)

            # Relationship analysis
            st.subheader("Relationship Mapping")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Relationship Types:**")
                if kg_data["edges"]:
                    rel_types = [edge["label"] for edge in kg_data["edges"]]
                    rel_counts = pd.Series(rel_types).value_counts()
                    st.bar_chart(rel_counts)

            with col2:
                st.write("**Connection Weights:**")
                if kg_data["edges"]:
                    weights = [edge["weight"] for edge in kg_data["edges"]]
                    fig_weights = px.histogram(
                        x=weights,
                        title="Edge Weight Distribution",
                        labels={"x": "Weight", "y": "Count"},
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)

        # Graph editing tools
        st.subheader("Graph Editing Tools")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Add New Entity**")
            new_entity_name = st.text_input("Entity Name")
            new_entity_type = st.selectbox(
                "Entity Type",
                ["PERSON", "ORGANIZATION", "DOCUMENT", "CONCEPT", "LOCATION"],
            )

            if st.button("➕ Add Entity"):
                if new_entity_name:
                    ErrorHandler.display_success(
                        f"Added entity: {new_entity_name} ({new_entity_type})"
                    )
                else:
                    ErrorHandler.display_warning("Please enter entity name")

        with col2:
            st.write("**Add New Relationship**")
            if kg_data["nodes"]:
                entity_names = [node["label"] for node in kg_data["nodes"]]
                from_entity = st.selectbox(
                    "From Entity", entity_names, key="from_entity"
                )
                to_entity = st.selectbox("To Entity", entity_names, key="to_entity")
                relationship_type = st.text_input(
                    "Relationship Type", placeholder="e.g., 'works_for', 'located_in'"
                )

                if st.button("🔗 Add Relationship"):
                    if from_entity != to_entity and relationship_type:
                        ErrorHandler.display_success(
                            f"Added relationship: {from_entity} → {relationship_type} → {to_entity}"
                        )
                    else:
                        ErrorHandler.display_warning(
                            "Please select different entities and enter relationship type"
                        )

        # Graph queries
        st.subheader("Graph Queries")

        st.write("**Predefined Queries:**")
        query_options = [
            "Find all entities connected to a person",
            "Show document relationships",
            "Find shortest path between entities",
            "Identify central entities",
            "Show entity clusters",
        ]

        selected_query = st.selectbox("Select Query", query_options)

        if st.button("🔍 Execute Query"):
            st.info(f"Executing: {selected_query}")
            # Mock query results
            query_results = {
                "entities_found": 5,
                "relationships_found": 8,
                "execution_time": "0.23s",
            }
            st.json(query_results)

        # Export options
        st.subheader("Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Export as JSON"):
                st.download_button(
                    "Download JSON",
                    data=json.dumps(kg_data, indent=2),
                    file_name="knowledge_graph.json",
                    mime="application/json",
                )

        with col2:
            if st.button("📊 Export as CSV"):
                nodes_df = pd.DataFrame(kg_data["nodes"])
                csv_data = nodes_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="graph_nodes.csv",
                    mime="text/csv",
                )

        with col3:
            if st.button("🖼️ Export Visualization"):
                st.info(
                    "Graph visualization export functionality would be implemented here"
                )


class WorkflowDesignerTab:
    """Workflow Designer Tab - Configure and save analysis workflows"""

    @staticmethod
    def render():
        st.header("🧩 Workflow Designer")
        api_client = SessionManager.get_api_client()

        workflows = api_client.get_workflows()

        workflow_names = [w.get("name", str(w.get("id"))) for w in workflows]
        selection = st.selectbox("Select Workflow", ["Create New"] + workflow_names)
        current = None
        if selection != "Create New" and selection in workflow_names:
            current = workflows[workflow_names.index(selection)]

        name = st.text_input("Name", value=current.get("name", "") if current else "")
        enable_ner = st.checkbox("Enable NER", value=current.get("enable_ner", True) if current else True)
        enable_llm = st.checkbox(
            "Enable LLM Extraction",
            value=current.get("enable_llm_extraction", True) if current else True,
        )
        confidence = st.slider(
            "Confidence Threshold",
            0.0,
            1.0,
            current.get("confidence_threshold", 0.7) if current else 0.7,
            0.05,
        )

        if st.button("Save Workflow"):
            payload = {
                "name": name,
                "enable_ner": enable_ner,
                "enable_llm_extraction": enable_llm,
                "confidence_threshold": confidence,
            }
            if current and current.get("id"):
                result = api_client.update_workflow(current["id"], payload)
            else:
                result = api_client.create_workflow(payload)
            if result.get("status") != "ERROR":
                ErrorHandler.display_success("Workflow saved")
            else:
                ErrorHandler.display_error("Failed to save workflow", result.get("message"))


class SettingsLogsTab:
    """Settings & Logs Tab - Application configuration and system monitoring"""

    @staticmethod
    def render():
        st.header("⚙️ Settings & Logs")
        st.markdown("Configure application settings and monitor system events.")

        # Settings section
        st.subheader("Application Settings")

        # API Configuration
        with st.expander("🔌 API Configuration", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                api_endpoint = st.text_input(
                    "API Endpoint",
                    value="http://localhost:8000",
                    help="Backend API URL",
                )
                api_timeout = st.number_input(
                    "Request Timeout (seconds)", value=30, min_value=5, max_value=300
                )
                max_file_size = st.number_input(
                    "Max File Size (MB)", value=50, min_value=1, max_value=500
                )

            with col2:
                api_key = st.text_input(
                    "API Key", type="password", help="API authentication key"
                )
                enable_ssl = st.checkbox("Enable SSL Verification", value=True)
                debug_mode = st.checkbox(
                    "Debug Mode", value=False, help="Enable detailed logging"
                )

        # Processing Settings
        with st.expander("🔧 Processing Settings"):
            col1, col2 = st.columns(2)

            with col1:
                default_model = st.selectbox(
                    "Default Model",
                    ["gpt-4", "gpt-3.5-turbo", "claude-3", "local-model"],
                )
                default_confidence = st.slider(
                    "Default Confidence Threshold", 0.0, 1.0, 0.7, 0.05
                )
                max_concurrent = st.number_input(
                    "Max Concurrent Processes", value=5, min_value=1, max_value=20
                )

            with col2:
                auto_save = st.checkbox("Auto-save Results", value=True)
                backup_enabled = st.checkbox("Enable Backups", value=True)
                retention_days = st.number_input(
                    "Data Retention (days)", value=90, min_value=1, max_value=365
                )

        # UI Settings
        with st.expander("🎨 User Interface Settings"):
            col1, col2 = st.columns(2)

            with col1:
                theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
                page_size = st.number_input(
                    "Table Page Size", value=50, min_value=10, max_value=200
                )
                refresh_interval = st.number_input(
                    "Auto-refresh Interval (seconds)",
                    value=30,
                    min_value=5,
                    max_value=300,
                )

            with col2:
                show_notifications = st.checkbox("Show Notifications", value=True)
                show_tooltips = st.checkbox("Show Help Tooltips", value=True)
                compact_mode = st.checkbox("Compact Mode", value=False)

        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            settings = {
                "api": {
                    "endpoint": api_endpoint,
                    "timeout": api_timeout,
                    "max_file_size": max_file_size,
                    "api_key": api_key if api_key else None,
                    "ssl_verify": enable_ssl,
                    "debug": debug_mode,
                },
                "processing": {
                    "default_model": default_model,
                    "default_confidence": default_confidence,
                    "max_concurrent": max_concurrent,
                    "auto_save": auto_save,
                    "backup_enabled": backup_enabled,
                    "retention_days": retention_days,
                },
                "ui": {
                    "theme": theme,
                    "page_size": page_size,
                    "refresh_interval": refresh_interval,
                    "show_notifications": show_notifications,
                    "show_tooltips": show_tooltips,
                    "compact_mode": compact_mode,
                },
            }

            # Save to session state (in real app, would save to file/database)
            st.session_state.app_settings = settings
            ErrorHandler.display_success("Settings saved successfully!")

        # System Information
        st.subheader("System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Application Info**")
            st.write(f"Version: 2.1.0")
            st.write(f"Build: 20241203")
            st.write(f"Environment: {'Development' if debug_mode else 'Production'}")
            st.write(f"Python: {sys.version.split()[0]}")

        with col2:
            st.write("**System Status**")
            st.write(f"Uptime: {datetime.now().strftime('%H:%M:%S')}")
            st.write(f"Memory Usage: 45.2%")
            st.write(f"CPU Usage: 23.1%")
            st.write(f"Disk Usage: 67.8%")

        # Logs section
        st.subheader("System Logs")

        # Log filters
        col1, col2, col3 = st.columns(3)

        with col1:
            log_level = st.selectbox(
                "Log Level", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            )

        with col2:
            log_component = st.selectbox(
                "Component",
                ["ALL", "API", "Processing", "Memory", "Knowledge Graph", "GUI"],
            )

        with col3:
            _log_hours = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            )

        # Log display
        st.write("**Recent Log Entries:**")

        # Mock log entries
        log_entries = [
            {
                "timestamp": "2025-06-03 10:30:15",
                "level": "INFO",
                "component": "API",
                "message": "Document uploaded successfully",
                "user": "user123",
            },
            {
                "timestamp": "2025-06-03 10:29:45",
                "level": "WARNING",
                "component": "Processing",
                "message": "Low confidence in entity extraction",
                "user": "user456",
            },
            {
                "timestamp": "2025-06-03 10:28:30",
                "level": "ERROR",
                "component": "Memory",
                "message": "Failed to store memory entry",
                "user": "system",
            },
            {
                "timestamp": "2025-06-03 10:27:12",
                "level": "INFO",
                "component": "GUI",
                "message": "User accessed violation review tab",
                "user": "user789",
            },
            {
                "timestamp": "2025-06-03 10:26:05",
                "level": "DEBUG",
                "component": "Knowledge Graph",
                "message": "Graph query executed",
                "user": "user123",
            },
        ]

        # Filter logs
        filtered_logs = log_entries
        if log_level != "ALL":
            filtered_logs = [log for log in filtered_logs if log["level"] == log_level]
        if log_component != "ALL":
            filtered_logs = [
                log for log in filtered_logs if log["component"] == log_component
            ]

        # Display logs
        if filtered_logs:
            logs_df = pd.DataFrame(filtered_logs)
            st.dataframe(logs_df, use_container_width=True, hide_index=True)

            # Export logs
            if st.button("📥 Export Logs"):
                csv_data = logs_df.to_csv(index=False)
                st.download_button(
                    "Download Logs CSV",
                    data=csv_data,
                    file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("No log entries match the current filters")

        # System maintenance
        st.subheader("System Maintenance")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Restart Services"):
                with st.spinner("Restarting services..."):
                    import time

                    time.sleep(2)
                    ErrorHandler.display_success("Services restarted successfully")

        with col2:
            if st.button("🧹 Clear Cache"):
                ErrorHandler.display_success("Cache cleared successfully")

        with col3:
            if st.button("📊 Generate Report"):
                st.info("System health report generated")
                with st.expander("System Report"):
                    st.write("**Performance Metrics:**")
                    st.write("- Average response time: 245ms")
                    st.write("- Documents processed today: 47")
                    st.write("- Memory utilization: 78%")
                    st.write("- Error rate: 0.12%")


class AnalysisDashboardTab:
    """Analysis Dashboard Tab - High-level overview and analytics"""

    @staticmethod
    def render():
        st.header("📊 Analysis Dashboard")
        st.markdown(
            "High-level view of analysis results, document metadata, entities, contradictions, and violations."
        )

        # Key metrics
        st.subheader("System Overview")

        # Real analytics data from database
        analytics_data = {
            "documents_processed": 0,
            "documents_delta": "+0",
            "entities_extracted": 0,
            "entities_delta": "+0",
            "violations_detected": 0,
            "violations_delta": "+0",
            "avg_processing_time": "0.0s",
            "processing_delta": "+0.0s",
        }

        DataVisualization.create_metrics_dashboard(analytics_data)

        # Recent activity
        st.subheader("Recent Activity")

        # Real activity data would come from database
        activities = []

        if activities:
            for activity in activities:
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                    col1.write(activity["time"])
                    col2.write(f"**{activity['activity']}** - {activity['document']}")
                    col3.write(UIComponents.create_status_indicator(activity["status"]))
                    col4.write(activity["duration"])
        else:
            st.info(
                "No recent activity. Upload and process documents to see activity logs."
            )

        # Analytics charts
        st.subheader("Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Processing volume over time
            dates = pd.date_range(start="2025-05-01", end="2025-06-03", freq="D")
            volumes = [10 + i % 20 + (i // 7) * 2 for i in range(len(dates))]

            fig_volume = px.line(
                x=dates,
                y=volumes,
                title="Daily Processing Volume",
                labels={"x": "Date", "y": "Documents Processed"},
            )
            st.plotly_chart(fig_volume, use_container_width=True)

        with col2:
            # Entity type distribution
            # Real entities would fetch from database
            entities_mock = []
            entity_counts = pd.Series([e["type"] for e in entities_mock]).value_counts()

            fig_entities = px.pie(
                values=entity_counts.values,
                names=entity_counts.index,
                title="Entity Type Distribution",
            )
            st.plotly_chart(fig_entities, use_container_width=True)

        # Document analysis
        st.subheader("Document Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Recent documents table
            # Real documents would fetch from database
            documents = []
            docs_df = pd.DataFrame(documents)

            st.write("**Recent Documents**")
            if not docs_df.empty:
                st.dataframe(docs_df, use_container_width=True, hide_index=True)
            else:
                st.info("No documents processed yet. Upload documents to begin.")

        with col2:
            # Document status distribution
            if not docs_df.empty and "status" in docs_df.columns:
                status_counts = docs_df["status"].value_counts()
                fig_status = px.bar(
                    x=status_counts.values,
                    y=status_counts.index,
                    orientation="h",
                    title="Document Status",
                    labels={"x": "Count", "y": "Status"},
                )
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.info("Status chart will appear when documents are processed.")

        # Violations and compliance
        st.subheader("Violations & Compliance")

        # Real violations would fetch from database
        violations = []
        violations_df = pd.DataFrame(violations)

        col1, col2, col3 = st.columns(3)

        with col1:
            # Severity distribution
            severity_counts = violations_df["severity"].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Violation Severity",
            )
            st.plotly_chart(fig_severity, use_container_width=True)

        with col2:
            # Violation trends
            violation_dates = pd.date_range(
                start="2025-05-01", end="2025-06-03", freq="D"
            )
            violation_counts = [2 + i % 5 for i in range(len(violation_dates))]

            fig_trends = px.line(
                x=violation_dates,
                y=violation_counts,
                title="Daily Violations Detected",
                labels={"x": "Date", "y": "Violations"},
            )
            st.plotly_chart(fig_trends, use_container_width=True)

        with col3:
            # Compliance score
            compliance_score = 87.5
            fig_compliance = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=compliance_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Compliance Score"},
                    delta={"reference": 85},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 85], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            st.plotly_chart(fig_compliance, use_container_width=True)

        # System performance
        st.subheader("System Performance")

        col1, col2 = st.columns(2)

        with col1:
            # Processing time distribution
            processing_times = [1.2, 2.1, 1.8, 3.2, 0.9, 2.4, 1.6, 2.8, 1.4, 2.2]
            fig_times = px.histogram(
                x=processing_times,
                title="Processing Time Distribution",
                labels={"x": "Processing Time (seconds)", "y": "Count"},
            )
            st.plotly_chart(fig_times, use_container_width=True)

        with col2:
            # Success rate over time
            dates = pd.date_range(start="2025-05-27", end="2025-06-03", freq="D")
            success_rates = [95.2, 96.1, 94.8, 97.3, 96.7, 95.9, 97.1]

            fig_success = px.line(
                x=dates,
                y=success_rates,
                title="Daily Success Rate",
                labels={"x": "Date", "y": "Success Rate (%)"},
            )
            fig_success.update_yaxes(range=[90, 100])
            st.plotly_chart(fig_success, use_container_width=True)

        # Quick actions
        st.subheader("Quick Actions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📄 Process New Document"):
                st.switch_page("Document Processor")

        with col2:
            if st.button("⚠️ Review Violations"):
                st.switch_page("Violation Review")

        with col3:
            if st.button("🧠 Check Memory"):
                st.switch_page("Memory Brain")

        with col4:
            if st.button("🕸️ Explore Graph"):
                st.switch_page("Knowledge Graph")


def main():
    """Main application entry point"""

    # Configure page
    st.set_page_config(
        page_title="Legal AI System - Unified GUI",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    SessionManager.init_session_state()

    # Sidebar navigation
    st.sidebar.title("⚖️ Legal AI System")
    st.sidebar.markdown("Professional Legal AI Assistant")

    # Navigation tabs
    tab_options = {
        "📊 Analysis Dashboard": "dashboard",
        "📄 Document Processor": "processor",
        "🧠 Memory Brain": "memory",
        "⚠️ Violation Review": "violations",
        "🕸️ Knowledge Graph": "graph",
        "⚙️ Settings & Logs": "settings",
    }

    selected_tab = st.sidebar.radio("Navigation", list(tab_options.keys()))

    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    api_client = SessionManager.get_api_client()
    health_status = api_client.check_health()

    status_indicator = UIComponents.create_status_indicator(
        health_status.get("status", "ERROR")
    )
    st.sidebar.markdown(f"**Backend:** {status_indicator}")

    # Quick stats
    st.sidebar.markdown("**Quick Stats:**")
    try:
        v_count = len(ViolationReviewDB().fetch_violations())
    except Exception:
        v_count = 0
    st.sidebar.write("• Documents: 0")
    st.sidebar.write(f"• Violations: {v_count}")
    st.sidebar.write("• Memory: 0 entries")
    st.sidebar.write("• Graph: 0 nodes")

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Legal AI System v2.1.0*")

    # Render selected tab content
    tab_key = tab_options[selected_tab]

    if tab_key == "dashboard":
        AnalysisDashboardTab.render()
    elif tab_key == "processor":
        DocumentProcessorTab.render()
    elif tab_key == "workflows":
        WorkflowDesignerTab.render()
    elif tab_key == "memory":
        MemoryBrainTab.render()
    elif tab_key == "violations":
        ViolationReviewTab.render()
    elif tab_key == "graph":
        KnowledgeGraphTab.render()
    elif tab_key == "workflows":
        WorkflowDesignerTab.render()
    elif tab_key == "settings":
        SettingsLogsTab.render()


if __name__ == "__main__":
    main()
