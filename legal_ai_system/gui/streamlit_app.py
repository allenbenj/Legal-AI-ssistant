import sys
import os

# Add the parent directory to sys.path to resolve imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import streamlit as st
except ImportError:
    print("Error: streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed. Install with: pip install requests")
    sys.exit(1)

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None

try:
    from legal_ai_system.config.settings import settings
except ImportError:
    class MockSettings:
        api_base_url = "http://localhost:8000"
    settings = MockSettings()

# MUST be first Streamlit command - configure page
st.set_page_config(
    page_title="Legal AI System",
    page_icon="⚖️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for red/black/grey theme
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .stSidebar {
        background-color: #2d2d2d;
    }
    
    .stButton > button {
        background-color: #dc143c;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #b01030;
        color: white;
    }
    
    .stSelectbox > div > div {
        background-color: #3d3d3d;
        color: white;
    }
    
    .stTextInput > div > div > input {
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #dc143c;
    }
    
    .stMetric {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc143c;
    }
    
    .stSuccess {
        background-color: #2d4a2d;
        border: 1px solid #4caf50;
    }
    
    .stError {
        background-color: #4a2d2d;
        border: 1px solid #dc143c;
    }
    
    .stWarning {
        background-color: #4a4a2d;
        border: 1px solid #ff9800;
    }
    
    .stInfo {
        background-color: #2d3a4a;
        border: 1px solid #2196f3;
    }
    
    h1, h2, h3 {
        color: #dc143c;
        font-weight: bold;
    }
    
    .stDataFrame {
        background-color: #2d2d2d;
    }
    
    .stExpander {
        background-color: #2d2d2d;
        border: 1px solid #555;
    }
</style>
""", unsafe_allow_html=True)

# Fix for Streamlit execution - use absolute imports
if sentry_sdk:
    sentry_sdk.init(
        dsn="https://2fbad862414aad747dba577c60110470@o4509439121489920.ingest.us.sentry.io/4509439123587072",
        send_default_pii=True,
    )

try:
    from legal_ai_system.core.constants import Constants
except ImportError:  # pragma: no cover - fallback for standalone execution
    class FallbackConstants:
        class Version:
            APP_VERSION = "2.1.0"
    Constants = FallbackConstants

import logging  # Using standard logging for this standalone part initially
from pathlib import Path
import time  # For simulate processing
import asyncio
from typing import Optional
import datetime

try:
    from legal_ai_system.services.realtime_analysis_workflow import RealTimeAnalysisWorkflow
except Exception:
    RealTimeAnalysisWorkflow = None  # type: ignore

workflow_for_gui: Optional[RealTimeAnalysisWorkflow] = None

# Using standard logging initially, can be augmented by detailed_logging if main system is run first
streamlit_logger = logging.getLogger("StreamlitAppGUI")
if not streamlit_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    streamlit_logger.addHandler(handler)
    streamlit_logger.setLevel(logging.INFO)


def setup_main_app_logging_gui() -> None:  # Renamed to avoid conflict if imported elsewhere
    """Configure basic logging for this Streamlit app entry point if not already done."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    log_dir / "streamlit_gui.log", encoding='utf-8')
            ]
        )
        streamlit_logger.info(
            "Basic logging configured by streamlit_app.py for GUI.")
    else:
        streamlit_logger.info(
            "Logging seems to be already configured. Streamlit GUI using existing setup.")


def check_gui_dependencies() -> bool:  # Renamed
    """Check if required GUI and core dependencies are available."""
    streamlit_logger.info("Checking core dependencies for Streamlit GUI.")
    required_packages = [
        'streamlit', 'requests', 'pandas', 'numpy',
    ]

    missing = []
    for package_name in required_packages:
        try:
            __import__(package_name)
            streamlit_logger.debug(f"Dependency check: {package_name} - OK.")
        except ImportError:
            missing.append(package_name)
            streamlit_logger.warning(
                f"Dependency check: {package_name} - MISSING.")

    if missing:
        streamlit_logger.error(
            f"Missing required packages for GUI: {', '.join(missing)}")
        print("to console as Streamlit might not be fully up yet")
        print(f"❌ Missing required GUI packages: {', '.join(missing)}")
        return False  # Do not auto-install in this environment for safety

    streamlit_logger.info("All checked GUI dependencies are available.")
    return True


def run_streamlit_app_content():
    """Defines the actual content and logic of the Streamlit application."""

    st.title("🏛️ Legal AI System Dashboard")
    st.caption("Professional Edition - Document Analysis & Knowledge Management")

    st.sidebar.header("🔥 LEGAL AI COMMAND CENTER")
    
    # Check if backend API is reachable
    try:
        r = requests.get(f"{settings.api_base_url}/api/v1/system/health", timeout=5)
        if r.status_code == 200:
            health_data = r.json()
            overall_status = health_data.get("overall_status", "UNKNOWN")
            
            if overall_status == "HEALTHY":
                st.sidebar.success("🟢 SYSTEM ONLINE")
            elif overall_status == "BUSY":
                st.sidebar.warning("🟡 SYSTEM BUSY")
            elif overall_status == "DEGRADED":
                st.sidebar.error("🟠 SYSTEM DEGRADED")
            else:
                st.sidebar.error("🔴 SYSTEM ERROR")
                
            # Show key metrics
            metrics = health_data.get("performance_metrics_summary", {})
            if metrics:
                st.sidebar.markdown(f"""
                **📊 LIVE METRICS**
                - Documents: {metrics.get('total_documents', 0)}
                - Success Rate: {metrics.get('success_rate', 0):.0f}%
                - Processing: {metrics.get('processing_documents', 0)}
                """)
        else:
            st.sidebar.error("🔴 API CONNECTION FAILED")
    except Exception as e:
        st.sidebar.error(f"🔴 SYSTEM OFFLINE\n{str(e)[:50]}...")
        streamlit_logger.warning(f"Health check failed: {e}")

    page = st.sidebar.radio(
        "Go to", [
            "Dashboard",
            "Document Upload",
            "Document Results",
            "Review Queue",
            "Workflow Designer",
            "Process Monitoring", 
            "System Status",
            "Settings",
        ]
    )

    if page == "Dashboard":
        st.header("System Overview")
        st.write(
            "Welcome to the Legal AI System. This dashboard provides an overview of system activities and performance.")
        
        # Get real document statistics
        try:
            response = requests.get(f"{settings.api_base_url}/api/v1/documents", timeout=5)
            if response.status_code == 200:
                documents = response.json()
                total_docs = len(documents)
                completed_docs = len([d for d in documents if d.get("status") == "completed"])
                processing_docs = len([d for d in documents if d.get("status") == "processing"])
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Documents Processed", value=str(completed_docs), delta=f"+{total_docs} total")
                col2.metric(label="Currently Processing", value=str(processing_docs))
                col3.metric(label="Success Rate", value=f"{(completed_docs/total_docs*100):.0f}%" if total_docs > 0 else "0%")

                st.subheader("Recent Activity")
                if documents:
                    # Show recent documents
                    for doc in documents[-5:]:  # Last 5 documents
                        status_icon = "✅" if doc.get("status") == "completed" else "⏳" if doc.get("status") == "processing" else "❌"
                        progress = doc.get("progress", 0) * 100
                        st.write(f"{status_icon} **{doc.get('filename', 'Unknown')}** - {doc.get('status', 'unknown')} ({progress:.0f}%)")
                else:
                    st.info("No documents processed yet. Upload documents to begin processing.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Documents Processed", value="--", delta="API Error")
                col2.metric(label="Active Workflows", value="--")
                col3.metric(label="Pending Reviews", value="--")
                st.error("Could not connect to backend API")
        except Exception as e:
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Documents Processed", value="--", delta="Connection Error")
            col2.metric(label="Active Workflows", value="--")
            col3.metric(label="Pending Reviews", value="--")
            st.error(f"Error: {e}")

    elif page == "Document Upload":
        st.header("📄 Document Upload & Processing")
        
        # Upload method selection
        upload_method = st.radio(
            "Choose upload method:",
            ["Single Document", "Multiple Files", "Zip Archive"],
            horizontal=True
        )
        
        if upload_method == "Single Document":
            uploaded_file = st.file_uploader("Choose a document to analyze", type=[
                'pdf', 'docx', 'txt', 'md'])

            if uploaded_file is not None:
                st.write(f"📄 Uploaded: {uploaded_file.name} ({uploaded_file.type})")

                with st.expander("Processing Options"):
                    st.checkbox("Enable NER", value=True, key="opt_ner")
                    st.checkbox("Enable LLM Extraction", value=True, key="opt_llm_extract")
                    st.checkbox("Enable Confidence Calibration", value=True, key="opt_conf_calib")
                    st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05, key="opt_conf_thresh")

                if st.button("🚀 Process Document", type="primary"):
                    with st.spinner("Sending document to backend for processing..."):
                        files = {
                            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        try:
                            upload_response = requests.post(
                                f"{settings.api_base_url}/api/v1/documents/upload",
                                files=files,
                                timeout=30,
                            )
                            upload_response.raise_for_status()
                            upload_data = upload_response.json()
                            document_id = upload_data.get("document_id")
                            st.success(f"✅ Document uploaded with ID: {document_id}")

                            proc_options = {
                                "enable_ner": st.session_state.opt_ner,
                                "enable_llm_extraction": st.session_state.opt_llm_extract,
                                "enable_confidence_calibration": st.session_state.opt_conf_calib,
                                "confidence_threshold": st.session_state.opt_conf_thresh,
                            }
                            process_response = requests.post(
                                f"{settings.api_base_url}/api/v1/documents/{document_id}/process",
                                json=proc_options,
                                timeout=30,
                            )
                            process_response.raise_for_status()
                            st.success(f"🚀 Processing started for document ID: {document_id}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ API Error: {e}")
                        except Exception as e:
                            st.error(f"❌ An error occurred: {e}")

        elif upload_method == "Multiple Files":
            st.subheader("📁 Multiple File Upload")
            st.info("💡 **How to select multiple files:** Click 'Browse files', then hold Ctrl (Windows/Linux) or Cmd (Mac) while clicking each file to select multiple files at once.")
            uploaded_files = st.file_uploader(
                "Select multiple documents (hold Ctrl/Cmd to select multiple)",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.write(f"📁 Selected {len(uploaded_files)} files:")
                for file in uploaded_files:
                    st.write(f"  • {file.name} ({file.type})")
                
                with st.expander("Batch Processing Options"):
                    st.checkbox("Enable NER", value=True, key="batch_opt_ner")
                    st.checkbox("Enable LLM Extraction", value=True, key="batch_opt_llm_extract")
                    st.checkbox("Enable Confidence Calibration", value=True, key="batch_opt_conf_calib")
                    st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05, key="batch_opt_conf_thresh")
                    process_sequentially = st.checkbox("Process sequentially (recommended for large batches)", value=True)
                
                if st.button(f"🚀 Process All {len(uploaded_files)} Documents", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    proc_options = {
                        "enable_ner": st.session_state.batch_opt_ner,
                        "enable_llm_extraction": st.session_state.batch_opt_llm_extract,
                        "enable_confidence_calibration": st.session_state.batch_opt_conf_calib,
                        "confidence_threshold": st.session_state.batch_opt_conf_thresh,
                    }
                    
                    uploaded_docs = []
                    failed_uploads = []
                    
                    # Upload all files first
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"Uploading {file.name}...")
                        try:
                            files = {'file': (file.name, file.getvalue(), file.type)}
                            upload_response = requests.post(
                                f"{settings.api_base_url}/api/v1/documents/upload",
                                files=files,
                                timeout=30,
                            )
                            upload_response.raise_for_status()
                            upload_data = upload_response.json()
                            uploaded_docs.append({
                                "id": upload_data.get("document_id"),
                                "name": file.name,
                                "options": proc_options
                            })
                        except Exception as e:
                            failed_uploads.append(f"{file.name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
                    
                    # Process all uploaded documents
                    processed_docs = []
                    failed_processing = []
                    
                    for i, doc in enumerate(uploaded_docs):
                        status_text.text(f"Processing {doc['name']}...")
                        try:
                            process_response = requests.post(
                                f"{settings.api_base_url}/api/v1/documents/{doc['id']}/process",
                                json=doc['options'],
                                timeout=30,
                            )
                            process_response.raise_for_status()
                            processed_docs.append(doc['name'])
                            
                            if not process_sequentially:
                                # For parallel processing, just continue
                                pass
                            else:
                                # For sequential processing, wait a bit
                                import time
                                time.sleep(1)
                                
                        except Exception as e:
                            failed_processing.append(f"{doc['name']}: {str(e)}")
                        
                        progress_bar.progress((len(uploaded_files) + i + 1) / (len(uploaded_files) * 2))
                    
                    # Show results
                    status_text.text("✅ Batch processing completed!")
                    
                    with results_container:
                        if processed_docs:
                            st.success(f"✅ Successfully processed {len(processed_docs)} documents:")
                            for doc_name in processed_docs:
                                st.write(f"  • {doc_name}")
                        
                        if failed_uploads:
                            st.error(f"❌ Failed to upload {len(failed_uploads)} files:")
                            for error in failed_uploads:
                                st.write(f"  • {error}")
                        
                        if failed_processing:
                            st.error(f"❌ Failed to process {len(failed_processing)} files:")
                            for error in failed_processing:
                                st.write(f"  • {error}")
                        
                        st.info("💡 Check the 'Document Results' page to view detailed analysis results.")

        elif upload_method == "Zip Archive":
            st.subheader("📦 Zip Archive Upload")
            uploaded_zip = st.file_uploader("Upload a ZIP file containing documents", type=['zip'])
            
            if uploaded_zip is not None:
                st.write(f"📦 Uploaded: {uploaded_zip.name}")
                
                try:
                    import zipfile
                    import io
                    
                    # Extract files from zip
                    zip_file = zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()))
                    file_list = zip_file.namelist()
                    
                    # Filter for supported file types
                    supported_files = [f for f in file_list if f.lower().endswith(('.pdf', '.docx', '.txt', '.md'))]
                    
                    st.write(f"📁 Found {len(supported_files)} supported documents in archive:")
                    for file_name in supported_files[:10]:  # Show first 10
                        st.write(f"  • {file_name}")
                    if len(supported_files) > 10:
                        st.write(f"  ... and {len(supported_files) - 10} more files")
                    
                    if supported_files:
                        with st.expander("Zip Processing Options"):
                            st.checkbox("Enable NER", value=True, key="zip_opt_ner")
                            st.checkbox("Enable LLM Extraction", value=True, key="zip_opt_llm_extract")
                            st.checkbox("Enable Confidence Calibration", value=True, key="zip_opt_conf_calib")
                            st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05, key="zip_opt_conf_thresh")
                        
                        if st.button(f"🚀 Process All {len(supported_files)} Documents from ZIP", type="primary"):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            proc_options = {
                                "enable_ner": st.session_state.zip_opt_ner,
                                "enable_llm_extraction": st.session_state.zip_opt_llm_extract,
                                "enable_confidence_calibration": st.session_state.zip_opt_conf_calib,
                                "confidence_threshold": st.session_state.zip_opt_conf_thresh,
                            }
                            
                            processed_count = 0
                            failed_count = 0
                            
                            for i, file_name in enumerate(supported_files):
                                status_text.text(f"Processing {file_name}...")
                                
                                try:
                                    # Extract file content
                                    file_content = zip_file.read(file_name)
                                    
                                    # Upload file
                                    files = {'file': (file_name, file_content, 'application/octet-stream')}
                                    upload_response = requests.post(
                                        f"{settings.api_base_url}/api/v1/documents/upload",
                                        files=files,
                                        timeout=30,
                                    )
                                    upload_response.raise_for_status()
                                    upload_data = upload_response.json()
                                    document_id = upload_data.get("document_id")
                                    
                                    # Process document
                                    process_response = requests.post(
                                        f"{settings.api_base_url}/api/v1/documents/{document_id}/process",
                                        json=proc_options,
                                        timeout=30,
                                    )
                                    process_response.raise_for_status()
                                    processed_count += 1
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Failed to process {file_name}: {str(e)}")
                                    failed_count += 1
                                
                                progress_bar.progress((i + 1) / len(supported_files))
                            
                            status_text.text("✅ ZIP processing completed!")
                            st.success(f"✅ Successfully processed {processed_count} documents")
                            if failed_count > 0:
                                st.warning(f"⚠️ Failed to process {failed_count} documents")
                            st.info("💡 Check the 'Document Results' page to view detailed analysis results.")
                    else:
                        st.warning("⚠️ No supported document types found in ZIP archive")
                        
                except Exception as e:
                    st.error(f"❌ Error processing ZIP file: {e}")
            else:
                st.info("💡 Upload a ZIP file containing PDF, DOCX, TXT, or MD files for batch processing")

    elif page == "Document Results":
        st.header("📊 Document Processing Results")
        st.write("View and analyze processed documents")
        
        # Add demo data button for testing
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🎭 Create Demo Data", help="Generate sample document analysis for testing"):
                try:
                    response = requests.post(f"{settings.api_base_url}/api/v1/documents/demo", timeout=10)
                    if response.status_code == 200:
                        st.success("✅ Demo document created! Refresh to see results.")
                        st.rerun()
                    else:
                        st.error("Failed to create demo data")
                except Exception as e:
                    st.error(f"Error creating demo data: {e}")
        
        # Get list of processed documents
        try:
            # First, let's get all document statuses from the backend
            response = requests.get(f"{settings.api_base_url}/api/v1/documents", timeout=10)
            if response.status_code == 200:
                documents = response.json()
                
                if documents:
                    # Create tabs for different views
                    overview_tab, details_tab, raw_data_tab = st.tabs(["📋 Overview", "🔍 Details", "🗃️ Raw Data"])
                    
                    with overview_tab:
                        st.subheader("Document Processing Overview")
                        
                        # Create a table of all documents
                        import pandas as pd
                        df_data = []
                        for doc in documents:
                            df_data.append({
                                "Document ID": doc.get("document_id", "N/A"),
                                "Filename": doc.get("filename", "N/A"),
                                "Status": doc.get("status", "N/A"),
                                "Progress": f"{doc.get('progress', 0)*100:.1f}%",
                                "Stage": doc.get("stage", "N/A"),
                                "Completed": "✅" if doc.get("status") == "completed" else "⏳"
                            })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                        
                        # Document selector for detailed view
                        st.subheader("Select Document for Detailed Analysis")
                        completed_docs = [doc for doc in documents if doc.get("status") == "completed"]
                        
                        if completed_docs:
                            doc_options = {f"{doc['filename']} ({doc['document_id']})": doc for doc in completed_docs}
                            selected_doc_name = st.selectbox("Choose a document:", list(doc_options.keys()))
                            selected_doc = doc_options[selected_doc_name]
                            
                            if st.button("Load Document Analysis"):
                                st.session_state['selected_doc'] = selected_doc
                        else:
                            st.info("No completed documents found. Upload and process a document first.")
                    
                    with details_tab:
                        if 'selected_doc' in st.session_state:
                            doc = st.session_state['selected_doc']
                            
                            st.subheader(f"Analysis Results: {doc['filename']}")
                            
                            # Get detailed results
                            doc_id = doc['document_id']
                            detail_response = requests.get(
                                f"{settings.api_base_url}/api/v1/documents/{doc_id}/status", 
                                timeout=10
                            )
                            
                            if detail_response.status_code == 200:
                                doc_details = detail_response.json()
                                # Handle both direct result_summary and nested structure
                                if 'result_summary' in doc_details:
                                    results = doc_details['result_summary']
                                else:
                                    # If the response IS the result summary
                                    results = doc_details
                                
                                if results:
                                    # Summary Section
                                    if 'summary' in results:
                                        st.subheader("📝 Document Summary")
                                        st.write(results['summary'])
                                    
                                    # Entities Section
                                    if 'entities' in results and results['entities']:
                                        st.subheader("🏷️ Extracted Entities")
                                        entities_df = pd.DataFrame(results['entities'])
                                        if not entities_df.empty:
                                            # Group by entity type
                                            for entity_type in entities_df['type'].unique():
                                                with st.expander(f"{entity_type} ({len(entities_df[entities_df['type'] == entity_type])} found)"):
                                                    type_entities = entities_df[entities_df['type'] == entity_type]
                                                    for _, entity in type_entities.iterrows():
                                                        st.write(f"• **{entity['entity']}** (confidence: {entity['confidence']:.2f})")
                                    
                                    # Legal Analysis Section
                                    if 'legal_analysis' in results and results['legal_analysis']:
                                        st.subheader("⚖️ Legal Analysis")
                                        analysis = results['legal_analysis']
                                        
                                        if 'analysis' in analysis:
                                            st.write("**Analysis:**")
                                            st.write(analysis['analysis'])
                                        
                                        if 'legal_issues' in analysis and analysis['legal_issues']:
                                            st.write("**Legal Issues:**")
                                            for issue in analysis['legal_issues']:
                                                st.write(f"• {issue}")
                                        
                                        if 'recommendations' in analysis and analysis['recommendations']:
                                            st.write("**Recommendations:**")
                                            for rec in analysis['recommendations']:
                                                st.write(f"• {rec}")
                                    
                                    # Document Stats
                                    st.subheader("📊 Document Statistics")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Text Length", f"{results.get('text_length', 0):,} chars")
                                    with col2:
                                        st.metric("Entities Found", len(results.get('entities', [])))
                                    with col3:
                                        st.metric("Processing Status", "✅ Complete" if results.get('processing_completed') else "❌ Failed")
                                else:
                                    st.warning("No detailed results found for this document.")
                            else:
                                st.error("Failed to load document details.")
                        else:
                            st.info("Select a document from the Overview tab to see detailed analysis.")
                    
                    with raw_data_tab:
                        if 'selected_doc' in st.session_state:
                            doc = st.session_state['selected_doc']
                            st.subheader(f"Raw Data: {doc['filename']}")
                            
                            # Get and display raw processing data
                            doc_id = doc['document_id']
                            detail_response = requests.get(
                                f"{settings.api_base_url}/api/v1/documents/{doc_id}/status", 
                                timeout=10
                            )
                            
                            if detail_response.status_code == 200:
                                doc_details = detail_response.json()
                                st.json(doc_details)
                            else:
                                st.error("Failed to load raw data.")
                        else:
                            st.info("Select a document from the Overview tab to see raw data.")
                
                else:
                    st.info("No documents found. Upload and process a document first.")
            else:
                st.error("Failed to connect to backend to retrieve document list.")
        except Exception as e:
            st.error(f"Error loading documents: {e}")

    elif page == "Review Queue":
        st.header("📥 Pending Review Items")
        st.write("Review and approve/reject AI-extracted entities and legal findings")
        
        # Get pending reviews from the backend
        try:
            response = requests.get(f"{settings.api_base_url}/api/v1/reviews/pending", timeout=10)
            if response.status_code == 200:
                pending_reviews = response.json()
                
                if pending_reviews:
                    st.subheader(f"📋 {len(pending_reviews)} Items Pending Review")
                    
                    # Priority filter
                    priority_filter = st.selectbox(
                        "Filter by Priority:",
                        ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        key="priority_filter"
                    )
                    
                    # Filter reviews by priority if selected
                    if priority_filter != "All":
                        filtered_reviews = [r for r in pending_reviews if r.get("priority") == priority_filter]
                    else:
                        filtered_reviews = pending_reviews
                    
                    if filtered_reviews:
                        for idx, item in enumerate(filtered_reviews):
                            priority = item.get("priority", "MEDIUM")
                            priority_color = {
                                "CRITICAL": "🔴",
                                "HIGH": "🟠", 
                                "MEDIUM": "🟡",
                                "LOW": "🟢"
                            }.get(priority, "⚪")
                            
                            # Use containers instead of nested expanders
                            with st.container():
                                st.markdown(f"### {priority_color} {item.get('entity_type', 'UNKNOWN')} - {item.get('entity_text', 'Unknown')[:50]}...")
                                
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    st.write(f"**Entity:** {item.get('entity_text', 'N/A')}")
                                    st.write(f"**Type:** {item.get('entity_type', 'N/A')}")
                                    st.write(f"**Confidence:** {item.get('confidence', 0):.2%}")
                                    st.write(f"**Context:** {item.get('context', 'N/A')}")
                                    st.write(f"**Document:** {item.get('source_document_id', 'N/A')}")
                                    st.write(f"**Priority:** {priority}")
                                
                                with col2:
                                    item_id = item.get("id", "unknown")
                                    
                                    if st.button(f"✅ Approve", key=f"approve_{item_id}_{idx}"):
                                        try:
                                            decision_response = requests.post(
                                                f"{settings.api_base_url}/api/v1/calibration/review",
                                                json={
                                                    "item_id": item_id,
                                                    "decision": "approved",
                                                    "reviewer_notes": "Approved via GUI"
                                                },
                                                timeout=10
                                            )
                                            if decision_response.status_code == 200:
                                                st.success("✅ Approved!")
                                                st.rerun()
                                            else:
                                                st.error("Failed to submit approval")
                                        except Exception as e:
                                            st.error(f"Error: {e}")
                                    
                                    if st.button(f"❌ Reject", key=f"reject_{item_id}_{idx}"):
                                        try:
                                            decision_response = requests.post(
                                                f"{settings.api_base_url}/api/v1/calibration/review",
                                                json={
                                                    "item_id": item_id,
                                                    "decision": "rejected",
                                                    "reviewer_notes": "Rejected via GUI"
                                                },
                                                timeout=10
                                            )
                                            if decision_response.status_code == 200:
                                                st.success("❌ Rejected!")
                                                st.rerun()
                                            else:
                                                st.error("Failed to submit rejection")
                                        except Exception as e:
                                            st.error(f"Error: {e}")
                                
                                with col3:
                                    # Show modify form in main area, not nested expander
                                    if st.button(f"✏️ Modify", key=f"modify_btn_{item_id}_{idx}"):
                                        st.session_state[f"modify_mode_{item_id}"] = True
                                    
                                    if st.session_state.get(f"modify_mode_{item_id}", False):
                                        st.markdown("**Modify Entity:**")
                                        modified_text = st.text_input(
                                            "Entity Text:",
                                            value=item.get('entity_text', ''),
                                            key=f"modify_text_{item_id}_{idx}"
                                        )
                                        modified_type = st.selectbox(
                                            "Type:",
                                            ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "LEGAL_ISSUE", "OTHER"],
                                            index=0,
                                            key=f"modify_type_{item_id}_{idx}"
                                        )
                                        modify_notes = st.text_area(
                                            "Notes:",
                                            key=f"modify_notes_{item_id}_{idx}",
                                            height=100
                                        )
                                        
                                        col_save, col_cancel = st.columns(2)
                                        with col_save:
                                            if st.button(f"💾 Save", key=f"save_{item_id}_{idx}"):
                                                try:
                                                    decision_response = requests.post(
                                                        f"{settings.api_base_url}/api/v1/calibration/review",
                                                        json={
                                                            "item_id": item_id,
                                                            "decision": "modified",
                                                            "modified_data": {
                                                                "entity_text": modified_text,
                                                                "entity_type": modified_type
                                                            },
                                                            "reviewer_notes": modify_notes
                                                        },
                                                        timeout=10
                                                    )
                                                    if decision_response.status_code == 200:
                                                        st.success("💾 Modified!")
                                                        st.session_state[f"modify_mode_{item_id}"] = False
                                                        st.rerun()
                                                    else:
                                                        st.error("Failed to submit modification")
                                                except Exception as e:
                                                    st.error(f"Error: {e}")
                                        
                                        with col_cancel:
                                            if st.button(f"❌ Cancel", key=f"cancel_{item_id}_{idx}"):
                                                st.session_state[f"modify_mode_{item_id}"] = False
                                                st.rerun()
                                
                                st.divider()
                    else:
                        st.info(f"No {priority_filter.lower()} priority items found.")
                else:
                    st.info("🎉 No items pending review! All extractions have been processed.")
                    
                    # Show some statistics
                    try:
                        stats_response = requests.get(f"{settings.api_base_url}/api/v1/reviews/stats", timeout=5)
                        if stats_response.status_code == 200:
                            stats = stats_response.json()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Reviewed", stats.get("total_reviewed", 0))
                            with col2:
                                st.metric("Auto-Approved", stats.get("auto_approved", 0))
                            with col3:
                                st.metric("Human Reviewed", stats.get("human_reviewed", 0))
                    except:
                        pass
            else:
                st.error(f"Failed to fetch reviews (HTTP {response.status_code})")
                
        except requests.exceptions.ConnectionError:
            st.error("🔴 Cannot connect to backend API. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Error loading reviews: {e}")
            
        # Manual refresh button
        if st.button("🔄 Refresh Reviews"):
            st.rerun()

    elif page == "Workflow Designer":
        st.header("🔧 Workflow Designer")
        st.write("Design custom processing workflows with granular control over each step")
        
        # Initialize workflow state
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
        
        # Load saved workflows section
        st.subheader("💾 Saved Workflows")
        try:
            response = requests.get(f"{settings.api_base_url}/api/v1/workflows", timeout=5)
            if response.status_code == 200:
                saved_workflows = response.json()
                if saved_workflows:
                    col_load, col_manage = st.columns([2, 1])
                    with col_load:
                        workflow_options = {f"{wf['name']} ({wf.get('id', 'Unknown ID')})": wf for wf in saved_workflows}
                        selected_workflow_name = st.selectbox("Load Saved Workflow:", [""] + list(workflow_options.keys()))
                        
                        if selected_workflow_name and st.button("📥 Load Workflow"):
                            selected_workflow = workflow_options[selected_workflow_name]
                            # Convert API workflow back to GUI format
                            st.session_state.workflow_steps = []
                            
                            # Add steps based on workflow configuration
                            step_id = 1
                            if selected_workflow.get("enable_ner"):
                                st.session_state.workflow_steps.append({
                                    "id": step_id,
                                    "tool": "Entity Recognition",
                                    "option": "AI-powered NER",
                                    "config": {"confidence_threshold": selected_workflow.get("confidence_threshold", 0.7), "parallel_processing": True}
                                })
                                step_id += 1
                            
                            if selected_workflow.get("enable_llm_extraction"):
                                st.session_state.workflow_steps.append({
                                    "id": step_id,
                                    "tool": "Legal Analysis",
                                    "option": "Contract analysis",
                                    "config": {"confidence_threshold": selected_workflow.get("confidence_threshold", 0.7), "parallel_processing": True}
                                })
                                step_id += 1
                            
                            if selected_workflow.get("enable_confidence_calibration"):
                                st.session_state.workflow_steps.append({
                                    "id": step_id,
                                    "tool": "Quality Review",
                                    "option": "Low confidence only",
                                    "config": {
                                        "confidence_threshold": selected_workflow.get("confidence_threshold", 0.7),
                                        "parallel_processing": True,
                                        "review_threshold": selected_workflow.get("confidence_threshold", 0.8),
                                        "auto_approve": True
                                    }
                                })
                            
                            st.success(f"Loaded workflow: {selected_workflow['name']}")
                            st.rerun()
                    
                    with col_manage:
                        st.write(f"**Available:** {len(saved_workflows)} workflows")
                        if st.button("🔄 Refresh List"):
                            st.rerun()
                else:
                    st.info("No saved workflows found. Create one below!")
            else:
                st.warning("Could not load saved workflows from backend")
        except Exception as e:
            st.warning(f"Error loading saved workflows: {e}")
        
        st.divider()
        st.subheader("📋 Available Processing Tools")
        
        # Available tools with descriptions
        available_tools = {
            "Text Extraction": {
                "description": "Extract text content from documents",
                "options": ["OCR", "Direct text", "Hybrid extraction"],
                "required": True
            },
            "Entity Recognition": {
                "description": "Identify entities like people, organizations, dates",
                "options": ["SpaCy NER", "AI-powered NER", "Custom patterns"],
                "required": False
            },
            "Legal Analysis": {
                "description": "Analyze legal content and identify issues",
                "options": ["Contract analysis", "Compliance check", "Risk assessment"],
                "required": False
            },
            "Citation Extraction": {
                "description": "Extract legal citations and references", 
                "options": ["Bluebook format", "ALWD format", "Custom format"],
                "required": False
            },
            "Summarization": {
                "description": "Generate document summaries",
                "options": ["Executive summary", "Key points", "Technical summary"],
                "required": False
            },
            "Classification": {
                "description": "Classify document type and content",
                "options": ["Document type", "Legal domain", "Urgency level"],
                "required": False
            },
            "Quality Review": {
                "description": "Human review of extractions",
                "options": ["All extractions", "Low confidence only", "High-risk only"],
                "required": False
            },
            "Export": {
                "description": "Export results in various formats",
                "options": ["JSON", "PDF report", "Excel spreadsheet"],
                "required": False
            },
            "Database Storage": {
                "description": "Store results in persistent database",
                "options": ["Violations & Memory", "Knowledge Graph", "Full Analytics"],
                "required": False
            },
            "XAI Processing": {
                "description": "Advanced AI processing with Grok models",
                "options": ["Grok-3-Mini", "Grok-3-Reasoning", "Custom XAI"],
                "required": False
            },
            "Real-time Monitoring": {
                "description": "WebSocket-based progress tracking",
                "options": ["Basic progress", "Granular stages", "Full analytics"],
                "required": False
            }
        }
        
        # Tool selection interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🛠️ Add Processing Step")
            selected_tool = st.selectbox("Select Tool:", list(available_tools.keys()))
            
            if selected_tool:
                tool_info = available_tools[selected_tool]
                st.write(f"**Description:** {tool_info['description']}")
                
                # Tool options
                selected_option = st.selectbox(f"{selected_tool} Option:", tool_info['options'])
                
                # Additional configuration
                with st.expander("⚙️ Advanced Configuration"):
                    confidence_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.7, 0.05)
                    parallel_processing = st.checkbox("Enable parallel processing", value=True)
                    
                    if selected_tool == "Entity Recognition":
                        entity_types = st.multiselect(
                            "Entity Types to Extract:",
                            ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "LEGAL_ISSUE"],
                            default=["PERSON", "ORGANIZATION"]
                        )
                    
                    if selected_tool == "Quality Review":
                        review_threshold = st.slider("Review Threshold:", 0.0, 1.0, 0.8, 0.05)
                        auto_approve = st.checkbox("Auto-approve high confidence items", value=True)
                    
                    elif selected_tool == "Database Storage":
                        violation_storage = st.checkbox("Store violation records", value=True)
                        memory_storage = st.checkbox("Store memory entries", value=True)
                        analytics_tracking = st.checkbox("Enable analytics tracking", value=True)
                    
                    elif selected_tool == "XAI Processing":
                        xai_temperature = st.slider("XAI Temperature:", 0.0, 1.0, 0.7, 0.1)
                        xai_max_tokens = st.number_input("Max Tokens:", min_value=1000, max_value=131072, value=4096, step=1000)
                        xai_reasoning = st.checkbox("Enable reasoning mode", value=False)
                    
                    elif selected_tool == "Real-time Monitoring":
                        broadcast_frequency = st.slider("Broadcast Frequency (sec):", 1, 10, 2, 1)
                        websocket_enabled = st.checkbox("Enable WebSocket broadcasting", value=True)
                        progress_granularity = st.selectbox("Progress Detail Level:", ["Basic", "Detailed", "Verbose"])
                
                if st.button(f"➕ Add {selected_tool} Step"):
                    step = {
                        "id": len(st.session_state.workflow_steps) + 1,
                        "tool": selected_tool,
                        "option": selected_option,
                        "config": {
                            "confidence_threshold": confidence_threshold,
                            "parallel_processing": parallel_processing
                        }
                    }
                    
                    # Add tool-specific config
                    if selected_tool == "Entity Recognition":
                        step["config"]["entity_types"] = entity_types
                    elif selected_tool == "Quality Review":
                        step["config"]["review_threshold"] = review_threshold
                        step["config"]["auto_approve"] = auto_approve
                    elif selected_tool == "Database Storage":
                        step["config"]["violation_storage"] = violation_storage
                        step["config"]["memory_storage"] = memory_storage
                        step["config"]["analytics_tracking"] = analytics_tracking
                    elif selected_tool == "XAI Processing":
                        step["config"]["xai_temperature"] = xai_temperature
                        step["config"]["xai_max_tokens"] = xai_max_tokens
                        step["config"]["xai_reasoning"] = xai_reasoning
                    elif selected_tool == "Real-time Monitoring":
                        step["config"]["broadcast_frequency"] = broadcast_frequency
                        step["config"]["websocket_enabled"] = websocket_enabled
                        step["config"]["progress_granularity"] = progress_granularity
                    
                    st.session_state.workflow_steps.append(step)
                    st.success(f"Added {selected_tool} step!")
                    st.rerun()
        
        with col2:
            st.subheader("📋 Current Workflow")
            
            if st.session_state.workflow_steps:
                # Workflow visualization
                for i, step in enumerate(st.session_state.workflow_steps):
                    with st.container():
                        col_step, col_actions = st.columns([3, 1])
                        
                        with col_step:
                            st.markdown(f"**{i+1}. {step['tool']}** - {step['option']}")
                            st.caption(f"Confidence: {step['config']['confidence_threshold']:.1%} | Parallel: {step['config']['parallel_processing']}")
                        
                        with col_actions:
                            if st.button("🗑️", key=f"delete_{step['id']}", help="Delete step"):
                                st.session_state.workflow_steps.remove(step)
                                st.rerun()
                            
                            if i > 0 and st.button("⬆️", key=f"up_{step['id']}", help="Move up"):
                                st.session_state.workflow_steps[i], st.session_state.workflow_steps[i-1] = \
                                    st.session_state.workflow_steps[i-1], st.session_state.workflow_steps[i]
                                st.rerun()
                            
                            if i < len(st.session_state.workflow_steps)-1 and st.button("⬇️", key=f"down_{step['id']}", help="Move down"):
                                st.session_state.workflow_steps[i], st.session_state.workflow_steps[i+1] = \
                                    st.session_state.workflow_steps[i+1], st.session_state.workflow_steps[i]
                                st.rerun()
                        
                        if i < len(st.session_state.workflow_steps) - 1:
                            st.markdown("⬇️")
                
                st.divider()
                
                # Workflow actions
                col_save, col_clear, col_run = st.columns(3)
                
                with col_save:
                    workflow_name = st.text_input("Workflow Name:", value="Custom Workflow")
                    if st.button("💾 Save Workflow"):
                        # Save workflow configuration to backend
                        try:
                            # Convert steps to API format
                            api_steps = []
                            for step in st.session_state.workflow_steps:
                                api_step = {
                                    "enable_ner": step["tool"] == "Entity Recognition",
                                    "enable_llm_extraction": step["tool"] in ["Entity Recognition", "Legal Analysis"],
                                    "enable_confidence_calibration": step["tool"] == "Quality Review",
                                    "confidence_threshold": step["config"]["confidence_threshold"]
                                }
                                if step["tool"] == "Entity Recognition":
                                    api_step["entity_types"] = step["config"].get("entity_types", [])
                                api_steps.append(api_step)
                            
                            workflow_config = {
                                "name": workflow_name,
                                "description": f"Custom workflow with {len(st.session_state.workflow_steps)} steps",
                                "enable_ner": any(step["tool"] == "Entity Recognition" for step in st.session_state.workflow_steps),
                                "enable_llm_extraction": any(step["tool"] in ["Entity Recognition", "Legal Analysis"] for step in st.session_state.workflow_steps),
                                "enable_confidence_calibration": any(step["tool"] == "Quality Review" for step in st.session_state.workflow_steps),
                                "confidence_threshold": next((step["config"]["confidence_threshold"] for step in st.session_state.workflow_steps if step["tool"] == "Quality Review"), 0.7)
                            }
                            
                            response = requests.post(
                                f"{settings.api_base_url}/api/v1/workflows",
                                json=workflow_config,
                                timeout=10
                            )
                            if response.status_code == 201:
                                saved_workflow = response.json()
                                st.success(f"✅ Workflow '{workflow_name}' saved successfully!")
                                st.info(f"🆔 Workflow ID: {saved_workflow.get('id', 'N/A')}")
                                # Store in session for quick access
                                if 'saved_workflows' not in st.session_state:
                                    st.session_state.saved_workflows = []
                                st.session_state.saved_workflows.append(saved_workflow)
                            else:
                                st.error(f"❌ Failed to save workflow: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Failed to save workflow: {e}")
                
                with col_clear:
                    st.write("")  # Spacing
                    if st.button("🗑️ Clear All Steps"):
                        st.session_state.workflow_steps = []
                        st.rerun()
                
                with col_run:
                    st.write("")  # Spacing
                    if st.button("▶️ Test Workflow"):
                        if st.session_state.workflow_steps:
                            with st.spinner("Testing workflow..."):
                                # Create a test document
                                test_text = "This is a test legal document involving John Doe and ABC Corporation. The contract was signed on January 15, 2024, for a total amount of $50,000."
                                
                                try:
                                    # Create a temporary test document
                                    files = {'file': ('test_document.txt', test_text.encode(), 'text/plain')}
                                    upload_response = requests.post(
                                        f"{settings.api_base_url}/api/v1/documents/upload",
                                        files=files,
                                        timeout=10,
                                    )
                                    
                                    if upload_response.status_code == 200:
                                        upload_data = upload_response.json()
                                        document_id = upload_data.get("document_id")
                                        
                                        # Build processing options from workflow
                                        proc_options = {
                                            "enable_ner": any(step["tool"] == "Entity Recognition" for step in st.session_state.workflow_steps),
                                            "enable_llm_extraction": any(step["tool"] in ["Entity Recognition", "Legal Analysis"] for step in st.session_state.workflow_steps),
                                            "enable_confidence_calibration": any(step["tool"] == "Quality Review" for step in st.session_state.workflow_steps),
                                            "confidence_threshold": next((step["config"]["confidence_threshold"] for step in st.session_state.workflow_steps if step["tool"] == "Quality Review"), 0.7)
                                        }
                                        
                                        # Process the test document
                                        process_response = requests.post(
                                            f"{settings.api_base_url}/api/v1/documents/{document_id}/process",
                                            json=proc_options,
                                            timeout=30,
                                        )
                                        
                                        if process_response.status_code == 202:
                                            st.success("✅ Workflow test started!")
                                            st.info(f"📄 Test Document ID: {document_id}")
                                            st.info("💡 Check the 'Process Monitoring' page to see real-time progress")
                                            st.info("📊 Check 'Document Results' page to see the test results")
                                        else:
                                            st.error(f"❌ Failed to start workflow test: {process_response.status_code}")
                                    else:
                                        st.error(f"❌ Failed to upload test document: {upload_response.status_code}")
                                
                                except Exception as e:
                                    st.error(f"❌ Workflow test failed: {e}")
                        else:
                            st.warning("⚠️ Please add at least one step to test the workflow")
            else:
                st.info("Add processing steps to build your workflow")
                st.markdown("""
                **💡 Quick Start Tips:**
                1. Start with **Text Extraction** (required for most workflows)
                2. Add **Entity Recognition** to identify key information
                3. Include **Legal Analysis** for legal documents
                4. End with **Quality Review** for human oversight
                5. Add **Export** to save results
                """)

    elif page == "Process Monitoring":
        st.header("📊 Process Monitoring")
        st.write("Real-time monitoring of document processing with granular visibility")
        
        # Real-time process monitoring
        try:
            response = requests.get(f"{settings.api_base_url}/api/v1/documents", timeout=5)
            if response.status_code == 200:
                documents = response.json()
                
                # Filter processing documents
                processing_docs = [d for d in documents if d.get("status") == "processing"]
                completed_docs = [d for d in documents if d.get("status") == "completed"]
                failed_docs = [d for d in documents if d.get("status") == "failed"]
                
                # Status overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔄 Processing", len(processing_docs))
                with col2:
                    st.metric("✅ Completed", len(completed_docs))
                with col3:
                    st.metric("❌ Failed", len(failed_docs))
                with col4:
                    st.metric("📋 Total", len(documents))
                
                if processing_docs:
                    st.subheader("🔄 Active Processing")
                    for doc in processing_docs:
                        with st.expander(f"📄 {doc.get('filename', 'Unknown')} ({doc.get('progress', 0)*100:.0f}%)", expanded=True):
                            col_info, col_progress = st.columns([1, 2])
                            
                            with col_info:
                                st.write(f"**Document ID:** {doc.get('document_id', 'N/A')}")
                                st.write(f"**Current Stage:** {doc.get('stage', 'Unknown')}")
                                st.write(f"**Status:** {doc.get('status', 'Unknown')}")
                            
                            with col_progress:
                                progress = doc.get('progress', 0)
                                st.progress(progress)
                                st.write(f"Progress: {progress*100:.1f}%")
                                
                                # Stage indicators
                                stages = ["uploaded", "text_extraction", "entity_extraction", "legal_analysis", "summarization", "human_review_processing", "completed"]
                                current_stage = doc.get('stage', 'unknown')
                                
                                stage_status = []
                                for stage in stages:
                                    if stages.index(stage) < stages.index(current_stage) if current_stage in stages else 0:
                                        stage_status.append("✅")
                                    elif stage == current_stage:
                                        stage_status.append("🔄")
                                    else:
                                        stage_status.append("⏸️")
                                
                                st.write(" ".join([f"{status} {stage}" for status, stage in zip(stage_status, stages)]))
                
                if completed_docs:
                    st.subheader("✅ Recently Completed")
                    for doc in completed_docs[-5:]:  # Last 5 completed
                        with st.container():
                            col_name, col_stats = st.columns([2, 1])
                            with col_name:
                                st.write(f"📄 **{doc.get('filename', 'Unknown')}**")
                                st.caption(f"ID: {doc.get('document_id', 'N/A')}")
                            with col_stats:
                                if 'result_summary' in doc:
                                    results = doc['result_summary']
                                    entities_count = len(results.get('entities', []))
                                    st.metric("Entities", entities_count)
                
                if failed_docs:
                    st.subheader("❌ Failed Processing")
                    for doc in failed_docs:
                        st.error(f"📄 **{doc.get('filename', 'Unknown')}** - {doc.get('error', 'Unknown error')}")
                
            else:
                st.error("Failed to fetch processing status")
                
        except Exception as e:
            st.error(f"Error loading process monitoring: {e}")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh every 5 seconds")
        if auto_refresh:
            import time
            time.sleep(5)
            st.rerun()

    elif page == "System Status":
        st.header("⚙️ SYSTEM STATUS & HEALTH")
        st.write("Real-time monitoring of all system components")
        
        try:
            response = requests.get(f"{settings.api_base_url}/api/v1/system/health", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                
                # Overall Status
                overall_status = status_data.get("overall_status", "UNKNOWN")
                if overall_status == "HEALTHY":
                    st.success(f"🟢 OVERALL STATUS: {overall_status}")
                elif overall_status == "BUSY":
                    st.warning(f"🟡 OVERALL STATUS: {overall_status}")
                else:
                    st.error(f"🔴 OVERALL STATUS: {overall_status}")
                
                # Metrics Dashboard
                metrics = status_data.get("performance_metrics_summary", {})
                if metrics:
                    st.subheader("📊 PERFORMANCE METRICS")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Documents", metrics.get("total_documents", 0))
                    with col2:
                        st.metric("Completed", metrics.get("completed_documents", 0))
                    with col3:
                        st.metric("Processing", metrics.get("processing_documents", 0))
                    with col4:
                        success_rate = metrics.get("success_rate", 0)
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric("Memory Usage", f"{metrics.get('memory_usage_mb', 0)} MB")
                    with col6:
                        st.metric("Avg Processing Time", f"{metrics.get('avg_processing_time_sec', 0)} sec")
                
                # Service Status
                services = status_data.get("services_status", {})
                if services:
                    st.subheader("🔧 SERVICE STATUS")
                    
                    for service_name, service_info in services.items():
                        with st.expander(f"{service_name.replace('_', ' ').title()}", expanded=True):
                            status = service_info.get("status", "unknown")
                            if status in ["running", "available", "configured"]:
                                st.success(f"✅ Status: {status}")
                            else:
                                st.error(f"❌ Status: {status}")
                            
                            details = service_info.get("details", "No details available")
                            st.write(f"**Details:** {details}")
                            
                            # Show additional info if available
                            for key, value in service_info.items():
                                if key not in ["status", "details"]:
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Raw Data Section
                with st.expander("🗃️ Raw System Data"):
                    st.json(status_data)
                    
            else:
                st.error(f"❌ Failed to fetch system status: HTTP {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ System Status Error: {e}")
            st.write("**Troubleshooting:**")
            st.write("1. Check if the backend server is running")
            st.write("2. Verify the API endpoint is accessible")
            st.write("3. Check network connectivity")

    elif page == "Settings":
        st.header("⚙️ Settings")
        st.write("Configure API keys and model preferences")
        
        # Create tabs for different settings
        api_tab, model_tab = st.tabs(["API Keys", "Model Settings"])
        
        with api_tab:
            st.subheader("API Keys")
            st.info("Your API keys are stored securely and only used for AI model requests.")
            
            # OpenAI settings
            with st.expander("🤖 OpenAI", expanded=True):
                openai_key = st.text_input(
                    "OpenAI API Key", 
                    type="password",
                    key="openai_key",
                    help="Get your API key from https://platform.openai.com/api-keys"
                )
                if st.button("Save OpenAI Key", key="save_openai"):
                    if openai_key:
                        # Store in session state for now
                        st.session_state['openai_api_key'] = openai_key
                        st.success("OpenAI API key saved!")
                    else:
                        st.error("Please enter an API key")
            
            # Anthropic settings
            with st.expander("🤖 Anthropic (Claude)"):
                anthropic_key = st.text_input(
                    "Anthropic API Key", 
                    type="password",
                    key="anthropic_key",
                    help="Get your API key from https://console.anthropic.com/"
                )
                if st.button("Save Anthropic Key", key="save_anthropic"):
                    if anthropic_key:
                        st.session_state['anthropic_api_key'] = anthropic_key
                        st.success("Anthropic API key saved!")
                    else:
                        st.error("Please enter an API key")
            
            # X.AI settings
            with st.expander("🤖 X.AI (Grok)"):
                xai_key = st.text_input(
                    "X.AI API Key", 
                    type="password",
                    key="xai_key",
                    help="Get your API key from https://x.ai/"
                )
                if st.button("Save X.AI Key", key="save_xai"):
                    if xai_key:
                        st.session_state['xai_api_key'] = xai_key
                        st.success("X.AI API key saved!")
                    else:
                        st.error("Please enter an API key")
        
        with model_tab:
            st.subheader("Model Selection")
            
            # Provider selection
            provider = st.selectbox(
                "AI Provider",
                ["openai", "anthropic", "xai", "ollama"],
                index=0,
                key="ai_provider",
                help="Choose your preferred AI provider"
            )
            
            # Model selection based on provider
            if provider == "openai":
                model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                    index=1,
                    key="openai_model"
                )
            elif provider == "anthropic":
                model = st.selectbox(
                    "Anthropic Model",
                    ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                    index=0,
                    key="anthropic_model"
                )
            elif provider == "xai":
                model = st.selectbox(
                    "X.AI Model",
                    ["grok-3-mini", "grok-3-mini-latest", "grok-beta", "grok-vision-beta"],
                    index=0,
                    key="xai_model"
                )
            else:  # ollama
                model = st.text_input(
                    "Ollama Model", 
                    value="llama3.2",
                    key="ollama_model",
                    help="Enter the name of your local Ollama model"
                )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    key="temperature",
                    help="Controls randomness in responses. Lower = more focused, Higher = more creative"
                )
                
                # Set max tokens based on provider
                if provider == "xai":
                    default_max_tokens = 131072
                    max_tokens_limit = 131072
                    help_text = "Maximum length of the AI response (Grok-3-mini supports up to 131,072 tokens)"
                else:
                    default_max_tokens = 4096
                    max_tokens_limit = 32000
                    help_text = "Maximum length of the AI response"
                
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=max_tokens_limit,
                    value=default_max_tokens,
                    step=100,
                    key="max_tokens",
                    help=help_text
                )
            
            if st.button("Save Model Settings", key="save_model"):
                # Send settings to backend
                try:
                    settings_data = {
                        "provider": provider,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    # Add API key if available
                    if provider == "openai" and 'openai_api_key' in st.session_state:
                        settings_data['api_key'] = st.session_state['openai_api_key']
                    elif provider == "anthropic" and 'anthropic_api_key' in st.session_state:
                        settings_data['api_key'] = st.session_state['anthropic_api_key']
                    elif provider == "xai" and 'xai_api_key' in st.session_state:
                        settings_data['api_key'] = st.session_state['xai_api_key']
                    
                    response = requests.post(
                        f"{settings.api_base_url}/api/v1/config/model",
                        json=settings_data,
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.success("✅ Model settings saved successfully!")
                        st.info(f"🤖 Using {provider} - {model}")
                    else:
                        st.error(f"Failed to save settings: {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to save settings: {e}")

    st.sidebar.markdown("---")
    st.sidebar.info("Legal AI System v2.1.0")


def main_streamlit_entry():
    """Main entry point for the Streamlit GUI application."""
    setup_main_app_logging_gui()
    streamlit_logger.info("Legal AI System Streamlit GUI starting...")

    if not check_gui_dependencies():
        streamlit_logger.critical(
            "Critical GUI dependencies missing. Streamlit app cannot start.")
        # Error already printed by check_gui_dependencies
        sys.exit(1)

    run_streamlit_app_content()


if __name__ == "__main__":
    # This makes streamlit_app.py directly runnable: `python legal_ai_system/gui/streamlit_app.py`
    # It's also the target for `streamlit run legal_ai_system/gui/streamlit_app.py`
    main_streamlit_entry()