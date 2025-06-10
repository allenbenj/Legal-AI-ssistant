import sys
import os
import sentry_sdk
# Add the parent directory to sys.path to resolve imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st

# MUST be first Streamlit command - configure page
st.set_page_config(
    page_title="Legal AI System",
    page_icon="⚖️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Fix for Streamlit execution - use absolute imports
sentry_sdk.init(
    dsn="https://2fbad862414aad747dba577c60110470@o4509439121489920.ingest.us.sentry.io/4509439123587072",
    # Add request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
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
    # This is a simplified setup. If detailed_logging is available and initialized by another part
    # of the system (e.g. if FastAPI starts first and initializes it), this might not be needed
    # or could integrate with it.
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Check if root logger already has handlers (e.g., from detailed_logging)
    # For a truly standalone Streamlit app, we might want to configure it.
    # If it's part of a larger system, rely on the system's logging config.
    # For this refactor, let's assume it can configure itself if no handlers exist.
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
        # Attempting to # The above code is a Python script that contains a print statement. However,
        # the print statement is empty, so it will not output anything when executed.
        print("to console as Streamlit might not be fully up yet")
        print(f"❌ Missing required GUI packages: {', '.join(missing)}")
        return False  # Do not auto-install in this environment for safety

    streamlit_logger.info("All checked GUI dependencies are available.")
    return True


def run_streamlit_app_content():
    """Defines the actual content and logic of the Streamlit application."""

    # `check_gui_dependencies` ensures `streamlit` is installed before this
    # function is executed. Therefore we can rely on the module level import of
    # `streamlit as st` and avoid a second import here. This also keeps linters
    # happy since the variable `st` is defined at the module level.

    st.title("🏛️ Legal AI System Dashboard")
    st.caption("Professional Edition - Document Analysis & Knowledge Management")

    st.sidebar.header("Navigation")
    # Check if services are available (conceptual, replace with actual service status check)
    # This would typically involve making an API call to the FastAPI backend's health check.
    backend_status = "API Not Connected"  # Placeholder
    try:
        # Conceptual: r = requests.get("http://localhost:8000/api/v1/system/health") # Assuming API is on port 8000
        # if r.status_code == 200 and r.json().get("overall_status") == "HEALTHY": backend_status = "API Connected"
        pass  # For now, skip actual API call
    except Exception:
        pass

    st.sidebar.info(f"Status: {backend_status}")

    page = st.sidebar.radio(
        "Go to", ["Dashboard", "Document Upload", "Knowledge Graph", "System Status"])

    if page == "Dashboard":
        st.header("System Overview")
        st.write(
            "Welcome to the Legal AI System. This dashboard provides an overview of system activities and performance.")
        # Placeholder for dashboard components
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Documents Processed", value="0", delta="0 today")
        col2.metric(label="Active Workflows", value="0")
        col3.metric(label="Pending Reviews", value="0")

        st.subheader("Recent Activity")
        st.info("No recent activity. Upload documents to begin processing.")

    elif page == "Document Upload":
        st.header("📄 Document Upload & Processing")
        uploaded_file = st.file_uploader("Choose a document to analyze", type=[
            'pdf', 'docx', 'txt', 'md'])

        if uploaded_file is not None:
            st.write(f"Uploaded: {uploaded_file.name} ({uploaded_file.type})")

            with st.expander("Processing Options"):
                # These would map to ProcessingRequest model for the backend
                st.checkbox("Enable NER", value=True, key="opt_ner")
                st.checkbox("Enable LLM Extraction",
                            value=True, key="opt_llm_extract")
                st.checkbox("Enable Confidence Calibration",
                            value=True, key="opt_conf_calib")
                st.slider("Confidence Threshold", 0.1, 1.0,
                            0.7, 0.05, key="opt_conf_thresh")

            if st.button("Process Document"):
                with st.spinner("Sending document to backend for processing..."):
                    # API Call to FastAPI backend's /documents/upload and /documents/{id}/process
                    # For this example, simulate the process.
                    # files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    # try:
                    #     upload_response = requests.post("http://localhost:8000/api/v1/documents/upload", files=files)
                    #     upload_response.raise_for_status()
                    #     upload_data = upload_response.json()
                    #     document_id = upload_data.get("document_id")
                    #     st.info(f"Document uploaded with ID: {document_id}. Initiating processing...")
                    #
                    #     proc_options = {
                    #         "enable_ner": st.session_state.opt_ner,
                    #         "enable_llm_extraction": st.session_state.opt_llm_extract,
                    #         # ... map other options
                    #     }
                    #     process_response = requests.post(f"http://localhost:8000/api/v1/documents/{document_id}/process",
                    #                                         json={"processing_options": proc_options})
                    #     process_response.raise_for_status()
                    #     st.success(f"Processing started for document ID: {document_id}. Check status page or notifications.")
                    # except requests.exceptions.RequestException as e:
                    #     st.error(f"API Error: {e}")
                    # except Exception as e:
                    #     st.error(f"An error occurred: {e}")

                    # Mocking the process
                    time.sleep(2)  # Simulate API call
                    st.success(
                        f"Document '{uploaded_file.name}' sent for processing! (Mocked)")
                    st.info(
                        "In a real system, you would monitor progress via status page or WebSockets.")

    elif page == "Knowledge Graph":
        st.header("🕸️ Knowledge Graph Explorer")
        st.write("Visualize and query the legal knowledge graph. (Conceptual)")
        st.image("https://via.placeholder.com/800x400.png?text=Knowledge+Graph+Visualization+Placeholder",
                 caption="Knowledge Graph (Placeholder - requires integration with a graph viz library and API)")

        query_st = st.text_input(
            "Search Knowledge Graph (e.g., 'entities related to John Doe')")
        if st.button("Search KG"):
            if query_st:
                st.write(f"Searching for: {query_st}")
                # API call to GraphQL or REST endpoint for KG search
                # st.json(mock_kg_search_result)
                st.info("Search results would appear here. (Mocked)")
            else:
                st.warning("Please enter a search query.")

    elif page == "System Status":
        st.header("⚙️ System Status & Health")
        st.write(
            "Monitor the health and performance of system components. (Conceptual)")
        # API call to FastAPI backend's /system/health
        # status_data = requests.get("http://localhost:8000/api/v1/system/health").json()
        # st.json(status_data)
        st.info(
            "System health details would be fetched from the API and displayed here.")
        st.json({
            "Overall Status": "HEALTHY (Mocked)", "API Backend": "Online",
            "LLM Provider": "Connected", "Database": "Operational"
        })

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

    # Conceptual: Initialize core system parts if this GUI is meant to run somewhat independently
    # or if it needs to configure things before the FastAPI backend is assumed to be up.
    # In a typical setup, Streamlit acts as a client to the FastAPI backend, so backend init is separate.
    # from legal_ai_system.core.system_initializer import initialize_system # Potentially
    # initialize_system(is_first_run_setup=False)

    run_streamlit_app_content()


if __name__ == "__main__":
    # This makes streamlit_app.py directly runnable: `python legal_ai_system/gui/streamlit_app.py`
    # It's also the target for `streamlit run legal_ai_system/gui/streamlit_app.py`
    main_streamlit_entry()
