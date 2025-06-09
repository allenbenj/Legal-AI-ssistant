"""
XAI (Grok) Integration for Legal AI System GUI
Provides direct integration with xAI's Grok models for document processing
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# Import existing configuration
try:
    import sys

    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from legal_ai_system.core.grok_config import (
        DEFAULT_GROK_MODEL,
        GROK_MODELS_CONFIG,
        get_optimized_prompt,
    )
except ImportError as e:
    logging.warning(f"Could not import Grok configuration: {e}")
    # Fallback configurations
    GROK_MODELS_CONFIG = {
        "grok-3-mini": {
            "model_name": "grok-3-mini",
            "context_length": 8192,
            "max_tokens": 4096,
            "temperature": 0.7,
            "use_case": "Fast, efficient legal analysis",
            "reasoning": False,
        },
        "grok-3-reasoning": {
            "model_name": "grok-3-reasoning",
            "context_length": 8192,
            "max_tokens": 4096,
            "temperature": 0.3,
            "use_case": "Complex legal reasoning and analysis",
            "reasoning": True,
        },
    }
    DEFAULT_GROK_MODEL = "grok-3-mini"

logger = logging.getLogger(__name__)


class XAIGrokClient:
    """Direct client for XAI's Grok API integration"""

    def __init__(
        self, api_key: Optional[str] = None, base_url: str = "https://api.x.ai/v1"
    ):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = base_url
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to XAI API"""
        if not self.api_key:
            return {"status": "ERROR", "message": "No API key provided"}

        try:
            # Test with a simple completion request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": DEFAULT_GROK_MODEL,
                    "messages": [{"role": "user", "content": "Test connection"}],
                    "max_tokens": 10,
                },
                timeout=10,
            )

            if response.status_code == 200:
                return {"status": "HEALTHY", "message": "XAI API connection successful"}
            else:
                return {
                    "status": "ERROR",
                    "message": f"API returned status {response.status_code}",
                }

        except Exception as e:
            logger.error(f"XAI API connection test failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    def analyze_document(
        self,
        document_text: str,
        analysis_type: str = "legal_analysis",
        model: str = DEFAULT_GROK_MODEL,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze document using Grok models"""
        if not self.api_key:
            return {"status": "ERROR", "message": "No API key configured"}

        try:
            # Get optimized prompt for the analysis type and model
            prompt = self._get_analysis_prompt(analysis_type, model, document_text)

            # Configure model parameters
            model_config = GROK_MODELS_CONFIG.get(
                model, GROK_MODELS_CONFIG[DEFAULT_GROK_MODEL]
            )

            # Prepare request
            request_data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": model_config["max_tokens"],
                "temperature": (
                    temperature
                    if temperature is not None
                    else model_config["temperature"]
                ),
            }

            # Make API call
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=request_data, timeout=60
            )

            response.raise_for_status()
            result = response.json()

            # Extract and format response
            if "choices" in result and len(result["choices"]) > 0:
                analysis_content = result["choices"][0]["message"]["content"]

                return {
                    "status": "SUCCESS",
                    "analysis": analysis_content,
                    "model_used": model,
                    "analysis_type": analysis_type,
                    "token_usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"status": "ERROR", "message": "No response content received"}

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    def detect_violations(
        self,
        document_text: str,
        model: str = DEFAULT_GROK_MODEL,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Detect legal violations in document"""
        return self.analyze_document(
            document_text, "violation_detection", model, temperature
        )

    def extract_entities(
        self,
        document_text: str,
        model: str = DEFAULT_GROK_MODEL,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extract legal entities from document"""
        try:
            prompt = f"""
            Extract legal entities from the following document. Identify:
            
            1. **People**: Names, roles, titles
            2. **Organizations**: Companies, agencies, courts
            3. **Locations**: Addresses, jurisdictions, venues
            4. **Dates**: Important dates and deadlines
            5. **Legal Concepts**: Laws, regulations, cases cited
            6. **Financial**: Amounts, damages, fees
            
            Format the response as JSON with categories and confidence scores.
            
            Document:
            {document_text}
            """

            model_config = GROK_MODELS_CONFIG.get(
                model, GROK_MODELS_CONFIG[DEFAULT_GROK_MODEL]
            )

            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": model_config["max_tokens"],
                    "temperature": temperature if temperature is not None else 0.1,
                },
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                entities_content = result["choices"][0]["message"]["content"]

                return {
                    "status": "SUCCESS",
                    "entities": entities_content,
                    "model_used": model,
                    "token_usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"status": "ERROR", "message": "No entities extracted"}

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    def summarize_document(
        self,
        document_text: str,
        model: str = DEFAULT_GROK_MODEL,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create document summary"""
        return self.analyze_document(
            document_text, "document_summary", model, temperature
        )

    def _get_analysis_prompt(
        self, analysis_type: str, model: str, document_text: str
    ) -> str:
        """Get optimized prompt for analysis type and model"""
        try:
            # Try to use the configured prompts if available
            if "get_optimized_prompt" in globals():
                return get_optimized_prompt(
                    analysis_type, model, document_text=document_text
                )
        except Exception as e:
            logger.warning(f"Could not get optimized prompt: {e}")

        # Fallback prompts
        fallback_prompts = {
            "legal_analysis": f"""
            You are a legal AI assistant. Analyze the following legal document with precision and accuracy.
            
            Focus on:
            1. Legal violations and misconduct
            2. Procedural issues
            3. Constitutional concerns
            4. Factual inconsistencies
            
            Document to analyze:
            {document_text}
            
            Provide a structured analysis with citations and reasoning.
            """,
            "violation_detection": f"""
            As a legal expert, examine this document for potential legal violations.
            
            Look for:
            - Constitutional violations
            - Procedural violations
            - Ethical violations
            - Evidence handling issues
            
            Document:
            {document_text}
            
            List each violation with:
            1. Type of violation
            2. Specific evidence
            3. Relevant legal standards
            4. Severity assessment
            """,
            "document_summary": f"""
            Summarize this legal document concisely but comprehensively.
            Include key parties, issues, holdings, and implications.
            
            Document:
            {document_text}
            
            Provide a structured summary with:
            - Case/Document overview
            - Key legal issues
            - Main arguments
            - Conclusions/Holdings
            - Implications
            """,
        }

        return fallback_prompts.get(analysis_type, fallback_prompts["legal_analysis"])

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Grok models"""
        models = []
        for model_name, config in GROK_MODELS_CONFIG.items():
            models.append(
                {
                    "name": model_name,
                    "display_name": model_name.replace("-", " ").title(),
                    "use_case": config["use_case"],
                    "reasoning": config.get("reasoning", False),
                    "context_length": config["context_length"],
                    "max_tokens": config["max_tokens"],
                }
            )
        return models


class XAIIntegratedGUI:
    """Enhanced GUI components with XAI integration"""

    @staticmethod
    def render_xai_settings():
        """Render XAI configuration settings"""
        st.subheader("🤖 XAI (Grok) Configuration")

        # API Key configuration
        col1, col2 = st.columns(2)

        with col1:
            api_key = st.text_input(
                "XAI API Key",
                type="password",
                value=st.session_state.get("xai_api_key", ""),
                help="Enter your XAI API key for Grok models",
            )
            if api_key != st.session_state.get("xai_api_key", ""):
                st.session_state.xai_api_key = api_key

        with col2:
            base_url = st.text_input(
                "API Base URL", value="https://api.x.ai/v1", help="XAI API endpoint URL"
            )

        # Test connection
        if st.button("🔍 Test XAI Connection"):
            if api_key:
                client = XAIGrokClient(api_key, base_url)
                test_result = client.test_connection()

                if test_result["status"] == "HEALTHY":
                    st.success(f"✅ {test_result['message']}")
                    st.session_state.xai_connected = True
                else:
                    st.error(f"❌ {test_result['message']}")
                    st.session_state.xai_connected = False
            else:
                st.warning("Please enter your XAI API key first")

        # Model selection
        st.subheader("Model Selection")

        if api_key:
            client = XAIGrokClient(api_key, base_url)
            available_models = client.get_available_models()

            col1, col2 = st.columns(2)

            with col1:
                selected_model = st.selectbox(
                    "Grok Model",
                    options=[model["name"] for model in available_models],
                    format_func=lambda x: next(
                        m["display_name"] for m in available_models if m["name"] == x
                    ),
                    help="Select Grok model for analysis",
                )
                st.session_state.selected_grok_model = selected_model

            with col2:
                if selected_model:
                    model_info = next(
                        m for m in available_models if m["name"] == selected_model
                    )
                    st.write("**Model Information:**")
                    st.write(f"• Use Case: {model_info['use_case']}")
                    st.write(
                        f"• Reasoning: {'Yes' if model_info['reasoning'] else 'No'}"
                    )
                    st.write(f"• Context: {model_info['context_length']:,} tokens")
                    st.write(f"• Max Output: {model_info['max_tokens']:,} tokens")

        return api_key, (
            selected_model if "selected_model" in locals() else DEFAULT_GROK_MODEL
        )

    @staticmethod
    def render_xai_document_processor():
        """Enhanced document processor with XAI integration"""
        st.header("📄 Document Processor (XAI Enhanced)")
        st.markdown("Upload documents for AI analysis using XAI's Grok models.")

        # XAI Settings
        api_key, selected_model = XAIIntegratedGUI.render_xai_settings()

        if not api_key:
            st.warning(
                "⚠️ Please configure your XAI API key in the settings above to use Grok models."
            )
            return

        # File upload
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Choose a document to analyze",
            type=["pdf", "docx", "txt", "md", "doc"],
            help="Supported formats: PDF, Word documents, Text files, Markdown",
        )

        if uploaded_file:
            st.success(
                f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)"
            )

            # Analysis options
            st.subheader("Analysis Options")

            col1, col2 = st.columns(2)

            with col1:
                analysis_types = st.multiselect(
                    "Analysis Types",
                    [
                        "Legal Analysis",
                        "Violation Detection",
                        "Entity Extraction",
                        "Document Summary",
                    ],
                    default=["Legal Analysis", "Violation Detection"],
                    help="Select types of analysis to perform",
                )

            with col2:
                custom_temperature = st.slider(
                    "Model Temperature",
                    0.0,
                    1.0,
                    GROK_MODELS_CONFIG.get(selected_model, {}).get("temperature", 0.7),
                    0.1,
                    help="Lower values = more focused, higher values = more creative",
                )

            # Process document
            if st.button("🚀 Analyze with Grok", type="primary"):
                if not analysis_types:
                    st.warning("Please select at least one analysis type")
                    return

                # Read file content
                try:
                    if uploaded_file.type == "text/plain":
                        document_text = str(uploaded_file.read(), "utf-8")
                    else:
                        # For other file types, you'd need additional libraries
                        # For demo purposes, treat as text
                        document_text = str(
                            uploaded_file.read(), "utf-8", errors="ignore"
                        )

                    if len(document_text.strip()) == 0:
                        st.error("Document appears to be empty")
                        return

                    # Initialize XAI client
                    client = XAIGrokClient(api_key)

                    # Perform selected analyses
                    results = {}

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, analysis_type in enumerate(analysis_types):
                        status_text.text(f"Performing {analysis_type}...")
                        progress_bar.progress((i + 1) / len(analysis_types))

                        if analysis_type == "Legal Analysis":
                            result = client.analyze_document(
                                document_text,
                                "legal_analysis",
                                selected_model,
                                temperature=custom_temperature,
                            )
                        elif analysis_type == "Violation Detection":
                            result = client.detect_violations(
                                document_text,
                                selected_model,
                                temperature=custom_temperature,
                            )
                        elif analysis_type == "Entity Extraction":
                            result = client.extract_entities(
                                document_text,
                                selected_model,
                                temperature=custom_temperature,
                            )
                        elif analysis_type == "Document Summary":
                            result = client.summarize_document(
                                document_text,
                                selected_model,
                                temperature=custom_temperature,
                            )

                        results[analysis_type] = result

                    progress_bar.progress(1.0)
                    status_text.text("Analysis complete!")

                    # Display results
                    st.subheader("Analysis Results")

                    for analysis_type, result in results.items():
                        with st.expander(f"📋 {analysis_type}", expanded=True):
                            if result["status"] == "SUCCESS":
                                st.markdown("**Analysis:**")
                                st.write(
                                    result.get("analysis")
                                    or result.get("entities")
                                    or "No content returned"
                                )

                                # Show metadata
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Model Used", result["model_used"])

                                if "token_usage" in result and result["token_usage"]:
                                    usage = result["token_usage"]
                                    col2.metric(
                                        "Tokens Used", usage.get("total_tokens", "N/A")
                                    )
                                    col3.metric(
                                        "Completion Tokens",
                                        usage.get("completion_tokens", "N/A"),
                                    )
                            else:
                                st.error(f"Analysis failed: {result['message']}")

                    # Save results to session state for later use
                    st.session_state.latest_analysis = {
                        "document_name": uploaded_file.name,
                        "model_used": selected_model,
                        "timestamp": datetime.now().isoformat(),
                        "results": results,
                    }

                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    logger.error(f"Document processing error: {e}")

        # Recent analyses
        if hasattr(st.session_state, "latest_analysis"):
            st.subheader("Recent Analysis")
            analysis = st.session_state.latest_analysis

            st.write(f"**Document:** {analysis['document_name']}")
            st.write(f"**Model:** {analysis['model_used']}")
            st.write(f"**Time:** {analysis['timestamp'][:19]}")

            if st.button("📥 Download Results as JSON"):
                json_data = json.dumps(analysis, indent=2)
                st.download_button(
                    label="Download Analysis Results",
                    data=json_data,
                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )


def integrate_xai_with_existing_gui():
    """Integration function to add XAI capabilities to existing GUI"""

    # Add XAI tab to the main navigation
    if "xai_tab_added" not in st.session_state:
        st.session_state.xai_tab_added = True

    # XAI-enhanced document processor
    if st.sidebar.checkbox(
        "🤖 Use XAI/Grok Models", help="Enable direct XAI integration"
    ):
        st.session_state.use_xai = True
        XAIIntegratedGUI.render_xai_document_processor()
    else:
        st.session_state.use_xai = False


# Utility function to check if XAI is properly configured
def check_xai_setup() -> Dict[str, Any]:
    """Check if XAI is properly set up"""
    setup_status = {
        "api_key_configured": bool(
            os.getenv("XAI_API_KEY") or st.session_state.get("xai_api_key")
        ),
        "config_available": "GROK_MODELS_CONFIG" in globals(),
        "models_available": len(GROK_MODELS_CONFIG) > 0,
        "ready": False,
    }

    setup_status["ready"] = all(
        [
            setup_status["api_key_configured"],
            setup_status["config_available"],
            setup_status["models_available"],
        ]
    )

    return setup_status
