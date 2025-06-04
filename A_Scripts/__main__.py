# legal_ai_system/__main__.py
#!/usr/bin/env python3
"""
Legal AI System - Module Entry Point
====================================
Run with: python -m legal_ai_system
This will typically launch the user interface.
"""

import sys
import os
from pathlib import Path

# Ensure the package root is in sys.path if running with `python legal_ai_system/__main__.py`
# This is usually not needed if running with `python -m legal_ai_system` from one level up.
# However, adding it can make direct script execution more robust.
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent)) # Add the directory containing 'legal_ai_system'

def run_system(): # Renamed from main for clarity
    """Main entry point when the 'legal_ai_system' package is run as a module."""
    # Attempt to import the primary GUI entry point
    try:
        # Assuming streamlit_app.py contains the main function to launch the GUI
        from legal_ai_system.gui.streamlit_app import main_streamlit_entry
        
        print(f"INFO: Launching Legal AI System GUI via: legal_ai_system.gui.streamlit_app.main_streamlit_entry()")
        # Call the main function from streamlit_app.py
        # This function should handle its own setup (logging, etc.) and then run Streamlit.
        main_streamlit_entry() 
        return 0 # Success
    except ImportError as e:
        print(f"ERROR: Failed to import the GUI entry point: {e}", file=sys.stderr)
        print("Please ensure all components are correctly installed and the project structure is intact.", file=sys.stderr)
        print("Try running: pip install -r requirements.txt", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while trying to launch the Legal AI System: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # This block is executed when the script is run directly, e.g. `python legal_ai_system/__main__.py`
    # It's also the entry point for `python -m legal_ai_system`
    
    # Optionally, handle command-line arguments here if you want to launch different parts,
    # e.g., `python -m legal_ai_system api` or `python -m legal_ai_system gui`
    # For now, it defaults to launching the GUI.
    
    # Example: Check for an argument to run API instead of GUI
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        print("INFO: (Not implemented in __main__.py) Request to launch API. Run legal_ai_system/main.py directly for API.")
        # To launch API:
        # from legal_ai_system.main import app as fastapi_app # Assuming main.py has `app`
        # import uvicorn
        # uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
        sys.exit(0) # Or implement API launch
    else:
        sys.exit(run_system())