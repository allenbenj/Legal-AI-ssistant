# Removed Backend Scripts

The project previously included two large FastAPI server implementations:

- `legal_ai_system/gui/main.py`
- `legal_ai_system/scripts/main.py`

These files duplicated functionality and have been removed to avoid confusion.
All desktop features are now consolidated in
`legal_ai_system/gui/legal_ai_pyqt6_enhanced.py`.

The previous `integrated_gui.py` module and its helper `sections` package have
been deprecated and removed. Use `legal_ai_pyqt6_enhanced.py` as the single
source for the PyQt6 application.
