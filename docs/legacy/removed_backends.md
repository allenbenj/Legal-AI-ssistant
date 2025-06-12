# Removed Backend Scripts

The project previously included two large FastAPI server implementations:

- `legal_ai_system/gui/main.py`
- `legal_ai_system/scripts/main.py`

These files duplicated functionality and have been removed to avoid confusion.
All desktop features are now consolidated in
`legal_ai_system/gui/legal_ai_pyqt6_integrated.py`.
