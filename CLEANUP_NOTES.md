# Cleanup Notes

The following items were identified for future cleanup and consolidation:

- `legal_ai_system/scripts` contains many helper scripts and platform-specific
  start commands (`*.sh`, `*.bat`, `*.ps1`). Review which entry points are still
  required and consider removing outdated duplicates such as `main2.py` and
  `main_gui_backup.py`.
- Temporary data directories under `legal_ai_system/data` are currently empty
  and should remain excluded from version control.
- Test file `test_consolidation_success_copy.py` was renamed from a
  copy left with spaces in its filename.
- Removed an empty `__int__.py` file that served no purpose.

These notes summarize items to revisit when continuing the project
organization work.
- Created `archive_backup_2025-05-24/` and `archive_legacy_components/` to hold
  obsolete scripts and backups referenced by tests.
- Moved `main2.py`, `main_gui_backup.py`, `streamlit_app_backup.py`, and
  `start_with_path_fix.py` into `archive_legacy_components/`.
- Archived the larger `scripts/main_gui.py` in favor of the simpler
  PyQt6 version under `legal_ai_system/gui/`.
- `.gitkeep` files added so empty directories persist in version control.
- Added a project `.gitignore` to ignore logs, coverage outputs, and cache
  directories generated during tests.
