# Dependency Audit

The following packages were listed in `requirements.txt` or `pyproject.toml` but are not imported anywhere in the codebase.
They have been removed to keep the dependency list lean.

- aiohttp
- anthropic
- bcrypt
- click
- ffmpeg-python
- flair
- httpx
- lancedb
- pyannote.audio
- pypdf
- python-dotenv
- rich
- sqlite-utils
- strawberry-graphql
- toml
- torch
- tqdm
- typing-extensions
- whisperx

This audit was performed by scanning all Python files for import statements and manually verifying
suspect packages. Dependency management files have been updated accordingly.
