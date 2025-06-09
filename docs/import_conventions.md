# Import Conventions

This document defines the standard naming conventions for import statements in the Legal AI System. Following these guidelines ensures a consistent and readable codebase.

## Ordering
1. **Standard library imports**
2. **Third-party imports**
3. **First-party imports** (`legal_ai_system` modules)

Use `isort` with the provided configuration in `pyproject.toml` to automatically enforce this ordering.

## Absolute Imports
Use absolute imports for all modules inside the project. Example:
```python
from legal_ai_system.core.models import LegalDocument
```
Avoid relative imports such as `from ..core import models`.

## Common Aliases
- `import numpy as np`
- `import pandas as pd`
- `import logging`
- `from pathlib import Path`

Additional aliases should be consistent across the project.

## Example
```python
import asyncio
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from legal_ai_system.core.base_agent import BaseAgent
from legal_ai_system.services.service_container import ServiceContainer
```
