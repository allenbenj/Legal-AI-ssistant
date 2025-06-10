import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types

for mod in [
    "legal_ai_system.integration_ready",
    "legal_ai_system.integration_ready.vector_store_enhanced",
    "faiss",
]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

sys.modules[
    "legal_ai_system.integration_ready.vector_store_enhanced"
].MemoryStore = object
