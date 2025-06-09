from dataclasses import dataclass
from pathlib import Path

@dataclass
class LegalAISettings:
    app_name: str = "Legal AI Assistant"
    version: str = "2.0.0"
    base_dir: Path = Path(__file__).resolve().parent.parent

settings = LegalAISettings()
