"""Compatibility layer that re-exports unified agent configuration helpers.

This thin wrapper imports everything from ``legal_ai_system.core.agent_unified_config``
and exposes the same public API. It avoids wildcard imports to keep linters quiet
while maintaining backwards compatibility for existing imports that reference this
module.
"""

from importlib import import_module

_core_cfg = import_module("..core.agent_unified_config", package=__name__)

__all__ = [name for name in dir(_core_cfg) if not name.startswith("_")]

globals().update({name: getattr(_core_cfg, name) for name in __all__})

