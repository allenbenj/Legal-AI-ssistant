"""Re-export constants from :mod:`legal_ai_system.core.constants`.

Historically this module used ``from ..core.constants import *`` which
triggered linter warnings about wildcard imports and manipulation of
``__all__``.  To maintain a stable public API while avoiding those warnings we
explicitly import the required symbols and define ``__all__`` here.
"""

from ..core.constants import (
    Constants,
    TimeConstants,
    SizeConstants,
    SecurityConstants,
    PerformanceConstants,
    DocumentConstants,
    LegalConstants,
    NetworkConstants,
    EnvironmentConstants,
    validate_positive_integer,
    validate_float_range,
    validate_timeout_seconds,
    validate_cache_size,
)

__all__ = [
    "Constants",
    "TimeConstants",
    "SizeConstants",
    "SecurityConstants",
    "PerformanceConstants",
    "DocumentConstants",
    "LegalConstants",
    "NetworkConstants",
    "EnvironmentConstants",
    "validate_positive_integer",
    "validate_float_range",
    "validate_timeout_seconds",
    "validate_cache_size",
]
