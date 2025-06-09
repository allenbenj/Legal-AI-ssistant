"""Public re-export of core constants without wildcard imports.

Historically this module relied on ``from ..core.constants import *`` which
caused linter warnings about unused imports.  To keep the public API stable and
avoid those warnings, we import the core constants module under an alias and
explicitly assign the attributes we wish to expose.
"""

from __future__ import annotations

from ..core import constants as _core_constants

# Re-export selected classes and helpers for backwards compatibility
Constants = _core_constants.Constants
TimeConstants = _core_constants.TimeConstants
SizeConstants = _core_constants.SizeConstants
SecurityConstants = _core_constants.SecurityConstants
PerformanceConstants = _core_constants.PerformanceConstants
DocumentConstants = _core_constants.DocumentConstants
LegalConstants = _core_constants.LegalConstants
NetworkConstants = _core_constants.NetworkConstants
EnvironmentConstants = _core_constants.EnvironmentConstants
validate_positive_integer = _core_constants.validate_positive_integer
validate_float_range = _core_constants.validate_float_range
validate_timeout_seconds = _core_constants.validate_timeout_seconds
validate_cache_size = _core_constants.validate_cache_size

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
