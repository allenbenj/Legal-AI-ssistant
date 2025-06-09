"""Public re-export of constants from :mod:`legal_ai_system`.

This module exposes the constant classes and helper validation functions found
in :mod:`legal_ai_system.config.constants` so that external packages can import
them from a stable location.  The previous implementation imported each symbol
individually which confused linters that expected those names to be used
locally.  Instead we import the target module as ``_base`` and re-export the
attributes we wish to expose.  This avoids linter warnings while maintaining a
clear mapping to the original implementation.
"""

from legal_ai_system.config import constants as _base


# Re-export the public constants classes and validators from ``_base``.
Constants = _base.Constants
TimeConstants = _base.TimeConstants
SizeConstants = _base.SizeConstants
SecurityConstants = _base.SecurityConstants
PerformanceConstants = _base.PerformanceConstants
DocumentConstants = _base.DocumentConstants
LegalConstants = _base.LegalConstants
NetworkConstants = _base.NetworkConstants
EnvironmentConstants = _base.EnvironmentConstants
validate_positive_integer = _base.validate_positive_integer
validate_float_range = _base.validate_float_range
validate_timeout_seconds = _base.validate_timeout_seconds
validate_cache_size = _base.validate_cache_size

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
