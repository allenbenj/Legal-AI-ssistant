"""Unified constants exported for public consumption.

This module exposes the constant classes and helper validation functions
defined in :mod:`legal_ai_system.core.constants`.  Prior revisions chained
imports through ``legal_ai_system.config.constants`` which duplicated code and
confused static analyzers.  The constants are now loaded directly from the core
implementation to provide a single authoritative source.
"""

from legal_ai_system.core import constants as _base


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
