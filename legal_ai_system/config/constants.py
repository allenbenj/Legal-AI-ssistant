"""Compatibility wrapper exposing constants from :mod:`legal_ai_system.constants`.

This module exists for backward compatibility with older import paths.
New code should import directly from :mod:`legal_ai_system.constants` or
:mod:`config.constants`.
"""

from legal_ai_system import constants as _base

# Re-export selected classes and helpers for legacy consumers
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

__all__ = _base.__all__
