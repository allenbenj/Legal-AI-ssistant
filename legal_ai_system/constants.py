"""Unified constants exported for public consumption.

This module provides a single public interface for all framework constants.
The actual values live in :mod:`legal_ai_system.core.constants`. Importing
from this module avoids exposing the internal ``core`` package structure and
prevents long re-export chains.
"""

from .core import constants as _core

# Re-export public constant classes and helper validators
Constants = _core.Constants
TimeConstants = _core.TimeConstants
SizeConstants = _core.SizeConstants
SecurityConstants = _core.SecurityConstants
PerformanceConstants = _core.PerformanceConstants
DocumentConstants = _core.DocumentConstants
LegalConstants = _core.LegalConstants
NetworkConstants = _core.NetworkConstants
EnvironmentConstants = _core.EnvironmentConstants
validate_positive_integer = _core.validate_positive_integer
validate_float_range = _core.validate_float_range
validate_timeout_seconds = _core.validate_timeout_seconds
validate_cache_size = _core.validate_cache_size

__all__ = _core.__all__
