"""Compatibility wrapper exposing constants from :mod:`config.constants`.

This module previously duplicated the definitions found in
``config.constants``.  To prevent code drift and simplify the import graph,
it now re-exports those symbols directly from the unified public module.
Existing import paths continue to work while new code should import from
``config.constants`` instead.
"""

from config import constants as _base

# Re-export selected classes and helpers for backwards compatibility
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
