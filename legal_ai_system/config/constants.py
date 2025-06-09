"""Re-export constants from :mod:`legal_ai_system.core.constants`.

This module previously relied on ``from ..core.constants import *`` which
triggered linter warnings about unused wildcard imports and manipulation of
``__all__``.  To maintain a stable public API while avoiding those warnings we
explicitly import and re-export the symbols defined in ``core.constants``.
"""

from ..core import constants as core_constants

Constants = core_constants.Constants
TimeConstants = core_constants.TimeConstants
SizeConstants = core_constants.SizeConstants
SecurityConstants = core_constants.SecurityConstants
PerformanceConstants = core_constants.PerformanceConstants
DocumentConstants = core_constants.DocumentConstants
LegalConstants = core_constants.LegalConstants
NetworkConstants = core_constants.NetworkConstants
EnvironmentConstants = core_constants.EnvironmentConstants
validate_positive_integer = core_constants.validate_positive_integer
validate_float_range = core_constants.validate_float_range
validate_timeout_seconds = core_constants.validate_timeout_seconds
validate_cache_size = core_constants.validate_cache_size

__all__ = core_constants.__all__
