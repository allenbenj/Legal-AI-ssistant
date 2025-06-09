# legal_ai_system/core/constants.py
#Legal AI System Constants - Centralized Configuration Values
#==========================================================
#This module contains all configuration constants used throughout the Legal AI System.
#All values include proper units and documentation to eliminate magic numbers.

#Following the DRY principle and 12-factor app methodology for maintainable,
#scalable, and self-documenting code.

from typing import Final
from enum import Enum

# =================== TIME CONSTANTS ===================

class TimeConstants:
    """Time-related constants with explicit units"""
    
    # Service timeouts (in seconds)
    DEFAULT_SERVICE_TIMEOUT_SECONDS: Final[int] = 300  # 5 minutes
    MAX_SERVICE_TIMEOUT_SECONDS: Final[int] = 600  # 10 minutes  
    MIN_SERVICE_TIMEOUT_SECONDS: Final[int] = 1    # 1 second
    
    # Service timeouts (in milliseconds) 
    DEFAULT_SERVICE_TIMEOUT_MS: Final[int] = 300_000  # 5 minutes in ms
    MAX_SERVICE_TIMEOUT_MS: Final[int] = 600_000      # 10 minutes in ms
    MIN_SERVICE_TIMEOUT_MS: Final[int] = 120_000      # 2 minutes in ms
    
    # Health monitoring intervals (in seconds)
    HEALTH_CHECK_INTERVAL_SECONDS: Final[float] = 30.0
    HEALTH_CHECK_TIMEOUT_SECONDS: Final[int] = 5
    
    # Retry and backoff (in seconds)
    DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0
    MAX_RETRY_DELAY_SECONDS: Final[float] = 60.0
    EXPONENTIAL_BACKOFF_MULTIPLIER: Final[float] = 2.0
    
    # Session and authentication timeouts
    SESSION_TIMEOUT_HOURS: Final[int] = 8
    SESSION_TIMEOUT_MINUTES: Final[int] = 30  # For short sessions
    ACCOUNT_LOCKOUT_DURATION_MINUTES: Final[int] = 30
    
    # Cache TTL
    DEFAULT_CACHE_TTL_HOURS: Final[int] = 24
    SHORT_CACHE_TTL_MINUTES: Final[int] = 15
    LONG_CACHE_TTL_DAYS: Final[int] = 7

# =================== SIZE AND MEMORY CONSTANTS ===================

class SizeConstants:
    """Size and memory-related constants with explicit units"""
    
    # Memory sizes (in bytes)
    KILOBYTE: Final[int] = 1024
    MEGABYTE: Final[int] = 1024 * KILOBYTE
    GIGABYTE: Final[int] = 1024 * MEGABYTE
    
    # JSON and payload limits (in bytes)
    MAX_JSON_PAYLOAD_BYTES: Final[int] = 1 * MEGABYTE  # 1 MB
    MAX_TEXT_INPUT_BYTES: Final[int] = 50_000          # ~50 KB
    MAX_FILE_SIZE_BYTES: Final[int] = 100 * MEGABYTE   # 100 MB
    
    # Cache sizes (counts and bytes)
    DEFAULT_CACHE_SIZE_ITEMS: Final[int] = 1000
    LARGE_CACHE_SIZE_ITEMS: Final[int] = 10_000
    MAX_CACHE_SIZE_MB: Final[int] = 500  # 500 MB
    
    # Service call history limits (item counts)
    SERVICE_CALL_HISTORY_LIMIT: Final[int] = 1000
    MAX_LOG_ENTRIES: Final[int] = 10_000
    
    # Text processing limits (character counts)
    MAX_CONTEXT_TOKENS: Final[int] = 32_000      # Standard LLM context window
    LARGE_CONTEXT_TOKENS: Final[int] = 128_000   # Large context models
    SUMMARY_MAX_LENGTH_CHARS: Final[int] = 500
    AUTO_SUMMARIZE_THRESHOLD_CHARS: Final[int] = 5000
    
    # Chunk sizes for document processing (character counts)
    DEFAULT_CHUNK_SIZE_CHARS: Final[int] = 1000
    CHUNK_OVERLAP_CHARS: Final[int] = 200
    LARGE_CHUNK_SIZE_CHARS: Final[int] = 4000

# =================== SECURITY CONSTANTS ===================

class SecurityConstants:
    """Security-related constants with proper validation limits"""
    
    # Password requirements
    MIN_PASSWORD_LENGTH_CHARS: Final[int] = 8
    MAX_PASSWORD_LENGTH_CHARS: Final[int] = 128
    
    # Cryptographic parameters
    PBKDF2_ITERATIONS: Final[int] = 100_000  # OWASP recommended minimum
    SALT_LENGTH_BYTES: Final[int] = 32       # 256 bits
    ENCRYPTION_KEY_LENGTH_BYTES: Final[int] = 32  # 256 bits
    
    # Authentication attempts and limits
    MAX_FAILED_LOGIN_ATTEMPTS: Final[int] = 5
    MAX_CONCURRENT_SESSIONS: Final[int] = 10
    
    # Rate limiting (requests per time period)
    RATE_LIMIT_PER_MINUTE: Final[int] = 100
    RATE_LIMIT_PER_HOUR: Final[int] = 1000
    
    # Token and session management
    SESSION_TOKEN_LENGTH_BYTES: Final[int] = 32
    CSRF_TOKEN_LENGTH_BYTES: Final[int] = 32

# =================== PERFORMANCE CONSTANTS ===================

class PerformanceConstants:
    """Performance tuning constants with operational context"""
    
    # Retry logic
    MAX_RETRY_ATTEMPTS: Final[int] = 3
    MAX_CIRCUIT_BREAKER_FAILURES: Final[int] = 5
    
    # Concurrency limits
    MAX_CONCURRENT_DOCUMENTS: Final[int] = 3
    MAX_CONCURRENT_REQUESTS: Final[int] = 10
    DEFAULT_BATCH_SIZE: Final[int] = 10
    LARGE_BATCH_SIZE: Final[int] = 100
    
    # Processing thresholds
    SIMILARITY_THRESHOLD_DEFAULT: Final[float] = 0.8
    CONFIDENCE_THRESHOLD_HIGH: Final[float] = 0.9
    CONFIDENCE_THRESHOLD_MEDIUM: Final[float] = 0.7
    CONFIDENCE_THRESHOLD_LOW: Final[float] = 0.5
    
    # ML and optimization parameters
    MIN_TRAINING_SAMPLES: Final[int] = 50
    MAX_OPTIMIZATION_AGE_HOURS: Final[int] = 24
    EMBEDDING_DIMENSION: Final[int] = 384  # Common embedding size

# =================== DOCUMENT PROCESSING CONSTANTS ===================

class DocumentConstants:
    """Document processing and format constants"""
    
    # File format limits
    MAX_PDF_PAGES: Final[int] = 1000
    MAX_DOCUMENT_SIZE_MB: Final[int] = 100
    
    # OCR and text extraction
    OCR_DPI: Final[int] = 300
    MAX_OCR_PAGES: Final[int] = 50
    
    # Auto-tagging confidence levels
    AUTO_TAG_CONFIDENCE_THRESHOLD: Final[float] = 0.7
    AUTO_APPROVE_THRESHOLD: Final[float] = 0.9
    MANUAL_REVIEW_THRESHOLD: Final[float] = 0.5

# =================== LEGAL DOMAIN CONSTANTS ===================

class LegalConstants:
    """Legal domain-specific constants"""
    
    # Citation and case processing
    MAX_CASE_REFERENCES: Final[int] = 100
    MAX_STATUTE_REFERENCES: Final[int] = 50
    
    # Entity limits
    MAX_ENTITIES_PER_DOCUMENT: Final[int] = 1000
    MAX_RELATIONSHIPS_PER_ENTITY: Final[int] = 100
    
    # Violation detection thresholds
    VIOLATION_CONFIDENCE_THRESHOLD: Final[float] = 0.8
    CRITICAL_VIOLATION_THRESHOLD: Final[float] = 0.95

# =================== NETWORK AND API CONSTANTS ===================

class NetworkConstants:
    """Network and API-related constants"""
    
    # HTTP timeouts (in seconds)
    API_REQUEST_TIMEOUT_SECONDS: Final[int] = 30
    API_CONNECTION_TIMEOUT_SECONDS: Final[int] = 10
    
    # Retry and backoff for network requests
    MAX_API_RETRIES: Final[int] = 3
    API_RETRY_DELAY_SECONDS: Final[float] = 1.0
    
    # Connection limits
    MAX_CONNECTIONS_PER_HOST: Final[int] = 10
    MAX_TOTAL_CONNECTIONS: Final[int] = 100

# =================== ENVIRONMENT-SPECIFIC OVERRIDES ===================

class EnvironmentConstants:
    """Environment-specific constants that can be overridden"""
    
    # Development vs Production differences
    DEV_CACHE_SIZE: Final[int] = 100
    PROD_CACHE_SIZE: Final[int] = 10_000
    
    DEV_MAX_REQUESTS_PER_MINUTE: Final[int] = 1000
    PROD_MAX_REQUESTS_PER_MINUTE: Final[int] = 100
    
    # Testing constants
    TEST_TIMEOUT_SECONDS: Final[int] = 5
    TEST_BATCH_SIZE: Final[int] = 5

# =================== VALIDATION HELPERS ===================

def validate_positive_integer(value: int, name: str) -> int:
    """Validate that a value is a positive integer"""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value

def validate_float_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """Validate that a float value is within a specified range"""
    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return float(value)

def validate_timeout_seconds(value: int) -> int:
    """Validate timeout value is within reasonable bounds"""
    return validate_positive_integer(value, "timeout_seconds")

def validate_cache_size(value: int) -> int:
    """Validate cache size is within reasonable bounds"""
    if value < 10 or value > 1_000_000:
        raise ValueError(f"Cache size must be between 10 and 1,000,000, got {value}")
    return value

# =================== CONSTANT GROUPS FOR EASY ACCESS ===================

class Constants:
    """Main constants class providing organized access to all constant groups"""
    
    Time = TimeConstants
    Size = SizeConstants  
    Security = SecurityConstants
    Performance = PerformanceConstants
    Document = DocumentConstants
    Legal = LegalConstants
    Network = NetworkConstants
    Environment = EnvironmentConstants

# Export for convenient importing
__all__ = [
    'Constants',
    'TimeConstants',
    'SizeConstants', 
    'SecurityConstants',
    'PerformanceConstants',
    'DocumentConstants',
    'LegalConstants',
    'NetworkConstants',
    'EnvironmentConstants',
    'validate_positive_integer',
    'validate_float_range',
    'validate_timeout_seconds',
    'validate_cache_size'
]