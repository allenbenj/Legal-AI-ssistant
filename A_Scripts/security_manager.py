# legal_ai_system/core/security_manager.py
"""Security Manager - Critical Security Fixes.

Addresses identified vulnerabilities: input validation, PII protection, authentication.
Provides comprehensive security services for the Legal AI System including encryption,
authentication, authorization, audit logging, and PII detection.
"""

import re
import hashlib
import secrets
# import logging # Replaced by detailed_logging
from typing import Any, Dict, List, Optional, Set, Union # Added Set, Union
from dataclasses import dataclass, field # Added field
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
from enum import Enum

# Use detailed_logging
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
# Import constants from the config module
from ..config.constants import Constants
from ..persistence.repositories.user_repository import UserRepository 
# Import exceptions for specific error types if needed
# from .unified_exceptions import SecurityError # Example

# Initialize logger for this module
security_logger_module = get_detailed_logger("SecurityManagerModule", LogCategory.SECURITY)


try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTOGRAPHY_AVAILABLE = True
    security_logger_module.info("Cryptography library loaded successfully.")
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    security_logger_module.warning("Cryptography library not available - encryption features will be disabled.")


class AccessLevel(Enum):
    """Access levels for role-based access control."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class User:
    """User entity with authentication and authorization."""
    user_id: str
    username: str
    email: str
    password_hash: str # Hex encoded
    salt: str # Hex encoded
    access_level: AccessLevel 
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())) # Added entry_id
    timestamp: datetime
    user_id: str # Should be 'unknown' if action is pre-auth or system
    action: str
    resource: Optional[str] = None # Made optional
    details: Dict[str, Any] = field(default_factory=dict) # Ensure default is factory
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: str = "success" # Added status: success, failure, attempt
    
class PIIDetector:
    """Detects and manages Personally Identifiable Information."""
    
    def __init__(self):
        self.logger = get_detailed_logger("PIIDetector", LogCategory.SECURITY)
        # Common PII patterns for legal documents
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3,4}\d{4}\b'), # Adjusted for 3 or 4 groups
            'address': re.compile(r'\b\d+\s+[A-Za-z0-9\s.,#-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl|Square|Sq|Terrace|Ter)\b', re.IGNORECASE), # Improved address
            'date_of_birth': re.compile(r'\b(?:(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:0?[1-9]|[12]\d|3[01]))[/-](?:19|20)\d{2}\b', re.IGNORECASE), # More robust DOB
            'case_number': re.compile(r'\b(?:Case|Docket|File|Matter|Index)\s*(?:No\.?|Number|Num\.?)?\s*:?\s*([A-Za-z0-9-]+(?:/[A-Za-z0-9-]+)*)\b', re.IGNORECASE) # Improved case number
        }
        self.logger.info("PIIDetector initialized with patterns.", parameters={'num_patterns': len(self.patterns)})

    @detailed_log_function(LogCategory.SECURITY)
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text and return findings."""
        findings: Dict[str, List[str]] = {} # Initialize as empty dict
        self.logger.debug("Starting PII detection.", parameters={'text_length': len(text)})
        
        for pii_type, pattern in self.patterns.items():
            try:
                # findall might return tuples if there are capture groups, ensure we get the full match
                matches = [match[0] if isinstance(match, tuple) else match for match in pattern.findall(text)]
                if matches:
                    if pii_type not in findings:
                        findings[pii_type] = []
                    findings[pii_type].extend(matches) # Use extend for list of matches
                    self.logger.trace(f"PII type '{pii_type}' detected.", parameters={'count': len(matches), 'first_match_preview': matches[0][:50] if matches else None})
            except Exception as e:
                self.logger.error(f"Error during PII detection for type '{pii_type}'.", exception=e)

        if findings:
            self.logger.info("PII detection complete. Findings present.", parameters={'pii_types_found': list(findings.keys())})
        else:
            self.logger.debug("PII detection complete. No findings.")
        return findings
    
    @detailed_log_function(LogCategory.SECURITY)
    def anonymize_text(self, text: str, replacement_char: str = "█") -> str:
        """Anonymize PII in text while preserving structure."""
        self.logger.debug("Starting PII anonymization.", parameters={'text_length': len(text)})
        anonymized_text = text # Create a new variable
        
        for pii_type, pattern in self.patterns.items():
            if pii_type == 'case_number': # Preserve case numbers for legal context
                self.logger.trace("Skipping anonymization for case_number.")
                continue
            
            try:
                # Use re.sub with a lambda to ensure replacement length matches original match length
                anonymized_text = pattern.sub(lambda m: replacement_char * len(m.group(0)), anonymized_text)
            except Exception as e:
                self.logger.error(f"Error during PII anonymization for type '{pii_type}'.", exception=e)
        
        self.logger.info("PII anonymization complete.")
        return anonymized_text

class InputValidator:
    """Validates and sanitizes all inputs to prevent injection attacks."""
    logger = get_detailed_logger("InputValidator", LogCategory.SECURITY)

    @staticmethod
    @detailed_log_function(LogCategory.SECURITY)
    def validate_json(json_string: str, max_size: int = Constants.Size.MAX_JSON_PAYLOAD_BYTES) -> Dict[str, Any]:
        """Safely validate and parse JSON with size limits."""
        InputValidator.logger.debug("Validating JSON input.", parameters={'string_length': len(json_string), 'max_size_bytes': max_size})
        if len(json_string.encode('utf-8')) > max_size: # Check byte size for UTF-8
            msg = f"JSON payload too large: {len(json_string.encode('utf-8'))} bytes (max: {max_size})"
            InputValidator.logger.error(msg)
            raise ValueError(msg)
        
        try:
            # Basic cleaning - more robust sanitization might be needed depending on context
            cleaned = re.sub(r'<script[^>]*>.*?</script>', '', json_string, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'javascript\s*:', '', cleaned, flags=re.IGNORECASE) # Added \s*
            
            parsed = json.loads(cleaned)
            
            if not isinstance(parsed, (dict, list)):
                msg = "Invalid JSON structure: Must be an object or array."
                InputValidator.logger.error(msg)
                raise ValueError(msg)
            
            InputValidator.logger.info("JSON validation successful.")
            return parsed
            
        except json.JSONDecodeError as e:
            InputValidator.logger.error("JSON decoding failed.", exception=e)
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e: # Catch other potential errors
            InputValidator.logger.error("Unexpected error during JSON validation.", exception=e)
            raise ValueError(f"JSON validation error: {str(e)}")
    
    @staticmethod
    @detailed_log_function(LogCategory.SECURITY)
    def validate_file_path(file_path_str: str, allowed_dirs: List[Union[str, Path]]) -> Path: # Renamed param
        """Validate file path to prevent directory traversal."""
        InputValidator.logger.debug("Validating file path.", parameters={'path_str': file_path_str, 'num_allowed_dirs': len(allowed_dirs)})
        try:
            # Resolve the path to get its absolute form and normalize (e.g., resolve symlinks)
            resolved_path = Path(file_path_str).resolve()

            # Check if path is within any of the allowed directories
            is_allowed = False
            for allowed_dir_str in allowed_dirs:
                resolved_allowed_dir = Path(allowed_dir_str).resolve()
                if resolved_allowed_dir.is_dir(): # Ensure allowed_dir is actually a directory
                    # Check if resolved_path is a subpath of resolved_allowed_dir
                    if resolved_path.is_relative_to(resolved_allowed_dir): # Python 3.9+
                        is_allowed = True
                        break
                else:
                    InputValidator.logger.warning(f"Configured allowed directory is not a directory.", parameters={'allowed_dir': str(resolved_allowed_dir)})
            
            if not is_allowed:
                msg = f"Path '{str(resolved_path)}' is not within allowed directories."
                InputValidator.logger.error(msg, parameters={'allowed_dirs': [str(d) for d in allowed_dirs]})
                raise ValueError(msg)
            
            # Further check against common traversal patterns for robustness, though resolve() helps
            if ".." in str(resolved_path.relative_to(Path(allowed_dirs[0]).resolve())): # Check relative path after confirming it's inside
                 msg = "Path traversal components ('..') detected within allowed structure."
                 InputValidator.logger.error(msg)
                 raise ValueError(msg)

            InputValidator.logger.info("File path validation successful.", parameters={'resolved_path': str(resolved_path)})
            return resolved_path
            
        except Exception as e: # Catch more general errors during path operations
            InputValidator.logger.error("File path validation failed.", parameters={'path_str': file_path_str}, exception=e)
            raise ValueError(f"Invalid or disallowed file path: {str(e)}")
    
    @staticmethod
    @detailed_log_function(LogCategory.SECURITY)
    def sanitize_text(text: str, max_length: int = Constants.Size.MAX_TEXT_INPUT_BYTES) -> str:
        """Sanitize text input for safe processing."""
        InputValidator.logger.debug("Sanitizing text input.", parameters={'original_length': len(text), 'max_length': max_length})
        if len(text) > max_length:
            text = text[:max_length]
            InputValidator.logger.warning("Text input truncated due to max length.", parameters={'new_length': len(text)})
        
        # More comprehensive sanitization (example using a basic allowlist approach for HTML-like tags)
        # This is a placeholder; real HTML sanitization needs a proper library like Bleach.
        # For non-HTML text, the primary concern is often control characters or specific injection strings.
        
        # Remove null bytes
        text = text.replace('\x00', '')

        # Basic script tag removal (can be bypassed, use dedicated library for HTML)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE) # Remove onX attributes

        # Normalize whitespace and strip
        sanitized_text = ' '.join(text.split()).strip()
        InputValidator.logger.info("Text sanitization complete.", parameters={'final_length': len(sanitized_text)})
        return sanitized_text

class EncryptionManager:
    """Manages encryption for sensitive data storage."""
    logger = get_detailed_logger("EncryptionManager", LogCategory.SECURITY)

    def __init__(self, password: str, salt: Optional[bytes] = None): # Allow providing salt
        if not CRYPTOGRAPHY_AVAILABLE:
            EncryptionManager.logger.critical("Cryptography library not available - ENCRYPTION IS DISABLED. THIS IS INSECURE.")
            self.cipher = None
            return
            
        self.password = password.encode('utf-8') # Ensure password is bytes
        # Use a securely stored salt in production, not hardcoded.
        # For this refactor, we'll use a hardcoded one but log a warning.
        self.salt = salt if salt else b'a_very_secure_salt_for_legal_ai' # Example fixed salt
        if not salt:
            EncryptionManager.logger.warning("Using a fixed salt for encryption. In production, use a unique, securely stored salt per key or a global one from secure config.")

        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=Constants.Security.ENCRYPTION_KEY_LENGTH_BYTES, # 32 bytes for AES-256
                salt=self.salt,
                iterations=Constants.Security.PBKDF2_ITERATIONS,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.password))
            self.cipher = Fernet(key)
            EncryptionManager.logger.info("EncryptionManager initialized successfully.")
        except Exception as e:
            EncryptionManager.logger.critical("Failed to initialize Fernet cipher.", exception=e)
            self.cipher = None # Ensure cipher is None if init fails
            raise RuntimeError("EncryptionManager could not be initialized.") from e # Fail hard if crypto setup fails
    
    @detailed_log_function(LogCategory.SECURITY)
    def encrypt(self, data_str: str) -> str: # Changed param name for clarity
        """Encrypt sensitive string data."""
        if not self.cipher:
            EncryptionManager.logger.error("Encryption attempted but cipher is not available. Returning plaintext. THIS IS INSECURE.")
            return data_str # Insecure fallback
            
        try:
            encrypted_bytes = self.cipher.encrypt(data_str.encode('utf-8'))
            # Return as string (base64 is safe for text fields)
            return encrypted_bytes.decode('utf-8') 
        except Exception as e:
            EncryptionManager.logger.error("Encryption failed.", exception=e)
            raise RuntimeError("Data encryption failed.") from e # Propagate error
    
    @detailed_log_function(LogCategory.SECURITY)
    def decrypt(self, encrypted_data_str: str) -> str: # Changed param name
        """Decrypt sensitive string data."""
        if not self.cipher:
            EncryptionManager.logger.error("Decryption attempted but cipher is not available. Returning encrypted data. THIS IS INSECURE.")
            return encrypted_data_str # Insecure fallback
            
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_data_str.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except Exception as e: # Catch specific cryptography errors if possible, e.g., InvalidToken
            EncryptionManager.logger.error("Decryption failed. Data might be corrupted or key mismatch.", exception=e)
            raise RuntimeError("Data decryption failed. Possible data corruption or key mismatch.") from e

class AuthenticationManager:
    logger = get_detailed_logger("AuthenticationManager", LogCategory.SECURITY)
    
    def __init__(self, user_repository: Optional[UserRepository] = None): # Inject UserRepository
        self.user_repository = user_repository
        self.config = {} # Placeholder for config access
        self._user_cache: Dict[str, User] = {}
        self._session_cache: Dict[str, Dict[str, Any]] = {} 
        self.audit_log_buffer: List[AuditLogEntry] = [] 
        self.max_failed_attempts = Constants.Security.MAX_FAILED_LOGIN_ATTEMPTS
        self.lockout_duration = timedelta(minutes=Constants.Time.ACCOUNT_LOCKOUT_DURATION_MINUTES)
        
        if not self.user_repository:
            AuthenticationManager.logger.warning("UserRepository not provided. AuthenticationManager will operate in a limited, non-persistent (in-memory) mode.")
        else:
            AuthenticationManager.logger.info("AuthenticationManager initialized with UserRepository for persistence.")
            # Consider pre-loading some active users or sessions into cache if needed.

    @detailed_log_function(LogCategory.SECURITY)
    async def create_user_async(self, username: str, email: str, password: str,
                               access_level: AccessLevel = AccessLevel.READ) -> str:
        # ... (password validation) ...
        if len(password) < Constants.Security.MIN_PASSWORD_LENGTH_CHARS: raise ValueError("...")

        salt_bytes = secrets.token_bytes(Constants.Security.SALT_LENGTH_BYTES)
        password_hash_bytes = hashlib.pbkdf2_hmac(
            'sha256', password.encode('utf-8'), salt_bytes, 
            Constants.Security.PBKDF2_ITERATIONS, dklen=Constants.Security.ENCRYPTION_KEY_LENGTH_BYTES
        )
        user_id = str(uuid.uuid4())
        user = User(
            user_id=user_id, username=username, email=email,
            password_hash=password_hash_bytes.hex(), salt=salt_bytes.hex(),
            access_level=access_level, created_at=datetime.now(timezone.utc), is_active=True
        )
        
        if self.user_repository:
            existing_user = await self.user_repository.get_user_by_username_async(username)
            if existing_user:
                raise ValueError(f"Username '{username}' already exists.")
            await self.user_repository.add_user_async(user)
            self._user_cache[user.user_id] = user # Update cache
        else: # In-memory fallback
            if username in [u.username for u in self._user_cache.values()]: raise ValueError(f"Username '{username}' already exists (in-memory).")
            self._user_cache[user_id] = user

        await self._log_audit_async(user_id, "user_created", details={'username': username, 'access_level': access_level.value})
        return user_id

    @detailed_log_function(LogCategory.SECURITY)
    async def authenticate_async(self, username: str, password: str, 
                                ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Optional[str]:
        log_details = {'username': username, 'ip': ip_address, 'ua': user_agent}
        user_obj: Optional[User] = self._user_cache.get(username, None) # Try username as cache key first
        if not user_obj or user_obj.username != username : # If not found by username, iterate (less efficient)
            user_obj = next((u for u in self._user_cache.values() if u.username == username), None)

        if not user_obj and self.user_repository:
            user_obj = await self.user_repository.get_user_by_username_async(username)
            if user_obj: self._user_cache[user_obj.user_id] = user_obj # Cache it

        if not user_obj:
            await self._log_audit_async("unknown", "login_failed_user_not_found", details=log_details, status="failure")
            return None
        
        # ... (active check, lockout check, password verification as before) ...
        # On successful auth:
        # user_obj.failed_attempts = 0; user_obj.locked_until = None; user_obj.last_login = now_utc
        # if self.user_repository:
        #    await self.user_repository.update_user_auth_status_async(...)

        # Session creation
        session_token = secrets.token_urlsafe(Constants.Security.SESSION_TOKEN_LENGTH_BYTES)
        session_data = {
            'user_id': user_obj.user_id, 'created_at': datetime.now(timezone.utc),
            'ip_address': ip_address, 'user_agent': user_agent,
            'expires_at': datetime.now(timezone.utc) + timedelta(hours=Constants.Time.SESSION_TIMEOUT_HOURS)
        }
        if self.user_repository:
            await self.user_repository.create_session_async(session_token, session_data)
        self._session_cache[session_token] = session_data # Always cache session

        await self._log_audit_async(user_obj.user_id, "login_successful", details=log_details)
        return session_token

    @detailed_log_function(LogCategory.SECURITY)
    async def validate_session_async(self, session_token: str) -> Optional[User]:
        session_data = self._session_cache.get(session_token)
        now = datetime.now(timezone.utc)

        if session_data and now <= session_data['expires_at']: # Check memory cache first
            user = self._user_cache.get(session_data['user_id'])
            if user and user.is_active: return user
            # If user not in cache, fetch from repo
            if self.user_repository:
                user = await self.user_repository.get_user_by_id_async(session_data['user_id'])
                if user and user.is_active: self._user_cache[user.user_id] = user; return user
            return None # User not found or inactive in repo
        elif session_data and now > session_data['expires_at']: # Expired in memory cache
            del self._session_cache[session_token]
            if self.user_repository: await self.user_repository.invalidate_session_async(session_token) # Invalidate in DB too
            return None

        # If not in memory cache or expired there, check persistent store
        if self.user_repository:
            persisted_session_data = await self.user_repository.get_session_async(session_token)
            if persisted_session_data and now <= persisted_session_data['expires_at']:
                self._session_cache[session_token] = persisted_session_data # Update memory cache
                user = self._user_cache.get(persisted_session_data['user_id'])
                if not user: 
                    user = await self.user_repository.get_user_by_id_async(persisted_session_data['user_id'])
                    if user: self._user_cache[user.user_id] = user
                if user and user.is_active: return user
        return None
    
    async def _log_audit_async(self, user_id: str, action: str, resource: Optional[str] = None, 
                               details: Optional[Dict[str, Any]] = None, 
                               ip_address: Optional[str] = None, user_agent: Optional[str] = None, status: str = "success"):
        # ... (buffering logic as before) ...
        # This method now needs to be async if _flush_audit_logs_async is async
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc), user_id=user_id, action=action,
            resource=resource, details=details or {}, ip_address=ip_address,
            user_agent=user_agent, status=status
        )
        self.audit_log_buffer.append(entry)
        if len(self.audit_log_buffer) >= self.config.get("audit_log_batch_size", 10): # Configurable batch size
            await self._flush_audit_logs_async()

    async def _flush_audit_logs_async(self): # Make sure this is called on shutdown too
        if not self.audit_log_buffer: return
        logs_to_write = self.audit_log_buffer[:] # Copy
        self.audit_log_buffer.clear()

        if self.user_repository:
            try:
                await self.user_repository.add_audit_logs_batch_async(logs_to_write)
            except Exception as e: # Catch specific DatabaseError
                AuthenticationManager.logger.error("Failed to flush audit logs to repository. Logs will be lost if app shuts down.", exception=e,
                                                 parameters={'num_logs_lost_if_not_retried': len(logs_to_write)})
                # Consider re-adding to buffer or writing to a fallback file log for critical audit trails
                # self.audit_log_buffer.extend(logs_to_write) # Re-add for next attempt (careful with infinite loops)
        else:
            AuthenticationManager.logger.warning("UserRepository not available. Audit logs buffered in memory only.", 
                                               parameters={'buffered_count': len(logs_to_write)})
    
    # Ensure a method to flush audit logs on application shutdown is called from SecurityManager.shutdown()
    async def flush_pending_audit_logs(self):
        await self._flush_audit_logs_async()


class SecurityManager:
    """Central security manager coordinating all security components."""
    logger = get_detailed_logger("SecurityManager", LogCategory.SECURITY)

    def __init__(self, encryption_password: str, allowed_directories: List[Union[str, Path]]): # Type hint for list elements
        self.pii_detector = PIIDetector()
        self.input_validator = InputValidator() # Static methods, but instance can hold config if needed
        self.encryption_manager = EncryptionManager(encryption_password)
        self.auth_manager = AuthenticationManager()
        
        # Ensure allowed_directories are resolved Paths
        self.allowed_directories = [Path(d).resolve() for d in allowed_directories]
        
        # Security configuration (could be loaded from ConfigurationManager)
        self.enable_pii_detection = True # Default
        self.enable_encryption_at_rest = True # Default for sensitive data
        self.require_authentication_for_all_apis = True # Default
        
        SecurityManager.logger.info(
            "SecurityManager initialized.", 
            parameters={
                'pii_detection': self.enable_pii_detection, 
                'encryption_at_rest': self.enable_encryption_at_rest, 
                'auth_required': self.require_authentication_for_all_apis,
                'num_allowed_dirs': len(self.allowed_directories)
            }
        )
    
    @detailed_log_function(LogCategory.SECURITY)
    def process_document_securely(self, content: str, user_session_token: str, # Renamed param
                                 document_path_str: str, # Renamed param
                                 ip_address: Optional[str] = None, 
                                 user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Process document with full security validation."""
        SecurityManager.logger.info("Processing document securely.", parameters={'doc_path': document_path_str})

        user = self.auth_manager.validate_session(user_session_token)
        if not user:
            self.auth_manager._log_audit("unknown_user", "doc_process_denied_invalid_session", resource=document_path_str, 
                                        details={'reason': 'Invalid session token'}, ip_address=ip_address, user_agent=user_agent, status="failure")
            raise PermissionError("Invalid session or session expired.")
        
        if not self.auth_manager.check_permission(user, AccessLevel.READ): # Assuming READ is enough to process
            self.auth_manager._log_audit(user.user_id, "doc_process_denied_insufficient_perms", resource=document_path_str, 
                                        details={'required_level': AccessLevel.READ.value, 'user_level': user.access_level.value}, 
                                        ip_address=ip_address, user_agent=user_agent, status="failure")
            raise PermissionError("Insufficient permissions to process document.")
        
        validated_path = self.input_validator.validate_file_path(document_path_str, self.allowed_directories)
        sanitized_content = self.input_validator.sanitize_text(content)
        
        pii_findings: Dict[str, List[str]] = {}
        processing_content = sanitized_content
        if self.enable_pii_detection:
            pii_findings = self.pii_detector.detect_pii(sanitized_content)
            if pii_findings:
                SecurityManager.logger.warning("PII detected in document.", 
                                               parameters={'user_id': user.user_id, 'doc_path': str(validated_path), 
                                                           'pii_types': list(pii_findings.keys())})
                # Anonymize for processing if PII found and policy dictates
                # This policy could be more nuanced (e.g., based on user role or document type)
                processing_content = self.pii_detector.anonymize_text(sanitized_content)
        
        encrypted_original_content = None
        if self.enable_encryption_at_rest:
            try:
                encrypted_original_content = self.encryption_manager.encrypt(sanitized_content)
            except Exception as enc_err: # Catch specific encryption error if EncryptionManager raises one
                SecurityManager.logger.error("Failed to encrypt original content.", exception=enc_err, 
                                             parameters={'doc_path': str(validated_path)})
                # Decide error handling: proceed without encryption, or fail? For now, log and proceed without.
                encrypted_original_content = sanitized_content # Fallback to unencrypted if critical
                                
        self.auth_manager._log_audit(user.user_id, "document_processed_securely", resource=str(validated_path), 
                                    details={'pii_types_found': list(pii_findings.keys()), 
                                             'content_anonymized': bool(pii_findings and self.enable_pii_detection)},
                                    ip_address=ip_address, user_agent=user_agent)
        
        return {
            "content_for_processing": processing_content, # Anonymized or original
            "original_content_encrypted_if_enabled": encrypted_original_content, # Encrypted original or original
            "pii_detected": pii_findings,
            "validated_document_path": str(validated_path),
            "processed_by_user_id": user.user_id,
            "processing_timestamp": datetime.now(tz=datetime.timezone.utc).isoformat()
        }
    
    @detailed_log_function(LogCategory.SECURITY)
    def parse_llm_response_securely(self, llm_response_str: str) -> Dict[str, Any]: # Renamed param
        """Safely parse LLM response with validation."""
        SecurityManager.logger.debug("Parsing LLM response securely.", parameters={'response_length': len(llm_response_str)})
        try:
            # Basic check for JSON-like structures first to avoid complex regex on huge non-JSON strings
            if not ('[' in llm_response_str and ']' in llm_response_str) and \
               not ('{' in llm_response_str and '}' in llm_response_str):
                SecurityManager.logger.warning("No JSON-like structures found in LLM response.")
                return {"entities": [], "error": "Response does not appear to contain JSON.", "success": False}

            # Attempt to find the main JSON part of the response
            # This regex tries to find the largest valid-looking JSON array or object
            # It's not perfect but better than just taking the first bracket.
            json_match = re.search(r'(\[.*\]|\{.*\})', llm_response_str, re.DOTALL | re.MULTILINE)
            
            if not json_match:
                SecurityManager.logger.warning("No valid JSON block found in LLM response via regex.")
                return {"entities": [], "error": "No valid JSON block extracted from response.", "success": False}
            
            extracted_json_str = json_match.group(1)
            parsed_data = self.input_validator.validate_json(extracted_json_str) # Max size default
            
            # Ensure it's a list for entities, or wrap if it's a single dict object
            if isinstance(parsed_data, dict):
                # Check if it's a dict that *contains* an entities list (common pattern)
                if "entities" in parsed_data and isinstance(parsed_data["entities"], list):
                    parsed_data = parsed_data["entities"]
                else: # Assume the dict itself is the entity or wrap it
                    parsed_data = [parsed_data] 
            
            if not isinstance(parsed_data, list): # Final check
                 SecurityManager.logger.error("Parsed LLM data is not a list of entities.", parameters={'parsed_type': type(parsed_data).__name__})
                 return {"entities": [], "error": "Parsed data is not in expected list format.", "success": False}

            SecurityManager.logger.info("LLM response parsed successfully.", parameters={'num_entities_parsed': len(parsed_data)})
            return {"entities": parsed_data, "success": True}
            
        except ValueError as ve: # Catch validation errors specifically
            SecurityManager.logger.error("LLM response JSON validation failed.", exception=ve)
            return {"entities": [], "error": f"JSON validation error: {str(ve)}", "success": False}
        except Exception as e:
            SecurityManager.logger.error("Unexpected error parsing LLM response.", exception=e)
            return {"entities": [], "error": f"LLM response parsing failed: {str(e)}", "success": False}
    
    # _log_document_access is part of AuthenticationManager now as _log_audit
    
    @detailed_log_function(LogCategory.SECURITY)
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        auth_metrics = self.auth_manager # Assuming auth_manager has a method for its stats
        
        # Example: Count recent audit logs for specific actions
        recent_failed_logins = sum(1 for entry in self.auth_manager.audit_log 
                                   if entry.action.startswith("login_failed") and 
                                   (datetime.now(tz=datetime.timezone.utc) - entry.timestamp).total_seconds() < 3600*24) # Last 24h

        metrics = {
            "active_sessions_count": len(self.auth_manager.active_sessions),
            "total_users_count": len(self.auth_manager.users),
            "audit_log_total_entries": len(self.auth_manager.audit_log),
            "failed_logins_last_24h": recent_failed_logins,
            "pii_detection_enabled": self.enable_pii_detection,
            "encryption_at_rest_enabled": self.enable_encryption_at_rest,
            "authentication_required_for_apis": self.require_authentication_for_all_apis,
            "cryptography_library_available": CRYPTOGRAPHY_AVAILABLE
        }
        SecurityManager.logger.info("Security metrics retrieved.", parameters=metrics)
        return metrics

    async def initialize(self) -> 'SecurityManager': # For service container compatibility
        SecurityManager.logger.info("SecurityManager (async) initialize called.")
        # Current implementation is synchronous.
        return self

    def health_check(self) -> Dict[str, Any]: # For service container compatibility
        SecurityManager.logger.debug("Performing security health check.")
        # Basic health checks
        crypto_healthy = CRYPTOGRAPHY_AVAILABLE if self.enable_encryption_at_rest else True # Healthy if not enabled or lib available
        auth_healthy = True # Assuming auth manager is always "healthy" in its current form
        
        overall_status = "healthy"
        if not crypto_healthy:
            overall_status = "degraded"
            
        return {
            "status": overall_status,
            "components": {
                "pii_detector": "active" if self.enable_pii_detection else "inactive",
                "input_validator": "active",
                "encryption_manager": "active_and_healthy" if crypto_healthy and self.enable_encryption_at_rest 
                                     else ("active_but_degraded" if self.enable_encryption_at_rest and not crypto_healthy 
                                           else "inactive"),
                "authentication_manager": "active" if auth_healthy else "error"
            },
            "cryptography_available": CRYPTOGRAPHY_AVAILABLE,
            "timestamp": datetime.now(tz=datetime.timezone.utc).isoformat()
        }