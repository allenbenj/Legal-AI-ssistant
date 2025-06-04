"""Security Manager - Critical Security Fixes.

Addresses identified vulnerabilities: input validation, PII protection, authentication.
Provides comprehensive security services for the Legal AI System including encryption,
authentication, authorization, audit logging, and PII detection.
"""

import re
import hashlib
import secrets
import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum

from .constants import Constants

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("Cryptography not available - encryption features disabled")

class AccessLevel(Enum):
    """Access levels for role-based access control.
    
    Defines hierarchical access levels from READ (basic access) to SUPER_ADMIN
    (full system access). Used throughout the system for authorization checks.
    """
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class User:
    """User entity with authentication and authorization.
    
    Represents a system user with comprehensive security attributes including
    password hashing, account lockout protection, and audit tracking.
    
    Attributes:
        user_id: Unique identifier for the user.
        username: Human-readable username for login.
        email: User's email address for notifications.
        password_hash: Securely hashed password using PBKDF2.
        salt: Random salt used for password hashing.
        access_level: User's permission level from AccessLevel enum.
        created_at: Timestamp when user account was created.
        last_login: Timestamp of user's last successful login.
        is_active: Whether the user account is currently active.
        failed_attempts: Number of consecutive failed login attempts.
        locked_until: Timestamp until which account is locked (if applicable).
    """
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    access_level: AccessLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking.
    
    Records all security-relevant events for compliance and forensic analysis.
    Essential for legal document processing systems requiring audit trails.
    
    Attributes:
        timestamp: When the event occurred.
        user_id: ID of user who performed the action.
        action: Type of action performed (e.g., 'document_processed').
        resource: Resource that was accessed or modified.
        details: Additional contextual information about the event.
        ip_address: IP address from which action was performed.
        user_agent: Browser/client user agent string.
    """
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class PIIDetector:
    """Detects and manages Personally Identifiable Information.
    
    Provides pattern-based detection of common PII types in legal documents
    including SSNs, addresses, phone numbers, and case-specific information.
    Supports both detection and anonymization workflows.
    """
    
    def __init__(self):
        # Common PII patterns for legal documents
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'address': re.compile(r'\b\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\.?\b', re.IGNORECASE),
            'date_of_birth': re.compile(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'),
            'case_number': re.compile(r'\b(?:Case|Docket|File)\s*(?:No\.?|Number)?\s*:?\s*([A-Z0-9-]+)\b', re.IGNORECASE)
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text and return findings.
        
        Args:
            text: Text content to analyze for PII.
            
        Returns:
            Dictionary mapping PII types to lists of detected instances.
            
        Example:
            >>> detector = PIIDetector()
            >>> findings = detector.detect_pii("Call John at 555-123-4567")
            >>> print(findings)
            {'phone': ['555-123-4567']}
        """
        findings = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                findings[pii_type] = matches
        
        return findings
    
    def anonymize_text(self, text: str, replacement_char: str = "█") -> str:
        """Anonymize PII in text while preserving structure.
        
        Args:
            text: Text content to anonymize.
            replacement_char: Character to use for redaction.
            
        Returns:
            Text with PII replaced by redaction characters.
            
        Note:
            Case numbers are preserved as they're essential for legal context.
        """
        anonymized = text
        
        for pii_type, pattern in self.patterns.items():
            if pii_type == 'case_number':
                # Preserve case numbers for legal context
                continue
            
            anonymized = pattern.sub(lambda m: replacement_char * len(m.group()), anonymized)
        
        return anonymized

class InputValidator:
    """Validates and sanitizes all inputs to prevent injection attacks.
    
    Provides comprehensive input validation including JSON parsing with size limits,
    file path validation with directory traversal protection, and text sanitization
    to prevent script injection attacks.
    """
    
    @staticmethod
    def validate_json(json_string: str, max_size: int = Constants.Size.MAX_JSON_PAYLOAD_BYTES) -> Dict[str, Any]:
        """Safely validate and parse JSON with size limits.
        
        Args:
            json_string: JSON string to validate and parse.
            max_size: Maximum allowed size in bytes.
            
        Returns:
            Parsed JSON as dictionary or list.
            
        Raises:
            ValueError: If JSON is invalid, too large, or contains dangerous content.
        """
        if len(json_string) > max_size:
            raise ValueError(f"JSON payload too large: {len(json_string)} bytes")
        
        try:
            # Remove any potential script tags or dangerous content
            cleaned = re.sub(r'<script[^>]*>.*?</script>', '', json_string, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'javascript:', '', cleaned, flags=re.IGNORECASE)
            
            parsed = json.loads(cleaned)
            
            # Validate structure
            if not isinstance(parsed, (dict, list)):
                raise ValueError("JSON must be object or array")
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning("json_validation_failed", error=str(e))
            raise ValueError(f"Invalid JSON: {str(e)}")
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_dirs: List[str]) -> Path:
        """Validate file path to prevent directory traversal.
        
        Args:
            file_path: File path to validate.
            allowed_dirs: List of allowed directory paths.
            
        Returns:
            Validated Path object if safe.
            
        Raises:
            ValueError: If path contains traversal attempts or is outside allowed directories.
        """
        try:
            path = Path(file_path).resolve()
            
            # Check for path traversal attempts
            if '..' in str(path) or str(path).startswith('/'):
                raise ValueError("Path traversal detected")
            
            # Ensure path is within allowed directories
            allowed = False
            for allowed_dir in allowed_dirs:
                allowed_path = Path(allowed_dir).resolve()
                try:
                    path.relative_to(allowed_path)
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                raise ValueError(f"Path not in allowed directories: {allowed_dirs}")
            
            return path
            
        except Exception as e:
            logger.warning("file_path_validation_failed", 
                          path=file_path, 
                          error=str(e))
            raise ValueError(f"Invalid file path: {str(e)}")
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = Constants.Size.MAX_TEXT_INPUT_BYTES) -> str:
        """Sanitize text input for safe processing.
        
        Args:
            text: Text content to sanitize.
            max_length: Maximum allowed length in characters.
            
        Returns:
            Sanitized text with dangerous content removed.
        """
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove potential script injections
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        return text.strip()

class EncryptionManager:
    """Manages encryption for sensitive data storage.
    
    Provides secure encryption/decryption for sensitive data using Fernet
    symmetric encryption with PBKDF2 key derivation. Handles cases where
    cryptography library is not available gracefully.
    """
    
    def __init__(self, password: str):
        """Initialize with master password.
        
        Args:
            password: Master password for key derivation.
            
        Note:
            If cryptography library is not available, encryption is disabled
            and operations return plaintext with warnings.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography not available - encryption disabled")
            self.cipher = None
            return
            
        self.password = password.encode()
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=Constants.Security.ENCRYPTION_KEY_LENGTH_BYTES,
            salt=b'legal_ai_salt',  # In production, use random salt per encryption
            iterations=Constants.Security.PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(self.kdf.derive(self.password))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Plaintext data to encrypt.
            
        Returns:
            Base64-encoded encrypted data.
            
        Raises:
            Exception: If encryption fails.
        """
        if not self.cipher:
            logger.warning("Encryption not available - returning plaintext")
            return data
            
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error("encryption_failed", error=str(e))
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data.
            
        Returns:
            Decrypted plaintext data.
            
        Raises:
            Exception: If decryption fails.
        """
        if not self.cipher:
            logger.warning("Decryption not available - returning data as-is")
            return encrypted_data
            
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error("decryption_failed", error=str(e))
            raise

class AuthenticationManager:
    """Manages user authentication and authorization.
    
    Provides comprehensive user management including secure password hashing,
    session management, account lockout protection, and audit logging.
    Implements industry-standard security practices for legal applications.
    """
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.max_failed_attempts = Constants.Security.MAX_FAILED_LOGIN_ATTEMPTS
        self.lockout_duration = timedelta(minutes=Constants.Time.ACCOUNT_LOCKOUT_DURATION_MINUTES)
    
    def create_user(self, username: str, email: str, password: str, 
                   access_level: AccessLevel = AccessLevel.READ) -> str:
        """Create a new user with secure password hashing.
        
        Args:
            username: Unique username for the user.
            email: User's email address.
            password: Plaintext password (will be securely hashed).
            access_level: Permission level for the user.
            
        Returns:
            Unique user ID for the created user.
            
        Raises:
            ValueError: If password doesn't meet security requirements.
        """
        # Validate password strength
        if len(password) < Constants.Security.MIN_PASSWORD_LENGTH_CHARS:
            raise ValueError(f"Password must be at least {Constants.Security.MIN_PASSWORD_LENGTH_CHARS} characters")
        
        if not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain uppercase letter")
        
        if not re.search(r'[a-z]', password):
            raise ValueError("Password must contain lowercase letter")
        
        if not re.search(r'\d', password):
            raise ValueError("Password must contain digit")
        
        # Generate secure salt and hash
        salt = secrets.token_hex(Constants.Security.SALT_LENGTH_BYTES)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                           password.encode(), 
                                           salt.encode(), 
                                           Constants.Security.PBKDF2_ITERATIONS)
        
        user_id = str(secrets.token_urlsafe(Constants.Security.SESSION_TOKEN_LENGTH_BYTES // 2))
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash.hex(),
            salt=salt,
            access_level=access_level,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        
        logger.info("user_created", 
                   user_id=user_id, 
                   username=username,
                   access_level=access_level.value)
        
        return user_id
    
    def authenticate(self, username: str, password: str, 
                    ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return session token.
        
        Args:
            username: Username to authenticate.
            password: Plaintext password to verify.
            ip_address: IP address of the authentication request.
            
        Returns:
            Session token if authentication successful, None otherwise.
            
        Note:
            Implements account lockout after multiple failed attempts.
        """
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self._log_auth_attempt(None, "failed_auth_invalid_user", ip_address)
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_auth_attempt(user.user_id, "failed_auth_locked", ip_address)
            return None
        
        # Verify password
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                          password.encode(),
                                          user.salt.encode(),
                                          Constants.Security.PBKDF2_ITERATIONS)
        
        if password_hash.hex() != user.password_hash:
            user.failed_attempts += 1
            
            if user.failed_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.now() + self.lockout_duration
                logger.warning("user_account_locked", 
                              user_id=user.user_id,
                              failed_attempts=user.failed_attempts)
            
            self._log_auth_attempt(user.user_id, "failed_auth_invalid_password", ip_address)
            return None
        
        # Reset failed attempts on successful auth
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Create session
        session_token = secrets.token_urlsafe(Constants.Security.SESSION_TOKEN_LENGTH_BYTES)
        self.active_sessions[session_token] = {
            'user_id': user.user_id,
            'created_at': datetime.now(),
            'ip_address': ip_address,
            'expires_at': datetime.now() + timedelta(hours=Constants.Time.SESSION_TIMEOUT_HOURS)
        }
        
        self._log_auth_attempt(user.user_id, "successful_auth", ip_address)
        
        logger.info("user_authenticated", 
                   user_id=user.user_id,
                   session_token=session_token[:8] + "...")
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session token and return user.
        
        Args:
            session_token: Session token to validate.
            
        Returns:
            User object if session is valid, None otherwise.
        """
        session = self.active_sessions.get(session_token)
        
        if not session:
            return None
        
        if datetime.now() > session['expires_at']:
            del self.active_sessions[session_token]
            return None
        
        user = self.users.get(session['user_id'])
        if not user or not user.is_active:
            return None
        
        return user
    
    def check_permission(self, user: User, required_level: AccessLevel) -> bool:
        """Check if user has required permission level.
        
        Args:
            user: User to check permissions for.
            required_level: Minimum required access level.
            
        Returns:
            True if user has sufficient permissions, False otherwise.
        """
        access_hierarchy = {
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SUPER_ADMIN: 4
        }
        
        return access_hierarchy[user.access_level] >= access_hierarchy[required_level]
    
    def _log_auth_attempt(self, user_id: Optional[str], action: str, 
                         ip_address: Optional[str]):
        """Log authentication attempt for audit."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=user_id or "unknown",
            action=action,
            resource="authentication",
            details={"ip_address": ip_address},
            ip_address=ip_address
        )
        self.audit_log.append(entry)

class SecurityManager:
    """Central security manager coordinating all security components.
    
    Provides unified security interface for the Legal AI System, coordinating
    PII detection, input validation, encryption, and authentication. Implements
    comprehensive document processing security with audit logging.
    """
    
    def __init__(self, encryption_password: str, allowed_directories: List[str]):
        self.pii_detector = PIIDetector()
        self.input_validator = InputValidator()
        self.encryption_manager = EncryptionManager(encryption_password)
        self.auth_manager = AuthenticationManager()
        self.allowed_directories = allowed_directories
        
        # Security configuration
        self.enable_pii_detection = True
        self.enable_encryption = True
        self.require_authentication = True
        
        logger.info(
            "SecurityManager initialized - PII detection: %s, Encryption: %s, Authentication: %s",
            self.enable_pii_detection,
            self.enable_encryption,
            self.require_authentication
        )
    
    def process_document_securely(self, content: str, user_session: str,
                                 document_path: str) -> Dict[str, Any]:
        """Process document with full security validation.
        
        Args:
            content: Document content to process.
            user_session: Session token for authentication.
            document_path: Path to the document being processed.
            
        Returns:
            Dictionary containing processed content, security metadata, and audit info.
            
        Raises:
            PermissionError: If session is invalid or user lacks permissions.
            ValueError: If document path or content is invalid.
        """
        # Validate session
        user = self.auth_manager.validate_session(user_session)
        if not user:
            raise PermissionError("Invalid session")
        
        # Check read permission
        if not self.auth_manager.check_permission(user, AccessLevel.READ):
            raise PermissionError("Insufficient permissions")
        
        # Validate file path
        safe_path = self.input_validator.validate_file_path(
            document_path, self.allowed_directories
        )
        
        # Sanitize content
        safe_content = self.input_validator.sanitize_text(content)
        
        # Detect PII
        pii_findings = {}
        if self.enable_pii_detection:
            pii_findings = self.pii_detector.detect_pii(safe_content)
            
            if pii_findings:
                logger.warning("pii_detected_in_document",
                              user_id=user.user_id,
                              document_path=str(safe_path),
                              pii_types=list(pii_findings.keys()))
        
        # Anonymize for processing if PII found
        processing_content = safe_content
        if pii_findings and self.enable_pii_detection:
            processing_content = self.pii_detector.anonymize_text(safe_content)
        
        # Log access
        self._log_document_access(user.user_id, str(safe_path), "document_processed")
        
        return {
            "content": processing_content,
            "original_content_encrypted": self.encryption_manager.encrypt(safe_content) if self.enable_encryption else safe_content,
            "pii_detected": pii_findings,
            "safe_path": str(safe_path),
            "processed_by": user.user_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def parse_llm_response_securely(self, response: str) -> Dict[str, Any]:
        """Safely parse LLM response with validation.
        
        Args:
            response: Raw LLM response containing JSON data.
            
        Returns:
            Dictionary with parsed entities and success/error status.
        """
        try:
            # Find JSON in response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not json_match:
                # Try object format
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            
            if not json_match:
                logger.warning("no_json_found_in_llm_response")
                return {"entities": [], "error": "No valid JSON found in response"}
            
            # Validate JSON securely
            parsed_data = self.input_validator.validate_json(json_match.group())
            
            # Ensure it's a list for entities
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            
            return {"entities": parsed_data, "success": True}
            
        except Exception as e:
            logger.error("llm_response_parsing_failed", error=str(e))
            return {"entities": [], "error": f"Parsing failed: {str(e)}"}
    
    def _log_document_access(self, user_id: str, document_path: str, action: str):
        """Log document access for audit trail."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=document_path,
            details={"action": action}
        )
        self.auth_manager.audit_log.append(entry)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring.
        
        Returns:
            Dictionary containing current security status and metrics.
        """
        return {
            "active_sessions": len(self.auth_manager.active_sessions),
            "total_users": len(self.auth_manager.users),
            "audit_log_entries": len(self.auth_manager.audit_log),
            "failed_auth_attempts_24h": len([
                entry for entry in self.auth_manager.audit_log
                if entry.action.startswith("failed_auth") and
                (datetime.now() - entry.timestamp).days < 1
            ]),
            "pii_detection_enabled": self.enable_pii_detection,
            "encryption_enabled": self.enable_encryption,
            "authentication_required": self.require_authentication
        }