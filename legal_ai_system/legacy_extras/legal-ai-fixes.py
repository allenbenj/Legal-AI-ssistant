# Legal AI System - Critical Fixes
# This file contains fixes for the startup errors

## Fix 1: SecurityManager - Replace custom logging with standard logging

# In core/security_manager.py, update the SecurityManager __init__ method:

import logging
import structlog

# Option 1: Use structlog if available, otherwise fallback to standard logging
try:
    logger = structlog.get_logger()
except:
    logger = logging.getLogger(__name__)

class SecurityManager:
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
        
        # Fixed logging - use standard Python logging format
        logger.info(
            "SecurityManager initialized - PII detection: %s, Encryption: %s, Authentication: %s",
            self.enable_pii_detection,
            self.enable_encryption,
            self.require_authentication
        )

## Fix 2: AuthenticationManager - Fix logging in methods

# In the same file, update AuthenticationManager methods:

class AuthenticationManager:
    def create_user(self, username: str, email: str, password: str, access_level: AccessLevel = AccessLevel.READ) -> str:
        # ... existing validation code ...
        
        # Fixed logging
        logger.info(
            "User created: %s (username: %s, access_level: %s)",
            user_id,
            username,
            access_level.value
        )
        
        return user_id
    
    def _log_auth_attempt(self, user_id: Optional[str], action: str, ip_address: Optional[str]):
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

## Fix 3: ServiceContainer - Add missing shutdown method

# In core/unified_services.py, add the shutdown method:

class ServiceContainer:
    # ... existing methods ...
    
    async def shutdown(self) -> None:
        """Shutdown all services in reverse order."""
        services_logger.info("🛑 SHUTTING DOWN ALL SERVICES 🛑")
        
        # Stop health monitoring
        self._health_monitor_running = False
        
        shutdown_count = 0
        failed_count = 0
        
        # Shutdown services in reverse order
        for service_name in self._shutdown_order:
            try:
                if service_name not in self._instances:
                    continue
                    
                services_logger.info(f"Shutting down service: {service_name}")
                instance = self._instances[service_name]
                
                # Call shutdown method if available
                if hasattr(instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(instance.shutdown):
                        await instance.shutdown()
                    else:
                        instance.shutdown()
                elif hasattr(instance, 'close'):
                    if asyncio.iscoroutinefunction(instance.close):
                        await instance.close()
                    else:
                        instance.close()
                
                # Update state
                if service_name in self._services:
                    self._services[service_name].state = ServiceState.STOPPED
                
                shutdown_count += 1
                lifecycle_logger.info(f"Service {service_name} shut down successfully")
                
            except Exception as e:
                failed_count += 1
                lifecycle_logger.error(f"Failed to shutdown service {service_name}", exception=e)
        
        # Clear instances
        self._instances.clear()
        
        services_logger.info(
            "Service shutdown complete",
            parameters={
                'shutdown_count': shutdown_count,
                'failed_count': failed_count
            }
        )

## Fix 4: API Main - Fix async monitoring issue

# In api/main.py, fix the system monitoring:

async def monitor_system():
    """Monitor system health periodically."""
    while True:
        try:
            # Fix: get_system_status is synchronous, not async
            status = service_container.get_system_status()
            
            # Log system health
            healthy_percentage = status.get('health_percentage', 0)
            if healthy_percentage < 80:
                logger.warning(f"System health degraded: {healthy_percentage}%")
            
            # Wait for next check
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"System monitoring error: {e}")
            await asyncio.sleep(60)  # Continue monitoring even on error

# Also in the lifespan function, fix the admin user creation:

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    try:
        logger.info("🚀 Starting Legal AI System API...")
        
        # Initialize services
        global service_container, security_manager
        service_container = get_service_container()
        
        # Initialize security manager
        security_manager = SecurityManager(
            encryption_password=settings.ENCRYPTION_KEY,
            allowed_directories=[settings.UPLOAD_DIR, settings.STORAGE_DIR]
        )
        
        # Create admin user if it doesn't exist
        try:
            admin_id = security_manager.auth_manager.create_user(
                username="admin",
                email="admin@legal-ai.com",
                password=settings.ADMIN_PASSWORD,
                access_level=AccessLevel.SUPER_ADMIN
            )
            logger.info(f"✅ Admin user created: {admin_id}")
        except Exception as e:
            # User might already exist or other error
            logger.info(f"ℹ️ Admin user already exists or creation failed: {e}")
        
        # Start monitoring task
        asyncio.create_task(monitor_system())
        
        logger.info("✅ Legal AI System API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Legal AI System API...")
        
        # Shutdown services
        if service_container:
            await service_container.shutdown()
        
        logger.info("👋 Legal AI System API shut down complete")

## Fix 5: Create a proper structlog configuration

# Create a new file: core/logging_config.py

import logging
import structlog
from typing import Any, Dict

def configure_logging():
    """Configure structlog for the application."""
    
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Then in your application startup, call configure_logging()

## Summary of Changes:

1. **SecurityManager**: Replaced custom logging parameters with standard Python logging format
2. **AuthenticationManager**: Fixed logging calls to use standard format
3. **ServiceContainer**: Added proper `shutdown()` method
4. **API Main**: Fixed async/await issue in monitoring and improved error handling
5. **Logging Configuration**: Added proper structlog configuration

## To Apply These Fixes:

1. Update `core/security_manager.py` with the logging fixes
2. Add the `shutdown()` method to `core/unified_services.py`
3. Fix the monitoring function in `api/main.py`
4. Create `core/logging_config.py` and call it during startup
5. Restart the API server

These fixes should resolve all the startup errors and allow the API to run properly.