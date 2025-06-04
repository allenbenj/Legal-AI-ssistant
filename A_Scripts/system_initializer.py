# legal_ai_system/core/system_initializer.py

#System Initializer - Creates databases, directories, and initializes basic logging.
#Ensures all required system components are properly initialized before use.


import os
import sqlite3
import logging # Using standard logging for basic setup
from pathlib import Path
from typing import Dict, Any, List # Added List
from datetime import datetime

# Determine project root dynamically - legal_ai_system/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR_INIT = PROJECT_ROOT / "logs"
STORAGE_DIR_INIT = PROJECT_ROOT / "storage"
DATABASES_DIR_INIT = STORAGE_DIR_INIT / "databases"

# Basic logger for this module
initializer_logger = logging.getLogger("SystemInitializer")

def setup_basic_logging(log_level: int = logging.INFO) -> str: # Added log_level param
    """Setup basic Python logging for early system stages and this script."""
    LOGS_DIR_INIT.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR_INIT / f"system_init_{timestamp}.log"
    
    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Ensure console output
        ]
    )
    
    # Test logging
    initializer_logger.info(f"Basic logging initialized. Log file: {log_file}")
    return str(log_file)

@detailed_log_function(LogCategory.SYSTEM) # Assuming detailed_logging is available when this is called by other parts of the system
def initialize_system(is_first_run_setup: bool = False) -> Dict[str, Any]: # Added param
    """Initialize all system components and return status."""
    # Setup basic logging first if not already configured by a higher-level entry point
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        setup_basic_logging()

    initializer_logger.info("🔧 INITIALIZING LEGAL AI SYSTEM...")
    
    results: Dict[str, Any] = { # Type hint for results
        'success': True,
        'errors': [],
        'created_paths': [], # Renamed from 'created'
        'initialized_components': [] # Renamed from 'initialized'
    }
    
    try:
        # 1. Create directory structure
        # Directories relative to PROJECT_ROOT
        directories_to_create = [
            LOGS_DIR_INIT,
            STORAGE_DIR_INIT,
            DATABASES_DIR_INIT,
            STORAGE_DIR_INIT / "vectors",
            STORAGE_DIR_INIT / "documents",
            STORAGE_DIR_INIT / "documents/inbox",
            STORAGE_DIR_INIT / "documents/processed",
            STORAGE_DIR_INIT / "results",
            STORAGE_DIR_INIT / "cache", # For ProcessingCache
            STORAGE_DIR_INIT / "embeddings_cache", # For EmbeddingManager
            STORAGE_DIR_INIT / "calibration", # For ConfidenceCalibrationManager
            PROJECT_ROOT / "temp" # General temp
        ]
        
        initializer_logger.info("Creating directory structure...")
        for dir_path_obj in directories_to_create:
            try:
                if not dir_path_obj.exists():
                    dir_path_obj.mkdir(parents=True, exist_ok=True)
                    results['created_paths'].append(str(dir_path_obj))
                    initializer_logger.info(f"Directory created.", parameters={'path': str(dir_path_obj)})
                else:
                    initializer_logger.debug(f"Directory already exists.", parameters={'path': str(dir_path_obj)})
            except Exception as e:
                msg = f"Failed to create directory {dir_path_obj}"
                initializer_logger.error(msg, exception=e)
                results['errors'].append(f"{msg}: {str(e)}")
                results['success'] = False


        results['initialized_components'].append('directories')
        
        # 2. Create SQLite databases
        initializer_logger.info("Creating databases...")
        database_results = _create_databases_core() # Renamed internal helper
        results['created_paths'].extend(database_results['created'])
        results['errors'].extend(database_results['errors'])
        if database_results['errors']: results['success'] = False
        results['initialized_components'].append('databases')

        # 3. Create configuration files if it's a first run setup
        if is_first_run_setup:
            initializer_logger.info("Performing first-run setup: Creating config files...")
            config_results = _create_config_files_core() # Renamed internal helper
            results['created_paths'].extend(config_results['created'])
            results['initialized_components'].append('config_files_example')
        else:
            initializer_logger.info("Skipping config file creation (not first run).")

        if results['success']:
            initializer_logger.info("🎉 SYSTEM INITIALIZATION COMPLETE!")
        else:
            initializer_logger.error("SYSTEM INITIALIZATION COMPLETED WITH ERRORS.")
        return results
        
    except Exception as e:
        error_msg = f"Critical system initialization failure."
        initializer_logger.critical(error_msg, exception=e)
        results['success'] = False
        results['errors'].append(f"{error_msg}: {str(e)}")
        return results

def _create_databases_core() -> Dict[str, List[str]]: # Renamed
    """Helper to create all required SQLite databases."""
    results: Dict[str, List[str]] = {'created': [], 'errors': []}
    
    # Define databases and their creation functions
    # Functions should take db_path (Path object) as argument
    databases_to_init: Dict[str, Callable[[Path], None]] = { # type: ignore[valid-type]
        'agent_memory.db': _create_agent_memory_db_schema, # Renamed
        'vector_store_meta.db': _create_vector_store_db_schema, # Renamed for clarity
        'document_metadata.db': _create_document_metadata_db_schema, # Renamed
        'knowledge_graph_cache.db': _create_knowledge_graph_db_schema, # Renamed
        'user_sessions.db': _create_user_sessions_db_schema, # Renamed
        'ml_optimizer.db': _create_ml_optimizer_db_schema, # For MLOptimizer
        'review_memory.db': _create_review_memory_db_schema, # For ReviewableMemory
    }
    
    for db_filename, creation_func in databases_to_init.items():
        db_file_path = DATABASES_DIR_INIT / db_filename
        try:
            if not db_file_path.exists():
                initializer_logger.debug(f"Database file '{db_filename}' not found. Creating...")
                creation_func(db_file_path)
                results['created'].append(str(db_file_path))
                initializer_logger.info(f"Database created.", parameters={'db_name': db_filename})
            else:
                initializer_logger.debug(f"Database already exists.", parameters={'db_name': db_filename})
                # Optionally, add schema validation/migration here for existing DBs
        except Exception as e:
            error_message = f"Failed to create/verify database '{db_filename}'"
            initializer_logger.error(error_message, exception=e)
            results['errors'].append(f"{error_message}: {str(e)}")
    
    return results

# Schema creation functions (internal helpers)
def _execute_schema(db_path: Path, schema_sql: str):
    """Executes SQL schema script on a given database path."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_sql)
        initializer_logger.debug(f"Schema executed successfully for {db_path.name}")
    except sqlite3.Error as e:
        initializer_logger.error(f"SQLite error executing schema for {db_path.name}", exception=e)
        raise # Re-raise to be caught by _create_databases_core

def _create_agent_memory_db_schema(db_path: Path):
    schema = """
        CREATE TABLE IF NOT EXISTS agent_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            metadata TEXT DEFAULT '{}',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_agent_mem_doc_agent_key ON agent_memories(doc_id, agent, key);
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            agent_type TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
    """
    _execute_schema(db_path, schema)

def _create_vector_store_db_schema(db_path: Path):
    schema = """
        CREATE TABLE IF NOT EXISTS vector_metadata (
            vector_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            content_preview TEXT,
            vector_norm REAL,
            dimension INTEGER,
            created_at TEXT,
            last_accessed TEXT,
            access_count INTEGER DEFAULT 0,
            source_file TEXT,
            document_type TEXT,
            tags TEXT,
            confidence_score REAL DEFAULT 1.0,
            embedding_model TEXT,
            custom_metadata TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_vs_meta_doc_id ON vector_metadata(document_id);
        CREATE INDEX IF NOT EXISTS idx_vs_meta_hash ON vector_metadata(content_hash);
    """
    _execute_schema(db_path, schema)

def _create_document_metadata_db_schema(db_path: Path):
    schema = """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY, /* Changed from auto-increment to text for flexibility */
            filename TEXT NOT NULL,
            file_path TEXT UNIQUE, /* Path should be unique if used as identifier */
            file_size INTEGER,
            file_type TEXT,
            file_hash TEXT UNIQUE, /* Hash of content for deduplication */
            processing_status TEXT DEFAULT 'pending', /* e.g., pending, processing, completed, failed */
            processed_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, /* For tracking metadata updates */
            source TEXT, /* e.g., upload, file_watch, api */
            tags TEXT, /* JSON list of tags */
            custom_metadata TEXT /* JSON for other metadata */
        );
        CREATE INDEX IF NOT EXISTS idx_doc_meta_status ON documents(processing_status);
        CREATE INDEX IF NOT EXISTS idx_doc_meta_file_type ON documents(file_type);

        CREATE TABLE IF NOT EXISTS document_analysis_results ( /* Renamed from document_analysis */
            analysis_id TEXT PRIMARY KEY, /* UUID for analysis result */
            document_id TEXT NOT NULL,
            agent_name TEXT NOT NULL, /* Which agent performed this analysis */
            analysis_type TEXT NOT NULL, /* e.g., ner, summarization, classification */
            result_summary TEXT, /* Short summary or key findings */
            full_result_path TEXT, /* Path to a file storing the full JSON/structured result if large */
            confidence_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_doc_analysis_doc_id_type ON document_analysis_results(document_id, analysis_type);
    """
    _execute_schema(db_path, schema)

def _create_knowledge_graph_db_schema(db_path: Path): # For local caching if Neo4j is down or not primary
    schema = """
        CREATE TABLE IF NOT EXISTS kg_entities_cache (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL, /* Canonical name */
            properties TEXT, /* JSON dump of properties */
            last_synced_neo4j DATETIME
        );
        CREATE INDEX IF NOT EXISTS idx_kg_entity_type_name ON kg_entities_cache(entity_type, name);

        CREATE TABLE IF NOT EXISTS kg_relationships_cache (
            relationship_id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            properties TEXT, /* JSON dump of properties */
            last_synced_neo4j DATETIME,
            FOREIGN KEY (source_entity_id) REFERENCES kg_entities_cache (entity_id) ON DELETE CASCADE,
            FOREIGN KEY (target_entity_id) REFERENCES kg_entities_cache (entity_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_kg_rel_source_type ON kg_relationships_cache(source_entity_id, relationship_type);
        CREATE INDEX IF NOT EXISTS idx_kg_rel_target_type ON kg_relationships_cache(target_entity_id, relationship_type);
    """
    _execute_schema(db_path, schema)

def _create_user_sessions_db_schema(db_path: Path):
    schema = """
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT, /* Link to a user table if you have one */
            session_data TEXT, /* JSON blob for session state */
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME, /* For session timeout */
            status TEXT DEFAULT 'active' /* active, expired, closed */
        );
        CREATE INDEX IF NOT EXISTS idx_user_session_user_id ON user_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_session_expires ON user_sessions(expires_at);

        CREATE TABLE IF NOT EXISTS session_activity_log ( /* Renamed from session_logs */
            log_id TEXT PRIMARY KEY, /* UUID for log entry */
            session_id TEXT NOT NULL,
            activity_type TEXT NOT NULL, /* e.g., login, doc_upload, search, agent_interaction */
            details TEXT, /* JSON details of the activity */
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions (session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_session_log_session_activity ON session_activity_log(session_id, activity_type);
    """
    _execute_schema(db_path, schema)

def _create_ml_optimizer_db_schema(db_path: Path):
    schema = """
        CREATE TABLE IF NOT EXISTS performance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_path TEXT NOT NULL,
            document_type TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            features_json TEXT NOT NULL,
            objective TEXT NOT NULL,
            composite_score REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_perf_doc_type_obj_score ON performance_records(document_type, objective, composite_score DESC);
        
        CREATE TABLE IF NOT EXISTS optimization_cache (
            cache_key TEXT PRIMARY KEY, /* Composite key of doc_type, features_hash, objective */
            parameters_json TEXT NOT NULL,
            expected_improvement REAL NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            samples_count INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_opt_cache_expires ON optimization_cache(expires_at);
    """
    _execute_schema(db_path, schema)

def _create_review_memory_db_schema(db_path: Path):
    schema = """
        CREATE TABLE IF NOT EXISTS review_items (
            item_id TEXT PRIMARY KEY,
            item_type TEXT NOT NULL, -- 'entity', 'relationship', 'finding'
            content TEXT NOT NULL, -- JSON content of the item
            confidence REAL NOT NULL,
            source_document_id TEXT NOT NULL, -- Link to documents table
            extraction_context TEXT, -- e.g., agent name, model used
            review_status TEXT NOT NULL, -- PENDING, APPROVED, REJECTED, MODIFIED, AUTO_APPROVED
            review_priority TEXT NOT NULL, -- CRITICAL, HIGH, MEDIUM, LOW
            created_at DATETIME NOT NULL,
            reviewed_at DATETIME,
            reviewer_id TEXT, -- Link to user table
            reviewer_notes TEXT,
            original_content TEXT, -- JSON of content before modification
            FOREIGN KEY (source_document_id) REFERENCES documents (document_id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_review_status_priority_created ON review_items(review_status, review_priority, created_at);

        CREATE TABLE IF NOT EXISTS legal_findings ( -- For more complex findings
            finding_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            finding_type TEXT NOT NULL, -- 'violation', 'contradiction', 'pattern'
            description TEXT NOT NULL,
            entities_involved TEXT, -- JSON list of entity_ids
            relationships_involved TEXT, -- JSON list of relationship_ids
            evidence_snippets TEXT, -- JSON list of text snippets
            confidence REAL NOT NULL,
            severity TEXT NOT NULL, -- CRITICAL, HIGH, MEDIUM, LOW
            created_at DATETIME NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, CONFIRMED, DISMISSED
            reviewed_by_user_id TEXT,
            reviewed_at DATETIME,
            FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_finding_doc_id_type_status ON legal_findings(document_id, finding_type, review_status);

        CREATE TABLE IF NOT EXISTS feedback_history ( -- For learning from reviews
            feedback_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL, -- Can be review_item_id or finding_id
            item_type_reviewed TEXT NOT NULL, -- 'review_item', 'legal_finding'
            original_confidence REAL,
            review_decision TEXT NOT NULL, -- e.g., APPROVED, REJECTED, MODIFIED
            confidence_adjustment REAL, -- If confidence was manually changed
            feedback_notes TEXT,
            created_at DATETIME NOT NULL,
            user_id TEXT -- User who provided feedback
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_item_id ON feedback_history(item_id);
    """
    _execute_schema(db_path, schema)


def _create_config_files_core() -> Dict[str, List[str]]: # Renamed
    """Helper to create default configuration files like .env.example."""
    results: Dict[str, List[str]] = {'created': []}
    
    env_example_path = PROJECT_ROOT / ".env.example"
    if not env_example_path.exists():
        env_content = """# Legal AI System Configuration Example
# Copy this file to .env and update with your actual values.
# Lines starting with # are comments.

# --- Core System ---
# APP_NAME="Legal AI Assistant"
# DEBUG=False
# LOG_LEVEL=INFO # TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- LLM Providers ---
# Primary LLM (options: ollama, openai, xai)
LEGAL_AI_LLM_PROVIDER=xai
LEGAL_AI_LLM_MODEL=grok-3-mini # Default for xai
# LEGAL_AI_LLM_MODEL=llama3.2 # Example for ollama
# LEGAL_AI_LLM_MODEL=gpt-4o # Example for openai

# xAI/Grok Settings (only if LLM_PROVIDER=xai)
LEGAL_AI_XAI_API_KEY="your_xai_api_key_here"
# LEGAL_AI_XAI_BASE_URL="https://api.x.ai/v1" # Default

# OpenAI Settings (only if LLM_PROVIDER=openai)
# LEGAL_AI_OPENAI_API_KEY="your_openai_api_key_here"
# LEGAL_AI_OPENAI_BASE_URL="httpsAI_API_KEY_HERE"

# Ollama Settings (only if LLM_PROVIDER=ollama or as fallback)
# LEGAL_AI_OLLAMA_HOST="http://localhost:11434" # Default
# LEGAL_AI_OLLAMA_TIMEOUT=60 # Default

# Fallback LLM (if primary fails)
LEGAL_AI_FALLBACK_PROVIDER=ollama
LEGAL_AI_FALLBACK_MODEL=llama3.2

# --- Databases ---
# Neo4j (for Knowledge Graph)
LEGAL_AI_NEO4J_URI="bolt://localhost:7687"
LEGAL_AI_NEO4J_USER="neo4j"
LEGAL_AI_NEO4J_PASSWORD="your_neo4j_password" # CHANGE THIS!
LEGAL_AI_NEO4J_DATABASE="neo4j"

# SQLite paths are relative to storage/databases/ (defined in settings.py)
# No need to set them in .env unless overriding defaults from settings.py

# --- Vector Storage ---
# LEGAL_AI_VECTOR_STORE_TYPE="hybrid" # faiss, lance, hybrid
# LEGAL_AI_EMBEDDING_MODEL="all-MiniLM-L6-v2" # Default

# --- Security ---
# Encryption password for sensitive data (IMPORTANT: Set a strong, unique password)
# LEGAL_AI_ENCRYPTION_PASSWORD="a_very_strong_and_secret_password_here" # Used by SecurityManager, not directly by settings
# Note: The salt for EncryptionManager is currently hardcoded in security_manager.py for simplicity in this refactor.
# In a production system, LEGAL_AI_ENCRYPTION_SALT (hex-encoded) should also be in .env and securely managed.

# --- API (FastAPI) ---
# LEGAL_AI_API_HOST="0.0.0.0"
# LEGAL_AI_API_PORT=8000
# LEGAL_AI_JWT_SECRET_KEY="a_very_secret_key_for_jwt_replace_me" # For FastAPI auth if used

# --- Development ---
# LEGAL_AI_ENABLE_TEST_MODE=False
"""
        try:
            with open(env_example_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            results['created'].append(str(env_example_path))
            initializer_logger.info(f"Created example environment file.", parameters={'path': str(env_example_path)})
        except IOError as e:
            initializer_logger.error(f"Failed to create .env.example file.", exception=e)
            results['errors'].append(f"Failed to create .env.example: {str(e)}")
            
    return results

@detailed_log_function(LogCategory.SYSTEM)
def test_system_health() -> Dict[str, Any]:
    """Test basic system health after initialization."""
    initializer_logger.info("Performing basic system health check...")
    health: Dict[str, Any] = { # Type hint
        'overall_status': "HEALTHY", # Renamed from 'overall'
        'component_checks': {}, # Renamed from 'components'
        'issues_found': [] # Renamed from 'errors'
    }
    
    # Test logging directory
    if LOGS_DIR_INIT.exists() and LOGS_DIR_INIT.is_dir():
        health['component_checks']['logging_directory'] = "OK"
    else:
        health['component_checks']['logging_directory'] = "ERROR: Missing or not a directory"
        health['issues_found'].append("Logging directory issue.")
        health['overall_status'] = "DEGRADED"
    
    # Test database directory and files
    if DATABASES_DIR_INIT.exists() and DATABASES_DIR_INIT.is_dir():
        health['component_checks']['databases_directory'] = "OK"
        # Check for at least one .db file as a proxy for successful DB creation
        db_files = list(DATABASES_DIR_INIT.glob("*.db"))
        if db_files:
            health['component_checks']['database_files_present'] = "OK"
            # Try connecting to one DB as a simple test
            try:
                with sqlite3.connect(db_files[0]) as conn:
                    conn.execute("SELECT 1;")
                health['component_checks']['database_connectivity_basic'] = "OK"
            except sqlite3.Error as e:
                health['component_checks']['database_connectivity_basic'] = f"ERROR: {str(e)}"
                health['issues_found'].append(f"Basic DB connectivity failed for {db_files[0].name}.")
                health['overall_status'] = "ERROR"

        else:
            health['component_checks']['database_files_present'] = "WARNING: No .db files found"
            health['issues_found'].append("No SQLite database files found in databases directory.")
            health['overall_status'] = "DEGRADED"
    else:
        health['component_checks']['databases_directory'] = "ERROR: Missing or not a directory"
        health['issues_found'].append("Databases directory issue.")
        health['overall_status'] = "ERROR"

    # Test other storage directories
    for dirname in ["vectors", "documents", "cache"]:
        dir_path = STORAGE_DIR_INIT / dirname
        if dir_path.exists() and dir_path.is_dir():
            health['component_checks'][f'{dirname}_directory'] = "OK"
        else:
            health['component_checks'][f'{dirname}_directory'] = "ERROR: Missing or not a directory"
            health['issues_found'].append(f"Storage directory '{dirname}' issue.")
            health['overall_status'] = "DEGRADED"
            
    if not health['issues_found']: # If no issues, status remains HEALTHY
        pass
    elif health['overall_status'] != "ERROR": # If only warnings, it's DEGRADED
        health['overall_status'] = "DEGRADED"

    initializer_logger.info("System health check complete.", parameters=health)
    return health

# Main execution block for standalone script usage
if __name__ == "__main__":
    # Setup basic logging for the script itself if run directly
    # This ensures that if detailed_logging isn't fully set up by the main app yet,
    # this script can still log its own actions.
    if not logging.getLogger("SystemInitializer").hasHandlers():
        # Need to configure the basicConfig for the root logger if no handlers exist
        # or specifically for the "SystemInitializer" logger.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

    results = initialize_system(is_first_run_setup=True) # Assume first run if executed directly
    
    print("\n" + "="*50)
    print("INITIALIZATION SCRIPT RESULTS:")
    print("="*50)
    
    if results['success']:
        print("✅ SUCCESS: System initialization script completed successfully.")
    else:
        print("❌ FAILURE: System initialization script encountered errors.")
    
    if results['created_paths']:
        print(f"\n📁 Created/Verified {len(results['created_paths'])} paths:")
        for item in results['created_paths']:
            print(f"  - {item}")
    
    if results['errors']:
        print(f"\n❌ Encountered {len(results['errors'])} errors during initialization:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Perform and print health check results
    print("\n" + "="*50)
    print("SYSTEM HEALTH CHECK RESULTS (POST-INITIALIZATION):")
    print("="*50)
    
    health_results = test_system_health()
    print(f"  Overall Status: {health_results['overall_status']}")
    for component, status in health_results['component_checks'].items():
        print(f"  - {component.replace('_', ' ').title()}: {status}")
    
    if health_results['issues_found']:
        print("\n  Detected Issues:")
        for issue in health_results['issues_found']:
            print(f"    - {issue}")
    elif health_results['overall_status'] == "HEALTHY":
         print("\n  ✅ System appears healthy and ready for further setup/use.")