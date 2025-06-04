"""
System Initializer - Creates databases, directories, and initializes logging
============================================================================
Ensures all required system components are properly initialized before use.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def initialize_system() -> Dict[str, Any]:
    """Initialize all system components and return status"""
    results = {
        'success': True,
        'errors': [],
        'created': [],
        'initialized': []
    }
    
    print("🔧 INITIALIZING LEGAL AI SYSTEM...")
    
    try:
        # 1. Create directory structure
        directories = [
            'logs',
            'storage',
            'storage/databases',
            'storage/vectors',
            'storage/documents',
            'storage/documents/inbox',
            'storage/documents/processed',
            'storage/results',
            'temp'
        ]
        
        for dir_path in directories:
            full_path = Path(dir_path)
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                results['created'].append(str(full_path))
                print(f"  ✅ Created directory: {full_path}")
            else:
                print(f"  📁 Directory exists: {full_path}")
        
        # 2. Initialize logging system
        setup_basic_logging()
        results['initialized'].append('logging_system')
        print("  ✅ Logging system initialized")
        
        # 3. Create SQLite databases
        database_results = create_databases()
        results['created'].extend(database_results['created'])
        results['errors'].extend(database_results['errors'])
        
        # 4. Create configuration files if they don't exist
        config_results = create_config_files()
        results['created'].extend(config_results['created'])
        
        print("🎉 SYSTEM INITIALIZATION COMPLETE!")
        return results
        
    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}"
        results['success'] = False
        results['errors'].append(error_msg)
        print(f"❌ {error_msg}")
        return results

def setup_basic_logging():
    """Setup basic logging that actually works"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"legal_ai_system_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )
    
    # Test logging
    logger = logging.getLogger("SystemInitializer")
    logger.info("Logging system initialized successfully")
    logger.info(f"Log file created: {log_file}")
    
    return str(log_file)

def create_databases() -> Dict[str, Any]:
    """Create all required SQLite databases"""
    results = {'created': [], 'errors': []}
    
    databases = {
        'agent_memory.db': create_agent_memory_db,
        'vector_store.db': create_vector_store_db,
        'document_metadata.db': create_document_metadata_db,
        'knowledge_graph.db': create_knowledge_graph_db,
        'user_sessions.db': create_user_sessions_db
    }
    
    db_dir = Path("storage/databases")
    
    for db_name, create_func in databases.items():
        db_path = db_dir / db_name
        try:
            if not db_path.exists():
                create_func(db_path)
                results['created'].append(str(db_path))
                print(f"  ✅ Created database: {db_name}")
            else:
                print(f"  📊 Database exists: {db_name}")
        except Exception as e:
            error_msg = f"Failed to create {db_name}: {str(e)}"
            results['errors'].append(error_msg)
            print(f"  ❌ {error_msg}")
    
    return results

def create_agent_memory_db(db_path: Path):
    """Create agent memory database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Agent memory table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT NOT NULL,
            session_id TEXT,
            memory_key TEXT NOT NULL,
            memory_value TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Agent sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            agent_type TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory(agent_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_memory_session ON agent_memory(session_id)')
    
    conn.commit()
    conn.close()

def create_vector_store_db(db_path: Path):
    """Create vector store metadata database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT UNIQUE NOT NULL,
            document_path TEXT,
            document_hash TEXT,
            vector_index INTEGER,
            embedding_model TEXT,
            chunk_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_index INTEGER,
            chunk_text TEXT,
            chunk_metadata TEXT,
            vector_id TEXT,
            FOREIGN KEY (document_id) REFERENCES document_vectors (document_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_document_metadata_db(db_path: Path):
    """Create document metadata database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT,
            file_size INTEGER,
            file_type TEXT,
            file_hash TEXT,
            processing_status TEXT DEFAULT 'pending',
            processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            analysis_result TEXT,
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (document_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_knowledge_graph_db(db_path: Path):
    """Create knowledge graph cache database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT UNIQUE NOT NULL,
            entity_type TEXT NOT NULL,
            entity_name TEXT NOT NULL,
            attributes TEXT,
            confidence_score REAL,
            source_document TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relationship_id TEXT UNIQUE NOT NULL,
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            confidence_score REAL,
            source_document TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_entity_id) REFERENCES entities (entity_id),
            FOREIGN KEY (target_entity_id) REFERENCES entities (entity_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user_sessions_db(db_path: Path):
    """Create user sessions database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            session_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_config_files() -> Dict[str, Any]:
    """Create default configuration files"""
    results = {'created': []}
    
    # Create .env.example if it doesn't exist
    env_example = Path(".env.example")
    if not env_example.exists():
        env_content = """# Legal AI System Configuration
# Copy this file to .env and update with your values

# LLM Configuration
XAI_API_KEY=your_xai_key_here
OPENAI_API_KEY=your_openai_key_here
LLM_PROVIDER=xai
LLM_MODEL=grok-3-mini

# Vector Storage
VECTOR_STORE_TYPE=hybrid
LANCE_DB_PATH=./storage/vectors/lancedb

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Logging
LOG_LEVEL=INFO
ENABLE_DETAILED_LOGGING=true

# Features
ENABLE_AUTO_TAGGING=true
ENABLE_FILE_WATCHING=false
"""
        with open(env_example, 'w') as f:
            f.write(env_content)
        results['created'].append(str(env_example))
        print(f"  ✅ Created: {env_example}")
    
    return results

def test_system_health() -> Dict[str, Any]:
    """Test system health and return status"""
    health = {
        'overall': True,
        'components': {},
        'errors': []
    }
    
    # Test logging
    try:
        logger = logging.getLogger("HealthCheck")
        logger.info("Health check: Logging system operational")
        health['components']['logging'] = True
    except Exception as e:
        health['components']['logging'] = False
        health['errors'].append(f"Logging: {str(e)}")
    
    # Test databases
    db_dir = Path("storage/databases")
    if db_dir.exists():
        db_files = list(db_dir.glob("*.db"))
        health['components']['databases'] = len(db_files) > 0
        if len(db_files) == 0:
            health['errors'].append("No database files found")
    else:
        health['components']['databases'] = False
        health['errors'].append("Database directory not found")
    
    # Test directories
    required_dirs = ['logs', 'storage', 'storage/vectors']
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    health['components']['directories'] = len(missing_dirs) == 0
    if missing_dirs:
        health['errors'].append(f"Missing directories: {missing_dirs}")
    
    # Overall health
    health['overall'] = all(health['components'].values())
    
    return health

if __name__ == "__main__":
    # Run initialization
    results = initialize_system()
    
    print("\n" + "="*50)
    print("INITIALIZATION RESULTS:")
    print("="*50)
    
    if results['success']:
        print("✅ SUCCESS: System initialized successfully")
    else:
        print("❌ FAILURE: System initialization had errors")
    
    if results['created']:
        print(f"\n📁 Created {len(results['created'])} items:")
        for item in results['created']:
            print(f"  - {item}")
    
    if results['errors']:
        print(f"\n❌ {len(results['errors'])} errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Test system health
    print("\n" + "="*50)
    print("SYSTEM HEALTH CHECK:")
    print("="*50)
    
    health = test_system_health()
    if health['overall']:
        print("✅ System is healthy and ready")
    else:
        print("❌ System has issues:")
        for error in health['errors']:
            print(f"  - {error}")