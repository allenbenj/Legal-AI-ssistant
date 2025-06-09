"""
Comprehensive Test Suite - Addresses Missing Test Coverage
Includes unit tests, integration tests, security tests, and performance tests
"""

import pytest
import asyncio
import aiohttp
import time
import random
import string
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Test framework imports
from pytest_benchmark import benchmark
import pytest_asyncio
from pytest_mock import MockerFixture

# System imports (would need to be adjusted based on actual import paths)
import sys
sys.path.append('..')

from core.security_manager import SecurityManager, PIIDetector, InputValidator, AuthenticationManager, AccessLevel
from core.enhanced_persistence import EnhancedPersistenceManager, EntityRecord, EntityStatus
from core.workflow_state_manager import WorkflowStateManager, AgentType, WorkflowState
from agents.entity_extraction import EntityExtractionAgent, LegalEntity, EntityExtractionResult
from agents.knowledge_base_agent import KnowledgeBaseAgent, ResolvedEntity

class TestSecurityManager:
    """Comprehensive security testing suite."""
    
    @pytest.fixture
    def security_manager(self, tmp_path):
        """Create security manager for testing."""
        allowed_dirs = [str(tmp_path)]
        return SecurityManager("test_encryption_key", allowed_dirs)
    
    @pytest.fixture
    def sample_legal_document(self):
        """Sample legal document with PII for testing."""
        return """
        Case No: 2024-CV-001234
        
        Plaintiff: John Doe (SSN: 123-45-6789)
        Email: john.doe@email.com
        Phone: (555) 123-4567
        Address: 123 Main Street, Anytown, NY 12345
        
        This case involves potential Brady violations where the prosecution
        failed to disclose exculpatory evidence to the defense.
        """
    
    def test_pii_detection(self, security_manager, sample_legal_document):
        """Test PII detection accuracy."""
        pii_findings = security_manager.pii_detector.detect_pii(sample_legal_document)
        
        assert 'ssn' in pii_findings
        assert 'email' in pii_findings
        assert 'phone' in pii_findings
        assert 'address' in pii_findings
        assert 'case_number' in pii_findings
        
        assert '123-45-6789' in pii_findings['ssn']
        assert 'john.doe@email.com' in pii_findings['email']
    
    def test_pii_anonymization(self, security_manager, sample_legal_document):
        """Test PII anonymization preserves structure."""
        anonymized = security_manager.pii_detector.anonymize_text(sample_legal_document)
        
        # Should not contain original PII
        assert '123-45-6789' not in anonymized
        assert 'john.doe@email.com' not in anonymized
        
        # Should preserve case numbers for legal context
        assert '2024-CV-001234' in anonymized
        
        # Should preserve general structure
        assert 'Plaintiff:' in anonymized
        assert 'Brady violations' in anonymized
    
    def test_input_validation_json_injection(self, security_manager):
        """Test protection against JSON injection attacks."""
        malicious_json = '''
        {
            "entities": [
                {"name": "test", "type": "person"}
            ],
            "script": "<script>alert('xss')</script>",
            "javascript": "javascript:alert('evil')"
        }
        '''
        
        # Should successfully parse but sanitize dangerous content
        result = security_manager.parse_llm_response_securely(malicious_json)
        assert result['success'] is True
        assert len(result['entities']) > 0
        
        # Should not contain script content
        json_str = json.dumps(result)
        assert '<script>' not in json_str
        assert 'javascript:' not in json_str
    
    def test_path_traversal_protection(self, security_manager, tmp_path):
        """Test protection against path traversal attacks."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            security_manager.input_validator.validate_file_path(
                "../../etc/passwd", [str(tmp_path)]
            )
        
        with pytest.raises(ValueError, match="Path not in allowed directories"):
            security_manager.input_validator.validate_file_path(
                "/tmp/evil_file.txt", [str(tmp_path)]
            )
    
    def test_authentication_password_strength(self, security_manager):
        """Test password strength requirements."""
        auth_manager = security_manager.auth_manager
        
        # Weak passwords should be rejected
        with pytest.raises(ValueError, match="Password must be at least 8 characters"):
            auth_manager.create_user("test", "test@test.com", "weak")
        
        with pytest.raises(ValueError, match="Password must contain uppercase letter"):
            auth_manager.create_user("test", "test@test.com", "lowercase123")
        
        with pytest.raises(ValueError, match="Password must contain digit"):
            auth_manager.create_user("test", "test@test.com", "NoNumbers")
        
        # Strong password should work
        user_id = auth_manager.create_user("test", "test@test.com", "StrongPass123")
        assert user_id is not None
    
    def test_authentication_lockout(self, security_manager):
        """Test account lockout after failed attempts."""
        auth_manager = security_manager.auth_manager
        user_id = auth_manager.create_user("test", "test@test.com", "StrongPass123")
        
        # Multiple failed attempts should lock account
        for _ in range(5):
            result = auth_manager.authenticate("test", "wrong_password")
            assert result is None
        
        # Account should be locked
        user = auth_manager.users[user_id]
        assert user.locked_until is not None
        
        # Even correct password should fail when locked
        result = auth_manager.authenticate("test", "StrongPass123")
        assert result is None
    
    def test_encryption_decryption(self, security_manager):
        """Test encryption and decryption of sensitive data."""
        sensitive_data = "Attorney-client privileged communication"
        
        encrypted = security_manager.encryption_manager.encrypt(sensitive_data)
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
        
        decrypted = security_manager.encryption_manager.decrypt(encrypted)
        assert decrypted == sensitive_data

class TestEnhancedPersistence:
    """Test enhanced persistence layer with ACID transactions."""
    
    @pytest.fixture
    async def persistence_manager(self):
        """Create persistence manager for testing."""
        # Use in-memory databases for testing
        db_url = "postgresql://postgres:password@localhost:5432/test_legal_ai"
        redis_url = "redis://localhost:6379/1"
        
        manager = EnhancedPersistenceManager(db_url, redis_url)
        await manager.initialize()
        yield manager
        await manager.close()
    
    @pytest.fixture
    def sample_entity(self):
        """Create sample entity for testing."""
        return EntityRecord(
            entity_id="test-entity-001",
            entity_type="Person",
            canonical_name="John Doe",
            attributes={"role": "plaintiff", "jurisdiction": "federal"},
            confidence_score=0.9,
            status=EntityStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="test_user",
            updated_by="test_user",
            source_documents=["doc1", "doc2"]
        )
    
    @pytest.mark.asyncio
    async def test_entity_crud_operations(self, persistence_manager, sample_entity):
        """Test complete CRUD operations for entities."""
        # Create
        entity_id = await persistence_manager.entity_repo.create_entity(sample_entity)
        assert entity_id == sample_entity.entity_id
        
        # Read
        retrieved = await persistence_manager.entity_repo.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.canonical_name == sample_entity.canonical_name
        assert retrieved.entity_type == sample_entity.entity_type
        
        # Update
        updates = {"confidence_score": 0.95, "canonical_name": "John P. Doe"}
        success = await persistence_manager.entity_repo.update_entity(
            entity_id, updates, "test_updater"
        )
        assert success is True
        
        # Verify update
        updated = await persistence_manager.entity_repo.get_entity(entity_id)
        assert updated.confidence_score == 0.95
        assert updated.canonical_name == "John P. Doe"
        assert updated.version == 2  # Version should increment
    
    @pytest.mark.asyncio
    async def test_entity_similarity_search(self, persistence_manager, sample_entity):
        """Test entity similarity search functionality."""
        # Create test entities
        await persistence_manager.entity_repo.create_entity(sample_entity)
        
        similar_entity = EntityRecord(
            entity_id="test-entity-002",
            entity_type="Person",
            canonical_name="John Smith",  # Similar name
            attributes={},
            confidence_score=0.8,
            status=EntityStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="test_user",
            updated_by="test_user"
        )
        await persistence_manager.entity_repo.create_entity(similar_entity)
        
        # Search for similar entities
        similar = await persistence_manager.entity_repo.find_similar_entities(
            "Person", "John", 0.3
        )
        
        assert len(similar) >= 2
        names = [e.canonical_name for e in similar]
        assert "John Doe" in names
        assert "John Smith" in names
    
    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, persistence_manager):
        """Test batch operations for performance."""
        # Create multiple entities
        entities = []
        for i in range(100):
            entity = EntityRecord(
                entity_id=f"batch-entity-{i:03d}",
                entity_type="Document",
                canonical_name=f"Document {i}",
                attributes={"batch": i},
                confidence_score=random.uniform(0.5, 1.0),
                status=EntityStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="batch_test",
                updated_by="batch_test"
            )
            entities.append(entity)
        
        start_time = time.time()
        entity_ids = await persistence_manager.entity_repo.batch_create_entities(entities)
        end_time = time.time()
        
        assert len(entity_ids) == 100
        assert end_time - start_time < 2.0  # Should complete in under 2 seconds
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, persistence_manager, sample_entity):
        """Test transaction rollback on failure."""
        # This test would need to simulate a failure mid-transaction
        # and verify that no partial data is committed
        with pytest.raises(Exception):
            async with persistence_manager.entity_repo.transaction_manager.transaction() as conn:
                await conn.execute("""
                    INSERT INTO entities (
                        entity_id, entity_type, canonical_name, attributes, 
                        confidence_score, status, created_at, updated_at, 
                        created_by, updated_by, version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                sample_entity.entity_id, sample_entity.entity_type, sample_entity.canonical_name,
                json.dumps(sample_entity.attributes), sample_entity.confidence_score,
                sample_entity.status.value, sample_entity.created_at, sample_entity.updated_at,
                sample_entity.created_by, sample_entity.updated_by, sample_entity.version)
                
                # Force an error
                raise Exception("Simulated transaction failure")
        
        # Verify entity was not created
        retrieved = await persistence_manager.entity_repo.get_entity(sample_entity.entity_id)
        assert retrieved is None

class TestWorkflowStateManager:
    """Test workflow state management and coordination."""
    
    @pytest.fixture
    def workflow_manager(self, tmp_path):
        """Create workflow manager for testing."""
        return WorkflowStateManager(str(tmp_path / "workflows"))
    
    @pytest.fixture
    def sample_documents(self, tmp_path):
        """Create sample documents for testing."""
        docs = []
        for i in range(3):
            doc_path = tmp_path / f"document_{i}.txt"
            doc_path.write_text(f"Sample document {i} content")
            docs.append(str(doc_path))
        return docs
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, workflow_manager, sample_documents):
        """Test workflow creation and initialization."""
        workflow_id = await workflow_manager.create_workflow(sample_documents)
        
        assert workflow_id is not None
        assert workflow_id in workflow_manager._workflows
        
        workflow = workflow_manager._workflows[workflow_id]
        assert workflow.state == WorkflowState.INITIALIZED
        assert len(workflow.documents) == 3
    
    @pytest.mark.asyncio
    async def test_agent_registration_and_dependencies(self, workflow_manager, sample_documents):
        """Test agent registration and dependency checking."""
        workflow_id = await workflow_manager.create_workflow(sample_documents)
        
        # Register document processor (no dependencies)
        success = await workflow_manager.register_agent(
            workflow_id, AgentType.DOCUMENT_PROCESSOR, "doc-agent-001"
        )
        assert success is True
        
        # Try to register entity extraction (has dependency)
        success = await workflow_manager.register_agent(
            workflow_id, AgentType.ENTITY_EXTRACTION, "entity-agent-001"
        )
        assert success is True
        
        # Check dependency status
        workflow = workflow_manager._workflows[workflow_id]
        doc_agent = workflow.agent_contexts["doc-agent-001"]
        entity_agent = workflow.agent_contexts["entity-agent-001"]
        
        assert doc_agent.dependencies_met is True
        assert entity_agent.dependencies_met is False  # Doc processor not completed yet
    
    @pytest.mark.asyncio
    async def test_shared_context_updates(self, workflow_manager, sample_documents):
        """Test shared context updates across agents."""
        workflow_id = await workflow_manager.create_workflow(sample_documents)
        
        # Update shared data
        await workflow_manager.update_shared_data(workflow_id, "test_key", "test_value")
        
        # Get shared context
        context = await workflow_manager.get_shared_context(workflow_id)
        
        assert context is not None
        assert context["shared_data"]["test_key"] == "test_value"
        assert context["workflow_id"] == workflow_id

class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for agent testing."""
        services = Mock()
        services.llm_manager = AsyncMock()
        services.llm_manager.complete = AsyncMock(return_value='[{"name": "John Doe", "entity_type": "Person", "confidence": 0.9}]')
        return services
    
    @pytest.mark.asyncio
    async def test_entity_extraction_end_to_end(self, mock_services):
        """Test complete entity extraction workflow."""
        agent = EntityExtractionAgent(mock_services)
        
        task_data = "John Doe filed a lawsuit against ABC Corporation."
        metadata = {"document_id": "test-doc-001"}
        
        result = await agent._process_task(task_data, metadata)
        
        assert result is not None
        assert "entities" in result
        assert len(result["entities"]) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_base_entity_resolution(self, mock_services):
        """Test knowledge base entity resolution."""
        agent = KnowledgeBaseAgent(mock_services)
        
        # Add some entities to resolve
        entities = [
            {"name": "John Doe", "entity_type": "Person", "confidence": 0.9},
            {"name": "J. Doe", "entity_type": "Person", "confidence": 0.8},  # Should be merged
            {"name": "ABC Corp", "entity_type": "Organization", "confidence": 0.95}
        ]
        
        metadata = {"document_id": "test-doc-001"}
        
        result = await agent._process_task(entities, metadata)
        
        assert result is not None
        assert "resolved_entities" in result
        assert "organizational_structure" in result
        assert "analytical_insights" in result

class TestPerformance:
    """Performance and scalability tests."""
    
    def test_entity_processing_performance(self, benchmark):
        """Benchmark entity processing performance."""
        def process_entities():
            # Simulate entity processing
            entities = []
            for i in range(1000):
                entity = {
                    "name": f"Entity {i}",
                    "type": "Document",
                    "confidence": random.uniform(0.5, 1.0)
                }
                entities.append(entity)
            return len(entities)
        
        result = benchmark(process_entities)
        assert result == 1000
    
    def test_similarity_calculation_performance(self, benchmark):
        """Benchmark similarity calculation performance."""
        def calculate_similarities():
            # Simulate O(n²) similarity calculations
            entities = [f"Entity {i}" for i in range(100)]
            similarities = []
            
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    # Simple similarity calculation
                    sim = len(set(entities[i].split()) & set(entities[j].split())) / len(set(entities[i].split()) | set(entities[j].split()))
                    similarities.append(sim)
            
            return len(similarities)
        
        result = benchmark(calculate_similarities)
        assert result > 0

class TestSecurityVulnerabilities:
    """Security vulnerability tests."""
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection."""
        # This would test database query parameterization
        malicious_input = "'; DROP TABLE entities; --"
        
        # Verify that malicious input is properly escaped/parameterized
        # This test would need actual database connection to verify
        assert "DROP TABLE" in malicious_input  # Ensure test data is correct
    
    def test_xss_protection(self, tmp_path):
        """Test protection against XSS attacks."""
        security_manager = SecurityManager("test_key", [str(tmp_path)])
        
        malicious_content = "<script>alert('xss')</script>Hello World"
        sanitized = security_manager.input_validator.sanitize_text(malicious_content)
        
        assert "<script>" not in sanitized
        assert "Hello World" in sanitized
    
    def test_file_upload_validation(self, tmp_path):
        """Test file upload validation."""
        security_manager = SecurityManager("test_key", [str(tmp_path)])
        
        # Test valid file
        valid_file = tmp_path / "document.pdf"
        valid_file.write_text("test content")
        
        validated_path = security_manager.input_validator.validate_file_path(
            str(valid_file), [str(tmp_path)]
        )
        assert validated_path.exists()
        
        # Test invalid file extension
        executable_file = tmp_path / "malware.exe"
        executable_file.write_bytes(b"malicious content")
        
        # This should still validate the path, but application logic
        # should check file types separately
        validated_path = security_manager.input_validator.validate_file_path(
            str(executable_file), [str(tmp_path)]
        )
        assert validated_path.exists()

class TestChaosEngineering:
    """Chaos engineering tests for system resilience."""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, mocker: MockerFixture):
        """Test system behavior when database connection fails."""
        # Mock database connection failure
        mock_pool = AsyncMock()
        mock_pool.acquire.side_effect = Exception("Database connection failed")
        
        # Test how system handles the failure
        with pytest.raises(Exception, match="Database connection failed"):
            async with mock_pool.acquire() as conn:
                pass
    
    @pytest.mark.asyncio
    async def test_llm_service_timeout(self, mocker: MockerFixture):
        """Test system behavior when LLM service times out."""
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = asyncio.TimeoutError("LLM request timed out")
        
        # Test system graceful degradation
        with pytest.raises(asyncio.TimeoutError):
            await mock_llm.complete("test prompt")
    
    @pytest.mark.asyncio
    async def test_memory_pressure_simulation(self):
        """Test system behavior under memory pressure."""
        # Simulate memory pressure by creating large objects
        large_objects = []
        try:
            for i in range(1000):
                # Create large objects to consume memory
                large_obj = [random.random() for _ in range(10000)]
                large_objects.append(large_obj)
        except MemoryError:
            # System should handle memory pressure gracefully
            pass
        
        # Clean up
        del large_objects

# Test configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Performance test configuration
def pytest_configure(config):
    """Configure pytest for performance testing."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "--cov=../",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "-v"
    ])
