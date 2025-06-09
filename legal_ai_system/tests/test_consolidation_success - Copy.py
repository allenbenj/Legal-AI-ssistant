#!/usr/bin/env python3
"""
Test Consolidation Success
Verifies that component consolidation completed successfully.
"""

import sys
from pathlib import Path
import pytest

# Add the legal_ai_system to Python path
sys.path.insert(0, str(Path(__file__).parent / "legal_ai_system"))

def test_canonical_imports():
    """Test that canonical components can be imported."""
    print("=== Testing Canonical Component Imports ===")
    
    success_count = 0
    total_tests = 0
    
    # Test DocumentProcessor (canonical)
    total_tests += 1
    try:
        from legal_ai_system.agents.document_processor_full import DocumentProcessorAgent
        print("✅ DocumentProcessorAgent (canonical) imported successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ DocumentProcessorAgent import failed: {e}")
    
    # Test VectorStore implementations
    total_tests += 1
    try:
        from legal_ai_system.core.enhanced_vector_store import EnhancedVectorStore
        print("✅ EnhancedVectorStore (primary) imported successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ EnhancedVectorStore import failed: {e}")
    
    total_tests += 1
    try:
        from legal_ai_system.core.vector_store_enhanced import VectorStore
        print("✅ VectorStore (production-ready) imported successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ VectorStore import failed: {e}")
    
    # Test VectorStoreManager (abstraction)
    total_tests += 1
    try:
        from legal_ai_system.core.vector_store_manager import VectorStoreManager
        print("✅ VectorStoreManager (abstraction) imported successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ VectorStoreManager import failed: {e}")
    
    # Test UnifiedMemoryManager
    total_tests += 1
    try:
        from legal_ai_system.core.unified_memory_manager import UnifiedMemoryManager
        print("✅ UnifiedMemoryManager (consolidated) imported successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ UnifiedMemoryManager import failed: {e}")
    
    return success_count, total_tests

def test_service_registration():
    """Test that consolidated components are registered in service container."""
    print("\n=== Testing Service Container Registration ===")
    
    try:
        import importlib
        if importlib.util.find_spec("legal_ai_system.core.unified_services") is None:
            pytest.skip("unified_services module not available")

        from legal_ai_system.core.unified_services import get_service_container, register_core_services
        
        # Register services
        register_core_services()
        container = get_service_container()
        
        # Check registrations
        expected_services = [
            "vector_store_manager",
            "unified_memory_manager"
        ]
        
        success_count = 0
        for service_name in expected_services:
            if service_name in container._services:
                metadata = container._services[service_name]
                print(f"✅ {service_name} registered - Priority: {metadata.priority.name}, Dependencies: {metadata.dependencies}")
                success_count += 1
            else:
                print(f"❌ {service_name} not found in service container")
        
        return success_count, len(expected_services)
        
    except Exception as e:
        print(f"❌ Service container test failed: {e}")
        return 0, len(expected_services)

def test_directory_structure():
    """Test that directories are properly organized."""
    print("\n=== Testing Directory Structure ===")
    
    base_path = Path(__file__).parent
    
    checks = [
        ("Production backup removed", not (base_path / "legal_ai_system" / "backup_2025-05-24").exists()),
        ("Backup archived", (base_path / "archive_backup_2025-05-24").exists()),
        ("Legacy components archived", (base_path / "archive_legacy_components").exists()),
        ("Production legal_pipeline removed", not (base_path / "legal_pipeline").exists()),
        ("Canonical DocumentProcessor exists", (base_path / "legal_ai_system" / "agents" / "document_processor_full.py").exists()),
        ("VectorStoreManager exists", (base_path / "legal_ai_system" / "core" / "vector_store_manager.py").exists()),
        ("UnifiedMemoryManager exists", (base_path / "legal_ai_system" / "core" / "unified_memory_manager.py").exists())
    ]
    
    success_count = 0
    for description, condition in checks:
        if condition:
            print(f"✅ {description}")
            success_count += 1
        else:
            print(f"❌ {description}")
    
    return success_count, len(checks)

def main():
    """Run all consolidation tests."""
    print("Component Consolidation Success Test")
    print("=" * 50)
    
    total_passed = 0
    total_tests = 0
    
    # Run tests
    tests = [
        test_canonical_imports,
        test_service_registration,
        test_directory_structure
    ]
    
    for test in tests:
        passed, total = test()
        total_passed += passed
        total_tests += total
    
    print(f"\n=== Consolidation Test Results ===")
    print(f"Passed: {total_passed}/{total_tests}")
    print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 CONSOLIDATION SUCCESSFUL!")
        print("✅ All redundant components consolidated following DRY principle")
        print("✅ Canonical implementations established as source of truth")
        print("✅ Service container integration complete")
        print("✅ Directory structure cleaned and organized")
        return 0
    else:
        print(f"\n⚠️  Consolidation partially complete: {total_tests - total_passed} issues remaining")
        return 1

if __name__ == "__main__":
    sys.exit(main())