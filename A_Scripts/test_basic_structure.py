#!/usr/bin/env python3
"""
Basic Structure Test for Legal AI System
========================================
Tests core modules without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_models_structure():
    """Test that models.py has correct structure."""
    print("✓ Testing models structure...")
    try:
        # Test direct import without dependencies
        import legal_ai_system.core.models as models
        
        # Check key classes exist
        assert hasattr(models, 'LegalDocument')
        assert hasattr(models, 'ExtractedEntity') 
        assert hasattr(models, 'ProcessingResult')
        assert hasattr(models, 'EntityType')
        assert hasattr(models, 'DocumentStatus')
        
        print("  ✓ All model classes found")
        
        # Test creating instances
        doc = models.LegalDocument()
        assert doc.id is not None
        print("  ✓ LegalDocument creation works")
        
        entity = models.ExtractedEntity(
            text="test", 
            entity_type=models.EntityType.PERSON,
            confidence=0.8
        )
        assert entity.confidence == 0.8
        print("  ✓ ExtractedEntity creation works")
        
        return True
    except Exception as e:
        print(f"  ✗ Models test failed: {e}")
        return False

def test_file_structure():
    """Test that required files and directories exist."""
    print("✓ Testing file structure...")
    
    required_files = [
        "legal_ai_system/__init__.py",
        "legal_ai_system/core/models.py",
        "legal_ai_system/core/base_agent.py", 
        "legal_ai_system/core/unified_exceptions.py",
        "legal_ai_system/extraction/hybrid_extractor.py",
        "legal_ai_system/memory/unified_memory_manager.py",
        "legal_ai_system/main.py",
        "legal_ai_system/requirements.txt"
    ]
    
    required_dirs = [
        "legal_ai_system/storage/documents",
        "legal_ai_system/storage/databases", 
        "legal_ai_system/storage/vectors",
        "legal_ai_system/logs",
        "legal_ai_system/models"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"  ✓ {dir_path}/")
    
    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False
        
    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False
    
    return True

def test_main_compilation():
    """Test that main.py compiles without syntax errors."""
    print("✓ Testing main.py compilation...")
    try:
        import py_compile
        py_compile.compile('legal_ai_system/main.py', doraise=True)
        print("  ✓ main.py compiles successfully")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ✗ main.py compilation failed: {e}")
        return False
    except FileNotFoundError as e:
        print(f"  ✗ main.py not found: {e}")
        return False

def main():
    """Run all basic structure tests."""
    print("Legal AI System - Basic Structure Test")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Main Compilation", test_main_compilation), 
        ("Models Structure", test_models_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED")
    
    print(f"\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic structure tests PASSED!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test full system: python main.py")
        print("3. Configure LLM providers and API keys")
        return True
    else:
        print("❌ Some tests FAILED - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)