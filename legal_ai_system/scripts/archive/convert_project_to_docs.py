#!/usr/bin/env python3
"""
Project Documentation Generator
Example script showing how to use PyToMdConverter to document the entire legal_ai_system
"""

from pathlib import Path
from legal_ai_system.legacy_extras.py_to_md_converter import PyToMdConverter
import logging

def main():
    """Convert the entire legal_ai_system to markdown documentation."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent  # Go up to legal_ai_system directory
    source_dir = project_root
    docs_dir = project_root / "docs" / "source_code"
    
    print(f"🔍 Converting Python project to documentation...")
    print(f"📂 Source: {source_dir}")
    print(f"📝 Target: {docs_dir}")
    
    try:
        # Create converter
        converter = PyToMdConverter(
            source_dir=str(source_dir),
            target_dir=str(docs_dir),
            include_subdirs=True
        )
        
        # Convert all Python files
        stats = converter.convert_all()
        
        # Create index file
        converter.create_index_file(stats)
        
        print(f"\n✅ Documentation generation complete!")
        print(f"📊 Statistics:")
        print(f"   • Total files found: {stats['total']}")
        print(f"   • Successfully converted: {stats['successful']}")
        print(f"   • Failed conversions: {stats['failed']}")
        print(f"📁 Documentation saved to: {docs_dir}")
        print(f"🔗 Open {docs_dir / 'README.md'} to see the index")
        
        return 0 if stats['failed'] == 0 else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    # Enable info-level logging
    logging.basicConfig(level=logging.INFO)
    
    exit(main())