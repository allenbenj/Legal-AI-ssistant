#!/usr/bin/env python3
"""
Python to Markdown Converter
Scans a folder for .py files and saves copies as .md files with path information
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import argparse
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PyToMdConverter:
    """Converts Python files to Markdown files with metadata."""
    
    def __init__(self, source_dir: str, target_dir: str, include_subdirs: bool = True):
        """
        Initialize the converter.
        
        Args:
            source_dir: Source directory to scan for .py files
            target_dir: Target directory to save .md files
            include_subdirs: Whether to recursively scan subdirectories
        """
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.include_subdirs = include_subdirs
        
        # Validate source directory exists
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized converter: {self.source_dir} -> {self.target_dir}")
    
    def find_python_files(self) -> List[Path]:
        """
        Find all .py files in the source directory.
        
        Returns:
            List of Path objects for .py files
        """
        python_files = []
        
        if self.include_subdirs:
            # Recursively find all .py files
            python_files = list(self.source_dir.rglob("*.py"))
        else:
            # Only scan the immediate directory
            python_files = list(self.source_dir.glob("*.py"))
        
        # Filter out __pycache__ and other common non-source directories
        filtered_files = []
        for file_path in python_files:
            # Skip files in __pycache__, .git, .venv, etc.
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Found {len(filtered_files)} Python files")
        return filtered_files
    
    def get_relative_path(self, file_path: Path) -> str:
        """
        Get the relative path from source directory.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            Relative path as string
        """
        try:
            return str(file_path.relative_to(self.source_dir))
        except ValueError:
            # If file is not under source_dir, return the absolute path
            return str(file_path)
    
    def create_markdown_content(self, py_file: Path) -> str:
        """
        Create markdown content from Python file.
        
        Args:
            py_file: Path to the Python file
            
        Returns:
            Markdown formatted content
        """
        try:
            # Read the Python file content
            with open(py_file, 'r', encoding='utf-8') as f:
                python_content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(py_file, 'r', encoding='latin-1') as f:
                    python_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {py_file}: {e}")
                python_content = f"# Error reading file: {e}"
        
        # Get file metadata
        stat = py_file.stat()
        file_size = stat.st_size
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        relative_path = self.get_relative_path(py_file)
        
        # Create markdown content with metadata header
        markdown_content = f"""# {py_file.name}

## File Information
- **Original Path**: `{py_file}`
- **Relative Path**: `{relative_path}`
- **File Size**: {file_size:,} bytes
- **Last Modified**: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Converted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Source Code

```python
{python_content}
```

---
*Converted from Python to Markdown by PyToMdConverter*
"""
        
        return markdown_content
    
    def get_target_path(self, py_file: Path) -> Path:
        """
        Get the target path for the markdown file.
        
        Args:
            py_file: Source Python file path
            
        Returns:
            Target markdown file path
        """
        # Get relative path to maintain directory structure
        relative_path = self.get_relative_path(py_file)
        
        # Change extension from .py to .md
        md_relative_path = Path(relative_path).with_suffix('.md')
        
        # Create full target path
        target_path = self.target_dir / md_relative_path
        
        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        return target_path
    
    def convert_file(self, py_file: Path) -> bool:
        """
        Convert a single Python file to Markdown.
        
        Args:
            py_file: Path to the Python file
            
        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            # Create markdown content
            markdown_content = self.create_markdown_content(py_file)
            
            # Get target path
            target_path = self.get_target_path(py_file)
            
            # Write markdown file
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Converted: {self.get_relative_path(py_file)} -> {target_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {py_file}: {e}")
            return False
    
    def convert_all(self) -> dict:
        """
        Convert all Python files to Markdown.
        
        Returns:
            Dictionary with conversion statistics
        """
        python_files = self.find_python_files()
        
        if not python_files:
            logger.warning("No Python files found to convert")
            return {"total": 0, "successful": 0, "failed": 0}
        
        successful = 0
        failed = 0
        
        logger.info(f"Starting conversion of {len(python_files)} files...")
        
        for py_file in python_files:
            if self.convert_file(py_file):
                successful += 1
            else:
                failed += 1
        
        stats = {
            "total": len(python_files),
            "successful": successful,
            "failed": failed
        }
        
        logger.info(f"Conversion complete: {successful} successful, {failed} failed")
        return stats
    
    def create_index_file(self, stats: dict) -> None:
        """
        Create an index file listing all converted files.
        
        Args:
            stats: Conversion statistics
        """
        try:
            python_files = self.find_python_files()
            
            index_content = f"""# Python to Markdown Conversion Index

## Conversion Summary
- **Total Files**: {stats['total']}
- **Successfully Converted**: {stats['successful']}
- **Failed**: {stats['failed']}
- **Source Directory**: `{self.source_dir}`
- **Target Directory**: `{self.target_dir}`
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Converted Files

"""
            
            for py_file in sorted(python_files):
                relative_path = self.get_relative_path(py_file)
                md_path = Path(relative_path).with_suffix('.md')
                
                # Check if file was successfully converted
                target_path = self.target_dir / md_path
                status = "✅" if target_path.exists() else "❌"
                
                index_content += f"- {status} [{relative_path}]({md_path})\n"
            
            # Write index file
            index_path = self.target_dir / "README.md"
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            logger.info(f"Created index file: {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to create index file: {e}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert Python files to Markdown with path information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .py files in current directory
  python py_to_md_converter.py . ./markdown_output
  
  # Convert only files in specific directory (no subdirs)
  python py_to_md_converter.py ./src ./docs --no-subdirs
  
  # Convert with verbose logging
  python py_to_md_converter.py ./legal_ai_system ./docs --verbose
        """
    )
    
    parser.add_argument(
        "source_dir",
        help="Source directory to scan for .py files"
    )
    
    parser.add_argument(
        "target_dir", 
        help="Target directory to save .md files"
    )
    
    parser.add_argument(
        "--no-subdirs",
        action="store_true",
        help="Don't recursively scan subdirectories"
    )
    
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Don't create index file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create converter
        converter = PyToMdConverter(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            include_subdirs=not args.no_subdirs
        )
        
        # Convert all files
        stats = converter.convert_all()
        
        # Create index file unless disabled
        if not args.no_index:
            converter.create_index_file(stats)
        
        # Print summary
        print(f"\n🎉 Conversion Complete!")
        print(f"📁 Source: {converter.source_dir}")
        print(f"📁 Target: {converter.target_dir}")
        print(f"📊 Files: {stats['successful']}/{stats['total']} successful")
        
        if stats['failed'] > 0:
            print(f"⚠️  Failed: {stats['failed']} files")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())