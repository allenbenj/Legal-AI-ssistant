#!/usr/bin/env python3
"""
Migration Tool for Legal AI System

This tool helps migrate from the existing scattered codebase to the new 
organized legal_ai_system structure. It analyzes existing files and 
provides consolidation recommendations.

Usage:
    python utils/migration_tool.py --analyze
    python utils/migration_tool.py --migrate --backup
    python utils/migration_tool.py --verify
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class FileAnalysis:
    """Analysis result for a single file"""
    source_path: Path
    suggested_target: Optional[Path]
    file_type: str
    quality_score: float
    dependencies: List[str]
    notes: List[str]
    action: str  # keep, migrate, consolidate, skip

@dataclass
class MigrationPlan:
    """Complete migration plan"""
    source_dirs: List[Path]
    target_dir: Path
    file_analyses: List[FileAnalysis]
    conflicts: List[Dict]
    recommendations: List[str]
    timestamp: str

class LegalAIMigrator:
    """Migrates existing codebase to new organized structure"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("/mnt/e/A_Scripts")
        self.target_dir = self.base_dir / "legal_ai_system"
        self.backup_dir = self.base_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # File type mappings
        self.file_type_map = {
            # Core system files
            "config.py": "config",
            "settings.py": "config", 
            "services.py": "core",
            "log_setup.py": "utils",
            
            # Agent files
            "agent_nodes.py": "agents",
            "agent_memory.py": "agents",
            "legal_agents.py": "agents",
            "document_processor.py": "agents",
            "violation_review.py": "agents",
            
            # GUI files
            "main_gui.py": "gui",
            "main_gui_refactored.py": "gui",
            "knowledge_graph_gui.py": "gui",
            "violation_review_gui.py": "gui",
            "memory_brain_gui.py": "gui",
            
            # Storage files
            "vector_store.py": "core",
            "memory_store.py": "core", 
            "knowledge_graph_builder.py": "core",
            
            # Utility files
            "document_utils.py": "utils",
            "embedding_utils.py": "utils",
            "classification_utils.py": "utils",
            
            # LangGraph files
            "langgraph_setup.py": "core",
            "langgraph_mcp_integration.py": "core",
            
            # Test files
            "test_*.py": "tests"
        }
        
        # Quality indicators
        self.quality_indicators = {
            "has_docstrings": 10,
            "has_type_hints": 8,
            "has_tests": 15,
            "recent_modified": 5,
            "proper_imports": 5,
            "error_handling": 8,
            "async_support": 7
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def discover_source_files(self) -> List[Path]:
        """Discover all relevant Python files in the source directories"""
        source_dirs = [
            self.base_dir / "Grok/L_Agent _Code_Store/Law-Parser",
            self.base_dir / "Grok/L_Agent _Code_Store",
            self.base_dir / "Backup/Scripts",
            self.base_dir / "legal_pipeline",
            self.base_dir / "Project_Alpha",
            self.base_dir / "Grok/Gemini"
        ]
        
        files = []
        for source_dir in source_dirs:
            if source_dir.exists():
                for py_file in source_dir.rglob("*.py"):
                    if not self._should_skip_file(py_file):
                        files.append(py_file)
        
        return files
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            "__pycache__",
            ".pyc",
            "test_delete",
            "backup",
            "temp",
            ".git"
        ]
        
        path_str = str(file_path).lower()
        return any(pattern in path_str for pattern in skip_patterns)
    
    def analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single file for migration"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Determine file type and target location
            file_type, target_path = self._classify_file(file_path, content)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(content, file_path)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(content)
            
            # Generate notes and action
            notes, action = self._generate_analysis_notes(file_path, content, quality_score)
            
            return FileAnalysis(
                source_path=file_path,
                suggested_target=target_path,
                file_type=file_type,
                quality_score=quality_score,
                dependencies=dependencies,
                notes=notes,
                action=action
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return FileAnalysis(
                source_path=file_path,
                suggested_target=None,
                file_type="unknown",
                quality_score=0.0,
                dependencies=[],
                notes=[f"Analysis failed: {e}"],
                action="skip"
            )
    
    def _classify_file(self, file_path: Path, content: str) -> Tuple[str, Optional[Path]]:
        """Classify file and suggest target location"""
        filename = file_path.name
        
        # Direct mapping
        for pattern, category in self.file_type_map.items():
            if pattern.endswith("*"):
                if filename.startswith(pattern[:-1]):
                    return category, self._get_target_path(category, filename)
            elif filename == pattern:
                return category, self._get_target_path(category, filename)
        
        # Content-based classification
        if "class.*Agent" in content or "BaseAgent" in content:
            return "agents", self._get_target_path("agents", filename)
        elif "QMainWindow" in content or "QWidget" in content:
            return "gui", self._get_target_path("gui", filename)
        elif "FAISS" in content or "vector" in content.lower():
            return "core", self._get_target_path("core", filename)
        elif "pytest" in content or "test_" in filename:
            return "tests", self._get_target_path("tests", filename)
        elif "def extract" in content or "def process" in content:
            return "utils", self._get_target_path("utils", filename)
        
        return "unknown", None
    
    def _get_target_path(self, category: str, filename: str) -> Path:
        """Get target path for file in new structure"""
        category_map = {
            "core": self.target_dir / "core",
            "agents": self.target_dir / "agents", 
            "gui": self.target_dir / "gui",
            "utils": self.target_dir / "utils",
            "config": self.target_dir / "config",
            "tests": self.target_dir / "tests",
            "storage": self.target_dir / "storage"
        }
        
        target_dir = category_map.get(category, self.target_dir / "misc")
        return target_dir / filename
    
    def _calculate_quality_score(self, content: str, file_path: Path) -> float:
        """Calculate quality score for file"""
        score = 0.0
        
        # Check for docstrings
        if '"""' in content or "'''" in content:
            score += self.quality_indicators["has_docstrings"]
        
        # Check for type hints
        if " -> " in content or ": str" in content or "from typing import" in content:
            score += self.quality_indicators["has_type_hints"]
        
        # Check for proper imports
        if "import " in content and not content.startswith("import sys"):
            score += self.quality_indicators["proper_imports"]
        
        # Check for error handling
        if "try:" in content and "except" in content:
            score += self.quality_indicators["error_handling"]
        
        # Check for async support
        if "async def" in content or "await " in content:
            score += self.quality_indicators["async_support"]
        
        # Check modification time (more recent = higher score)
        try:
            mod_time = file_path.stat().st_mtime
            days_old = (datetime.now().timestamp() - mod_time) / (24 * 3600)
            if days_old < 30:  # Modified in last 30 days
                score += self.quality_indicators["recent_modified"]
        except:
            pass
        
        # Normalize to 0-100
        max_score = sum(self.quality_indicators.values())
        return min(100.0, (score / max_score) * 100)
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import dependencies from file"""
        dependencies = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Clean up the import line
                dep = line.split('#')[0].strip()  # Remove comments
                dependencies.append(dep)
        
        return dependencies
    
    def _generate_analysis_notes(self, file_path: Path, content: str, quality_score: float) -> Tuple[List[str], str]:
        """Generate analysis notes and recommended action"""
        notes = []
        
        # Quality assessment
        if quality_score > 80:
            notes.append("High quality code - good candidate for migration")
            action = "migrate"
        elif quality_score > 60:
            notes.append("Good quality code - migrate with minor cleanup")
            action = "migrate"
        elif quality_score > 40:
            notes.append("Moderate quality - needs refactoring before migration")
            action = "consolidate"
        else:
            notes.append("Low quality - consider rewriting")
            action = "skip"
        
        # Specific checks
        if "# TODO" in content or "FIXME" in content:
            notes.append("Contains TODOs or FIXMEs")
        
        if len(content.split('\n')) > 1000:
            notes.append("Large file - consider splitting into modules")
        
        if "class " in content and "def " in content:
            class_count = content.count("class ")
            if class_count > 3:
                notes.append(f"Multiple classes ({class_count}) - consider splitting")
        
        # Check for duplicates
        similar_files = self._find_similar_files(file_path)
        if similar_files:
            notes.append(f"Similar files found: {similar_files}")
            action = "consolidate"
        
        return notes, action
    
    def _find_similar_files(self, file_path: Path) -> List[str]:
        """Find files with similar names (potential duplicates)"""
        # This is a simplified implementation
        # In practice, you might want to compare file contents
        base_name = file_path.stem
        parent_dir = file_path.parent
        
        similar = []
        for other_file in parent_dir.glob("*.py"):
            if other_file != file_path and base_name in other_file.stem:
                similar.append(other_file.name)
        
        return similar
    
    def create_migration_plan(self) -> MigrationPlan:
        """Create comprehensive migration plan"""
        self.logger.info("Discovering source files...")
        source_files = self.discover_source_files()
        
        self.logger.info(f"Analyzing {len(source_files)} files...")
        file_analyses = []
        
        for file_path in source_files:
            analysis = self.analyze_file(file_path)
            file_analyses.append(analysis)
            self.logger.debug(f"Analyzed {file_path.name}: {analysis.action} (score: {analysis.quality_score:.1f})")
        
        # Detect conflicts
        conflicts = self._detect_conflicts(file_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_analyses, conflicts)
        
        return MigrationPlan(
            source_dirs=[self.base_dir / "Grok", self.base_dir / "Backup", self.base_dir / "legal_pipeline"],
            target_dir=self.target_dir,
            file_analyses=file_analyses,
            conflicts=conflicts,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _detect_conflicts(self, analyses: List[FileAnalysis]) -> List[Dict]:
        """Detect potential conflicts in migration"""
        conflicts = []
        target_paths = {}
        
        for analysis in analyses:
            if analysis.suggested_target:
                target = str(analysis.suggested_target)
                if target in target_paths:
                    conflicts.append({
                        "type": "duplicate_target",
                        "target": target,
                        "sources": [target_paths[target], str(analysis.source_path)],
                        "recommendation": "Choose best quality file or consolidate"
                    })
                else:
                    target_paths[target] = str(analysis.source_path)
        
        return conflicts
    
    def _generate_recommendations(self, analyses: List[FileAnalysis], conflicts: List[Dict]) -> List[str]:
        """Generate migration recommendations"""
        recommendations = []
        
        # Statistics
        total_files = len(analyses)
        migrate_count = len([a for a in analyses if a.action == "migrate"])
        consolidate_count = len([a for a in analyses if a.action == "consolidate"])
        skip_count = len([a for a in analyses if a.action == "skip"])
        
        recommendations.append(f"Migration Summary:")
        recommendations.append(f"  - Total files analyzed: {total_files}")
        recommendations.append(f"  - Ready to migrate: {migrate_count}")
        recommendations.append(f"  - Need consolidation: {consolidate_count}")
        recommendations.append(f"  - Recommend skipping: {skip_count}")
        recommendations.append(f"  - Conflicts detected: {len(conflicts)}")
        
        # High priority recommendations
        high_quality = [a for a in analyses if a.quality_score > 80]
        if high_quality:
            recommendations.append(f"\nHigh Priority Migration (Quality > 80):")
            for analysis in high_quality[:5]:  # Top 5
                recommendations.append(f"  - {analysis.source_path.name} → {analysis.suggested_target}")
        
        # Consolidation opportunities
        consolidate_files = [a for a in analyses if a.action == "consolidate"]
        if consolidate_files:
            recommendations.append(f"\nConsolidation Opportunities:")
            for analysis in consolidate_files[:3]:  # Top 3
                recommendations.append(f"  - {analysis.source_path.name}: {'; '.join(analysis.notes)}")
        
        return recommendations
    
    def save_migration_plan(self, plan: MigrationPlan, output_file: Path = None) -> Path:
        """Save migration plan to JSON file"""
        if not output_file:
            output_file = self.base_dir / f"migration_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        plan_dict = asdict(plan)
        
        # Convert Path objects to strings
        for analysis in plan_dict['file_analyses']:
            if analysis['source_path']:
                analysis['source_path'] = str(analysis['source_path'])
            if analysis['suggested_target']:
                analysis['suggested_target'] = str(analysis['suggested_target'])
        
        plan_dict['source_dirs'] = [str(d) for d in plan_dict['source_dirs']]
        plan_dict['target_dir'] = str(plan_dict['target_dir'])
        
        with open(output_file, 'w') as f:
            json.dump(plan_dict, f, indent=2)
        
        self.logger.info(f"Migration plan saved to {output_file}")
        return output_file
    
    def execute_migration(self, plan: MigrationPlan, create_backup: bool = True) -> bool:
        """Execute the migration plan"""
        try:
            if create_backup:
                self.logger.info(f"Creating backup at {self.backup_dir}")
                self._create_backup()
            
            self.logger.info("Executing migration...")
            
            # Create target directories
            self._create_target_structure()
            
            # Migrate files
            migrated = 0
            for analysis in plan.file_analyses:
                if analysis.action == "migrate" and analysis.suggested_target:
                    try:
                        self._migrate_file(analysis.source_path, analysis.suggested_target)
                        migrated += 1
                    except Exception as e:
                        self.logger.error(f"Failed to migrate {analysis.source_path}: {e}")
            
            self.logger.info(f"Successfully migrated {migrated} files")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False
    
    def _create_backup(self) -> None:
        """Create backup of existing files"""
        source_dirs = [
            self.base_dir / "Grok/L_Agent _Code_Store",
            self.base_dir / "Backup/Scripts",
            self.base_dir / "legal_pipeline"
        ]
        
        for source_dir in source_dirs:
            if source_dir.exists():
                target_backup = self.backup_dir / source_dir.name
                shutil.copytree(source_dir, target_backup)
    
    def _create_target_structure(self) -> None:
        """Create target directory structure"""
        directories = [
            "core", "agents", "gui", "utils", "config", 
            "tests", "storage", "storage/databases", 
            "storage/vectors", "storage/documents"
        ]
        
        for directory in directories:
            (self.target_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _migrate_file(self, source: Path, target: Path) -> None:
        """Migrate a single file"""
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.logger.debug(f"Migrated {source.name} → {target}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Legal AI System Migration Tool")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing codebase")
    parser.add_argument("--migrate", action="store_true", help="Execute migration")
    parser.add_argument("--plan-file", help="Load migration plan from file")
    parser.add_argument("--backup", action="store_true", help="Create backup before migration")
    parser.add_argument("--output", help="Output file for migration plan")
    parser.add_argument("--base-dir", help="Base directory (default: /mnt/e/A_Scripts)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else Path("/mnt/e/A_Scripts")
    migrator = LegalAIMigrator(base_dir)
    
    if args.analyze:
        print("Analyzing existing codebase...")
        plan = migrator.create_migration_plan()
        
        # Save plan
        output_file = Path(args.output) if args.output else None
        plan_file = migrator.save_migration_plan(plan, output_file)
        
        # Print summary
        print(f"\nMigration Analysis Complete!")
        print(f"Plan saved to: {plan_file}")
        print("\nRecommendations:")
        for rec in plan.recommendations:
            print(rec)
    
    elif args.migrate:
        if args.plan_file:
            print(f"Loading migration plan from {args.plan_file}")
            # Implementation for loading plan would go here
        else:
            print("Creating new migration plan...")
            plan = migrator.create_migration_plan()
        
        if input("Proceed with migration? (y/N): ").lower() == 'y':
            success = migrator.execute_migration(plan, args.backup)
            print("Migration completed successfully!" if success else "Migration failed!")
        else:
            print("Migration cancelled.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()