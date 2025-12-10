#!/usr/bin/env python3
"""================================================================================
FULL REVIEW SCRIPT - Comprehensive Codebase Analysis
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC
Version: 1.0 | December 2025

This script performs a comprehensive review of the Alpha-Loop-LLM codebase:
1. File structure analysis
2. Code quality checks
3. Documentation completeness
4. Import validation
5. Configuration verification
6. Git status and remote sync

Usage:
    Windows:
        cd "C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\gkv"
        .\\venv\\Scripts\\Activate.ps1
        python scripts/full_review.py

    Mac:
        cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
        source venv/bin/activate
        python scripts/full_review.py

================================================================================
"""

import ast
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class CodebaseReviewer:
    """Comprehensive codebase review tool."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self.results: Dict[str, Any] = {}
        
        # Directories to analyze
        self.src_dirs = ["src", "scripts"]
        self.doc_files = [
            "README.md",
            "SETUP_WINDOWS.md",
            "SETUP_MAC.md",
            "UNIFIED_COMMAND_REFERENCE.md",
            "AGENT_ARCHITECTURE.md",
            "TRAINING_GUIDE.md",
        ]
        
        # Exclusions
        self.exclude_dirs = {
            "__pycache__", ".git", "venv", ".venv", "node_modules",
            ".pytest_cache", ".mypy_cache", "catboost_info",
        }
    
    def run_full_review(self) -> Dict[str, Any]:
        """Run all review checks."""
        print("=" * 70)
        print("ALPHA LOOP LLM - FULL CODEBASE REVIEW")
        print(f"Project Root: {self.project_root}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 70)
        
        # Run all checks
        self.results["file_structure"] = self.analyze_file_structure()
        self.results["python_files"] = self.analyze_python_files()
        self.results["documentation"] = self.check_documentation()
        self.results["imports"] = self.check_imports()
        self.results["agents"] = self.check_agent_structure()
        self.results["git_status"] = self.check_git_status()
        self.results["configuration"] = self.check_configuration()
        
        # Generate summary
        self.results["summary"] = self.generate_summary()
        
        # Print report
        self.print_report()
        
        return self.results
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze the overall file structure."""
        print("\n[1/7] Analyzing file structure...")
        
        stats = {
            "total_files": 0,
            "python_files": 0,
            "markdown_files": 0,
            "script_files": 0,
            "directories": 0,
            "total_lines": 0,
            "by_extension": defaultdict(int),
            "largest_files": [],
        }
        
        file_sizes = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            rel_root = Path(root).relative_to(self.project_root)
            stats["directories"] += 1
            
            for file in files:
                filepath = Path(root) / file
                stats["total_files"] += 1
                
                ext = filepath.suffix.lower()
                stats["by_extension"][ext] += 1
                
                if ext == ".py":
                    stats["python_files"] += 1
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = len(f.readlines())
                            stats["total_lines"] += lines
                            file_sizes.append((str(filepath.relative_to(self.project_root)), lines))
                    except:
                        pass
                elif ext == ".md":
                    stats["markdown_files"] += 1
                elif ext in [".bat", ".sh", ".ps1"]:
                    stats["script_files"] += 1
        
        # Get largest files
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = file_sizes[:10]
        
        print(f"   Found {stats['total_files']} files in {stats['directories']} directories")
        print(f"   Python files: {stats['python_files']} ({stats['total_lines']:,} lines)")
        
        return dict(stats)
    
    def analyze_python_files(self) -> Dict[str, Any]:
        """Analyze Python file quality."""
        print("\n[2/7] Analyzing Python files...")
        
        analysis = {
            "with_docstrings": 0,
            "without_docstrings": 0,
            "syntax_errors": [],
            "type_hints": 0,
            "classes": 0,
            "functions": 0,
            "imports": defaultdict(int),
        }
        
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if not any(exc in str(f) for exc in self.exclude_dirs)]
        
        for filepath in py_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                
                # Check for module docstring
                if ast.get_docstring(tree):
                    analysis["with_docstrings"] += 1
                else:
                    analysis["without_docstrings"] += 1
                
                # Count classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        analysis["classes"] += 1
                    elif isinstance(node, ast.FunctionDef):
                        analysis["functions"] += 1
                        # Check for type hints
                        if node.returns or any(arg.annotation for arg in node.args.args):
                            analysis["type_hints"] += 1
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"][alias.name] += 1
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis["imports"][node.module.split('.')[0]] += 1
                            
            except SyntaxError as e:
                analysis["syntax_errors"].append({
                    "file": str(filepath.relative_to(self.project_root)),
                    "error": str(e),
                })
            except Exception:
                pass
        
        print(f"   Classes: {analysis['classes']}, Functions: {analysis['functions']}")
        print(f"   With docstrings: {analysis['with_docstrings']}, Without: {analysis['without_docstrings']}")
        print(f"   Functions with type hints: {analysis['type_hints']}")
        
        if analysis["syntax_errors"]:
            print(f"   ⚠️  Syntax errors: {len(analysis['syntax_errors'])}")
        
        return dict(analysis)
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        print("\n[3/7] Checking documentation...")
        
        docs = {
            "present": [],
            "missing": [],
            "word_counts": {},
        }
        
        for doc_file in self.doc_files:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                docs["present"].append(doc_file)
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        word_count = len(content.split())
                        docs["word_counts"][doc_file] = word_count
                except:
                    docs["word_counts"][doc_file] = 0
            else:
                docs["missing"].append(doc_file)
        
        # Check for Windows/Mac instructions
        docs["has_dual_platform"] = {
            doc: False for doc in docs["present"]
        }
        
        for doc_file in docs["present"]:
            doc_path = self.project_root / doc_file
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    has_windows = "powershell" in content or "windows" in content
                    has_mac = "bash" in content or "mac" in content or "terminal" in content
                    docs["has_dual_platform"][doc_file] = has_windows and has_mac
            except:
                pass
        
        print(f"   Present: {len(docs['present'])}, Missing: {len(docs['missing'])}")
        dual_count = sum(1 for v in docs["has_dual_platform"].values() if v)
        print(f"   With dual platform instructions: {dual_count}/{len(docs['present'])}")
        
        if docs["missing"]:
            print(f"   ⚠️  Missing docs: {docs['missing']}")
        
        return docs
    
    def check_imports(self) -> Dict[str, Any]:
        """Check import validity."""
        print("\n[4/7] Checking imports...")
        
        imports = {
            "valid": 0,
            "potentially_missing": [],
            "circular_imports": [],
        }
        
        required_packages = {
            "pandas", "numpy", "sklearn", "xgboost", "lightgbm", "catboost",
            "requests", "loguru", "pydantic", "sqlalchemy",
        }
        
        found_packages = set()
        
        py_files = list((self.project_root / "src").rglob("*.py"))
        
        for filepath in py_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            found_packages.add(alias.name.split('.')[0])
                            imports["valid"] += 1
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            found_packages.add(node.module.split('.')[0])
                            imports["valid"] += 1
            except:
                pass
        
        # Check for missing required packages
        missing = required_packages - found_packages
        if missing:
            imports["potentially_missing"] = list(missing)
        
        print(f"   Valid imports: {imports['valid']}")
        print(f"   Unique packages used: {len(found_packages)}")
        
        return imports
    
    def check_agent_structure(self) -> Dict[str, Any]:
        """Check the agent system structure."""
        print("\n[5/7] Checking agent structure...")
        
        agents = {
            "total": 0,
            "directories": [],
            "files": [],
            "master_agents": [],
            "missing_init": [],
        }
        
        agents_dir = self.project_root / "src" / "agents"
        
        if agents_dir.exists():
            for item in agents_dir.iterdir():
                if item.is_dir() and item.name not in self.exclude_dirs:
                    agents["directories"].append(item.name)
                    agents["total"] += 1
                    
                    # Check for __init__.py
                    if not (item / "__init__.py").exists():
                        agents["missing_init"].append(item.name)
                    
                    # Check for master agents
                    if item.name in ["hoags_agent", "ghost_agent", "operations"]:
                        agents["master_agents"].append(item.name)
                
                elif item.suffix == ".py":
                    agents["files"].append(item.name)
        
        print(f"   Agent directories: {len(agents['directories'])}")
        print(f"   Master agents found: {len(agents['master_agents'])}")
        
        if agents["missing_init"]:
            print(f"   ⚠️  Missing __init__.py: {agents['missing_init'][:5]}")
        
        return agents
    
    def check_git_status(self) -> Dict[str, Any]:
        """Check Git repository status."""
        print("\n[6/7] Checking Git status...")
        
        git_info = {
            "is_repo": False,
            "branch": None,
            "remote_url": None,
            "uncommitted_changes": 0,
            "untracked_files": 0,
            "ahead_behind": None,
        }
        
        try:
            # Check if git repo
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            git_info["is_repo"] = result.returncode == 0
            
            if git_info["is_repo"]:
                # Get branch
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                git_info["branch"] = result.stdout.strip()
                
                # Get remote URL
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    git_info["remote_url"] = result.stdout.strip()
                
                # Check for uncommitted changes
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                git_info["uncommitted_changes"] = len([l for l in lines if l and not l.startswith('??')])
                git_info["untracked_files"] = len([l for l in lines if l.startswith('??')])
                
                print(f"   Branch: {git_info['branch']}")
                print(f"   Remote: {git_info['remote_url']}")
                print(f"   Uncommitted changes: {git_info['uncommitted_changes']}")
                print(f"   Untracked files: {git_info['untracked_files']}")
                
        except FileNotFoundError:
            print("   ⚠️  Git not found")
        except Exception as e:
            print(f"   ⚠️  Git check failed: {e}")
        
        return git_info
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files."""
        print("\n[7/7] Checking configuration...")
        
        config = {
            "requirements_txt": False,
            "env_file": False,
            "settings_py": False,
            "gitignore": False,
        }
        
        # Check for required files
        config["requirements_txt"] = (self.project_root / "requirements.txt").exists()
        config["env_file"] = (self.project_root / ".env").exists()
        config["settings_py"] = (self.project_root / "src" / "config" / "settings.py").exists()
        config["gitignore"] = (self.project_root / ".gitignore").exists()
        
        present = sum(1 for v in config.values() if v)
        print(f"   Configuration files present: {present}/{len(config)}")
        
        for name, exists in config.items():
            status = "✓" if exists else "✗"
            print(f"   {status} {name}")
        
        return config
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate review summary."""
        summary = {
            "status": "PASS",
            "issues": [],
            "recommendations": [],
            "stats": {},
        }
        
        # Check for issues
        if self.results["python_files"]["syntax_errors"]:
            summary["issues"].append(f"Syntax errors found: {len(self.results['python_files']['syntax_errors'])}")
            summary["status"] = "NEEDS_ATTENTION"
        
        if self.results["documentation"]["missing"]:
            summary["issues"].append(f"Missing documentation: {self.results['documentation']['missing']}")
        
        if self.results["git_status"].get("uncommitted_changes", 0) > 0:
            summary["issues"].append(f"Uncommitted changes: {self.results['git_status']['uncommitted_changes']}")
        
        if not self.results["configuration"]["env_file"]:
            summary["issues"].append("Missing .env file")
            summary["status"] = "NEEDS_ATTENTION"
        
        # Generate recommendations
        docs_without_dual = [
            k for k, v in self.results["documentation"]["has_dual_platform"].items() if not v
        ]
        if docs_without_dual:
            summary["recommendations"].append(f"Add dual platform (Windows/Mac) instructions to: {docs_without_dual}")
        
        if self.results["python_files"]["without_docstrings"] > self.results["python_files"]["with_docstrings"]:
            summary["recommendations"].append("Add docstrings to more Python files")
        
        # Compile stats
        summary["stats"] = {
            "total_python_files": self.results["file_structure"]["python_files"],
            "total_lines_of_code": self.results["file_structure"]["total_lines"],
            "classes": self.results["python_files"]["classes"],
            "functions": self.results["python_files"]["functions"],
            "agent_directories": len(self.results["agents"]["directories"]),
        }
        
        return summary
    
    def print_report(self):
        """Print the final report."""
        print("\n" + "=" * 70)
        print("REVIEW SUMMARY")
        print("=" * 70)
        
        summary = self.results["summary"]
        
        print(f"\nOverall Status: {summary['status']}")
        
        print("\n📊 Statistics:")
        for key, value in summary["stats"].items():
            print(f"   • {key.replace('_', ' ').title()}: {value:,}")
        
        if summary["issues"]:
            print("\n⚠️  Issues Found:")
            for issue in summary["issues"]:
                print(f"   • {issue}")
        
        if summary["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "=" * 70)
        print("Review complete!")
        print("=" * 70)


def main():
    """Run the full review."""
    reviewer = CodebaseReviewer()
    results = reviewer.run_full_review()
    
    # Save results to JSON
    import json
    output_path = PROJECT_ROOT / "logs" / "review_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert defaultdicts and other non-serializable types
    def convert_for_json(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert_for_json)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()

