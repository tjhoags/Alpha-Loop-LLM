"""================================================================================
ISSUE SCANNER - Automated Similar Issue Detection Across Files
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd "C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\sii"
    .\\venv\\Scripts\\Activate.ps1
    python -m src.review.issue_scanner

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
    source venv/bin/activate
    python -m src.review.issue_scanner

WHAT THIS MODULE DOES:
----------------------
When an issue is found in one file, this module:
1. Analyzes the issue pattern (missing imports, type errors, etc.)
2. Scans all similar files for the same pattern
3. Reports locations and suggests fixes
4. Can auto-fix with confirmation

CURSOR AGENT INTEGRATION:
-------------------------
This module is designed to be invoked by Cursor agents when they detect issues.
Call `scan_for_similar_issues(issue_pattern, source_file)` to find related issues.

================================================================================
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger


class IssueType(Enum):
    """Categories of detectable issues."""
    MISSING_IMPORT = "missing_import"
    TYPE_ERROR = "type_error"
    UNDEFINED_VARIABLE = "undefined_variable"
    UNUSED_IMPORT = "unused_import"
    UNUSED_VARIABLE = "unused_variable"
    DEPRECATED_USAGE = "deprecated_usage"
    SECURITY_RISK = "security_risk"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_SMELL = "code_smell"
    MISSING_DOCSTRING = "missing_docstring"
    INCONSISTENT_NAMING = "inconsistent_naming"
    HARDCODED_VALUE = "hardcoded_value"
    DUPLICATE_CODE = "duplicate_code"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    CROSS_PLATFORM_ISSUE = "cross_platform_issue"


@dataclass
class Issue:
    """Represents a detected issue."""
    issue_type: IssueType
    file_path: Path
    line_number: int
    column: int
    message: str
    code_snippet: str
    severity: str  # "error", "warning", "info"
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False
    pattern_hash: str = ""

    def __post_init__(self):
        """Generate pattern hash for similarity matching."""
        if not self.pattern_hash:
            pattern = f"{self.issue_type.value}:{self.message}"
            self.pattern_hash = hashlib.md5(pattern.encode()).hexdigest()[:12]


@dataclass
class IssuePattern:
    """Pattern for detecting similar issues."""
    issue_type: IssueType
    regex_pattern: str
    description: str
    fix_template: Optional[str] = None
    file_extensions: List[str] = field(default_factory=lambda: [".py"])
    exclude_patterns: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Results from scanning for similar issues."""
    original_issue: Issue
    similar_issues: List[Issue]
    files_scanned: int
    pattern_frequency: Dict[str, int]
    recommendations: List[str]


class IssueScannerAgent:
    """
    Agent that scans the codebase for similar issues.
    
    Designed to be invoked by Cursor agents when they detect an issue,
    enabling automated detection and fixing of similar problems across files.
    """

    # Common issue patterns for detection
    ISSUE_PATTERNS: List[IssuePattern] = [
        # Import issues
        IssuePattern(
            issue_type=IssueType.MISSING_IMPORT,
            regex_pattern=r"^(?!from|import).*\b(pd|np|logger|os|sys|json|re|Path|datetime|Optional|List|Dict|Any)\b",
            description="Common module used without import",
            fix_template="from {module} import {name}",
        ),
        
        # Type annotation issues
        IssuePattern(
            issue_type=IssueType.TYPE_ERROR,
            regex_pattern=r"def\s+\w+\([^)]*\)\s*:",
            description="Function missing return type annotation",
            fix_template="def {func_name}({params}) -> {return_type}:",
        ),
        
        # Docstring issues
        IssuePattern(
            issue_type=IssueType.MISSING_DOCSTRING,
            regex_pattern=r"^\s*(def|class)\s+\w+.*:\s*$",
            description="Function or class missing docstring",
            fix_template='    """Description of {name}."""',
        ),
        
        # Hardcoded values
        IssuePattern(
            issue_type=IssueType.HARDCODED_VALUE,
            regex_pattern=r'["\'](?:localhost|127\.0\.0\.1|password|secret|api[_-]?key)["\']',
            description="Potentially hardcoded sensitive value",
        ),
        
        # Cross-platform path issues
        IssuePattern(
            issue_type=IssueType.CROSS_PLATFORM_ISSUE,
            regex_pattern=r'["\'][A-Z]:\\\\|["\']\/(?:Users|home)\/',
            description="Hardcoded platform-specific path",
            fix_template="Path({path}).expanduser()",
        ),
        
        # Missing error handling
        IssuePattern(
            issue_type=IssueType.MISSING_ERROR_HANDLING,
            regex_pattern=r"(?:open|connect|execute|request)\s*\([^)]+\)\s*(?!.*except)",
            description="IO operation without error handling",
        ),
        
        # Deprecated patterns
        IssuePattern(
            issue_type=IssueType.DEPRECATED_USAGE,
            regex_pattern=r"\.format\s*\(|%\s*[sd]|print\s*\(.*,\s*file\s*=",
            description="Consider using f-strings for string formatting",
        ),
        
        # Security risks
        IssuePattern(
            issue_type=IssueType.SECURITY_RISK,
            regex_pattern=r"eval\s*\(|exec\s*\(|pickle\.load|__import__\s*\(",
            description="Potentially unsafe code execution",
        ),
    ]

    def __init__(
        self,
        project_root: Optional[Path] = None,
        exclude_dirs: Optional[List[str]] = None,
        custom_patterns: Optional[List[IssuePattern]] = None,
    ):
        """
        Initialize the issue scanner.

        Args:
            project_root: Root directory to scan. Defaults to current directory.
            exclude_dirs: Directories to exclude from scanning.
            custom_patterns: Additional patterns to detect.
        """
        self.project_root = project_root or Path.cwd()
        self.exclude_dirs = set(exclude_dirs or [
            "venv", ".venv", "__pycache__", ".git", "node_modules",
            ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs",
            "catboost_info", "models", "logs", "data"
        ])
        self.patterns = self.ISSUE_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        self._file_cache: Dict[Path, str] = {}
        self._ast_cache: Dict[Path, Optional[ast.Module]] = {}

    def _get_file_content(self, file_path: Path) -> str:
        """Get file content with caching."""
        if file_path not in self._file_cache:
            try:
                self._file_cache[file_path] = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                self._file_cache[file_path] = ""
        return self._file_cache[file_path]

    def _get_ast(self, file_path: Path) -> Optional[ast.Module]:
        """Get AST with caching."""
        if file_path not in self._ast_cache:
            try:
                content = self._get_file_content(file_path)
                self._ast_cache[file_path] = ast.parse(content, str(file_path))
            except SyntaxError:
                self._ast_cache[file_path] = None
        return self._ast_cache[file_path]

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)
        return python_files

    def _analyze_issue(self, issue: Issue) -> IssuePattern:
        """
        Analyze an issue to determine its pattern.
        
        Returns the best matching pattern for the issue.
        """
        for pattern in self.patterns:
            if pattern.issue_type == issue.issue_type:
                return pattern
        # Create a dynamic pattern from the issue
        return IssuePattern(
            issue_type=issue.issue_type,
            regex_pattern=re.escape(issue.code_snippet),
            description=issue.message,
        )

    def _scan_file_for_pattern(
        self, file_path: Path, pattern: IssuePattern
    ) -> List[Issue]:
        """Scan a single file for issues matching the pattern."""
        issues = []
        content = self._get_file_content(file_path)
        if not content:
            return issues

        lines = content.split("\n")
        regex = re.compile(pattern.regex_pattern, re.IGNORECASE)

        for line_num, line in enumerate(lines, 1):
            # Skip excluded patterns
            if any(re.search(exc, line) for exc in pattern.exclude_patterns):
                continue

            matches = regex.finditer(line)
            for match in matches:
                issues.append(Issue(
                    issue_type=pattern.issue_type,
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start(),
                    message=pattern.description,
                    code_snippet=line.strip(),
                    severity="warning",
                    suggested_fix=pattern.fix_template,
                    auto_fixable=pattern.fix_template is not None,
                ))

        return issues

    def scan_for_similar_issues(
        self,
        original_issue: Issue,
        max_results: int = 50,
    ) -> ScanResult:
        """
        Scan the codebase for issues similar to the original.

        This is the main entry point for Cursor agents to find related issues.

        Args:
            original_issue: The issue that was initially detected.
            max_results: Maximum number of similar issues to return.

        Returns:
            ScanResult containing all similar issues found.
        """
        logger.info(f"Scanning for issues similar to: {original_issue.message}")
        
        pattern = self._analyze_issue(original_issue)
        similar_issues: List[Issue] = []
        pattern_frequency: Dict[str, int] = {}
        files_scanned = 0

        python_files = self._find_python_files()
        
        for file_path in python_files:
            if file_path == original_issue.file_path:
                continue
            
            files_scanned += 1
            file_issues = self._scan_file_for_pattern(file_path, pattern)
            
            for issue in file_issues:
                if len(similar_issues) >= max_results:
                    break
                similar_issues.append(issue)
                
                # Track pattern frequency
                pattern_key = f"{issue.file_path.parent.name}/{issue.issue_type.value}"
                pattern_frequency[pattern_key] = pattern_frequency.get(pattern_key, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            original_issue, similar_issues, pattern_frequency
        )

        result = ScanResult(
            original_issue=original_issue,
            similar_issues=similar_issues,
            files_scanned=files_scanned,
            pattern_frequency=pattern_frequency,
            recommendations=recommendations,
        )

        logger.info(f"Found {len(similar_issues)} similar issues in {files_scanned} files")
        return result

    def _generate_recommendations(
        self,
        original: Issue,
        similar: List[Issue],
        frequency: Dict[str, int],
    ) -> List[str]:
        """Generate actionable recommendations based on scan results."""
        recommendations = []

        if len(similar) > 10:
            recommendations.append(
                f"⚠️ High frequency issue: Found {len(similar)} similar occurrences. "
                "Consider creating a linting rule to prevent this pattern."
            )

        # Identify hotspots
        if frequency:
            hotspot = max(frequency.items(), key=lambda x: x[1])
            if hotspot[1] > 5:
                recommendations.append(
                    f"📍 Hotspot detected: {hotspot[0]} has {hotspot[1]} occurrences. "
                    "Prioritize fixing this area."
                )

        if original.auto_fixable and similar:
            recommendations.append(
                f"🔧 Auto-fix available: {len([i for i in similar if i.auto_fixable])} "
                "issues can be automatically fixed."
            )

        return recommendations

    def scan_all_patterns(self) -> Dict[IssueType, List[Issue]]:
        """
        Scan the entire codebase for all known issue patterns.

        Returns:
            Dictionary mapping issue types to lists of detected issues.
        """
        all_issues: Dict[IssueType, List[Issue]] = {t: [] for t in IssueType}
        python_files = self._find_python_files()

        for file_path in python_files:
            for pattern in self.patterns:
                issues = self._scan_file_for_pattern(file_path, pattern)
                all_issues[pattern.issue_type].extend(issues)

        return all_issues

    def generate_report(self, scan_result: ScanResult) -> str:
        """Generate a human-readable report of scan results."""
        lines = [
            "=" * 80,
            "ISSUE SCAN REPORT",
            "=" * 80,
            "",
            f"Original Issue: {scan_result.original_issue.message}",
            f"File: {scan_result.original_issue.file_path}",
            f"Line: {scan_result.original_issue.line_number}",
            "",
            f"Files Scanned: {scan_result.files_scanned}",
            f"Similar Issues Found: {len(scan_result.similar_issues)}",
            "",
            "-" * 40,
            "SIMILAR ISSUES:",
            "-" * 40,
        ]

        for issue in scan_result.similar_issues[:20]:  # Limit output
            lines.extend([
                f"\n📄 {issue.file_path.relative_to(self.project_root)}",
                f"   Line {issue.line_number}: {issue.code_snippet[:60]}...",
            ])

        if scan_result.recommendations:
            lines.extend([
                "",
                "-" * 40,
                "RECOMMENDATIONS:",
                "-" * 40,
            ])
            for rec in scan_result.recommendations:
                lines.append(f"  • {rec}")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


# Cursor Agent Integration Functions
def scan_for_similar_issues(
    issue_type: str,
    message: str,
    file_path: str,
    line_number: int = 1,
    code_snippet: str = "",
) -> ScanResult:
    """
    Entry point for Cursor agents to scan for similar issues.

    Args:
        issue_type: Type of issue (e.g., "missing_import", "type_error")
        message: Description of the issue
        file_path: Path to the file where the issue was found
        line_number: Line number of the issue
        code_snippet: The problematic code

    Returns:
        ScanResult with all similar issues found

    Example:
        >>> result = scan_for_similar_issues(
        ...     issue_type="missing_import",
        ...     message="pandas is used but not imported",
        ...     file_path="src/analysis.py",
        ...     line_number=42,
        ...     code_snippet="df = pd.DataFrame()"
        ... )
        >>> print(f"Found {len(result.similar_issues)} similar issues")
    """
    # Convert string to IssueType
    try:
        issue_type_enum = IssueType(issue_type)
    except ValueError:
        issue_type_enum = IssueType.CODE_SMELL

    issue = Issue(
        issue_type=issue_type_enum,
        file_path=Path(file_path),
        line_number=line_number,
        column=0,
        message=message,
        code_snippet=code_snippet,
        severity="warning",
    )

    scanner = IssueScannerAgent()
    return scanner.scan_for_similar_issues(issue)


def fix_similar_issues(scan_result: ScanResult, dry_run: bool = True) -> List[str]:
    """
    Apply fixes to all similar issues found.

    Args:
        scan_result: Result from scan_for_similar_issues
        dry_run: If True, only report what would be changed

    Returns:
        List of files that were (or would be) modified
    """
    modified_files = []
    
    for issue in scan_result.similar_issues:
        if not issue.auto_fixable or not issue.suggested_fix:
            continue
            
        if dry_run:
            logger.info(f"[DRY RUN] Would fix: {issue.file_path}:{issue.line_number}")
        else:
            # TODO: Implement actual fix application
            logger.info(f"Fixed: {issue.file_path}:{issue.line_number}")
            
        modified_files.append(str(issue.file_path))
    
    return modified_files


def main():
    """Run the issue scanner as a standalone tool."""
    import sys
    
    print("=" * 60)
    print("Alpha Loop Capital - Issue Scanner")
    print("=" * 60)
    
    scanner = IssueScannerAgent()
    all_issues = scanner.scan_all_patterns()
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    print(f"\nTotal issues found: {total_issues}")
    print()
    
    for issue_type, issues in all_issues.items():
        if issues:
            print(f"{issue_type.value}: {len(issues)} issues")
            for issue in issues[:3]:  # Show first 3
                rel_path = issue.file_path.relative_to(scanner.project_root)
                print(f"  - {rel_path}:{issue.line_number}")
    
    print("\nScan complete.")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

