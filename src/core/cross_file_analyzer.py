"""================================================================================
CROSS-FILE ANALYZER - Issue Detection and Propagation System
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC
Version: 1.0 | December 2025

This module provides functionality for:
1. Detecting issues/patterns in code files
2. Finding similar issues across the codebase
3. Suggesting or automatically applying fixes
4. Tracking issue propagation and resolution

When an issue is found in one file, this system automatically scans for
similar issues in other files and can apply consistent fixes.

================================================================================
"""

import ast
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of issues that can be detected."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TYPE_MISMATCH = "type_mismatch"
    UNDEFINED_VARIABLE = "undefined_variable"
    UNUSED_IMPORT = "unused_import"
    DEPRECATED_USAGE = "deprecated_usage"
    INCONSISTENT_NAMING = "inconsistent_naming"
    MISSING_DOCSTRING = "missing_docstring"
    HARDCODED_VALUE = "hardcoded_value"
    DUPLICATE_CODE = "duplicate_code"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    STYLE_VIOLATION = "style_violation"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    CONFIGURATION_ISSUE = "configuration_issue"
    CUSTOM = "custom"


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"  # Must fix immediately
    HIGH = "high"          # Fix before deployment
    MEDIUM = "medium"      # Should fix soon
    LOW = "low"            # Nice to fix
    INFO = "info"          # Informational only


class FixStatus(Enum):
    """Status of issue fixes."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IssuePattern:
    """A pattern that identifies a specific type of issue."""
    pattern_id: str
    issue_type: IssueType
    description: str
    regex_pattern: Optional[str] = None
    ast_pattern: Optional[str] = None
    file_extensions: List[str] = field(default_factory=lambda: [".py"])
    severity: IssueSeverity = IssueSeverity.MEDIUM
    fix_template: Optional[str] = None
    auto_fixable: bool = False
    
    def matches_file(self, filepath: str) -> bool:
        """Check if this pattern applies to the given file."""
        return any(filepath.endswith(ext) for ext in self.file_extensions)


@dataclass
class DetectedIssue:
    """A detected issue in a file."""
    issue_id: str
    pattern_id: str
    issue_type: IssueType
    severity: IssueSeverity
    filepath: str
    line_number: int
    column: Optional[int] = None
    description: str = ""
    code_snippet: str = ""
    suggested_fix: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    fix_status: FixStatus = FixStatus.PENDING
    
    def to_dict(self) -> Dict:
        return {
            "issue_id": self.issue_id,
            "pattern_id": self.pattern_id,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "filepath": self.filepath,
            "line_number": self.line_number,
            "description": self.description,
            "code_snippet": self.code_snippet,
            "suggested_fix": self.suggested_fix,
            "fix_status": self.fix_status.value,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class FixResult:
    """Result of applying a fix."""
    issue_id: str
    filepath: str
    success: bool
    original_code: str
    new_code: str
    message: str = ""
    applied_at: datetime = field(default_factory=datetime.now)


class CrossFileAnalyzer:
    """
    Cross-File Analyzer for detecting and fixing similar issues across codebase.
    
    This class enables agents to:
    1. Register issue patterns they've identified
    2. Scan the entire codebase for similar patterns
    3. Apply consistent fixes across all affected files
    4. Track and report on issue resolution
    
    Usage:
        analyzer = CrossFileAnalyzer(project_root="/path/to/project")
        
        # Register a pattern when an issue is found
        pattern = analyzer.register_pattern(
            issue_type=IssueType.DEPRECATED_USAGE,
            description="Using deprecated 'eval()' function",
            regex_pattern=r"eval\s*\(",
            fix_template="ast.literal_eval(",
        )
        
        # Scan codebase for similar issues
        issues = analyzer.scan_for_pattern(pattern.pattern_id)
        
        # Apply fixes
        results = analyzer.apply_fixes(issues, auto_approve=True)
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Cross-File Analyzer.
        
        Args:
            project_root: Root directory of the project to analyze
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Pattern storage
        self.patterns: Dict[str, IssuePattern] = {}
        
        # Issue storage
        self.detected_issues: Dict[str, DetectedIssue] = {}
        
        # Fix history
        self.fix_history: List[FixResult] = []
        
        # Exclusion patterns (directories/files to skip)
        self.exclusions: Set[str] = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".pytest_cache", ".mypy_cache", "dist", "build", "*.pyc",
            "*.egg-info", ".tox", ".coverage", "htmlcov",
        }
        
        # Statistics
        self.scan_count = 0
        self.fix_count = 0
        self.files_analyzed = 0
        
        # Initialize common patterns
        self._init_common_patterns()
        
        logger.info(f"CrossFileAnalyzer initialized for {self.project_root}")
    
    def _init_common_patterns(self):
        """Initialize commonly useful issue patterns."""
        
        # Pattern: Hardcoded passwords/secrets
        self.register_pattern(
            issue_type=IssueType.SECURITY_ISSUE,
            description="Potential hardcoded password or secret",
            regex_pattern=r'(password|secret|api_key|apikey|token)\s*=\s*["\'][^"\']+["\']',
            severity=IssueSeverity.CRITICAL,
            auto_fixable=False,
        )
        
        # Pattern: Print statements (should use logging)
        self.register_pattern(
            issue_type=IssueType.STYLE_VIOLATION,
            description="Using print() instead of logging",
            regex_pattern=r'^\s*print\s*\(',
            severity=IssueSeverity.LOW,
            fix_template="logger.info(",
            auto_fixable=True,
        )
        
        # Pattern: Bare except clause
        self.register_pattern(
            issue_type=IssueType.MISSING_ERROR_HANDLING,
            description="Bare except clause catches all exceptions",
            regex_pattern=r'except\s*:',
            severity=IssueSeverity.MEDIUM,
            fix_template="except Exception:",
            auto_fixable=True,
        )
        
        # Pattern: TODO comments
        self.register_pattern(
            issue_type=IssueType.CUSTOM,
            description="TODO comment found - needs attention",
            regex_pattern=r'#\s*TODO[:\s]',
            severity=IssueSeverity.INFO,
            auto_fixable=False,
        )
        
        # Pattern: FIXME comments
        self.register_pattern(
            issue_type=IssueType.CUSTOM,
            description="FIXME comment found - needs fixing",
            regex_pattern=r'#\s*FIXME[:\s]',
            severity=IssueSeverity.HIGH,
            auto_fixable=False,
        )
        
        # Pattern: Magic numbers
        self.register_pattern(
            issue_type=IssueType.HARDCODED_VALUE,
            description="Magic number detected - consider using named constant",
            regex_pattern=r'(?<!["\'\w])\b(?:1000|100|60|24|7|365|1024|4096)\b(?!["\'\w])',
            severity=IssueSeverity.LOW,
            auto_fixable=False,
        )
        
        # Pattern: SQL injection risk
        self.register_pattern(
            issue_type=IssueType.SECURITY_ISSUE,
            description="Potential SQL injection - use parameterized queries",
            regex_pattern=r'execute\s*\(\s*["\'].*?\%s.*?["\'].*?\%',
            severity=IssueSeverity.CRITICAL,
            auto_fixable=False,
        )
        
        # Pattern: Missing type hints
        self.register_pattern(
            issue_type=IssueType.STYLE_VIOLATION,
            description="Function missing type hints",
            regex_pattern=r'def\s+\w+\s*\([^)]*\)\s*:',
            severity=IssueSeverity.INFO,
            auto_fixable=False,
        )
        
    def register_pattern(
        self,
        issue_type: IssueType,
        description: str,
        regex_pattern: Optional[str] = None,
        ast_pattern: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        fix_template: Optional[str] = None,
        auto_fixable: bool = False,
    ) -> IssuePattern:
        """Register a new issue pattern for detection.
        
        Args:
            issue_type: Type of issue this pattern detects
            description: Human-readable description
            regex_pattern: Regex pattern to match
            ast_pattern: AST pattern to match (for Python)
            file_extensions: File types to scan
            severity: Issue severity level
            fix_template: Template for automatic fix
            auto_fixable: Whether this can be auto-fixed
            
        Returns:
            The created IssuePattern
        """
        pattern_id = hashlib.sha256(
            f"{issue_type.value}{description}{regex_pattern}".encode()
        ).hexdigest()[:12]
        
        pattern = IssuePattern(
            pattern_id=pattern_id,
            issue_type=issue_type,
            description=description,
            regex_pattern=regex_pattern,
            ast_pattern=ast_pattern,
            file_extensions=file_extensions or [".py"],
            severity=severity,
            fix_template=fix_template,
            auto_fixable=auto_fixable,
        )
        
        self.patterns[pattern_id] = pattern
        logger.info(f"Registered pattern: {pattern_id} - {description}")
        
        return pattern
    
    def scan_file(self, filepath: str) -> List[DetectedIssue]:
        """Scan a single file for all registered patterns.
        
        Args:
            filepath: Path to the file to scan
            
        Returns:
            List of detected issues
        """
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return issues
        
        for pattern in self.patterns.values():
            if not pattern.matches_file(filepath):
                continue
                
            if pattern.regex_pattern:
                # Regex-based detection
                try:
                    regex = re.compile(pattern.regex_pattern, re.MULTILINE | re.IGNORECASE)
                    for i, line in enumerate(lines, 1):
                        match = regex.search(line)
                        if match:
                            issue_id = hashlib.sha256(
                                f"{pattern.pattern_id}{filepath}{i}{match.group()}".encode()
                            ).hexdigest()[:16]
                            
                            issue = DetectedIssue(
                                issue_id=issue_id,
                                pattern_id=pattern.pattern_id,
                                issue_type=pattern.issue_type,
                                severity=pattern.severity,
                                filepath=filepath,
                                line_number=i,
                                column=match.start(),
                                description=pattern.description,
                                code_snippet=line.strip(),
                                suggested_fix=pattern.fix_template,
                            )
                            
                            issues.append(issue)
                            self.detected_issues[issue_id] = issue
                            
                except re.error as e:
                    logger.error(f"Invalid regex in pattern {pattern.pattern_id}: {e}")
        
        self.files_analyzed += 1
        return issues
    
    def scan_directory(
        self,
        directory: Optional[str] = None,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
    ) -> List[DetectedIssue]:
        """Scan a directory for all registered patterns.
        
        Args:
            directory: Directory to scan (defaults to project root)
            recursive: Whether to scan subdirectories
            file_extensions: File extensions to include
            
        Returns:
            List of all detected issues
        """
        scan_dir = Path(directory) if directory else self.project_root
        all_issues = []
        
        def should_skip(path: Path) -> bool:
            """Check if path should be skipped."""
            for excl in self.exclusions:
                if excl in str(path):
                    return True
            return False
        
        def scan_path(path: Path):
            if should_skip(path):
                return
                
            if path.is_file():
                if file_extensions:
                    if not any(str(path).endswith(ext) for ext in file_extensions):
                        return
                issues = self.scan_file(str(path))
                all_issues.extend(issues)
                
            elif path.is_dir() and recursive:
                for child in path.iterdir():
                    scan_path(child)
        
        scan_path(scan_dir)
        self.scan_count += 1
        
        logger.info(
            f"Scan complete: {len(all_issues)} issues found in "
            f"{self.files_analyzed} files"
        )
        
        return all_issues
    
    def scan_for_pattern(
        self,
        pattern_id: str,
        directory: Optional[str] = None,
    ) -> List[DetectedIssue]:
        """Scan for a specific pattern across the codebase.
        
        Args:
            pattern_id: ID of the pattern to scan for
            directory: Directory to scan (defaults to project root)
            
        Returns:
            List of issues matching the pattern
        """
        if pattern_id not in self.patterns:
            logger.error(f"Pattern not found: {pattern_id}")
            return []
        
        all_issues = self.scan_directory(directory)
        return [i for i in all_issues if i.pattern_id == pattern_id]
    
    def find_similar_issues(
        self,
        reference_issue: DetectedIssue,
        similarity_threshold: float = 0.8,
    ) -> List[DetectedIssue]:
        """Find issues similar to a reference issue.
        
        When an agent finds and fixes an issue, this method finds similar
        issues in other files that may need the same fix.
        
        Args:
            reference_issue: The issue to find similar ones to
            similarity_threshold: How similar issues must be (0-1)
            
        Returns:
            List of similar issues
        """
        similar = []
        
        # First, scan for the same pattern
        pattern_matches = self.scan_for_pattern(reference_issue.pattern_id)
        
        # Filter out the reference issue itself
        pattern_matches = [
            i for i in pattern_matches
            if i.issue_id != reference_issue.issue_id
        ]
        
        # Add code similarity check
        for issue in pattern_matches:
            similarity = self._calculate_code_similarity(
                reference_issue.code_snippet,
                issue.code_snippet
            )
            if similarity >= similarity_threshold:
                issue.context["similarity_score"] = similarity
                similar.append(issue)
        
        logger.info(
            f"Found {len(similar)} similar issues to {reference_issue.issue_id}"
        )
        
        return similar
    
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets.
        
        Uses a simple Jaccard similarity on tokens.
        """
        def tokenize(code: str) -> Set[str]:
            # Simple tokenization
            return set(re.findall(r'\b\w+\b', code.lower()))
        
        tokens1 = tokenize(code1)
        tokens2 = tokenize(code2)
        
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def apply_fix(
        self,
        issue: DetectedIssue,
        custom_fix: Optional[str] = None,
        backup: bool = True,
    ) -> FixResult:
        """Apply a fix to a detected issue.
        
        Args:
            issue: The issue to fix
            custom_fix: Custom fix code (overrides pattern's fix_template)
            backup: Whether to create a backup of the file
            
        Returns:
            FixResult with details of the fix attempt
        """
        fix_code = custom_fix or issue.suggested_fix
        
        if not fix_code:
            return FixResult(
                issue_id=issue.issue_id,
                filepath=issue.filepath,
                success=False,
                original_code=issue.code_snippet,
                new_code="",
                message="No fix template available",
            )
        
        try:
            # Read file
            with open(issue.filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if backup:
                backup_path = f"{issue.filepath}.bak"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            
            # Get the line to fix
            line_idx = issue.line_number - 1
            if 0 <= line_idx < len(lines):
                original_line = lines[line_idx]
                
                # Apply fix (simple regex replacement)
                pattern = self.patterns.get(issue.pattern_id)
                if pattern and pattern.regex_pattern:
                    new_line = re.sub(
                        pattern.regex_pattern,
                        fix_code,
                        original_line,
                        count=1
                    )
                else:
                    # Direct replacement
                    new_line = original_line.replace(
                        issue.code_snippet,
                        fix_code
                    )
                
                lines[line_idx] = new_line
                
                # Write file
                with open(issue.filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                # Update issue status
                issue.fix_status = FixStatus.APPLIED
                
                result = FixResult(
                    issue_id=issue.issue_id,
                    filepath=issue.filepath,
                    success=True,
                    original_code=original_line.strip(),
                    new_code=new_line.strip(),
                    message="Fix applied successfully",
                )
                
                self.fix_history.append(result)
                self.fix_count += 1
                
                logger.info(f"Fixed issue {issue.issue_id} in {issue.filepath}")
                
                return result
            
        except Exception as e:
            logger.error(f"Error applying fix to {issue.filepath}: {e}")
            issue.fix_status = FixStatus.FAILED
            
            return FixResult(
                issue_id=issue.issue_id,
                filepath=issue.filepath,
                success=False,
                original_code=issue.code_snippet,
                new_code="",
                message=str(e),
            )
        
        return FixResult(
            issue_id=issue.issue_id,
            filepath=issue.filepath,
            success=False,
            original_code=issue.code_snippet,
            new_code="",
            message="Line not found",
        )
    
    def apply_fixes_batch(
        self,
        issues: List[DetectedIssue],
        auto_approve: bool = False,
        dry_run: bool = False,
    ) -> List[FixResult]:
        """Apply fixes to multiple issues.
        
        Args:
            issues: List of issues to fix
            auto_approve: Apply fixes without confirmation
            dry_run: Only show what would be fixed
            
        Returns:
            List of FixResult objects
        """
        results = []
        
        # Filter to auto-fixable issues
        fixable_issues = [
            i for i in issues
            if i.pattern_id in self.patterns
            and self.patterns[i.pattern_id].auto_fixable
            and i.suggested_fix
        ]
        
        logger.info(f"Processing {len(fixable_issues)} auto-fixable issues")
        
        for issue in fixable_issues:
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would fix {issue.filepath}:{issue.line_number} - "
                    f"{issue.description}"
                )
                results.append(FixResult(
                    issue_id=issue.issue_id,
                    filepath=issue.filepath,
                    success=True,
                    original_code=issue.code_snippet,
                    new_code=issue.suggested_fix or "",
                    message="[DRY RUN] Fix would be applied",
                ))
            elif auto_approve:
                result = self.apply_fix(issue)
                results.append(result)
            else:
                logger.info(
                    f"Issue pending approval: {issue.filepath}:{issue.line_number}"
                )
                issue.fix_status = FixStatus.PENDING
        
        return results
    
    def propagate_fix(
        self,
        original_issue: DetectedIssue,
        fix_code: str,
        similarity_threshold: float = 0.8,
        auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """Propagate a fix to similar issues across the codebase.
        
        When an agent fixes an issue, this method:
        1. Finds all similar issues in other files
        2. Applies the same fix (if auto_apply) or suggests it
        
        Args:
            original_issue: The issue that was fixed
            fix_code: The code that fixed the original issue
            similarity_threshold: How similar issues must be
            auto_apply: Whether to automatically apply fixes
            
        Returns:
            Summary of propagation results
        """
        # Find similar issues
        similar_issues = self.find_similar_issues(
            original_issue,
            similarity_threshold
        )
        
        results = {
            "original_issue": original_issue.issue_id,
            "similar_found": len(similar_issues),
            "fixes_applied": 0,
            "fixes_pending": 0,
            "details": [],
        }
        
        for issue in similar_issues:
            if auto_apply:
                fix_result = self.apply_fix(issue, custom_fix=fix_code)
                if fix_result.success:
                    results["fixes_applied"] += 1
                results["details"].append(fix_result)
            else:
                issue.suggested_fix = fix_code
                results["fixes_pending"] += 1
                results["details"].append({
                    "issue_id": issue.issue_id,
                    "filepath": issue.filepath,
                    "line": issue.line_number,
                    "status": "pending_approval",
                })
        
        logger.info(
            f"Fix propagation: {results['fixes_applied']} applied, "
            f"{results['fixes_pending']} pending from {results['similar_found']} similar issues"
        )
        
        return results
    
    def get_issues_by_severity(
        self,
        min_severity: IssueSeverity = IssueSeverity.LOW,
    ) -> Dict[str, List[DetectedIssue]]:
        """Get all issues grouped by severity.
        
        Args:
            min_severity: Minimum severity to include
            
        Returns:
            Dict mapping severity to issues
        """
        severity_order = [
            IssueSeverity.CRITICAL,
            IssueSeverity.HIGH,
            IssueSeverity.MEDIUM,
            IssueSeverity.LOW,
            IssueSeverity.INFO,
        ]
        
        min_idx = severity_order.index(min_severity)
        allowed_severities = severity_order[:min_idx + 1]
        
        result = {sev.value: [] for sev in allowed_severities}
        
        for issue in self.detected_issues.values():
            if issue.severity in allowed_severities:
                result[issue.severity.value].append(issue)
        
        return result
    
    def get_issues_by_file(self) -> Dict[str, List[DetectedIssue]]:
        """Get all issues grouped by file."""
        result = {}
        
        for issue in self.detected_issues.values():
            if issue.filepath not in result:
                result[issue.filepath] = []
            result[issue.filepath].append(issue)
        
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all detected issues.
        
        Returns:
            Report dictionary with statistics and issue details
        """
        by_severity = self.get_issues_by_severity()
        by_file = self.get_issues_by_file()
        
        # Count by type
        by_type = {}
        for issue in self.detected_issues.values():
            type_name = issue.issue_type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
        
        # Count by status
        by_status = {}
        for issue in self.detected_issues.values():
            status = issue.fix_status.value
            if status not in by_status:
                by_status[status] = 0
            by_status[status] += 1
        
        return {
            "summary": {
                "total_issues": len(self.detected_issues),
                "files_with_issues": len(by_file),
                "files_analyzed": self.files_analyzed,
                "scans_performed": self.scan_count,
                "fixes_applied": self.fix_count,
                "patterns_registered": len(self.patterns),
            },
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "by_type": by_type,
            "by_status": by_status,
            "top_files": sorted(
                [(f, len(issues)) for f, issues in by_file.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "timestamp": datetime.now().isoformat(),
        }
    
    def clear_issues(self):
        """Clear all detected issues and reset counters."""
        self.detected_issues.clear()
        self.files_analyzed = 0
        logger.info("Cleared all detected issues")


# =============================================================================
# AGENT INTEGRATION
# =============================================================================

class AgentIssueHandler:
    """
    Integration layer for agents to use the CrossFileAnalyzer.
    
    This class provides a simple interface for agents to:
    1. Report issues they find during operation
    2. Request scans for similar issues
    3. Automatically propagate fixes
    
    Usage in an agent:
        from src.core.cross_file_analyzer import AgentIssueHandler
        
        handler = AgentIssueHandler(agent_name="HOAGS")
        
        # When you find an issue
        issue = handler.report_issue(
            filepath="src/ml/train.py",
            line_number=42,
            issue_type=IssueType.DEPRECATED_USAGE,
            description="Using deprecated function",
            code_snippet="old_function()",
        )
        
        # Find similar issues
        similar = handler.find_similar(issue)
        
        # After fixing original, propagate fix
        handler.propagate_fix(issue, "new_function()")
    """
    
    def __init__(self, agent_name: str, project_root: Optional[str] = None):
        """Initialize the handler for a specific agent.
        
        Args:
            agent_name: Name of the agent using this handler
            project_root: Root directory of the project
        """
        self.agent_name = agent_name
        self.analyzer = get_cross_file_analyzer(project_root)
        
        logger.info(f"AgentIssueHandler initialized for {agent_name}")
    
    def report_issue(
        self,
        filepath: str,
        line_number: int,
        issue_type: IssueType,
        description: str,
        code_snippet: str,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        suggested_fix: Optional[str] = None,
    ) -> DetectedIssue:
        """Report an issue found during agent operation.
        
        Args:
            filepath: Path to the file with the issue
            line_number: Line number of the issue
            issue_type: Type of issue
            description: Description of the issue
            code_snippet: The problematic code
            severity: Issue severity
            suggested_fix: Suggested fix if known
            
        Returns:
            DetectedIssue object
        """
        issue_id = hashlib.sha256(
            f"{self.agent_name}{filepath}{line_number}{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        # Create a pattern for this issue type
        pattern = self.analyzer.register_pattern(
            issue_type=issue_type,
            description=description,
            regex_pattern=re.escape(code_snippet),
            severity=severity,
            fix_template=suggested_fix,
            auto_fixable=suggested_fix is not None,
        )
        
        issue = DetectedIssue(
            issue_id=issue_id,
            pattern_id=pattern.pattern_id,
            issue_type=issue_type,
            severity=severity,
            filepath=filepath,
            line_number=line_number,
            description=description,
            code_snippet=code_snippet,
            suggested_fix=suggested_fix,
            context={"reported_by": self.agent_name},
        )
        
        self.analyzer.detected_issues[issue_id] = issue
        
        logger.info(
            f"[{self.agent_name}] Reported issue: {issue_type.value} in "
            f"{filepath}:{line_number}"
        )
        
        return issue
    
    def find_similar(
        self,
        issue: DetectedIssue,
        similarity_threshold: float = 0.8,
    ) -> List[DetectedIssue]:
        """Find similar issues across the codebase.
        
        Args:
            issue: Reference issue to find similar ones to
            similarity_threshold: How similar issues must be
            
        Returns:
            List of similar issues
        """
        return self.analyzer.find_similar_issues(issue, similarity_threshold)
    
    def propagate_fix(
        self,
        issue: DetectedIssue,
        fix_code: str,
        auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """Propagate a fix to similar issues.
        
        Args:
            issue: The issue that was fixed
            fix_code: The code that fixed it
            auto_apply: Whether to automatically apply to similar issues
            
        Returns:
            Summary of propagation results
        """
        return self.analyzer.propagate_fix(
            issue,
            fix_code,
            auto_apply=auto_apply,
        )
    
    def scan_codebase(
        self,
        directory: Optional[str] = None,
    ) -> List[DetectedIssue]:
        """Scan the codebase for all registered patterns.
        
        Args:
            directory: Directory to scan (defaults to project root)
            
        Returns:
            List of all detected issues
        """
        return self.analyzer.scan_directory(directory)
    
    def get_report(self) -> Dict[str, Any]:
        """Get a report of all detected issues."""
        return self.analyzer.generate_report()


# =============================================================================
# SINGLETON
# =============================================================================

_analyzer_instance: Optional[CrossFileAnalyzer] = None


def get_cross_file_analyzer(project_root: Optional[str] = None) -> CrossFileAnalyzer:
    """Get the singleton CrossFileAnalyzer instance.
    
    Args:
        project_root: Root directory (only used on first call)
        
    Returns:
        CrossFileAnalyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = CrossFileAnalyzer(project_root)
    return _analyzer_instance


def reset_analyzer():
    """Reset the singleton instance (for testing)."""
    global _analyzer_instance
    _analyzer_instance = None

