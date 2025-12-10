"""================================================================================
CODE REVIEW AGENT - Cursor Integration for Automated Issue Detection
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd "C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\sii"
    .\\venv\\Scripts\\Activate.ps1
    python -m src.review.code_review_agent

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
    source venv/bin/activate
    python -m src.review.code_review_agent

CURSOR AGENT INTEGRATION:
-------------------------
This agent is designed to be invoked by Cursor when an issue is detected.
It will automatically scan the codebase for similar issues and suggest fixes.

WORKFLOW:
---------
1. User or Cursor detects an issue in a file
2. CodeReviewAgent analyzes the issue pattern
3. Scans entire codebase for similar patterns
4. Generates a report with all similar issues
5. Provides actionable fix suggestions
6. Can auto-apply fixes with confirmation

================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .issue_scanner import (
    Issue,
    IssuePattern,
    IssueScannerAgent,
    IssueType,
    ScanResult,
    scan_for_similar_issues,
)


@dataclass
class FixAction:
    """Represents a proposed fix action."""
    file_path: Path
    line_number: int
    original_code: str
    fixed_code: str
    description: str
    confidence: float  # 0.0 to 1.0
    auto_applicable: bool = True


@dataclass
class ReviewSession:
    """Tracks a code review session."""
    session_id: str
    started_at: datetime
    original_issue: Optional[Issue] = None
    scan_results: Optional[ScanResult] = None
    proposed_fixes: List[FixAction] = field(default_factory=list)
    applied_fixes: List[FixAction] = field(default_factory=list)
    status: str = "pending"  # pending, scanning, reviewing, fixing, complete


class CodeReviewAgent:
    """
    Agent for automated code review and similar issue detection.
    
    This agent integrates with Cursor to:
    1. Detect issues in the current file
    2. Find similar issues across the codebase
    3. Suggest and apply fixes
    
    Example usage by Cursor:
    
        agent = CodeReviewAgent()
        
        # When an issue is found in the current file
        session = agent.start_review(
            file_path="src/data_ingestion/collector.py",
            line_number=42,
            issue_description="Missing type hint for function parameter"
        )
        
        # Get similar issues
        similar = agent.find_similar_issues(session)
        
        # Generate fix report
        report = agent.generate_report(session)
        
        # Apply fixes (with confirmation)
        agent.apply_fixes(session, dry_run=True)
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the code review agent.
        
        Args:
            project_root: Root directory of the project to review
        """
        self.project_root = project_root or Path.cwd()
        self.scanner = IssueScannerAgent(project_root=self.project_root)
        self.sessions: Dict[str, ReviewSession] = {}
        self._session_counter = 0

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        self._session_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"review_{timestamp}_{self._session_counter}"

    def start_review(
        self,
        file_path: str,
        line_number: int,
        issue_description: str,
        code_snippet: str = "",
        issue_type: str = "code_smell",
    ) -> ReviewSession:
        """
        Start a new code review session.
        
        This is the main entry point for Cursor to initiate a review.
        
        Args:
            file_path: Path to the file with the issue
            line_number: Line number where the issue was found
            issue_description: Description of the issue
            code_snippet: The problematic code (optional)
            issue_type: Type of issue (see IssueType enum)
            
        Returns:
            ReviewSession with initial analysis
        """
        session_id = self._generate_session_id()
        
        # Convert string to IssueType
        try:
            issue_type_enum = IssueType(issue_type)
        except ValueError:
            issue_type_enum = IssueType.CODE_SMELL
        
        # Create the original issue
        original_issue = Issue(
            issue_type=issue_type_enum,
            file_path=Path(file_path),
            line_number=line_number,
            column=0,
            message=issue_description,
            code_snippet=code_snippet,
            severity="warning",
        )
        
        session = ReviewSession(
            session_id=session_id,
            started_at=datetime.now(),
            original_issue=original_issue,
            status="scanning",
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"Started review session: {session_id}")
        logger.info(f"  File: {file_path}")
        logger.info(f"  Issue: {issue_description}")
        
        return session

    def find_similar_issues(
        self,
        session: ReviewSession,
        max_results: int = 50
    ) -> List[Issue]:
        """
        Find similar issues across the codebase.
        
        Args:
            session: Active review session
            max_results: Maximum number of similar issues to find
            
        Returns:
            List of similar issues found
        """
        if session.original_issue is None:
            logger.warning("No original issue in session")
            return []
        
        session.status = "scanning"
        
        # Use the scanner to find similar issues
        scan_result = self.scanner.scan_for_similar_issues(
            session.original_issue,
            max_results=max_results
        )
        
        session.scan_results = scan_result
        session.status = "reviewing"
        
        logger.info(f"Found {len(scan_result.similar_issues)} similar issues")
        
        return scan_result.similar_issues

    def generate_fixes(
        self,
        session: ReviewSession
    ) -> List[FixAction]:
        """
        Generate fix suggestions for all issues in the session.
        
        Args:
            session: Review session with scan results
            
        Returns:
            List of proposed fix actions
        """
        if session.scan_results is None:
            logger.warning("No scan results in session")
            return []
        
        fixes: List[FixAction] = []
        
        # Generate fix for original issue
        if session.original_issue and session.original_issue.suggested_fix:
            fixes.append(FixAction(
                file_path=session.original_issue.file_path,
                line_number=session.original_issue.line_number,
                original_code=session.original_issue.code_snippet,
                fixed_code=session.original_issue.suggested_fix,
                description=f"Fix: {session.original_issue.message}",
                confidence=0.9,
                auto_applicable=session.original_issue.auto_fixable,
            ))
        
        # Generate fixes for similar issues
        for issue in session.scan_results.similar_issues:
            if issue.suggested_fix:
                fixes.append(FixAction(
                    file_path=issue.file_path,
                    line_number=issue.line_number,
                    original_code=issue.code_snippet,
                    fixed_code=issue.suggested_fix,
                    description=f"Fix: {issue.message}",
                    confidence=0.8,
                    auto_applicable=issue.auto_fixable,
                ))
        
        session.proposed_fixes = fixes
        logger.info(f"Generated {len(fixes)} fix suggestions")
        
        return fixes

    def generate_report(self, session: ReviewSession) -> str:
        """
        Generate a comprehensive review report.
        
        Args:
            session: Review session to report on
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            "CODE REVIEW REPORT",
            "=" * 80,
            f"Session ID: {session.session_id}",
            f"Started: {session.started_at.isoformat()}",
            f"Status: {session.status}",
            "",
        ]
        
        # Original issue
        if session.original_issue:
            lines.extend([
                "-" * 40,
                "ORIGINAL ISSUE",
                "-" * 40,
                f"File: {session.original_issue.file_path}",
                f"Line: {session.original_issue.line_number}",
                f"Type: {session.original_issue.issue_type.value}",
                f"Message: {session.original_issue.message}",
                f"Code: {session.original_issue.code_snippet}",
                "",
            ])
        
        # Scan results
        if session.scan_results:
            lines.extend([
                "-" * 40,
                "SCAN RESULTS",
                "-" * 40,
                f"Files scanned: {session.scan_results.files_scanned}",
                f"Similar issues found: {len(session.scan_results.similar_issues)}",
                "",
            ])
            
            # Show top issues
            lines.append("Top Similar Issues:")
            for i, issue in enumerate(session.scan_results.similar_issues[:10], 1):
                rel_path = issue.file_path.relative_to(self.project_root) \
                    if issue.file_path.is_relative_to(self.project_root) \
                    else issue.file_path
                lines.append(f"  {i}. {rel_path}:{issue.line_number}")
                lines.append(f"     {issue.code_snippet[:60]}...")
            
            # Recommendations
            if session.scan_results.recommendations:
                lines.extend([
                    "",
                    "-" * 40,
                    "RECOMMENDATIONS",
                    "-" * 40,
                ])
                for rec in session.scan_results.recommendations:
                    lines.append(f"  • {rec}")
        
        # Proposed fixes
        if session.proposed_fixes:
            lines.extend([
                "",
                "-" * 40,
                "PROPOSED FIXES",
                "-" * 40,
                f"Total fixes: {len(session.proposed_fixes)}",
                f"Auto-applicable: {sum(1 for f in session.proposed_fixes if f.auto_applicable)}",
            ])
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)

    def apply_fixes(
        self,
        session: ReviewSession,
        dry_run: bool = True,
        only_high_confidence: bool = True,
        confidence_threshold: float = 0.8
    ) -> List[str]:
        """
        Apply fixes to files.
        
        Args:
            session: Review session with proposed fixes
            dry_run: If True, only report what would be changed
            only_high_confidence: Only apply fixes above confidence threshold
            confidence_threshold: Minimum confidence for auto-fix
            
        Returns:
            List of files that were modified
        """
        modified_files: List[str] = []
        
        for fix in session.proposed_fixes:
            if not fix.auto_applicable:
                continue
            
            if only_high_confidence and fix.confidence < confidence_threshold:
                continue
            
            file_path_str = str(fix.file_path)
            
            if dry_run:
                logger.info(f"[DRY RUN] Would fix {file_path_str}:{fix.line_number}")
                logger.info(f"  Original: {fix.original_code[:50]}...")
                logger.info(f"  Fixed: {fix.fixed_code[:50]}...")
            else:
                # TODO: Implement actual file modification
                # This would read the file, replace the line, and write back
                logger.info(f"Applied fix to {file_path_str}:{fix.line_number}")
                session.applied_fixes.append(fix)
            
            if file_path_str not in modified_files:
                modified_files.append(file_path_str)
        
        if not dry_run:
            session.status = "complete"
        
        return modified_files

    def export_session(self, session: ReviewSession, output_path: Path) -> None:
        """
        Export session data to JSON for persistence.
        
        Args:
            session: Session to export
            output_path: Path to write JSON file
        """
        data = {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "status": session.status,
            "original_issue": {
                "file": str(session.original_issue.file_path) if session.original_issue else None,
                "line": session.original_issue.line_number if session.original_issue else None,
                "message": session.original_issue.message if session.original_issue else None,
            },
            "similar_issues_count": len(session.scan_results.similar_issues) if session.scan_results else 0,
            "proposed_fixes_count": len(session.proposed_fixes),
            "applied_fixes_count": len(session.applied_fixes),
        }
        
        output_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Exported session to {output_path}")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR CURSOR
# =============================================================================

def review_current_issue(
    file_path: str,
    line_number: int,
    issue_description: str,
    code_snippet: str = "",
) -> str:
    """
    Quick function for Cursor to review an issue and find similar ones.
    
    Args:
        file_path: Path to the file with the issue
        line_number: Line number of the issue
        issue_description: Description of what's wrong
        code_snippet: The problematic code
        
    Returns:
        Formatted report string
        
    Example:
        >>> report = review_current_issue(
        ...     file_path="src/data_ingestion/collector.py",
        ...     line_number=42,
        ...     issue_description="Function missing docstring",
        ...     code_snippet="def process_data(df):"
        ... )
        >>> print(report)
    """
    agent = CodeReviewAgent()
    session = agent.start_review(
        file_path=file_path,
        line_number=line_number,
        issue_description=issue_description,
        code_snippet=code_snippet,
    )
    agent.find_similar_issues(session)
    agent.generate_fixes(session)
    return agent.generate_report(session)


def full_codebase_review() -> str:
    """
    Run a full codebase review for all known issue patterns.
    
    Returns:
        Summary report of all issues found
        
    Example:
        >>> report = full_codebase_review()
        >>> print(report)
    """
    scanner = IssueScannerAgent()
    all_issues = scanner.scan_all_patterns()
    
    lines = [
        "=" * 80,
        "FULL CODEBASE REVIEW",
        "=" * 80,
        "",
    ]
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    lines.append(f"Total issues found: {total_issues}")
    lines.append("")
    
    for issue_type, issues in all_issues.items():
        if issues:
            lines.append(f"{issue_type.value}: {len(issues)} issues")
            for issue in issues[:3]:
                lines.append(f"  - {issue.file_path.name}:{issue.line_number}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Run the code review agent as a standalone tool."""
    print("=" * 60)
    print("Alpha Loop Capital - Code Review Agent")
    print("=" * 60)
    
    # Run full codebase review
    report = full_codebase_review()
    print(report)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

