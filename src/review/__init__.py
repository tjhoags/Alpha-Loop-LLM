"""================================================================================
REVIEW MODULE - Multi-Agent Code Review System with Issue Detection
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd "C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\sii"
    .\\venv\\Scripts\\Activate.ps1
    python -m src.review.orchestrator

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
    source venv/bin/activate
    python -m src.review.orchestrator

WHAT THIS MODULE DOES:
----------------------
Provides a comprehensive multi-agent code review system that:
1. Scans the entire project for issues regardless of context window limits
2. Uses multiple LLM agents to review and reach consensus on changes
3. Tests changes on both Windows and Mac platforms
4. Enforces docstring standards with execution instructions
5. Ensures ML model weight interpretations are clearly documented
6. **NEW**: Finds and fixes similar issues across the entire codebase

ISSUE SCANNER:
--------------
When an issue is found in one file, use the issue scanner to find similar
issues across the codebase:

    from src.review.issue_scanner import scan_for_similar_issues
    
    result = scan_for_similar_issues(
        issue_type="missing_import",
        message="pandas is used but not imported",
        file_path="src/analysis.py",
        line_number=42,
        code_snippet="df = pd.DataFrame()"
    )
    print(f"Found {len(result.similar_issues)} similar issues")

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Code analysis and bug detection
    - Docstring generation and formatting
    - Cross-platform compatibility checks
    - Security vulnerability scanning
    - Performance optimization suggestions
    - Similar issue detection and batch fixes

NOT APPROVED:
    - Modifying trading logic without explicit approval
    - Changing risk parameters or position limits
    - Altering API keys or credentials
    - Removing safety checks or validation

WEIGHT CONSIDERATIONS:
----------------------
This module does not contain ML models. It uses LLM APIs for code analysis.
Ensure API rate limits are respected when processing large codebases.

================================================================================
"""

from .agents import ConsensusManager, ReviewAgent
from .analyzers import BugDetector, CrossPlatformChecker, DocstringAnalyzer
from .orchestrator import ReviewOrchestrator
from .reporters import ChangeProposal, ReviewReport
from .issue_scanner import (
    IssueScannerAgent,
    Issue,
    IssueType,
    IssuePattern,
    ScanResult,
    scan_for_similar_issues,
    fix_similar_issues,
)

__all__ = [
    # Core Review
    "ReviewOrchestrator",
    "ReviewAgent",
    "ConsensusManager",
    "DocstringAnalyzer",
    "BugDetector",
    "CrossPlatformChecker",
    "ReviewReport",
    "ChangeProposal",
    # Issue Scanner
    "IssueScannerAgent",
    "Issue",
    "IssueType",
    "IssuePattern",
    "ScanResult",
    "scan_for_similar_issues",
    "fix_similar_issues",
]
