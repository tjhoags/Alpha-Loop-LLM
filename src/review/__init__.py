"""================================================================================
REVIEW MODULE - Multi-Agent Code Review System
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -m src.review.orchestrator

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
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

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Code analysis and bug detection
    - Docstring generation and formatting
    - Cross-platform compatibility checks
    - Security vulnerability scanning
    - Performance optimization suggestions

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

__all__ = [
    "ReviewOrchestrator",
    "ReviewAgent",
    "ConsensusManager",
    "DocstringAnalyzer",
    "BugDetector",
    "CrossPlatformChecker",
    "ReviewReport",
    "ChangeProposal",
]
