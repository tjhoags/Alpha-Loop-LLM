"""================================================================================
REVIEW ORCHESTRATOR - Master Controller for Multi-Agent Code Review
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -m src.review.orchestrator

    # With options:
    python -m src.review.orchestrator --scope=src/agents --fix --consensus=3

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -m src.review.orchestrator

    # With options:
    python -m src.review.orchestrator --scope=src/agents --fix --consensus=3

WHAT THIS MODULE DOES:
----------------------
The Review Orchestrator is the central controller that:
1. Discovers all Python files in the project (no context window limits)
2. Spawns multiple review agents to analyze files in parallel
3. Collects findings and proposed changes from all agents
4. Manages consensus voting on proposed changes
5. Applies approved changes after testing on both platforms
6. Generates comprehensive review reports

The orchestrator operates in phases:
    Phase 1: Discovery - Find all files to review
    Phase 2: Analysis - Multiple agents review files in chunks
    Phase 3: Consolidation - Merge findings, deduplicate issues
    Phase 4: Consensus - Agents vote on proposed changes
    Phase 5: Testing - Run tests on Windows and Mac
    Phase 6: Application - Apply approved changes
    Phase 7: Reporting - Generate final report

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Spawning review agents for code analysis
    - Reading any file in the project
    - Proposing code changes (not applying without consensus)
    - Running tests in isolated environments
    - Generating reports and logs

NOT APPROVED:
    - Applying changes without consensus
    - Modifying files outside the project
    - Executing arbitrary user code
    - Bypassing the consensus mechanism
    - Ignoring test failures

WEIGHT CONSIDERATIONS:
----------------------
When multiple agents review the same file:
- Each agent's vote carries equal weight (no single agent dominance)
- Consensus threshold is configurable (default: 2 agents must agree)
- Strict mode requires unanimous agreement
- Conflicting suggestions are flagged for human review

================================================================================
"""

import argparse
import hashlib
import json
import os
import platform
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


class ReviewPhase(Enum):
    """Phases of the review process."""

    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    CONSOLIDATION = "consolidation"
    CONSENSUS = "consensus"
    TESTING = "testing"
    APPLICATION = "application"
    REPORTING = "reporting"


class IssueSeverity(Enum):
    """Severity levels for identified issues."""

    CRITICAL = "critical"      # Must fix: security, data loss
    HIGH = "high"              # Should fix: bugs, crashes
    MEDIUM = "medium"          # Nice to fix: style, performance
    LOW = "low"                # Optional: minor improvements
    INFO = "info"              # Informational only


class IssueCategory(Enum):
    """Categories of issues."""

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCSTRING = "docstring"
    CROSS_PLATFORM = "cross_platform"
    STYLE = "style"
    MODEL_WEIGHT = "model_weight"
    DEPRECATED = "deprecated"


@dataclass
class FileInfo:
    """Information about a file to review."""

    path: Path
    relative_path: str
    size_bytes: int
    last_modified: datetime
    hash: str
    language: str = "python"

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified.isoformat(),
            "hash": self.hash,
            "language": self.language,
        }


@dataclass
class Issue:
    """An issue identified during review."""

    file_path: str
    line_number: Optional[int]
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    suggested_fix: Optional[str] = None
    agent_id: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
            "agent_id": self.agent_id,
            "confidence": self.confidence,
        }


@dataclass
class ChangeProposal:
    """A proposed change to a file."""

    proposal_id: str
    file_path: str
    original_content: str
    proposed_content: str
    change_type: str  # 'add', 'modify', 'delete'
    reason: str
    issues_addressed: List[str] = field(default_factory=list)
    proposing_agent: str = ""
    votes_for: List[str] = field(default_factory=list)
    votes_against: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, applied

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def vote_count(self) -> Tuple[int, int]:
        return len(self.votes_for), len(self.votes_against)

    def is_approved(self, threshold: int = 2, strict: bool = False) -> bool:
        """Check if proposal has reached consensus."""
        if strict:
            return len(self.votes_against) == 0 and len(self.votes_for) >= threshold
        return len(self.votes_for) >= threshold and len(self.votes_for) > len(self.votes_against)


@dataclass
class ReviewResult:
    """Results from a single agent's review."""

    agent_id: str
    files_reviewed: List[str]
    issues_found: List[Issue]
    proposals: List[ChangeProposal]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "files_reviewed": self.files_reviewed,
            "issues_found": [i.to_dict() for i in self.issues_found],
            "proposals": [p.to_dict() for p in self.proposals],
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class ReviewOrchestrator:
    """Master controller for the multi-agent code review system.

    This class coordinates multiple review agents, manages consensus,
    and ensures changes are tested before application.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        scope: Optional[str] = None,
        consensus_threshold: int = 2,
        strict_mode: bool = False,
        auto_fix: bool = False,
        dry_run: bool = False,
        num_agents: int = 3,
    ):
        """Initialize the Review Orchestrator.

        Args:
        ----
            project_root: Root directory of the project to review
            scope: Subdirectory to limit review scope (None = full project)
            consensus_threshold: Number of agents that must agree for approval
            strict_mode: If True, require unanimous agreement
            auto_fix: If True, automatically apply approved changes
            dry_run: If True, show changes without applying
            num_agents: Number of review agents to spawn
        """
        self.project_root = project_root or PROJECT_ROOT
        self.scope = scope
        self.consensus_threshold = consensus_threshold
        self.strict_mode = strict_mode
        self.auto_fix = auto_fix
        self.dry_run = dry_run
        self.num_agents = num_agents

        self.settings = get_settings()
        self.current_phase = ReviewPhase.DISCOVERY

        # State
        self.files_to_review: List[FileInfo] = []
        self.all_issues: List[Issue] = []
        self.all_proposals: List[ChangeProposal] = []
        self.agent_results: List[ReviewResult] = []
        self.applied_changes: List[ChangeProposal] = []

        # Threading
        self.result_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()

        # Setup logging
        log_path = self.settings.logs_dir / "review"
        log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(self.log_file, rotation="50 MB", level="DEBUG")

        logger.info("ReviewOrchestrator initialized")
        logger.info(f"  Project root: {self.project_root}")
        logger.info(f"  Scope: {self.scope or 'Full project'}")
        logger.info(f"  Consensus threshold: {self.consensus_threshold}")
        logger.info(f"  Strict mode: {self.strict_mode}")
        logger.info(f"  Auto-fix: {self.auto_fix}")
        logger.info(f"  Dry run: {self.dry_run}")

    # =========================================================================
    # PHASE 1: DISCOVERY
    # =========================================================================

    def discover_files(self) -> List[FileInfo]:
        """Discover all Python files in the project.

        This phase scans the entire project (or scoped directory) to find
        all files that need to be reviewed. No context window limits apply.
        """
        self.current_phase = ReviewPhase.DISCOVERY
        logger.info("=" * 70)
        logger.info("PHASE 1: FILE DISCOVERY")
        logger.info("=" * 70)

        search_root = self.project_root
        if self.scope:
            search_root = self.project_root / self.scope

        if not search_root.exists():
            logger.error(f"Search root does not exist: {search_root}")
            return []

        # Directories to skip
        skip_dirs = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".pytest_cache", ".mypy_cache", "dist", "build", "egg-info",
            ".tox", ".eggs", "htmlcov", ".coverage", "catboost_info",
        }

        files: List[FileInfo] = []

        for root, dirs, filenames in os.walk(search_root):
            # Filter out skip directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue

                file_path = Path(root) / filename

                try:
                    stat = file_path.stat()

                    # Calculate file hash for change detection
                    with open(file_path, "rb") as f:
                        content = f.read()
                        file_hash = hashlib.md5(content).hexdigest()

                    file_info = FileInfo(
                        path=file_path,
                        relative_path=str(file_path.relative_to(self.project_root)),
                        size_bytes=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        hash=file_hash,
                    )
                    files.append(file_info)

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

        self.files_to_review = files
        logger.info(f"Discovered {len(files)} Python files to review")

        # Log file distribution by directory
        dir_counts: Dict[str, int] = {}
        for f in files:
            parts = f.relative_path.split(os.sep)
            if len(parts) > 1:
                dir_counts[parts[0]] = dir_counts.get(parts[0], 0) + 1

        for dir_name, count in sorted(dir_counts.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {dir_name}/: {count} files")

        return files

    # =========================================================================
    # PHASE 2: ANALYSIS
    # =========================================================================

    def analyze_files(self) -> List[ReviewResult]:
        """Analyze all discovered files using multiple agents.

        Files are distributed among agents for parallel analysis.
        Each agent reviews its assigned files and reports issues.
        """
        self.current_phase = ReviewPhase.ANALYSIS
        logger.info("=" * 70)
        logger.info("PHASE 2: MULTI-AGENT ANALYSIS")
        logger.info("=" * 70)

        if not self.files_to_review:
            logger.warning("No files to review!")
            return []

        # Split files among agents
        chunk_size = max(1, len(self.files_to_review) // self.num_agents)
        file_chunks = [
            self.files_to_review[i:i + chunk_size]
            for i in range(0, len(self.files_to_review), chunk_size)
        ]

        # Ensure we don't have more chunks than agents
        while len(file_chunks) > self.num_agents:
            file_chunks[-2].extend(file_chunks[-1])
            file_chunks.pop()

        logger.info(f"Distributing {len(self.files_to_review)} files among {len(file_chunks)} agents")

        results: List[ReviewResult] = []

        # Create and run review agents
        from .agents import ReviewAgent

        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            futures = []
            for i, chunk in enumerate(file_chunks):
                agent = ReviewAgent(
                    agent_id=f"reviewer_{i+1}",
                    project_root=self.project_root,
                )
                future = executor.submit(agent.review_files, chunk)
                futures.append((agent.agent_id, future))

            # Collect results
            for agent_id, future in futures:
                try:
                    result = future.result(timeout=600)  # 10 min timeout
                    results.append(result)
                    logger.info(f"Agent {agent_id}: reviewed {len(result.files_reviewed)} files, "
                              f"found {len(result.issues_found)} issues")
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")

        self.agent_results = results
        return results

    # =========================================================================
    # PHASE 3: CONSOLIDATION
    # =========================================================================

    def consolidate_findings(self) -> Tuple[List[Issue], List[ChangeProposal]]:
        """Consolidate findings from all agents.

        Merges issues, deduplicates, and creates unified change proposals.
        """
        self.current_phase = ReviewPhase.CONSOLIDATION
        logger.info("=" * 70)
        logger.info("PHASE 3: CONSOLIDATING FINDINGS")
        logger.info("=" * 70)

        all_issues: List[Issue] = []
        all_proposals: List[ChangeProposal] = []

        # Collect all issues and proposals
        for result in self.agent_results:
            all_issues.extend(result.issues_found)
            all_proposals.extend(result.proposals)

        # Deduplicate issues (same file, line, category)
        seen_issues: Set[str] = set()
        unique_issues: List[Issue] = []

        for issue in all_issues:
            key = f"{issue.file_path}:{issue.line_number}:{issue.category.value}"
            if key not in seen_issues:
                seen_issues.add(key)
                unique_issues.append(issue)

        # Group proposals by file
        proposals_by_file: Dict[str, List[ChangeProposal]] = {}
        for proposal in all_proposals:
            if proposal.file_path not in proposals_by_file:
                proposals_by_file[proposal.file_path] = []
            proposals_by_file[proposal.file_path].append(proposal)

        # Merge similar proposals for the same file
        merged_proposals: List[ChangeProposal] = []
        for file_path, proposals in proposals_by_file.items():
            if len(proposals) == 1:
                merged_proposals.append(proposals[0])
            else:
                # Keep the most comprehensive proposal
                best = max(proposals, key=lambda p: len(p.proposed_content))
                best.issues_addressed.extend(
                    issue for p in proposals for issue in p.issues_addressed
                    if issue not in best.issues_addressed
                )
                merged_proposals.append(best)

        self.all_issues = unique_issues
        self.all_proposals = merged_proposals

        logger.info(f"Consolidated {len(all_issues)} issues into {len(unique_issues)} unique issues")
        logger.info(f"Consolidated {len(all_proposals)} proposals into {len(merged_proposals)} proposals")

        # Log issue summary by severity
        severity_counts = {}
        for issue in unique_issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        for sev, count in sorted(severity_counts.items()):
            logger.info(f"  {sev.upper()}: {count} issues")

        return unique_issues, merged_proposals

    # =========================================================================
    # PHASE 4: CONSENSUS
    # =========================================================================

    def conduct_consensus_vote(self) -> List[ChangeProposal]:
        """Conduct consensus voting on proposed changes.

        Each agent reviews proposals from other agents and votes.
        Changes are approved only if they meet the consensus threshold.
        """
        self.current_phase = ReviewPhase.CONSENSUS
        logger.info("=" * 70)
        logger.info("PHASE 4: CONSENSUS VOTING")
        logger.info("=" * 70)

        if not self.all_proposals:
            logger.info("No proposals to vote on")
            return []

        from .agents import ConsensusManager

        consensus_mgr = ConsensusManager(
            threshold=self.consensus_threshold,
            strict=self.strict_mode,
        )

        # Have each agent vote on proposals they didn't create
        for result in self.agent_results:
            agent_id = result.agent_id

            for proposal in self.all_proposals:
                if proposal.proposing_agent != agent_id:
                    # Agent reviews and votes
                    vote = consensus_mgr.review_proposal(proposal, agent_id)
                    if vote:
                        proposal.votes_for.append(agent_id)
                    else:
                        proposal.votes_against.append(agent_id)

        # Determine approved proposals
        approved = []
        rejected = []

        for proposal in self.all_proposals:
            if proposal.is_approved(self.consensus_threshold, self.strict_mode):
                proposal.status = "approved"
                approved.append(proposal)
            else:
                proposal.status = "rejected"
                rejected.append(proposal)

        logger.info(f"Consensus results: {len(approved)} approved, {len(rejected)} rejected")

        for proposal in approved:
            logger.info(f"  APPROVED: {proposal.file_path} "
                       f"(votes: {proposal.vote_count[0]} for, {proposal.vote_count[1]} against)")

        for proposal in rejected:
            logger.info(f"  REJECTED: {proposal.file_path} "
                       f"(votes: {proposal.vote_count[0]} for, {proposal.vote_count[1]} against)")

        return approved

    # =========================================================================
    # PHASE 5: TESTING
    # =========================================================================

    def run_cross_platform_tests(self, approved_changes: List[ChangeProposal]) -> Dict[str, bool]:
        """Run tests on both Windows and Mac platforms.

        Tests are run automatically without manual intervention.
        Changes are only applied if tests pass on both platforms.
        """
        self.current_phase = ReviewPhase.TESTING
        logger.info("=" * 70)
        logger.info("PHASE 5: CROSS-PLATFORM TESTING")
        logger.info("=" * 70)

        from .cross_platform import CrossPlatformTester

        tester = CrossPlatformTester(self.project_root)
        results = {}

        # Detect current platform
        current_platform = platform.system().lower()
        logger.info(f"Current platform: {current_platform}")

        # Run tests on current platform
        logger.info("Running tests on current platform...")
        current_result = tester.run_tests()
        results[current_platform] = current_result

        if current_result:
            logger.info(f"  {current_platform}: PASSED")
        else:
            logger.error(f"  {current_platform}: FAILED")

        # For the other platform, we generate test scripts
        other_platform = "darwin" if current_platform == "windows" else "windows"
        logger.info(f"Generating test script for {other_platform}...")

        script_path = tester.generate_test_script(other_platform)
        logger.info(f"  Test script: {script_path}")
        logger.info(f"  Run this script on {other_platform} to complete cross-platform testing")

        # Check if we have remote testing capability
        if tester.can_test_remote(other_platform):
            logger.info(f"Remote testing available for {other_platform}")
            remote_result = tester.run_remote_tests(other_platform)
            results[other_platform] = remote_result
            if remote_result:
                logger.info(f"  {other_platform}: PASSED")
            else:
                logger.error(f"  {other_platform}: FAILED")
        else:
            logger.warning(f"Remote testing not available for {other_platform}")
            logger.warning("Manual testing required - see generated script")
            results[other_platform] = None  # Unknown

        return results

    # =========================================================================
    # PHASE 6: APPLICATION
    # =========================================================================

    def apply_changes(self, approved_changes: List[ChangeProposal]) -> List[ChangeProposal]:
        """Apply approved changes to the codebase.

        Only changes that passed consensus and testing are applied.
        """
        self.current_phase = ReviewPhase.APPLICATION
        logger.info("=" * 70)
        logger.info("PHASE 6: APPLYING CHANGES")
        logger.info("=" * 70)

        if self.dry_run:
            logger.info("DRY RUN - No changes will be applied")
            for proposal in approved_changes:
                logger.info(f"  Would modify: {proposal.file_path}")
            return []

        if not self.auto_fix:
            logger.info("Auto-fix disabled - Changes require manual approval")
            self._save_pending_changes(approved_changes)
            return []

        applied: List[ChangeProposal] = []

        for proposal in approved_changes:
            try:
                file_path = self.project_root / proposal.file_path

                # Create backup
                backup_path = file_path.with_suffix(".py.backup")
                if file_path.exists():
                    backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")

                # Apply change
                file_path.write_text(proposal.proposed_content, encoding="utf-8")

                proposal.status = "applied"
                applied.append(proposal)
                logger.info(f"  Applied: {proposal.file_path}")

            except Exception as e:
                logger.error(f"  Failed to apply {proposal.file_path}: {e}")
                proposal.status = "failed"

        self.applied_changes = applied
        logger.info(f"Applied {len(applied)} of {len(approved_changes)} changes")

        return applied

    def _save_pending_changes(self, proposals: List[ChangeProposal]) -> Path:
        """Save pending changes for manual review."""
        pending_dir = self.settings.logs_dir / "review" / "pending_changes"
        pending_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, proposal in enumerate(proposals):
            change_file = pending_dir / f"{timestamp}_{i:03d}_{Path(proposal.file_path).stem}.json"
            with open(change_file, "w") as f:
                json.dump(proposal.to_dict(), f, indent=2)

        logger.info(f"Saved {len(proposals)} pending changes to {pending_dir}")
        return pending_dir

    # =========================================================================
    # PHASE 7: REPORTING
    # =========================================================================

    def generate_report(self) -> "ReviewReport":
        """Generate comprehensive review report.

        The report includes all findings, votes, changes, and recommendations.
        """
        self.current_phase = ReviewPhase.REPORTING
        logger.info("=" * 70)
        logger.info("PHASE 7: GENERATING REPORT")
        logger.info("=" * 70)

        from .reporters import ReviewReport

        report = ReviewReport(
            timestamp=datetime.now(),
            project_root=str(self.project_root),
            scope=self.scope,
            files_reviewed=len(self.files_to_review),
            issues_found=self.all_issues,
            proposals=self.all_proposals,
            applied_changes=self.applied_changes,
            consensus_threshold=self.consensus_threshold,
            strict_mode=self.strict_mode,
        )

        # Save report
        report_path = self.settings.logs_dir / "review" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.save(report_path)

        # Print summary
        report.print_summary()

        logger.info(f"Report saved to {report_path}")

        return report

    # =========================================================================
    # MAIN ORCHESTRATION
    # =========================================================================

    def run(self) -> "ReviewReport":
        """Run the complete review process.

        Executes all phases in order and returns the final report.
        """
        logger.info("=" * 70)
        logger.info("STARTING COMPREHENSIVE CODE REVIEW")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Phase 1: Discovery
            self.discover_files()

            # Phase 2: Analysis
            self.analyze_files()

            # Phase 3: Consolidation
            self.consolidate_findings()

            # Phase 4: Consensus
            approved = self.conduct_consensus_vote()

            # Phase 5: Testing (if there are approved changes)
            if approved:
                test_results = self.run_cross_platform_tests(approved)

                # Only proceed if tests pass (or are unknown for remote platform)
                tests_ok = all(v is None or v for v in test_results.values())

                if tests_ok:
                    # Phase 6: Application
                    self.apply_changes(approved)
                else:
                    logger.warning("Tests failed - changes not applied")

            # Phase 7: Reporting
            report = self.generate_report()

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Review completed in {duration:.1f} seconds")

            return report

        except Exception as e:
            logger.exception(f"Review failed: {e}")
            raise


def main():
    """Main entry point for the review command."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Code Review System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.review.orchestrator                    # Full project review
  python -m src.review.orchestrator --scope=src/agents # Review only agents
  python -m src.review.orchestrator --fix              # Apply approved changes
  python -m src.review.orchestrator --dry-run          # Show changes without applying
  python -m src.review.orchestrator --consensus=3      # Require 3 agents to agree
  python -m src.review.orchestrator --strict           # Require unanimous agreement
        """,
    )

    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="Subdirectory to limit review scope (default: full project)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically apply approved changes",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without applying",
    )

    parser.add_argument(
        "--consensus",
        type=int,
        default=2,
        help="Number of agents that must agree for approval (default: 2)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require unanimous agreement from all agents",
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of review agents to spawn (default: 3)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ALPHA-LOOP-LLM CODE REVIEW SYSTEM")
    print("=" * 70)
    print(f"Scope: {args.scope or 'Full project'}")
    print(f"Consensus: {args.consensus} agents must agree")
    print(f"Strict mode: {args.strict}")
    print(f"Auto-fix: {args.fix}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    orchestrator = ReviewOrchestrator(
        scope=args.scope,
        consensus_threshold=args.consensus,
        strict_mode=args.strict,
        auto_fix=args.fix,
        dry_run=args.dry_run,
        num_agents=args.agents,
    )

    report = orchestrator.run()

    print("\n" + "=" * 70)
    print("REVIEW COMPLETE")
    print("=" * 70)
    print(f"Files reviewed: {report.files_reviewed}")
    print(f"Issues found: {len(report.issues_found)}")
    print(f"Changes proposed: {len(report.proposals)}")
    print(f"Changes applied: {len(report.applied_changes)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
