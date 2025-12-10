"""================================================================================
REVIEW REPORTERS - Report Generation for Code Reviews
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -c "from src.review.reporters import ReviewReport; print(ReviewReport.__doc__)"

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -c "from src.review.reporters import ReviewReport; print(ReviewReport.__doc__)"

WHAT THIS MODULE DOES:
----------------------
Generates comprehensive review reports including:
1. Summary statistics (files reviewed, issues found, changes applied)
2. Issue breakdown by category and severity
3. Change proposals with vote counts and consensus status
4. Recommendations for manual review
5. Cross-platform test results
6. Export to JSON, Markdown, and HTML formats

Reports serve as documentation of the review process and
provide audit trails for all changes made.

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Generating reports from review data
    - Exporting to various formats
    - Creating visualizations of issues
    - Summarizing findings

NOT APPROVED:
    - Modifying source code
    - Executing changes
    - Accessing external systems
    - Sending reports externally without approval

WEIGHT CONSIDERATIONS:
----------------------
This module does not use ML models.
Report data is presented factually without weighting.
Issue severity is as determined by the review agents.

================================================================================
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings

# Import types from orchestrator
from .orchestrator import ChangeProposal, Issue, IssueCategory, IssueSeverity


@dataclass
class ReviewReport:
    """Comprehensive report of a code review session.

    Contains all findings, proposals, votes, and applied changes
    from a review orchestration run.
    """

    # Metadata
    timestamp: datetime
    project_root: str
    scope: Optional[str]

    # Statistics
    files_reviewed: int

    # Findings
    issues_found: List[Issue]
    proposals: List[ChangeProposal]
    applied_changes: List[ChangeProposal]

    # Configuration
    consensus_threshold: int
    strict_mode: bool

    # Optional additional data
    test_results: Dict[str, bool] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "project_root": self.project_root,
            "scope": self.scope,
            "files_reviewed": self.files_reviewed,
            "issues_found": [i.to_dict() for i in self.issues_found],
            "proposals": [p.to_dict() for p in self.proposals],
            "applied_changes": [c.to_dict() for c in self.applied_changes],
            "consensus_threshold": self.consensus_threshold,
            "strict_mode": self.strict_mode,
            "test_results": self.test_results,
            "duration_seconds": self.duration_seconds,
            "summary": self.get_summary(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the review."""
        # Count issues by severity
        severity_counts = Counter(i.severity.value for i in self.issues_found)

        # Count issues by category
        category_counts = Counter(i.category.value for i in self.issues_found)

        # Count proposal statuses
        proposal_statuses = Counter(p.status for p in self.proposals)

        return {
            "files_reviewed": self.files_reviewed,
            "total_issues": len(self.issues_found),
            "issues_by_severity": dict(severity_counts),
            "issues_by_category": dict(category_counts),
            "total_proposals": len(self.proposals),
            "proposals_by_status": dict(proposal_statuses),
            "changes_applied": len(self.applied_changes),
            "tests_passed": all(v for v in self.test_results.values() if v is not None),
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ReviewReport":
        """Load report from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct issues
        issues = []
        for i_data in data.get("issues_found", []):
            issues.append(Issue(
                file_path=i_data["file_path"],
                line_number=i_data.get("line_number"),
                category=IssueCategory[i_data["category"].upper()],
                severity=IssueSeverity[i_data["severity"].upper()],
                title=i_data["title"],
                description=i_data["description"],
                suggested_fix=i_data.get("suggested_fix"),
                agent_id=i_data.get("agent_id", ""),
                confidence=i_data.get("confidence", 1.0),
            ))

        # Reconstruct proposals
        proposals = []
        for p_data in data.get("proposals", []):
            proposals.append(ChangeProposal(**p_data))

        applied = []
        for c_data in data.get("applied_changes", []):
            applied.append(ChangeProposal(**c_data))

        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            project_root=data["project_root"],
            scope=data.get("scope"),
            files_reviewed=data["files_reviewed"],
            issues_found=issues,
            proposals=proposals,
            applied_changes=applied,
            consensus_threshold=data.get("consensus_threshold", 2),
            strict_mode=data.get("strict_mode", False),
            test_results=data.get("test_results", {}),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    def print_summary(self) -> None:
        """Print a summary of the review to console."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("REVIEW SUMMARY")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.duration_seconds:.1f} seconds")
        print(f"Scope: {self.scope or 'Full project'}")
        print("")

        print("-" * 70)
        print("FILES")
        print("-" * 70)
        print(f"Files reviewed: {summary['files_reviewed']}")
        print("")

        print("-" * 70)
        print("ISSUES FOUND")
        print("-" * 70)
        print(f"Total issues: {summary['total_issues']}")
        print("")
        print("By Severity:")
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = summary["issues_by_severity"].get(sev, 0)
            if count > 0:
                print(f"  {sev.upper():12s}: {count}")

        print("")
        print("By Category:")
        for cat, count in sorted(summary["issues_by_category"].items(), key=lambda x: -x[1]):
            print(f"  {cat:15s}: {count}")

        print("")
        print("-" * 70)
        print("CHANGE PROPOSALS")
        print("-" * 70)
        print(f"Total proposals: {summary['total_proposals']}")
        for status, count in summary["proposals_by_status"].items():
            print(f"  {status:12s}: {count}")

        print("")
        print(f"Changes applied: {summary['changes_applied']}")

        print("")
        print("-" * 70)
        print("CROSS-PLATFORM TESTS")
        print("-" * 70)
        if self.test_results:
            for platform, result in self.test_results.items():
                status = "PASSED" if result else ("FAILED" if result is False else "NOT RUN")
                print(f"  {platform:12s}: {status}")
        else:
            print("  No test results available")

        print("")
        print("=" * 70)

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        summary = self.get_summary()
        lines = []

        lines.append("# Code Review Report")
        lines.append("")
        lines.append(f"**Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Duration:** {self.duration_seconds:.1f} seconds")
        lines.append(f"**Scope:** {self.scope or 'Full project'}")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Files reviewed: {summary['files_reviewed']}")
        lines.append(f"- Total issues: {summary['total_issues']}")
        lines.append(f"- Proposals: {summary['total_proposals']}")
        lines.append(f"- Changes applied: {summary['changes_applied']}")
        lines.append("")

        # Issues by severity
        lines.append("## Issues by Severity")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = summary["issues_by_severity"].get(sev, 0)
            lines.append(f"| {sev.upper()} | {count} |")
        lines.append("")

        # Critical and High issues detail
        critical_high = [i for i in self.issues_found
                        if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]]

        if critical_high:
            lines.append("## Critical and High Severity Issues")
            lines.append("")
            for issue in critical_high:
                lines.append(f"### {issue.title}")
                lines.append("")
                lines.append(f"- **File:** `{issue.file_path}`")
                if issue.line_number:
                    lines.append(f"- **Line:** {issue.line_number}")
                lines.append(f"- **Severity:** {issue.severity.value.upper()}")
                lines.append(f"- **Category:** {issue.category.value}")
                lines.append("")
                lines.append(issue.description)
                if issue.suggested_fix:
                    lines.append("")
                    lines.append(f"**Suggested fix:** {issue.suggested_fix}")
                lines.append("")

        # Applied changes
        if self.applied_changes:
            lines.append("## Applied Changes")
            lines.append("")
            for change in self.applied_changes:
                lines.append(f"- `{change.file_path}`: {change.reason}")
            lines.append("")

        # Test results
        lines.append("## Cross-Platform Test Results")
        lines.append("")
        if self.test_results:
            lines.append("| Platform | Status |")
            lines.append("|----------|--------|")
            for platform, result in self.test_results.items():
                status = "PASSED" if result else ("FAILED" if result is False else "NOT RUN")
                lines.append(f"| {platform} | {status} |")
        else:
            lines.append("No test results available.")
        lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML report."""
        summary = self.get_summary()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Review Report - {self.timestamp.strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.critical {{ background: linear-gradient(135deg, #f44336 0%, #e91e63 100%); }}
        .stat-card.success {{ background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%); }}
        .stat-number {{ font-size: 36px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .severity-critical {{ color: #f44336; font-weight: bold; }}
        .severity-high {{ color: #ff9800; font-weight: bold; }}
        .severity-medium {{ color: #2196F3; }}
        .severity-low {{ color: #9e9e9e; }}
        .issue-card {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        .issue-card.critical {{ border-left: 4px solid #f44336; }}
        .issue-card.high {{ border-left: 4px solid #ff9800; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="report">
        <h1>Code Review Report</h1>
        <p>
            <strong>Date:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Duration:</strong> {self.duration_seconds:.1f} seconds<br>
            <strong>Scope:</strong> {self.scope or 'Full project'}
        </p>

        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-number">{summary['files_reviewed']}</div>
                <div class="stat-label">Files Reviewed</div>
            </div>
            <div class="stat-card {'critical' if summary['issues_by_severity'].get('critical', 0) > 0 else ''}">
                <div class="stat-number">{summary['total_issues']}</div>
                <div class="stat-label">Issues Found</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{summary['total_proposals']}</div>
                <div class="stat-label">Proposals</div>
            </div>
            <div class="stat-card success">
                <div class="stat-number">{summary['changes_applied']}</div>
                <div class="stat-label">Changes Applied</div>
            </div>
        </div>

        <h2>Issues by Severity</h2>
        <table>
            <tr>
                <th>Severity</th>
                <th>Count</th>
            </tr>
"""
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = summary["issues_by_severity"].get(sev, 0)
            html += f"""            <tr>
                <td class="severity-{sev}">{sev.upper()}</td>
                <td>{count}</td>
            </tr>
"""

        html += """        </table>

        <h2>Critical and High Severity Issues</h2>
"""

        critical_high = [i for i in self.issues_found
                        if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]]

        if critical_high:
            for issue in critical_high[:20]:  # Limit to first 20
                html += f"""        <div class="issue-card {issue.severity.value}">
            <h3>{issue.title}</h3>
            <p><strong>File:</strong> <code>{issue.file_path}</code>
            {f'<strong>Line:</strong> {issue.line_number}' if issue.line_number else ''}</p>
            <p>{issue.description}</p>
            {f'<p><strong>Suggested fix:</strong> {issue.suggested_fix}</p>' if issue.suggested_fix else ''}
        </div>
"""
        else:
            html += "        <p>No critical or high severity issues found.</p>\n"

        html += """
        <h2>Cross-Platform Test Results</h2>
        <table>
            <tr>
                <th>Platform</th>
                <th>Status</th>
            </tr>
"""
        for platform, result in self.test_results.items():
            status = "PASSED" if result else ("FAILED" if result is False else "NOT RUN")
            color = "#4CAF50" if result else ("#f44336" if result is False else "#9e9e9e")
            html += f"""            <tr>
                <td>{platform}</td>
                <td style="color: {color}; font-weight: bold;">{status}</td>
            </tr>
"""

        html += """        </table>
    </div>
</body>
</html>
"""
        return html

    def export_all_formats(self, output_dir: Path) -> Dict[str, Path]:
        """Export report to all formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")

        paths = {}

        # JSON
        json_path = output_dir / f"report_{timestamp}.json"
        self.save(json_path)
        paths["json"] = json_path

        # Markdown
        md_path = output_dir / f"report_{timestamp}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        paths["markdown"] = md_path
        logger.info(f"Markdown report saved to {md_path}")

        # HTML
        html_path = output_dir / f"report_{timestamp}.html"
        html_path.write_text(self.to_html(), encoding="utf-8")
        paths["html"] = html_path
        logger.info(f"HTML report saved to {html_path}")

        return paths


class DocstringReporter:
    """Generate reports specifically about docstring compliance.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self.settings = get_settings()

    def generate_compliance_report(self, issues: List[Issue]) -> str:
        """Generate a docstring compliance report."""
        docstring_issues = [i for i in issues if i.category == IssueCategory.DOCSTRING]

        lines = []
        lines.append("# Docstring Compliance Report")
        lines.append("")
        lines.append(f"**Total files with docstring issues:** {len(set(i.file_path for i in docstring_issues))}")
        lines.append(f"**Total docstring issues:** {len(docstring_issues)}")
        lines.append("")

        # Group by issue type
        missing_docstrings = [i for i in docstring_issues if "missing" in i.title.lower()]
        incomplete_docstrings = [i for i in docstring_issues if "incomplete" in i.title.lower() or "missing section" in i.title.lower()]
        other_issues = [i for i in docstring_issues if i not in missing_docstrings and i not in incomplete_docstrings]

        if missing_docstrings:
            lines.append("## Files Missing Docstrings")
            lines.append("")
            for issue in missing_docstrings:
                lines.append(f"- `{issue.file_path}`")
            lines.append("")

        if incomplete_docstrings:
            lines.append("## Files with Incomplete Docstrings")
            lines.append("")
            for issue in incomplete_docstrings:
                lines.append(f"- `{issue.file_path}`: {issue.description}")
            lines.append("")

        if other_issues:
            lines.append("## Other Docstring Issues")
            lines.append("")
            for issue in other_issues:
                lines.append(f"- `{issue.file_path}`: {issue.title}")
            lines.append("")

        lines.append("## Required Docstring Format")
        lines.append("")
        lines.append("Every Python file should have a module docstring with these sections:")
        lines.append("")
        lines.append("```python")
        lines.append('"""')
        lines.append("================================================================================")
        lines.append("MODULE NAME - Brief Description")
        lines.append("================================================================================")
        lines.append("")
        lines.append("HOW TO RUN:")
        lines.append("-----------")
        lines.append("Windows (PowerShell):")
        lines.append("    cd C:\\\\Users\\\\tom\\\\Alpha-Loop-LLM\\\\Alpha-Loop-LLM-1")
        lines.append("    .\\\\venv\\\\Scripts\\\\activate")
        lines.append("    python -m module.path")
        lines.append("")
        lines.append("Mac (Terminal):")
        lines.append("    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1")
        lines.append("    source venv/bin/activate")
        lines.append("    python -m module.path")
        lines.append("")
        lines.append("WHAT THIS MODULE DOES:")
        lines.append("----------------------")
        lines.append("[Description]")
        lines.append("")
        lines.append("MODEL INTERPRETATION:")
        lines.append("---------------------")
        lines.append("APPROVED: [List of approved operations]")
        lines.append("NOT APPROVED: [List of not approved operations]")
        lines.append("")
        lines.append("WEIGHT CONSIDERATIONS:")
        lines.append("----------------------")
        lines.append("[ML weight documentation]")
        lines.append("")
        lines.append("================================================================================")
        lines.append('"""')
        lines.append("```")

        return "\n".join(lines)
