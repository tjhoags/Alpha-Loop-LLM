"""================================================================================
REVIEW AGENTS - LLM-Powered Code Review Agents
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -c "from src.review.agents import ReviewAgent; print(ReviewAgent('test').agent_id)"

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -c "from src.review.agents import ReviewAgent; print(ReviewAgent('test').agent_id)"

WHAT THIS MODULE DOES:
----------------------
Provides LLM-powered review agents that:
1. Analyze Python files for bugs, issues, and improvements
2. Check docstring compliance against required format
3. Identify cross-platform compatibility issues
4. Generate change proposals with suggested fixes
5. Vote on proposals from other agents (consensus mechanism)

Each ReviewAgent operates independently and can review files in parallel.
The ConsensusManager coordinates voting between agents.

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Static code analysis (syntax, style, patterns)
    - Docstring format validation
    - Cross-platform path checking
    - Security pattern detection
    - ML weight documentation validation

NOT APPROVED:
    - Executing arbitrary code from reviewed files
    - Modifying files directly (only proposals)
    - Accessing external systems during review
    - Bypassing consensus requirements

WEIGHT CONSIDERATIONS:
----------------------
Agent voting weights are EQUAL by default:
- No single agent can override consensus
- Each agent's vote counts as 1
- Threshold determines minimum agreeing agents
- Strict mode requires ALL agents to agree (no dissent)

If using different LLM models for agents, be aware:
- Different models may have different biases
- Consider using the same model for fairness
- Or weight more conservative models higher for safety

================================================================================
"""

import ast
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings

# Import types from orchestrator
from .orchestrator import (
    ChangeProposal,
    FileInfo,
    Issue,
    IssueCategory,
    IssueSeverity,
    ReviewResult,
)


class ReviewAgentType(Enum):
    """Types of review agents with different specializations."""

    GENERAL = "general"           # General code review
    SECURITY = "security"         # Security-focused review
    PERFORMANCE = "performance"   # Performance-focused review
    DOCSTRING = "docstring"       # Docstring compliance review
    CROSS_PLATFORM = "cross_platform"  # Cross-platform compatibility


# Required docstring sections
REQUIRED_DOCSTRING_SECTIONS = [
    "HOW TO RUN:",
    "Windows (PowerShell):",
    "Mac (Terminal):",
    "WHAT THIS MODULE DOES:",
    "MODEL INTERPRETATION:",
    "APPROVED:",
    "NOT APPROVED:",
    "WEIGHT CONSIDERATIONS:",
]


@dataclass
class AgentVote:
    """A vote cast by an agent on a proposal."""

    agent_id: str
    proposal_id: str
    vote: bool  # True = approve, False = reject
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ReviewAgent:
    """An LLM-powered code review agent.

    Each agent independently reviews files and generates issues/proposals.
    Agents can also vote on proposals from other agents.
    """

    def __init__(
        self,
        agent_id: str,
        project_root: Optional[Path] = None,
        agent_type: ReviewAgentType = ReviewAgentType.GENERAL,
        use_llm: bool = True,
    ):
        """Initialize a review agent.

        Args:
        ----
            agent_id: Unique identifier for this agent
            project_root: Root directory of the project
            agent_type: Specialization type for this agent
            use_llm: Whether to use LLM for analysis (vs rule-based only)
        """
        self.agent_id = agent_id
        self.project_root = project_root or PROJECT_ROOT
        self.agent_type = agent_type
        self.use_llm = use_llm
        self.settings = get_settings()

        # LLM client (lazy initialization)
        self._llm_client = None

        logger.debug(f"ReviewAgent {agent_id} initialized (type: {agent_type.value})")

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None and self.use_llm:
            self._llm_client = self._initialize_llm()
        return self._llm_client

    def _initialize_llm(self):
        """Initialize the LLM client for code analysis."""
        # Try Anthropic first, then OpenAI
        try:
            import anthropic
            if self.settings.anthropic_api_key:
                return anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        except ImportError:
            pass

        try:
            import openai
            if self.settings.openai_api_key:
                return openai.OpenAI(api_key=self.settings.openai_api_key)
        except ImportError:
            pass

        logger.warning(f"Agent {self.agent_id}: No LLM client available, using rule-based analysis only")
        return None

    # =========================================================================
    # FILE REVIEW
    # =========================================================================

    def review_files(self, files: List[FileInfo]) -> ReviewResult:
        """Review a list of files and generate issues/proposals.

        Args:
        ----
            files: List of FileInfo objects to review

        Returns:
        -------
            ReviewResult containing all findings
        """
        start_time = time.time()

        issues: List[Issue] = []
        proposals: List[ChangeProposal] = []
        files_reviewed: List[str] = []

        for file_info in files:
            try:
                file_issues, file_proposals = self.review_single_file(file_info)
                issues.extend(file_issues)
                proposals.extend(file_proposals)
                files_reviewed.append(file_info.relative_path)

            except Exception as e:
                logger.error(f"Agent {self.agent_id}: Error reviewing {file_info.relative_path}: {e}")
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=None,
                    category=IssueCategory.BUG,
                    severity=IssueSeverity.HIGH,
                    title="File review error",
                    description=f"Could not review file: {e!s}",
                    agent_id=self.agent_id,
                ))

        duration = time.time() - start_time

        return ReviewResult(
            agent_id=self.agent_id,
            files_reviewed=files_reviewed,
            issues_found=issues,
            proposals=proposals,
            duration_seconds=duration,
        )

    def review_single_file(self, file_info: FileInfo) -> Tuple[List[Issue], List[ChangeProposal]]:
        """Review a single file.

        Args:
        ----
            file_info: Information about the file to review

        Returns:
        -------
            Tuple of (issues, proposals)
        """
        issues: List[Issue] = []
        proposals: List[ChangeProposal] = []

        try:
            content = file_info.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_info.path.read_text(encoding="latin-1")

        # 1. Check docstring compliance
        docstring_issues, docstring_proposal = self._check_docstring(file_info, content)
        issues.extend(docstring_issues)
        if docstring_proposal:
            proposals.append(docstring_proposal)

        # 2. Check for syntax errors
        syntax_issues = self._check_syntax(file_info, content)
        issues.extend(syntax_issues)

        # 3. Check cross-platform compatibility
        platform_issues = self._check_cross_platform(file_info, content)
        issues.extend(platform_issues)

        # 4. Check for common bugs
        bug_issues = self._check_common_bugs(file_info, content)
        issues.extend(bug_issues)

        # 5. Check security issues
        security_issues = self._check_security(file_info, content)
        issues.extend(security_issues)

        # 6. Check ML model weight documentation
        weight_issues = self._check_weight_documentation(file_info, content)
        issues.extend(weight_issues)

        # 7. LLM-based deep analysis (if enabled)
        if self.use_llm and self.llm_client:
            llm_issues = self._llm_analyze(file_info, content)
            issues.extend(llm_issues)

        return issues, proposals

    # =========================================================================
    # DOCSTRING CHECKING
    # =========================================================================

    def _check_docstring(
        self,
        file_info: FileInfo,
        content: str,
    ) -> Tuple[List[Issue], Optional[ChangeProposal]]:
        """Check if file has compliant docstring with execution instructions.

        Required format includes:
        - HOW TO RUN section with Windows and Mac instructions
        - WHAT THIS MODULE DOES section
        - MODEL INTERPRETATION section with APPROVED/NOT APPROVED
        - WEIGHT CONSIDERATIONS section
        """
        issues: List[Issue] = []

        # Skip __init__.py files with minimal content
        if file_info.path.name == "__init__.py" and len(content.strip()) < 100:
            return issues, None

        # Extract module docstring
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
        except SyntaxError:
            return issues, None

        if not docstring:
            issues.append(Issue(
                file_path=file_info.relative_path,
                line_number=1,
                category=IssueCategory.DOCSTRING,
                severity=IssueSeverity.MEDIUM,
                title="Missing module docstring",
                description="File lacks a module-level docstring with execution instructions",
                agent_id=self.agent_id,
            ))

            # Generate proposal for missing docstring
            proposal = self._generate_docstring_proposal(file_info, content, None)
            return issues, proposal

        # Check for required sections
        missing_sections = []
        for section in REQUIRED_DOCSTRING_SECTIONS:
            if section not in docstring:
                missing_sections.append(section)

        if missing_sections:
            issues.append(Issue(
                file_path=file_info.relative_path,
                line_number=1,
                category=IssueCategory.DOCSTRING,
                severity=IssueSeverity.MEDIUM,
                title="Incomplete docstring",
                description=f"Docstring missing required sections: {', '.join(missing_sections)}",
                agent_id=self.agent_id,
            ))

            proposal = self._generate_docstring_proposal(file_info, content, docstring)
            return issues, proposal

        # Check for cd command in Windows section
        if "cd C:\\" not in docstring and "cd C:/" not in docstring:
            issues.append(Issue(
                file_path=file_info.relative_path,
                line_number=1,
                category=IssueCategory.DOCSTRING,
                severity=IssueSeverity.LOW,
                title="Missing cd command in Windows instructions",
                description="Windows instructions should start with 'cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1'",
                agent_id=self.agent_id,
            ))

        return issues, None

    def _generate_docstring_proposal(
        self,
        file_info: FileInfo,
        content: str,
        existing_docstring: Optional[str],
    ) -> ChangeProposal:
        """Generate a proposal to add/fix the docstring."""
        # Determine module name from path
        module_parts = file_info.relative_path.replace("\\", "/").replace(".py", "").split("/")
        module_name = module_parts[-1].upper().replace("_", " ")

        # Generate the compliant docstring
        new_docstring = f'''"""
================================================================================
{module_name} - [Brief Description]
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\\\Users\\\\tom\\\\Alpha-Loop-LLM\\\\Alpha-Loop-LLM-1
    .\\\\venv\\\\Scripts\\\\activate
    python -m {'.'.join(module_parts[:-1])}.{module_parts[-1]}

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -m {'.'.join(module_parts[:-1])}.{module_parts[-1]}

WHAT THIS MODULE DOES:
----------------------
[Describe the module's functionality in natural language]

MODEL INTERPRETATION:
---------------------
APPROVED:
    - [List operations this module is designed to perform]
    - [List acceptable use cases]

NOT APPROVED:
    - [List operations this module should NOT be used for]
    - [List potential misuse cases to avoid]

WEIGHT CONSIDERATIONS:
----------------------
[Document any ML model weighting considerations]
[Note any biases or edge cases to be aware of]

================================================================================
"""
'''

        # Create the proposed content
        if existing_docstring:
            # Replace existing docstring
            # Find the docstring in the original content
            proposed_content = re.sub(
                r'^([\s\S]*?)("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
                f"\\1{new_docstring}",
                content,
                count=1,
            )
        else:
            # Add new docstring at the top
            proposed_content = new_docstring + "\n" + content

        return ChangeProposal(
            proposal_id=f"{self.agent_id}_{hashlib.md5(file_info.relative_path.encode()).hexdigest()[:8]}",
            file_path=file_info.relative_path,
            original_content=content,
            proposed_content=proposed_content,
            change_type="modify",
            reason="Add compliant docstring with execution instructions and model interpretation",
            issues_addressed=["Missing or incomplete docstring"],
            proposing_agent=self.agent_id,
        )

    # =========================================================================
    # SYNTAX CHECKING
    # =========================================================================

    def _check_syntax(self, file_info: FileInfo, content: str) -> List[Issue]:
        """Check for Python syntax errors."""
        issues: List[Issue] = []

        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(Issue(
                file_path=file_info.relative_path,
                line_number=e.lineno,
                category=IssueCategory.BUG,
                severity=IssueSeverity.CRITICAL,
                title="Syntax error",
                description=f"Python syntax error: {e.msg}",
                suggested_fix=f"Fix syntax at line {e.lineno}: {e.text.strip() if e.text else 'unknown'}",
                agent_id=self.agent_id,
            ))

        return issues

    # =========================================================================
    # CROSS-PLATFORM CHECKING
    # =========================================================================

    def _check_cross_platform(self, file_info: FileInfo, content: str) -> List[Issue]:
        """Check for cross-platform compatibility issues."""
        issues: List[Issue] = []

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for hardcoded Windows paths
            if re.search(r'["\']C:\\\\', line) or re.search(r'["\']C:/', line):
                # Skip if it's in a docstring or comment about instructions
                if "HOW TO RUN" not in content[:content.find(line)] or '"""' not in content[content.find(line):]:
                    if "#" not in line and '"""' not in line and "'''" not in line:
                        issues.append(Issue(
                            file_path=file_info.relative_path,
                            line_number=i,
                            category=IssueCategory.CROSS_PLATFORM,
                            severity=IssueSeverity.MEDIUM,
                            title="Hardcoded Windows path",
                            description="Line contains hardcoded Windows path. Use pathlib.Path or os.path for cross-platform compatibility.",
                            suggested_fix="Use Path.home() or environment variables instead of hardcoded paths",
                            agent_id=self.agent_id,
                        ))

            # Check for os.system with Windows-specific commands
            if "os.system" in line:
                win_commands = ["dir", "cls", "copy", "del", "move", "type", "ren"]
                for cmd in win_commands:
                    if f'"{cmd}' in line.lower() or f"'{cmd}" in line.lower():
                        issues.append(Issue(
                            file_path=file_info.relative_path,
                            line_number=i,
                            category=IssueCategory.CROSS_PLATFORM,
                            severity=IssueSeverity.MEDIUM,
                            title="Windows-specific system command",
                            description=f"os.system uses Windows-specific command '{cmd}'",
                            suggested_fix="Use subprocess with platform checks or Python equivalents (shutil, os)",
                            agent_id=self.agent_id,
                        ))

            # Check for subprocess with shell=True and Windows commands
            if "subprocess" in line and "shell=True" in line:
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=i,
                    category=IssueCategory.CROSS_PLATFORM,
                    severity=IssueSeverity.LOW,
                    title="subprocess with shell=True",
                    description="Using shell=True can cause cross-platform issues",
                    suggested_fix="Use subprocess with a list of arguments instead of shell=True when possible",
                    agent_id=self.agent_id,
                ))

        return issues

    # =========================================================================
    # BUG CHECKING
    # =========================================================================

    def _check_common_bugs(self, file_info: FileInfo, content: str) -> List[Issue]:
        """Check for common bug patterns."""
        issues: List[Issue] = []

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Mutable default arguments
            if re.search(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|set\(\))", line):
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=i,
                    category=IssueCategory.BUG,
                    severity=IssueSeverity.HIGH,
                    title="Mutable default argument",
                    description="Using mutable default argument (list/dict/set). This can cause unexpected behavior.",
                    suggested_fix="Use None as default and initialize inside function: 'def f(x=None): x = x or []'",
                    agent_id=self.agent_id,
                ))

            # Bare except
            if re.search(r"except\s*:", line):
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=i,
                    category=IssueCategory.BUG,
                    severity=IssueSeverity.MEDIUM,
                    title="Bare except clause",
                    description="Bare except catches all exceptions including KeyboardInterrupt and SystemExit",
                    suggested_fix="Use 'except Exception:' or catch specific exceptions",
                    agent_id=self.agent_id,
                ))

            # Division without zero check in obvious cases
            if re.search(r"/\s*\w+\s*[^\s]", line) and "if" not in lines[max(0,i-3):i]:
                # This is a heuristic and may have false positives
                pass

            # Print statements in production code (not in if __name__ blocks)
            if line.strip().startswith("print(") and "__main__" not in content[max(0,content.find(line)-200):content.find(line)]:
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=i,
                    category=IssueCategory.STYLE,
                    severity=IssueSeverity.LOW,
                    title="Print statement in production code",
                    description="Consider using logging instead of print for production code",
                    suggested_fix="Use logger.info() or logger.debug() instead",
                    agent_id=self.agent_id,
                ))

        return issues

    # =========================================================================
    # SECURITY CHECKING
    # =========================================================================

    def _check_security(self, file_info: FileInfo, content: str) -> List[Issue]:
        """Check for security issues."""
        issues: List[Issue] = []

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Hardcoded credentials (basic patterns)
            cred_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
                (r'api_key\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']', "Hardcoded API key"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
                (r'token\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']', "Hardcoded token"),
            ]

            for pattern, desc in cred_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if in docstring or comment
                    if "#" in line.split("=")[0] or '"""' in line or "'''" in line:
                        continue
                    # Skip if it's loading from env
                    if "os.getenv" in line or "os.environ" in line:
                        continue

                    issues.append(Issue(
                        file_path=file_info.relative_path,
                        line_number=i,
                        category=IssueCategory.SECURITY,
                        severity=IssueSeverity.CRITICAL,
                        title=desc,
                        description="Potential hardcoded credential found. Use environment variables instead.",
                        suggested_fix="Use os.getenv('VARIABLE_NAME') or settings configuration",
                        agent_id=self.agent_id,
                    ))

            # SQL injection patterns
            if re.search(r'execute\s*\(\s*["\'].*%s|execute\s*\(\s*f["\']', line):
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=i,
                    category=IssueCategory.SECURITY,
                    severity=IssueSeverity.HIGH,
                    title="Potential SQL injection",
                    description="String formatting in SQL query. Use parameterized queries instead.",
                    suggested_fix="Use cursor.execute('SELECT * FROM t WHERE id = ?', (id,))",
                    agent_id=self.agent_id,
                ))

            # eval/exec usage
            if re.search(r"\beval\s*\(|\bexec\s*\(", line):
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=i,
                    category=IssueCategory.SECURITY,
                    severity=IssueSeverity.HIGH,
                    title="Use of eval/exec",
                    description="eval() and exec() can execute arbitrary code and are security risks",
                    suggested_fix="Use ast.literal_eval() for safe evaluation or refactor to avoid eval/exec",
                    agent_id=self.agent_id,
                ))

        return issues

    # =========================================================================
    # ML WEIGHT DOCUMENTATION CHECKING
    # =========================================================================

    def _check_weight_documentation(self, file_info: FileInfo, content: str) -> List[Issue]:
        """Check for proper ML model weight documentation."""
        issues: List[Issue] = []

        # Check if this is an ML-related file
        ml_indicators = [
            "sklearn", "xgboost", "lightgbm", "catboost", "torch", "tensorflow",
            "keras", "model.fit", "model.train", "predict", "classifier", "regressor",
            "neural", "weights", "bias", "gradient",
        ]

        is_ml_file = any(ind in content.lower() for ind in ml_indicators)

        if not is_ml_file:
            return issues

        # Check for weight documentation in docstring
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree) or ""
        except SyntaxError:
            return issues

        if "WEIGHT CONSIDERATIONS" not in docstring:
            issues.append(Issue(
                file_path=file_info.relative_path,
                line_number=1,
                category=IssueCategory.MODEL_WEIGHT,
                severity=IssueSeverity.MEDIUM,
                title="Missing weight considerations documentation",
                description="ML file lacks WEIGHT CONSIDERATIONS section in docstring",
                suggested_fix="Add WEIGHT CONSIDERATIONS section documenting model biases and edge cases",
                agent_id=self.agent_id,
            ))

        # Check for class weights in classification
        if "classifier" in content.lower() or "classification" in content.lower():
            if "class_weight" not in content and "sample_weight" not in content:
                issues.append(Issue(
                    file_path=file_info.relative_path,
                    line_number=None,
                    category=IssueCategory.MODEL_WEIGHT,
                    severity=IssueSeverity.LOW,
                    title="No class weight handling",
                    description="Classification model may need class_weight parameter for imbalanced data",
                    suggested_fix="Consider adding class_weight='balanced' or custom weights for imbalanced datasets",
                    agent_id=self.agent_id,
                ))

        return issues

    # =========================================================================
    # LLM ANALYSIS
    # =========================================================================

    def _llm_analyze(self, file_info: FileInfo, content: str) -> List[Issue]:
        """Use LLM for deeper code analysis."""
        issues: List[Issue] = []

        if not self.llm_client:
            return issues

        # Truncate very large files
        max_chars = 50000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"

        prompt = f"""Analyze this Python file for bugs, issues, and improvements.
Focus on:
1. Logic errors and bugs
2. Performance issues
3. Missing error handling
4. Code that won't work on both Windows and Mac
5. Security vulnerabilities

File: {file_info.relative_path}

```python
{content}
```

Return a JSON array of issues found. Each issue should have:
- line_number (int or null)
- severity (critical/high/medium/low)
- category (bug/security/performance/style)
- title (short description)
- description (detailed explanation)
- suggested_fix (how to fix it)

Return only the JSON array, no other text.
"""

        try:
            # Try Anthropic
            if hasattr(self.llm_client, "messages"):
                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.content[0].text

            # Try OpenAI
            elif hasattr(self.llm_client, "chat"):
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.choices[0].message.content

            else:
                return issues

            # Parse JSON response
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\[[\s\S]*\]", result_text)
            if json_match:
                llm_issues = json.loads(json_match.group())

                for issue_data in llm_issues:
                    try:
                        issues.append(Issue(
                            file_path=file_info.relative_path,
                            line_number=issue_data.get("line_number"),
                            category=IssueCategory[issue_data.get("category", "bug").upper()],
                            severity=IssueSeverity[issue_data.get("severity", "low").upper()],
                            title=issue_data.get("title", "LLM-detected issue"),
                            description=issue_data.get("description", ""),
                            suggested_fix=issue_data.get("suggested_fix"),
                            agent_id=self.agent_id,
                            confidence=0.8,  # LLM suggestions have lower confidence
                        ))
                    except (KeyError, ValueError) as e:
                        logger.debug(f"Skipping malformed LLM issue: {e}")

        except Exception as e:
            logger.warning(f"LLM analysis failed for {file_info.relative_path}: {e}")

        return issues

    # =========================================================================
    # VOTING
    # =========================================================================

    def vote_on_proposal(self, proposal: ChangeProposal) -> AgentVote:
        """Vote on a proposal from another agent.

        Args:
        ----
            proposal: The change proposal to vote on

        Returns:
        -------
            AgentVote with this agent's decision
        """
        # Don't vote on own proposals
        if proposal.proposing_agent == self.agent_id:
            return AgentVote(
                agent_id=self.agent_id,
                proposal_id=proposal.proposal_id,
                vote=True,  # Auto-approve own proposals
                reason="Self-proposed",
                confidence=1.0,
            )

        # Analyze the proposed change
        vote = True
        reason = "Change appears reasonable"
        confidence = 0.8

        # Check if the change introduces new issues
        try:
            # Try to parse the proposed content
            ast.parse(proposal.proposed_content)
        except SyntaxError as e:
            vote = False
            reason = f"Proposed change has syntax error: {e.msg}"
            confidence = 1.0

        # Check if docstring is properly formatted
        if proposal.change_type == "modify" and "docstring" in proposal.reason.lower():
            required_present = sum(
                1 for section in REQUIRED_DOCSTRING_SECTIONS
                if section in proposal.proposed_content
            )

            if required_present < len(REQUIRED_DOCSTRING_SECTIONS):
                vote = False
                reason = f"Proposed docstring missing {len(REQUIRED_DOCSTRING_SECTIONS) - required_present} required sections"
                confidence = 0.9

        # Size sanity check
        size_change = len(proposal.proposed_content) - len(proposal.original_content)
        if abs(size_change) > len(proposal.original_content):
            # More than doubling or halving the file
            confidence = 0.5
            reason = f"Large size change ({size_change:+d} chars) - review carefully"

        return AgentVote(
            agent_id=self.agent_id,
            proposal_id=proposal.proposal_id,
            vote=vote,
            reason=reason,
            confidence=confidence,
        )


class ConsensusManager:
    """Manages consensus voting between review agents.

    Ensures that changes are only applied when enough agents agree.
    """

    def __init__(self, threshold: int = 2, strict: bool = False):
        """Initialize the consensus manager.

        Args:
        ----
            threshold: Minimum votes needed for approval
            strict: If True, require unanimous agreement
        """
        self.threshold = threshold
        self.strict = strict
        self.votes: Dict[str, List[AgentVote]] = {}  # proposal_id -> votes

    def review_proposal(self, proposal: ChangeProposal, agent_id: str) -> bool:
        """Have an agent review and vote on a proposal.

        This is a simplified version - in production, this would
        delegate to the actual agent for analysis.

        Args:
        ----
            proposal: The proposal to review
            agent_id: The reviewing agent's ID

        Returns:
        -------
            True if agent votes to approve, False otherwise
        """
        # Create a temporary agent for voting
        agent = ReviewAgent(agent_id=agent_id, use_llm=False)
        vote = agent.vote_on_proposal(proposal)

        # Record the vote
        if proposal.proposal_id not in self.votes:
            self.votes[proposal.proposal_id] = []
        self.votes[proposal.proposal_id].append(vote)

        return vote.vote

    def get_consensus(self, proposal: ChangeProposal) -> Tuple[bool, str]:
        """Determine if a proposal has reached consensus.

        Args:
        ----
            proposal: The proposal to check

        Returns:
        -------
            Tuple of (is_approved, reason)
        """
        votes = self.votes.get(proposal.proposal_id, [])

        votes_for = sum(1 for v in votes if v.vote)
        votes_against = sum(1 for v in votes if not v.vote)

        if self.strict:
            if votes_against > 0:
                return False, f"Strict mode: {votes_against} agent(s) voted against"
            if votes_for >= self.threshold:
                return True, f"Unanimous approval ({votes_for} votes)"
            return False, f"Not enough votes ({votes_for}/{self.threshold})"

        if votes_for >= self.threshold and votes_for > votes_against:
            return True, f"Approved ({votes_for} for, {votes_against} against)"

        return False, f"Rejected ({votes_for} for, {votes_against} against, need {self.threshold})"

    def get_vote_summary(self, proposal: ChangeProposal) -> Dict[str, Any]:
        """Get a summary of votes for a proposal."""
        votes = self.votes.get(proposal.proposal_id, [])

        return {
            "proposal_id": proposal.proposal_id,
            "file_path": proposal.file_path,
            "votes_for": [v.agent_id for v in votes if v.vote],
            "votes_against": [v.agent_id for v in votes if not v.vote],
            "reasons": {v.agent_id: v.reason for v in votes},
            "threshold": self.threshold,
            "strict": self.strict,
        }
