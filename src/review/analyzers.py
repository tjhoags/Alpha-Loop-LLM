"""================================================================================
ANALYZERS - Specialized Code Analysis Tools
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -c "from src.review.analyzers import DocstringAnalyzer; print('OK')"

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -c "from src.review.analyzers import DocstringAnalyzer; print('OK')"

WHAT THIS MODULE DOES:
----------------------
Provides specialized analyzers for different aspects of code review:

1. DocstringAnalyzer: Validates docstring compliance with required format
2. BugDetector: Identifies common bug patterns in Python code
3. CrossPlatformChecker: Finds platform-specific code that may not work on both OS
4. SecurityScanner: Detects potential security vulnerabilities
5. WeightDocumentationChecker: Ensures ML models have proper weight documentation

Each analyzer can be used independently or as part of the orchestrated review.

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Static analysis of Python source code
    - AST parsing and inspection
    - Pattern matching for common issues
    - Generating fix suggestions

NOT APPROVED:
    - Executing analyzed code
    - Modifying files directly
    - Network operations during analysis
    - Accessing external systems

WEIGHT CONSIDERATIONS:
----------------------
This module does not use ML models.
All analysis is rule-based and deterministic.
Issue severity is assigned based on predefined rules.

================================================================================
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .orchestrator import Issue, IssueCategory, IssueSeverity

# Required docstring sections for compliance
REQUIRED_SECTIONS = [
    "HOW TO RUN:",
    "Windows (PowerShell):",
    "Mac (Terminal):",
    "WHAT THIS MODULE DOES:",
    "MODEL INTERPRETATION:",
    "APPROVED:",
    "NOT APPROVED:",
    "WEIGHT CONSIDERATIONS:",
]


class DocstringAnalyzer:
    """Analyzes Python files for docstring compliance.

    Checks that every Python file has the required docstring format
    including execution instructions for both Windows and Mac.
    """

    def __init__(self):
        self.required_sections = REQUIRED_SECTIONS

    def analyze(self, file_path: Path, content: str) -> List[Issue]:
        """Analyze a file's docstring for compliance.

        Args:
        ----
            file_path: Path to the file being analyzed
            content: The file content

        Returns:
        -------
            List of issues found
        """
        issues = []
        relative_path = str(file_path)

        # Skip __init__.py files with minimal content
        if file_path.name == "__init__.py" and len(content.strip()) < 100:
            return issues

        # Try to extract module docstring
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
        except SyntaxError:
            # Syntax errors are handled elsewhere
            return issues

        if not docstring:
            issues.append(Issue(
                file_path=relative_path,
                line_number=1,
                category=IssueCategory.DOCSTRING,
                severity=IssueSeverity.MEDIUM,
                title="Missing module docstring",
                description="File lacks a module-level docstring with execution instructions. "
                           "Every .py file must have a docstring with HOW TO RUN instructions.",
                suggested_fix="Add a compliant module docstring at the top of the file",
            ))
            return issues

        # Check for required sections
        missing_sections = []
        for section in self.required_sections:
            if section not in docstring:
                missing_sections.append(section)

        if missing_sections:
            issues.append(Issue(
                file_path=relative_path,
                line_number=1,
                category=IssueCategory.DOCSTRING,
                severity=IssueSeverity.MEDIUM,
                title="Incomplete module docstring",
                description=f"Docstring missing required sections: {', '.join(missing_sections)}",
                suggested_fix="Add the missing sections to the module docstring",
            ))

        # Check for proper cd command in Windows section
        if "HOW TO RUN:" in docstring:
            if "cd C:\\" not in docstring and "cd C:/" not in docstring:
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=1,
                    category=IssueCategory.DOCSTRING,
                    severity=IssueSeverity.LOW,
                    title="Missing Windows cd command",
                    description="Windows instructions should include 'cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1'",
                    suggested_fix="Add the cd command to the Windows PowerShell section",
                ))

            if "cd ~" not in docstring and "cd $HOME" not in docstring:
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=1,
                    category=IssueCategory.DOCSTRING,
                    severity=IssueSeverity.LOW,
                    title="Missing Mac cd command",
                    description="Mac instructions should include 'cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1'",
                    suggested_fix="Add the cd command to the Mac Terminal section",
                ))

        # Check for venv activation instructions
        if "venv" not in docstring.lower():
            issues.append(Issue(
                file_path=relative_path,
                line_number=1,
                category=IssueCategory.DOCSTRING,
                severity=IssueSeverity.LOW,
                title="Missing venv activation",
                description="Docstring should include virtual environment activation instructions",
                suggested_fix="Add venv activation commands for both platforms",
            ))

        return issues

    def generate_compliant_docstring(self, file_path: Path, existing_docstring: Optional[str] = None) -> str:
        """Generate a compliant docstring for a file.

        Args:
        ----
            file_path: Path to the file
            existing_docstring: Existing docstring content to preserve

        Returns:
        -------
            A compliant docstring string
        """
        # Extract module name
        parts = str(file_path).replace("\\", "/").replace(".py", "").split("/")

        # Find src directory index
        try:
            src_idx = parts.index("src")
            module_parts = parts[src_idx:]
        except ValueError:
            module_parts = parts[-3:] if len(parts) >= 3 else parts

        module_name = parts[-1].upper().replace("_", " ")
        module_path = ".".join(module_parts)

        # Try to extract existing description
        description = "[Description of what this module does]"
        if existing_docstring:
            # Try to find any descriptive text
            lines = existing_docstring.split("\n")
            for line in lines:
                if len(line.strip()) > 20 and not any(s in line for s in ["===", "---", "HOW TO", "WHAT THIS"]):
                    description = line.strip()
                    break

        docstring = f'''"""
================================================================================
{module_name} - Brief Description
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\\\Users\\\\tom\\\\Alpha-Loop-LLM\\\\Alpha-Loop-LLM-1
    .\\\\venv\\\\Scripts\\\\activate
    python -m {module_path}

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -m {module_path}

WHAT THIS MODULE DOES:
----------------------
{description}

MODEL INTERPRETATION:
---------------------
APPROVED:
    - [List operations this module is designed for]

NOT APPROVED:
    - [List operations this module should NOT be used for]

WEIGHT CONSIDERATIONS:
----------------------
[Document any ML model weighting considerations here]

================================================================================
"""'''

        return docstring


class BugDetector:
    """Detects common bug patterns in Python code.
    """

    def analyze(self, file_path: Path, content: str) -> List[Issue]:
        """Analyze code for common bug patterns.

        Args:
        ----
            file_path: Path to the file
            content: File content

        Returns:
        -------
            List of issues found
        """
        issues = []
        relative_path = str(file_path)
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Mutable default arguments
            if re.search(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|set\(\))", line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.BUG,
                    severity=IssueSeverity.HIGH,
                    title="Mutable default argument",
                    description="Using mutable default argument (list/dict/set). "
                               "This can cause unexpected behavior as the default is shared across calls.",
                    suggested_fix="Use None as default and initialize inside function: 'def f(x=None): x = x or []'",
                ))

            # Bare except
            if re.match(r"\s*except\s*:", line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.BUG,
                    severity=IssueSeverity.MEDIUM,
                    title="Bare except clause",
                    description="Bare except catches all exceptions including KeyboardInterrupt and SystemExit",
                    suggested_fix="Use 'except Exception:' or catch specific exceptions",
                ))

            # == None instead of is None
            if re.search(r"==\s*None|None\s*==", line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.STYLE,
                    severity=IssueSeverity.LOW,
                    title="Comparison to None",
                    description="Use 'is None' instead of '== None' for None comparisons",
                    suggested_fix="Replace '== None' with 'is None'",
                ))

            # String formatting in logging
            if re.search(r'logger\.\w+\(f["\']|logging\.\w+\(f["\']', line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.PERFORMANCE,
                    severity=IssueSeverity.LOW,
                    title="f-string in logging",
                    description="Using f-string in logging call. String is formatted even if log level is disabled.",
                    suggested_fix="Use logger.info('message %s', value) instead of logger.info(f'message {value}')",
                ))

            # TODO/FIXME/HACK comments
            if re.search(r"#\s*(TODO|FIXME|HACK|XXX)", line, re.IGNORECASE):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.STYLE,
                    severity=IssueSeverity.INFO,
                    title="TODO/FIXME comment found",
                    description=f"Found comment marker: {line.strip()[:50]}...",
                    suggested_fix="Address the TODO/FIXME or create a tracking issue",
                ))

            # Potential division by zero
            if re.search(r"/\s*[a-zA-Z_]\w*\s*(?![.\[])", line) and "if" not in line:
                # This is a heuristic and may have false positives
                pass

        return issues


class CrossPlatformChecker:
    """Checks for cross-platform compatibility issues.
    """

    WINDOWS_COMMANDS = ["dir", "cls", "copy", "del", "move", "type", "ren", "mkdir", "rmdir"]
    UNIX_COMMANDS = ["ls", "clear", "cp", "rm", "mv", "cat", "chmod", "chown"]

    def analyze(self, file_path: Path, content: str) -> List[Issue]:
        """Analyze code for cross-platform issues.

        Args:
        ----
            file_path: Path to the file
            content: File content

        Returns:
        -------
            List of issues found
        """
        issues = []
        relative_path = str(file_path)
        lines = content.split("\n")

        # Track if we're in a docstring
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines, 1):
            # Track docstrings
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    in_docstring = True
                    docstring_char = '"""' if '"""' in line else "'''"
                    # Check if single-line docstring
                    if line.count(docstring_char) >= 2:
                        in_docstring = False
                else:
                    if docstring_char in line:
                        in_docstring = False
                continue

            if in_docstring:
                continue

            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Hardcoded Windows paths
            if re.search(r'["\']C:\\\\|["\']C:/', line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.CROSS_PLATFORM,
                    severity=IssueSeverity.MEDIUM,
                    title="Hardcoded Windows path",
                    description="Hardcoded Windows path found. Use pathlib.Path or os.path for cross-platform compatibility.",
                    suggested_fix="Use Path.home() / 'subdir' or os.path.expanduser('~')",
                ))

            # Hardcoded Unix paths (outside of docstrings)
            if re.search(r'["\']/Users/|["\']/home/', line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.CROSS_PLATFORM,
                    severity=IssueSeverity.MEDIUM,
                    title="Hardcoded Unix path",
                    description="Hardcoded Unix path found. Use pathlib.Path or os.path for cross-platform compatibility.",
                    suggested_fix="Use Path.home() / 'subdir' or os.path.expanduser('~')",
                ))

            # os.system with platform-specific commands
            if "os.system" in line:
                for cmd in self.WINDOWS_COMMANDS:
                    if f'"{cmd}' in line.lower() or f"'{cmd}" in line.lower():
                        issues.append(Issue(
                            file_path=relative_path,
                            line_number=i,
                            category=IssueCategory.CROSS_PLATFORM,
                            severity=IssueSeverity.MEDIUM,
                            title="Windows-specific command",
                            description=f"os.system uses Windows-specific command '{cmd}'",
                            suggested_fix="Use shutil or os module equivalents, or check platform first",
                        ))

            # subprocess with shell=True
            if "subprocess" in line and "shell=True" in line:
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.CROSS_PLATFORM,
                    severity=IssueSeverity.LOW,
                    title="subprocess with shell=True",
                    description="shell=True can cause cross-platform issues due to different shell environments",
                    suggested_fix="Use a list of arguments instead of shell=True when possible",
                ))

            # Windows-specific imports
            if "import winreg" in line or "import msvcrt" in line or "import winsound" in line:
                # Check if there's a platform guard
                context = "\n".join(lines[max(0, i-5):i])
                if "platform" not in context.lower() and "sys.platform" not in context:
                    issues.append(Issue(
                        file_path=relative_path,
                        line_number=i,
                        category=IssueCategory.CROSS_PLATFORM,
                        severity=IssueSeverity.HIGH,
                        title="Unguarded Windows import",
                        description="Windows-specific import without platform check",
                        suggested_fix="Add 'if sys.platform == \"win32\":' before the import",
                    ))

            # Unix-specific imports
            if "import fcntl" in line or "import pwd" in line or "import grp" in line:
                context = "\n".join(lines[max(0, i-5):i])
                if "platform" not in context.lower() and "sys.platform" not in context:
                    issues.append(Issue(
                        file_path=relative_path,
                        line_number=i,
                        category=IssueCategory.CROSS_PLATFORM,
                        severity=IssueSeverity.HIGH,
                        title="Unguarded Unix import",
                        description="Unix-specific import without platform check",
                        suggested_fix="Add 'if sys.platform != \"win32\":' before the import",
                    ))

        return issues


class SecurityScanner:
    """Scans for potential security vulnerabilities.
    """

    CREDENTIAL_PATTERNS = [
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'api_key\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']', "Hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        (r'token\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']', "Hardcoded token"),
        (r'aws_access_key_id\s*=\s*["\'][A-Z0-9]{20}["\']', "Hardcoded AWS key"),
    ]

    def analyze(self, file_path: Path, content: str) -> List[Issue]:
        """Scan for security vulnerabilities.

        Args:
        ----
            file_path: Path to the file
            content: File content

        Returns:
        -------
            List of issues found
        """
        issues = []
        relative_path = str(file_path)
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Skip if loading from environment
            if "os.getenv" in line or "os.environ" in line or "getenv" in line:
                continue

            # Check credential patterns
            for pattern, description in self.CREDENTIAL_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(Issue(
                        file_path=relative_path,
                        line_number=i,
                        category=IssueCategory.SECURITY,
                        severity=IssueSeverity.CRITICAL,
                        title=description,
                        description="Potential hardcoded credential found. Never commit credentials to source control.",
                        suggested_fix="Use environment variables: os.getenv('VARIABLE_NAME')",
                    ))

            # SQL injection
            if re.search(r'execute\s*\(\s*["\'].*%s|execute\s*\(\s*f["\']|execute\s*\(\s*["\'].*\.format', line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.SECURITY,
                    severity=IssueSeverity.HIGH,
                    title="Potential SQL injection",
                    description="String formatting in SQL query. Use parameterized queries to prevent SQL injection.",
                    suggested_fix="Use cursor.execute('SELECT * FROM t WHERE id = ?', (id,))",
                ))

            # eval/exec
            if re.search(r"\beval\s*\(|\bexec\s*\(", line):
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.SECURITY,
                    severity=IssueSeverity.HIGH,
                    title="Use of eval/exec",
                    description="eval() and exec() can execute arbitrary code and are security risks",
                    suggested_fix="Use ast.literal_eval() for safe evaluation, or refactor to avoid eval/exec",
                ))

            # Pickle loading (deserialization attack)
            if "pickle.load" in line or "pickle.loads" in line:
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=i,
                    category=IssueCategory.SECURITY,
                    severity=IssueSeverity.MEDIUM,
                    title="Pickle deserialization",
                    description="pickle.load can execute arbitrary code. Only load pickles from trusted sources.",
                    suggested_fix="Consider using JSON or other safe serialization formats for untrusted data",
                ))

            # Shell injection via subprocess
            if "subprocess" in line and "shell=True" in line:
                if re.search(r'subprocess\.\w+\([^)]*\+|subprocess\.\w+\(.*f["\']', line):
                    issues.append(Issue(
                        file_path=relative_path,
                        line_number=i,
                        category=IssueCategory.SECURITY,
                        severity=IssueSeverity.HIGH,
                        title="Potential shell injection",
                        description="Concatenating strings in subprocess with shell=True can lead to shell injection",
                        suggested_fix="Use a list of arguments without shell=True",
                    ))

        return issues


class WeightDocumentationChecker:
    """Checks that ML model code has proper weight documentation.
    """

    ML_INDICATORS = [
        "sklearn", "xgboost", "lightgbm", "catboost", "torch", "tensorflow",
        "keras", "model.fit", "classifier", "regressor", "neural",
        "weights", "bias", "gradient", "optimizer", "loss_function",
    ]

    def analyze(self, file_path: Path, content: str) -> List[Issue]:
        """Check ML files for weight documentation.

        Args:
        ----
            file_path: Path to the file
            content: File content

        Returns:
        -------
            List of issues found
        """
        issues = []
        relative_path = str(file_path)

        # Check if this is an ML-related file
        content_lower = content.lower()
        is_ml_file = any(ind in content_lower for ind in self.ML_INDICATORS)

        if not is_ml_file:
            return issues

        # Extract docstring
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree) or ""
        except SyntaxError:
            return issues

        # Check for weight documentation
        if "WEIGHT CONSIDERATIONS" not in docstring:
            issues.append(Issue(
                file_path=relative_path,
                line_number=1,
                category=IssueCategory.MODEL_WEIGHT,
                severity=IssueSeverity.MEDIUM,
                title="Missing weight documentation",
                description="ML-related file lacks WEIGHT CONSIDERATIONS section in docstring",
                suggested_fix="Add WEIGHT CONSIDERATIONS section documenting model biases, weight handling, and edge cases",
            ))

        # Check for class weight handling in classification
        if "classifier" in content_lower or "classification" in content_lower:
            if "class_weight" not in content and "sample_weight" not in content:
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=None,
                    category=IssueCategory.MODEL_WEIGHT,
                    severity=IssueSeverity.LOW,
                    title="No class weight handling",
                    description="Classification model may need class_weight for imbalanced data",
                    suggested_fix="Consider class_weight='balanced' or custom weights",
                ))

        # Check for feature importance documentation
        if "feature_importance" in content or "feature_importances" in content:
            if "interpretation" not in docstring.lower() and "importance" not in docstring.lower():
                issues.append(Issue(
                    file_path=relative_path,
                    line_number=None,
                    category=IssueCategory.MODEL_WEIGHT,
                    severity=IssueSeverity.LOW,
                    title="Missing feature importance documentation",
                    description="Model uses feature importance but lacks interpretation documentation",
                    suggested_fix="Document how feature importances should be interpreted",
                ))

        return issues
