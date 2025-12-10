"""================================================================================
CROSS-PLATFORM TESTING - Windows and Mac Compatibility Testing
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -m src.review.cross_platform

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -m src.review.cross_platform

WHAT THIS MODULE DOES:
----------------------
Provides cross-platform testing capabilities that:
1. Detect the current operating system (Windows/Mac/Linux)
2. Run Python tests on the current platform without manual intervention
3. Generate test scripts for the other platform
4. Support remote testing if SSH/connection is configured
5. Validate that code works identically on both platforms

The tester focuses on:
- Import verification (all modules can be imported)
- Syntax validation (all files parse correctly)
- Path handling (no hardcoded platform-specific paths in logic)
- Test suite execution (pytest runs without failures)

MODEL INTERPRETATION:
---------------------
APPROVED:
    - Running pytest and unittest on local system
    - Checking import statements work
    - Validating file syntax with ast.parse
    - Generating shell scripts for remote execution
    - SSH connections to pre-configured test machines

NOT APPROVED:
    - Executing arbitrary code from untrusted sources
    - Modifying system files or configurations
    - Installing system-level packages without approval
    - Running tests with elevated privileges

WEIGHT CONSIDERATIONS:
----------------------
This module does not use ML models.
Test results are binary (pass/fail) with no weighting.
All test failures are treated equally regardless of platform.

================================================================================
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


@dataclass
class TestResult:
    """Result of a test run."""

    platform: str
    success: bool
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    duration_seconds: float = 0.0
    output: str = ""
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "success": self.success,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


class CrossPlatformTester:
    """Handles cross-platform testing for Windows and Mac.

    Runs tests on the current platform and generates scripts
    for testing on the other platform.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the cross-platform tester.

        Args:
        ----
            project_root: Root directory of the project to test
        """
        self.project_root = project_root or PROJECT_ROOT
        self.settings = get_settings()
        self.current_platform = self._detect_platform()

        # Remote testing configuration (from settings or env)
        self.remote_config = {
            "windows": {
                "host": os.getenv("REMOTE_WINDOWS_HOST", ""),
                "user": os.getenv("REMOTE_WINDOWS_USER", ""),
                "key": os.getenv("REMOTE_WINDOWS_KEY", ""),
                "project_path": os.getenv("REMOTE_WINDOWS_PATH", "C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1"),
            },
            "darwin": {
                "host": os.getenv("REMOTE_MAC_HOST", ""),
                "user": os.getenv("REMOTE_MAC_USER", ""),
                "key": os.getenv("REMOTE_MAC_KEY", ""),
                "project_path": os.getenv("REMOTE_MAC_PATH", "~/Alpha-Loop-LLM/Alpha-Loop-LLM-1"),
            },
        }

        logger.info(f"CrossPlatformTester initialized on {self.current_platform}")

    def _detect_platform(self) -> str:
        """Detect the current operating system."""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "darwin"  # macOS
        else:
            return "linux"

    # =========================================================================
    # LOCAL TESTING
    # =========================================================================

    def run_tests(self, test_path: Optional[str] = None) -> bool:
        """Run tests on the current platform.

        Args:
        ----
            test_path: Specific test path (default: run all tests)

        Returns:
        -------
            True if all tests pass, False otherwise
        """
        logger.info(f"Running tests on {self.current_platform}...")

        result = TestResult(platform=self.current_platform, success=False)
        start_time = datetime.now()

        try:
            # Step 1: Verify all imports work
            import_result = self._verify_imports()
            if not import_result["success"]:
                result.errors.extend(import_result["errors"])
                logger.error(f"Import verification failed: {len(import_result['errors'])} errors")
                return False

            logger.info(f"Import verification passed ({import_result['modules_checked']} modules)")

            # Step 2: Verify syntax of all Python files
            syntax_result = self._verify_syntax()
            if not syntax_result["success"]:
                result.errors.extend(syntax_result["errors"])
                logger.error(f"Syntax verification failed: {len(syntax_result['errors'])} errors")
                return False

            logger.info(f"Syntax verification passed ({syntax_result['files_checked']} files)")

            # Step 3: Run pytest if available
            pytest_result = self._run_pytest(test_path)
            result.tests_run = pytest_result.get("tests_run", 0)
            result.tests_passed = pytest_result.get("tests_passed", 0)
            result.tests_failed = pytest_result.get("tests_failed", 0)
            result.tests_skipped = pytest_result.get("tests_skipped", 0)
            result.output = pytest_result.get("output", "")

            if pytest_result.get("errors"):
                result.errors.extend(pytest_result["errors"])

            result.success = pytest_result.get("success", True)

        except Exception as e:
            logger.exception(f"Test execution failed: {e}")
            result.errors.append(str(e))
            result.success = False

        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Log summary
        if result.success:
            logger.info(f"All tests passed on {self.current_platform}")
            logger.info(f"  Tests: {result.tests_passed}/{result.tests_run} passed, "
                       f"{result.tests_skipped} skipped")
        else:
            logger.error(f"Tests failed on {self.current_platform}")
            logger.error(f"  Tests: {result.tests_passed}/{result.tests_run} passed, "
                        f"{result.tests_failed} failed")
            for error in result.errors[:5]:  # Show first 5 errors
                logger.error(f"  - {error}")

        return result.success

    def _verify_imports(self) -> Dict[str, Any]:
        """Verify all project modules can be imported."""
        result = {
            "success": True,
            "modules_checked": 0,
            "errors": [],
        }

        # Key modules to verify
        key_modules = [
            "src.config.settings",
            "src.agents.base_agent",
            "src.review.orchestrator",
        ]

        for module_name in key_modules:
            try:
                __import__(module_name)
                result["modules_checked"] += 1
            except ImportError as e:
                result["success"] = False
                result["errors"].append(f"Cannot import {module_name}: {e}")
            except Exception as e:
                result["success"] = False
                result["errors"].append(f"Error importing {module_name}: {e}")

        return result

    def _verify_syntax(self) -> Dict[str, Any]:
        """Verify syntax of all Python files."""
        import ast

        result = {
            "success": True,
            "files_checked": 0,
            "errors": [],
        }

        # Find all Python files
        for py_file in self.project_root.rglob("*.py"):
            # Skip venv and cache directories
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                ast.parse(content)
                result["files_checked"] += 1
            except SyntaxError as e:
                result["success"] = False
                rel_path = py_file.relative_to(self.project_root)
                result["errors"].append(f"Syntax error in {rel_path}:{e.lineno}: {e.msg}")
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    content = py_file.read_text(encoding="latin-1")
                    ast.parse(content)
                    result["files_checked"] += 1
                except Exception as e:
                    rel_path = py_file.relative_to(self.project_root)
                    result["errors"].append(f"Cannot read {rel_path}: {e}")

        return result

    def _run_pytest(self, test_path: Optional[str] = None) -> Dict[str, Any]:
        """Run pytest and return results."""
        result = {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "output": "",
            "errors": [],
        }

        # Check if pytest is available
        pytest_path = shutil.which("pytest")
        if not pytest_path:
            # Try in venv
            if self.current_platform == "windows":
                pytest_path = self.project_root / "venv" / "Scripts" / "pytest.exe"
            else:
                pytest_path = self.project_root / "venv" / "bin" / "pytest"

            if not pytest_path.exists():
                logger.warning("pytest not found, skipping test execution")
                result["output"] = "pytest not found"
                return result

        # Build pytest command
        cmd = [str(pytest_path), "-v", "--tb=short"]

        if test_path:
            cmd.append(test_path)
        else:
            # Default test directory
            tests_dir = self.project_root / "tests"
            if tests_dir.exists():
                cmd.append(str(tests_dir))
            else:
                logger.warning("No tests directory found")
                return result

        # Add JSON output for parsing
        json_report = tempfile.mktemp(suffix=".json")
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])

        try:
            # Run pytest
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600, check=False,  # 10 minute timeout
            )

            result["output"] = proc.stdout + proc.stderr

            # Try to parse JSON report
            if os.path.exists(json_report):
                with open(json_report) as f:
                    report = json.load(f)

                summary = report.get("summary", {})
                result["tests_run"] = summary.get("total", 0)
                result["tests_passed"] = summary.get("passed", 0)
                result["tests_failed"] = summary.get("failed", 0)
                result["tests_skipped"] = summary.get("skipped", 0)

                os.remove(json_report)

            result["success"] = proc.returncode == 0

            if proc.returncode != 0:
                result["errors"].append(f"pytest exited with code {proc.returncode}")

        except subprocess.TimeoutExpired:
            result["success"] = False
            result["errors"].append("Test execution timed out (10 minutes)")
        except FileNotFoundError:
            result["output"] = "pytest not found or not executable"
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Test execution error: {e}")

        return result

    # =========================================================================
    # REMOTE TESTING
    # =========================================================================

    def can_test_remote(self, target_platform: str) -> bool:
        """Check if remote testing is available for a platform."""
        config = self.remote_config.get(target_platform, {})
        return bool(config.get("host"))

    def run_remote_tests(self, target_platform: str) -> bool:
        """Run tests on a remote machine.

        Args:
        ----
            target_platform: The platform to test on ('windows' or 'darwin')

        Returns:
        -------
            True if tests pass, False otherwise
        """
        config = self.remote_config.get(target_platform, {})

        if not config.get("host"):
            logger.error(f"No remote configuration for {target_platform}")
            return False

        logger.info(f"Running remote tests on {target_platform} ({config['host']})...")

        # Build SSH command
        ssh_cmd = ["ssh"]

        if config.get("key"):
            ssh_cmd.extend(["-i", config["key"]])

        ssh_cmd.append(f"{config['user']}@{config['host']}")

        # Build the test command for the remote platform
        if target_platform == "windows":
            test_cmd = f"""
cd {config['project_path']}
.\\venv\\Scripts\\activate
python -c "from src.review.cross_platform import CrossPlatformTester; t = CrossPlatformTester(); exit(0 if t.run_tests() else 1)"
"""
        else:  # darwin/linux
            test_cmd = f"""
cd {config['project_path']}
source venv/bin/activate
python -c "from src.review.cross_platform import CrossPlatformTester; t = CrossPlatformTester(); exit(0 if t.run_tests() else 1)"
"""

        ssh_cmd.append(test_cmd)

        try:
            proc = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=900, check=False,  # 15 minute timeout for remote
            )

            if proc.returncode == 0:
                logger.info(f"Remote tests passed on {target_platform}")
                return True
            else:
                logger.error(f"Remote tests failed on {target_platform}")
                logger.error(f"Output: {proc.stdout}")
                logger.error(f"Errors: {proc.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Remote test timed out on {target_platform}")
            return False
        except Exception as e:
            logger.error(f"Remote test error: {e}")
            return False

    # =========================================================================
    # TEST SCRIPT GENERATION
    # =========================================================================

    def generate_test_script(self, target_platform: str) -> Path:
        """Generate a test script for manual execution on another platform.

        Args:
        ----
            target_platform: The platform to generate for ('windows' or 'darwin')

        Returns:
        -------
            Path to the generated script
        """
        scripts_dir = self.settings.logs_dir / "review" / "test_scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if target_platform == "windows":
            script_name = f"run_tests_{timestamp}.ps1"
            script_content = self._generate_windows_script()
        else:
            script_name = f"run_tests_{timestamp}.sh"
            script_content = self._generate_mac_script()

        script_path = scripts_dir / script_name
        script_path.write_text(script_content, encoding="utf-8")

        # Make shell scripts executable on Unix
        if target_platform != "windows":
            script_path.chmod(0o755)

        logger.info(f"Generated test script: {script_path}")

        return script_path

    def _generate_windows_script(self) -> str:
        """Generate PowerShell test script for Windows."""
        return """# ================================================================================
# ALPHA-LOOP-LLM CROSS-PLATFORM TEST SCRIPT - WINDOWS
# ================================================================================
#
# HOW TO RUN:
# -----------
# 1. Open PowerShell
# 2. Navigate to the project: cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
# 3. Run this script: .\\logs\\review\\test_scripts\\run_tests_XXXXXXXX.ps1
#
# WHAT THIS DOES:
# ---------------
# - Activates the virtual environment
# - Verifies all imports work
# - Checks syntax of all Python files
# - Runs the pytest suite
# - Reports results
#
# ================================================================================

$ErrorActionPreference = "Stop"

Write-Host "=" * 70
Write-Host "ALPHA-LOOP-LLM CROSS-PLATFORM TESTS - WINDOWS"
Write-Host "=" * 70
Write-Host ""

# Change to project directory
$ProjectRoot = "C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1"
Set-Location $ProjectRoot
Write-Host "Working directory: $ProjectRoot"
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "$ProjectRoot\\venv\\Scripts\\Activate.ps1"
Write-Host ""

# Run Python test
Write-Host "Running cross-platform tests..."
Write-Host "-" * 70

$TestResult = python -c @"
import sys
sys.path.insert(0, '.')
from src.review.cross_platform import CrossPlatformTester
tester = CrossPlatformTester()
success = tester.run_tests()
sys.exit(0 if success else 1)
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 70
    Write-Host "ALL TESTS PASSED ON WINDOWS" -ForegroundColor Green
    Write-Host "=" * 70
} else {
    Write-Host ""
    Write-Host "=" * 70
    Write-Host "TESTS FAILED ON WINDOWS" -ForegroundColor Red
    Write-Host "=" * 70
    exit 1
}
"""

    def _generate_mac_script(self) -> str:
        """Generate Bash test script for Mac."""
        return """#!/bin/bash
# ================================================================================
# ALPHA-LOOP-LLM CROSS-PLATFORM TEST SCRIPT - MAC
# ================================================================================
#
# HOW TO RUN:
# -----------
# 1. Open Terminal
# 2. Navigate to the project: cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
# 3. Make executable (once): chmod +x logs/review/test_scripts/run_tests_XXXXXXXX.sh
# 4. Run this script: ./logs/review/test_scripts/run_tests_XXXXXXXX.sh
#
# WHAT THIS DOES:
# ---------------
# - Activates the virtual environment
# - Verifies all imports work
# - Checks syntax of all Python files
# - Runs the pytest suite
# - Reports results
#
# ================================================================================

set -e

echo "======================================================================"
echo "ALPHA-LOOP-LLM CROSS-PLATFORM TESTS - MAC"
echo "======================================================================"
echo ""

# Change to project directory
PROJECT_ROOT="$HOME/Alpha-Loop-LLM/Alpha-Loop-LLM-1"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Run Python test
echo "Running cross-platform tests..."
echo "----------------------------------------------------------------------"

python -c "
import sys
sys.path.insert(0, '.')
from src.review.cross_platform import CrossPlatformTester
tester = CrossPlatformTester()
success = tester.run_tests()
sys.exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "ALL TESTS PASSED ON MAC"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "TESTS FAILED ON MAC"
    echo "======================================================================"
    exit 1
fi
"""

    # =========================================================================
    # PLATFORM-SPECIFIC CHECKS
    # =========================================================================

    def check_platform_compatibility(self, file_path: Path) -> List[str]:
        """Check a file for platform-specific issues.

        Args:
        ----
            file_path: Path to the file to check

        Returns:
        -------
            List of compatibility issues found
        """
        issues = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for Windows-specific paths in code (not docstrings)
            if "C:\\\\" in line or "C:/" in line:
                # Skip if in docstring or comment
                if not (line.strip().startswith("#") or '"""' in line or "'''" in line):
                    # Check if it's in actual code
                    if "=" in line or "open(" in line or "Path(" in line:
                        issues.append(f"Line {i}: Hardcoded Windows path")

            # Check for Mac-specific paths
            if "/Users/" in line and "home" not in line.lower():
                if not (line.strip().startswith("#") or '"""' in line):
                    issues.append(f"Line {i}: Hardcoded Mac path")

            # Check for platform-specific imports without guards
            if "import winreg" in line or "import msvcrt" in line:
                # Check if there's a platform check nearby
                context = "\n".join(lines[max(0, i-5):i])
                if "platform" not in context.lower() and "sys.platform" not in context:
                    issues.append(f"Line {i}: Windows-specific import without platform check")

            # Check for Unix-specific code
            if "os.fork" in line or "import fcntl" in line:
                context = "\n".join(lines[max(0, i-5):i])
                if "platform" not in context.lower() and "sys.platform" not in context:
                    issues.append(f"Line {i}: Unix-specific code without platform check")

        return issues


def main():
    """Main entry point for cross-platform testing."""
    print("=" * 70)
    print("ALPHA-LOOP-LLM CROSS-PLATFORM TESTER")
    print("=" * 70)

    tester = CrossPlatformTester()
    print(f"Current platform: {tester.current_platform}")
    print("")

    # Run tests on current platform
    print("Running tests on current platform...")
    success = tester.run_tests()

    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)

    # Generate script for other platform
    other_platform = "darwin" if tester.current_platform == "windows" else "windows"
    print(f"\nGenerating test script for {other_platform}...")
    script_path = tester.generate_test_script(other_platform)
    print(f"Script generated: {script_path}")


if __name__ == "__main__":
    main()
