# FIX - Alpha Loop Debugging Protocol

## TRIGGER
When user says "fix", "debug", "broken", "error", or "not working"

## EXECUTION PROTOCOL

### STEP 1: MODEL AGENTS DIAGNOSE
- GHOST Agent: Analyze error patterns and hidden issues
- NOBUS Agent: Run fault injection to identify weak points
- KILLJOY Agent: Check if risk limits are being violated
- ORCHESTRATOR Agent: Coordinate multi-agent diagnosis

### STEP 2: FIX AND TEST CODE
- Fix the identified issue
- Write automated test to verify fix
- Run test on BOTH Windows AND Mac paths
- Ensure code is operational WITHOUT any user intervention
- Verify no input prompts, no manual steps required
- Test:
  - Windows: `python -c "from module import X; X.test()"`
  - Mac: `python3 -c "from module import X; X.test()"`

### STEP 3: FIX SOURCE ISSUE
- Identify ROOT CAUSE, not just symptoms
- Check if issue is:
  - Wrong file path (Windows vs Mac)
  - Missing import
  - API key not loaded
  - Database connection issue
  - Virtual environment not activated
- Fix the source, not a workaround

### STEP 4: FIND SIMILAR ISSUES
- Search for same pattern across ALL code files:
  ```bash
  grep -r "PATTERN" src/ scripts/
  ```
- Check for:
  - Same hardcoded paths
  - Same import errors
  - Same logic bugs
  - Same missing error handling
- Fix ALL occurrences, not just the reported one

### STEP 5: SCAN ALL FILES
- Systematically scan ALL files implemented by other agents
- Check categories:
  - `src/agents/**/*.py` - All agent files
  - `src/core/**/*.py` - Core infrastructure
  - `src/data_ingestion/**/*.py` - Data sources
  - `src/ml/**/*.py` - ML training
  - `src/trading/**/*.py` - Trading execution
  - `scripts/**/*.py` - All scripts
  - `scripts/**/*.bat` - Windows scripts
  - `scripts/**/*.sh` - Mac scripts
  - `scripts/**/*.ps1` - PowerShell scripts
- Run linter: `read_lints` on each directory
- Run import check: `python -c "import src.module"`
- Document findings and fix proactively

## CROSS-PLATFORM REQUIREMENTS

### Path Handling
```python
# WRONG - Hardcoded Windows path
path = "C:\\Users\\tom\\Alpha-Loop-LLM"

# CORRECT - Cross-platform
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(PROJECT_ROOT, "subfolder", "file.py")
```

### Script Instructions Must Include:
1. `cd` to project directory (with correct path for platform)
2. Virtual environment activation
3. Environment variable loading
4. Actual command to run
5. Expected output / next steps

### Windows (.bat / .ps1)
```batch
REM Step 1: Navigate to project
cd /d "%~dp0.."

REM Step 2: Activate venv
call venv\Scripts\activate.bat

REM Step 3: Run script
python scripts\my_script.py
```

### Mac (.sh)
```bash
# Step 1: Navigate to project
cd "$(dirname "$0")/.."

# Step 2: Activate venv
source venv/bin/activate

# Step 3: Run script
python scripts/my_script.py
```

## VALIDATION CHECKLIST

Before marking as fixed:
- [ ] Code runs without errors
- [ ] No user intervention required
- [ ] Works on Windows
- [ ] Works on Mac
- [ ] Similar issues fixed across codebase
- [ ] All imports verified
- [ ] Linter passes
- [ ] Tests pass (if applicable)

## REPORTING

After fix, report:
1. **Root Cause**: What was the actual issue
2. **Files Changed**: List all modified files
3. **Similar Issues Found**: Count and location
4. **Tests Run**: What was validated
5. **Next Steps**: Any remaining work

