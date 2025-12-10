# /review - Comprehensive Multi-Agent Code Review Command

Execute a full project-wide code review using multiple LLM agents with consensus-based approval.

## What This Command Does

1. **Full Project Scan**: Reviews ALL files in the project (not limited by context window)
2. **Multi-Agent Consensus**: Multiple reviewer agents must agree before changes are applied
3. **Cross-Platform Testing**: Automatically tests on both Windows and Mac without manual intervention
4. **Docstring Enforcement**: Ensures every .py file has proper natural language instructions
5. **Debug & Fix**: Identifies bugs, issues, and automatically proposes fixes

## Usage

```
/review [options]
```

### Options
- `--scope=<path>` - Limit review to specific directory (default: entire project)
- `--fix` - Automatically apply agreed-upon fixes
- `--dry-run` - Show what would be changed without applying
- `--consensus=<n>` - Number of agents that must agree (default: 2)
- `--strict` - Require unanimous agreement from all agents

## Execution Instructions

Run the review system:

```bash
# Windows (PowerShell)
cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
.\venv\Scripts\activate
python -m src.review.orchestrator

# Mac (Terminal)
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
source venv/bin/activate
python -m src.review.orchestrator
```

## Review Categories

1. **Syntax & Bugs**: Python syntax errors, logical bugs, runtime errors
2. **Docstring Compliance**: Every .py must have execution instructions
3. **Cross-Platform**: Windows/Mac compatibility issues
4. **Security**: Credential exposure, injection vulnerabilities
5. **Performance**: Inefficient code patterns, memory leaks
6. **Model Weights**: Ensure no weighting/interpretation issues in ML code

## Required Docstring Format

Every .py file MUST have a header docstring with:

```python
"""
================================================================================
MODULE NAME - Brief Description
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python -m path.to.module

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python -m path.to.module

WHAT THIS MODULE DOES:
----------------------
[Natural language description of functionality]

MODEL INTERPRETATION:
---------------------
APPROVED: [List of operations/interpretations this module approves]
NOT APPROVED: [List of operations this module should NOT be used for]

WEIGHT CONSIDERATIONS:
----------------------
[Any ML model weighting considerations to avoid bias/issues]

================================================================================
"""
```

## Prompt for Review Agents

Review the Alpha-Loop-LLM codebase comprehensively:

1. Scan all Python files in the project
2. For each file, check:
   - Does it have the required docstring header with execution instructions?
   - Are there any bugs or syntax errors?
   - Is it cross-platform compatible (Windows + Mac)?
   - Are there security vulnerabilities?
   - Are ML model interpretations clearly documented?
   - Are weight considerations properly noted?

3. Generate a review report with:
   - Files needing docstring updates
   - Bugs found and proposed fixes
   - Cross-platform issues
   - Security concerns
   - Model interpretation gaps

4. Before applying ANY changes:
   - Present findings to other review agents
   - Require consensus agreement
   - Log all approved/rejected changes

5. After consensus:
   - Apply approved changes
   - Run cross-platform tests (Windows + Mac)
   - Verify tests pass without manual intervention
   - Generate final report

Use the review orchestrator at `src/review/orchestrator.py` to coordinate this process.
