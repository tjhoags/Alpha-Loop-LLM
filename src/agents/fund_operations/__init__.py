"""
================================================================================
FUND OPERATIONS AGENTS - Chris Friedman's Accounting & Operations Division
================================================================================
Author: Chris Friedman | Alpha Loop Capital, LLC

This module contains agents responsible for all fund accounting, operations,
taxation, and reporting functions. These agents report directly to Chris Friedman.

Agents:
- SANTAS_HELPER: Head of Fund Operations, leads team of fund accountants
- CPA: Tax, Audit, and Internal Reporting specialist

NOTE: Agents are now in their own directories. This module re-exports for
backwards compatibility.
================================================================================
"""

# Import from authoritative locations
from src.agents.santas_helper_agent.santas_helper_agent import (
    SantasHelperAgent,
    get_santas_helper,
)
from src.agents.cpa_agent.cpa_agent import (
    CPAAgent,
    get_cpa,
)

__all__ = [
    "SantasHelperAgent",
    "get_santas_helper",
    "CPAAgent",
    "get_cpa",
]
