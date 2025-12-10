"""
Operations Division Agents
==========================
Reports to: Chris Friedman (Principal/COO)

FRIEDS is Chris Friedman's master authority agent, partner to Tom's HOAGS.
SANTAS_HELPER and CPA are senior agents under FRIEDS.

Hierarchy:
  CHRIS FRIEDMAN (Principal)
       |
    FRIEDS (Master Authority - Partner to HOAGS)
       |
    +--+--+
    |     |
SANTAS  CPA
HELPER
"""

# FRIEDS is in this directory
from .frieds_agent import FriedsAgent, get_frieds

# SANTAS_HELPER and CPA are in their own directories - re-export for compatibility
from src.agents.santas_helper_agent.santas_helper_agent import (
    SantasHelperAgent,
    get_santas_helper,
)
from src.agents.cpa_agent.cpa_agent import (
    CPAAgent,
    get_cpa,
)

__all__ = [
    "FriedsAgent",
    "get_frieds",
    "SantasHelperAgent",
    "get_santas_helper",
    "CPAAgent",
    "get_cpa"
]
