"""
================================================================================
CO-EXECUTIVE ASSISTANTS - MARGOT_ROBBIE & ANNA_KENDRICK
================================================================================
Author: Alpha Loop Capital, LLC

Co-EAs that report to both KAT (Tom's EA) and SHYLA (Chris's EA).
They handle shared research, drafting, admin, and scheduling tasks.

SECURITY MODEL:
- READ-ONLY access by default
- NO actions without WRITTEN PERMISSION from owner
- Full audit trail on all activities
================================================================================
"""

from src.agents.co_assistants.margot_robbie import MargotRobbieAgent, get_margot
from src.agents.co_assistants.anna_kendrick import AnnaKendrickAgent, get_anna

__all__ = [
    "MargotRobbieAgent",
    "get_margot",
    "AnnaKendrickAgent",
    "get_anna",
]

