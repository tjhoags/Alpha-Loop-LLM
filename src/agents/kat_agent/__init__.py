"""
================================================================================
KAT AGENT - Tom Hogan's Executive Assistant
================================================================================
Author: Tom Hogan (Founder & CIO) | Alpha Loop Capital, LLC

KAT handles everything Tom asks for - professional and personal.
Personal data training is GATED and requires explicit permission from Tom.

SECURITY MODEL:
- READ-ONLY access by default
- NO actions without WRITTEN PERMISSION from Tom
- Full audit trail on all activities
================================================================================
"""

from src.agents.kat_agent.kat_agent import KatAgent, get_kat

__all__ = [
    "KatAgent",
    "get_kat",
]

