"""
Executive Assistant Suite
=========================
Alpha Loop Capital, LLC

Personal and professional executive assistants for the owners.
These agents have READ-ONLY access to local files and software by default.
NO USAGE or actions permitted without WRITTEN PERMISSION from the owner.

SECURITY MODEL:
- All actions require explicit written permission
- Read-only access by default
- Full audit trail of all requests and permissions
- Permission tokens expire and must be renewed

EXECUTIVE ASSISTANTS:
- KAT: Tom Hogan's Executive Assistant (Founder & CIO)
- SHYLA: Chris Friedman's Executive Assistant (COO)

CO-EXECUTIVE ASSISTANTS (Report to both KAT and SHYLA):
- MARGOT_ROBBIE: Co-Executive Assistant
- ANNA_KENDRICK: Co-Executive Assistant

Authors: Tom Hogan (Founder & CIO) & Chris Friedman (COO)
"""

from .base_executive_assistant import (
    BaseExecutiveAssistant,
    PermissionLevel,
    PermissionToken,
    PermissionRequest,
    AccessScope,
)
from .kat import KatAssistant, get_kat
from .shyla import ShylaAssistant, get_shyla
from .margot_robbie import MargotRobbieAssistant, get_margot_robbie
from .anna_kendrick import AnnaKendrickAssistant, get_anna_kendrick

__all__ = [
    # Base classes
    "BaseExecutiveAssistant",
    "PermissionLevel",
    "PermissionToken",
    "PermissionRequest",
    "AccessScope",
    # Primary Executive Assistants
    "KatAssistant",
    "get_kat",
    "ShylaAssistant",
    "get_shyla",
    # Co-Executive Assistants
    "MargotRobbieAssistant",
    "get_margot_robbie",
    "AnnaKendrickAssistant",
    "get_anna_kendrick",
]
