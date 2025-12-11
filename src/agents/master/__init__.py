"""Master Tier Agents - Tier 1
Ultimate decision authority in the ALC ecosystem.
"""

from src.agents.hoags_agent.hoags_agent import HoagsAgent, get_hoags
from src.agents.ghost_agent.ghost_agent import GhostAgent, get_ghost
from src.agents.operations.frieds_agent import FriedsAgent, get_frieds

__all__ = [
    "HoagsAgent",
    "get_hoags",
    "GhostAgent",
    "get_ghost",
    "FriedsAgent",
    "get_frieds",
]

