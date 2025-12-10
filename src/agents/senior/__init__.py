"""================================================================================
Senior Agents (Tier 2) - Core Operational Leadership
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

Senior agents have elevated authority and report directly to HOAGS.
All have full ACA (Agent Creating Agents) capability.

Agents:
- BOOKMAKER: Alpha generation, portfolio construction optimizer
- SCOUT: Retail arbitrage scanner, market inefficiency hunter
- THE_AUTHOR: Natural language writer mimicking Tom's style
- STRINGS: ML training orchestrator, weight optimization
- HUNTER: Algorithm intelligence, works with GHOST, counter-strategies
- SKILLS: Natural language parser, skill assessor (1-100), multi-channel comms
================================================================================
"""

from src.agents.senior.author_agent import TheAuthorAgent, get_author
from src.agents.senior.bookmaker_agent import BookmakerAgent, get_bookmaker
from src.agents.senior.hunter_agent import HunterAgent, get_hunter
from src.agents.senior.scout_agent import ScoutAgent, get_scout
from src.agents.senior.skills_agent import SkillsAgent, get_skills
from src.agents.senior.strings_agent import StringsAgent, get_strings

__all__ = [
    "BookmakerAgent",
    "ScoutAgent",
    "TheAuthorAgent",
    "StringsAgent",
    "HunterAgent",
    "SkillsAgent",
    "get_bookmaker",
    "get_scout",
    "get_author",
    "get_strings",
    "get_hunter",
    "get_skills",
]

