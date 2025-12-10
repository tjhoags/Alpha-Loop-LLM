"""
================================================================================
SHYLA AGENT - Chris Friedman's Executive Assistant
================================================================================
Author: Chris Friedman | Alpha Loop Capital, LLC

SHYLA handles everything Chris asks for - professional and personal.
Personal data training is GATED and requires explicit permission from Chris.

Sub-agents:
- COFFEE_BREAK: Schedule optimization, meeting gaps, break reminders
- BEAN_COUNTER: Time tracking, expense reports, budget monitoring
================================================================================
"""

from src.agents.shyla_agent.shyla_agent import ShylaAgent, get_shyla

__all__ = [
    "ShylaAgent",
    "get_shyla",
]

