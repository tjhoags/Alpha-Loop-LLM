"""
================================================================================
ALPHA LOOP HUB - Internal Communication & Control Center
================================================================================
Author: Alpha Loop Capital, LLC

The Alpha Loop Hub is the primary interface for Tom and Chris to communicate
with all agents and manage operations.

Features:
- Real-time agent communication
- Dashboard with key metrics
- Task and calendar management
- Learning insights
- Integration controls
================================================================================
"""

from src.app.hub_app import AlphaLoopHub, get_hub_app

__all__ = [
    "AlphaLoopHub",
    "get_hub_app",
]

