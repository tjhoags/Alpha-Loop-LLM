
"""
ConsumerSectorAgent - sector specialization.
"""
from typing import Any, Dict, List
from src.agents.sectors.base_sector import BaseSectorAgent


class ConsumerSectorAgent(BaseSectorAgent):
    sector_name = "Consumer"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="ConsumerSectorAgent", user_id=user_id)
