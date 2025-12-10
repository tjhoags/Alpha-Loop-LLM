
"""EmergingMarketsAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class EmergingMarketsAgent(BaseSectorAgent):
    sector_name = "Emerging Markets"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="EmergingMarketsAgent", user_id=user_id)
