
"""EnergySectorAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class EnergySectorAgent(BaseSectorAgent):
    sector_name = "Energy"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="EnergySectorAgent", user_id=user_id)
