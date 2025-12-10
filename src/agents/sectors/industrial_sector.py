
"""IndustrialSectorAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class IndustrialSectorAgent(BaseSectorAgent):
    sector_name = "Industrial"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="IndustrialSectorAgent", user_id=user_id)
