
"""HealthcareSectorAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class HealthcareSectorAgent(BaseSectorAgent):
    sector_name = "Healthcare"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="HealthcareSectorAgent", user_id=user_id)
