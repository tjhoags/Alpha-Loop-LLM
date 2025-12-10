
"""TechSectorAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class TechSectorAgent(BaseSectorAgent):
    sector_name = "Tech"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="TechSectorAgent", user_id=user_id)
