
"""RealestateSectorAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class RealestateSectorAgent(BaseSectorAgent):
    sector_name = "Realestate"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="RealestateSectorAgent", user_id=user_id)
