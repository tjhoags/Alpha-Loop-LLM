
"""MaterialsSectorAgent - sector specialization.
"""
from src.agents.sectors.base_sector import BaseSectorAgent


class MaterialsSectorAgent(BaseSectorAgent):
    sector_name = "Materials"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="MaterialsSectorAgent", user_id=user_id)
