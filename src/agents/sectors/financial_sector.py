
"""
FinancialSectorAgent - sector specialization.
"""
from typing import Any, Dict, List
from src.agents.sectors.base_sector import BaseSectorAgent


class FinancialSectorAgent(BaseSectorAgent):
    sector_name = "Financial"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="FinancialSectorAgent", user_id=user_id)
