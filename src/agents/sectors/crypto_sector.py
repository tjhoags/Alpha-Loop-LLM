
"""
CryptoSectorAgent - sector specialization.
"""
from typing import Any, Dict, List
from src.agents.sectors.base_sector import BaseSectorAgent


class CryptoSectorAgent(BaseSectorAgent):
    sector_name = "Crypto"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="CryptoSectorAgent", user_id=user_id)
