
"""
BaseSectorAgent - shared logic for sector-level specialization.
"""
from typing import Any, Dict, List
from src.core.agent_base import BaseAgent, AgentTier


class BaseSectorAgent(BaseAgent):
    sector_name: str = "GENERAL"

    def __init__(self, name: str, user_id: str = "TJH"):
        super().__init__(
            name=name,
            tier=AgentTier.SECTOR,
            capabilities=[f"sector_{self.sector_name.lower()}"],
            user_id=user_id,
        )

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "agent": self.name,
            "sector": self.sector_name,
            "task": task.get("type", "analyze_sector"),
        }

    def get_capabilities(self) -> List[str]:
        return self.capabilities
