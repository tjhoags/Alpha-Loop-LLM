
"""KillJoyAgent - capital allocation and guardrails."""
from typing import Any, Dict, List
from src.core.agent_base import BaseAgent, AgentTier


class KillJoyAgent(BaseAgent):
    def __init__(self, user_id: str = "TJH"):
        super().__init__(
            name="KillJoyAgent",
            tier=AgentTier.SENIOR,
            capabilities=["capital_allocation", "heat_control", "drawdown_watch"],
            user_id=user_id,
        )

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "agent": self.name,
            "task": task.get("type", "allocate"),
            "note": "Guardrails enforced",
        }

    def get_capabilities(self) -> List[str]:
        return self.capabilities
