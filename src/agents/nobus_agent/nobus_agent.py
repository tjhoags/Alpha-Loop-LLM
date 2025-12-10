
"""NOBUSAgent - resilience and fault injection."""
from typing import Any, Dict, List
from src.core.agent_base import BaseAgent, AgentTier


class NOBUSAgent(BaseAgent):
    def __init__(self, user_id: str = "TJH"):
        super().__init__(
            name="NOBUSAgent",
            tier=AgentTier.SUPPORT,
            capabilities=["fault_injection", "stress_test"],
            user_id=user_id,
        )

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "agent": self.name,
            "task": task.get("type", "stress_test"),
            "note": "Simulated failure/edge-case applied",
        }

    def get_capabilities(self) -> List[str]:
        return self.capabilities
