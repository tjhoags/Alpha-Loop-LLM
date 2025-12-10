
"""BaseStrategyAgent - foundation for strategy agents.
"""
from typing import Any, Dict, List

from src.core.agent_base import AgentTier, BaseAgent


class BaseStrategyAgent(BaseAgent):
    strategy_name: str = "generic"

    def __init__(self, name: str, user_id: str = "TJH"):
        super().__init__(
            name=name,
            tier=AgentTier.STRATEGY,
            capabilities=[f"strategy_{self.strategy_name}"],
            user_id=user_id,
        )

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "agent": self.name,
            "strategy": self.strategy_name,
            "task": task.get("type", "generate_signal"),
        }

    def get_capabilities(self) -> List[str]:
        return self.capabilities
