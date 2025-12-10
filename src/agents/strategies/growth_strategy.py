
"""GrowthStrategyAgent - strategy specialization.
"""
from src.agents.strategies.base_strategy import BaseStrategyAgent


class GrowthStrategyAgent(BaseStrategyAgent):
    strategy_name = "Growth"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="GrowthStrategyAgent", user_id=user_id)
