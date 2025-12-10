
"""ValueStrategyAgent - strategy specialization.
"""
from src.agents.strategies.base_strategy import BaseStrategyAgent


class ValueStrategyAgent(BaseStrategyAgent):
    strategy_name = "Value"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="ValueStrategyAgent", user_id=user_id)
