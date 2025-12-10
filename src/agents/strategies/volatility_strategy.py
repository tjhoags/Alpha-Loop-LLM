
"""VolatilityStrategyAgent - strategy specialization.
"""
from src.agents.strategies.base_strategy import BaseStrategyAgent


class VolatilityStrategyAgent(BaseStrategyAgent):
    strategy_name = "Volatility"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="VolatilityStrategyAgent", user_id=user_id)
