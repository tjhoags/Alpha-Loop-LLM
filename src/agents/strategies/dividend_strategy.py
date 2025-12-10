
"""DividendStrategyAgent - strategy specialization.
"""
from src.agents.strategies.base_strategy import BaseStrategyAgent


class DividendStrategyAgent(BaseStrategyAgent):
    strategy_name = "Dividend"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="DividendStrategyAgent", user_id=user_id)
