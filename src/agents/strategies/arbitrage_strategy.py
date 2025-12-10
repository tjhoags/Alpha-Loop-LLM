
"""
ArbitrageStrategyAgent - strategy specialization.
"""
from typing import Any, Dict, List
from src.agents.strategies.base_strategy import BaseStrategyAgent


class ArbitrageStrategyAgent(BaseStrategyAgent):
    strategy_name = "Arbitrage"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="ArbitrageStrategyAgent", user_id=user_id)
