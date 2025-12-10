
"""
MomentumStrategyAgent - strategy specialization.
"""
from typing import Any, Dict, List
from src.agents.strategies.base_strategy import BaseStrategyAgent


class MomentumStrategyAgent(BaseStrategyAgent):
    strategy_name = "Momentum"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="MomentumStrategyAgent", user_id=user_id)
