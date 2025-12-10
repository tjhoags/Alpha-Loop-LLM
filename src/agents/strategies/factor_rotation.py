
"""
FactorRotationAgent - strategy specialization.
"""
from typing import Any, Dict, List
from src.agents.strategies.base_strategy import BaseStrategyAgent


class FactorRotationAgent(BaseStrategyAgent):
    strategy_name = "Factor Rotation"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="FactorRotationAgent", user_id=user_id)
