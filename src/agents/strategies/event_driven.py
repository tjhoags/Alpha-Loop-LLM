
"""EventDrivenAgent - strategy specialization.
"""
from src.agents.strategies.base_strategy import BaseStrategyAgent


class EventDrivenAgent(BaseStrategyAgent):
    strategy_name = "Event Driven"

    def __init__(self, user_id: str = "TJH"):
        super().__init__(name="EventDrivenAgent", user_id=user_id)
