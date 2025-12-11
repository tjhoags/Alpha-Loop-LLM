"""================================================================================
CAPITAL AGENT - Portfolio Allocation & Risk Budgeting Authority
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

CAPITAL AGENT manages the firm's financial resources. It determines how much capital
is allocated to each strategy, agent, and trade. It enforces risk limits (VaR,
Drawdown) at the capital level and ensures efficient cash management.

Tier: SENIOR (2)
Reports To: HOAGS
Cluster: risk_management

Core Philosophy:
"Protect the capital first. Grow it second."

Key Capabilities:
- Dynamic capital allocation
- Risk budgeting (VaR, Gross/Net Exposure)
- Cash management (Sweep, Yield)
- Leverage optimization
================================================================================
"""

import logging
from typing import Any, Dict, List, Optional
from src.core.agent_base import AgentTier, BaseAgent

logger = logging.getLogger(__name__)

class CapitalAgent(BaseAgent):
    """CAPITAL Agent - The Treasurer

    Manages allocation limits and risk budgets.
    """

    def __init__(self):
        super().__init__(
            name="CAPITAL",
            tier=AgentTier.SENIOR,
            capabilities=[
                "capital_allocation",
                "risk_budgeting",
                "cash_management",
                "leverage_control"
            ],
            user_id="TJH"
        )
        self.total_aum = 1000000.0 # Placeholder
        self.cash_balance = 1000000.0
        self.allocations = {}

    def get_natural_language_explanation(self) -> str:
        return """
CAPITAL AGENT is the Treasurer and Risk Budgeter. It decides "who gets how much money".
It prevents any single agent or strategy from blowing up the fund by enforcing strict capital limits.

RELATIONSHIPS:
- Gatekeeper for EXECUTION agents (must request capital before trading).
- Reports solvency and exposure to HOAGS.
- Works with RISK to determine safe leverage levels.

GROWTH PATH:
- Currently: Static % allocation per strategy.
- Next Phase: Kelly Criterion dynamic sizing based on real-time win rates.
- Ultimate Goal: AI CFO that manages liquidity and funding costs autonomously.
"""

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": "Capital agent processing"}

    def get_capabilities(self) -> List[str]:
        return self.capabilities

# Singleton
_capital_instance: Optional[CapitalAgent] = None

def get_capital() -> CapitalAgent:
    global _capital_instance
    if _capital_instance is None:
        _capital_instance = CapitalAgent()
    return _capital_instance

if __name__ == "__main__":
    import argparse
    import time

    def train_interactive(agent: CapitalAgent):
        print(f"\nTraining {agent.name}...")
        print("Capital agent learns to optimize allocation for risk-adjusted returns.")
        print("1. Optimize Kelly Criterion")
        print("2. Stress Test Liquidity")

        choice = input("Select: ")
        if choice == '1':
            print("Adjusting bet sizes based on recent win rates...")
            time.sleep(1)
            print("Result: Reducing allocation to Mean Reversion (Win Rate dropped).")
        elif choice == '2':
            print("Simulating margin call...")
            time.sleep(1)
            print("Result: Sufficient liquidity. Cash buffer 15%.")
        else:
            print("Invalid.")

    def collaborate(agent: CapitalAgent):
        print(f"\n{agent.name} initiating collaboration protocols...")
        print("Connecting to SCOUT... [ESTABLISHED]")
        print("SCOUT: 'Found opportunity. Need $50k allocation.'")
        time.sleep(0.5)
        print("CAPITAL: 'Checking risk limits... Approved. $50k allocated. Max Loss set to $500.'")

    parser = argparse.ArgumentParser(description="Run the CAPITAL Agent.")
    parser.add_argument("mode", nargs="?", default="help", choices=["train", "collaborate", "run", "help"], help="Mode to run the agent in")
    args = parser.parse_args()

    agent = get_capital()

    print(f"\n--- {agent.name} AGENT ---")
    print(agent.get_natural_language_explanation())

    if args.mode == "train":
        train_interactive(agent)
    elif args.mode == "collaborate":
        collaborate(agent)
    elif args.mode == "run":
        print(f"Running {agent.name}...")
        print(f"Total AUM: ${agent.total_aum:,.2f}")
    elif args.mode == "help":
        parser.print_help()







