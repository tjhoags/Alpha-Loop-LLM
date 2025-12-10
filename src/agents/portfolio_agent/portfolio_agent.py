"""
================================================================================
PORTFOLIO AGENT - Portfolio Management & Optimization
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Track current positions
- Calculate rebalancing needs
- Portfolio optimization
- Performance attribution

Tier: SENIOR (2)
Reports To: HOAGS, KILLJOY
Cluster: portfolio

Core Philosophy:
"Know what you own. Own what you know."

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT PORTFOLIO_AGENT DOES:
    PORTFOLIO_AGENT is the "book keeper" of Alpha Loop Capital. It
    tracks every position, calculates performance, and determines
    when rebalancing is needed.
    
    While KILLJOY focuses on risk limits, PORTFOLIO_AGENT focuses on
    portfolio construction - ensuring the portfolio is optimally
    allocated to capture the signals from strategy agents.
    
    Think of PORTFOLIO_AGENT as the "portfolio manager" who maintains
    the overall portfolio structure and ensures proper diversification.

KEY FUNCTIONS:
    1. process() - Main entry point. Routes to appropriate portfolio
       management method.
       
    2. _get_positions() - Returns current positions with quantities
       and values.
       
    3. _calculate_rebalance() - Determines trades needed to reach
       target allocation.
       
    4. _update_position() - Updates position after a trade fills.

RELATIONSHIPS WITH OTHER AGENTS:
    - EXECUTION_AGENT: Receives filled trade notifications to update
      positions.
      
    - KILLJOY: Works closely with KILLJOY. PORTFOLIO_AGENT proposes
      rebalancing, KILLJOY checks risk.
      
    - STRINGS: Receives optimal weights from STRINGS for portfolio
      construction.
      
    - ALL STRATEGY AGENTS: Aggregates signals from all strategy agents
      to build the target portfolio.

PATHS OF GROWTH/TRANSFORMATION:
    1. MEAN-VARIANCE OPTIMIZATION: Implement proper Markowitz
       optimization for portfolio construction.
       
    2. BLACK-LITTERMAN: Incorporate views from strategy agents into
       optimization framework.
       
    3. FACTOR EXPOSURE: Track and manage factor exposures
       (momentum, value, volatility, etc.)
       
    4. TAX-LOSS HARVESTING: Automated harvesting of losses for
       tax efficiency.
       
    5. TRANSACTION COST OPTIMIZATION: Balance rebalancing precision
       against trading costs.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr
    
    # Activate virtual environment:
    .\\venv\\Scripts\\activate
    
    # Train PORTFOLIO_AGENT individually:
    python -m src.training.agent_training_utils --agent PORTFOLIO_AGENT
    
    # Train portfolio management pipeline:
    python -m src.training.agent_training_utils --agents PORTFOLIO_AGENT,KILLJOY,STRINGS

RUNNING THE AGENT:
    from src.agents.portfolio_agent.portfolio_agent import PortfolioAgent
    
    portfolio = PortfolioAgent()
    
    # Get current positions
    result = portfolio.process({"type": "get_positions"})
    
    # Update a position
    result = portfolio.process({
        "type": "update_position",
        "ticker": "AAPL",
        "quantity": 100,
        "avg_price": 175.50
    })
    
    # Calculate rebalancing
    result = portfolio.process({
        "type": "calculate_rebalance",
        "target_allocation": {"AAPL": 0.08, "NVDA": 0.05}
    })

================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier
from typing import Dict, Any, List


class PortfolioAgent(BaseAgent):
    """
    Senior Agent - Portfolio Management
    
    Tracks positions and manages portfolio allocation.
    """
    
    def __init__(self, user_id: str = "TJH"):
        """Initialize PortfolioAgent."""
        super().__init__(
            name="PortfolioAgent",
            tier=AgentTier.SENIOR,
            capabilities=[
                "position_tracking",
                "rebalancing",
                "portfolio_optimization",
                "performance_attribution",
            ],
            user_id=user_id
        )
        self.positions: Dict[str, Dict] = {}
        self.logger.info("PortfolioAgent initialized")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process portfolio task.
        
        Args:
            task: Task dictionary
            
        Returns:
            Portfolio result
        """
        task_type = task.get('type', 'get_positions')
        
        if task_type == 'get_positions':
            return self._get_positions()
        elif task_type == 'calculate_rebalance':
            return self._calculate_rebalance(task)
        elif task_type == 'update_position':
            return self._update_position(task)
        else:
            return {'success': False, 'error': 'Unknown task type'}
    
    def _get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return {
            'success': True,
            'positions': self.positions,
            'total_positions': len(self.positions),
        }
    
    def _calculate_rebalance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate rebalancing trades needed.
        
        Args:
            task: Task with target allocation
            
        Returns:
            Rebalancing trades
        """
        target_allocation = task.get('target_allocation', {})
        
        self.logger.info("Calculating rebalancing trades")
        
        # Placeholder for actual rebalancing logic
        rebalance_trades = []
        
        return {
            'success': True,
            'trades_needed': len(rebalance_trades),
            'trades': rebalance_trades,
        }
    
    def _update_position(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update a position."""
        ticker = task.get('ticker', 'UNKNOWN')
        quantity = task.get('quantity', 0)
        avg_price = task.get('avg_price', 0)
        
        self.positions[ticker] = {
            'quantity': quantity,
            'avg_price': avg_price,
        }
        
        return {
            'success': True,
            'ticker': ticker,
            'position': self.positions[ticker],
        }
    
    def get_capabilities(self) -> List[str]:
        """Return PortfolioAgent capabilities."""
        return self.capabilities

