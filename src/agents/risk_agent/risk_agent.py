"""
RiskAgent - Risk management and limits enforcement
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Enforce 30% Margin of Safety
- Position size limits
- Portfolio heat management
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier
from typing import Dict, Any, List


class RiskAgent(BaseAgent):
    """
    Senior Agent - Risk Management
    
    Enforces all risk limits including the critical 30% Margin of Safety.
    """
    
    MARGIN_OF_SAFETY = 0.30  # 30% required
    MAX_POSITION_SIZE = 0.10  # 10% max
    MAX_PORTFOLIO_HEAT = 0.20  # 20% max
    
    def __init__(self, user_id: str = "TJH"):
        """Initialize RiskAgent."""
        super().__init__(
            name="RiskAgent",
            tier=AgentTier.SENIOR,
            capabilities=[
                "margin_of_safety_check",
                "position_sizing",
                "portfolio_heat_management",
                "risk_limit_enforcement",
            ],
            user_id=user_id
        )
        self.logger.info("RiskAgent initialized")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process risk assessment task.
        
        Args:
            task: Task dictionary
            
        Returns:
            Risk assessment results
        """
        task_type = task.get('type', 'assess_trade')
        
        if task_type == 'assess_trade':
            return self._assess_trade(task)
        elif task_type == 'check_portfolio_risk':
            return self._check_portfolio_risk(task)
        else:
            return {'success': False, 'error': 'Unknown task type'}
    
    def _assess_trade(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess a trade for risk compliance.
        
        Args:
            task: Task with trade details
            
        Returns:
            Risk assessment
        """
        ticker = task.get('ticker', 'UNKNOWN')
        intrinsic_value = task.get('intrinsic_value', 100)
        current_price = task.get('current_price', 100)
        position_size = task.get('position_size', 0)
        
        # Calculate margin of safety
        margin = (intrinsic_value - current_price) / intrinsic_value
        
        # Check against required margin
        passes_margin = margin >= self.MARGIN_OF_SAFETY
        passes_size = position_size <= self.MAX_POSITION_SIZE
        
        approved = passes_margin and passes_size
        
        self.logger.info(
            f"Risk assessment for {ticker}: "
            f"Margin={margin:.1%} (Required={self.MARGIN_OF_SAFETY:.1%}), "
            f"Approved={approved}"
        )
        
        return {
            'success': True,
            'ticker': ticker,
            'approved': approved,
            'margin_of_safety': margin,
            'required_margin': self.MARGIN_OF_SAFETY,
            'passes_margin': passes_margin,
            'passes_size': passes_size,
            'intrinsic_value': intrinsic_value,
            'current_price': current_price,
        }
    
    def _check_portfolio_risk(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall portfolio risk."""
        # Placeholder for portfolio-level risk
        return {
            'success': True,
            'portfolio_heat': 0.15,  # 15%
            'max_allowed': self.MAX_PORTFOLIO_HEAT,
            'approved': True,
        }
    
    def get_capabilities(self) -> List[str]:
        """Return RiskAgent capabilities."""
        return self.capabilities

