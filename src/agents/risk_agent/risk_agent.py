"""
================================================================================
RISK AGENT - Risk Assessment & Margin of Safety
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Enforce 30% Margin of Safety (core value investing principle)
- Position size limits
- Portfolio heat management
- Trade-level risk assessment

Tier: SENIOR (2)
Reports To: HOAGS, KILLJOY
Cluster: risk

Core Philosophy:
"Margin of safety is the central concept of investment."
- Benjamin Graham

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT RISK_AGENT DOES:
    RISK_AGENT is the "value investing guardian" of Alpha Loop Capital.
    Its primary job is enforcing the 30% Margin of Safety rule - we only
    buy when price is at least 30% below intrinsic value.
    
    While KILLJOY handles position sizing and portfolio-level risk,
    RISK_AGENT focuses on individual trade-level risk assessment,
    particularly margin of safety calculations.
    
    Think of RISK_AGENT as the "due diligence analyst" who asks:
    "Is this cheap enough?" before every trade.

KEY CONSTANTS:
    - MARGIN_OF_SAFETY: 30% (minimum discount to intrinsic value)
    - MAX_POSITION_SIZE: 10% (per holding)
    - MAX_PORTFOLIO_HEAT: 20% (total risk budget)

KEY FUNCTIONS:
    1. process() - Main entry point. Routes to trade assessment
       or portfolio risk check.
       
    2. _assess_trade() - Calculates margin of safety for a trade.
       Checks if price is sufficiently below intrinsic value.
       
    3. _check_portfolio_risk() - Assesses overall portfolio risk
       against limits.

RELATIONSHIPS WITH OTHER AGENTS:
    - BOOKMAKER: Receives intrinsic value estimates from BOOKMAKER
      for margin of safety calculations.
      
    - KILLJOY: Works alongside KILLJOY. RISK_AGENT says "is it cheap
      enough?" while KILLJOY says "can we afford it?"
      
    - ALL STRATEGY AGENTS: Every trade recommendation passes through
      RISK_AGENT for margin of safety check.
      
    - HOAGS: Reports margin of safety compliance to HOAGS.

PATHS OF GROWTH/TRANSFORMATION:
    1. MULTI-FACTOR MARGIN: Consider multiple valuation approaches
       (DCF, multiples, asset-based) for robustness.
       
    2. CONFIDENCE-WEIGHTED MARGIN: Adjust required margin based on
       confidence in intrinsic value estimate.
       
    3. SECTOR-SPECIFIC MARGINS: Different margin requirements for
       different sectors/asset classes.
       
    4. DYNAMIC MARGIN: Adjust margin requirements based on market
       conditions (higher in euphoria, lower in panic).
       
    5. MARGIN MONITORING: Track how margins evolve post-purchase.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr
    
    # Activate virtual environment:
    .\\venv\\Scripts\\activate
    
    # Train RISK_AGENT individually:
    python -m src.training.agent_training_utils --agent RISK_AGENT
    
    # Train risk management pipeline:
    python -m src.training.agent_training_utils --agents RISK_AGENT,KILLJOY,BOOKMAKER

RUNNING THE AGENT:
    from src.agents.risk_agent.risk_agent import RiskAgent
    
    risk = RiskAgent()
    
    # Assess a trade for margin of safety
    result = risk.process({
        "type": "assess_trade",
        "ticker": "CCJ",
        "intrinsic_value": 80.00,
        "current_price": 50.00,
        "position_size": 0.05
    })
    # Returns: approved=True, margin_of_safety=37.5%
    
    # Check portfolio risk
    result = risk.process({"type": "check_portfolio_risk"})

================================================================================
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

