"""
ExecutionAgent - Broker execution (IBKR, Coinbase)
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Execute trades via IBKR
- Execute crypto trades via Coinbase
- Order management
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier
from typing import Dict, Any, List


class ExecutionAgent(BaseAgent):
    """
    Senior Agent - Trade Execution
    
    Interfaces with brokers to execute approved trades.
    """
    
    def __init__(self, user_id: str = "TJH"):
        """Initialize ExecutionAgent."""
        super().__init__(
            name="ExecutionAgent",
            tier=AgentTier.SENIOR,
            capabilities=[
                "ibkr_execution",
                "coinbase_execution",
                "order_management",
                "execution_quality_monitoring",
            ],
            user_id=user_id
        )
        self.paper_account = "7497"
        self.live_account = "7496"
        self.default_mode = "PAPER"  # Always start with paper
        self.logger.info(f"ExecutionAgent initialized (Mode: {self.default_mode})")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process execution task.
        
        Args:
            task: Task dictionary
            
        Returns:
            Execution result
        """
        broker = task.get('broker', 'ibkr')
        
        if broker == 'ibkr':
            return self._execute_ibkr(task)
        elif broker == 'coinbase':
            return self._execute_coinbase(task)
        else:
            return {'success': False, 'error': f'Unknown broker: {broker}'}
    
    def _execute_ibkr(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade via IBKR.
        
        Args:
            task: Task with trade details
            
        Returns:
            Execution result
        """
        ticker = task.get('ticker', 'UNKNOWN')
        action = task.get('action', 'BUY')
        quantity = task.get('quantity', 0)
        mode = task.get('mode', self.default_mode)
        
        account = self.paper_account if mode == "PAPER" else self.live_account
        
        self.logger.info(
            f"Executing {action} {quantity} {ticker} via IBKR "
            f"(Account: {account}, Mode: {mode})"
        )
        
        # Placeholder for actual IBKR execution via ib_insync
        return {
            'success': True,
            'broker': 'ibkr',
            'account': account,
            'mode': mode,
            'ticker': ticker,
            'action': action,
            'quantity': quantity,
            'order_id': 'ORD-12345',  # From actual execution
            'status': 'FILLED',
        }
    
    def _execute_coinbase(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute crypto trade via Coinbase.
        
        Args:
            task: Task with trade details
            
        Returns:
            Execution result
        """
        symbol = task.get('symbol', 'BTC-USD')
        side = task.get('side', 'buy')
        amount = task.get('amount', 0)
        
        self.logger.info(f"Executing {side} {amount} {symbol} via Coinbase")
        
        # Placeholder for actual Coinbase execution
        return {
            'success': True,
            'broker': 'coinbase',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'order_id': 'CB-12345',  # From actual execution
            'status': 'filled',
        }
    
    def get_capabilities(self) -> List[str]:
        """Return ExecutionAgent capabilities."""
        return self.capabilities

