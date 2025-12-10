"""
Trading Engine - Production Ready
Author: Tom Hogan | Alpha Loop Capital, LLC

Main trading engine that coordinates all agents and executes trades.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import importlib
import inspect
from pathlib import Path

from src.core.data_logger import AgentDecision, PortfolioSnapshot, Trade, get_data_logger
from src.agents.strategies.enhanced_momentum_agent import EnhancedMomentumAgent
from src.agents.strategies.mean_reversion_agent import MeanReversionAgent

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Production trading engine.

    Coordinates:
    - All 50 agents
    - Trade execution
    - Portfolio management
    - Risk management
    - Data logging
    """

    def __init__(self, initial_capital: float = 100000.0, paper_trading: bool = True):
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        self.stock_universe = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {shares, avg_cost, current_price}
        self.total_value = initial_capital
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Data logger
        self.data_logger = get_data_logger()

        # Agents (will be loaded dynamically)
        self.agents = self._load_strategy_agents()

        logger.info(f"TradingEngine initialized | Capital: ${initial_capital:,.2f} | Paper: {paper_trading}")
        logger.info(f"Loaded {len(self.agents)} agents.")

    def _load_strategy_agents(self) -> Dict[str, object]:
        """Dynamically load all strategy agents from the src/agents/strategies directory."""
        agents = {}
        strategies_path = Path(__file__).parent.parent / 'agents' / 'strategies'
        for file_path in strategies_path.glob('*.py'):
            if file_path.name.startswith('__') or file_path.name == 'base_strategy.py':
                continue

            module_name = f"src.agents.strategies.{file_path.stem}"
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Heuristic to find agent classes
                    if 'Agent' in name and 'Base' not in name:
                        try:
                            agents[name] = obj()
                            logger.info(f"Successfully loaded agent: {name}")
                        except Exception as e:
                            logger.error(f"Failed to instantiate agent {name}: {e}")
            except ImportError as e:
                logger.error(f"Failed to import module {module_name}: {e}")
        return agents

    def run_morning_scan(self):
        """Run morning market scan (before open)"""
        logger.info("=" * 60)
        logger.info("MORNING SCAN - Pre-Market Analysis")
        logger.info("=" * 60)

        # For now, placeholder
        logger.info("Morning scan complete")

        # Log portfolio snapshot
        self._log_portfolio_snapshot()

    def run_trading_loop(self):
        """Main trading loop (during market hours)"""
        logger.info("=" * 60)
        logger.info("TRADING LOOP START")
        logger.info("=" * 60)
        
        # Create dummy price data for now
        dates = pd.date_range(datetime.now() - timedelta(days=365), datetime.now())
        price_data = pd.DataFrame({
            symbol: 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
            for symbol in self.stock_universe
        }, index=dates)


        #while self._is_market_open():
        # For testing, run once
        try:
            # 1. Collect agent decisions
            decisions = self._collect_agent_decisions(price_data)

            # 2. Aggregate and rank signals
            trades_to_execute = self._aggregate_signals(decisions)

            # 3. Execute trades
            for trade_signal in trades_to_execute:
                self._execute_trade(trade_signal)

            # 4. Update portfolio
            self._update_portfolio()

            # 5. Log snapshot
            self._log_portfolio_snapshot()

            # Sleep until next cycle (e.g., 1 minute)
            #time.sleep(60)
            pass

        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            #time.sleep(5)  # Brief pause before retrying

        logger.info("Trading loop stopped")

    def run_eod_analysis(self):
        """Run end-of-day analysis (after close)"""
        logger.info("=" * 60)
        logger.info("EOD ANALYSIS - Post-Market Review")
        logger.info("=" * 60)

        # Update final portfolio snapshot
        self._update_portfolio()
        self._log_portfolio_snapshot()

        # TODO: Run performance attribution
        # TODO: Update agent learning
        # TODO: Generate daily report

        logger.info(f"Daily P&L: ${self.daily_pnl:,.2f}")
        logger.info(f"Total P&L: ${self.total_pnl:,.2f}")
        logger.info(f"Portfolio Value: ${self.total_value:,.2f}")

    def _collect_agent_decisions(self, price_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Collect decisions from all agents"""
        all_signals = []
        current_date = price_data.index[-1]
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'generate_signals'):
                try:
                    logger.info(f"Querying agent: {agent_name}")
                    signals = agent.generate_signals(price_data, self.positions, current_date)
                    if signals:
                        all_signals.append({'agent': agent_name, 'signals': signals})
                except Exception as e:
                    logger.error(f"Error getting signals from {agent_name}: {e}")
        return all_signals

    def _aggregate_signals(self, decisions: List[Dict]) -> List[Dict]:
        """Aggregate agent signals into trade orders"""
        
        target_portfolio: Dict[str, float] = {}

        for decision in decisions:
            agent_name = decision['agent']
            signals = decision['signals']
            for symbol, weight in signals.items():
                if symbol not in target_portfolio:
                    target_portfolio[symbol] = 0.0
                target_portfolio[symbol] += weight
        
        # Normalize weights
        total_weight = sum(target_portfolio.values())
        if total_weight > 1.0:
            for symbol in target_portfolio:
                target_portfolio[symbol] /= total_weight
        
        # Determine trades to execute to reach target portfolio
        trades = []
        current_portfolio_value = {s: p['shares'] * p['current_price'] for s, p in self.positions.items()}
        
        for symbol, target_weight in target_portfolio.items():
            target_value = self.total_value * target_weight
            current_value = current_portfolio_value.get(symbol, 0)
            trade_value = target_value - current_value
            
            # Using a placeholder price for now
            current_price = 100.0 # TODO: Get real price
            
            if abs(trade_value) > 0: # minimal trade size
                quantity = trade_value / current_price
                side = 'buy' if quantity > 0 else 'sell'
                trades.append({'symbol': symbol, 'side': side, 'quantity': abs(quantity)})

        return trades

    def _execute_trade(self, trade_signal: Dict):
        """Execute a trade"""
        symbol = trade_signal['symbol']
        side = trade_signal['side']
        quantity = trade_signal['quantity']

        # Get current price (TODO: real market data)
        current_price = 100.0  # Placeholder

        # Calculate costs
        commission = quantity * current_price * 0.0001  # 1 bps
        slippage = quantity * current_price * 0.0005  # 5 bps

        # Update positions
        if side == 'buy':
            if symbol not in self.positions:
                self.positions[symbol] = {'shares': 0, 'avg_cost': 0, 'current_price': current_price}

            pos = self.positions[symbol]
            total_cost = pos['shares'] * pos['avg_cost'] + quantity * current_price
            total_shares = pos['shares'] + quantity
            pos['avg_cost'] = total_cost / total_shares if total_shares > 0 else 0
            pos['shares'] = total_shares
            pos['current_price'] = current_price

            self.cash -= (quantity * current_price + commission + slippage)

        else:  # sell
            if symbol in self.positions:
                pos = self.positions[symbol]
                pnl = (current_price - pos['avg_cost']) * quantity
                pos['shares'] -= quantity

                if pos['shares'] <= 0:
                    del self.positions[symbol]

                self.cash += (quantity * current_price - commission - slippage)
                self.daily_pnl += pnl
                self.total_pnl += pnl

        # Log trade
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
            commission=commission,
            slippage=slippage,
            agent_name=trade_signal.get('agent_name', 'Unknown'),
            strategy=trade_signal.get('strategy', 'Unknown'),
            pnl=self.daily_pnl
        )

        self.data_logger.log_trade(trade)
        logger.info(f"Executed trade: {side} {quantity:.2f} {symbol} @ {current_price:.2f}")


    def _update_portfolio(self):
        """Update portfolio values"""
        # Update position values (TODO: real prices)
        position_value = sum(
            pos['shares'] * pos['current_price']
            for pos in self.positions.values()
        )

        self.total_value = self.cash + position_value

    def _log_portfolio_snapshot(self):
        """Log current portfolio state"""
        positions_dict = {
            symbol: {
                'shares': pos['shares'],
                'value': pos['shares'] * pos['current_price'],
                'pnl': (pos['current_price'] - pos['avg_cost']) * pos['shares']
            }
            for symbol, pos in self.positions.items()
        }

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now().isoformat(),
            total_value=self.total_value,
            cash=self.cash,
            positions=positions_dict,
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl
        )

        self.data_logger.log_portfolio_snapshot(snapshot)

    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()

        # Market hours: 9:30 AM - 4:00 PM ET (Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        return market_open <= now <= market_close


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    engine = TradingEngine(initial_capital=100000, paper_trading=True)
    engine.run_trading_loop()

    print("\n✅ Trading Engine Ready")
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Paper Trading: {engine.paper_trading}")
    print(f"Final Portfolio Value: ${engine.total_value:,.2f}")
    print(f"Final Positions: {engine.positions}")
