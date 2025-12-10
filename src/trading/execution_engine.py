"""================================================================================
EXECUTION ENGINE - IBKR Trading Integration
================================================================================
This is the core trading engine that:
1. Connects to Interactive Brokers (TWS/Gateway)
2. Receives signals from SignalGenerator
3. Passes trades through RiskManager for approval
4. Executes orders and tracks fills

Important:
---------
- Port 7497 = Paper Trading (SAFE)
- Port 7496 = Live Trading (REAL MONEY)
================================================================================

"""

import time
from datetime import datetime
from datetime import time as dt_time
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

try:
    from ib_insync import IB, Contract, Crypto, LimitOrder, MarketOrder, Stock
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Running in simulation mode.")

from src.config.settings import get_settings
from src.database.connection import get_engine
from src.risk.risk_manager import RiskManager
from src.trading.signal_generator import Signal, SignalGenerator


class ExecutionEngine:
    """Production trading engine with IBKR integration.

    Features:
    - Real-time signal processing
    - Risk-checked order execution
    - Position tracking
    - P&L monitoring
    """

    def __init__(self, simulation_mode: bool = False):
        self.settings = get_settings()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()

        self.simulation_mode = simulation_mode or not IB_AVAILABLE
        self.ib: Optional["IB"] = None
        self._connected = False
        self._running = False

        # Track our orders
        self.pending_orders: Dict[int, dict] = {}
        self.filled_orders: List[dict] = []

        logger.info("ExecutionEngine initialized")
        logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else 'LIVE IBKR'}")
        logger.info(f"IBKR Port: {self.settings.ibkr_port} ({'PAPER' if self.settings.ibkr_port == 7497 else 'LIVE'})")

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        if self.simulation_mode:
            logger.info("Running in SIMULATION mode - no IBKR connection")
            self._connected = True
            return True

        if not IB_AVAILABLE:
            logger.error("ib_insync not installed. Cannot connect to IBKR.")
            return False

        try:
            self.ib = IB()
            self.ib.connect(
                host=self.settings.ibkr_host,
                port=self.settings.ibkr_port,
                clientId=self.settings.ibkr_client_id,
            )

            # Set up event handlers
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.errorEvent += self._on_error

            self._connected = True
            logger.info(f"Connected to IBKR at {self.settings.ibkr_host}:{self.settings.ibkr_port}")

            # Log account info
            if self.ib.isConnected():
                accounts = self.ib.managedAccounts()
                logger.info(f"Connected accounts: {accounts}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")
        self._connected = False

    def _on_order_status(self, trade) -> None:
        """Handle order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled

        logger.info(f"Order {order_id}: {status} (filled: {filled})")

        if status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            self._on_order_filled(order_id, fill_price)

    def _on_order_filled(self, order_id: int, fill_price: float) -> None:
        """Process a filled order."""
        if order_id in self.pending_orders:
            order_info = self.pending_orders.pop(order_id)
            order_info["fill_price"] = fill_price
            order_info["fill_time"] = datetime.now()
            order_info["status"] = "FILLED"
            self.filled_orders.append(order_info)

            # Update risk manager
            symbol = order_info["symbol"]
            quantity = order_info["quantity"]
            side = order_info["side"]

            if side == "BUY":
                self.risk_manager.open_position(symbol, quantity, fill_price)
            else:
                self.risk_manager.close_position(symbol, fill_price)

            logger.info(f"Order filled: {side} {quantity} {symbol} @ ${fill_price:.2f}")

    def _on_error(self, reqId, errorCode, errorString, contract) -> None:
        """Handle IBKR errors."""
        # Common non-critical errors
        NON_CRITICAL = [2104, 2106, 2158, 2119]

        if errorCode in NON_CRITICAL:
            logger.debug(f"IBKR info {errorCode}: {errorString}")
        else:
            logger.error(f"IBKR error {errorCode}: {errorString}")

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def _create_contract(self, symbol: str) -> Optional["Contract"]:
        """Create IBKR contract for a symbol."""
        if not IB_AVAILABLE:
            return None

        if "-USD" in symbol:  # Crypto
            base = symbol.split("-")[0]
            return Crypto(base, "PAXOS", "USD")
        else:  # Stock
            return Stock(symbol, "SMART", "USD")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        if self.simulation_mode:
            # Return simulated price from database
            return self._get_last_price_from_db(symbol)

        if not self.ib or not self.ib.isConnected():
            return None

        contract = self._create_contract(symbol)
        if not contract:
            return None

        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data

            price = ticker.last or ticker.close
            self.ib.cancelMktData(contract)

            return float(price) if price else None
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def _get_last_price_from_db(self, symbol: str) -> Optional[float]:
        """Get last price from database (for simulation)."""
        try:
            engine = get_engine()
            query = """
            SELECT TOP 1 [close]
            FROM price_bars
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            """
            result = pd.read_sql(query, engine, params={"symbol": symbol})
            if not result.empty:
                return float(result.iloc[0]["close"])
        except Exception as e:
            logger.error(f"Failed to get DB price for {symbol}: {e}")
        return None

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data for signal generation."""
        try:
            engine = get_engine()
            query = """
            SELECT symbol, timestamp, [open], high, low, [close], volume
            FROM price_bars
            WHERE symbol = :symbol
              AND timestamp >= DATEADD(day, :days, GETDATE())
            ORDER BY timestamp ASC
            """
            return pd.read_sql(query, engine, params={"symbol": symbol, "days": -abs(days)})
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    def execute_signal(self, signal: Signal) -> bool:
        """Execute a trading signal after risk checks.
        Returns True if order was placed.
        """
        if not signal.is_actionable:
            logger.debug(f"Signal for {signal.symbol} not actionable (conf={signal.confidence:.2f})")
            return False

        # Get current price
        price = self.get_current_price(signal.symbol)
        if not price:
            logger.warning(f"Cannot execute signal for {signal.symbol}: no price available")
            return False

        # Calculate position size
        position_value = self.risk_manager.calculate_position_size(
            symbol=signal.symbol,
            signal_confidence=signal.confidence,
        )

        if position_value <= 0:
            logger.info(f"Risk manager rejected position for {signal.symbol}")
            return False

        quantity = self.risk_manager.shares_from_value(position_value, price)
        if quantity <= 0:
            logger.info(f"Position size too small for {signal.symbol}")
            return False

        # Determine side
        side = "BUY" if signal.direction == "LONG" else "SELL"

        # Risk approval
        approved, reason = self.risk_manager.approve_trade(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            price=price,
        )

        if not approved:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return False

        # Place order
        return self._place_order(signal.symbol, side, quantity, price)

    def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "MARKET",
    ) -> bool:
        """Place an order with IBKR."""
        order_info = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "submit_time": datetime.now(),
            "status": "SUBMITTED",
        }

        if self.simulation_mode:
            # Simulate immediate fill
            logger.info(f"[SIMULATION] Order: {side} {quantity} {symbol} @ ${price:.2f}")
            order_info["fill_price"] = price
            order_info["fill_time"] = datetime.now()
            order_info["status"] = "FILLED"
            self.filled_orders.append(order_info)

            # Update risk manager
            if side == "BUY":
                self.risk_manager.open_position(symbol, quantity, price)
            else:
                self.risk_manager.close_position(symbol, price)

            return True

        # Real IBKR order
        if not self.ib or not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return False

        try:
            contract = self._create_contract(symbol)
            if not contract:
                return False

            self.ib.qualifyContracts(contract)

            if order_type == "MARKET":
                order = MarketOrder(side, quantity)
            else:
                order = LimitOrder(side, quantity, price)

            trade = self.ib.placeOrder(contract, order)
            order_id = trade.order.orderId

            self.pending_orders[order_id] = order_info
            logger.info(f"Order placed: {side} {quantity} {symbol} (ID: {order_id})")

            return True

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return False

    # =========================================================================
    # MAIN TRADING LOOP
    # =========================================================================

    def is_market_open(self) -> bool:
        """Check if US stock market is open."""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()

        return market_open <= current_time <= market_close

    def run_trading_cycle(self) -> None:
        """Run one trading cycle: generate signals and execute."""
        logger.info("=" * 60)
        logger.info("Starting trading cycle")

        # Check kill switches
        can_trade, reason = self.risk_manager.check_kill_switches()
        if not can_trade:
            logger.warning(f"Trading disabled: {reason}")
            return

        # Get price data for all symbols
        price_data = {}
        for symbol in self.settings.target_symbols:
            df = self.get_historical_data(symbol, days=30)
            if not df.empty:
                price_data[symbol] = df

        if not price_data:
            logger.warning("No price data available")
            return

        # Generate signals
        signals = self.signal_generator.generate_all_signals(price_data)

        # Execute actionable signals
        executed = 0
        for signal in signals:
            if signal.is_actionable:
                if self.execute_signal(signal):
                    executed += 1

        logger.info(f"Executed {executed} trades this cycle")
        self.risk_manager.log_status()

    def start(self) -> None:
        """Start the trading engine."""
        logger.info("=" * 60)
        logger.info("STARTING EXECUTION ENGINE")
        logger.info("=" * 60)

        # Connect to IBKR
        if not self.connect():
            logger.error("Failed to connect. Running in simulation mode.")
            self.simulation_mode = True
            self._connected = True

        self._running = True

        # Main loop
        try:
            while self._running:
                if self.is_market_open() or self.simulation_mode:
                    self.run_trading_cycle()
                else:
                    logger.info("Market closed. Waiting...")

                # Sleep between cycles (5 minutes)
                time.sleep(300)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the trading engine."""
        self._running = False
        self.disconnect()

        # Log final stats
        self.risk_manager.log_status()
        logger.info(f"Total orders filled: {len(self.filled_orders)}")
        logger.info("Execution engine stopped")


def main():
    """Entry point for trading engine."""
    settings = get_settings()
    logger.add(
        settings.logs_dir / "trading_engine.log",
        rotation="20 MB",
        level=settings.log_level,
    )

    # SAFETY: Default to simulation for testing
    engine = ExecutionEngine(simulation_mode=True)
    engine.start()


if __name__ == "__main__":
    main()


