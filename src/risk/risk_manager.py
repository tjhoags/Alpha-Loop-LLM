"""================================================================================
INSTITUTIONAL RISK MANAGER
================================================================================
This module implements hard risk controls that protect capital. Every trade
must pass through these checks before execution.

Key Features:
- Daily loss kill switch (2% max)
- Drawdown protection (5% max)
- Position sizing via Kelly Criterion
- Exposure limits per position and total
- Real-time P&L tracking
================================================================================
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, Tuple

from loguru import logger

from src.config.settings import get_settings


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity > 0:  # Long
            return self.quantity * (self.current_price - self.entry_price)
        else:  # Short
            return abs(self.quantity) * (self.entry_price - self.current_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        cost_basis = abs(self.quantity) * self.entry_price
        return self.unrealized_pnl / cost_basis if cost_basis > 0 else 0.0


@dataclass
class RiskState:
    """Tracks the current risk state of the portfolio."""

    equity: float = 100000.0  # Starting equity
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    peak_equity: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades_today: int = 0
    last_reset_date: date = field(default_factory=date.today)

    @property
    def current_equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.equity + unrealized

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak (as positive decimal, e.g., 0.05 = 5%)"""
        if self.peak_equity <= 0:
            return 0.0
        return max(0, (self.peak_equity - self.current_equity) / self.peak_equity)

    @property
    def daily_loss_pct(self) -> float:
        """Daily loss as positive decimal"""
        if self.equity <= 0:
            return 0.0
        return max(0, -self.daily_pnl / self.equity)

    @property
    def gross_exposure(self) -> float:
        """Total market value of all positions"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        """Gross exposure as % of equity"""
        return self.gross_exposure / self.equity if self.equity > 0 else 0.0


class RiskManager:
    """Institutional-grade risk manager with kill switches and position sizing.

    GRADING CRITERIA FOR RISK:
    - Must never exceed daily loss limit (2%)
    - Must never exceed max drawdown (5%)
    - Position sizes must respect Kelly-based limits
    - Must maintain diversification (max 10 positions)
    """

    def __init__(self, initial_equity: float = 100000.0):
        self.settings = get_settings()
        self.state = RiskState(
            equity=initial_equity,
            peak_equity=initial_equity,
        )
        self._trading_halted = False
        self._halt_reason = ""
        logger.info(f"RiskManager initialized with equity=${initial_equity:,.2f}")
        logger.info(f"Max daily loss: {self.settings.max_daily_loss_pct*100}%")
        logger.info(f"Max drawdown: {self.settings.max_drawdown_pct*100}%")
        logger.info(f"Max position size: {self.settings.max_position_size_pct*100}%")

    # =========================================================================
    # KILL SWITCHES
    # =========================================================================

    def check_kill_switches(self) -> Tuple[bool, str]:
        """Returns (can_trade, reason).
        If can_trade is False, ALL trading must stop immediately.
        """
        # Check if new day - reset daily counters
        today = date.today()
        if self.state.last_reset_date != today:
            self._reset_daily()

        # Kill Switch 1: Daily Loss Limit
        if self.state.daily_loss_pct >= self.settings.max_daily_loss_pct:
            self._halt_trading(f"DAILY LOSS LIMIT BREACHED: {self.state.daily_loss_pct*100:.2f}%")
            return False, self._halt_reason

        # Kill Switch 2: Max Drawdown
        if self.state.drawdown >= self.settings.max_drawdown_pct:
            self._halt_trading(f"MAX DRAWDOWN BREACHED: {self.state.drawdown*100:.2f}%")
            return False, self._halt_reason

        # Already halted?
        if self._trading_halted:
            return False, self._halt_reason

        return True, "OK"

    def _halt_trading(self, reason: str) -> None:
        """Emergency halt all trading."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.critical(f"[ALERT] TRADING HALTED: {reason}")

    def _reset_daily(self) -> None:
        """Reset daily counters for new trading day."""
        self.state.last_reset_date = date.today()
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Daily risk counters reset.")

    # =========================================================================
    # POSITION SIZING - Kelly Criterion
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        win_rate: float = 0.52,
        avg_win_loss_ratio: float = 1.2,
    ) -> int:
        """Calculate position size using Kelly Criterion with safety caps.

        Kelly Formula: f* = (p * b - q) / b
        where:
            p = win probability
            b = win/loss ratio
            q = 1 - p (loss probability)

        We use quarter-Kelly for safety (kelly_fraction_cap = 0.25)
        """
        # Pre-flight checks
        can_trade, reason = self.check_kill_switches()
        if not can_trade:
            logger.warning(f"Position size = 0 due to kill switch: {reason}")
            return 0

        # Check position limits
        if len(self.state.positions) >= self.settings.max_positions:
            logger.warning(f"Max positions ({self.settings.max_positions}) reached.")
            return 0

        # Already have position in this symbol?
        if symbol in self.state.positions:
            logger.warning(f"Already have position in {symbol}.")
            return 0

        # Kelly calculation
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio

        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, kelly_fraction)  # Can't be negative

        # Apply safety cap (quarter-Kelly)
        safe_fraction = kelly_fraction * self.settings.kelly_fraction_cap

        # Apply signal confidence
        adjusted_fraction = safe_fraction * signal_confidence

        # Apply max position size limit
        max_fraction = self.settings.max_position_size_pct
        final_fraction = min(adjusted_fraction, max_fraction)

        # Calculate dollar amount
        position_value = self.state.current_equity * final_fraction

        logger.debug(
            f"Kelly sizing for {symbol}: "
            f"raw_kelly={kelly_fraction:.3f}, "
            f"safe={safe_fraction:.3f}, "
            f"conf_adj={adjusted_fraction:.3f}, "
            f"final={final_fraction:.3f}, "
            f"value=${position_value:,.2f}",
        )

        return int(position_value)  # Return dollar amount to allocate

    def shares_from_value(self, value: float, price: float) -> int:
        """Convert dollar value to share count."""
        if price <= 0:
            return 0
        return int(value / price)

    # =========================================================================
    # POSITION TRACKING
    # =========================================================================

    def open_position(self, symbol: str, quantity: int, price: float) -> bool:
        """Record opening a new position."""
        can_trade, reason = self.check_kill_switches()
        if not can_trade:
            return False

        if symbol in self.state.positions:
            logger.warning(f"Cannot open duplicate position in {symbol}")
            return False

        self.state.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price,
        )
        self.state.trades_today += 1
        logger.info(f"Opened position: {quantity} shares of {symbol} @ ${price:.2f}")
        return True

    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close position and return realized P&L."""
        if symbol not in self.state.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self.state.positions[symbol]
        pos.current_price = price
        realized_pnl = pos.unrealized_pnl

        # Update state
        self.state.equity += realized_pnl
        self.state.daily_pnl += realized_pnl
        self.state.total_pnl += realized_pnl

        # Update peak equity if we're at new high
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        del self.state.positions[symbol]
        self.state.trades_today += 1

        logger.info(
            f"Closed position: {symbol} @ ${price:.2f}, "
            f"P&L=${realized_pnl:,.2f} ({pos.unrealized_pnl_pct*100:.2f}%)",
        )
        return realized_pnl

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.state.positions:
                self.state.positions[symbol].current_price = price

    # =========================================================================
    # TRADE APPROVAL
    # =========================================================================

    def approve_trade(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: int,
        price: float,
    ) -> Tuple[bool, str]:
        """Final approval check before sending order.
        Returns (approved, reason).
        """
        # Kill switch check
        can_trade, reason = self.check_kill_switches()
        if not can_trade:
            return False, reason

        # Position check
        trade_value = quantity * price
        trade_pct = trade_value / self.state.current_equity

        if trade_pct > self.settings.max_position_size_pct:
            return False, f"Trade exceeds max position size ({trade_pct*100:.1f}% > {self.settings.max_position_size_pct*100}%)"

        # Exposure check
        new_exposure = self.state.gross_exposure + trade_value
        new_exposure_pct = new_exposure / self.state.current_equity
        if new_exposure_pct > 1.0:  # Can't exceed 100% exposure
            return False, f"Would exceed 100% exposure ({new_exposure_pct*100:.1f}%)"

        logger.info(f"Trade APPROVED: {side} {quantity} {symbol} @ ${price:.2f}")
        return True, "APPROVED"

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_status(self) -> Dict:
        """Get current risk status for logging/monitoring."""
        return {
            "equity": self.state.current_equity,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": -self.state.daily_loss_pct if self.state.daily_pnl < 0 else self.state.daily_pnl / self.state.equity,
            "total_pnl": self.state.total_pnl,
            "drawdown_pct": self.state.drawdown,
            "exposure_pct": self.state.exposure_pct,
            "position_count": len(self.state.positions),
            "trades_today": self.state.trades_today,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
        }

    def log_status(self) -> None:
        """Log current risk status."""
        status = self.get_status()
        logger.info(
            f"[RISK] Status | "
            f"Equity: ${status['equity']:,.2f} | "
            f"Daily P&L: ${status['daily_pnl']:,.2f} ({status['daily_pnl_pct']*100:+.2f}%) | "
            f"DD: {status['drawdown_pct']*100:.2f}% | "
            f"Exposure: {status['exposure_pct']*100:.1f}% | "
            f"Positions: {status['position_count']}",
        )
        if status["trading_halted"]:
            logger.critical(f"[ALERT] TRADING HALTED: {status['halt_reason']}")


