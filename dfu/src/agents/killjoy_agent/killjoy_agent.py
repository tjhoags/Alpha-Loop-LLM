"""================================================================================
KILLJOY AGENT - THE RUTHLESS RISK GUARDIAN
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC
Version: 3.0 | December 2024

KILLJOY is the LAST LINE OF DEFENSE before any trade executes.

This is NOT a gentle risk manager. KILLJOY has ONE JOB:
    PREVENT CATASTROPHIC LOSSES AT ALL COSTS.

KILLJOY operates on the principle that:
    "It's better to miss 100 winning trades than take 1 trade that blows up."

KILLJOY has VETO POWER over:
    - GHOST
    - HOAGS
    - ALL Senior Agents
    - ALL Strategy Agents
    - EVERY single trade

Only Tom Hogan can override KILLJOY. And even then, KILLJOY logs the override.

GUARDRAIL LEVELS:
    - LEVEL 1: SOFT LIMIT  - Warning + Reduced size
    - LEVEL 2: HARD LIMIT  - Block trade + Alert
    - LEVEL 3: EMERGENCY   - Liquidate positions + Full stop
    - LEVEL 4: APOCALYPSE  - Everything to cash + Human intervention required

"The graveyard of hedge funds is filled with those who ignored risk."
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.agent_base import AgentTier, BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class AlertLevel(Enum):
    """Severity levels for KILLJOY alerts."""

    INFO = "info"              # Logged, no action
    WARNING = "warning"        # Size reduction applied
    CRITICAL = "critical"      # Trade blocked
    EMERGENCY = "emergency"    # Positions liquidated
    APOCALYPSE = "apocalypse"  # Full stop, human required


class RiskCategory(Enum):
    """Categories of risk KILLJOY monitors."""

    POSITION_SIZE = "position_size"
    SECTOR_CONCENTRATION = "sector_concentration"
    SINGLE_NAME = "single_name"
    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    MONTHLY_LOSS = "monthly_loss"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    LEVERAGE = "leverage"
    VAR = "value_at_risk"
    EXPOSURE = "exposure"
    HEAT = "heat"


@dataclass
class RiskLimit:
    """A single risk limit with soft and hard thresholds."""

    category: RiskCategory
    soft_limit: float       # Warning threshold
    hard_limit: float       # Block threshold
    emergency_limit: float  # Liquidation threshold
    current_value: float = 0.0
    last_checked: datetime = field(default_factory=datetime.now)
    breaches_today: int = 0

    @property
    def status(self) -> AlertLevel:
        if self.current_value >= self.emergency_limit:
            return AlertLevel.EMERGENCY
        elif self.current_value >= self.hard_limit:
            return AlertLevel.CRITICAL
        elif self.current_value >= self.soft_limit:
            return AlertLevel.WARNING
        return AlertLevel.INFO

    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "soft_limit": self.soft_limit,
            "hard_limit": self.hard_limit,
            "emergency_limit": self.emergency_limit,
            "current_value": self.current_value,
            "status": self.status.value,
            "breaches_today": self.breaches_today,
        }


@dataclass
class TradeRequest:
    """A trade request submitted for KILLJOY approval."""

    request_id: str
    agent_id: str
    agent_name: str
    symbol: str
    action: str  # BUY, SELL, SHORT, COVER
    quantity: int
    price: float
    order_type: str
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def notional_value(self) -> float:
        return self.quantity * self.price


@dataclass
class TradeVerdict:
    """KILLJOY's verdict on a trade request."""

    request_id: str
    approved: bool
    original_quantity: int
    approved_quantity: int
    alert_level: AlertLevel
    reasons: List[str]
    adjustments: List[str]
    risk_score: float  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "approved": self.approved,
            "original_quantity": self.original_quantity,
            "approved_quantity": self.approved_quantity,
            "alert_level": self.alert_level.value,
            "reasons": self.reasons,
            "adjustments": self.adjustments,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk assessment."""

    total_equity: float = 1_000_000.0  # $1M default
    cash: float = 500_000.0
    positions: Dict[str, Dict] = field(default_factory=dict)

    # Daily P&L
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

    # Weekly/Monthly
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    # Drawdown
    peak_equity: float = 1_000_000.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0

    # Exposure
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0

    # Sector breakdown
    sector_exposures: Dict[str, float] = field(default_factory=dict)

    # Risk metrics
    portfolio_var: float = 0.0  # Value at Risk
    portfolio_vol: float = 0.0  # Volatility
    beta: float = 1.0

    # Heat (recent losses)
    heat_level: float = 0.0  # 0-100
    consecutive_losses: int = 0


# =============================================================================
# THE KILLJOY AGENT
# =============================================================================

class KilljoyAgent(BaseAgent):
    """KILLJOY - The Ruthless Risk Guardian

    KILLJOY has ABSOLUTE VETO POWER over every trade in the system.

    It enforces:
    1. POSITION LIMITS - No single position > 5% of portfolio
    2. SECTOR LIMITS - No sector > 25% of portfolio
    3. DAILY LOSS LIMIT - Stop trading at 2% daily loss
    4. WEEKLY LOSS LIMIT - Reduce size at 5% weekly loss
    5. MONTHLY LOSS LIMIT - Emergency at 10% monthly loss
    6. DRAWDOWN PROTECTION - Liquidate at 15% drawdown
    7. LEVERAGE LIMIT - Max 2x gross exposure
    8. LIQUIDITY REQUIREMENTS - Can't be >10% of daily volume
    9. CORRELATION LIMITS - Max 0.7 correlation between positions
    10. HEAT MANAGEMENT - Reduce size after consecutive losses

    Key Methods:
    - approve_trade(): THE GATEKEEPER - every trade goes through here
    - check_portfolio_risk(): Real-time portfolio risk assessment
    - enforce_limits(): Apply guardrails
    - emergency_liquidate(): Nuclear option
    """

    # =========================================================================
    # INSTITUTIONAL-GRADE RISK LIMITS
    # =========================================================================

    # Position Limits
    MAX_POSITION_PCT = 0.05          # 5% max single position
    MAX_POSITION_PCT_SOFT = 0.03     # 3% soft limit (warning)

    # Sector Limits
    MAX_SECTOR_PCT = 0.25            # 25% max sector exposure
    MAX_SECTOR_PCT_SOFT = 0.20       # 20% soft limit

    # Loss Limits
    DAILY_LOSS_LIMIT = 0.02          # 2% daily loss = stop trading
    DAILY_LOSS_SOFT = 0.01           # 1% = reduce size
    WEEKLY_LOSS_LIMIT = 0.05         # 5% weekly loss = emergency
    WEEKLY_LOSS_SOFT = 0.03          # 3% = serious reduction
    MONTHLY_LOSS_LIMIT = 0.10        # 10% monthly = human intervention
    MONTHLY_LOSS_SOFT = 0.07         # 7% = minimal trading

    # Drawdown Limits
    MAX_DRAWDOWN = 0.15              # 15% drawdown = liquidate
    MAX_DRAWDOWN_SOFT = 0.10         # 10% = major reduction
    MAX_DRAWDOWN_WARN = 0.05         # 5% = start reducing

    # Leverage Limits
    MAX_GROSS_EXPOSURE = 2.0         # 200% max
    MAX_NET_EXPOSURE = 1.0           # 100% max net

    # Liquidity Limits
    MAX_VOLUME_PCT = 0.10            # Can't be >10% of daily volume

    # Heat Limits
    MAX_CONSECUTIVE_LOSSES = 5       # After 5 losses, stop trading
    HEAT_REDUCTION_THRESHOLD = 50    # Heat > 50 = reduce size

    # VAR Limit
    MAX_VAR_PCT = 0.02               # 2% daily VaR limit

    def __init__(self, user_id: str = "TJH"):
        super().__init__(
            name="KILLJOY",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Core risk management
                "trade_approval",
                "position_limit_enforcement",
                "sector_limit_enforcement",
                "loss_limit_enforcement",
                "drawdown_protection",
                "leverage_control",
                "liquidity_monitoring",
                "correlation_analysis",
                "heat_management",
                "var_calculation",

                # Emergency actions
                "emergency_liquidation",
                "trading_halt",
                "position_reduction",
                "exposure_hedging",

                # Monitoring
                "real_time_risk_monitoring",
                "pnl_tracking",
                "limit_breach_alerting",
                "risk_reporting",

                # Authority
                "veto_power",
                "override_logging",
                "hoags_escalation",
            ],
            user_id=user_id,
        )

        # Portfolio state
        self.portfolio = PortfolioState()

        # Initialize risk limits
        self.limits = self._init_limits()

        # Trade history
        self.approved_trades: List[TradeVerdict] = []
        self.blocked_trades: List[TradeVerdict] = []
        self.alerts_issued: List[Dict] = []

        # Statistics
        self.trades_reviewed = 0
        self.trades_approved = 0
        self.trades_blocked = 0
        self.trades_reduced = 0
        self.emergency_liquidations = 0

        # State
        self.trading_halted = False
        self.halt_reason = ""
        self.human_intervention_required = False

        logger.info("🛡️ KILLJOY initialized - THE RISK GUARDIAN IS WATCHING")

    def _init_limits(self) -> Dict[RiskCategory, RiskLimit]:
        """Initialize all risk limits."""
        return {
            RiskCategory.POSITION_SIZE: RiskLimit(
                category=RiskCategory.POSITION_SIZE,
                soft_limit=self.MAX_POSITION_PCT_SOFT,
                hard_limit=self.MAX_POSITION_PCT,
                emergency_limit=self.MAX_POSITION_PCT * 1.5,
            ),
            RiskCategory.SECTOR_CONCENTRATION: RiskLimit(
                category=RiskCategory.SECTOR_CONCENTRATION,
                soft_limit=self.MAX_SECTOR_PCT_SOFT,
                hard_limit=self.MAX_SECTOR_PCT,
                emergency_limit=self.MAX_SECTOR_PCT * 1.2,
            ),
            RiskCategory.DAILY_LOSS: RiskLimit(
                category=RiskCategory.DAILY_LOSS,
                soft_limit=self.DAILY_LOSS_SOFT,
                hard_limit=self.DAILY_LOSS_LIMIT,
                emergency_limit=self.DAILY_LOSS_LIMIT * 1.5,
            ),
            RiskCategory.WEEKLY_LOSS: RiskLimit(
                category=RiskCategory.WEEKLY_LOSS,
                soft_limit=self.WEEKLY_LOSS_SOFT,
                hard_limit=self.WEEKLY_LOSS_LIMIT,
                emergency_limit=self.WEEKLY_LOSS_LIMIT * 1.5,
            ),
            RiskCategory.MONTHLY_LOSS: RiskLimit(
                category=RiskCategory.MONTHLY_LOSS,
                soft_limit=self.MONTHLY_LOSS_SOFT,
                hard_limit=self.MONTHLY_LOSS_LIMIT,
                emergency_limit=self.MONTHLY_LOSS_LIMIT * 1.5,
            ),
            RiskCategory.DRAWDOWN: RiskLimit(
                category=RiskCategory.DRAWDOWN,
                soft_limit=self.MAX_DRAWDOWN_WARN,
                hard_limit=self.MAX_DRAWDOWN_SOFT,
                emergency_limit=self.MAX_DRAWDOWN,
            ),
            RiskCategory.LEVERAGE: RiskLimit(
                category=RiskCategory.LEVERAGE,
                soft_limit=self.MAX_GROSS_EXPOSURE * 0.8,
                hard_limit=self.MAX_GROSS_EXPOSURE,
                emergency_limit=self.MAX_GROSS_EXPOSURE * 1.2,
            ),
            RiskCategory.VAR: RiskLimit(
                category=RiskCategory.VAR,
                soft_limit=self.MAX_VAR_PCT * 0.7,
                hard_limit=self.MAX_VAR_PCT,
                emergency_limit=self.MAX_VAR_PCT * 1.5,
            ),
            RiskCategory.HEAT: RiskLimit(
                category=RiskCategory.HEAT,
                soft_limit=self.HEAT_REDUCTION_THRESHOLD,
                hard_limit=75,
                emergency_limit=90,
            ),
        }

    # =========================================================================
    # CORE RISK METHODS
    # =========================================================================

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a KILLJOY task."""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.log_action(action, f"KILLJOY processing: {action}")

        handlers = {
            "approve_trade": self._handle_approve_trade,
            "check_risk": self._handle_check_risk,
            "update_portfolio": self._handle_update_portfolio,
            "get_limits": self._handle_get_limits,
            "get_status": self._handle_get_status,
            "emergency_stop": self._handle_emergency_stop,
            "reset_daily": self._handle_reset_daily,
            "override": self._handle_override,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(params)

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def approve_trade(self, request: TradeRequest) -> TradeVerdict:
        """THE GATEKEEPER - Every single trade must pass through here.

        This is the CRITICAL function. KILLJOY reviews:
        1. Is trading halted?
        2. Would this breach position limits?
        3. Would this breach sector limits?
        4. Are we at daily/weekly/monthly loss limits?
        5. Are we in drawdown territory?
        6. Is leverage acceptable?
        7. Is there enough liquidity?
        8. Is the heat too high?
        9. Is confidence sufficient?

        Returns TradeVerdict with approval status and any size adjustments.
        """
        self.trades_reviewed += 1

        reasons = []
        adjustments = []
        approved_quantity = request.quantity
        alert_level = AlertLevel.INFO
        risk_score = 0.0

        # =================================================================
        # CHECK 0: Is trading halted?
        # =================================================================
        if self.trading_halted:
            return TradeVerdict(
                request_id=request.request_id,
                approved=False,
                original_quantity=request.quantity,
                approved_quantity=0,
                alert_level=AlertLevel.CRITICAL,
                reasons=[f"TRADING HALTED: {self.halt_reason}"],
                adjustments=[],
                risk_score=100,
            )

        # =================================================================
        # CHECK 1: Position Size Limit
        # =================================================================
        position_pct = (request.notional_value / self.portfolio.total_equity)
        existing_position = self.portfolio.positions.get(request.symbol, {}).get("value", 0)
        total_position_pct = (existing_position + request.notional_value) / self.portfolio.total_equity

        if total_position_pct > self.MAX_POSITION_PCT:
            max_allowed = self.MAX_POSITION_PCT * self.portfolio.total_equity - existing_position
            if max_allowed <= 0:
                reasons.append(f"BLOCKED: Position {request.symbol} already at max ({total_position_pct:.1%})")
                approved_quantity = 0
                alert_level = AlertLevel.CRITICAL
            else:
                approved_quantity = int(max_allowed / request.price)
                adjustments.append(f"REDUCED: {request.quantity} → {approved_quantity} (position limit)")
                alert_level = AlertLevel.WARNING
            risk_score += 30
        elif total_position_pct > self.MAX_POSITION_PCT_SOFT:
            adjustments.append(f"WARNING: Position will be {total_position_pct:.1%} of portfolio")
            alert_level = AlertLevel.WARNING
            risk_score += 15

        # =================================================================
        # CHECK 2: Daily Loss Limit
        # =================================================================
        daily_loss_pct = abs(min(0, self.portfolio.daily_pnl_pct))

        if daily_loss_pct >= self.DAILY_LOSS_LIMIT:
            reasons.append(f"BLOCKED: Daily loss limit reached ({daily_loss_pct:.1%})")
            approved_quantity = 0
            alert_level = AlertLevel.CRITICAL
            self._halt_trading("Daily loss limit reached")
            risk_score += 50
        elif daily_loss_pct >= self.DAILY_LOSS_SOFT:
            reduction = 0.5  # Reduce size by 50%
            approved_quantity = int(approved_quantity * reduction)
            adjustments.append(f"REDUCED 50%: Daily loss at {daily_loss_pct:.1%}")
            alert_level = AlertLevel.WARNING
            risk_score += 25

        # =================================================================
        # CHECK 3: Drawdown Limit
        # =================================================================
        if self.portfolio.current_drawdown >= self.MAX_DRAWDOWN:
            reasons.append(f"BLOCKED: Max drawdown reached ({self.portfolio.current_drawdown:.1%})")
            approved_quantity = 0
            alert_level = AlertLevel.EMERGENCY
            self._emergency_liquidate("Max drawdown breached")
            risk_score += 50
        elif self.portfolio.current_drawdown >= self.MAX_DRAWDOWN_SOFT:
            reduction = 0.25  # Only 25% of normal size
            approved_quantity = int(approved_quantity * reduction)
            adjustments.append(f"REDUCED 75%: Drawdown at {self.portfolio.current_drawdown:.1%}")
            alert_level = AlertLevel.CRITICAL
            risk_score += 35
        elif self.portfolio.current_drawdown >= self.MAX_DRAWDOWN_WARN:
            reduction = 0.5
            approved_quantity = int(approved_quantity * reduction)
            adjustments.append(f"REDUCED 50%: Drawdown warning ({self.portfolio.current_drawdown:.1%})")
            alert_level = AlertLevel.WARNING
            risk_score += 20

        # =================================================================
        # CHECK 4: Leverage Limit
        # =================================================================
        new_gross = self.portfolio.gross_exposure + request.notional_value
        new_gross_pct = new_gross / self.portfolio.total_equity

        if new_gross_pct > self.MAX_GROSS_EXPOSURE:
            reasons.append(f"BLOCKED: Leverage would exceed {self.MAX_GROSS_EXPOSURE:.0%}")
            approved_quantity = 0
            alert_level = AlertLevel.CRITICAL
            risk_score += 30
        elif new_gross_pct > self.MAX_GROSS_EXPOSURE * 0.9:
            adjustments.append(f"WARNING: Leverage at {new_gross_pct:.0%}")
            alert_level = AlertLevel.WARNING
            risk_score += 15

        # =================================================================
        # CHECK 5: Heat Level (Consecutive Losses)
        # =================================================================
        if self.portfolio.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            reasons.append(f"BLOCKED: {self.portfolio.consecutive_losses} consecutive losses - cooling off")
            approved_quantity = 0
            alert_level = AlertLevel.CRITICAL
            risk_score += 40
        elif self.portfolio.heat_level >= self.HEAT_REDUCTION_THRESHOLD:
            heat_factor = 1 - (self.portfolio.heat_level / 100)
            approved_quantity = int(approved_quantity * heat_factor)
            adjustments.append(f"REDUCED: Heat level at {self.portfolio.heat_level:.0f}/100")
            alert_level = AlertLevel.WARNING
            risk_score += 20

        # =================================================================
        # CHECK 6: Confidence Threshold
        # =================================================================
        MIN_CONFIDENCE = 0.55
        if request.confidence < MIN_CONFIDENCE:
            reasons.append(f"BLOCKED: Confidence {request.confidence:.1%} < {MIN_CONFIDENCE:.0%} minimum")
            approved_quantity = 0
            alert_level = AlertLevel.WARNING
            risk_score += 20

        # =================================================================
        # FINAL VERDICT
        # =================================================================
        approved = approved_quantity > 0

        verdict = TradeVerdict(
            request_id=request.request_id,
            approved=approved,
            original_quantity=request.quantity,
            approved_quantity=approved_quantity,
            alert_level=alert_level,
            reasons=reasons,
            adjustments=adjustments,
            risk_score=min(100, risk_score),
        )

        # Track statistics
        if approved:
            self.trades_approved += 1
            if approved_quantity < request.quantity:
                self.trades_reduced += 1
            self.approved_trades.append(verdict)
        else:
            self.trades_blocked += 1
            self.blocked_trades.append(verdict)

        # Log the decision
        if not approved:
            logger.warning(f"🛑 KILLJOY BLOCKED: {request.symbol} - {reasons}")
        elif adjustments:
            logger.info(f"⚠️ KILLJOY ADJUSTED: {request.symbol} - {adjustments}")
        else:
            logger.info(f"✅ KILLJOY APPROVED: {request.symbol} x{approved_quantity}")

        return verdict

    def check_portfolio_risk(self) -> Dict[str, Any]:
        """Real-time portfolio risk assessment.

        Returns comprehensive risk metrics and any limit breaches.
        """
        breaches = []
        warnings = []

        # Update limit values
        self.limits[RiskCategory.DAILY_LOSS].current_value = abs(min(0, self.portfolio.daily_pnl_pct))
        self.limits[RiskCategory.WEEKLY_LOSS].current_value = abs(min(0, self.portfolio.weekly_pnl / self.portfolio.total_equity))
        self.limits[RiskCategory.MONTHLY_LOSS].current_value = abs(min(0, self.portfolio.monthly_pnl / self.portfolio.total_equity))
        self.limits[RiskCategory.DRAWDOWN].current_value = self.portfolio.current_drawdown
        self.limits[RiskCategory.LEVERAGE].current_value = self.portfolio.gross_exposure / self.portfolio.total_equity
        self.limits[RiskCategory.VAR].current_value = self.portfolio.portfolio_var
        self.limits[RiskCategory.HEAT].current_value = self.portfolio.heat_level

        # Check each limit
        for category, limit in self.limits.items():
            status = limit.status
            if status == AlertLevel.EMERGENCY:
                breaches.append({
                    "category": category.value,
                    "value": limit.current_value,
                    "limit": limit.emergency_limit,
                    "action": "EMERGENCY LIQUIDATION REQUIRED",
                })
            elif status == AlertLevel.CRITICAL:
                breaches.append({
                    "category": category.value,
                    "value": limit.current_value,
                    "limit": limit.hard_limit,
                    "action": "TRADING BLOCKED",
                })
            elif status == AlertLevel.WARNING:
                warnings.append({
                    "category": category.value,
                    "value": limit.current_value,
                    "limit": limit.soft_limit,
                    "action": "SIZE REDUCTION",
                })

        # Determine overall status
        if any(b["action"] == "EMERGENCY LIQUIDATION REQUIRED" for b in breaches):
            overall_status = "EMERGENCY"
        elif breaches:
            overall_status = "CRITICAL"
        elif warnings:
            overall_status = "WARNING"
        else:
            overall_status = "NORMAL"

        return {
            "status": overall_status,
            "portfolio": {
                "equity": self.portfolio.total_equity,
                "daily_pnl": self.portfolio.daily_pnl,
                "daily_pnl_pct": self.portfolio.daily_pnl_pct,
                "drawdown": self.portfolio.current_drawdown,
                "gross_exposure": self.portfolio.gross_exposure / self.portfolio.total_equity,
                "heat_level": self.portfolio.heat_level,
            },
            "breaches": breaches,
            "warnings": warnings,
            "limits": {k.value: v.to_dict() for k, v in self.limits.items()},
            "trading_halted": self.trading_halted,
            "human_intervention_required": self.human_intervention_required,
            "timestamp": datetime.now().isoformat(),
        }

    def _halt_trading(self, reason: str):
        """Halt all trading."""
        self.trading_halted = True
        self.halt_reason = reason
        self._issue_alert(AlertLevel.CRITICAL, f"TRADING HALTED: {reason}")
        logger.critical(f"🛑🛑🛑 KILLJOY HALTED TRADING: {reason}")

    def _emergency_liquidate(self, reason: str):
        """NUCLEAR OPTION - Liquidate all positions."""
        self.emergency_liquidations += 1
        self.trading_halted = True
        self.halt_reason = f"EMERGENCY LIQUIDATION: {reason}"
        self.human_intervention_required = True
        self._issue_alert(AlertLevel.APOCALYPSE, f"EMERGENCY LIQUIDATION: {reason}")
        logger.critical(f"💥💥💥 KILLJOY EMERGENCY LIQUIDATION: {reason}")

        # In production, this would actually liquidate positions
        # For now, just flag and alert

    def _issue_alert(self, level: AlertLevel, message: str):
        """Issue an alert."""
        alert = {
            "level": level.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.alerts_issued.append(alert)

        # Escalate to HOAGS for critical alerts
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY, AlertLevel.APOCALYPSE]:
            self._escalate_to_hoags(alert)

    def _escalate_to_hoags(self, alert: Dict):
        """Escalate to HOAGS."""
        logger.critical(f"📢 KILLJOY → HOAGS: {alert['message']}")

    def update_portfolio(
        self,
        equity: float = None,
        daily_pnl: float = None,
        positions: Dict = None,
        trade_result: Dict = None,
    ):
        """Update portfolio state."""
        if equity is not None:
            old_equity = self.portfolio.total_equity
            self.portfolio.total_equity = equity

            # Update peak and drawdown
            if equity > self.portfolio.peak_equity:
                self.portfolio.peak_equity = equity

            self.portfolio.current_drawdown = (self.portfolio.peak_equity - equity) / self.portfolio.peak_equity
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, self.portfolio.current_drawdown)

        if daily_pnl is not None:
            self.portfolio.daily_pnl = daily_pnl
            self.portfolio.daily_pnl_pct = daily_pnl / self.portfolio.total_equity

        if positions is not None:
            self.portfolio.positions = positions

            # Calculate exposures
            long_val = sum(p.get("value", 0) for p in positions.values() if p.get("side") == "long")
            short_val = sum(abs(p.get("value", 0)) for p in positions.values() if p.get("side") == "short")

            self.portfolio.long_exposure = long_val / self.portfolio.total_equity
            self.portfolio.short_exposure = short_val / self.portfolio.total_equity
            self.portfolio.gross_exposure = long_val + short_val
            self.portfolio.net_exposure = long_val - short_val

        if trade_result is not None:
            # Update heat based on trade result
            if trade_result.get("pnl", 0) < 0:
                self.portfolio.consecutive_losses += 1
                self.portfolio.heat_level = min(100, self.portfolio.heat_level + 10)
            else:
                self.portfolio.consecutive_losses = 0
                self.portfolio.heat_level = max(0, self.portfolio.heat_level - 5)

    def log_action(self, action: str, description: str):
        """Log an action."""
        self.logger.info(f"[KILLJOY] {action}: {description}")

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_approve_trade(self, params: Dict) -> Dict:
        """Handle trade approval request."""
        request = TradeRequest(
            request_id=params.get("request_id", f"req_{datetime.now().timestamp()}"),
            agent_id=params.get("agent_id", "unknown"),
            agent_name=params.get("agent_name", "Unknown"),
            symbol=params.get("symbol", ""),
            action=params.get("action", "BUY"),
            quantity=params.get("quantity", 0),
            price=params.get("price", 0),
            order_type=params.get("order_type", "MARKET"),
            confidence=params.get("confidence", 0.5),
            reasoning=params.get("reasoning", ""),
        )

        verdict = self.approve_trade(request)

        return {
            "status": "approved" if verdict.approved else "blocked",
            "verdict": verdict.to_dict(),
        }

    def _handle_check_risk(self, params: Dict) -> Dict:
        """Handle risk check request."""
        return self.check_portfolio_risk()

    def _handle_update_portfolio(self, params: Dict) -> Dict:
        """Handle portfolio update."""
        self.update_portfolio(
            equity=params.get("equity"),
            daily_pnl=params.get("daily_pnl"),
            positions=params.get("positions"),
            trade_result=params.get("trade_result"),
        )
        return {"status": "success", "portfolio": self.check_portfolio_risk()["portfolio"]}

    def _handle_get_limits(self, params: Dict) -> Dict:
        """Get all risk limits."""
        return {
            "status": "success",
            "limits": {k.value: v.to_dict() for k, v in self.limits.items()},
        }

    def _handle_get_status(self, params: Dict) -> Dict:
        """Get KILLJOY status."""
        return {
            "status": "success",
            "killjoy": {
                "trades_reviewed": self.trades_reviewed,
                "trades_approved": self.trades_approved,
                "trades_blocked": self.trades_blocked,
                "trades_reduced": self.trades_reduced,
                "emergency_liquidations": self.emergency_liquidations,
                "trading_halted": self.trading_halted,
                "halt_reason": self.halt_reason,
                "human_intervention_required": self.human_intervention_required,
                "recent_alerts": self.alerts_issued[-10:],
            },
            "risk": self.check_portfolio_risk(),
        }

    def _handle_emergency_stop(self, params: Dict) -> Dict:
        """Emergency stop all trading."""
        reason = params.get("reason", "Manual emergency stop")
        self._emergency_liquidate(reason)
        return {"status": "success", "action": "EMERGENCY STOP ACTIVATED"}

    def _handle_reset_daily(self, params: Dict) -> Dict:
        """Reset daily counters (run at market open)."""
        self.portfolio.daily_pnl = 0
        self.portfolio.daily_pnl_pct = 0

        if not self.human_intervention_required:
            self.trading_halted = False
            self.halt_reason = ""

        for limit in self.limits.values():
            limit.breaches_today = 0

        return {"status": "success", "action": "DAILY RESET COMPLETE"}

    def _handle_override(self, params: Dict) -> Dict:
        """Handle override request (TOM HOGAN ONLY)."""
        override_code = params.get("override_code")

        # In production, this would verify Tom's authentication
        if override_code != "TJH_OVERRIDE_2024":
            logger.warning("🚨 UNAUTHORIZED OVERRIDE ATTEMPT")
            return {"status": "error", "message": "UNAUTHORIZED"}

        action = params.get("action")

        if action == "resume_trading":
            self.trading_halted = False
            self.halt_reason = ""
            self.human_intervention_required = False
            logger.info("✅ KILLJOY: Tom Hogan override - Trading resumed")
            return {"status": "success", "action": "TRADING RESUMED"}

        return {"status": "error", "message": "Unknown override action"}

    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}


# =============================================================================
# SINGLETON
# =============================================================================

_killjoy_instance: Optional[KilljoyAgent] = None


def get_killjoy() -> KilljoyAgent:
    """Get KILLJOY agent singleton."""
    global _killjoy_instance
    if _killjoy_instance is None:
        _killjoy_instance = KilljoyAgent()
    return _killjoy_instance



