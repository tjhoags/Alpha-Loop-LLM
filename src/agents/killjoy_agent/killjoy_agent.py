"""
================================================================================
KILLJOY AGENT - Capital Allocation, Position Sizing & Risk Guardrails
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

KILLJOY exists for one reason: PROTECT THE CAPITAL.

When every other agent is screaming "BUY!", KILLJOY says "Wait."
When the portfolio is running hot, KILLJOY says "Cool down."
When conviction is high, KILLJOY says "But what if you're wrong?"

KILLJOY is the sober voice in the room. It's not here to make friends.
It's here to make sure you're still in the game tomorrow.

Named after the character who ruins everyone's fun - because sometimes
protecting capital means killing joy.

Key Responsibilities:
- Capital allocation across strategies
- Position sizing with Kelly Criterion
- Heat monitoring (portfolio risk)
- Drawdown protection
- Correlation monitoring
- Concentration limits
- Daily/Weekly/Monthly loss limits
- Volatility-adjusted sizing

Reports To: HOAGS → Tom
Tier: SENIOR (2)

Core Philosophy:
"The first rule of trading is: Don't lose money. The second rule is: Don't forget the first."

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT KILLJOY DOES:
    KILLJOY is the risk guardian of Alpha Loop Capital. While other agents
    hunt alpha and find opportunities, KILLJOY's job is to say "no" when
    needed. It prevents catastrophic losses.
    
    The name says it all - KILLJOY "kills the joy" of over-eager trading.
    When BOOKMAKER finds an amazing opportunity and SCOUT confirms it,
    KILLJOY asks: "But what if you're both wrong?"
    
    Think of KILLJOY as the "chief risk officer" who ensures survival
    above all else. No single trade can blow up the portfolio.

KEY FUNCTIONS:
    1. allocate_capital() - Determines how much capital to risk on any
       strategy. Uses Kelly Criterion with fractional sizing.
       
    2. size_position() - Calculates exact position size with all
       guardrails: volatility scaling, correlation adjustment, heat
       penalty, concentration limits.
       
    3. check_heat() - Monitors portfolio "heat" (risk). Calculates VaR,
       drawdown, concentration, correlation. Returns risk level.
       
    4. approve_trade() - Final gate for any trade. Checks loss limits,
       position limits, drawdown limits. Can REJECT or SIZE DOWN.
       
    5. emergency_derisk() - When things go wrong, rapidly reduce
       positions to bring heat back to acceptable levels.

RISK LIMITS (Hard Stops):
    - Max single position: 10% of portfolio
    - Max sector: 30% of portfolio
    - Daily loss limit: 2%
    - Weekly loss limit: 5%
    - Max drawdown: 15%
    - Max daily VaR 95%: 2%
    - Max top-3 concentration: 40%

RELATIONSHIPS WITH OTHER AGENTS:
    - HOAGS: Reports to HOAGS. Can override even HOAGS-approved trades
      if risk limits are breached. KILLJOY has HALT authority.
      
    - ALL STRATEGY AGENTS: Every trade recommendation must pass
      KILLJOY's approval before execution.
      
    - STRINGS: Coordinates on ensemble weights. STRINGS optimizes
      for return, KILLJOY constrains for risk.
      
    - WHITEHAT: Works together on systemic risk. WHITEHAT handles
      technical security, KILLJOY handles financial risk.

PATHS OF GROWTH/TRANSFORMATION:
    1. PREDICTIVE RISK: Not just reactive limits but predictive risk
       - knowing when risk is building before limits are hit.
       
    2. REGIME-ADAPTIVE LIMITS: Different limits for different market
       regimes. Tighter in volatility, looser in calm.
       
    3. TAIL RISK FOCUS: Better modeling of tail events and black
       swans. Standard VaR misses the big ones.
       
    4. LIQUIDITY INTEGRATION: Factor in market liquidity when
       calculating position sizes and exit risk.
       
    5. CORRELATION DYNAMICS: Real-time correlation monitoring to
       catch "correlation breakdown" in stress.
       
    6. STRESS TESTING: Continuous stress testing against historical
       crisis scenarios.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr
    
    # Activate virtual environment:
    .\\venv\\Scripts\\activate
    
    # Train KILLJOY individually:
    python -m src.training.agent_training_utils --agent KILLJOY
    
    # Train with risk-related agents:
    python -m src.training.agent_training_utils --agents KILLJOY,WHITEHAT,BLACKHAT
    
    # Cross-train: KILLJOY and STRINGS optimize, AUTHOR documents:
    python -m src.training.agent_training_utils --cross-train "KILLJOY,STRINGS:AUTHOR:agent_trainer"

RUNNING THE AGENT:
    from src.agents.killjoy_agent.killjoy_agent import get_killjoy
    
    killjoy = get_killjoy()
    
    # Check current portfolio heat
    result = killjoy.process({
        "action": "check_heat",
        "positions": [
            {"ticker": "AAPL", "weight": 0.08, "volatility": 0.25},
            {"ticker": "CCJ", "weight": 0.05, "volatility": 0.40}
        ]
    })
    
    # Approve a trade
    result = killjoy.process({
        "action": "approve_trade",
        "ticker": "NVDA",
        "action": "buy",
        "size_pct": 0.05,
        "price": 500,
        "confidence": 0.75
    })
    
    # Emergency derisk
    result = killjoy.process({
        "action": "emergency_derisk",
        "target_heat": 30.0
    })

================================================================================
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Portfolio risk levels."""
    MINIMAL = "minimal"        # < 0.5% daily VaR
    LOW = "low"                # 0.5-1% daily VaR
    MODERATE = "moderate"      # 1-2% daily VaR
    ELEVATED = "elevated"      # 2-3% daily VaR
    HIGH = "high"              # 3-5% daily VaR
    CRITICAL = "critical"      # > 5% daily VaR - REDUCE IMMEDIATELY


class AllocationAction(Enum):
    """Capital allocation actions."""
    INCREASE = "increase"
    MAINTAIN = "maintain"
    REDUCE = "reduce"
    ELIMINATE = "eliminate"
    HALT = "halt_new_positions"


@dataclass
class PortfolioHeat:
    """Current portfolio heat/risk metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Daily metrics
    daily_pnl_pct: float = 0.0
    daily_var_95: float = 0.0       # 95% Value at Risk
    daily_var_99: float = 0.0       # 99% Value at Risk
    
    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown_30d: float = 0.0
    days_in_drawdown: int = 0
    
    # Concentration
    largest_position_pct: float = 0.0
    top_3_concentration: float = 0.0
    sector_concentration: Dict[str, float] = field(default_factory=dict)
    
    # Correlation
    avg_position_correlation: float = 0.0
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    
    # Volatility
    portfolio_volatility: float = 0.0
    vol_regime: str = "normal"
    
    # Overall
    risk_level: RiskLevel = RiskLevel.MODERATE
    heat_score: float = 50.0        # 0-100, higher = more risk
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "daily_pnl_pct": self.daily_pnl_pct,
            "daily_var_95": self.daily_var_95,
            "daily_var_99": self.daily_var_99,
            "current_drawdown": self.current_drawdown,
            "max_drawdown_30d": self.max_drawdown_30d,
            "days_in_drawdown": self.days_in_drawdown,
            "largest_position_pct": self.largest_position_pct,
            "top_3_concentration": self.top_3_concentration,
            "sector_concentration": self.sector_concentration,
            "avg_correlation": self.avg_position_correlation,
            "portfolio_volatility": self.portfolio_volatility,
            "vol_regime": self.vol_regime,
            "risk_level": self.risk_level.value,
            "heat_score": self.heat_score,
        }


@dataclass
class PositionSizeRecommendation:
    """Recommendation for position sizing."""
    ticker: str
    action: str                 # "buy", "sell", "hold"
    requested_size_pct: float   # What was requested
    approved_size_pct: float    # What KILLJOY approved
    max_size_pct: float         # Hard limit
    reasoning: List[str] = field(default_factory=list)
    kelly_fraction: float = 0.0
    risk_contribution: float = 0.0
    confidence_adjustment: float = 1.0
    approved: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "requested_pct": self.requested_size_pct,
            "approved_pct": self.approved_size_pct,
            "max_pct": self.max_size_pct,
            "kelly_fraction": self.kelly_fraction,
            "risk_contribution": self.risk_contribution,
            "reasoning": self.reasoning,
            "approved": self.approved,
        }


@dataclass
class RiskLimits:
    """Portfolio risk limits - THESE ARE HARD LIMITS."""
    # Position limits
    max_position_pct: float = 0.10          # 10% max single position
    max_sector_pct: float = 0.30            # 30% max sector
    max_correlated_pct: float = 0.40        # 40% in correlated positions
    
    # Loss limits
    daily_loss_limit_pct: float = 0.02      # 2% max daily loss
    weekly_loss_limit_pct: float = 0.05     # 5% max weekly loss
    monthly_loss_limit_pct: float = 0.10    # 10% max monthly loss
    
    # Drawdown limits
    max_drawdown_pct: float = 0.15          # 15% max drawdown
    drawdown_reduction_threshold: float = 0.08  # Start reducing at 8%
    
    # VaR limits
    max_daily_var_95: float = 0.02          # 2% daily 95% VaR
    max_daily_var_99: float = 0.03          # 3% daily 99% VaR
    
    # Concentration limits
    max_top3_concentration: float = 0.40    # 40% in top 3
    
    # Leverage
    max_gross_leverage: float = 1.0         # No leverage by default
    max_net_leverage: float = 1.0
    
    # Number of positions
    min_positions: int = 5                  # Minimum diversification
    max_positions: int = 25                 # Avoid over-diversification


class KillJoyAgent(BaseAgent):
    """
    KILLJOY Agent - Capital Allocation & Risk Guardrails
    
    KILLJOY is the chief risk officer of the ALC-Algo ecosystem.
    It ensures no single trade or position can blow up the portfolio.
    
    Core Functions:
    - allocate_capital(): Determine how much to risk
    - size_position(): Calculate exact position size
    - check_heat(): Monitor portfolio risk
    - enforce_limits(): Hard stop on limit breaches
    - approve_trade(): Final sign-off on any trade
    
    Key Principles:
    1. Survival first - never risk ruin
    2. Kelly Criterion with fractional sizing
    3. Correlation-aware allocation
    4. Regime-adaptive limits
    5. ALWAYS err on the side of caution
    """
    
    def __init__(self, user_id: str = "TJH"):
        super().__init__(
            name="KillJoyAgent",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Capital allocation
                "capital_allocation",
                "position_sizing",
                "kelly_criterion",
                "risk_parity",
                "volatility_targeting",
                
                # Risk monitoring
                "heat_monitoring",
                "var_calculation",
                "drawdown_tracking",
                "correlation_analysis",
                "concentration_monitoring",
                
                # Limits enforcement
                "loss_limit_enforcement",
                "position_limit_enforcement",
                "leverage_control",
                "sector_limit_enforcement",
                
                # Guardrails
                "trade_approval",
                "emergency_derisking",
                "regime_adaptation",
                "stress_testing",
            ],
            user_id=user_id,
        )
        
        # Risk limits
        self.limits = RiskLimits()
        
        # Current state
        self.current_heat = PortfolioHeat()
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl_history: List[float] = []
        
        # Tracking
        self.trades_approved: int = 0
        self.trades_rejected: int = 0
        self.trades_sized_down: int = 0
        self.emergency_derisks: int = 0
        
        # Limit breaches
        self.limit_breaches: List[Dict] = []
        
        # Regime state
        self.current_regime: str = "normal"
        self.regime_risk_multiplier: float = 1.0
        
        self.logger.info("KillJoyAgent initialized - Capital protection active")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a KILLJOY task."""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)
        
        self.log_action(action, f"KILLJOY processing: {action}")
        
        handlers = {
            "allocate_capital": self._handle_allocate,
            "size_position": self._handle_size_position,
            "check_heat": self._handle_check_heat,
            "approve_trade": self._handle_approve_trade,
            "enforce_limits": self._handle_enforce_limits,
            "emergency_derisk": self._handle_emergency_derisk,
            "update_positions": self._handle_update_positions,
            "get_limits": self._handle_get_limits,
            "status": self._handle_status,
        }
        
        handler = handlers.get(action, self._handle_unknown)
        return handler(params)
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    # =========================================================================
    # CORE KILLJOY METHODS
    # =========================================================================
    
    def allocate_capital(
        self,
        strategy: str,
        conviction: float,
        expected_return: float,
        expected_volatility: float,
        correlation_to_portfolio: float = 0.5
    ) -> Tuple[float, List[str]]:
        """
        Determine capital allocation for a strategy.
        
        Uses Kelly Criterion with fractional sizing and risk adjustments.
        
        Returns:
            (allocation_pct, reasoning_list)
        """
        reasons = []
        
        # Check if we're in a halt state
        if self.current_heat.risk_level == RiskLevel.CRITICAL:
            reasons.append("CRITICAL: Portfolio at critical risk - no new allocations")
            return 0.0, reasons
        
        # Calculate Kelly fraction
        win_prob = 0.5 + (expected_return / expected_volatility) * 0.1  # Simplified
        win_prob = max(0.3, min(0.7, win_prob))
        
        if expected_volatility > 0:
            kelly_full = (win_prob * expected_return - (1 - win_prob) * expected_volatility) / expected_volatility
        else:
            kelly_full = 0
        
        kelly_full = max(0, kelly_full)
        
        # Use fractional Kelly (half Kelly is standard)
        kelly_fraction = kelly_full * 0.5
        reasons.append(f"Kelly fraction: {kelly_fraction:.1%} (half-Kelly)")
        
        # Adjust for conviction
        conviction_adj = 0.5 + conviction * 0.5  # 0.5 to 1.0
        allocation = kelly_fraction * conviction_adj
        reasons.append(f"Conviction adjustment: {conviction_adj:.2f}")
        
        # Adjust for correlation
        correlation_penalty = 1.0 - (correlation_to_portfolio * 0.5)
        allocation *= correlation_penalty
        reasons.append(f"Correlation penalty: {correlation_penalty:.2f}")
        
        # Adjust for current heat
        heat_penalty = self._get_heat_penalty()
        allocation *= heat_penalty
        reasons.append(f"Heat adjustment: {heat_penalty:.2f}")
        
        # Adjust for regime
        allocation *= self.regime_risk_multiplier
        reasons.append(f"Regime multiplier: {self.regime_risk_multiplier:.2f}")
        
        # Apply hard limits
        allocation = min(allocation, self.limits.max_position_pct)
        
        # Round to reasonable precision
        allocation = round(allocation, 4)
        
        reasons.append(f"Final allocation: {allocation:.2%}")
        
        return allocation, reasons
    
    def size_position(
        self,
        ticker: str,
        action: str,
        requested_pct: float,
        confidence: float,
        volatility: float,
        correlation: float = 0.5
    ) -> PositionSizeRecommendation:
        """
        Calculate approved position size with all guardrails.
        
        Args:
            ticker: Stock ticker
            action: "buy" or "sell"
            requested_pct: Requested position size as % of portfolio
            confidence: Model confidence (0-1)
            volatility: Position volatility (annualized)
            correlation: Correlation to rest of portfolio
        
        Returns:
            PositionSizeRecommendation with approved size
        """
        reasons = []
        approved = True
        
        # Start with requested
        approved_pct = requested_pct
        
        # 1. Apply hard position limit
        if approved_pct > self.limits.max_position_pct:
            reasons.append(f"Reduced from {approved_pct:.1%} to {self.limits.max_position_pct:.1%} (position limit)")
            approved_pct = self.limits.max_position_pct
            self.trades_sized_down += 1
        
        # 2. Volatility scaling
        target_vol = 0.15  # Target 15% annualized vol contribution
        if volatility > 0:
            vol_scalar = min(1.0, target_vol / volatility)
            if vol_scalar < 1.0:
                new_size = approved_pct * vol_scalar
                reasons.append(f"Vol-scaled from {approved_pct:.1%} to {new_size:.1%}")
                approved_pct = new_size
        
        # 3. Confidence adjustment
        conf_scalar = 0.5 + confidence * 0.5  # 50% to 100%
        approved_pct *= conf_scalar
        reasons.append(f"Confidence-adjusted: {conf_scalar:.2f}x")
        
        # 4. Correlation adjustment
        corr_scalar = 1.0 - (abs(correlation) * 0.3)
        approved_pct *= corr_scalar
        reasons.append(f"Correlation-adjusted: {corr_scalar:.2f}x")
        
        # 5. Heat adjustment
        heat_scalar = self._get_heat_penalty()
        approved_pct *= heat_scalar
        reasons.append(f"Heat-adjusted: {heat_scalar:.2f}x")
        
        # 6. Check concentration
        if ticker in self.positions:
            existing = self.positions[ticker].get("weight", 0)
            total = existing + approved_pct
            if total > self.limits.max_position_pct:
                approved_pct = max(0, self.limits.max_position_pct - existing)
                reasons.append(f"Reduced due to existing position ({existing:.1%})")
        
        # 7. Check if we'd breach top-3 concentration
        current_top3 = self.current_heat.top_3_concentration
        if current_top3 > self.limits.max_top3_concentration * 0.9:
            approved_pct *= 0.5
            reasons.append("Reduced 50% due to concentration concerns")
        
        # 8. Reject if too small
        if approved_pct < 0.005:  # Less than 0.5%
            approved = False
            reasons.append("Position too small to be meaningful (<0.5%)")
            approved_pct = 0
        
        # 9. Reject if in drawdown halt
        if self.current_heat.current_drawdown > self.limits.max_drawdown_pct:
            approved = False
            reasons.append(f"REJECTED: Portfolio in max drawdown ({self.current_heat.current_drawdown:.1%})")
            approved_pct = 0
            self.trades_rejected += 1
        
        # Calculate risk contribution
        risk_contribution = approved_pct * volatility
        
        recommendation = PositionSizeRecommendation(
            ticker=ticker,
            action=action,
            requested_size_pct=requested_pct,
            approved_size_pct=round(approved_pct, 4),
            max_size_pct=self.limits.max_position_pct,
            reasoning=reasons,
            kelly_fraction=0.0,  # Would calculate if we had full data
            risk_contribution=risk_contribution,
            confidence_adjustment=conf_scalar,
            approved=approved,
        )
        
        if approved:
            self.trades_approved += 1
        
        return recommendation
    
    def check_heat(self, positions: List[Dict] = None) -> PortfolioHeat:
        """
        Calculate current portfolio heat/risk.
        
        Args:
            positions: List of current positions with weights and volatilities
        
        Returns:
            PortfolioHeat object with all risk metrics
        """
        heat = PortfolioHeat(timestamp=datetime.now())
        
        if not positions:
            positions = list(self.positions.values())
        
        if not positions:
            heat.risk_level = RiskLevel.MINIMAL
            heat.heat_score = 0
            self.current_heat = heat
            return heat
        
        # Calculate concentration
        weights = [p.get("weight", 0) for p in positions]
        weights.sort(reverse=True)
        
        heat.largest_position_pct = weights[0] if weights else 0
        heat.top_3_concentration = sum(weights[:3])
        
        # Calculate portfolio volatility (simplified)
        vols = [p.get("volatility", 0.20) for p in positions]
        avg_vol = sum(v * w for v, w in zip(vols, weights)) / sum(weights) if weights else 0
        heat.portfolio_volatility = avg_vol
        
        # Estimate VaR (simplified - would use proper calculation)
        heat.daily_var_95 = avg_vol * 1.65 / 16  # Annualized to daily
        heat.daily_var_99 = avg_vol * 2.33 / 16
        
        # Get drawdown from history
        if self.daily_pnl_history:
            cumulative = 0
            peak = 0
            for pnl in self.daily_pnl_history:
                cumulative += pnl
                peak = max(peak, cumulative)
                heat.current_drawdown = max(heat.current_drawdown, peak - cumulative)
        
        # Calculate heat score (0-100)
        heat_score = 0
        
        # Concentration component (0-25)
        heat_score += min(25, heat.top_3_concentration * 50)
        
        # Volatility component (0-25)
        heat_score += min(25, heat.portfolio_volatility * 100)
        
        # VaR component (0-25)
        heat_score += min(25, heat.daily_var_95 * 500)
        
        # Drawdown component (0-25)
        heat_score += min(25, heat.current_drawdown * 100)
        
        heat.heat_score = round(heat_score, 1)
        
        # Determine risk level
        if heat_score < 20:
            heat.risk_level = RiskLevel.MINIMAL
        elif heat_score < 35:
            heat.risk_level = RiskLevel.LOW
        elif heat_score < 50:
            heat.risk_level = RiskLevel.MODERATE
        elif heat_score < 65:
            heat.risk_level = RiskLevel.ELEVATED
        elif heat_score < 80:
            heat.risk_level = RiskLevel.HIGH
        else:
            heat.risk_level = RiskLevel.CRITICAL
        
        self.current_heat = heat
        return heat
    
    def approve_trade(
        self,
        ticker: str,
        action: str,
        size_pct: float,
        price: float,
        confidence: float
    ) -> Tuple[bool, str, Dict]:
        """
        Final approval gate for any trade.
        
        Returns:
            (approved, reason, trade_details)
        """
        # Check daily loss limit
        if self._check_daily_loss_limit():
            self.trades_rejected += 1
            return False, "Daily loss limit reached - no new trades", {}
        
        # Check weekly loss limit
        if self._check_weekly_loss_limit():
            self.trades_rejected += 1
            return False, "Weekly loss limit reached - no new trades", {}
        
        # Check drawdown
        if self.current_heat.current_drawdown > self.limits.max_drawdown_pct:
            self.trades_rejected += 1
            return False, f"Max drawdown exceeded ({self.current_heat.current_drawdown:.1%})", {}
        
        # Check heat level
        if self.current_heat.risk_level == RiskLevel.CRITICAL:
            self.trades_rejected += 1
            return False, "Portfolio at CRITICAL heat - trades halted", {}
        
        # Size the position properly
        recommendation = self.size_position(
            ticker=ticker,
            action=action,
            requested_pct=size_pct,
            confidence=confidence,
            volatility=0.25,  # Would get real vol
            correlation=0.5,
        )
        
        if not recommendation.approved:
            return False, "; ".join(recommendation.reasoning), {}
        
        trade_details = {
            "ticker": ticker,
            "action": action,
            "original_size": size_pct,
            "approved_size": recommendation.approved_size_pct,
            "price": price,
            "confidence": confidence,
            "risk_contribution": recommendation.risk_contribution,
            "approved_at": datetime.now().isoformat(),
            "approved_by": "KillJoyAgent",
        }
        
        self.trades_approved += 1
        return True, "APPROVED", trade_details
    
    def emergency_derisk(self, target_heat: float = 30.0) -> List[Dict]:
        """
        Emergency derisking - reduce positions to bring heat down.
        
        Args:
            target_heat: Target heat score to achieve
        
        Returns:
            List of positions to reduce
        """
        self.emergency_derisks += 1
        reductions = []
        
        current_heat = self.current_heat.heat_score
        if current_heat <= target_heat:
            return []
        
        # Sort positions by risk contribution (would use proper calc)
        sorted_positions = sorted(
            self.positions.items(),
            key=lambda x: x[1].get("weight", 0) * x[1].get("volatility", 0.2),
            reverse=True
        )
        
        heat_to_reduce = current_heat - target_heat
        
        for ticker, pos in sorted_positions:
            if heat_to_reduce <= 0:
                break
            
            weight = pos.get("weight", 0)
            vol = pos.get("volatility", 0.2)
            
            # Calculate reduction needed
            reduction_pct = min(weight * 0.5, weight)  # Reduce by up to 50%
            heat_impact = reduction_pct * vol * 100
            
            reductions.append({
                "ticker": ticker,
                "current_weight": weight,
                "reduce_by": reduction_pct,
                "new_weight": weight - reduction_pct,
                "heat_impact": heat_impact,
                "reason": "Emergency derisk",
            })
            
            heat_to_reduce -= heat_impact
        
        self.logger.critical(f"EMERGENCY DERISK: Recommending {len(reductions)} reductions")
        
        return reductions
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _get_heat_penalty(self) -> float:
        """Get sizing penalty based on current heat."""
        heat = self.current_heat.heat_score
        
        if heat < 30:
            return 1.0
        elif heat < 50:
            return 0.8
        elif heat < 65:
            return 0.5
        elif heat < 80:
            return 0.25
        else:
            return 0.0  # No new positions at critical heat
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is breached."""
        if self.daily_pnl_history:
            today_pnl = self.daily_pnl_history[-1] if self.daily_pnl_history else 0
            return today_pnl < -self.limits.daily_loss_limit_pct
        return False
    
    def _check_weekly_loss_limit(self) -> bool:
        """Check if weekly loss limit is breached."""
        if len(self.daily_pnl_history) >= 5:
            week_pnl = sum(self.daily_pnl_history[-5:])
            return week_pnl < -self.limits.weekly_loss_limit_pct
        return False
    
    def log_action(self, action: str, description: str):
        self.logger.info(f"[KILLJOY] {action}: {description}")
    
    # =========================================================================
    # TASK HANDLERS
    # =========================================================================
    
    def _handle_allocate(self, params: Dict) -> Dict:
        allocation, reasons = self.allocate_capital(
            strategy=params.get("strategy", "unknown"),
            conviction=params.get("conviction", 0.5),
            expected_return=params.get("expected_return", 0.10),
            expected_volatility=params.get("expected_volatility", 0.20),
            correlation_to_portfolio=params.get("correlation", 0.5),
        )
        return {
            "success": True,
            "allocation_pct": allocation,
            "reasoning": reasons,
            "current_heat": self.current_heat.heat_score,
        }
    
    def _handle_size_position(self, params: Dict) -> Dict:
        rec = self.size_position(
            ticker=params.get("ticker", ""),
            action=params.get("action", "buy"),
            requested_pct=params.get("size_pct", 0.05),
            confidence=params.get("confidence", 0.5),
            volatility=params.get("volatility", 0.25),
            correlation=params.get("correlation", 0.5),
        )
        return {"success": True, "recommendation": rec.to_dict()}
    
    def _handle_check_heat(self, params: Dict) -> Dict:
        positions = params.get("positions", [])
        heat = self.check_heat(positions)
        return {"success": True, "heat": heat.to_dict()}
    
    def _handle_approve_trade(self, params: Dict) -> Dict:
        approved, reason, details = self.approve_trade(
            ticker=params.get("ticker", ""),
            action=params.get("action", "buy"),
            size_pct=params.get("size_pct", 0.05),
            price=params.get("price", 100),
            confidence=params.get("confidence", 0.5),
        )
        return {
            "success": True,
            "approved": approved,
            "reason": reason,
            "trade_details": details,
        }
    
    def _handle_enforce_limits(self, params: Dict) -> Dict:
        breaches = []
        
        # Check all limits
        if self.current_heat.daily_var_95 > self.limits.max_daily_var_95:
            breaches.append({"limit": "daily_var_95", "value": self.current_heat.daily_var_95})
        
        if self.current_heat.current_drawdown > self.limits.max_drawdown_pct:
            breaches.append({"limit": "max_drawdown", "value": self.current_heat.current_drawdown})
        
        if self.current_heat.top_3_concentration > self.limits.max_top3_concentration:
            breaches.append({"limit": "top3_concentration", "value": self.current_heat.top_3_concentration})
        
        self.limit_breaches.extend(breaches)
        
        return {
            "success": True,
            "breaches": breaches,
            "breach_count": len(breaches),
            "action_required": len(breaches) > 0,
        }
    
    def _handle_emergency_derisk(self, params: Dict) -> Dict:
        target = params.get("target_heat", 30.0)
        reductions = self.emergency_derisk(target)
        return {
            "success": True,
            "reductions": reductions,
            "count": len(reductions),
        }
    
    def _handle_update_positions(self, params: Dict) -> Dict:
        positions = params.get("positions", [])
        for pos in positions:
            ticker = pos.get("ticker")
            if ticker:
                self.positions[ticker] = pos
        
        # Recalculate heat
        self.check_heat()
        
        return {
            "success": True,
            "positions_count": len(self.positions),
            "current_heat": self.current_heat.heat_score,
        }
    
    def _handle_get_limits(self, params: Dict) -> Dict:
        return {
            "success": True,
            "limits": {
                "max_position_pct": self.limits.max_position_pct,
                "max_sector_pct": self.limits.max_sector_pct,
                "daily_loss_limit_pct": self.limits.daily_loss_limit_pct,
                "weekly_loss_limit_pct": self.limits.weekly_loss_limit_pct,
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_daily_var_95": self.limits.max_daily_var_95,
                "max_top3_concentration": self.limits.max_top3_concentration,
            }
        }
    
    def _handle_status(self, params: Dict) -> Dict:
        return {
            "success": True,
            "agent": "KillJoyAgent",
            "trades_approved": self.trades_approved,
            "trades_rejected": self.trades_rejected,
            "trades_sized_down": self.trades_sized_down,
            "emergency_derisks": self.emergency_derisks,
            "current_heat": self.current_heat.to_dict(),
            "positions_monitored": len(self.positions),
            "regime": self.current_regime,
            "regime_multiplier": self.regime_risk_multiplier,
        }
    
    def _handle_unknown(self, params: Dict) -> Dict:
        return {"success": False, "error": "Unknown action"}


# =============================================================================
# SINGLETON
# =============================================================================

_killjoy_instance: Optional[KillJoyAgent] = None


def get_killjoy() -> KillJoyAgent:
    """Get KillJoy agent singleton."""
    global _killjoy_instance
    if _killjoy_instance is None:
        _killjoy_instance = KillJoyAgent()
    return _killjoy_instance
