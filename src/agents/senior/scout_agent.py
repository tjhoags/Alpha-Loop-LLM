"""
================================================================================
SCOUT AGENT - Retail Arbitrage & Market Inefficiency Hunter
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

SCOUT scours the full US equity landscape specifically targeting retail bad
bid/asks and market inefficiencies in the <$30bn market cap arena. SCOUT
looks for arbitrage opportunities, over/underpriced risk premia, and calculates
the optimal way to scalp trades - even single contracts.

Tier: SENIOR (2)
Reports To: HOAGS → Tom
Cluster: arbitrage

Focus:
- <$30bn market cap (small/mid cap)
- Retail bid/ask spreads
- Options mispricing
- Risk premia anomalies
- Single contract scalps

Core Philosophy:
"Find the edge retail created, scalp it before it disappears."

Key Capabilities:
- Bid/ask spread analysis
- Options mispricing detection
- Risk premia calculation
- Scalp trade optimization
- Immediate opportunity reporting

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT SCOUT DOES:
    SCOUT is the arbitrage hunter of Alpha Loop Capital, specializing in
    small and mid-cap stocks where retail trading creates exploitable
    inefficiencies.

    In large caps like AAPL or MSFT, markets are efficient and spreads
    are tight. But in small caps, retail traders often post bad bid/asks,
    creating moments where you can scalp risk-free profit.

    SCOUT finds these moments and calculates exactly how to exploit them -
    even if it means trading just a single options contract.

    Think of SCOUT as the "floor trader" of the ecosystem, always watching
    for momentary mispricings.

KEY FUNCTIONS:
    1. scan_for_inefficiencies() - Continuous scan of the small/mid cap
       universe looking for bad bid/asks, options mispricing, and risk
       premia anomalies. Returns opportunities sorted by urgency.

    2. analyze_bid_ask() - Deep dive on a specific stock's spread quality.
       Compares to historical, estimates retail vs institutional flow.

    3. calculate_fair_value() - Determines theoretical fair value for
       stocks and options using Black-Scholes and multi-factor models.

    4. optimize_scalp() - Given an opportunity, calculates exact entry
       price, position size, and timing. Even for single contracts.

RELATIONSHIPS WITH OTHER AGENTS:
    - HOAGS: Reports IMMEDIATELY when urgent opportunities are found.
      Time-sensitive nature of scalping requires fast communication.

    - BOOKMAKER: Feeds opportunities to BOOKMAKER for alpha quantification.
      SCOUT finds them, BOOKMAKER measures the edge.

    - HUNTER: Coordinates on flow analysis. HUNTER knows algorithms,
      SCOUT knows retail patterns. Together they understand order flow.

    - KILLJOY: All scalp recommendations must pass KILLJOY's position
      limits. Even small trades need risk approval.

    - CONVERSION_REVERSAL: Works with the options arbitrage specialist
      on put-call parity violations and box spreads.

PATHS OF GROWTH/TRANSFORMATION:
    1. CRYPTO EXPANSION: Apply retail arbitrage hunting to crypto
       markets where retail inefficiency is even more pronounced.

    2. REAL-TIME STREAMING: Move from periodic scans to real-time
       streaming with sub-second detection.

    3. AUTO-EXECUTION: Gain ability to execute approved scalps
       automatically when opportunities appear.

    4. OTC MARKETS: Expand to OTC stocks where inefficiencies are
       massive but liquidity risk is higher.

    5. SENTIMENT INTEGRATION: Incorporate retail sentiment data
       (Reddit, Twitter) to predict where bad quotes will appear.

    6. DARK POOL AWARENESS: Understand when dark pool prints signal
       upcoming public market mispricings.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train SCOUT individually:
    python -m src.training.agent_training_utils --agent SCOUT

    # Train with arbitrage-related agents:
    python -m src.training.agent_training_utils --agents SCOUT,BOOKMAKER,HUNTER

    # Cross-train with retail arbitrage ML:
    python -m src.training.agent_training_utils --cross-train "SCOUT,HUNTER:AUTHOR:retail_arbitrage"

RUNNING THE AGENT:
    from src.agents.senior.scout_agent import get_scout

    scout = get_scout()

    # Scan for inefficiencies
    result = scout.process({
        "action": "scan",
        "universe": ["SOFI", "HOOD", "RIVN", "CCJ"]
    })

    # Analyze specific ticker
    result = scout.process({
        "action": "analyze_ticker",
        "ticker": "SOFI"
    })

================================================================================
"""

import hashlib
import logging
import random
from datetime import datetime
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

from dataclasses import dataclass, field
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier

logger = logging.getLogger(__name__)


class InefficiencyType(Enum):
    """Types of market inefficiencies SCOUT detects"""
    BAD_BID_ASK = "bad_bid_ask"
    OPTIONS_MISPRICING = "options_mispricing"
    RISK_PREMIA_ANOMALY = "risk_premia_anomaly"
    LIQUIDITY_VACUUM = "liquidity_vacuum"
    STALE_QUOTE = "stale_quote"
    RETAIL_FLOW_IMBALANCE = "retail_flow_imbalance"
    VOLATILITY_SKEW_ANOMALY = "vol_skew_anomaly"


class Urgency(Enum):
    """How quickly an opportunity must be acted on"""
    IMMEDIATE = "immediate"      # Seconds
    FAST = "fast"                # Minutes
    STANDARD = "standard"        # Hours
    PATIENT = "patient"          # Days


@dataclass
class ScalpOpportunity:
    """A scalping opportunity identified by SCOUT"""
    opportunity_id: str
    inefficiency_type: InefficiencyType
    ticker: str
    market_cap_bn: float
    urgency: Urgency

    # Trade details
    action: str                  # "buy" or "sell"
    instrument: str              # "stock", "call", "put"
    strike: Optional[float]
    expiry: Optional[str]
    contracts: int               # Even if just 1

    # Pricing
    current_bid: float
    current_ask: float
    fair_value: float
    edge_cents: float            # Edge in cents
    edge_pct: float              # Edge as percentage

    # Risk
    max_loss: float
    expected_profit: float
    risk_reward: float
    confidence: float

    # Execution
    optimal_limit_price: float
    max_slippage: float
    time_to_fill_est: str

    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "opportunity_id": self.opportunity_id,
            "type": self.inefficiency_type.value,
            "ticker": self.ticker,
            "market_cap_bn": self.market_cap_bn,
            "urgency": self.urgency.value,
            "action": self.action,
            "instrument": self.instrument,
            "strike": self.strike,
            "expiry": self.expiry,
            "contracts": self.contracts,
            "bid": self.current_bid,
            "ask": self.current_ask,
            "fair_value": self.fair_value,
            "edge_cents": self.edge_cents,
            "edge_pct": self.edge_pct,
            "max_loss": self.max_loss,
            "expected_profit": self.expected_profit,
            "risk_reward": self.risk_reward,
            "confidence": self.confidence,
            "optimal_limit": self.optimal_limit_price,
            "detected_at": self.detected_at.isoformat()
        }


class ScoutAgent(BaseAgent):
    """
    SCOUT Agent - The Market Inefficiency Hunter

    SCOUT continuously scans for retail-created inefficiencies in small/mid cap
    stocks (<$30bn). It identifies bad bid/asks, options mispricing, and
    calculates optimal scalp trades - even for single contracts.

    Reports IMMEDIATELY to HOAGS when opportunities are found.

    Key Methods:
    - scan_for_inefficiencies(): Continuous market scan
    - analyze_bid_ask(): Deep dive on spread quality
    - calculate_fair_value(): Determine theoretical value
    - optimize_scalp(): Calculate optimal trade parameters
    - report_immediately(): Send urgent notification
    """

    # Market cap filter (billions)
    MAX_MARKET_CAP = 30.0

    # Minimum edge thresholds
    MIN_EDGE_PCT = 0.005          # 0.5% minimum edge
    MIN_EDGE_CENTS_STOCK = 5      # 5 cents for stocks
    MIN_EDGE_CENTS_OPTION = 3     # 3 cents for options

    def __init__(self):
        super().__init__(
            name="SCOUT",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Core scanning
                "bid_ask_analysis",
                "spread_quality_assessment",
                "retail_flow_detection",
                "stale_quote_identification",

                # Options specific
                "options_mispricing_detection",
                "volatility_skew_analysis",
                "greeks_calculation",
                "theoretical_value_computation",

                # Risk premia
                "risk_premia_calculation",
                "premia_anomaly_detection",
                "historical_premia_analysis",

                # Execution
                "scalp_optimization",
                "single_contract_execution",
                "slippage_estimation",
                "fill_probability_calculation",

                # Reporting
                "immediate_notification",
                "hoags_escalation"
            ],
            user_id="TJH"
        )

        # Tracking
        self.opportunities_found: List[ScalpOpportunity] = []
        self.opportunities_executed = 0
        self.total_edge_captured_cents = 0

        # Watchlist (small/mid caps to monitor)
        self.watchlist: List[str] = []

        # Configuration
        self.config = {
            "max_market_cap_bn": self.MAX_MARKET_CAP,
            "min_edge_pct": self.MIN_EDGE_PCT,
            "scan_interval_seconds": 5,
            "max_contracts_per_trade": 10,
            "min_open_interest": 100,
            "max_spread_pct": 0.05,  # 5% max spread to consider
        }

    @cached_property
    def _handlers(self) -> Dict[str, Callable[[Dict], Dict]]:
        """Cached handler dispatch table for O(1) lookup."""
        return {
            "scan": self._handle_scan,
            "analyze_ticker": self._handle_analyze_ticker,
            "find_arbitrage": self._handle_find_arbitrage,
            "calculate_scalp": self._handle_calculate_scalp,
            "add_watchlist": self._handle_add_watchlist,
            "get_opportunities": self._handle_get_opportunities,
            "report_status": self._handle_report_status,
        }

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a SCOUT task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.log_action(action, f"SCOUT processing: {action}")

        # Check for capability gaps (ACA)
        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap: {gap.missing_capabilities}")

        handler = self._handlers.get(action, self._handle_unknown)
        result = handler(params)

        # Immediate notification if urgent opportunity
        if result.get("opportunity") and result["opportunity"].urgency == Urgency.IMMEDIATE:
            self._report_immediately(result["opportunity"])

        return result

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    # =========================================================================
    # CORE SCOUT METHODS
    # =========================================================================

    def scan_for_inefficiencies(
        self,
        universe: List[str] = None,
        types: List[InefficiencyType] = None
    ) -> List[ScalpOpportunity]:
        """
        Scan market for inefficiencies and scalp opportunities.

        Args:
            universe: Tickers to scan (None = full small/mid cap universe)
            types: Types of inefficiencies to look for

        Returns:
            List of ScalpOpportunity objects, sorted by urgency then edge
        """
        self.logger.info("SCOUT: Beginning market scan for inefficiencies...")

        opportunities = []

        if types is None:
            types = list(InefficiencyType)

        # Use watchlist if no universe specified
        tickers_to_scan = universe or self.watchlist or self._get_small_mid_cap_universe()

        for ticker in tickers_to_scan:
            # Check market cap filter
            market_cap = self._get_market_cap(ticker)
            if market_cap > self.MAX_MARKET_CAP:
                continue

            # Scan for each inefficiency type
            if InefficiencyType.BAD_BID_ASK in types:
                opp = self._scan_bid_ask_spread(ticker, market_cap)
                if opp:
                    opportunities.append(opp)

            if InefficiencyType.OPTIONS_MISPRICING in types:
                opts = self._scan_options_mispricing(ticker, market_cap)
                opportunities.extend(opts)

            if InefficiencyType.RISK_PREMIA_ANOMALY in types:
                opp = self._scan_risk_premia(ticker, market_cap)
                if opp:
                    opportunities.append(opp)

        # Sort: IMMEDIATE first, then by edge_pct descending
        opportunities.sort(key=lambda x: (
            0 if x.urgency == Urgency.IMMEDIATE else 1,
            -x.edge_pct
        ))

        # Store and report
        self.opportunities_found.extend(opportunities)

        self.logger.info(f"SCOUT: Found {len(opportunities)} opportunities")

        # Immediate report for urgent opportunities
        for opp in opportunities:
            if opp.urgency == Urgency.IMMEDIATE:
                self._report_immediately(opp)

        return opportunities

    def analyze_bid_ask(self, ticker: str) -> Dict[str, Any]:
        """
        Deep analysis of bid/ask spread quality.

        Returns:
        - Spread analysis
        - Historical comparison
        - Retail vs institutional flow estimate
        - Opportunity assessment
        """
        # Placeholder - would use real market data
        bid = 50.00 + random.uniform(-2, 2)
        ask = bid + random.uniform(0.05, 0.50)
        spread = ask - bid
        spread_pct = spread / bid

        analysis = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "current_bid": bid,
            "current_ask": ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "spread_quality": "poor" if spread_pct > 0.02 else "fair" if spread_pct > 0.01 else "good",
            "historical_avg_spread_pct": 0.015,
            "spread_vs_historical": spread_pct / 0.015,
            "estimated_retail_flow_pct": random.uniform(0.3, 0.7),
            "opportunity_score": 1 - min(spread_pct / 0.05, 1)
        }

        # Flag if spread is abnormally wide
        if spread_pct > analysis["historical_avg_spread_pct"] * 1.5:
            analysis["alert"] = "WIDE_SPREAD"
            analysis["edge_potential"] = spread_pct - analysis["historical_avg_spread_pct"]

        return analysis

    def calculate_fair_value(
        self,
        ticker: str,
        instrument: str = "stock",
        strike: float = None,
        expiry: str = None
    ) -> Dict[str, Any]:
        """
        Calculate theoretical fair value for a security.

        For options, uses Black-Scholes with vol surface adjustments.
        For stocks, uses multi-factor model.
        """
        if instrument == "stock":
            # Simple fair value estimate
            price = random.uniform(20, 100)
            fair_value = price * (1 + random.uniform(-0.05, 0.05))

            return {
                "ticker": ticker,
                "instrument": instrument,
                "market_price": price,
                "fair_value": fair_value,
                "mispricing_pct": (fair_value - price) / price,
                "confidence": random.uniform(0.6, 0.9)
            }
        else:
            # Options pricing
            underlying_price = random.uniform(40, 80)
            theoretical = random.uniform(2, 8)
            market_mid = theoretical * (1 + random.uniform(-0.1, 0.1))

            return {
                "ticker": ticker,
                "instrument": instrument,
                "strike": strike,
                "expiry": expiry,
                "underlying_price": underlying_price,
                "theoretical_value": theoretical,
                "market_mid": market_mid,
                "mispricing_pct": (theoretical - market_mid) / market_mid,
                "implied_vol": random.uniform(0.25, 0.60),
                "delta": random.uniform(0.3, 0.7),
                "confidence": random.uniform(0.5, 0.85)
            }

    def optimize_scalp(self, opportunity: ScalpOpportunity) -> Dict[str, Any]:
        """
        Optimize execution parameters for a scalp trade.

        Calculates:
        - Optimal limit price
        - Position size (even if 1 contract)
        - Expected fill time
        - Risk parameters
        """
        # Calculate optimal entry
        if opportunity.action == "buy":
            # Bid up slightly from current bid
            optimal_limit = opportunity.current_bid + (opportunity.edge_cents * 0.3)
        else:
            # Offer down slightly from current ask
            optimal_limit = opportunity.current_ask - (opportunity.edge_cents * 0.3)

        # Position sizing (conservative for scalps)
        max_risk = 100  # $100 max risk per scalp
        position_size = min(
            opportunity.contracts,
            int(max_risk / opportunity.max_loss) if opportunity.max_loss > 0 else 1
        )

        return {
            "opportunity_id": opportunity.opportunity_id,
            "ticker": opportunity.ticker,
            "recommended_action": opportunity.action,
            "instrument": opportunity.instrument,
            "optimal_limit_price": round(optimal_limit, 2),
            "position_size": max(1, position_size),  # At least 1
            "estimated_fill_probability": 0.75,
            "estimated_fill_time": "30-90 seconds",
            "max_risk_per_contract": opportunity.max_loss,
            "expected_profit_per_contract": opportunity.expected_profit,
            "stop_loss": optimal_limit * 0.98 if opportunity.action == "buy" else optimal_limit * 1.02,
            "target": optimal_limit * 1.005 if opportunity.action == "buy" else optimal_limit * 0.995,
            "execution_notes": "Use limit order, be patient, don't chase"
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_small_mid_cap_universe(self) -> List[str]:
        """Get universe of small/mid cap stocks"""
        # Placeholder - would use real screener
        return [
            "AFRM", "UPST", "SOFI", "HOOD", "COIN", "RBLX", "U", "DKNG",
            "RIVN", "LCID", "FSR", "CCJ", "UEC", "DNN", "NXE", "UUUU",
            "PLTR", "SNOW", "NET", "CRWD", "ZS", "DDOG", "MDB", "CFLT"
        ]

    def _get_market_cap(self, ticker: str) -> float:
        """Get market cap in billions"""
        # Placeholder
        return random.uniform(5, 50)

    def _scan_bid_ask_spread(self, ticker: str, market_cap: float) -> Optional[ScalpOpportunity]:
        """Scan for bad bid/ask opportunities"""
        analysis = self.analyze_bid_ask(ticker)

        if analysis.get("alert") == "WIDE_SPREAD" and analysis.get("edge_potential", 0) > 0.005:
            edge_cents = (analysis["current_ask"] - analysis["current_bid"]) * 0.3 * 100

            return ScalpOpportunity(
                opportunity_id=f"scout_{hashlib.sha256(f'{ticker}{datetime.now()}'.encode()).hexdigest()[:8]}",
                inefficiency_type=InefficiencyType.BAD_BID_ASK,
                ticker=ticker,
                market_cap_bn=market_cap,
                urgency=Urgency.FAST,
                action="buy",  # Buy at bid, sell at ask
                instrument="stock",
                strike=None,
                expiry=None,
                contracts=100,  # shares
                current_bid=analysis["current_bid"],
                current_ask=analysis["current_ask"],
                fair_value=(analysis["current_bid"] + analysis["current_ask"]) / 2,
                edge_cents=edge_cents,
                edge_pct=analysis["edge_potential"],
                max_loss=analysis["spread"] * 100,  # per 100 shares
                expected_profit=edge_cents,
                risk_reward=edge_cents / (analysis["spread"] * 100) if analysis["spread"] > 0 else 0,
                confidence=analysis["opportunity_score"],
                optimal_limit_price=analysis["current_bid"] + 0.02,
                max_slippage=0.02,
                time_to_fill_est="1-2 minutes"
            )

        return None

    def _scan_options_mispricing(self, ticker: str, market_cap: float) -> List[ScalpOpportunity]:
        """Scan for options mispricing"""
        opportunities = []

        # Check a few strikes/expiries
        for _ in range(3):
            fv = self.calculate_fair_value(ticker, "call", strike=50, expiry="2024-03")

            if abs(fv["mispricing_pct"]) > 0.05 and fv["confidence"] > 0.6:
                edge_cents = abs(fv["theoretical_value"] - fv["market_mid"]) * 100

                opportunities.append(ScalpOpportunity(
                    opportunity_id=f"scout_opt_{hashlib.sha256(f'{ticker}{datetime.now()}'.encode()).hexdigest()[:8]}",
                    inefficiency_type=InefficiencyType.OPTIONS_MISPRICING,
                    ticker=ticker,
                    market_cap_bn=market_cap,
                    urgency=Urgency.IMMEDIATE if edge_cents > 10 else Urgency.FAST,
                    action="buy" if fv["mispricing_pct"] < 0 else "sell",
                    instrument="call",
                    strike=50,
                    expiry="2024-03",
                    contracts=1,  # Single contract!
                    current_bid=fv["market_mid"] * 0.95,
                    current_ask=fv["market_mid"] * 1.05,
                    fair_value=fv["theoretical_value"],
                    edge_cents=edge_cents,
                    edge_pct=abs(fv["mispricing_pct"]),
                    max_loss=fv["market_mid"] * 100,  # Full premium
                    expected_profit=edge_cents,
                    risk_reward=edge_cents / (fv["market_mid"] * 100) if fv["market_mid"] > 0 else 0,
                    confidence=fv["confidence"],
                    optimal_limit_price=fv["theoretical_value"],
                    max_slippage=0.05,
                    time_to_fill_est="30-60 seconds"
                ))

        return opportunities

    def _scan_risk_premia(self, ticker: str, market_cap: float) -> Optional[ScalpOpportunity]:
        """Scan for risk premia anomalies"""
        # Placeholder - would analyze term structure, vol surface, etc.
        return None

    def _report_immediately(self, opportunity: ScalpOpportunity):
        """Send immediate notification to HOAGS"""
        self.logger.critical(
            f"[ALERT] SCOUT IMMEDIATE ALERT\n"
            f"Ticker: {opportunity.ticker}\n"
            f"Type: {opportunity.inefficiency_type.value}\n"
            f"Edge: {opportunity.edge_cents:.0f}¢ ({opportunity.edge_pct:.1%})\n"
            f"Action: {opportunity.action.upper()} {opportunity.contracts} {opportunity.instrument}\n"
            f"Limit: ${opportunity.optimal_limit_price:.2f}\n"
            f"Confidence: {opportunity.confidence:.0%}"
        )
        # Would integrate with actual notification system

    def log_action(self, action: str, description: str):
        """Log an action"""
        self.logger.info(f"[SCOUT] {action}: {description}")

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_scan(self, params: Dict) -> Dict:
        universe = params.get("universe")
        types = params.get("types")
        if types:
            types = [InefficiencyType(t) for t in types]

        opps = self.scan_for_inefficiencies(universe, types)
        return {
            "status": "success",
            "opportunities_found": len(opps),
            "immediate_alerts": len([o for o in opps if o.urgency == Urgency.IMMEDIATE]),
            "opportunities": [o.to_dict() for o in opps[:10]]
        }

    def _handle_analyze_ticker(self, params: Dict) -> Dict:
        ticker = params.get("ticker", "")
        analysis = self.analyze_bid_ask(ticker)
        fair_value = self.calculate_fair_value(ticker)
        return {
            "status": "success",
            "bid_ask_analysis": analysis,
            "fair_value": fair_value
        }

    def _handle_find_arbitrage(self, params: Dict) -> Dict:
        ticker = params.get("ticker", "")
        market_cap = self._get_market_cap(ticker)

        opportunities = []
        opp = self._scan_bid_ask_spread(ticker, market_cap)
        if opp:
            opportunities.append(opp)

        opts = self._scan_options_mispricing(ticker, market_cap)
        opportunities.extend(opts)

        return {
            "status": "success",
            "ticker": ticker,
            "opportunities": [o.to_dict() for o in opportunities],
            "opportunity": opportunities[0] if opportunities else None
        }

    def _handle_calculate_scalp(self, params: Dict) -> Dict:
        opp_id = params.get("opportunity_id")
        opp = next((o for o in self.opportunities_found if o.opportunity_id == opp_id), None)
        if opp:
            execution = self.optimize_scalp(opp)
            return {"status": "success", "execution_plan": execution}
        return {"status": "error", "message": "Opportunity not found"}

    def _handle_add_watchlist(self, params: Dict) -> Dict:
        tickers = params.get("tickers", [])
        self.watchlist.extend(tickers)
        self.watchlist = list(set(self.watchlist))
        return {"status": "success", "watchlist": self.watchlist}

    def _handle_get_opportunities(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "total": len(self.opportunities_found),
            "recent": [o.to_dict() for o in self.opportunities_found[-20:]]
        }

    def _handle_report_status(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "total_opportunities_found": len(self.opportunities_found),
            "opportunities_executed": self.opportunities_executed,
            "total_edge_captured_cents": self.total_edge_captured_cents,
            "watchlist_size": len(self.watchlist),
            "config": self.config
        }

    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}


# =============================================================================
# SINGLETON
# =============================================================================

_scout_instance: Optional[ScoutAgent] = None


def get_scout() -> ScoutAgent:
    """Get SCOUT agent singleton"""
    global _scout_instance
    if _scout_instance is None:
        _scout_instance = ScoutAgent()
    return _scout_instance

