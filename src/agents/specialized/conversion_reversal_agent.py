"""
================================================================================
CONVERSION/REVERSAL ARBITRAGE AGENT
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Specialized agent for detecting and executing options arbitrage strategies:
1. Conversions - Long stock + Long put + Short call = Locked profit
2. Reversals - Short stock + Long call + Short put = Locked profit
3. Box Spreads - Risk-free profit from mispriced options
4. Put-Call Parity Violations - Exploit mathematical relationships

These are TRUE ARBITRAGE strategies with mathematically locked profits,
not just statistical edge. The challenge is finding them in small/mid caps
where retail flow creates temporary mispricings.

Tier: STRATEGY (5)
Reports To: SCOUT → HOAGS
Cluster: arbitrage

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT THIS AGENT DOES:
    ConversionReversalAgent hunts for TRUE arbitrage opportunities in
    options markets. Unlike statistical arbitrage which has risk, these
    are mathematically locked profits that arise from temporary mispricings.
    
    The key insight: In small/mid cap stocks (<$30bn), retail order flow
    can create mispricings that violate put-call parity. We exploit these
    before market makers correct them.

KEY FUNCTIONS:
    1. scan_universe() - Scans a universe of tickers for arbitrage
    2. process_opportunity() - Evaluates a specific opportunity
    3. detect_conversion() - Finds long stock + put + short call arbs
    4. detect_reversal() - Finds short stock + call + short put arbs
    5. detect_box_spread() - Finds risk-free profit from 4-leg spreads
    6. calculate_pcp_violation() - Measures put-call parity violations

RELATIONSHIPS WITH OTHER AGENTS:
    - SCOUT: Reports to SCOUT for escalation to HOAGS
    - EXECUTION_AGENT: Sends multi-leg execution instructions
    - DATA_AGENT: Requires real-time options chain data
    - KILLJOY: Submits trades for approval (though true arb = low risk)

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr
    
    # Activate virtual environment:
    .\\venv\\Scripts\\activate
    
    # Train CONVERSION_REVERSAL individually:
    python -m src.training.agent_training_utils --agent CONVERSION_REVERSAL
    
    # Train arbitrage pipeline:
    python -m src.training.agent_training_utils --agents CONVERSION_REVERSAL,SCOUT,EXECUTION_AGENT
    
    # Cross-train with capital allocation:
    python -m src.training.agent_training_utils --cross-train "CONVERSION_REVERSAL,SCOUT:AUTHOR:capital_agent"

RUNNING THE AGENT:
    from src.agents.specialized.conversion_reversal_agent import ConversionReversalAgent
    
    arb_agent = ConversionReversalAgent()
    
    # Scan universe for arbitrage
    result = arb_agent.process({
        "type": "scan_universe",
        "tickers": ["AAPL", "NVDA", "AMD"],
        "min_profit": 0.01
    })
    
    # Process specific opportunity
    result = arb_agent.process({
        "type": "process_opportunity",
        "ticker": "AMD",
        "strike": 150,
        "expiry": "2024-01-19"
    })

================================================================================
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from src.core.agent_base import BaseAgent, AgentTier, ThinkingMode
from src.ml.options_arbitrage_features import (
    ConversionReversalOpportunity,
    calculate_put_call_parity_violation,
    detect_conversion_opportunity,
    detect_reversal_opportunity,
    detect_box_spread_opportunity,
    scan_for_conversion_reversals
)


logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of options arbitrage."""
    CONVERSION = "conversion"
    REVERSAL = "reversal"
    BOX_SPREAD = "box_spread"
    PUT_CALL_PARITY = "put_call_parity"
    DIVIDEND_ARB = "dividend_arb"
    CALENDAR_SPREAD = "calendar_spread"


class ExecutionUrgency(Enum):
    """How quickly to execute."""
    IMMEDIATE = "immediate"  # Seconds - true arb, execute now
    FAST = "fast"            # Minutes - edge may disappear
    STANDARD = "standard"    # Can wait for better fills


@dataclass
class ArbitrageSignal:
    """Signal for an options arbitrage opportunity."""
    signal_id: str
    arb_type: ArbitrageType
    symbol: str
    underlying_price: float
    timestamp: datetime

    # Strike/Expiry info
    strike: float
    strike_2: Optional[float]  # For box spreads
    expiry: str
    days_to_expiry: int

    # Option prices
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float

    # Profit calculation
    gross_profit: float
    transaction_costs: float
    net_profit: float
    annualized_return: float

    # Risk
    capital_required: float
    max_risk: float  # Should be ~0 for true arbitrage

    # Execution
    urgency: ExecutionUrgency
    confidence: float
    execution_legs: List[Dict[str, Any]]

    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "type": self.arb_type.value,
            "symbol": self.symbol,
            "strike": self.strike,
            "expiry": self.expiry,
            "dte": self.days_to_expiry,
            "gross_profit": round(self.gross_profit, 2),
            "net_profit": round(self.net_profit, 2),
            "annualized_return": round(self.annualized_return, 4),
            "capital_required": round(self.capital_required, 2),
            "urgency": self.urgency.value,
            "confidence": round(self.confidence, 3),
            "legs": self.execution_legs
        }


class ConversionReversalAgent(BaseAgent):
    """
    Agent specialized in options arbitrage strategies.

    Scans options chains for violations of put-call parity and other
    mathematical relationships that guarantee profit.
    """

    # Thresholds
    MIN_NET_PROFIT = 5.0  # Minimum $5 net profit per contract
    MIN_ANNUALIZED_RETURN = 0.02  # 2% annualized minimum
    MAX_SPREAD_PCT = 0.10  # 10% max bid-ask spread

    # Transaction cost assumptions
    COMMISSION_PER_CONTRACT = 0.65
    STOCK_COMMISSION = 0.0  # Most brokers free for stocks now
    BORROW_RATE = 0.01  # 1% annual short borrow rate

    def __init__(self):
        super().__init__(
            name="ConversionReversal",
            tier=AgentTier.STRATEGY,
            capabilities=[
                # Detection
                "conversion_detection",
                "reversal_detection",
                "box_spread_detection",
                "parity_violation_detection",

                # Analysis
                "options_pricing",
                "greeks_calculation",
                "implied_volatility_analysis",
                "cost_analysis",

                # Execution
                "multi_leg_execution",
                "execution_sequencing",
                "fill_optimization",
                "slippage_estimation",

                # Risk
                "pin_risk_assessment",
                "early_assignment_risk",
                "dividend_risk_analysis"
            ],
            user_id="TJH"
        )

        # Signal tracking
        self.signals_generated: List[ArbitrageSignal] = []
        self.opportunities_captured = 0
        self.total_profit_captured = 0.0

        # Configuration
        self.config = {
            "min_net_profit": self.MIN_NET_PROFIT,
            "min_annualized_return": self.MIN_ANNUALIZED_RETURN,
            "max_spread_pct": self.MAX_SPREAD_PCT,
            "commission_per_contract": self.COMMISSION_PER_CONTRACT,
            "borrow_rate": self.BORROW_RATE,
            "risk_free_rate": 0.05,  # 5%
        }

        # Primary thinking mode
        self.primary_thinking_mode = ThinkingMode.STRUCTURAL

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task for the Conversion/Reversal Agent."""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.log_action(action, f"Processing: {action}")

        handlers = {
            "scan_chain": self._handle_scan_chain,
            "analyze_strike": self._handle_analyze_strike,
            "check_parity": self._handle_check_parity,
            "find_box_spread": self._handle_find_box_spread,
            "get_signals": self._handle_get_signals,
            "execute_signal": self._handle_execute_signal,
            "get_status": self._handle_get_status,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(params)

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    # =========================================================================
    # CORE ARBITRAGE DETECTION
    # =========================================================================

    def scan_options_chain(
        self,
        symbol: str,
        underlying_price: float,
        options_chain: List[Dict]
    ) -> List[ArbitrageSignal]:
        """
        Scan an options chain for arbitrage opportunities.

        Args:
            symbol: Underlying symbol
            underlying_price: Current stock price
            options_chain: List of option quotes with bids/asks

        Returns:
            List of ArbitrageSignal objects
        """
        signals = []
        r = self.config["risk_free_rate"]

        # Group by strike and expiry
        chain_by_strike = {}
        for opt in options_chain:
            key = (opt.get("strike"), opt.get("expiry"))
            if key not in chain_by_strike:
                chain_by_strike[key] = {"calls": [], "puts": []}

            if opt.get("option_type") == "call":
                chain_by_strike[key]["calls"].append(opt)
            else:
                chain_by_strike[key]["puts"].append(opt)

        # Check each strike/expiry combination
        for (strike, expiry), options in chain_by_strike.items():
            if not options["calls"] or not options["puts"]:
                continue

            call = options["calls"][0]
            put = options["puts"][0]

            # Extract prices
            call_bid = call.get("bid", 0)
            call_ask = call.get("ask", 0)
            put_bid = put.get("bid", 0)
            put_ask = put.get("ask", 0)

            # Skip if no market
            if call_bid == 0 or put_bid == 0:
                continue

            # Calculate DTE
            try:
                expiry_date = datetime.strptime(str(expiry), "%Y-%m-%d")
                dte = (expiry_date - datetime.now()).days
            except:
                dte = 30

            T = dte / 365

            # Check conversion
            conv = detect_conversion_opportunity(
                underlying_price, strike, T, r,
                call_bid, call_ask, put_bid, put_ask,
                borrow_rate=self.config["borrow_rate"],
                transaction_cost=self.config["commission_per_contract"] * 0.03
            )

            if conv and conv["net_profit"] > self.MIN_NET_PROFIT:
                signals.append(self._create_signal(
                    symbol, underlying_price, strike, str(expiry), dte,
                    call_bid, call_ask, put_bid, put_ask,
                    ArbitrageType.CONVERSION, conv
                ))

            # Check reversal
            rev = detect_reversal_opportunity(
                underlying_price, strike, T, r,
                call_bid, call_ask, put_bid, put_ask,
                short_rebate=self.config["borrow_rate"] * 0.5,
                transaction_cost=self.config["commission_per_contract"] * 0.03
            )

            if rev and rev["net_profit"] > self.MIN_NET_PROFIT:
                signals.append(self._create_signal(
                    symbol, underlying_price, strike, str(expiry), dte,
                    call_bid, call_ask, put_bid, put_ask,
                    ArbitrageType.REVERSAL, rev
                ))

        # Sort by profit
        signals.sort(key=lambda s: -s.net_profit)

        logger.info(f"Scanned {len(chain_by_strike)} strikes, found {len(signals)} opportunities")
        return signals

    def check_put_call_parity(
        self,
        symbol: str,
        underlying_price: float,
        strike: float,
        expiry: str,
        call_mid: float,
        put_mid: float,
        dividend: float = 0
    ) -> Dict[str, Any]:
        """
        Check for put-call parity violations.

        Put-Call Parity: C - P = S - K*e^(-rT) - D

        Returns analysis of any violations.
        """
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (expiry_date - datetime.now()).days
        except:
            dte = 30

        T = dte / 365
        r = self.config["risk_free_rate"]

        parity = calculate_put_call_parity_violation(
            underlying_price, strike, T, r, call_mid, put_mid, dividend
        )

        return {
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "dte": dte,
            "underlying": underlying_price,
            "call_mid": call_mid,
            "put_mid": put_mid,
            "parity_analysis": parity,
            "arbitrage_available": not parity["parity_holds"],
            "violation_direction": parity["direction"],
            "potential_profit_pct": parity["violation_pct"]
        }

    def find_box_spread_opportunities(
        self,
        symbol: str,
        options_chain: List[Dict],
        underlying_price: float
    ) -> List[ArbitrageSignal]:
        """
        Find box spread arbitrage opportunities.

        A box spread combines a bull call spread with a bear put spread.
        The payoff is always (K2 - K1), so it should cost that amount
        discounted by the risk-free rate.
        """
        signals = []
        r = self.config["risk_free_rate"]

        # Group options by expiry
        by_expiry = {}
        for opt in options_chain:
            exp = opt.get("expiry")
            if exp not in by_expiry:
                by_expiry[exp] = []
            by_expiry[exp].append(opt)

        # For each expiry, check all strike pairs
        for expiry, options in by_expiry.items():
            # Separate calls and puts by strike
            calls_by_strike = {o["strike"]: o for o in options if o.get("option_type") == "call"}
            puts_by_strike = {o["strike"]: o for o in options if o.get("option_type") == "put"}

            strikes = sorted(set(calls_by_strike.keys()) & set(puts_by_strike.keys()))

            # Check each pair of strikes
            for i, k1 in enumerate(strikes):
                for k2 in strikes[i+1:]:
                    if k1 not in calls_by_strike or k2 not in calls_by_strike:
                        continue
                    if k1 not in puts_by_strike or k2 not in puts_by_strike:
                        continue

                    try:
                        expiry_date = datetime.strptime(str(expiry), "%Y-%m-%d")
                        dte = (expiry_date - datetime.now()).days
                    except:
                        dte = 30

                    T = dte / 365

                    box = detect_box_spread_opportunity(
                        k1, k2, T, r,
                        calls_by_strike[k1].get("bid", 0),
                        calls_by_strike[k1].get("ask", 0),
                        calls_by_strike[k2].get("bid", 0),
                        calls_by_strike[k2].get("ask", 0),
                        puts_by_strike[k1].get("bid", 0),
                        puts_by_strike[k1].get("ask", 0),
                        puts_by_strike[k2].get("bid", 0),
                        puts_by_strike[k2].get("ask", 0),
                        transaction_cost=self.config["commission_per_contract"] * 0.04
                    )

                    if box and box["net_profit"] > self.MIN_NET_PROFIT:
                        signal = ArbitrageSignal(
                            signal_id=f"box_{uuid.uuid4().hex[:8]}",
                            arb_type=ArbitrageType.BOX_SPREAD,
                            symbol=symbol,
                            underlying_price=underlying_price,
                            timestamp=datetime.now(),
                            strike=k1,
                            strike_2=k2,
                            expiry=str(expiry),
                            days_to_expiry=dte,
                            call_bid=calls_by_strike[k1].get("bid", 0),
                            call_ask=calls_by_strike[k1].get("ask", 0),
                            put_bid=puts_by_strike[k1].get("bid", 0),
                            put_ask=puts_by_strike[k1].get("ask", 0),
                            gross_profit=box.get("net_profit", 0) + self.config["commission_per_contract"] * 4,
                            transaction_costs=self.config["commission_per_contract"] * 4,
                            net_profit=box["net_profit"],
                            annualized_return=box.get("annualized_return", 0),
                            capital_required=k2 - k1,
                            max_risk=0,  # Box spread is risk-free
                            urgency=ExecutionUrgency.FAST,
                            confidence=0.95,
                            execution_legs=[
                                {"action": "buy", "type": "call", "strike": k1},
                                {"action": "sell", "type": "call", "strike": k2},
                                {"action": "buy", "type": "put", "strike": k2},
                                {"action": "sell", "type": "put", "strike": k1},
                            ]
                        )
                        signals.append(signal)
                        self.signals_generated.append(signal)

        signals.sort(key=lambda s: -s.net_profit)
        return signals

    def _create_signal(
        self,
        symbol: str,
        underlying: float,
        strike: float,
        expiry: str,
        dte: int,
        call_bid: float,
        call_ask: float,
        put_bid: float,
        put_ask: float,
        arb_type: ArbitrageType,
        arb_result: Dict
    ) -> ArbitrageSignal:
        """Create an ArbitrageSignal from detection results."""

        # Determine execution legs
        if arb_type == ArbitrageType.CONVERSION:
            legs = [
                {"action": "buy", "type": "stock", "quantity": 100},
                {"action": "buy", "type": "put", "strike": strike, "expiry": expiry},
                {"action": "sell", "type": "call", "strike": strike, "expiry": expiry},
            ]
        else:  # Reversal
            legs = [
                {"action": "sell_short", "type": "stock", "quantity": 100},
                {"action": "buy", "type": "call", "strike": strike, "expiry": expiry},
                {"action": "sell", "type": "put", "strike": strike, "expiry": expiry},
            ]

        # Determine urgency based on profit size
        if arb_result["net_profit"] > 20:
            urgency = ExecutionUrgency.IMMEDIATE
        elif arb_result["net_profit"] > 10:
            urgency = ExecutionUrgency.FAST
        else:
            urgency = ExecutionUrgency.STANDARD

        signal = ArbitrageSignal(
            signal_id=f"{arb_type.value}_{uuid.uuid4().hex[:8]}",
            arb_type=arb_type,
            symbol=symbol,
            underlying_price=underlying,
            timestamp=datetime.now(),
            strike=strike,
            strike_2=None,
            expiry=expiry,
            days_to_expiry=dte,
            call_bid=call_bid,
            call_ask=call_ask,
            put_bid=put_bid,
            put_ask=put_ask,
            gross_profit=arb_result.get("gross_profit", 0),
            transaction_costs=self.config["commission_per_contract"] * 3,
            net_profit=arb_result["net_profit"],
            annualized_return=arb_result.get("annualized_return", 0),
            capital_required=arb_result.get("capital_required", underlying),
            max_risk=0,  # True arbitrage has no risk
            urgency=urgency,
            confidence=0.90 if arb_result["net_profit"] > 10 else 0.75,
            execution_legs=legs
        )

        self.signals_generated.append(signal)
        return signal

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_scan_chain(self, params: Dict) -> Dict:
        """Scan an options chain for opportunities."""
        symbol = params.get("symbol", "")
        underlying_price = params.get("underlying_price", 0)
        options_chain = params.get("options_chain", [])

        signals = self.scan_options_chain(symbol, underlying_price, options_chain)

        return {
            "status": "success",
            "symbol": symbol,
            "opportunities_found": len(signals),
            "signals": [s.to_dict() for s in signals[:10]]
        }

    def _handle_analyze_strike(self, params: Dict) -> Dict:
        """Analyze a specific strike for arbitrage."""
        symbol = params.get("symbol", "")
        underlying = params.get("underlying_price", 0)
        strike = params.get("strike", 0)
        expiry = params.get("expiry", "")
        call_bid = params.get("call_bid", 0)
        call_ask = params.get("call_ask", 0)
        put_bid = params.get("put_bid", 0)
        put_ask = params.get("put_ask", 0)

        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (expiry_date - datetime.now()).days
        except:
            dte = 30

        T = dte / 365
        r = self.config["risk_free_rate"]

        conv = detect_conversion_opportunity(
            underlying, strike, T, r,
            call_bid, call_ask, put_bid, put_ask
        )

        rev = detect_reversal_opportunity(
            underlying, strike, T, r,
            call_bid, call_ask, put_bid, put_ask
        )

        return {
            "status": "success",
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "conversion": conv,
            "reversal": rev,
            "best_opportunity": "conversion" if (conv and (not rev or conv["net_profit"] > rev["net_profit"])) else "reversal" if rev else None
        }

    def _handle_check_parity(self, params: Dict) -> Dict:
        """Check put-call parity."""
        result = self.check_put_call_parity(
            params.get("symbol", ""),
            params.get("underlying_price", 0),
            params.get("strike", 0),
            params.get("expiry", ""),
            params.get("call_mid", 0),
            params.get("put_mid", 0),
            params.get("dividend", 0)
        )
        return {"status": "success", "parity_check": result}

    def _handle_find_box_spread(self, params: Dict) -> Dict:
        """Find box spread opportunities."""
        signals = self.find_box_spread_opportunities(
            params.get("symbol", ""),
            params.get("options_chain", []),
            params.get("underlying_price", 0)
        )
        return {
            "status": "success",
            "opportunities_found": len(signals),
            "signals": [s.to_dict() for s in signals[:5]]
        }

    def _handle_get_signals(self, params: Dict) -> Dict:
        """Get recent signals."""
        max_age = params.get("max_age_seconds", 600)
        active = [
            s for s in self.signals_generated
            if (datetime.now() - s.timestamp).total_seconds() < max_age
        ]
        return {
            "status": "success",
            "count": len(active),
            "signals": [s.to_dict() for s in active]
        }

    def _handle_execute_signal(self, params: Dict) -> Dict:
        """Mark a signal as executed."""
        signal_id = params.get("signal_id", "")
        signal = next((s for s in self.signals_generated if s.signal_id == signal_id), None)

        if signal:
            self.opportunities_captured += 1
            self.total_profit_captured += signal.net_profit
            return {
                "status": "success",
                "signal_id": signal_id,
                "profit_captured": signal.net_profit
            }
        return {"status": "error", "message": "Signal not found"}

    def _handle_get_status(self, params: Dict) -> Dict:
        """Get agent status."""
        return {
            "status": "success",
            "agent": self.name,
            "signals_generated": len(self.signals_generated),
            "opportunities_captured": self.opportunities_captured,
            "total_profit_captured": self.total_profit_captured,
            "config": self.config
        }

    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}

    def log_action(self, action: str, description: str):
        logger.info(f"[ConversionReversal] {action}: {description}")


# Singleton
_conv_rev_agent: Optional[ConversionReversalAgent] = None


def get_conversion_reversal_agent() -> ConversionReversalAgent:
    """Get the Conversion/Reversal Agent singleton."""
    global _conv_rev_agent
    if _conv_rev_agent is None:
        _conv_rev_agent = ConversionReversalAgent()
    return _conv_rev_agent
