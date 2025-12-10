"""================================================================================
OPTIONS ARBITRAGE FEATURE ENGINEERING
================================================================================
Author: Alpha Loop Capital, LLC

Features for detecting options arbitrage opportunities:
1. Conversions - Long stock + long put + short call (locked profit)
2. Reversals - Short stock + long call + short put (locked profit)
3. Box Spreads - Risk-free profit from mispriced options
4. Put-Call Parity Violations
5. Dividend Arbitrage
6. Calendar Spread Mispricing
7. Volatility Surface Anomalies

These strategies exploit mathematical relationships that MUST hold in efficient
markets. When retail flow or low liquidity breaks these relationships, we profit.

Target: Small/mid cap options with wide spreads and low institutional presence.
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


# Black-Scholes helper functions
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(0, call)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0, put)


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       option_type: str = "call", max_iter: int = 100) -> float:
    """Calculate implied volatility using Newton-Raphson."""
    if T <= 0:
        return 0.0

    sigma = 0.3  # Initial guess

    for _ in range(max_iter):
        if option_type == "call":
            bs_price = black_scholes_call(S, K, T, r, sigma)
        else:
            bs_price = black_scholes_put(S, K, T, r, sigma)

        # Vega
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)

        if vega < 1e-10:
            break

        diff = bs_price - price
        if abs(diff) < 1e-6:
            break

        sigma = sigma - diff / vega
        sigma = max(0.01, min(5.0, sigma))  # Bound sigma

    return sigma


@dataclass
class ConversionReversalOpportunity:
    """Detected conversion or reversal arbitrage opportunity."""

    symbol: str
    underlying_price: float
    strike: float
    expiry: str
    days_to_expiry: int

    # Option prices
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float

    # Arbitrage type
    arb_type: str  # "conversion", "reversal", "box_spread"

    # P&L
    theoretical_pnl: float
    pnl_after_costs: float
    annualized_return: float

    # Risk metrics
    max_risk: float
    capital_required: float

    # Execution
    confidence: float
    urgency: str

    detected_at: datetime = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying_price,
            "strike": self.strike,
            "expiry": self.expiry,
            "dte": self.days_to_expiry,
            "type": self.arb_type,
            "theoretical_pnl": self.theoretical_pnl,
            "net_pnl": self.pnl_after_costs,
            "annualized_return": self.annualized_return,
            "confidence": self.confidence,
            "urgency": self.urgency,
        }


def calculate_put_call_parity_violation(
    S: float,
    K: float,
    T: float,
    r: float,
    call_mid: float,
    put_mid: float,
    dividend: float = 0.0,
) -> Dict[str, float]:
    """Check for put-call parity violations.

    Put-Call Parity: C - P = S - K*e^(-rT) - D

    Where:
    - C = call price
    - P = put price
    - S = stock price
    - K = strike
    - r = risk-free rate
    - T = time to expiry
    - D = present value of dividends

    Violations indicate arbitrage opportunity.
    """
    # Present value of strike
    pv_strike = K * np.exp(-r * T)

    # Theoretical relationship
    synthetic_forward = S - dividend - pv_strike

    # Actual relationship from market prices
    market_relationship = call_mid - put_mid

    # Violation
    violation = market_relationship - synthetic_forward
    violation_pct = abs(violation) / S if S > 0 else 0

    return {
        "synthetic_forward": synthetic_forward,
        "market_relationship": market_relationship,
        "violation": violation,
        "violation_pct": violation_pct,
        "parity_holds": abs(violation_pct) < 0.005,  # Within 0.5%
        "direction": "call_rich" if violation > 0 else "put_rich",
    }


def detect_conversion_opportunity(
    S: float,
    K: float,
    T: float,
    r: float,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    borrow_rate: float = 0.01,
    transaction_cost: float = 0.01,
) -> Optional[Dict]:
    """Detect conversion arbitrage opportunity.

    Conversion: Buy stock + Buy put + Sell call (at same strike/expiry)
    Lock in profit if: Stock + Put - Call > K * e^(-rT) + costs

    This exploits when calls are overpriced relative to puts.
    """
    # Cost to establish conversion
    # Buy stock at S, buy put at ask, sell call at bid
    cost = S + put_ask - call_bid

    # Guaranteed payoff at expiry = Strike (exercise put or get called)
    payoff = K

    # Present value of payoff
    pv_payoff = payoff * np.exp(-r * T)

    # Carrying costs (borrow rate for stock)
    carry_cost = S * borrow_rate * T

    # Total costs
    total_costs = transaction_cost * 3 + carry_cost  # 3 legs

    # Net profit
    gross_profit = pv_payoff - cost
    net_profit = gross_profit - total_costs

    if net_profit > 0:
        # Annualize the return
        capital = S  # Capital tied up in stock
        if T > 0 and capital > 0:
            annualized = (net_profit / capital) * (365 / (T * 365))
        else:
            annualized = 0

        return {
            "type": "conversion",
            "cost": cost,
            "payoff": payoff,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "annualized_return": annualized,
            "capital_required": S,
            "edge_pct": net_profit / S if S > 0 else 0,
        }

    return None


def detect_reversal_opportunity(
    S: float,
    K: float,
    T: float,
    r: float,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    short_rebate: float = 0.005,
    transaction_cost: float = 0.01,
) -> Optional[Dict]:
    """Detect reversal arbitrage opportunity.

    Reversal: Short stock + Buy call + Sell put (at same strike/expiry)
    Lock in profit if: Call - Put - Stock < -K * e^(-rT) - costs

    This exploits when puts are overpriced relative to calls.
    """
    # Credit from establishing reversal
    # Short stock at S, buy call at ask, sell put at bid
    credit = S + put_bid - call_ask

    # Guaranteed payoff at expiry = -Strike (buy back stock at strike)
    cost_to_close = K

    # Present value
    pv_cost = cost_to_close * np.exp(-r * T)

    # Short stock rebate (interest on short proceeds)
    rebate = S * short_rebate * T

    # Total costs
    total_costs = transaction_cost * 3

    # Net profit
    gross_profit = credit - pv_cost
    net_profit = gross_profit + rebate - total_costs

    if net_profit > 0:
        capital = S  # Margin requirement approximation
        if T > 0 and capital > 0:
            annualized = (net_profit / capital) * (365 / (T * 365))
        else:
            annualized = 0

        return {
            "type": "reversal",
            "credit": credit,
            "cost_to_close": cost_to_close,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "annualized_return": annualized,
            "capital_required": S,
            "edge_pct": net_profit / S if S > 0 else 0,
        }

    return None


def detect_box_spread_opportunity(
    K1: float,
    K2: float,
    T: float,
    r: float,
    call_K1_bid: float,
    call_K1_ask: float,
    call_K2_bid: float,
    call_K2_ask: float,
    put_K1_bid: float,
    put_K1_ask: float,
    put_K2_bid: float,
    put_K2_ask: float,
    transaction_cost: float = 0.01,
) -> Optional[Dict]:
    """Detect box spread arbitrage opportunity.

    Box Spread: Bull call spread + Bear put spread
    - Buy call K1, sell call K2 (bull spread)
    - Buy put K2, sell put K1 (bear spread)

    Payoff always = K2 - K1 (width of strikes)
    Should cost (K2 - K1) * e^(-rT)

    If market price differs, arbitrage exists.
    """
    # Cost to establish box (buy low strike call/put, sell high strike call/put)
    box_cost = (
        call_K1_ask - call_K2_bid +  # Bull call spread
        put_K2_ask - put_K1_bid      # Bear put spread
    )

    # Guaranteed payoff
    box_payoff = K2 - K1

    # Fair value
    fair_value = box_payoff * np.exp(-r * T)

    # Costs
    total_costs = transaction_cost * 4  # 4 legs

    # Check for mispricing
    if box_cost < fair_value - total_costs:
        # Box is underpriced - buy it
        net_profit = fair_value - box_cost - total_costs
        return {
            "type": "box_long",
            "box_cost": box_cost,
            "fair_value": fair_value,
            "net_profit": net_profit,
            "annualized_return": (net_profit / box_cost) / T if T > 0 and box_cost > 0 else 0,
            "edge_pct": net_profit / fair_value if fair_value > 0 else 0,
        }

    # Check if selling box is profitable
    box_credit = (
        call_K1_bid - call_K2_ask +
        put_K2_bid - put_K1_ask
    )

    if box_credit > fair_value + total_costs:
        net_profit = box_credit - fair_value - total_costs
        return {
            "type": "box_short",
            "box_credit": box_credit,
            "fair_value": fair_value,
            "net_profit": net_profit,
            "annualized_return": (net_profit / fair_value) / T if T > 0 and fair_value > 0 else 0,
            "edge_pct": net_profit / fair_value if fair_value > 0 else 0,
        }

    return None


def add_options_arbitrage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add options arbitrage detection features to a DataFrame of options data.

    Expected columns:
    - symbol, underlying_price, strike, expiry
    - call_bid, call_ask, put_bid, put_ask
    - days_to_expiry (or expiry date to calculate)

    Returns DataFrame with arbitrage features added.
    """
    df = df.copy()

    required = ["underlying_price", "strike", "call_bid", "call_ask", "put_bid", "put_ask"]
    if not all(c in df.columns for c in required):
        logger.warning("Missing required columns for options arbitrage features")
        return df

    # Risk-free rate assumption
    r = 0.05  # 5% risk-free rate

    # Time to expiry
    if "days_to_expiry" not in df.columns:
        if "expiry" in df.columns:
            df["expiry_date"] = pd.to_datetime(df["expiry"])
            df["days_to_expiry"] = (df["expiry_date"] - datetime.now()).dt.days
        else:
            df["days_to_expiry"] = 30  # Default

    df["T"] = df["days_to_expiry"] / 365

    # Mid prices
    df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2
    df["put_mid"] = (df["put_bid"] + df["put_ask"]) / 2

    # Spreads
    df["call_spread"] = df["call_ask"] - df["call_bid"]
    df["put_spread"] = df["put_ask"] - df["put_bid"]
    df["call_spread_pct"] = df["call_spread"] / (df["call_mid"] + 0.01)
    df["put_spread_pct"] = df["put_spread"] / (df["put_mid"] + 0.01)

    # Put-Call Parity check
    pv_strike = df["strike"] * np.exp(-r * df["T"])
    df["synthetic_forward"] = df["underlying_price"] - pv_strike
    df["market_forward"] = df["call_mid"] - df["put_mid"]
    df["parity_violation"] = df["market_forward"] - df["synthetic_forward"]
    df["parity_violation_pct"] = abs(df["parity_violation"]) / df["underlying_price"]

    # Conversion/Reversal signals
    # Conversion profitable when call rich (sell call, buy put)
    df["conversion_edge"] = (
        df["call_bid"] - df["put_ask"] -
        (df["underlying_price"] - pv_strike)
    )

    # Reversal profitable when put rich (sell put, buy call)
    df["reversal_edge"] = (
        df["put_bid"] - df["call_ask"] +
        (df["underlying_price"] - pv_strike)
    )

    # Flag opportunities
    min_edge = 0.10  # Minimum $0.10 edge
    df["conversion_signal"] = (df["conversion_edge"] > min_edge).astype(int)
    df["reversal_signal"] = (df["reversal_edge"] > min_edge).astype(int)

    # Implied volatility features
    df["call_iv"] = df.apply(
        lambda row: implied_volatility(
            row["call_mid"], row["underlying_price"], row["strike"],
            row["T"], r, "call",
        ) if row["T"] > 0 else 0,
        axis=1,
    )

    df["put_iv"] = df.apply(
        lambda row: implied_volatility(
            row["put_mid"], row["underlying_price"], row["strike"],
            row["T"], r, "put",
        ) if row["T"] > 0 else 0,
        axis=1,
    )

    # IV skew (put vs call IV at same strike)
    df["iv_skew"] = df["put_iv"] - df["call_iv"]
    df["iv_skew_pct"] = df["iv_skew"] / (df["call_iv"] + 0.01)

    # Moneyness
    df["moneyness"] = df["underlying_price"] / df["strike"]
    df["otm_call"] = (df["moneyness"] < 1).astype(int)
    df["otm_put"] = (df["moneyness"] > 1).astype(int)

    # Greeks approximations
    df["approx_delta_call"] = norm.cdf(
        np.log(df["moneyness"]) / (df["call_iv"] * np.sqrt(df["T"]) + 0.01),
    )
    df["approx_delta_put"] = df["approx_delta_call"] - 1

    # Arbitrage opportunity score
    df["arb_opportunity_score"] = (
        df["parity_violation_pct"] * 10 +
        df["conversion_signal"] * 5 +
        df["reversal_signal"] * 5 +
        abs(df["iv_skew_pct"]) * 2 +
        (df["call_spread_pct"] > 0.1).astype(int) * 3 +
        (df["put_spread_pct"] > 0.1).astype(int) * 3
    )

    # Normalize
    df["arb_opportunity_normalized"] = (
        df["arb_opportunity_score"] /
        (df["arb_opportunity_score"].rolling(100, min_periods=1).max() + 1)
    ).clip(0, 1)

    logger.info(f"Added options arbitrage features: {len(df)} rows")
    return df


def scan_for_conversion_reversals(
    options_data: pd.DataFrame,
    underlying_prices: Dict[str, float],
    min_edge: float = 0.05,
    max_spread_pct: float = 0.10,
) -> List[ConversionReversalOpportunity]:
    """Scan options data for conversion/reversal opportunities.

    Args:
    ----
        options_data: DataFrame with options chain data
        underlying_prices: Dict of symbol -> current price
        min_edge: Minimum edge (%) to flag opportunity
        max_spread_pct: Maximum bid-ask spread to consider

    Returns:
    -------
        List of ConversionReversalOpportunity objects
    """
    opportunities = []
    r = 0.05  # Risk-free rate

    for symbol in options_data["symbol"].unique():
        symbol_data = options_data[options_data["symbol"] == symbol]
        S = underlying_prices.get(symbol, symbol_data["underlying_price"].iloc[0])

        for _, row in symbol_data.iterrows():
            K = row["strike"]
            T = row.get("days_to_expiry", 30) / 365

            # Skip if spreads too wide
            call_spread_pct = (row["call_ask"] - row["call_bid"]) / (row["call_bid"] + 0.01)
            put_spread_pct = (row["put_ask"] - row["put_bid"]) / (row["put_bid"] + 0.01)

            if call_spread_pct > max_spread_pct or put_spread_pct > max_spread_pct:
                continue

            # Check conversion
            conv = detect_conversion_opportunity(
                S, K, T, r,
                row["call_bid"], row["call_ask"],
                row["put_bid"], row["put_ask"],
            )

            if conv and conv["edge_pct"] > min_edge:
                opportunities.append(ConversionReversalOpportunity(
                    symbol=symbol,
                    underlying_price=S,
                    strike=K,
                    expiry=str(row.get("expiry", "")),
                    days_to_expiry=int(T * 365),
                    call_bid=row["call_bid"],
                    call_ask=row["call_ask"],
                    put_bid=row["put_bid"],
                    put_ask=row["put_ask"],
                    arb_type="conversion",
                    theoretical_pnl=conv["gross_profit"],
                    pnl_after_costs=conv["net_profit"],
                    annualized_return=conv["annualized_return"],
                    max_risk=S * 0.05,  # 5% max risk estimate
                    capital_required=conv["capital_required"],
                    confidence=min(0.95, 0.5 + conv["edge_pct"] * 5),
                    urgency="immediate" if conv["edge_pct"] > 0.02 else "fast",
                ))

            # Check reversal
            rev = detect_reversal_opportunity(
                S, K, T, r,
                row["call_bid"], row["call_ask"],
                row["put_bid"], row["put_ask"],
            )

            if rev and rev["edge_pct"] > min_edge:
                opportunities.append(ConversionReversalOpportunity(
                    symbol=symbol,
                    underlying_price=S,
                    strike=K,
                    expiry=str(row.get("expiry", "")),
                    days_to_expiry=int(T * 365),
                    call_bid=row["call_bid"],
                    call_ask=row["call_ask"],
                    put_bid=row["put_bid"],
                    put_ask=row["put_ask"],
                    arb_type="reversal",
                    theoretical_pnl=rev["gross_profit"],
                    pnl_after_costs=rev["net_profit"],
                    annualized_return=rev["annualized_return"],
                    max_risk=S * 0.05,
                    capital_required=rev["capital_required"],
                    confidence=min(0.95, 0.5 + rev["edge_pct"] * 5),
                    urgency="immediate" if rev["edge_pct"] > 0.02 else "fast",
                ))

    # Sort by edge
    opportunities.sort(key=lambda x: -x.pnl_after_costs)

    logger.info(f"Found {len(opportunities)} conversion/reversal opportunities")
    return opportunities


@dataclass
class ButterflyOpportunity:
    """Detected butterfly spread arbitrage opportunity."""
    symbol: str
    underlying_price: float
    strike_low: float
    strike_mid: float
    strike_high: float
    expiry: str
    spread_type: str  # "call_butterfly", "put_butterfly", "iron_butterfly"
    cost: float
    max_profit: float
    max_loss: float
    breakeven_low: float
    breakeven_high: float
    edge_pct: float
    confidence: float


@dataclass
class IronCondorOpportunity:
    """Detected iron condor opportunity."""
    symbol: str
    underlying_price: float
    put_strike_low: float
    put_strike_high: float
    call_strike_low: float
    call_strike_high: float
    expiry: str
    credit_received: float
    max_profit: float
    max_loss: float
    breakeven_low: float
    breakeven_high: float
    prob_profit: float
    edge_pct: float


@dataclass
class CalendarSpreadOpportunity:
    """Detected calendar spread (time spread) opportunity."""
    symbol: str
    underlying_price: float
    strike: float
    near_expiry: str
    far_expiry: str
    spread_type: str  # "call_calendar", "put_calendar"
    cost: float
    near_iv: float
    far_iv: float
    iv_differential: float
    theta_edge: float
    confidence: float


def detect_butterfly_opportunity(
    S: float,
    K_low: float,
    K_mid: float,
    K_high: float,
    call_K_low_bid: float,
    call_K_low_ask: float,
    call_K_mid_bid: float,
    call_K_mid_ask: float,
    call_K_high_bid: float,
    call_K_high_ask: float,
    transaction_cost: float = 0.02,
) -> Optional[Dict]:
    """Detect call butterfly spread opportunity.

    Butterfly: Buy 1 K_low call, Sell 2 K_mid calls, Buy 1 K_high call
    Max profit at K_mid, limited loss = premium paid

    Arbitrage exists if cost < 0 (credit) or cost << max profit
    """
    # Must have equal wing widths
    if abs((K_mid - K_low) - (K_high - K_mid)) > 0.01:
        return None

    wing_width = K_mid - K_low

    # Cost to establish (buy wings, sell body)
    cost = (
        call_K_low_ask +  # Buy low strike
        call_K_high_ask -  # Buy high strike
        2 * call_K_mid_bid  # Sell 2 middle strikes
    )

    # Max profit = wing width - cost
    max_profit = wing_width - cost

    # Max loss = cost (if positive) or unlimited if credit
    max_loss = max(cost, 0)

    total_costs = transaction_cost * 4  # 4 legs

    # Breakevens
    breakeven_low = K_low + cost
    breakeven_high = K_high - cost

    net_profit = max_profit - total_costs

    # Flag if favorable risk/reward
    if net_profit > 0 and (net_profit / (max_loss + 0.01)) > 0.5:
        return {
            "type": "call_butterfly",
            "cost": cost,
            "max_profit": max_profit,
            "net_profit": net_profit,
            "max_loss": max_loss,
            "breakeven_low": breakeven_low,
            "breakeven_high": breakeven_high,
            "wing_width": wing_width,
            "risk_reward": net_profit / (max_loss + 0.01),
        }

    return None


def detect_iron_butterfly_opportunity(
    S: float,
    K: float,  # ATM strike
    K_low: float,  # OTM put strike
    K_high: float,  # OTM call strike
    call_K_bid: float,
    call_K_ask: float,
    put_K_bid: float,
    put_K_ask: float,
    call_K_high_bid: float,
    call_K_high_ask: float,
    put_K_low_bid: float,
    put_K_low_ask: float,
    transaction_cost: float = 0.02,
) -> Optional[Dict]:
    """Detect iron butterfly opportunity.

    Iron Butterfly: Sell ATM straddle + Buy OTM strangle
    - Sell call at K
    - Sell put at K
    - Buy call at K_high
    - Buy put at K_low

    Credit received upfront, max loss limited by wings.
    """
    # Credit from selling ATM straddle
    straddle_credit = call_K_bid + put_K_bid

    # Cost of buying wings
    wing_cost = call_K_high_ask + put_K_low_ask

    net_credit = straddle_credit - wing_cost
    total_costs = transaction_cost * 4

    if net_credit <= total_costs:
        return None

    # Max profit = net credit
    max_profit = net_credit - total_costs

    # Max loss = wing width - net credit
    wing_width_up = K_high - K
    wing_width_down = K - K_low
    max_loss = max(wing_width_up, wing_width_down) - net_credit

    # Breakevens
    breakeven_low = K - net_credit
    breakeven_high = K + net_credit

    if max_profit > 0 and max_loss > 0:
        return {
            "type": "iron_butterfly",
            "credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven_low": breakeven_low,
            "breakeven_high": breakeven_high,
            "risk_reward": max_profit / max_loss,
            "width_up": wing_width_up,
            "width_down": wing_width_down,
        }

    return None


def detect_iron_condor_opportunity(
    S: float,
    put_K_low: float,
    put_K_high: float,
    call_K_low: float,
    call_K_high: float,
    put_K_low_bid: float,
    put_K_low_ask: float,
    put_K_high_bid: float,
    put_K_high_ask: float,
    call_K_low_bid: float,
    call_K_low_ask: float,
    call_K_high_bid: float,
    call_K_high_ask: float,
    transaction_cost: float = 0.02,
) -> Optional[Dict]:
    """Detect iron condor opportunity.

    Iron Condor: Bull put spread + Bear call spread
    - Sell put at put_K_high (higher strike)
    - Buy put at put_K_low (lower strike)
    - Sell call at call_K_low (lower strike)
    - Buy call at call_K_high (higher strike)

    Profits if price stays between put_K_high and call_K_low
    """
    # Put spread credit (sell high, buy low)
    put_credit = put_K_high_bid - put_K_low_ask

    # Call spread credit (sell low, buy high)
    call_credit = call_K_low_bid - call_K_high_ask

    net_credit = put_credit + call_credit
    total_costs = transaction_cost * 4

    if net_credit <= total_costs:
        return None

    # Max profit = net credit
    max_profit = net_credit - total_costs

    # Max loss = wider wing width - net credit
    put_width = put_K_high - put_K_low
    call_width = call_K_high - call_K_low
    max_wing_width = max(put_width, call_width)
    max_loss = max_wing_width - net_credit

    # Breakevens
    breakeven_low = put_K_high - net_credit
    breakeven_high = call_K_low + net_credit

    # Probability of profit estimate (simplified)
    # If price stays in profit zone
    profit_zone_width = breakeven_high - breakeven_low
    total_range = call_K_high - put_K_low
    prob_profit = profit_zone_width / total_range if total_range > 0 else 0

    if max_profit > 0 and max_loss > 0 and prob_profit > 0.3:
        return {
            "type": "iron_condor",
            "credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven_low": breakeven_low,
            "breakeven_high": breakeven_high,
            "prob_profit": prob_profit,
            "risk_reward": max_profit / max_loss,
            "put_width": put_width,
            "call_width": call_width,
        }

    return None


def detect_calendar_spread_opportunity(
    S: float,
    K: float,
    near_call_bid: float,
    near_call_ask: float,
    far_call_bid: float,
    far_call_ask: float,
    near_T: float,
    far_T: float,
    near_iv: float,
    far_iv: float,
    transaction_cost: float = 0.02,
) -> Optional[Dict]:
    """Detect calendar spread (time spread) opportunity.

    Calendar: Sell near-term option, Buy far-term option (same strike)
    Profits from time decay differential and IV changes.

    Arbitrage exists when near-term IV >> far-term IV (unusual)
    or when theta differential is exploitable.
    """
    # Cost to establish (buy far, sell near)
    cost = far_call_ask - near_call_bid
    total_costs = transaction_cost * 2

    if cost <= 0:
        # Credit calendar - unusual, flag it
        return {
            "type": "credit_calendar",
            "cost": cost,  # Actually a credit
            "near_iv": near_iv,
            "far_iv": far_iv,
            "iv_differential": near_iv - far_iv,
            "edge_pct": abs(cost) / S if S > 0 else 0,
            "near_T": near_T,
            "far_T": far_T,
        }

    # IV differential check
    iv_diff = near_iv - far_iv

    # Theta differential (approximate)
    # Near-term options decay faster
    # Edge exists when selling high IV near-term vs buying low IV far-term
    if iv_diff > 0.05:  # Near IV 5%+ higher than far
        theta_edge = iv_diff * np.sqrt(near_T) * S * 0.01  # Rough estimate

        return {
            "type": "call_calendar",
            "cost": cost,
            "near_iv": near_iv,
            "far_iv": far_iv,
            "iv_differential": iv_diff,
            "theta_edge": theta_edge,
            "edge_pct": theta_edge / cost if cost > 0 else 0,
            "near_T": near_T,
            "far_T": far_T,
        }

    return None


def scan_for_complex_spreads(
    options_data: pd.DataFrame,
    underlying_prices: Dict[str, float],
    min_edge: float = 0.02,
) -> Dict[str, List]:
    """Scan for butterfly, iron condor, and calendar spread opportunities.

    Args:
        options_data: DataFrame with full options chain
        underlying_prices: Dict of symbol -> current price
        min_edge: Minimum edge to flag

    Returns:
        Dict with lists of opportunities by type
    """
    butterflies = []
    iron_condors = []
    calendars = []

    r = 0.05

    for symbol in options_data["underlying_symbol"].unique() if "underlying_symbol" in options_data.columns else options_data["symbol"].unique():
        S = underlying_prices.get(symbol, 100)
        symbol_data = options_data[
            (options_data.get("underlying_symbol", options_data.get("symbol")) == symbol)
        ]

        # Get unique strikes sorted
        strikes = sorted(symbol_data["strike"].unique())

        if len(strikes) < 4:
            continue

        # Scan for butterflies (need 3 consecutive strikes)
        for i in range(len(strikes) - 2):
            K_low, K_mid, K_high = strikes[i], strikes[i + 1], strikes[i + 2]

            # Check if equal spacing
            if abs((K_mid - K_low) - (K_high - K_mid)) > 1:
                continue

            try:
                low_data = symbol_data[symbol_data["strike"] == K_low].iloc[0]
                mid_data = symbol_data[symbol_data["strike"] == K_mid].iloc[0]
                high_data = symbol_data[symbol_data["strike"] == K_high].iloc[0]

                butterfly = detect_butterfly_opportunity(
                    S, K_low, K_mid, K_high,
                    low_data["call_bid"], low_data["call_ask"],
                    mid_data["call_bid"], mid_data["call_ask"],
                    high_data["call_bid"], high_data["call_ask"],
                )

                if butterfly and butterfly.get("risk_reward", 0) > 0.5:
                    butterflies.append(ButterflyOpportunity(
                        symbol=symbol,
                        underlying_price=S,
                        strike_low=K_low,
                        strike_mid=K_mid,
                        strike_high=K_high,
                        expiry=str(low_data.get("expiry", "")),
                        spread_type=butterfly["type"],
                        cost=butterfly["cost"],
                        max_profit=butterfly["max_profit"],
                        max_loss=butterfly["max_loss"],
                        breakeven_low=butterfly["breakeven_low"],
                        breakeven_high=butterfly["breakeven_high"],
                        edge_pct=butterfly["risk_reward"],
                        confidence=min(0.9, butterfly["risk_reward"]),
                    ))
            except (IndexError, KeyError):
                continue

        # Scan for iron condors (need 4 strikes)
        for i in range(len(strikes) - 3):
            put_K_low = strikes[i]
            put_K_high = strikes[i + 1]
            call_K_low = strikes[i + 2]
            call_K_high = strikes[i + 3]

            # Only if symmetric around current price
            if not (put_K_high < S < call_K_low):
                continue

            try:
                p_low = symbol_data[symbol_data["strike"] == put_K_low].iloc[0]
                p_high = symbol_data[symbol_data["strike"] == put_K_high].iloc[0]
                c_low = symbol_data[symbol_data["strike"] == call_K_low].iloc[0]
                c_high = symbol_data[symbol_data["strike"] == call_K_high].iloc[0]

                condor = detect_iron_condor_opportunity(
                    S,
                    put_K_low, put_K_high, call_K_low, call_K_high,
                    p_low["put_bid"], p_low["put_ask"],
                    p_high["put_bid"], p_high["put_ask"],
                    c_low["call_bid"], c_low["call_ask"],
                    c_high["call_bid"], c_high["call_ask"],
                )

                if condor and condor.get("risk_reward", 0) > 0.3:
                    iron_condors.append(IronCondorOpportunity(
                        symbol=symbol,
                        underlying_price=S,
                        put_strike_low=put_K_low,
                        put_strike_high=put_K_high,
                        call_strike_low=call_K_low,
                        call_strike_high=call_K_high,
                        expiry=str(p_low.get("expiry", "")),
                        credit_received=condor["credit"],
                        max_profit=condor["max_profit"],
                        max_loss=condor["max_loss"],
                        breakeven_low=condor["breakeven_low"],
                        breakeven_high=condor["breakeven_high"],
                        prob_profit=condor["prob_profit"],
                        edge_pct=condor["risk_reward"],
                    ))
            except (IndexError, KeyError):
                continue

    logger.info(f"Found {len(butterflies)} butterflies, {len(iron_condors)} iron condors, {len(calendars)} calendars")

    return {
        "butterflies": butterflies,
        "iron_condors": iron_condors,
        "calendars": calendars,
    }


def get_options_arbitrage_feature_names() -> List[str]:
    """Get list of all options arbitrage feature names."""
    return [
        # Spreads
        "call_spread",
        "put_spread",
        "call_spread_pct",
        "put_spread_pct",

        # Put-Call Parity
        "synthetic_forward",
        "market_forward",
        "parity_violation",
        "parity_violation_pct",

        # Conversion/Reversal
        "conversion_edge",
        "reversal_edge",
        "conversion_signal",
        "reversal_signal",

        # Implied Volatility
        "call_iv",
        "put_iv",
        "iv_skew",
        "iv_skew_pct",

        # Moneyness
        "moneyness",
        "otm_call",
        "otm_put",

        # Greeks
        "approx_delta_call",
        "approx_delta_put",

        # Composite
        "arb_opportunity_score",
        "arb_opportunity_normalized",
    ]


if __name__ == "__main__":
    # Test the module
    logger.info("Testing options arbitrage detection...")

    # Create sample options data
    sample_data = pd.DataFrame([
        {
            "symbol": "SOFI",
            "underlying_price": 8.50,
            "strike": 8.00,
            "expiry": "2024-03-15",
            "days_to_expiry": 45,
            "call_bid": 0.85,
            "call_ask": 0.90,
            "put_bid": 0.30,
            "put_ask": 0.35,
        },
        {
            "symbol": "SOFI",
            "underlying_price": 8.50,
            "strike": 9.00,
            "expiry": "2024-03-15",
            "days_to_expiry": 45,
            "call_bid": 0.40,
            "call_ask": 0.45,
            "put_bid": 0.85,
            "put_ask": 0.92,
        },
        {
            "symbol": "HOOD",
            "underlying_price": 12.00,
            "strike": 12.00,
            "expiry": "2024-03-15",
            "days_to_expiry": 45,
            "call_bid": 1.10,
            "call_ask": 1.20,
            "put_bid": 1.05,
            "put_ask": 1.15,
        },
    ])

    # Add features
    df_feat = add_options_arbitrage_features(sample_data)
    print("\nOptions arbitrage features:")
    print(df_feat[["symbol", "strike", "parity_violation_pct",
                   "conversion_signal", "reversal_signal", "arb_opportunity_score"]])

    # Scan for opportunities
    underlying = {"SOFI": 8.50, "HOOD": 12.00}
    opps = scan_for_conversion_reversals(sample_data, underlying, min_edge=0.01)

    print(f"\nFound {len(opps)} opportunities:")
    for opp in opps:
        print(f"  {opp.symbol} {opp.strike} {opp.arb_type}: ${opp.pnl_after_costs:.2f}")
