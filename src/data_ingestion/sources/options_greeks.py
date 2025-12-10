"""================================================================================
OPTIONS DATA WITH GREEKS
================================================================================
Fetches options chain data and calculates Greeks:
- Delta (price sensitivity)
- Gamma (delta sensitivity)
- Theta (time decay)
- Vega (volatility sensitivity)
- Rho (interest rate sensitivity)
- Implied Volatility
- Delta-Adjusted VaR
- Convexity (for options portfolios)
================================================================================
"""

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes option pricing."""
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict:
    """Calculate all Greeks for an option."""
    if T <= 0:
        return {
            "delta": 1.0 if (option_type == "call" and S > K) else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta (per day, negative for time decay)
    if option_type == "call":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% rate change
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    """Calculate implied volatility from market price."""
    def price_diff(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(price_diff, 0.001, 5.0)
        return iv
    except:
        return np.nan


def delta_adjusted_var(positions: pd.DataFrame, underlying_price: float, volatility: float, confidence: float = 0.95) -> float:
    """Calculate Delta-Adjusted Value at Risk for options portfolio.

    VaR = |Delta| * S * sigma * sqrt(T) * Z(confidence)
    """
    if positions.empty:
        return 0.0

    # Calculate portfolio delta
    portfolio_delta = (positions["delta"] * positions["quantity"]).sum()

    # Average time to expiration
    avg_T = positions["time_to_expiry"].mean() if "time_to_expiry" in positions.columns else 1/252

    # Z-score for confidence level
    z_score = norm.ppf(confidence)

    # Delta-adjusted VaR
    var = abs(portfolio_delta) * underlying_price * volatility * np.sqrt(avg_T) * z_score

    return var


def portfolio_convexity(positions: pd.DataFrame, underlying_price: float) -> float:
    """Calculate portfolio convexity (Gamma exposure).
    Higher convexity = more non-linear price sensitivity.
    """
    if positions.empty:
        return 0.0

    portfolio_gamma = (positions["gamma"] * positions["quantity"] * underlying_price ** 2).sum()
    return portfolio_gamma


def fetch_options_chain(symbol: str, expiration_date: str = None) -> pd.DataFrame:
    """Fetch options chain data (if available via API).
    For now, this is a placeholder - you may need to use IBKR or other sources.
    """
    # Alpha Vantage doesn't have options chain API
    # You'll need to integrate with IBKR or another options data provider
    logger.warning(f"Options chain fetching not yet implemented for {symbol}")
    return pd.DataFrame()


def enrich_options_with_greeks(options_df: pd.DataFrame, underlying_price: float, risk_free_rate: float = 0.05) -> pd.DataFrame:
    """Enrich options DataFrame with Greeks."""
    if options_df.empty:
        return options_df

    df = options_df.copy()

    # Calculate time to expiration (assume expiration_date column exists)
    if "expiration_date" in df.columns:
        df["time_to_expiry"] = (pd.to_datetime(df["expiration_date"]) - pd.Timestamp.now()).dt.days / 365.0
    else:
        df["time_to_expiry"] = 30 / 365.0  # Default 30 days

    # Get implied volatility if available, otherwise use historical
    if "implied_volatility" not in df.columns:
        df["implied_volatility"] = 0.20  # Default 20% IV

    # Calculate Greeks for each option
    greeks_list = []
    for idx, row in df.iterrows():
        greeks = calculate_greeks(
            S=underlying_price,
            K=row["strike"],
            T=row["time_to_expiry"],
            r=risk_free_rate,
            sigma=row["implied_volatility"],
            option_type=row.get("option_type", "call").lower(),
        )
        greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list)
    df = pd.concat([df, greeks_df], axis=1)

    logger.info(f"Enriched {len(df)} options with Greeks")
    return df

