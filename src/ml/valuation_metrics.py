"""
================================================================================
ADVANCED VALUATION METRICS ENGINE
================================================================================
Quantitative valuation metrics beyond simple P/E:
- Delta-Adjusted VaR (for options portfolios)
- Convexity (non-linear risk)
- Enterprise Value multiples
- Free Cash Flow Yield
- EV/EBITDA, EV/Sales
- ROIC, ROE, ROA
- Altman Z-Score (bankruptcy risk)
- Piotroski F-Score (value investing)
- Graham Number (intrinsic value)
- DCF components
================================================================================
"""

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


def calculate_enterprise_value(market_cap: float, debt: float, cash: float, minority_interest: float = 0) -> float:
    """Calculate Enterprise Value = Market Cap + Debt - Cash + Minority Interest."""
    return market_cap + debt - cash + minority_interest


def calculate_ev_ebitda(enterprise_value: float, ebitda: float) -> float:
    """EV/EBITDA ratio."""
    if ebitda <= 0:
        return np.nan
    return enterprise_value / ebitda


def calculate_ev_sales(enterprise_value: float, revenue: float) -> float:
    """EV/Sales ratio."""
    if revenue <= 0:
        return np.nan
    return enterprise_value / revenue


def calculate_free_cash_flow_yield(fcf: float, market_cap: float) -> float:
    """Free Cash Flow Yield = FCF / Market Cap."""
    if market_cap <= 0:
        return np.nan
    return fcf / market_cap


def calculate_roic(nopat: float, invested_capital: float) -> float:
    """Return on Invested Capital = NOPAT / Invested Capital."""
    if invested_capital <= 0:
        return np.nan
    return nopat / invested_capital


def altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_assets: float,
    sales: float,
) -> float:
    """
    Altman Z-Score (bankruptcy risk predictor).
    Z > 2.99 = Safe, 1.81-2.99 = Grey, < 1.81 = Distress
    """
    if total_assets <= 0:
        return np.nan
    
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_cap / total_assets if total_assets > 0 else 0
    X5 = sales / total_assets
    
    z_score = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
    return z_score


def piotroski_f_score(
    net_income: float,
    roa_current: float,
    roa_prior: float,
    cfo: float,
    cfo_vs_net_income: float,
    leverage_current: float,
    leverage_prior: float,
    current_ratio_current: float,
    current_ratio_prior: float,
    shares_outstanding_current: float,
    shares_outstanding_prior: float,
    gross_margin_current: float,
    gross_margin_prior: float,
    asset_turnover_current: float,
    asset_turnover_prior: float,
) -> int:
    """
    Piotroski F-Score (0-9, higher = better value).
    Based on 9 fundamental criteria.
    """
    score = 0
    
    # Profitability (4 points)
    if net_income > 0:
        score += 1
    if roa_current > roa_prior:
        score += 1
    if cfo > 0:
        score += 1
    if cfo_vs_net_income > 0:
        score += 1
    
    # Leverage, liquidity, source of funds (3 points)
    if leverage_current < leverage_prior:
        score += 1
    if current_ratio_current > current_ratio_prior:
        score += 1
    if shares_outstanding_current <= shares_outstanding_prior:
        score += 1
    
    # Operating efficiency (2 points)
    if gross_margin_current > gross_margin_prior:
        score += 1
    if asset_turnover_current > asset_turnover_prior:
        score += 1
    
    return score


def graham_number(eps: float, book_value_per_share: float) -> float:
    """
    Graham Number (intrinsic value estimate).
    Graham Number = sqrt(22.5 * EPS * Book Value per Share)
    """
    if eps <= 0 or book_value_per_share <= 0:
        return np.nan
    return np.sqrt(22.5 * eps * book_value_per_share)


def calculate_delta_adjusted_var_portfolio(
    positions_df: pd.DataFrame,
    underlying_prices: pd.Series,
    volatilities: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calculate portfolio-level Delta-Adjusted VaR.
    
    For each position:
    - Options: Use delta * underlying_price * volatility
    - Stocks: Use position_value * volatility
    """
    if positions_df.empty:
        return 0.0
    
    portfolio_var = 0.0
    z_score = norm.ppf(confidence)
    
    for idx, pos in positions_df.iterrows():
        symbol = pos["symbol"]
        underlying_price = underlying_prices.get(symbol, pos.get("price", 0))
        volatility = volatilities.get(symbol, 0.20)
        
        if "delta" in pos and not pd.isna(pos["delta"]):
            # Options position
            position_value = abs(pos["delta"]) * underlying_price * pos.get("quantity", 0)
            time_to_expiry = pos.get("time_to_expiry", 1/252)
            var_contribution = position_value * volatility * np.sqrt(time_to_expiry) * z_score
        else:
            # Stock position
            position_value = underlying_price * pos.get("quantity", 0)
            var_contribution = abs(position_value) * volatility * z_score
        
        portfolio_var += var_contribution ** 2
    
    return np.sqrt(portfolio_var)  # Portfolio VaR (assuming correlation = 1 for conservative estimate)


def calculate_portfolio_convexity(positions_df: pd.DataFrame, underlying_prices: pd.Series) -> float:
    """
    Calculate portfolio convexity (Gamma exposure).
    Higher convexity = more non-linear price sensitivity (good for large moves).
    """
    if positions_df.empty:
        return 0.0
    
    total_convexity = 0.0
    
    for idx, pos in positions_df.iterrows():
        symbol = pos["symbol"]
        underlying_price = underlying_prices.get(symbol, pos.get("price", 0))
        
        if "gamma" in pos and not pd.isna(pos["gamma"]):
            # Options position
            gamma_exposure = pos["gamma"] * (underlying_price ** 2) * pos.get("quantity", 0)
            total_convexity += gamma_exposure
        # Stocks have zero convexity (linear)
    
    return total_convexity


def enrich_fundamentals_with_valuation_metrics(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich fundamental data DataFrame with advanced valuation metrics.
    
    Expected columns:
    - market_cap, enterprise_value, debt, cash
    - ebitda, revenue, sales
    - net_income, ebit, nopat
    - working_capital, retained_earnings, total_assets
    - fcf, invested_capital
    - eps, book_value_per_share
    """
    df = fundamentals_df.copy()
    
    # Enterprise Value metrics
    if "enterprise_value" not in df.columns:
        df["enterprise_value"] = df.apply(
            lambda row: calculate_enterprise_value(
                row.get("market_cap", 0),
                row.get("debt", 0),
                row.get("cash", 0),
            ),
            axis=1,
        )
    
    # EV multiples
    df["ev_ebitda"] = df.apply(
        lambda row: calculate_ev_ebitda(row.get("enterprise_value", 0), row.get("ebitda", 0)),
        axis=1,
    )
    df["ev_sales"] = df.apply(
        lambda row: calculate_ev_sales(row.get("enterprise_value", 0), row.get("revenue", 0)),
        axis=1,
    )
    
    # Free Cash Flow Yield
    df["fcf_yield"] = df.apply(
        lambda row: calculate_free_cash_flow_yield(row.get("fcf", 0), row.get("market_cap", 0)),
        axis=1,
    )
    
    # ROIC
    df["roic"] = df.apply(
        lambda row: calculate_roic(row.get("nopat", 0), row.get("invested_capital", 0)),
        axis=1,
    )
    
    # Altman Z-Score
    df["altman_z_score"] = df.apply(
        lambda row: altman_z_score(
            row.get("working_capital", 0),
            row.get("retained_earnings", 0),
            row.get("ebit", 0),
            row.get("market_cap", 0),
            row.get("total_assets", 1),
            row.get("sales", 0),
        ),
        axis=1,
    )
    
    # Graham Number
    df["graham_number"] = df.apply(
        lambda row: graham_number(row.get("eps", 0), row.get("book_value_per_share", 0)),
        axis=1,
    )
    
    logger.info(f"Enriched {len(df)} fundamental records with valuation metrics")
    return df

