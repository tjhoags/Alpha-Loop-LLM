"""================================================================================
RETAIL INEFFICIENCY FEATURE ENGINEERING
================================================================================
Author: Alpha Loop Capital, LLC

Features specifically designed to detect and exploit retail trader inefficiencies
in small/mid cap stocks (<$10B market cap).

Target Inefficiencies:
1. Bad Bid/Ask Spreads - Wide spreads from low liquidity
2. Odd Lot Imbalances - Retail orders < 100 shares
3. Retail Flow Detection - Time-of-day patterns, round numbers
4. Stale Quotes - Market makers not updating
5. Momentum Chasing - Retail piling in late
6. Panic Selling - Retail capitulation signals

These features feed ML models that identify when retail activity creates
exploitable mispricings.
================================================================================
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def add_retail_inefficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that detect retail trading inefficiencies.

    These features are most useful for small/mid cap stocks where
    retail flow has a larger impact on price.

    Args:
    ----
        df: DataFrame with OHLCV data

    Returns:
    -------
        DataFrame with retail inefficiency features added
    """
    df = df.copy()
    df.sort_values("timestamp", inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain: {required}")

    # =========================================================================
    # BID-ASK SPREAD PROXIES
    # =========================================================================

    # Corwin-Schultz spread estimator (from high/low prices)
    # Better than simple (H-L)/C for estimating true spread
    df["beta"] = (np.log(df["high"] / df["low"]) ** 2).rolling(2).sum()
    df["gamma"] = np.log(
        df["high"].rolling(2).max() / df["low"].rolling(2).min(),
    ) ** 2

    # Avoid division issues
    df["alpha"] = (
        (np.sqrt(2 * df["beta"]) - np.sqrt(df["beta"])) /
        (3 - 2 * np.sqrt(2))
    ) - np.sqrt(df["gamma"] / (3 - 2 * np.sqrt(2)))

    df["corwin_schultz_spread"] = (
        2 * (np.exp(df["alpha"]) - 1) / (1 + np.exp(df["alpha"]))
    ).clip(0, 0.2)  # Cap at 20%

    # Simple spread proxy (H-L relative to close)
    df["hl_spread_proxy"] = (df["high"] - df["low"]) / df["close"]

    # Spread vs rolling average (anomaly detection)
    df["spread_ma_20"] = df["hl_spread_proxy"].rolling(20).mean()
    df["spread_z_score"] = (
        (df["hl_spread_proxy"] - df["spread_ma_20"]) /
        df["hl_spread_proxy"].rolling(20).std().replace(0, 1e-10)
    )

    # Wide spread indicator (spread > 2 std above mean)
    df["wide_spread_flag"] = (df["spread_z_score"] > 2).astype(int)

    # =========================================================================
    # RETAIL FLOW INDICATORS
    # =========================================================================

    # Small volume bars (potential retail activity)
    vol_median = df["volume"].rolling(20).median()
    df["small_volume_bar"] = (df["volume"] < vol_median * 0.3).astype(int)

    # Volume clustering at round prices (retail behavior)
    df["close_cents"] = (df["close"] * 100) % 100
    df["round_price"] = (
        (df["close_cents"] < 5) |
        (df["close_cents"] > 95) |
        (abs(df["close_cents"] - 25) < 5) |
        (abs(df["close_cents"] - 50) < 5) |
        (abs(df["close_cents"] - 75) < 5)
    ).astype(int)

    # Time-based retail patterns
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])

        # Retail active hours (market open, lunch, close)
        df["market_open_30min"] = (
            (ts.dt.hour == 9) & (ts.dt.minute < 30)
        ).astype(int)
        df["lunch_hour"] = (
            (ts.dt.hour >= 12) & (ts.dt.hour < 13)
        ).astype(int)
        df["market_close_30min"] = (
            (ts.dt.hour == 15) & (ts.dt.minute >= 30)
        ).astype(int)

        # Retail tends to trade more at open/close
        df["retail_active_period"] = (
            df["market_open_30min"] | df["market_close_30min"]
        ).astype(int)

    # =========================================================================
    # ODD LOT INDICATORS (proxy via volume patterns)
    # =========================================================================

    # Volume not divisible by 100 suggests odd lots
    # We can't see this directly, but low volume + high price impact suggests it
    df["price_impact"] = abs(df["close"].pct_change()) / (
        df["volume"] / df["volume"].rolling(20).mean() + 0.01
    )

    # High impact on low volume = likely retail
    df["high_retail_impact"] = (
        (df["price_impact"] > df["price_impact"].rolling(20).quantile(0.9)) &
        (df["volume"] < vol_median)
    ).astype(int)

    # =========================================================================
    # STALE QUOTE DETECTION
    # =========================================================================

    # Price not moving despite volume (stale quotes)
    df["price_change"] = df["close"].pct_change().abs()
    df["vol_change"] = df["volume"].pct_change().abs()

    df["stale_quote_signal"] = (
        (df["price_change"] < 0.001) &  # Price barely moved
        (df["volume"] > vol_median * 0.5)  # Decent volume
    ).astype(int)

    # Multiple stale bars in a row
    df["stale_quote_streak"] = df["stale_quote_signal"].rolling(3).sum()

    # =========================================================================
    # MOMENTUM CHASING INDICATORS
    # =========================================================================

    # Strong price move + increasing volume = retail chasing
    df["return_5"] = df["close"].pct_change(5)
    df["volume_trend_5"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

    df["momentum_chase_long"] = (
        (df["return_5"] > 0.05) &  # Up 5%+ in 5 bars
        (df["volume_trend_5"] > 1.5)  # Volume 50%+ above average
    ).astype(int)

    df["momentum_chase_short"] = (
        (df["return_5"] < -0.05) &  # Down 5%+ in 5 bars
        (df["volume_trend_5"] > 1.5)  # Volume spike on down move
    ).astype(int)

    # Late momentum entry (price already extended)
    df["price_extension"] = (df["close"] - df["close"].rolling(20).mean()) / (
        df["close"].rolling(20).std() + 1e-10
    )
    df["overextended"] = (abs(df["price_extension"]) > 2).astype(int)

    # =========================================================================
    # PANIC SELLING / CAPITULATION
    # =========================================================================

    # High volume down bar after extended decline
    df["return_10"] = df["close"].pct_change(10)
    df["capitulation_signal"] = (
        (df["return_10"] < -0.10) &  # Down 10%+ over 10 bars
        (df["volume"] > df["volume"].rolling(20).mean() * 2) &  # 2x volume
        (df["close"] < df["open"])  # Down bar
    ).astype(int)

    # Reversal setup after capitulation
    df["capitulation_reversal"] = (
        df["capitulation_signal"].shift(1) == 1
    ).astype(int)

    # =========================================================================
    # LIQUIDITY VACUUM DETECTION
    # =========================================================================

    # Amihud illiquidity (price impact per dollar volume)
    dollar_volume = df["close"] * df["volume"]
    df["amihud_illiquidity"] = (
        abs(df["close"].pct_change()) / (dollar_volume + 1)
    ) * 1e6  # Scale up

    df["amihud_ma_10"] = df["amihud_illiquidity"].rolling(10).mean()
    df["amihud_spike"] = (
        df["amihud_illiquidity"] > df["amihud_ma_10"] * 2
    ).astype(int)

    # Liquidity drought (very low volume + wide spread)
    df["liquidity_vacuum"] = (
        (df["volume"] < vol_median * 0.2) &
        (df["hl_spread_proxy"] > df["spread_ma_20"] * 1.5)
    ).astype(int)

    # =========================================================================
    # RETAIL SENTIMENT PROXIES
    # =========================================================================

    # Gap and trap (gap up/down that reverses)
    df["gap_up"] = (df["open"] > df["high"].shift(1)).astype(int)
    df["gap_down"] = (df["open"] < df["low"].shift(1)).astype(int)

    df["gap_up_failed"] = (
        (df["gap_up"] == 1) &
        (df["close"] < df["open"])  # Closed red
    ).astype(int)

    df["gap_down_failed"] = (
        (df["gap_down"] == 1) &
        (df["close"] > df["open"])  # Closed green
    ).astype(int)

    # Multiple down bars (retail panic)
    df["down_bar"] = (df["close"] < df["open"]).astype(int)
    df["consecutive_down"] = df["down_bar"].rolling(5).sum()

    # =========================================================================
    # COMPOSITE RETAIL INEFFICIENCY SCORE
    # =========================================================================

    # Aggregate signal: higher = more retail inefficiency present
    df["retail_inefficiency_score"] = (
        df["wide_spread_flag"] * 2 +
        df["high_retail_impact"] * 2 +
        df["stale_quote_signal"] * 1 +
        df["momentum_chase_long"] * 1.5 +
        df["momentum_chase_short"] * 1.5 +
        df["capitulation_signal"] * 3 +
        df["liquidity_vacuum"] * 2 +
        df["gap_up_failed"] * 1.5 +
        df["gap_down_failed"] * 1.5
    )

    # Normalize to 0-1 range
    score_max = df["retail_inefficiency_score"].rolling(100).max()
    df["retail_inefficiency_normalized"] = (
        df["retail_inefficiency_score"] / (score_max + 1)
    ).clip(0, 1)

    # Clean up intermediate columns
    drop_cols = ["beta", "gamma", "alpha", "close_cents"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Handle NaN
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)

    logger.info(f"Added retail inefficiency features: {len(df)} rows")
    return df


def get_retail_inefficiency_feature_names() -> List[str]:
    """Get list of all retail inefficiency feature names."""
    return [
        # Spread features
        "corwin_schultz_spread",
        "hl_spread_proxy",
        "spread_ma_20",
        "spread_z_score",
        "wide_spread_flag",

        # Retail flow
        "small_volume_bar",
        "round_price",
        "market_open_30min",
        "lunch_hour",
        "market_close_30min",
        "retail_active_period",

        # Odd lot proxies
        "price_impact",
        "high_retail_impact",

        # Stale quotes
        "stale_quote_signal",
        "stale_quote_streak",

        # Momentum chasing
        "momentum_chase_long",
        "momentum_chase_short",
        "price_extension",
        "overextended",

        # Capitulation
        "capitulation_signal",
        "capitulation_reversal",

        # Liquidity
        "amihud_illiquidity",
        "amihud_ma_10",
        "amihud_spike",
        "liquidity_vacuum",

        # Sentiment
        "gap_up",
        "gap_down",
        "gap_up_failed",
        "gap_down_failed",
        "consecutive_down",

        # Composite
        "retail_inefficiency_score",
        "retail_inefficiency_normalized",
    ]


def create_retail_arbitrage_target(
    df: pd.DataFrame,
    horizon: int = 5,
    min_return: float = 0.01,
) -> pd.DataFrame:
    """Create target variable for retail arbitrage model.

    Target = 1 if:
    - Retail inefficiency is high AND
    - Price moves favorably over horizon

    This targets situations where retail activity creates
    temporary mispricings we can exploit.

    Args:
    ----
        df: DataFrame with retail inefficiency features
        horizon: Forward bars to measure return
        min_return: Minimum return to consider profitable

    Returns:
    -------
        DataFrame with target column added
    """
    df = df.copy()

    # Future return
    df["future_return"] = df["close"].pct_change(horizon).shift(-horizon)

    # Profitable long opportunity (high inefficiency + price goes up)
    df["target_long"] = (
        (df["retail_inefficiency_normalized"] > 0.5) &
        (df["future_return"] > min_return)
    ).astype(int)

    # Profitable short opportunity (overextension + price reverts)
    df["target_short"] = (
        (df["overextended"] == 1) &
        (df["future_return"] < -min_return)
    ).astype(int)

    # Combined target: 1 = long opportunity, -1 = short, 0 = no trade
    df["target"] = np.where(
        df["target_long"] == 1, 1,
        np.where(df["target_short"] == 1, -1, 0),
    )

    # Binary target for simple classification
    df["target_binary"] = (df["future_return"] > 0).astype(int)

    df.dropna(subset=["future_return"], inplace=True)

    return df


def filter_small_mid_cap_data(
    df: pd.DataFrame,
    symbols: List[str] = None,
    max_market_cap_bn: float = 10.0,
) -> pd.DataFrame:
    """Filter data to small/mid cap stocks.

    Args:
    ----
        df: DataFrame with symbol column
        symbols: Specific symbols to include (overrides market cap filter)
        max_market_cap_bn: Maximum market cap in billions

    Returns:
    -------
        Filtered DataFrame
    """
    if symbols:
        return df[df["symbol"].isin(symbols)]

    # Default small/mid cap symbols
    small_mid_symbols = [
        # High retail activity small caps
        "AFRM", "UPST", "SOFI", "HOOD", "COIN", "RBLX", "DKNG",
        "RIVN", "LCID", "IONQ", "JOBY", "LILM", "EVGO", "QS",
        "PLTR", "PATH", "CFLT", "S", "DOCN", "GTLB", "ESTC",

        # Meme stocks (high retail)
        "GME", "AMC", "BBBY", "KOSS", "EXPR", "BB",

        # Small cap energy
        "UEC", "DNN", "NXE", "UUUU", "SMR", "LEU", "CCJ",

        # Small cap biotech
        "RXRX", "DNA", "BEAM", "CRSP", "NTLA", "VERV",

        # Small cap fintech
        "LC", "OPEN", "RDFN", "UWMC", "RKT",

        # Small cap EV
        "FSR", "GOEV", "ARVL", "REE", "WKHS",
    ]

    if "symbol" in df.columns:
        return df[df["symbol"].isin(small_mid_symbols)]

    return df


if __name__ == "__main__":
    # Test feature engineering
    import yfinance as yf

    logger.info("Testing retail inefficiency features...")

    # Get sample data
    ticker = yf.Ticker("SOFI")
    df = ticker.history(period="6mo", interval="1h")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "timestamp", "datetime": "timestamp"})

    # Add features
    df_feat = add_retail_inefficiency_features(df)

    print(f"\nFeatures added: {len(get_retail_inefficiency_feature_names())}")
    print("Sample inefficiency scores:")
    print(df_feat[["timestamp", "close", "retail_inefficiency_normalized"]].tail(10))

    # Create targets
    df_target = create_retail_arbitrage_target(df_feat)
    print("\nTarget distribution:")
    print(df_target["target"].value_counts())
