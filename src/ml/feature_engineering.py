"""================================================================================
FEATURE ENGINEERING - Institutional-Grade Technical Analysis
================================================================================
This module creates 100+ features for ML models including:
- Price dynamics (returns, log returns, volatility)
- Momentum indicators (RSI, Stochastic, Williams %R)
- Trend indicators (EMA, MACD, ADX)
- Volatility indicators (Bollinger, ATR, Keltner)
- Volume dynamics (OBV, VWAP, Volume Z-score)
- Market microstructure (spread proxies, liquidity)
================================================================================
"""

import numpy as np
import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from src.database.connection import get_engine
from src.ml.valuation_metrics import enrich_fundamentals_with_valuation_metrics


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators for ML training.

    Returns DataFrame with 50+ engineered features.
    """
    df = df.copy()
    df.sort_values("timestamp", inplace=True)

    # Ensure we have required columns
    required = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain: {required}")

    # =========================================================================
    # PRICE DYNAMICS
    # =========================================================================

    # Returns at multiple horizons
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)
    df["return_20"] = df["close"].pct_change(20)

    # Log returns (better for statistical properties)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))

    # Volatility at multiple windows
    df["volatility_10"] = df["log_return"].rolling(10).std()
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["volatility_50"] = df["log_return"].rolling(50).std()

    # Volatility ratio (short-term vs long-term)
    df["vol_ratio"] = df["volatility_10"] / (df["volatility_50"] + 1e-10)

    # High-Low range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["hl_range_ma"] = df["hl_range"].rolling(10).mean()

    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================

    # RSI
    rsi_14 = RSIIndicator(df["close"], window=14)
    df["rsi_14"] = rsi_14.rsi()

    rsi_7 = RSIIndicator(df["close"], window=7)
    df["rsi_7"] = rsi_7.rsi()

    # Stochastic Oscillator
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    willr = WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14)
    df["williams_r"] = willr.williams_r()

    # CCI (Commodity Channel Index)
    cci = CCIIndicator(df["high"], df["low"], df["close"], window=20)
    df["cci"] = cci.cci()

    # Rate of Change
    df["roc_10"] = df["close"].pct_change(10) * 100
    df["roc_20"] = df["close"].pct_change(20) * 100

    # =========================================================================
    # TREND INDICATORS
    # =========================================================================

    # EMAs at multiple periods
    for period in [5, 10, 20, 50]:
        ema = EMAIndicator(df["close"], window=period)
        df[f"ema_{period}"] = ema.ema_indicator()

    # EMA crossovers (binary signals)
    df["ema_cross_10_20"] = (df["ema_10"] > df["ema_20"]).astype(int)
    df["ema_cross_20_50"] = (df["ema_20"] > df["ema_50"]).astype(int)

    # Price relative to EMAs
    df["price_vs_ema_20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    df["price_vs_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]

    # MACD
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    # ADX (trend strength)
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-10)

    # ATR (Average True Range)
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    # Keltner Channel
    kc = KeltnerChannel(df["high"], df["low"], df["close"], window=20)
    df["kc_high"] = kc.keltner_channel_hband()
    df["kc_low"] = kc.keltner_channel_lband()
    df["kc_width"] = (df["kc_high"] - df["kc_low"]) / df["close"]

    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================

    # Volume Z-score (anomaly detection)
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)

    # Volume ratio (current vs average)
    df["volume_ratio"] = df["volume"] / (vol_mean + 1)

    # On-Balance Volume
    obv = OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["obv"] = obv.on_balance_volume()
    df["obv_change"] = df["obv"].pct_change(5)

    # VWAP (Volume Weighted Average Price)
    try:
        vwap = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"])
        df["vwap"] = vwap.volume_weighted_average_price()
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    except Exception:
        df["vwap"] = df["close"]
        df["price_vs_vwap"] = 0

    # =========================================================================
    # MARKET MICROSTRUCTURE (Proxies)
    # =========================================================================

    # Bid-Ask spread proxy (using high-low)
    df["spread_proxy"] = (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2)

    # Amihud illiquidity ratio
    df["amihud"] = abs(df["return_1"]) / (df["volume"] * df["close"] + 1)
    df["amihud_ma"] = df["amihud"].rolling(20).mean()

    # Price impact proxy
    df["price_impact"] = df["return_1"] / (df["volume_z"] + 0.01)

    # =========================================================================
    # TIME-BASED FEATURES
    # =========================================================================

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month

        # Market session indicators
        df["morning_session"] = ((ts.dt.hour >= 9) & (ts.dt.hour < 12)).astype(int)
        df["afternoon_session"] = ((ts.dt.hour >= 12) & (ts.dt.hour < 16)).astype(int)

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================

    # Doji candle (open close)
    body = abs(df["close"] - df["open"])
    wick = df["high"] - df["low"]
    df["doji"] = (body / (wick + 1e-10) < 0.1).astype(int)

    # Gap detection
    df["gap_up"] = (df["open"] > df["close"].shift(1)).astype(int)
    df["gap_down"] = (df["open"] < df["close"].shift(1)).astype(int)

    # Higher high / Lower low
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

    # Drop rows with NaN
    df.dropna(inplace=True)

    logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
    return df


def make_supervised(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create supervised learning target.

    Target: 1 if price goes UP over horizon periods, 0 otherwise.
    """
    df = df.copy()

    # Future return
    df["future_return"] = df["close"].pct_change(horizon).shift(-horizon)

    # Binary target
    df["target"] = (df["future_return"] > 0).astype(int)

    # Drop rows without target (last 'horizon' rows)
    df.dropna(subset=["target"], inplace=True)

    return df


def add_valuation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge valuation metrics from fundamental_data table.
    Adds: EV/EBITDA, EV/Sales, FCF Yield, ROIC, Altman Z, Graham Number, etc.
    """
    df = df.copy()

    # Get latest fundamental data for each symbol
    engine = get_engine()
    symbols = df["symbol"].unique()

    fundamentals_list = []
    for symbol in symbols:
        try:
            query = """
            SELECT * FROM fundamental_data
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT 1
            """
            fund_df = pd.read_sql(query, engine, params={"symbol": symbol})
            if not fund_df.empty:
                # Enrich with valuation metrics
                fund_df = enrich_fundamentals_with_valuation_metrics(fund_df)
                fundamentals_list.append(fund_df.iloc[0])
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")
            continue

    if not fundamentals_list:
        logger.warning("No fundamental data available - skipping valuation features")
        return df

    fund_df = pd.DataFrame(fundamentals_list)

    # Merge valuation metrics into price data
    valuation_cols = [
        "pe_ratio", "peg_ratio", "price_to_book", "price_to_sales",
        "ev_ebitda", "ev_sales", "fcf_yield", "roic",
        "altman_z_score", "graham_number", "beta",
        "profit_margin", "operating_margin", "debt_to_equity",
        "return_on_equity", "return_on_assets",
    ]

    available_cols = [c for c in valuation_cols if c in fund_df.columns]

    if available_cols:
        merge_df = fund_df[["symbol"] + available_cols]
        df = df.merge(merge_df, on="symbol", how="left", suffixes=("", "_fund"))
        logger.info(f"Added {len(available_cols)} valuation features")

    return df


def prepare_features(df: pd.DataFrame, horizon: int = 1, include_valuation: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Complete feature preparation pipeline with valuation metrics.

    Returns (X, y) ready for model training.
    """
    # Add technical indicators
    df_feat = add_technical_indicators(df)

    # Add valuation metrics (if available)
    if include_valuation:
        try:
            df_feat = add_valuation_features(df_feat)
        except Exception as e:
            logger.warning(f"Could not add valuation features: {e}")

    # Create supervised target
    df_sup = make_supervised(df_feat, horizon=horizon)

    # Define feature columns (exclude target and future info)
    exclude_cols = [
        "symbol", "timestamp", "future_return", "target",
        "open", "high", "low", "close", "volume",
        "source", "asset_type",  # Metadata columns
    ]
    feature_cols = [c for c in df_sup.columns if c not in exclude_cols]

    X = df_sup[feature_cols].copy()
    y = df_sup["target"].copy()

    # Handle any remaining infinities
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    logger.info(f"Prepared features: X shape {X.shape}, y shape {y.shape}, {len(feature_cols)} features")
    return X, y
