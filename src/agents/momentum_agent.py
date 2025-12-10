"""================================================================================
MOMENTUM AGENT
================================================================================
Trend-following trading agent that captures momentum effects:

1. Price Momentum: Stocks that went up continue to go up (3-12 month horizon)
2. Earnings Momentum: Post-earnings drift
3. Volume Momentum: Price moves confirmed by volume
4. Breakout Detection: New highs/lows with momentum
5. Cross-Sectional Momentum: Relative strength vs sector/market

Academic Basis:
- Jegadeesh & Titman (1993): Returns to Buying Winners and Selling Losers
- Carhart (1997): Four-factor model including momentum
- Asness et al. (2013): Value and Momentum Everywhere

Strategy:
- BUY: Strong uptrend with acceleration
- SELL: Trend reversal signals
- POSITION SIZE: Based on momentum strength and confidence

================================================================================
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class MomentumAgent(BaseAgent):
    """Momentum-based trading agent.

    Captures trend-following alpha through:
    - Price momentum (various lookbacks)
    - Momentum acceleration/deceleration
    - Volume confirmation
    - Relative strength
    - Trend strength indicators
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "momentum"
        config.use_momentum_features = True
        config.use_volume_features = True

        super().__init__(config)

        # Momentum-specific parameters
        self.short_lookback = 5
        self.medium_lookback = 20
        self.long_lookback = 60
        self.very_long_lookback = 252  # 1 year

        logger.info(f"MomentumAgent initialized with lookbacks: {self.short_lookback}/{self.medium_lookback}/{self.long_lookback}/{self.very_long_lookback}")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-specific features.

        Feature Categories:
        1. Price Momentum (various horizons)
        2. Momentum Acceleration
        3. Moving Average Crossovers
        4. Trend Strength (ADX)
        5. Volume Confirmation
        6. Relative Strength
        7. Breakout Signals
        """
        features = pd.DataFrame(index=df.index)

        # =====================================================================
        # 1. PRICE MOMENTUM (Core momentum features)
        # =====================================================================

        # Simple returns over various horizons
        features["mom_1d"] = df["close"].pct_change(1)
        features["mom_5d"] = df["close"].pct_change(5)
        features["mom_10d"] = df["close"].pct_change(10)
        features["mom_20d"] = df["close"].pct_change(20)
        features["mom_60d"] = df["close"].pct_change(60)
        features["mom_120d"] = df["close"].pct_change(120)
        features["mom_252d"] = df["close"].pct_change(252)

        # Momentum minus most recent month (skip last month for reversal)
        features["mom_12_1"] = df["close"].pct_change(252) - df["close"].pct_change(21)

        # Log momentum (more stable for large moves)
        features["log_mom_20d"] = np.log(df["close"] / df["close"].shift(20))
        features["log_mom_60d"] = np.log(df["close"] / df["close"].shift(60))

        # =====================================================================
        # 2. MOMENTUM ACCELERATION (Rate of change of momentum)
        # =====================================================================

        # Momentum of momentum
        mom_20 = df["close"].pct_change(20)
        features["mom_accel_20d"] = mom_20 - mom_20.shift(20)

        mom_60 = df["close"].pct_change(60)
        features["mom_accel_60d"] = mom_60 - mom_60.shift(60)

        # First and second derivatives of price
        features["price_velocity"] = df["close"].diff(5) / 5
        features["price_acceleration"] = features["price_velocity"].diff(5) / 5

        # Momentum consistency (how often positive over lookback)
        features["mom_consistency_20d"] = (df["close"].pct_change(1) > 0).rolling(20).mean()
        features["mom_consistency_60d"] = (df["close"].pct_change(1) > 0).rolling(60).mean()

        # =====================================================================
        # 3. MOVING AVERAGE CROSSOVERS
        # =====================================================================

        # Simple moving averages
        features["sma_5"] = df["close"].rolling(5).mean()
        features["sma_10"] = df["close"].rolling(10).mean()
        features["sma_20"] = df["close"].rolling(20).mean()
        features["sma_50"] = df["close"].rolling(50).mean()
        features["sma_200"] = df["close"].rolling(200).mean()

        # Exponential moving averages
        features["ema_12"] = df["close"].ewm(span=12).mean()
        features["ema_26"] = df["close"].ewm(span=26).mean()

        # Price relative to moving averages
        features["price_to_sma_20"] = df["close"] / features["sma_20"] - 1
        features["price_to_sma_50"] = df["close"] / features["sma_50"] - 1
        features["price_to_sma_200"] = df["close"] / features["sma_200"] - 1

        # MA crossover signals
        features["sma_5_20_cross"] = (features["sma_5"] > features["sma_20"]).astype(int)
        features["sma_20_50_cross"] = (features["sma_20"] > features["sma_50"]).astype(int)
        features["sma_50_200_cross"] = (features["sma_50"] > features["sma_200"]).astype(int)

        # Golden cross / death cross distance
        features["golden_cross_dist"] = (features["sma_50"] - features["sma_200"]) / features["sma_200"]

        # MACD
        features["macd"] = features["ema_12"] - features["ema_26"]
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]
        features["macd_cross"] = (features["macd"] > features["macd_signal"]).astype(int)

        # =====================================================================
        # 4. TREND STRENGTH (ADX and related)
        # =====================================================================

        # True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Average True Range
        features["atr_14"] = tr.rolling(14).mean()
        features["atr_pct"] = features["atr_14"] / df["close"]

        # Directional Movement
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr_14 = tr.rolling(14).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / atr_14
        minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / atr_14

        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        features["di_diff"] = plus_di - minus_di

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        features["adx"] = pd.Series(dx).rolling(14).mean()

        # Trend strength classification
        features["strong_trend"] = (features["adx"] > 25).astype(int)
        features["trend_direction"] = np.sign(features["di_diff"])

        # =====================================================================
        # 5. VOLUME CONFIRMATION
        # =====================================================================

        # Volume moving averages
        features["vol_sma_20"] = df["volume"].rolling(20).mean()
        features["vol_ratio"] = df["volume"] / features["vol_sma_20"]

        # Volume momentum
        features["vol_mom_10d"] = df["volume"].pct_change(10)

        # On-Balance Volume
        obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        features["obv"] = obv
        features["obv_sma_20"] = obv.rolling(20).mean()
        features["obv_trend"] = (obv > features["obv_sma_20"]).astype(int)

        # Volume-weighted momentum
        features["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        features["price_to_vwap"] = df["close"] / features["vwap"] - 1

        # Volume on up days vs down days
        up_vol = df["volume"].where(df["close"] > df["close"].shift(1), 0)
        down_vol = df["volume"].where(df["close"] < df["close"].shift(1), 0)
        features["up_down_vol_ratio"] = up_vol.rolling(20).sum() / (down_vol.rolling(20).sum() + 1)

        # =====================================================================
        # 6. RELATIVE STRENGTH
        # =====================================================================

        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # RSI divergence from price
        features["rsi_mom"] = features["rsi_14"] - features["rsi_14"].shift(14)
        features["rsi_overbought"] = (features["rsi_14"] > 70).astype(int)
        features["rsi_oversold"] = (features["rsi_14"] < 30).astype(int)

        # Rate of Change
        features["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
        features["roc_20"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20) * 100

        # =====================================================================
        # 7. BREAKOUT SIGNALS
        # =====================================================================

        # Donchian Channel
        features["donchian_high_20"] = df["high"].rolling(20).max()
        features["donchian_low_20"] = df["low"].rolling(20).min()
        features["donchian_mid"] = (features["donchian_high_20"] + features["donchian_low_20"]) / 2

        # Breakout signals
        features["new_high_20"] = (df["high"] >= features["donchian_high_20"]).astype(int)
        features["new_low_20"] = (df["low"] <= features["donchian_low_20"]).astype(int)

        # 52-week high/low
        features["high_252"] = df["high"].rolling(252).max()
        features["low_252"] = df["low"].rolling(252).min()
        features["pct_from_high_252"] = (df["close"] - features["high_252"]) / features["high_252"]
        features["pct_from_low_252"] = (df["close"] - features["low_252"]) / features["low_252"]
        features["near_52w_high"] = (features["pct_from_high_252"] > -0.05).astype(int)

        # Bollinger Bands
        bb_sma = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        features["bb_upper"] = bb_sma + 2 * bb_std
        features["bb_lower"] = bb_sma - 2 * bb_std
        features["bb_pct"] = (df["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"] + 1e-10)
        features["bb_breakout_up"] = (df["close"] > features["bb_upper"]).astype(int)
        features["bb_breakout_down"] = (df["close"] < features["bb_lower"]).astype(int)

        # =====================================================================
        # 8. VOLATILITY FEATURES
        # =====================================================================

        # Historical volatility
        features["hvol_10"] = features["mom_1d"].rolling(10).std() * np.sqrt(252)
        features["hvol_20"] = features["mom_1d"].rolling(20).std() * np.sqrt(252)
        features["hvol_60"] = features["mom_1d"].rolling(60).std() * np.sqrt(252)

        # Volatility ratio (short vs long)
        features["vol_ratio_10_60"] = features["hvol_10"] / (features["hvol_60"] + 1e-10)

        # Parkinson volatility (high-low)
        features["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean(),
        ) * np.sqrt(252)

        # =====================================================================
        # 9. PATTERN RECOGNITION
        # =====================================================================

        # Higher highs and higher lows (uptrend)
        features["higher_high"] = (df["high"] > df["high"].shift(1)).rolling(5).sum()
        features["higher_low"] = (df["low"] > df["low"].shift(1)).rolling(5).sum()
        features["uptrend_score"] = features["higher_high"] + features["higher_low"]

        # Lower highs and lower lows (downtrend)
        features["lower_high"] = (df["high"] < df["high"].shift(1)).rolling(5).sum()
        features["lower_low"] = (df["low"] < df["low"].shift(1)).rolling(5).sum()
        features["downtrend_score"] = features["lower_high"] + features["lower_low"]

        # Trend score
        features["trend_score"] = features["uptrend_score"] - features["downtrend_score"]

        # Drop intermediate columns
        cols_to_drop = ["sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                        "ema_12", "ema_26", "vol_sma_20", "obv", "obv_sma_20",
                        "donchian_high_20", "donchian_low_20", "donchian_mid",
                        "high_252", "low_252", "bb_upper", "bb_lower"]
        features = features.drop(columns=[c for c in cols_to_drop if c in features.columns], errors="ignore")

        return features

    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """Generate prediction from features.

        Returns
        -------
            (direction, confidence) tuple
        """
        if self.model is None:
            # Fallback to rule-based if no model
            return self._rule_based_predict(features)

        try:
            # Scale features
            feature_cols = [c for c in self.feature_names if c in features.columns]
            X = features[feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)

            # Get probability
            proba = self.model.predict_proba(X_scaled)[0]

            # Class 1 = up, Class 0 = down
            up_prob = proba[1]

            if up_prob > 0.5 + (self.config.confidence_threshold - 0.5):
                return ("BUY", up_prob)
            elif up_prob < 0.5 - (self.config.confidence_threshold - 0.5):
                return ("SELL", 1 - up_prob)
            else:
                return ("HOLD", 0.5)

        except Exception as e:
            logger.warning(f"Prediction error: {e}, falling back to rule-based")
            return self._rule_based_predict(features)

    def _rule_based_predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """Simple rule-based prediction when model not available."""
        try:
            row = features.iloc[-1] if len(features) > 0 else features

            score = 0.0

            # Momentum signals
            if "mom_20d" in row and row["mom_20d"] > 0.05:
                score += 0.2
            if "mom_60d" in row and row["mom_60d"] > 0.10:
                score += 0.15

            # Trend signals
            if "sma_50_200_cross" in row and row["sma_50_200_cross"] == 1:
                score += 0.15
            if "adx" in row and row["adx"] > 25:
                score += 0.1

            # RSI signals
            if "rsi_14" in row:
                if row["rsi_14"] > 50 and row["rsi_14"] < 70:
                    score += 0.1
                elif row["rsi_14"] < 30:
                    score -= 0.2

            # Volume confirmation
            if "vol_ratio" in row and row["vol_ratio"] > 1.5:
                score += 0.1

            confidence = 0.5 + score
            confidence = max(0.0, min(1.0, confidence))

            if confidence > 0.6:
                return ("BUY", confidence)
            elif confidence < 0.4:
                return ("SELL", 1 - confidence)
            else:
                return ("HOLD", 0.5)

        except Exception as e:
            logger.error(f"Rule-based prediction error: {e}")
            return ("HOLD", 0.5)

    def get_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate overall momentum score for a symbol.

        Returns score from -1 (strong downtrend) to +1 (strong uptrend)
        """
        score = 0.0

        try:
            # Short-term momentum
            if len(df) >= 5:
                mom_5 = df["close"].pct_change(5).iloc[-1]
                score += np.clip(mom_5 * 5, -0.2, 0.2)

            # Medium-term momentum
            if len(df) >= 20:
                mom_20 = df["close"].pct_change(20).iloc[-1]
                score += np.clip(mom_20 * 2, -0.3, 0.3)

            # Long-term momentum
            if len(df) >= 60:
                mom_60 = df["close"].pct_change(60).iloc[-1]
                score += np.clip(mom_60, -0.3, 0.3)

            # Trend
            if len(df) >= 50:
                sma_20 = df["close"].rolling(20).mean().iloc[-1]
                sma_50 = df["close"].rolling(50).mean().iloc[-1]
                if sma_20 > sma_50:
                    score += 0.2
                else:
                    score -= 0.2

        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")

        return np.clip(score, -1.0, 1.0)


