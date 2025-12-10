"""================================================================================
MEAN REVERSION AGENT
================================================================================
Statistical arbitrage agent that profits from prices reverting to mean:

1. Price Mean Reversion: Oversold/overbought conditions
2. Pair Trading: Relative value between correlated assets
3. Z-Score Trading: Standard deviation from mean
4. Bollinger Band Reversals: Touches bands then reverts
5. RSI Extremes: Overbought/oversold reversals

Academic Basis:
- De Bondt & Thaler (1985): Overreaction hypothesis
- Lo & MacKinlay (1990): When are contrarian profits due to stock overreaction?
- Poterba & Summers (1988): Mean reversion in stock prices

================================================================================
"""

from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class MeanReversionAgent(BaseAgent):
    """Mean reversion trading agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "mean_reversion"
        config.use_mean_reversion_features = True

        super().__init__(config)

        # Mean reversion specific parameters
        self.zscore_entry_threshold = 2.0
        self.zscore_exit_threshold = 0.5
        self.lookback_period = 20

        logger.info("MeanReversionAgent initialized")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion features."""
        features = pd.DataFrame(index=df.index)

        # =====================================================================
        # Z-SCORE FEATURES
        # =====================================================================

        for period in [10, 20, 50]:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            features[f"zscore_{period}"] = (df["close"] - sma) / (std + 1e-10)

        # Z-score of returns
        ret = df["close"].pct_change()
        for period in [10, 20, 50]:
            features[f"ret_zscore_{period}"] = (ret - ret.rolling(period).mean()) / (ret.rolling(period).std() + 1e-10)

        # =====================================================================
        # BOLLINGER BAND FEATURES
        # =====================================================================

        for period in [20, 50]:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            upper = sma + 2 * std
            lower = sma - 2 * std

            features[f"bb_pct_{period}"] = (df["close"] - lower) / (upper - lower + 1e-10)
            features[f"bb_width_{period}"] = (upper - lower) / sma
            features[f"bb_above_{period}"] = (df["close"] > upper).astype(int)
            features[f"bb_below_{period}"] = (df["close"] < lower).astype(int)

        # =====================================================================
        # RSI FEATURES (Mean Reversion Perspective)
        # =====================================================================

        for period in [7, 14, 21]:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            features[f"rsi_{period}"] = rsi
            features[f"rsi_overbought_{period}"] = (rsi > 70).astype(int)
            features[f"rsi_oversold_{period}"] = (rsi < 30).astype(int)
            features[f"rsi_extreme_{period}"] = ((rsi > 80) | (rsi < 20)).astype(int)

        # =====================================================================
        # STOCHASTIC FEATURES
        # =====================================================================

        for period in [14, 21]:
            low_min = df["low"].rolling(period).min()
            high_max = df["high"].rolling(period).max()

            features[f"stoch_k_{period}"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
            features[f"stoch_d_{period}"] = features[f"stoch_k_{period}"].rolling(3).mean()

        # =====================================================================
        # PRICE GAP FEATURES
        # =====================================================================

        # Distance from moving averages
        for period in [10, 20, 50, 200]:
            sma = df["close"].rolling(period).mean()
            features[f"dist_sma_{period}"] = (df["close"] - sma) / sma

        # Distance from high/low
        features["dist_from_high_20"] = (df["close"] - df["high"].rolling(20).max()) / df["close"]
        features["dist_from_low_20"] = (df["close"] - df["low"].rolling(20).min()) / df["close"]

        # =====================================================================
        # REVERSION SIGNALS
        # =====================================================================

        # Consecutive days up/down
        up_days = (df["close"] > df["close"].shift(1)).astype(int)
        down_days = (df["close"] < df["close"].shift(1)).astype(int)

        features["consec_up"] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
        features["consec_down"] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

        # Oversold after big drop
        features["big_drop_3d"] = (df["close"].pct_change(3) < -0.05).astype(int)
        features["big_rise_3d"] = (df["close"].pct_change(3) > 0.05).astype(int)

        # Volume exhaustion
        features["vol_sma_20"] = df["volume"].rolling(20).mean()
        features["vol_spike"] = df["volume"] / features["vol_sma_20"]
        features["vol_exhaustion"] = ((features["vol_spike"] > 2) & (features["big_drop_3d"] == 1)).astype(int)

        # Clean up
        features = features.drop(columns=["vol_sma_20"], errors="ignore")

        return features

    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """Generate mean reversion signal."""
        if self.model is None:
            return self._rule_based_predict(features)

        try:
            feature_cols = [c for c in self.feature_names if c in features.columns]
            X = features[feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)

            proba = self.model.predict_proba(X_scaled)[0]
            up_prob = proba[1]

            if up_prob > self.config.confidence_threshold:
                return ("BUY", up_prob)
            elif up_prob < 1 - self.config.confidence_threshold:
                return ("SELL", 1 - up_prob)
            else:
                return ("HOLD", 0.5)

        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return self._rule_based_predict(features)

    def _rule_based_predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """Rule-based mean reversion signal."""
        try:
            row = features.iloc[-1] if len(features) > 0 else features

            score = 0.0

            # Z-score signals
            if "zscore_20" in row:
                z = row["zscore_20"]
                if z < -2:
                    score += 0.3  # Oversold, expect reversion up
                elif z > 2:
                    score -= 0.3  # Overbought, expect reversion down

            # RSI signals
            if "rsi_14" in row:
                rsi = row["rsi_14"]
                if rsi < 30:
                    score += 0.25
                elif rsi > 70:
                    score -= 0.25

            # Bollinger band signals
            if "bb_below_20" in row and row["bb_below_20"] == 1:
                score += 0.2
            if "bb_above_20" in row and row["bb_above_20"] == 1:
                score -= 0.2

            confidence = 0.5 + score
            confidence = max(0.0, min(1.0, confidence))

            if confidence > 0.6:
                return ("BUY", confidence)
            elif confidence < 0.4:
                return ("SELL", 1 - confidence)
            else:
                return ("HOLD", 0.5)

        except Exception:
            return ("HOLD", 0.5)


