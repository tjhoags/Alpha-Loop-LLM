"""================================================================================
VOLATILITY AGENT
================================================================================
Volatility-based trading strategies:

1. VIX Trading: Long/short volatility
2. Volatility Mean Reversion: Vol tends to mean-revert
3. Volatility Clustering: Vol begets vol
4. Options Strategies: Straddles, strangles
5. Regime Detection: Low vol vs high vol environments

================================================================================
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class VolatilityAgent(BaseAgent):
    """Volatility-based trading agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "volatility"
        config.use_volatility_features = True
        super().__init__(config)
        logger.info("VolatilityAgent initialized")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility features."""
        features = self._base_features(df)

        ret = df["close"].pct_change()

        # Realized volatility
        for period in [5, 10, 20, 60]:
            features[f"rvol_{period}"] = ret.rolling(period).std() * np.sqrt(252)

        # Volatility ratios
        features["vol_ratio_5_20"] = features["rvol_5"] / (features["rvol_20"] + 1e-10)
        features["vol_ratio_10_60"] = features["rvol_10"] / (features["rvol_60"] + 1e-10)

        # Parkinson volatility
        features["parkinson"] = np.sqrt((np.log(df["high"]/df["low"])**2).rolling(20).mean() / (4*np.log(2))) * np.sqrt(252)

        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        features["atr_14"] = tr.rolling(14).mean()
        features["atr_pct"] = features["atr_14"] / df["close"]

        # Volatility regime
        vol_sma = features["rvol_20"].rolling(60).mean()
        features["high_vol_regime"] = (features["rvol_20"] > vol_sma * 1.5).astype(int)
        features["low_vol_regime"] = (features["rvol_20"] < vol_sma * 0.7).astype(int)

        return features

    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        if self.model is not None:
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
            except Exception:
                pass
        return ("HOLD", 0.5)


