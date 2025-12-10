"""================================================================================
LIQUIDITY AGENT
================================================================================
Market microstructure alpha:

1. Bid-Ask Spread Analysis
2. Order Flow Imbalance
3. Volume Profile Trading
4. Liquidity Provision
5. Market Impact Modeling

================================================================================
"""

from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class LiquidityAgent(BaseAgent):
    """Liquidity-based trading agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "liquidity"
        super().__init__(config)
        logger.info("LiquidityAgent initialized")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate liquidity features."""
        features = self._base_features(df)

        # Volume features
        features["vol_sma_10"] = df["volume"].rolling(10).mean()
        features["vol_sma_20"] = df["volume"].rolling(20).mean()
        features["vol_ratio"] = df["volume"] / features["vol_sma_20"]

        # Dollar volume
        features["dollar_volume"] = df["close"] * df["volume"]
        features["dollar_vol_sma"] = features["dollar_volume"].rolling(20).mean()

        # Amihud illiquidity
        ret_abs = df["close"].pct_change().abs()
        features["amihud"] = (ret_abs / features["dollar_volume"]).rolling(20).mean()

        # High-low spread proxy
        features["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        features["hl_spread_sma"] = features["hl_spread"].rolling(20).mean()

        # Volume-weighted price
        features["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        features["price_vs_vwap"] = df["close"] / features["vwap"] - 1

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


