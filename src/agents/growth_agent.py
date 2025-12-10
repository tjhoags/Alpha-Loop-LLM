"""================================================================================
GROWTH AGENT
================================================================================
Growth stock selection strategies:

1. Revenue growth acceleration
2. Earnings growth
3. Price momentum (growth stocks trend)
4. High P/E with justification
5. TAM expansion plays

================================================================================
"""

from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class GrowthAgent(BaseAgent):
    """Growth investing agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "growth"
        config.use_momentum_features = True
        super().__init__(config)
        logger.info("GrowthAgent initialized")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate growth features."""
        features = self._base_features(df)

        # Growth stocks have strong momentum
        features["mom_60"] = df["close"].pct_change(60)
        features["mom_120"] = df["close"].pct_change(120)
        features["mom_252"] = df["close"].pct_change(252)

        # New highs (growth stocks make new highs)
        features["at_52w_high"] = (df["close"] >= df["high"].rolling(252).max() * 0.95).astype(int)

        # Trend strength
        features["sma_50"] = df["close"].rolling(50).mean()
        features["sma_200"] = df["close"].rolling(200).mean()
        features["trend_strength"] = features["sma_50"] / features["sma_200"]
        features["above_sma_200"] = (df["close"] > features["sma_200"]).astype(int)

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
            except:
                pass
        return ("HOLD", 0.5)


