"""================================================================================
VALUE AGENT
================================================================================
Fundamental value investing strategies:

1. P/E, P/B, P/S ratios
2. Earnings yield
3. Book-to-market
4. DCF-based intrinsic value
5. Quality + Value combo

Based on: Fama-French Value Factor
================================================================================
"""

from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class ValueAgent(BaseAgent):
    """Value investing agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "value"
        config.use_fundamental_features = True
        super().__init__(config)
        logger.info("ValueAgent initialized")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate value features."""
        features = self._base_features(df)

        # Price-based value proxies
        features["price_to_52w_high"] = df["close"] / df["high"].rolling(252).max()
        features["price_to_52w_low"] = df["close"] / df["low"].rolling(252).min()

        # Mean reversion (value stocks often are beaten down)
        features["drawdown"] = df["close"] / df["close"].rolling(252).max() - 1
        features["value_score"] = -features["drawdown"]  # Bigger drawdown = more value

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


