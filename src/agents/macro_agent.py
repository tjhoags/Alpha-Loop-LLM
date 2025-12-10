"""================================================================================
MACRO AGENT
================================================================================
Economic indicator-based trading agent:

1. Fed Policy: Interest rates, QE/QT
2. Economic Data: GDP, CPI, Employment
3. Yield Curve: 2s10s spread, inversions
4. Dollar Strength: DXY correlation
5. Global Macro: International factors

================================================================================
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base_agent import AgentConfig, BaseAgent


class MacroAgent(BaseAgent):
    """Macro-based trading agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.agent_type = "macro"
        config.use_macro_features = True
        super().__init__(config)
        logger.info("MacroAgent initialized")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate macro features."""
        features = self._base_features(df)

        # Macro proxies from price data
        features["trend_strength"] = df["close"].rolling(50).mean() / df["close"].rolling(200).mean()
        features["regime"] = np.where(features["trend_strength"] > 1, 1, -1)

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


