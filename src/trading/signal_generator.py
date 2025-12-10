"""
================================================================================
SIGNAL GENERATOR - Ensemble ML + Sentiment Fusion
================================================================================
This module generates trading signals by combining:
1. Multiple ML model predictions (XGBoost, LightGBM, CatBoost)
2. NLP sentiment analysis from news/research
3. Technical indicator confirmations

Signals are graded by confidence (0.0 to 1.0) and only high-confidence
signals are passed to execution.
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.ml.feature_engineering import add_technical_indicators


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    direction: str  # "LONG", "SHORT", or "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    ml_score: float  # Average ML prediction
    sentiment_score: float  # Sentiment contribution
    timestamp: datetime
    model_votes: Dict[str, float]  # Individual model predictions
    
    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if confidence > 0.6 and direction is not neutral."""
        return self.confidence >= 0.6 and self.direction != "NEUTRAL"


class SignalGenerator:
    """
    Generates trading signals using ensemble ML models and sentiment.
    
    SIGNAL GRADING:
    - ML Score: Average probability from all models
    - Sentiment Score: FinBERT sentiment on recent news
    - Technical Confirmation: RSI, MACD alignment
    
    Final confidence = weighted combination of all factors
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.models: Dict[str, Dict] = {}  # symbol -> {model_name: model}
        self.model_metadata: Dict[str, Dict] = {}  # symbol -> {model_name: metadata}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all trained models from the models directory."""
        models_dir = self.settings.models_dir
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return
        
        # Find all model files
        model_files = list(models_dir.glob("*.pkl"))
        if not model_files:
            logger.warning("No trained models found. Run training first.")
            return
        
        for model_path in model_files:
            try:
                payload = joblib.load(model_path)
                model = payload.get("model")
                metadata = payload.get("metadata", {})
                
                symbol = metadata.get("symbol", "UNKNOWN")
                model_type = metadata.get("type", model_path.stem.split("_")[-2])
                
                if symbol not in self.models:
                    self.models[symbol] = {}
                    self.model_metadata[symbol] = {}
                
                self.models[symbol][model_type] = model
                self.model_metadata[symbol][model_type] = metadata
                
                logger.debug(f"Loaded model: {symbol}/{model_type}")
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
        
        total_models = sum(len(m) for m in self.models.values())
        logger.info(f"Loaded {total_models} models across {len(self.models)} symbols")
    
    def reload_models(self) -> None:
        """Reload models (call after overnight training completes)."""
        self.models.clear()
        self.model_metadata.clear()
        self._load_models()
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction (same as training)."""
        return add_technical_indicators(df)
    
    def _ensemble_predict(self, symbol: str, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Get ensemble prediction from all models for a symbol.
        Returns (average_probability, individual_votes).
        """
        if symbol not in self.models:
            logger.warning(f"No models found for {symbol}")
            return 0.5, {}
        
        votes = {}
        for model_name, model in self.models[symbol].items():
            try:
                # Get probability of class 1 (price going up)
                prob = model.predict_proba(X)[:, 1][-1]  # Last prediction
                votes[model_name] = float(prob)
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}/{model_name}: {e}")
                votes[model_name] = 0.5
        
        if not votes:
            return 0.5, {}
        
        # Simple average (could weight by model CV scores)
        avg_prob = np.mean(list(votes.values()))
        return float(avg_prob), votes
    
    def _get_technical_confirmation(self, df: pd.DataFrame) -> float:
        """
        Get technical indicator confirmation score.
        Returns score in [-1, 1] where positive = bullish, negative = bearish.
        """
        if df.empty:
            return 0.0
        
        latest = df.iloc[-1]
        score = 0.0
        
        # RSI
        if "rsi_14" in latest:
            rsi = latest["rsi_14"]
            if rsi < 30:  # Oversold - bullish
                score += 0.3
            elif rsi > 70:  # Overbought - bearish
                score -= 0.3
        
        # MACD
        if "macd_diff" in latest:
            macd_diff = latest["macd_diff"]
            if macd_diff > 0:
                score += 0.2
            else:
                score -= 0.2
        
        # EMA trend
        if "ema_12" in latest and "ema_26" in latest:
            if latest["ema_12"] > latest["ema_26"]:
                score += 0.2  # Bullish trend
            else:
                score -= 0.2  # Bearish trend
        
        return np.clip(score, -1.0, 1.0)
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        sentiment_score: float = 0.0
    ) -> Signal:
        """
        Generate a trading signal for a symbol.
        
        Args:
            symbol: Stock/crypto symbol
            price_data: DataFrame with OHLCV data
            sentiment_score: Pre-computed sentiment score [-1, 1]
        
        Returns:
            Signal object with direction and confidence
        """
        timestamp = datetime.now()
        
        # Prepare features
        try:
            df_features = self._prepare_features(price_data)
        except Exception as e:
            logger.error(f"Feature preparation failed for {symbol}: {e}")
            return Signal(
                symbol=symbol,
                direction="NEUTRAL",
                confidence=0.0,
                ml_score=0.5,
                sentiment_score=0.0,
                timestamp=timestamp,
                model_votes={}
            )
        
        if df_features.empty or len(df_features) < 10:
            return Signal(
                symbol=symbol,
                direction="NEUTRAL",
                confidence=0.0,
                ml_score=0.5,
                sentiment_score=0.0,
                timestamp=timestamp,
                model_votes={}
            )
        
        # Get feature columns
        feature_cols = [
            "return_1", "return_5", "volatility_10", "volume_z",
            "rsi_14", "ema_12", "ema_26", "macd", "macd_signal", "macd_diff"
        ]
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df_features.columns]
        X = df_features[available_cols].iloc[-1:].copy()
        
        # ML Prediction
        ml_prob, votes = self._ensemble_predict(symbol, X)
        
        # Technical confirmation
        tech_score = self._get_technical_confirmation(df_features)
        
        # Combine signals
        # ML contributes 60%, sentiment 25%, technicals 15%
        ML_WEIGHT = 0.60
        SENTIMENT_WEIGHT = 0.25
        TECH_WEIGHT = 0.15
        
        # Convert ML probability to directional score [-1, 1]
        ml_directional = (ml_prob - 0.5) * 2  # 0.7 -> 0.4, 0.3 -> -0.4
        
        # Combined score
        combined_score = (
            ML_WEIGHT * ml_directional +
            SENTIMENT_WEIGHT * sentiment_score +
            TECH_WEIGHT * tech_score
        )
        
        # Determine direction and confidence
        if combined_score > 0.1:
            direction = "LONG"
            confidence = min(combined_score, 1.0)
        elif combined_score < -0.1:
            direction = "SHORT"
            confidence = min(abs(combined_score), 1.0)
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        signal = Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            ml_score=ml_prob,
            sentiment_score=sentiment_score,
            timestamp=timestamp,
            model_votes=votes
        )
        
        logger.info(
            f"Signal for {symbol}: {direction} | "
            f"Conf={confidence:.2f} | "
            f"ML={ml_prob:.3f} | "
            f"Sent={sentiment_score:.2f} | "
            f"Tech={tech_score:.2f}"
        )
        
        return signal
    
    def generate_all_signals(
        self,
        price_data: Dict[str, pd.DataFrame],
        sentiment_scores: Optional[Dict[str, float]] = None
    ) -> List[Signal]:
        """
        Generate signals for all symbols in portfolio.
        
        Args:
            price_data: Dict of symbol -> price DataFrame
            sentiment_scores: Optional dict of symbol -> sentiment
        
        Returns:
            List of Signal objects, sorted by confidence
        """
        sentiment_scores = sentiment_scores or {}
        signals = []
        
        for symbol, df in price_data.items():
            sent_score = sentiment_scores.get(symbol, 0.0)
            signal = self.generate_signal(symbol, df, sent_score)
            signals.append(signal)
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        # Log summary
        actionable = [s for s in signals if s.is_actionable]
        logger.info(
            f"Generated {len(signals)} signals, {len(actionable)} actionable"
        )
        
        return signals
    
    def get_model_performance(self, symbol: str) -> Dict:
        """Get performance metrics for models on a symbol."""
        if symbol not in self.model_metadata:
            return {}
        
        performance = {}
        for model_name, meta in self.model_metadata[symbol].items():
            performance[model_name] = {
                "cv_auc": meta.get("cv_auc", 0),
                "cv_acc": meta.get("cv_acc", 0),
                "timestamp": meta.get("timestamp", "unknown"),
                "status": meta.get("status", "unknown")
            }
        return performance


