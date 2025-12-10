"""
================================================================================
BEHAVIORAL & PSYCHOLOGICAL FEATURE ENGINEERING
================================================================================
Quantified behavioral finance features for ML training:

EMOTIONAL INDICATORS:
- Fear/Greed proxy from price action
- Panic selling detection
- FOMO buying detection
- Capitulation signals

CROWD PSYCHOLOGY:
- Herding behavior metrics
- Information cascade proxies
- Social proof indicators

COGNITIVE BIAS SIGNALS:
- Anchoring to round numbers
- Recency bias in momentum
- Loss aversion patterns

GAME THEORY:
- Short squeeze probability
- Institutional vs retail divergence
- Information asymmetry proxies

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger


class BehavioralFeatureEngine:
    """
    Generates behavioral/psychological features for ML models.
    """
    
    @staticmethod
    def add_emotional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add emotional state indicators derived from price action.
        
        These proxy for market participant emotions without needing
        external sentiment data.
        """
        df = df.copy()
        
        # =================================================================
        # FEAR INDICATORS
        # =================================================================
        
        # Panic Selling Detection
        # High volume + large down move = panic
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1)
        df["return_1d"] = df["close"].pct_change()
        
        # Panic = high volume + big down day
        df["panic_score"] = np.where(
            (df["volume_ratio"] > 2) & (df["return_1d"] < -0.02),
            df["volume_ratio"] * abs(df["return_1d"]) * 10,
            0
        )
        df["panic_ma_5"] = df["panic_score"].rolling(5).mean()
        
        # Fear from consecutive down days
        df["down_day"] = (df["return_1d"] < 0).astype(int)
        df["consecutive_down"] = df["down_day"].rolling(5).sum()
        df["fear_streak"] = df["consecutive_down"] / 5  # 0 to 1
        
        # Volatility spike (fear indicator)
        df["volatility_20"] = df["return_1d"].rolling(20).std()
        df["volatility_ma_50"] = df["volatility_20"].rolling(50).mean()
        df["vol_spike"] = df["volatility_20"] / (df["volatility_ma_50"] + 0.001)
        df["fear_vol"] = np.where(df["vol_spike"] > 1.5, df["vol_spike"] - 1, 0)
        
        # =================================================================
        # GREED INDICATORS
        # =================================================================
        
        # FOMO Detection (buying into strength)
        # High volume + large up move after already up = FOMO
        df["return_5d"] = df["close"].pct_change(5)
        df["fomo_score"] = np.where(
            (df["volume_ratio"] > 1.5) & (df["return_1d"] > 0.02) & (df["return_5d"] > 0.05),
            df["volume_ratio"] * df["return_1d"] * 10,
            0
        )
        df["fomo_ma_5"] = df["fomo_score"].rolling(5).mean()
        
        # Consecutive up days (greed building)
        df["up_day"] = (df["return_1d"] > 0).astype(int)
        df["consecutive_up"] = df["up_day"].rolling(5).sum()
        df["greed_streak"] = df["consecutive_up"] / 5
        
        # New highs indicator
        df["high_20"] = df["high"].rolling(20).max()
        df["at_20d_high"] = (df["close"] >= df["high_20"] * 0.99).astype(int)
        
        # =================================================================
        # COMPOSITE FEAR/GREED
        # =================================================================
        
        # Fear score (0 to 1, higher = more fear)
        df["fear_composite"] = (
            0.30 * df["fear_streak"] +
            0.30 * np.clip(df["fear_vol"] / 2, 0, 1) +
            0.40 * np.clip(df["panic_ma_5"] / 0.5, 0, 1)
        )
        
        # Greed score (0 to 1, higher = more greed)
        df["greed_composite"] = (
            0.30 * df["greed_streak"] +
            0.30 * df["at_20d_high"] +
            0.40 * np.clip(df["fomo_ma_5"] / 0.3, 0, 1)
        )
        
        # Net sentiment (-1 to 1, negative = fear, positive = greed)
        df["emotional_state"] = df["greed_composite"] - df["fear_composite"]
        
        return df
    
    @staticmethod
    def add_crowd_psychology_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add crowd behavior and herding indicators.
        """
        df = df.copy()
        
        # =================================================================
        # HERDING INDICATORS
        # =================================================================
        
        # Return clustering (everyone doing same thing)
        df["return_abs"] = abs(df["return_1d"])
        df["return_abs_ma"] = df["return_abs"].rolling(20).mean()
        
        # Low dispersion in moves = herding
        df["return_std_10"] = df["return_1d"].rolling(10).std()
        df["return_std_50"] = df["return_1d"].rolling(50).std()
        df["herding_proxy"] = 1 - (df["return_std_10"] / (df["return_std_50"] + 0.001))
        df["herding_proxy"] = np.clip(df["herding_proxy"], 0, 1)
        
        # Volume clustering (everyone trading at same time)
        df["volume_std_10"] = df["volume"].rolling(10).std()
        df["volume_std_50"] = df["volume"].rolling(50).std()
        df["volume_clustering"] = df["volume_std_10"] / (df["volume_std_50"] + 1)
        
        # =================================================================
        # INFORMATION CASCADE PROXY
        # =================================================================
        
        # Sequence of same-direction moves (cascade building)
        df["direction"] = np.sign(df["return_1d"])
        df["same_direction_streak"] = 0
        
        # Calculate streaks
        streak = 0
        prev_dir = 0
        streaks = []
        for dir_val in df["direction"].fillna(0):
            if dir_val == prev_dir and dir_val != 0:
                streak += 1
            else:
                streak = 1
            streaks.append(streak)
            prev_dir = dir_val
        df["same_direction_streak"] = streaks
        
        # Cascade strength (longer streak + diminishing price impact)
        df["price_impact"] = abs(df["return_1d"]) / (df["volume_ratio"] + 0.01)
        df["price_impact_ma"] = df["price_impact"].rolling(5).mean()
        df["price_impact_change"] = df["price_impact_ma"].pct_change(5)
        
        # Cascade score: many same-direction moves with decreasing impact
        df["cascade_score"] = np.where(
            (df["same_direction_streak"] > 3) & (df["price_impact_change"] < 0),
            df["same_direction_streak"] / 10,
            0
        )
        
        # =================================================================
        # CONTRARIAN OPPORTUNITY
        # =================================================================
        
        # When herding is extreme, contrarian plays work
        df["contrarian_signal"] = np.where(
            (df["herding_proxy"] > 0.7) & (df["cascade_score"] > 0.3),
            -df["direction"],  # Go opposite of herd
            0
        )
        
        return df
    
    @staticmethod
    def add_cognitive_bias_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that detect/exploit cognitive biases.
        """
        df = df.copy()
        
        # =================================================================
        # ANCHORING BIAS
        # =================================================================
        
        # Distance from round numbers (people anchor to $10, $50, $100, etc.)
        df["price"] = df["close"]
        
        # Find nearest round number
        df["round_10"] = (df["price"] / 10).round() * 10
        df["round_50"] = (df["price"] / 50).round() * 50
        df["round_100"] = (df["price"] / 100).round() * 100
        
        # Distance from round numbers (normalized)
        df["dist_from_round_10"] = abs(df["price"] - df["round_10"]) / df["price"]
        df["dist_from_round_50"] = abs(df["price"] - df["round_50"]) / df["price"]
        df["dist_from_round_100"] = abs(df["price"] - df["round_100"]) / df["price"]
        
        # Near round number = potential support/resistance
        df["near_round_number"] = (
            (df["dist_from_round_10"] < 0.02) |
            (df["dist_from_round_50"] < 0.03) |
            (df["dist_from_round_100"] < 0.03)
        ).astype(int)
        
        # 52-week high/low anchoring
        df["high_52w"] = df["high"].rolling(252, min_periods=20).max()
        df["low_52w"] = df["low"].rolling(252, min_periods=20).min()
        df["dist_from_52w_high"] = (df["high_52w"] - df["price"]) / df["high_52w"]
        df["dist_from_52w_low"] = (df["price"] - df["low_52w"]) / df["low_52w"]
        
        # Near 52w high = resistance, Near 52w low = support
        df["near_52w_high"] = (df["dist_from_52w_high"] < 0.05).astype(int)
        df["near_52w_low"] = (df["dist_from_52w_low"] < 0.10).astype(int)
        
        # =================================================================
        # RECENCY BIAS
        # =================================================================
        
        # Compare recent vs long-term returns
        df["return_5d"] = df["price"].pct_change(5)
        df["return_20d"] = df["price"].pct_change(20)
        df["return_60d"] = df["price"].pct_change(60)
        
        # Recency weight: How much recent returns dominate perception
        df["recency_weight"] = abs(df["return_5d"]) / (abs(df["return_60d"]) / 12 + 0.001)
        df["recency_weight"] = np.clip(df["recency_weight"], 0, 10) / 10
        
        # Recency bias opportunity: Recent moves overweighted
        df["recency_overreaction"] = np.where(
            df["recency_weight"] > 0.7,
            -np.sign(df["return_5d"]),  # Fade recent move
            0
        )
        
        # =================================================================
        # LOSS AVERSION
        # =================================================================
        
        # Estimate percentage of holders underwater
        # Proxy: How far below recent highs
        df["drawdown_from_high"] = (df["high_52w"] - df["price"]) / df["high_52w"]
        
        # Break-even pressure: Many underwater + price approaching break-even
        df["breakeven_pressure"] = np.where(
            (df["drawdown_from_high"] > 0.15) & (df["return_5d"] > 0.05),
            df["drawdown_from_high"] * df["return_5d"] * 10,
            0
        )
        
        # Disposition effect: Tendency to sell winners, hold losers
        df["winner"] = (df["return_20d"] > 0.10).astype(int)
        df["loser"] = (df["return_20d"] < -0.10).astype(int)
        
        return df
    
    @staticmethod
    def add_game_theory_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add game theory derived features.
        """
        df = df.copy()
        
        # =================================================================
        # SHORT SQUEEZE SETUP
        # =================================================================
        
        # We don't have short interest data, but can proxy from price action
        # Sharp down move followed by sharp up = potential squeeze
        
        df["sharp_down"] = (df["return_1d"] < -0.05).astype(int)
        df["sharp_up"] = (df["return_1d"] > 0.05).astype(int)
        
        # Down then up pattern
        df["sharp_down_prev_5"] = df["sharp_down"].rolling(5).sum()
        df["squeeze_setup"] = np.where(
            (df["sharp_down_prev_5"] >= 2) & (df["return_1d"] > 0.03),
            1,
            0
        )
        
        # Squeeze momentum (up move accelerating)
        df["up_acceleration"] = df["return_1d"] - df["return_1d"].shift(1)
        df["squeeze_momentum"] = np.where(
            (df["squeeze_setup"] == 1) & (df["up_acceleration"] > 0),
            df["up_acceleration"] * 10,
            0
        )
        
        # =================================================================
        # COORDINATION GAME (MOMENTUM)
        # =================================================================
        
        # When momentum is strong, coordination to follow it increases
        df["momentum_20d"] = df["return_20d"]
        df["momentum_strength"] = abs(df["momentum_20d"])
        
        # Coordination score: Strong momentum + volume confirmation
        df["coordination_score"] = np.where(
            df["momentum_strength"] > 0.10,
            df["momentum_strength"] * df["volume_ratio"],
            0
        )
        
        # =================================================================
        # INFORMATION ASYMMETRY PROXY
        # =================================================================
        
        # Large moves on low volume = potential informed trading
        df["move_per_volume"] = abs(df["return_1d"]) / (np.log1p(df["volume"]) / 10)
        df["informed_trading_proxy"] = np.where(
            df["move_per_volume"] > df["move_per_volume"].rolling(50).quantile(0.9),
            1,
            0
        )
        
        return df
    
    @staticmethod
    def add_all_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all behavioral features at once.
        """
        df = BehavioralFeatureEngine.add_emotional_features(df)
        df = BehavioralFeatureEngine.add_crowd_psychology_features(df)
        df = BehavioralFeatureEngine.add_cognitive_bias_features(df)
        df = BehavioralFeatureEngine.add_game_theory_features(df)
        
        # Drop intermediate columns to keep feature set clean
        cols_to_keep = [
            # Original OHLCV
            "symbol", "timestamp", "open", "high", "low", "close", "volume",
            
            # Emotional features
            "panic_score", "panic_ma_5", "fear_streak", "fear_vol",
            "fomo_score", "fomo_ma_5", "greed_streak", "at_20d_high",
            "fear_composite", "greed_composite", "emotional_state",
            
            # Crowd psychology
            "herding_proxy", "volume_clustering", "same_direction_streak",
            "cascade_score", "contrarian_signal",
            
            # Cognitive bias
            "near_round_number", "near_52w_high", "near_52w_low",
            "dist_from_52w_high", "dist_from_52w_low",
            "recency_weight", "recency_overreaction",
            "breakeven_pressure", "winner", "loser",
            
            # Game theory
            "squeeze_setup", "squeeze_momentum",
            "coordination_score", "informed_trading_proxy",
            
            # Keep returns for later
            "return_1d", "return_5d", "return_20d", "volume_ratio"
        ]
        
        available_cols = [c for c in cols_to_keep if c in df.columns]
        
        logger.info(f"Added {len(available_cols) - 7} behavioral features")
        return df[available_cols]


def prepare_behavioral_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare behavioral features for ML training.
    Combines technical + behavioral features.
    """
    from src.ml.feature_engineering import add_technical_indicators, make_supervised
    
    # Add technical indicators first
    df_tech = add_technical_indicators(df)
    
    # Add behavioral features
    df_behavioral = BehavioralFeatureEngine.add_all_behavioral_features(df)
    
    # Merge on timestamp
    df_combined = df_tech.merge(
        df_behavioral.drop(columns=["symbol", "open", "high", "low", "close", "volume"], errors="ignore"),
        on="timestamp",
        how="left"
    )
    
    # Create supervised target
    df_sup = make_supervised(df_combined, horizon=1)
    
    # Define feature columns
    exclude_cols = ["symbol", "timestamp", "future_return", "target", 
                    "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df_sup.columns if c not in exclude_cols]
    
    X = df_sup[feature_cols].copy()
    y = df_sup["target"].copy()
    
    # Handle infinities and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    logger.info(f"Prepared {len(feature_cols)} features (technical + behavioral)")
    return X, y


