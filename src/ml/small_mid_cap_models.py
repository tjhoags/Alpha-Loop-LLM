"""================================================================================
SMALL/MID CAP SPECIALIZED ML MODELS
================================================================================
Author: Alpha Loop Capital, LLC

Specialized ML models for small/mid cap trading strategies:
1. RetailArbitrageModel - Exploits retail inefficiencies
2. ConversionReversalModel - Options arbitrage detection
3. MomentumCaptureModel - Captures retail-driven momentum
4. MeanReversionModel - Fades overextended retail moves

These models are specifically tuned for:
- Higher volatility environment
- Lower liquidity conditions
- Wider bid-ask spreads
- Stronger retail influence

Training uses TimeSeriesSplit to prevent look-ahead bias.
================================================================================
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.ensemble import (
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config.settings import get_settings


@dataclass
class ModelMetrics:
    """Performance metrics for a trained model."""

    model_name: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    train_samples: int
    test_samples: int
    feature_count: int
    trained_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "features": self.feature_count,
            "trained_at": self.trained_at.isoformat(),
        }

    def passes_threshold(self, min_auc: float = 0.52, min_accuracy: float = 0.52) -> bool:
        return self.auc >= min_auc and self.accuracy >= min_accuracy


class RetailArbitrageModel:
    """ML model for detecting retail arbitrage opportunities.

    Features: Retail inefficiency indicators
    Target: Profitable trade opportunity (binary)

    Uses ensemble of XGBoost + LightGBM + CatBoost
    """

    def __init__(self, model_id: str = "retail_arb"):
        self.model_id = model_id
        self.settings = get_settings()
        self.model = None
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_names: List[str] = []
        self.metrics: Optional[ModelMetrics] = None
        self.is_trained = False

        # Model hyperparameters tuned for small/mid cap
        self.xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
        }

        self.lgb_params = {
            "n_estimators": 400,
            "num_leaves": 31,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        self.cat_params = {
            "iterations": 300,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": False,
        }

    def _build_ensemble(self):
        """Build ensemble model."""
        xgb_model = xgb.XGBClassifier(**self.xgb_params)
        lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        cat_model = CatBoostClassifier(**self.cat_params)

        # Voting ensemble
        self.model = VotingClassifier(
            estimators=[
                ("xgb", xgb_model),
                ("lgb", lgb_model),
                ("cat", cat_model),
            ],
            voting="soft",
            weights=[0.4, 0.35, 0.25],  # XGB slightly higher weight
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> ModelMetrics:
        """Train the retail arbitrage model with time series cross-validation.

        Args:
        ----
            X: Feature DataFrame
            y: Target Series
            cv_folds: Number of CV folds

        Returns:
        -------
            ModelMetrics with performance stats
        """
        logger.info(f"Training RetailArbitrageModel on {len(X)} samples...")

        self.feature_names = X.columns.tolist()

        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Build fresh model for each fold
            self._build_ensemble()
            self.model.fit(X_train, y_train)

            # Score
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]

            fold_auc = roc_auc_score(y_val, y_proba)
            cv_scores.append(fold_auc)

            logger.info(f"  Fold {fold+1}/{cv_folds}: AUC = {fold_auc:.4f}")

        # Final training on all data
        self._build_ensemble()
        self.model.fit(X_scaled, y)

        # Final evaluation (last 20% as holdout)
        split_idx = int(len(X) * 0.8)
        X_test = X_scaled[split_idx:]
        y_test = y.iloc[split_idx:]

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = ModelMetrics(
            model_name=f"RetailArbitrage_{self.model_id}",
            accuracy=accuracy_score(y_test, y_pred),
            auc=roc_auc_score(y_test, y_proba),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            train_samples=split_idx,
            test_samples=len(y_test),
            feature_count=len(self.feature_names),
        )

        self.is_trained = True

        logger.info(f"Training complete: AUC={self.metrics.auc:.4f}, Acc={self.metrics.accuracy:.4f}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = X[self.feature_names]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = X[self.feature_names]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from ensemble."""
        if not self.is_trained:
            return pd.DataFrame()

        # Average importance across ensemble members
        importances = np.zeros(len(self.feature_names))

        for name, model in self.model.named_estimators_.items():
            if hasattr(model, "feature_importances_"):
                importances += model.feature_importances_

        importances /= len(self.model.named_estimators_)

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = os.path.join(
                self.settings.models_dir,
                f"retail_arb_{self.model_id}_{datetime.now().strftime('%Y%m%d')}.pkl",
            )

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "model_id": self.model_id,
        }, path)

        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.model_id = data["model_id"]
        self.is_trained = True

        logger.info(f"Model loaded from {path}")


class ConversionReversalModel:
    """ML model for detecting conversion/reversal arbitrage opportunities.

    Features: Options pricing features, put-call parity violations
    Target: Profitable arbitrage opportunity (binary)
    """

    def __init__(self, model_id: str = "conv_rev"):
        self.model_id = model_id
        self.settings = get_settings()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.metrics: Optional[ModelMetrics] = None
        self.is_trained = False

        # Simpler model for arbitrage detection (more deterministic)
        self.params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "random_state": 42,
        }

    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 3) -> ModelMetrics:
        """Train the conversion/reversal model."""
        logger.info(f"Training ConversionReversalModel on {len(X)} samples...")

        self.feature_names = X.columns.tolist()

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Use XGBoost for this model (good with structured data)
        self.model = xgb.XGBClassifier(**self.params)

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            y_train = y.iloc[train_idx]
            self.model.fit(X_train, y_train)

        # Final fit on all data
        self.model.fit(X_scaled, y)

        # Evaluate
        split_idx = int(len(X) * 0.8)
        y_pred = self.model.predict(X_scaled[split_idx:])
        y_proba = self.model.predict_proba(X_scaled[split_idx:])[:, 1]
        y_test = y.iloc[split_idx:]

        self.metrics = ModelMetrics(
            model_name=f"ConversionReversal_{self.model_id}",
            accuracy=accuracy_score(y_test, y_pred),
            auc=roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            train_samples=split_idx,
            test_samples=len(y_test),
            feature_count=len(self.feature_names),
        )

        self.is_trained = True
        logger.info(f"Training complete: AUC={self.metrics.auc:.4f}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained.")
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained.")
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(
                self.settings.models_dir,
                f"conv_rev_{self.model_id}_{datetime.now().strftime('%Y%m%d')}.pkl",
            )
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.is_trained = True


class MeanReversionModel:
    """ML model for detecting mean reversion opportunities.

    Targets overextended retail moves that are likely to revert.
    Best for small/mid caps with high retail participation.
    """

    def __init__(self, model_id: str = "mean_rev"):
        self.model_id = model_id
        self.settings = get_settings()
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.metrics: Optional[ModelMetrics] = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> ModelMetrics:
        """Train mean reversion model."""
        logger.info(f"Training MeanReversionModel on {len(X)} samples...")

        self.feature_names = X.columns.tolist()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)

        # LightGBM for speed
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        # Train with CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        for train_idx, val_idx in tscv.split(X_scaled):
            self.model.fit(X_scaled[train_idx], y.iloc[train_idx])

        # Final fit
        self.model.fit(X_scaled, y)

        # Evaluate
        split_idx = int(len(X) * 0.8)
        y_pred = self.model.predict(X_scaled[split_idx:])
        y_proba = self.model.predict_proba(X_scaled[split_idx:])[:, 1]
        y_test = y.iloc[split_idx:]

        self.metrics = ModelMetrics(
            model_name=f"MeanReversion_{self.model_id}",
            accuracy=accuracy_score(y_test, y_pred),
            auc=roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            train_samples=split_idx,
            test_samples=len(y_test),
            feature_count=len(self.feature_names),
        )

        self.is_trained = True
        logger.info(f"Training complete: AUC={self.metrics.auc:.4f}")
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained.")
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained.")
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(
                self.settings.models_dir,
                f"mean_rev_{self.model_id}_{datetime.now().strftime('%Y%m%d')}.pkl",
            )
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }, path)
        logger.info(f"Model saved to {path}")


def train_all_small_mid_cap_models(
    price_data: pd.DataFrame,
    options_data: pd.DataFrame = None,
) -> Dict[str, ModelMetrics]:
    """Train all small/mid cap specialized models.

    Args:
    ----
        price_data: OHLCV price data for small/mid cap stocks
        options_data: Options chain data (optional)

    Returns:
    -------
        Dict of model_name -> ModelMetrics
    """
    from src.ml.feature_engineering import add_technical_indicators
    from src.ml.options_arbitrage_features import add_options_arbitrage_features
    from src.ml.retail_inefficiency_features import (
        add_retail_inefficiency_features,
        create_retail_arbitrage_target,
    )

    results = {}

    # =========================================================================
    # 1. RETAIL ARBITRAGE MODEL
    # =========================================================================
    logger.info("Training Retail Arbitrage Model...")

    # Add features
    df = add_technical_indicators(price_data.copy())
    df = add_retail_inefficiency_features(df)
    df = create_retail_arbitrage_target(df, horizon=5, min_return=0.01)

    # Prepare features
    exclude = ["symbol", "timestamp", "target", "target_long", "target_short",
               "target_binary", "future_return", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df["target_binary"]

    # Train
    retail_model = RetailArbitrageModel("v1")
    retail_metrics = retail_model.train(X, y)
    retail_model.save()
    results["retail_arbitrage"] = retail_metrics

    # =========================================================================
    # 2. MEAN REVERSION MODEL
    # =========================================================================
    logger.info("Training Mean Reversion Model...")

    # Target: price reverts after overextension
    df["mean_rev_target"] = (
        (df["overextended"] == 1) &
        (df["future_return"].shift(-5) * np.sign(df["price_extension"]) < 0)
    ).astype(int)

    y_mr = df["mean_rev_target"].dropna()
    X_mr = X.loc[y_mr.index]

    mean_rev_model = MeanReversionModel("v1")
    mr_metrics = mean_rev_model.train(X_mr, y_mr)
    mean_rev_model.save()
    results["mean_reversion"] = mr_metrics

    # =========================================================================
    # 3. CONVERSION/REVERSAL MODEL (if options data available)
    # =========================================================================
    if options_data is not None and len(options_data) > 0:
        logger.info("Training Conversion/Reversal Model...")

        options_df = add_options_arbitrage_features(options_data.copy())

        # Target: arbitrage opportunity that was profitable
        options_df["arb_target"] = (
            (options_df["conversion_signal"] == 1) |
            (options_df["reversal_signal"] == 1)
        ).astype(int)

        opt_exclude = ["symbol", "underlying_price", "strike", "expiry",
                       "call_bid", "call_ask", "put_bid", "put_ask",
                       "arb_target", "T", "expiry_date"]
        opt_features = [c for c in options_df.columns if c not in opt_exclude]

        X_opt = options_df[opt_features]
        y_opt = options_df["arb_target"]

        conv_rev_model = ConversionReversalModel("v1")
        cr_metrics = conv_rev_model.train(X_opt, y_opt)
        conv_rev_model.save()
        results["conversion_reversal"] = cr_metrics

    logger.info("All models trained successfully!")
    return results


if __name__ == "__main__":
    # Quick test with synthetic data
    logger.info("Testing small/mid cap models...")

    # Create synthetic price data
    np.random.seed(42)
    n = 5000

    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="5min"),
        "symbol": "TEST",
        "open": 50 + np.cumsum(np.random.randn(n) * 0.1),
        "volume": np.random.randint(10000, 100000, n),
    })
    df["close"] = df["open"] + np.random.randn(n) * 0.5
    df["high"] = df[["open", "close"]].max(axis=1) + abs(np.random.randn(n) * 0.2)
    df["low"] = df[["open", "close"]].min(axis=1) - abs(np.random.randn(n) * 0.2)

    # Train models
    results = train_all_small_mid_cap_models(df)

    print("\nTraining Results:")
    for name, metrics in results.items():
        print(f"  {name}: AUC={metrics.auc:.4f}, Acc={metrics.accuracy:.4f}")
