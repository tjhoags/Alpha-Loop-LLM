from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "xgboost": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        eval_metric="logloss",
                    ),
                ),
            ],
        ),
        "lightgbm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=63,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                    ),
                ),
            ],
        ),
        "catboost": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    CatBoostClassifier(
                        depth=6,
                        iterations=300,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        verbose=False,
                        random_seed=random_state,
                    ),
                ),
            ],
        ),
    }


def evaluate_model(model: Pipeline, X, y) -> Tuple[float, float]:
    preds = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, prob)
    return acc, auc


def time_series_cv(model: Pipeline, X, y, splits: int = 3) -> Tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=splits)
    accs = []
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        acc, auc = evaluate_model(model, X_test, y_test)
        accs.append(acc)
        aucs.append(auc)
    return float(np.mean(accs)), float(np.mean(aucs))


def save_model(model: Pipeline, name: str, models_dir: Path, metadata: dict | None = None) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    fname = models_dir / f"{name}.pkl"
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, fname)
    logger.info(f"Saved model: {fname}")
    return fname


def calculate_risk_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate risk metrics for a return series.

    Args:
    ----
        returns: Array of returns

    Returns:
    -------
        Dict with risk metrics
    """
    if len(returns) == 0:
        return {
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    # Annualized volatility (assuming daily returns)
    volatility = float(np.std(returns) * np.sqrt(252))

    # Sharpe ratio (assuming risk-free rate of 0)
    mean_return = float(np.mean(returns) * 252)  # Annualized
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Higher moments
    skewness = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)) if np.std(returns) > 0 else 0.0
    kurtosis = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3) if np.std(returns) > 0 else 0.0

    return {
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

