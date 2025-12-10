"""================================================================================
OPTIONS STRATEGY TRAINING SCRIPT
================================================================================
Alpha Loop Capital, LLC

Trains options-specific trading strategies:
- Conversion/Reversal Arbitrage
- Put-Call Parity Violations
- Volatility Surface Anomalies
- Box Spread Arbitrage
- Calendar Spread Mispricing
- Greeks-based Strategies

Usage:
    python scripts/train_options_strategies.py --continuous --interval 15
================================================================================
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.database.connection import get_engine


# =============================================================================
# OPTIONS STRATEGY DEFINITIONS
# =============================================================================

OPTIONS_STRATEGIES = {
    "conversion_reversal": {
        "description": "Synthetic arbitrage via conversions and reversals",
        "underlyings": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"],
        "min_edge_bps": 5,
    },
    "put_call_parity": {
        "description": "Exploit put-call parity violations",
        "underlyings": ["SPY", "QQQ", "IWM", "AAPL", "TSLA"],
        "min_edge_bps": 3,
    },
    "volatility_surface": {
        "description": "Trade vol surface anomalies",
        "underlyings": ["SPY", "QQQ", "VIX"],
        "min_edge_bps": 10,
    },
    "box_spread": {
        "description": "Risk-free box spread arbitrage",
        "underlyings": ["SPY", "QQQ", "IWM"],
        "min_edge_bps": 2,
    },
    "calendar_spread": {
        "description": "Exploit calendar spread mispricing",
        "underlyings": ["SPY", "QQQ", "AAPL", "MSFT"],
        "min_edge_bps": 5,
    },
    "delta_hedging": {
        "description": "ML-enhanced delta hedging",
        "underlyings": ["SPY", "QQQ", "IWM"],
        "min_edge_bps": 1,
    },
}


# =============================================================================
# BLACK-SCHOLES PRICING
# =============================================================================

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return max(0, S - K)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict:
    """Calculate option Greeks."""
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta (per day)
    if option_type == "call":
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    # Vega (per 1% vol change)
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100

    # Rho (per 1% rate change)
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def implied_volatility(price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    """Calculate implied volatility using Newton-Raphson."""
    from scipy.stats import norm

    if T <= 0:
        return 0.0

    # Initial guess
    sigma = 0.3

    for _ in range(100):
        if option_type == "call":
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_put(S, K, T, r, sigma)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)

        if vega < 1e-10:
            break

        diff = model_price - price
        if abs(diff) < 1e-6:
            break

        sigma = sigma - diff / vega
        sigma = max(0.01, min(sigma, 5.0))

    return sigma


# =============================================================================
# ARBITRAGE DETECTION
# =============================================================================

def detect_conversion_reversal(
    S: float,
    K: float,
    T: float,
    r: float,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
) -> Dict:
    """Detect conversion/reversal arbitrage opportunities."""
    # Forward price
    F = S * np.exp(r * T)
    PV_K = K * np.exp(-r * T)

    # Conversion: Long stock + Long put + Short call
    # Profit if: S + P - C > K * exp(-rT)
    conversion_cost = S + put_ask - call_bid
    conversion_value = PV_K
    conversion_edge = conversion_value - conversion_cost

    # Reversal: Short stock + Short put + Long call
    # Profit if: C - P - S > -K * exp(-rT)
    reversal_cost = call_ask - put_bid - S
    reversal_value = -PV_K
    reversal_edge = -reversal_cost - reversal_value

    return {
        "conversion_edge": conversion_edge,
        "reversal_edge": reversal_edge,
        "conversion_signal": 1 if conversion_edge > 0 else 0,
        "reversal_signal": 1 if reversal_edge > 0 else 0,
        "forward": F,
        "pv_strike": PV_K,
    }


def detect_put_call_parity_violation(
    S: float,
    K: float,
    T: float,
    r: float,
    call_mid: float,
    put_mid: float,
) -> Dict:
    """Detect put-call parity violations."""
    # Put-Call Parity: C - P = S - K * exp(-rT)
    theoretical_diff = S - K * np.exp(-r * T)
    actual_diff = call_mid - put_mid
    violation = actual_diff - theoretical_diff

    return {
        "parity_violation": violation,
        "violation_pct": violation / S * 100,
        "signal": 1 if abs(violation) > 0.01 * S else 0,
    }


def detect_box_spread_arbitrage(
    K1: float,
    K2: float,
    T: float,
    r: float,
    call_K1_ask: float,
    call_K2_bid: float,
    put_K1_bid: float,
    put_K2_ask: float,
) -> Dict:
    """Detect box spread arbitrage (risk-free)."""
    # Box spread value should equal PV of strike difference
    box_value = (K2 - K1) * np.exp(-r * T)

    # Long box: Buy call K1, Sell call K2, Sell put K1, Buy put K2
    long_box_cost = call_K1_ask - call_K2_bid - put_K1_bid + put_K2_ask
    long_box_edge = box_value - long_box_cost

    # Short box: Sell call K1, Buy call K2, Buy put K1, Sell put K2
    short_box_cost = -call_K1_ask + call_K2_bid + put_K1_bid - put_K2_ask
    short_box_edge = -box_value - short_box_cost

    return {
        "box_value": box_value,
        "long_box_edge": long_box_edge,
        "short_box_edge": short_box_edge,
        "signal": 1 if max(long_box_edge, short_box_edge) > 0 else 0,
    }


# =============================================================================
# DATA GENERATION (Synthetic for training)
# =============================================================================

def generate_synthetic_options_data(
    underlyings: List[str],
    n_strikes: int = 10,
    n_expiries: int = 4,
) -> pd.DataFrame:
    """Generate synthetic options data for training."""
    records = []

    for underlying in underlyings:
        # Base price varies by underlying
        base_prices = {
            "SPY": 450, "QQQ": 380, "IWM": 200, "AAPL": 180,
            "MSFT": 370, "NVDA": 480, "TSLA": 250, "AMD": 120,
        }
        S = base_prices.get(underlying, 100) * (1 + np.random.normal(0, 0.02))

        for days_to_expiry in [7, 14, 30, 60]:
            T = days_to_expiry / 365
            r = 0.05  # Risk-free rate

            # Generate strikes around ATM
            for strike_pct in np.linspace(0.9, 1.1, n_strikes):
                K = round(S * strike_pct, 2)

                # Random IV (with skew)
                moneyness = K / S
                iv_base = 0.20 + 0.1 * (1 - moneyness)**2  # Vol smile
                iv = iv_base * (1 + np.random.normal(0, 0.1))
                iv = max(0.05, min(iv, 1.0))

                # Calculate theoretical prices
                call_theo = black_scholes_call(S, K, T, r, iv)
                put_theo = black_scholes_put(S, K, T, r, iv)

                # Add bid-ask spread (wider for OTM)
                spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
                call_spread = call_theo * spread_pct
                put_spread = put_theo * spread_pct

                # Sometimes introduce arbitrage opportunities (5% of time)
                arb_noise = 0
                if np.random.random() < 0.05:
                    arb_noise = np.random.choice([-1, 1]) * 0.02 * S

                call_bid = max(0.01, call_theo - call_spread / 2 + arb_noise)
                call_ask = call_theo + call_spread / 2
                put_bid = max(0.01, put_theo - put_spread / 2 - arb_noise)
                put_ask = put_theo + put_spread / 2

                # Calculate Greeks
                greeks = calculate_greeks(S, K, T, r, iv, "call")

                records.append({
                    "underlying": underlying,
                    "underlying_price": S,
                    "strike": K,
                    "expiry_days": days_to_expiry,
                    "T": T,
                    "iv": iv,
                    "call_bid": call_bid,
                    "call_ask": call_ask,
                    "call_mid": (call_bid + call_ask) / 2,
                    "put_bid": put_bid,
                    "put_ask": put_ask,
                    "put_mid": (put_bid + put_ask) / 2,
                    "moneyness": moneyness,
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "theta": greeks["theta"],
                    "vega": greeks["vega"],
                    "has_arb": 1 if arb_noise != 0 else 0,
                })

    return pd.DataFrame(records)


def add_options_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add options-specific features for ML training."""
    df = df.copy()

    r = 0.05  # Assume constant risk-free rate

    # Arbitrage detection features
    arb_features = []
    for _, row in df.iterrows():
        conv_rev = detect_conversion_reversal(
            row["underlying_price"], row["strike"], row["T"], r,
            row["call_bid"], row["call_ask"], row["put_bid"], row["put_ask"]
        )

        pcp = detect_put_call_parity_violation(
            row["underlying_price"], row["strike"], row["T"], r,
            row["call_mid"], row["put_mid"]
        )

        arb_features.append({
            "conversion_edge": conv_rev["conversion_edge"],
            "reversal_edge": conv_rev["reversal_edge"],
            "conversion_signal": conv_rev["conversion_signal"],
            "reversal_signal": conv_rev["reversal_signal"],
            "pcp_violation": pcp["parity_violation"],
            "pcp_violation_pct": pcp["violation_pct"],
        })

    arb_df = pd.DataFrame(arb_features)
    df = pd.concat([df.reset_index(drop=True), arb_df], axis=1)

    # Additional features
    df["call_spread_pct"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"]
    df["put_spread_pct"] = (df["put_ask"] - df["put_bid"]) / df["put_mid"]
    df["iv_rank"] = df.groupby("underlying")["iv"].rank(pct=True)
    df["delta_abs"] = abs(df["delta"])
    df["gamma_dollar"] = df["gamma"] * df["underlying_price"]**2 / 100

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_options_model(
    strategy_name: str,
    df: pd.DataFrame,
    model_id: str = "v1"
) -> Dict:
    """Train an options strategy model."""
    logger.info(f"Training {strategy_name} options strategy...")

    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
        import joblib

        # Add features
        df = add_options_features(df)

        # Create target based on strategy
        if strategy_name == "conversion_reversal":
            df["target"] = ((df["conversion_signal"] == 1) | (df["reversal_signal"] == 1)).astype(int)
        elif strategy_name == "put_call_parity":
            df["target"] = (abs(df["pcp_violation_pct"]) > 0.5).astype(int)
        elif strategy_name == "volatility_surface":
            df["target"] = (df["iv_rank"] > 0.8).astype(int)  # High IV opportunities
        else:
            df["target"] = df["has_arb"]

        # Feature columns
        exclude_cols = [
            "underlying", "underlying_price", "strike", "expiry_days", "T",
            "target", "has_arb", "conversion_signal", "reversal_signal"
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]

        X = df[feature_cols].fillna(0)
        y = df["target"]

        if len(y.unique()) < 2:
            logger.warning(f"Single class in target for {strategy_name}")
            return {"success": False, "error": "Single class target"}

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Save model
        settings = get_settings()
        model_path = settings.models_dir / f"options_{strategy_name}_{model_id}.pkl"
        joblib.dump({
            "model": model,
            "feature_cols": feature_cols,
            "metrics": {
                "accuracy": accuracy,
                "auc": auc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            },
            "trained_at": datetime.now().isoformat(),
        }, model_path)

        logger.info(f"  {strategy_name}: AUC={auc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")

        return {
            "success": True,
            "strategy": strategy_name,
            "accuracy": accuracy,
            "auc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "samples": len(df),
        }

    except Exception as e:
        logger.error(f"Failed to train {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "strategy": strategy_name, "error": str(e)}


def train_all_options_strategies(parallel: bool = True) -> Dict[str, Dict]:
    """Train all options strategy models."""
    logger.info("=" * 70)
    logger.info("OPTIONS STRATEGY TRAINING")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    results = {}

    if parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for strategy_name, strategy_config in OPTIONS_STRATEGIES.items():
                # Generate synthetic data for this strategy
                df = generate_synthetic_options_data(strategy_config["underlyings"])

                future = executor.submit(train_options_model, strategy_name, df)
                futures[future] = strategy_name

            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    result = future.result()
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"{strategy_name} failed: {e}")
                    results[strategy_name] = {"success": False, "error": str(e)}
    else:
        for strategy_name, strategy_config in OPTIONS_STRATEGIES.items():
            df = generate_synthetic_options_data(strategy_config["underlyings"])
            result = train_options_model(strategy_name, df)
            results[strategy_name] = result

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("OPTIONS TRAINING SUMMARY")
    logger.info("=" * 70)

    successful = 0
    for name, result in results.items():
        if result.get("success"):
            successful += 1
            auc = result.get("auc", 0)
            status = "[PASS]" if auc >= 0.52 else "[FAIL]"
            logger.info(f"  {name}: AUC={auc:.4f} {status}")
        else:
            logger.info(f"  {name}: FAILED - {result.get('error', 'Unknown')}")

    logger.info(f"\nSuccessful: {successful}/{len(OPTIONS_STRATEGIES)}")
    logger.info("=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Options Strategies")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=15, help="Training interval (seconds)")
    parser.add_argument("--parallel", action="store_true", default=True, help="Train in parallel")
    parser.add_argument("--strategies", type=str, help="Comma-separated strategies")

    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/options_training_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )

    # Filter strategies
    if args.strategies:
        strategy_filter = [s.strip().lower() for s in args.strategies.split(",")]
        global OPTIONS_STRATEGIES
        OPTIONS_STRATEGIES = {k: v for k, v in OPTIONS_STRATEGIES.items() if k in strategy_filter}

    logger.info("=" * 70)
    logger.info("ALPHA LOOP CAPITAL - OPTIONS STRATEGY TRAINING")
    logger.info("=" * 70)
    logger.info(f"Strategies: {list(OPTIONS_STRATEGIES.keys())}")
    logger.info(f"Interval: {args.interval}s")
    logger.info("=" * 70)

    if args.continuous:
        logger.info(f"Running in continuous mode (interval: {args.interval}s)")

        while True:
            try:
                results = train_all_options_strategies(args.parallel)

                if results:
                    successful = sum(1 for r in results.values() if r.get("success"))
                    logger.info(f"Cycle complete: {successful}/{len(results)} successful")

                logger.info(f"Next training in {args.interval} seconds...")
                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Training interrupted")
                break
            except Exception as e:
                logger.error(f"Training cycle failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"Retrying in {args.interval} seconds...")
                time.sleep(args.interval)
    else:
        results = train_all_options_strategies(args.parallel)

        if results:
            successful = sum(1 for r in results.values() if r.get("success"))
            sys.exit(0 if successful > 0 else 1)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
