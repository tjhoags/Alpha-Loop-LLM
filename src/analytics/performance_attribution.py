"""
Performance Attribution System

Decomposing portfolio returns to understand what drove performance.

Why Basic Approaches Fail:
- "We made money" isn't enough - need to know WHY
- Can't improve without knowing what worked and what didn't
- Luck vs skill - attribution separates them
- Risk-adjusted returns matter more than raw returns

Our Creative Philosophy:
- Brinson-Fachler attribution (allocation + selection + interaction)
- Factor attribution (Fama-French + Momentum + Quality)
- Sector/industry attribution
- Individual position attribution
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, Information Ratio)
- Alpha decomposition (skill vs beta exposure)
- Transaction cost attribution

Elite institutions demand attribution:
- Every hedge fund investor wants attribution reports
- SEC requires performance attribution for RIAs
- Top funds attribute daily, not monthly

Author: Tom Hogan
Date: 2025-12-09
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Performance attribution methods"""

    BRINSON_FACHLER = "brinson_fachler"  # Allocation + Selection + Interaction
    FACTOR_ATTRIBUTION = "factor_attribution"  # Fama-French factors
    SECTOR_ATTRIBUTION = "sector_attribution"  # Sector-based
    SECURITY_SELECTION = "security_selection"  # Individual positions
    RISK_ATTRIBUTION = "risk_attribution"  # Risk contribution to return


class PerformanceMetric(Enum):
    """Performance metrics"""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    TREYNOR_RATIO = "treynor_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"


@dataclass
class AttributionResult:
    """Performance attribution result"""

    method: AttributionMethod
    total_return: float  # Portfolio total return
    benchmark_return: float  # Benchmark return
    active_return: float  # Portfolio - Benchmark
    attribution_breakdown: Dict[str, float]  # Component → contribution
    risk_adjusted_metrics: Dict[PerformanceMetric, float]
    top_contributors: List[Tuple[str, float]]  # Top 10 positive contributors
    top_detractors: List[Tuple[str, float]]  # Top 10 negative contributors
    alpha: float  # Risk-adjusted excess return
    beta: float  # Market exposure
    r_squared: float  # % of variance explained by market
    tracking_error: float  # Volatility of active returns
    information_ratio: float  # Active return / Tracking error
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BrinsonAttribution:
    """Brinson-Fachler attribution components"""

    allocation_effect: float  # Over/underweight sectors
    selection_effect: float  # Picking winners within sectors
    interaction_effect: float  # Combined allocation + selection
    total_active_return: float  # Sum of all effects


@dataclass
class FactorAttribution:
    """Factor-based attribution"""

    market_contribution: float  # Market factor (CAPM)
    size_contribution: float  # SMB (Small Minus Big)
    value_contribution: float  # HML (High Minus Low)
    momentum_contribution: float  # UMD (Up Minus Down)
    quality_contribution: float  # QMJ (Quality Minus Junk)
    residual_alpha: float  # Unexplained (true alpha)
    factor_exposures: Dict[str, float]  # Factor → beta


class PerformanceAttributionEngine:
    """
    Institutional-grade performance attribution.

    Features:
    - Multiple attribution methods (Brinson, Factor, Sector)
    - Risk-adjusted performance metrics
    - Alpha decomposition (skill vs luck)
    - Transaction cost impact
    - Drawdown analysis
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,  # Current 1-year Treasury
        confidence_level: float = 0.95,  # For confidence intervals
    ):
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level

        logger.info("PerformanceAttributionEngine initialized")

    def attribute_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        holdings: Optional[pd.DataFrame] = None,  # Date × Asset weights
        sector_weights: Optional[pd.DataFrame] = None,  # Portfolio vs Benchmark sectors
        factor_returns: Optional[pd.DataFrame] = None,  # Fama-French factors
        method: AttributionMethod = AttributionMethod.BRINSON_FACHLER,
    ) -> AttributionResult:
        """
        Attribute portfolio performance.

        Args:
            portfolio_returns: Portfolio returns (time series)
            benchmark_returns: Benchmark returns (time series)
            holdings: Portfolio holdings over time (for security-level attribution)
            sector_weights: Sector weights (Portfolio vs Benchmark)
            factor_returns: Factor returns (Market, SMB, HML, UMD, QMJ)
            method: Attribution method

        Returns:
            AttributionResult with performance breakdown
        """
        # Align returns
        aligned = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        portfolio_ret = aligned["portfolio"]
        benchmark_ret = aligned["benchmark"]

        # Calculate basic metrics
        total_return = (1 + portfolio_ret).prod() - 1
        benchmark_return = (1 + benchmark_ret).prod() - 1
        active_return = total_return - benchmark_return

        # Calculate risk-adjusted metrics
        risk_metrics = self._calculate_risk_metrics(
            portfolio_ret, benchmark_ret, aligned["portfolio"] - aligned["benchmark"]
        )

        # Calculate alpha and beta
        alpha, beta, r_squared = self._calculate_alpha_beta(portfolio_ret, benchmark_ret)

        # Tracking error and information ratio
        active_returns = portfolio_ret - benchmark_ret
        tracking_error = active_returns.std() * np.sqrt(252)  # Annualized
        information_ratio = (
            (active_return / tracking_error) if tracking_error > 0 else 0.0
        )

        # Attribution breakdown by method
        if method == AttributionMethod.BRINSON_FACHLER and sector_weights is not None:
            attribution = self._brinson_attribution(sector_weights)
            attribution_breakdown = {
                "allocation": attribution.allocation_effect,
                "selection": attribution.selection_effect,
                "interaction": attribution.interaction_effect,
            }
        elif method == AttributionMethod.FACTOR_ATTRIBUTION and factor_returns is not None:
            attribution = self._factor_attribution(portfolio_ret, factor_returns)
            attribution_breakdown = {
                "market": attribution.market_contribution,
                "size": attribution.size_contribution,
                "value": attribution.value_contribution,
                "momentum": attribution.momentum_contribution,
                "quality": attribution.quality_contribution,
                "alpha": attribution.residual_alpha,
            }
        elif method == AttributionMethod.SECURITY_SELECTION and holdings is not None:
            attribution_breakdown = self._security_attribution(holdings, portfolio_ret)
        else:
            # Default: simple active return
            attribution_breakdown = {"active_return": float(active_return)}

        # Top contributors and detractors
        if isinstance(attribution_breakdown, dict):
            sorted_items = sorted(
                attribution_breakdown.items(), key=lambda x: x[1], reverse=True
            )
            top_contributors = sorted_items[:10]
            top_detractors = sorted_items[-10:]
        else:
            top_contributors = []
            top_detractors = []

        return AttributionResult(
            method=method,
            total_return=float(total_return),
            benchmark_return=float(benchmark_return),
            active_return=float(active_return),
            attribution_breakdown=attribution_breakdown,
            risk_adjusted_metrics=risk_metrics,
            top_contributors=top_contributors,
            top_detractors=top_detractors,
            alpha=float(alpha),
            beta=float(beta),
            r_squared=float(r_squared),
            tracking_error=float(tracking_error),
            information_ratio=float(information_ratio),
        )

    def _calculate_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        active_returns: pd.Series,
    ) -> Dict[PerformanceMetric, float]:
        """Calculate comprehensive risk-adjusted performance metrics"""
        metrics = {}

        # Total return
        metrics[PerformanceMetric.TOTAL_RETURN] = (1 + portfolio_returns).prod() - 1

        # Sharpe Ratio
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe = (
            (excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            if excess_returns.std() > 0
            else 0.0
        )
        metrics[PerformanceMetric.SHARPE_RATIO] = sharpe

        # Sortino Ratio (downside deviation only)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (
            (portfolio_returns.mean() * 252 - self.risk_free_rate) / downside_std
            if downside_std > 0
            else 0.0
        )
        metrics[PerformanceMetric.SORTINO_RATIO] = sortino

        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics[PerformanceMetric.MAX_DRAWDOWN] = max_drawdown

        # Calmar Ratio (return / max drawdown)
        annual_return = (1 + portfolio_returns).prod() ** (
            252 / len(portfolio_returns)
        ) - 1
        calmar = (
            annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        )
        metrics[PerformanceMetric.CALMAR_RATIO] = calmar

        # Information Ratio (already calculated in main function)
        tracking_error = active_returns.std() * np.sqrt(252)
        active_return = (1 + active_returns).prod() - 1
        info_ratio = (
            active_return / tracking_error if tracking_error > 0 else 0.0
        )
        metrics[PerformanceMetric.INFORMATION_RATIO] = info_ratio

        # Win Rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        metrics[PerformanceMetric.WIN_RATE] = win_rate

        # Profit Factor
        gains = portfolio_returns[portfolio_returns > 0].sum()
        losses = abs(portfolio_returns[portfolio_returns < 0].sum())
        profit_factor = gains / losses if losses != 0 else 0.0
        metrics[PerformanceMetric.PROFIT_FACTOR] = profit_factor

        return metrics

    def _calculate_alpha_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Calculate alpha and beta using regression.

        CAPM: R_p = α + β*R_m + ε
        """
        # Excess returns
        excess_portfolio = portfolio_returns - self.risk_free_rate / 252
        excess_benchmark = benchmark_returns - self.risk_free_rate / 252

        # Linear regression
        if len(excess_portfolio) > 1 and excess_benchmark.std() > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                excess_benchmark, excess_portfolio
            )

            beta = slope
            alpha = intercept * 252  # Annualized
            r_squared = r_value**2
        else:
            beta = 1.0
            alpha = 0.0
            r_squared = 0.0

        return alpha, beta, r_squared

    def _brinson_attribution(
        self, sector_weights: pd.DataFrame
    ) -> BrinsonAttribution:
        """
        Brinson-Fachler attribution.

        Decompose active return into:
        1. Allocation: Over/underweight sectors
        2. Selection: Picking winners within sectors
        3. Interaction: Combined effect

        Formula:
        - Allocation = (w_p - w_b) * (R_b - R_B)
        - Selection = w_b * (R_p - R_b)
        - Interaction = (w_p - w_b) * (R_p - R_b)

        where:
        - w_p = portfolio sector weight
        - w_b = benchmark sector weight
        - R_p = portfolio sector return
        - R_b = benchmark sector return
        - R_B = total benchmark return
        """
        # Expected columns: sector, portfolio_weight, benchmark_weight,
        #                   portfolio_return, benchmark_return

        if sector_weights.empty:
            return BrinsonAttribution(0.0, 0.0, 0.0, 0.0)

        # Total benchmark return (weighted average)
        total_benchmark_return = (
            sector_weights["benchmark_weight"] * sector_weights["benchmark_return"]
        ).sum()

        # Allocation effect
        allocation = (
            (sector_weights["portfolio_weight"] - sector_weights["benchmark_weight"])
            * (sector_weights["benchmark_return"] - total_benchmark_return)
        ).sum()

        # Selection effect
        selection = (
            sector_weights["benchmark_weight"]
            * (sector_weights["portfolio_return"] - sector_weights["benchmark_return"])
        ).sum()

        # Interaction effect
        interaction = (
            (sector_weights["portfolio_weight"] - sector_weights["benchmark_weight"])
            * (sector_weights["portfolio_return"] - sector_weights["benchmark_return"])
        ).sum()

        total_active = allocation + selection + interaction

        return BrinsonAttribution(
            allocation_effect=float(allocation),
            selection_effect=float(selection),
            interaction_effect=float(interaction),
            total_active_return=float(total_active),
        )

    def _factor_attribution(
        self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> FactorAttribution:
        """
        Factor-based attribution using Fama-French factors.

        R_p = α + β_MKT*MKT + β_SMB*SMB + β_HML*HML + β_UMD*UMD + β_QMJ*QMJ + ε

        Factor contributions = β_i * mean(Factor_i)
        """
        # Align returns
        aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        y = aligned.iloc[:, 0].values  # Portfolio returns

        # Prepare factor matrix
        factor_names = ["MKT", "SMB", "HML", "UMD", "QMJ"]
        X_factors = []
        available_factors = []

        for factor in factor_names:
            if factor in aligned.columns:
                X_factors.append(aligned[factor].values)
                available_factors.append(factor)

        if len(X_factors) == 0:
            # No factors available - return zero attribution
            return FactorAttribution(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

        X = np.column_stack(X_factors)

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            alpha = coeffs[0] * 252  # Annualized
            betas = coeffs[1:]
        except:
            alpha = 0.0
            betas = np.zeros(len(available_factors))

        # Calculate factor contributions
        factor_contributions = {}
        for i, factor in enumerate(available_factors):
            mean_factor_return = aligned[factor].mean() * 252  # Annualized
            contribution = betas[i] * mean_factor_return
            factor_contributions[factor] = float(contribution)

        # Extract specific factors
        market_contrib = factor_contributions.get("MKT", 0.0)
        size_contrib = factor_contributions.get("SMB", 0.0)
        value_contrib = factor_contributions.get("HML", 0.0)
        momentum_contrib = factor_contributions.get("UMD", 0.0)
        quality_contrib = factor_contributions.get("QMJ", 0.0)

        # Factor exposures (betas)
        factor_exposures = {
            factor: float(betas[i]) for i, factor in enumerate(available_factors)
        }

        return FactorAttribution(
            market_contribution=market_contrib,
            size_contribution=size_contrib,
            value_contribution=value_contrib,
            momentum_contribution=momentum_contrib,
            quality_contribution=quality_contrib,
            residual_alpha=float(alpha),
            factor_exposures=factor_exposures,
        )

    def _security_attribution(
        self, holdings: pd.DataFrame, portfolio_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Security-level attribution.

        Contribution = Weight * Return
        """
        # Expected: holdings has columns [date, asset, weight, return]
        if holdings.empty:
            return {}

        # Group by asset
        attribution = {}
        for asset in holdings["asset"].unique():
            asset_data = holdings[holdings["asset"] == asset]

            # Average weight
            avg_weight = asset_data["weight"].mean()

            # Total return (if return column exists)
            if "return" in asset_data.columns:
                total_return = (1 + asset_data["return"]).prod() - 1
            else:
                total_return = 0.0

            # Contribution
            contribution = avg_weight * total_return
            attribution[asset] = float(contribution)

        return attribution

    def calculate_drawdown_analysis(
        self, returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis.

        Returns:
        - Current drawdown
        - Max drawdown
        - Max drawdown duration
        - Average drawdown
        - Drawdown frequency
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Max drawdown
        max_dd = drawdown.min()

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_dd_duration = (
            np.mean(drawdown_periods) if drawdown_periods else 0
        )

        # Drawdown frequency
        dd_frequency = len(drawdown_periods) / len(returns) * 252  # Annualized

        # Average drawdown magnitude
        avg_dd_magnitude = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0

        return {
            "current_drawdown": float(current_dd),
            "max_drawdown": float(max_dd),
            "max_drawdown_duration_days": int(max_dd_duration),
            "avg_drawdown_duration_days": float(avg_dd_duration),
            "drawdown_frequency_per_year": float(dd_frequency),
            "avg_drawdown_magnitude": float(avg_dd_magnitude),
        }

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252,  # 1 year
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Useful for seeing how performance evolves over time.
        """
        rolling_data = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window : i]
            window_benchmark = benchmark_returns.iloc[i - window : i]

            # Sharpe ratio
            excess = window_returns - self.risk_free_rate / 252
            sharpe = (
                (excess.mean() / excess.std() * np.sqrt(252))
                if excess.std() > 0
                else 0.0
            )

            # Alpha and beta
            alpha, beta, r_sq = self._calculate_alpha_beta(
                window_returns, window_benchmark
            )

            # Drawdown
            cumulative = (1 + window_returns).cumprod()
            running_max = cumulative.expanding().max()
            dd = ((cumulative - running_max) / running_max).min()

            rolling_data.append(
                {
                    "date": returns.index[i],
                    "sharpe_ratio": sharpe,
                    "alpha": alpha,
                    "beta": beta,
                    "r_squared": r_sq,
                    "max_drawdown": dd,
                }
            )

        return pd.DataFrame(rolling_data).set_index("date")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n_days = len(dates)

    # Simulate portfolio and benchmark returns
    portfolio_returns = pd.Series(
        np.random.randn(n_days) * 0.012 + 0.0006, index=dates
    )
    benchmark_returns = pd.Series(
        np.random.randn(n_days) * 0.010 + 0.0004, index=dates
    )

    # Sample sector weights for Brinson attribution
    sectors = ["Technology", "Healthcare", "Financials", "Energy"]
    sector_data = {
        "sector": sectors,
        "portfolio_weight": [0.35, 0.25, 0.25, 0.15],
        "benchmark_weight": [0.30, 0.25, 0.30, 0.15],
        "portfolio_return": [0.15, 0.10, 0.05, -0.05],
        "benchmark_return": [0.12, 0.08, 0.06, -0.03],
    }
    sector_weights = pd.DataFrame(sector_data)

    # Sample factor returns (Fama-French)
    factor_data = {
        "MKT": np.random.randn(n_days) * 0.01 + 0.0003,
        "SMB": np.random.randn(n_days) * 0.005,
        "HML": np.random.randn(n_days) * 0.005,
        "UMD": np.random.randn(n_days) * 0.006,
        "QMJ": np.random.randn(n_days) * 0.004,
    }
    factor_returns = pd.DataFrame(factor_data, index=dates)

    # Initialize engine
    engine = PerformanceAttributionEngine()

    # Test 1: Brinson Attribution
    result_brinson = engine.attribute_performance(
        portfolio_returns,
        benchmark_returns,
        sector_weights=sector_weights,
        method=AttributionMethod.BRINSON_FACHLER,
    )

    print("\n=== Brinson-Fachler Attribution ===")
    print(f"Total Return: {result_brinson.total_return:.2%}")
    print(f"Benchmark Return: {result_brinson.benchmark_return:.2%}")
    print(f"Active Return: {result_brinson.active_return:.2%}")
    print(f"Alpha: {result_brinson.alpha:.2%}")
    print(f"Beta: {result_brinson.beta:.2f}")
    print(f"Information Ratio: {result_brinson.information_ratio:.2f}")
    print("\nAttribution Breakdown:")
    for component, value in result_brinson.attribution_breakdown.items():
        print(f"  {component}: {value:.2%}")

    # Test 2: Factor Attribution
    result_factor = engine.attribute_performance(
        portfolio_returns,
        benchmark_returns,
        factor_returns=factor_returns,
        method=AttributionMethod.FACTOR_ATTRIBUTION,
    )

    print("\n=== Factor Attribution ===")
    print(f"Total Return: {result_factor.total_return:.2%}")
    print("\nFactor Contributions:")
    for factor, contrib in result_factor.attribution_breakdown.items():
        print(f"  {factor}: {contrib:.2%}")

    # Test 3: Risk Metrics
    print("\n=== Risk-Adjusted Metrics ===")
    for metric, value in result_brinson.risk_adjusted_metrics.items():
        if metric == PerformanceMetric.TOTAL_RETURN:
            print(f"{metric.value}: {value:.2%}")
        elif metric in [PerformanceMetric.WIN_RATE]:
            print(f"{metric.value}: {value:.1%}")
        else:
            print(f"{metric.value}: {value:.2f}")

    # Test 4: Drawdown Analysis
    dd_analysis = engine.calculate_drawdown_analysis(portfolio_returns)
    print("\n=== Drawdown Analysis ===")
    for key, value in dd_analysis.items():
        if "magnitude" in key or "drawdown" in key and "duration" not in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.1f}")

    print("\n✅ Performance Attribution System - Tom Hogan")
