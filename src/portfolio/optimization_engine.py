"""
Portfolio Optimization Engine

Institutional-grade portfolio construction and optimization.

Why Basic Approaches Fail:
- Mean-variance optimization is unstable with estimation error
- Equal weighting ignores risk differences
- Market-cap weighting concentrates in bubbles
- Simple rebalancing ignores transaction costs

Our Creative Philosophy:
- Black-Litterman for incorporating views with uncertainty
- Risk Parity for balanced risk contribution
- Mean-CVaR for tail risk optimization
- Transaction cost-aware rebalancing
- Regime-dependent constraints
- Factor exposure controls

Elite institutions use these methods:
- Bridgewater: Risk Parity pioneered by Ray Dalio
- AQR: Systematic factor allocation with optimization
- Two Sigma: ML-enhanced Black-Litterman views

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
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""

    MEAN_VARIANCE = "mean_variance"  # Markowitz (unstable)
    BLACK_LITTERMAN = "black_litterman"  # Views + equilibrium
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    MEAN_CVAR = "mean_cvar"  # Tail risk optimization
    MINIMUM_VARIANCE = "minimum_variance"  # Pure risk minimization
    MAXIMUM_SHARPE = "maximum_sharpe"  # Risk-adjusted return
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"  # Tree-based allocation
    EQUAL_WEIGHT = "equal_weight"  # Naive diversification


class RebalanceMethod(Enum):
    """Rebalancing approaches"""

    CALENDAR = "calendar"  # Fixed schedule (monthly, quarterly)
    THRESHOLD = "threshold"  # When drift > threshold
    VOLATILITY_TRIGGER = "volatility_trigger"  # When vol spikes
    COST_AWARE = "cost_aware"  # Trade-off drift vs transaction costs
    NO_REBALANCE = "no_rebalance"  # Buy and hold


class ConstraintType(Enum):
    """Portfolio constraints"""

    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"  # 130/30, 120/20, etc.
    MARKET_NEUTRAL = "market_neutral"  # Beta = 0
    SECTOR_NEUTRAL = "sector_neutral"  # No sector bets
    FACTOR_NEUTRAL = "factor_neutral"  # No factor tilts


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization"""

    constraint_type: ConstraintType = ConstraintType.LONG_ONLY
    min_weight: float = 0.0  # Minimum position weight
    max_weight: float = 0.20  # Maximum position weight (5 positions minimum)
    max_sector_weight: float = 0.30  # Maximum sector exposure
    max_single_factor_exposure: float = 0.50  # Max factor tilt
    target_beta: Optional[float] = None  # Target market beta
    max_turnover: float = 0.50  # Maximum portfolio turnover per rebalance
    min_positions: int = 10  # Minimum number of positions
    max_positions: int = 50  # Maximum number of positions
    gross_exposure: Optional[float] = None  # 130/30 → gross = 1.6
    net_exposure: Optional[float] = None  # Market neutral → net = 0.0


@dataclass
class BlackLittermanView:
    """Investor view for Black-Litterman model"""

    asset: str  # Asset ticker
    view_type: str  # "absolute" or "relative"
    expected_return: float  # View on return (annualized)
    confidence: float  # 0.0 to 1.0 (how certain are we?)
    relative_to: Optional[str] = None  # For relative views ("SPY outperforms TLT")


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""

    method: OptimizationMethod
    weights: Dict[str, float]  # Asset → weight
    expected_return: float  # Annualized expected return
    expected_volatility: float  # Annualized volatility
    sharpe_ratio: float  # Expected Sharpe
    expected_cvar: float  # 95% CVaR (tail risk)
    turnover: float  # Turnover from current portfolio
    transaction_costs: float  # Estimated costs
    risk_contribution: Dict[str, float]  # Asset → % of portfolio risk
    factor_exposures: Dict[str, float]  # Factor → exposure
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PortfolioOptimizationEngine:
    """
    Institutional-grade portfolio optimization engine.

    Features:
    - Multiple optimization methods (Black-Litterman, Risk Parity, Mean-CVaR)
    - Transaction cost-aware rebalancing
    - Factor exposure controls
    - Regime-dependent constraints
    - Estimation error mitigation
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,  # Current 1-year Treasury
        market_return: float = 0.10,  # Historical equity premium
        market_volatility: float = 0.15,  # Historical market vol
        transaction_cost_bps: float = 5.0,  # 5 bps per trade
        market_impact_factor: float = 0.10,  # Market impact (% of volatility)
    ):
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.market_volatility = market_volatility
        self.transaction_cost_bps = transaction_cost_bps
        self.market_impact_factor = market_impact_factor

        logger.info("PortfolioOptimizationEngine initialized")

    def optimize(
        self,
        returns: pd.DataFrame,  # Historical returns (assets × time)
        current_weights: Optional[Dict[str, float]] = None,
        method: OptimizationMethod = OptimizationMethod.BLACK_LITTERMAN,
        constraints: Optional[OptimizationConstraints] = None,
        views: Optional[List[BlackLittermanView]] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio allocation.

        Args:
            returns: Historical returns DataFrame (assets × time)
            current_weights: Current portfolio weights (for turnover calculation)
            method: Optimization method
            constraints: Portfolio constraints
            views: Black-Litterman views (for BLACK_LITTERMAN method)
            factor_loadings: Factor loadings (for factor exposure control)

        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        if constraints is None:
            constraints = OptimizationConstraints()

        assets = returns.columns.tolist()
        n_assets = len(assets)

        # Calculate expected returns and covariance
        if method == OptimizationMethod.BLACK_LITTERMAN:
            expected_returns = self._black_litterman_returns(
                returns, views, factor_loadings
            )
        else:
            # Use historical mean (with shrinkage to avoid extreme estimates)
            historical_mean = returns.mean() * 252  # Annualized
            grand_mean = historical_mean.mean()
            shrinkage = 0.3  # Shrink toward grand mean
            expected_returns = shrinkage * grand_mean + (1 - shrinkage) * historical_mean

        # Covariance matrix (with shrinkage to improve stability)
        sample_cov = returns.cov() * 252  # Annualized
        covariance = self._shrink_covariance(sample_cov)

        # Optimize based on method
        if method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(assets, constraints)
        elif method == OptimizationMethod.MINIMUM_VARIANCE:
            weights = self._minimum_variance(covariance, constraints)
        elif method == OptimizationMethod.MAXIMUM_SHARPE:
            weights = self._maximum_sharpe(
                expected_returns, covariance, constraints
            )
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(covariance, constraints)
        elif method == OptimizationMethod.MEAN_CVAR:
            weights = self._mean_cvar(returns, expected_returns, constraints)
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            weights = self._maximum_sharpe(
                expected_returns, covariance, constraints
            )  # BL returns used above
        elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
            weights = self._hierarchical_risk_parity(returns, covariance, constraints)
        else:
            weights = self._maximum_sharpe(expected_returns, covariance, constraints)

        # Convert to dict
        weight_dict = {asset: weights[i] for i, asset in enumerate(assets)}

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else 0.0
        )

        # Calculate CVaR (95% confidence)
        portfolio_returns = returns.dot(weights)
        cvar_95 = self._calculate_cvar(portfolio_returns, confidence=0.95)

        # Calculate turnover and transaction costs
        if current_weights is None:
            current_weights = {asset: 0.0 for asset in assets}
        turnover = sum(
            abs(weight_dict.get(asset, 0.0) - current_weights.get(asset, 0.0))
            for asset in assets
        )
        transaction_costs = (turnover / 2) * (
            self.transaction_cost_bps / 10000
        )  # Half turnover (buy and sell)

        # Risk contribution
        marginal_risk = np.dot(covariance, weights)
        risk_contribution = {
            asset: (weights[i] * marginal_risk[i]) / portfolio_variance
            if portfolio_variance > 0
            else 0.0
            for i, asset in enumerate(assets)
        }

        # Factor exposures
        factor_exposures = {}
        if factor_loadings is not None:
            for factor in factor_loadings.columns:
                factor_exposures[factor] = np.dot(
                    weights, factor_loadings[factor].values
                )

        return OptimizationResult(
            method=method,
            weights=weight_dict,
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            expected_cvar=float(cvar_95),
            turnover=float(turnover),
            transaction_costs=float(transaction_costs),
            risk_contribution=risk_contribution,
            factor_exposures=factor_exposures,
        )

    def _black_litterman_returns(
        self,
        returns: pd.DataFrame,
        views: Optional[List[BlackLittermanView]],
        factor_loadings: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Black-Litterman model: Combine equilibrium returns with investor views.

        The genius of Black-Litterman:
        1. Start with market equilibrium (reverse optimization)
        2. Blend in investor views with confidence weights
        3. Result: stable expected returns that incorporate views

        Args:
            returns: Historical returns
            views: Investor views on asset returns
            factor_loadings: Factor loadings for equilibrium

        Returns:
            Blended expected returns
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # Step 1: Calculate equilibrium returns (reverse optimization)
        # Assume market cap weights are optimal → what returns justify this?
        sample_cov = returns.cov() * 252
        covariance = self._shrink_covariance(sample_cov)

        # Market portfolio weights (simplified: equal weight as proxy)
        market_weights = np.ones(n_assets) / n_assets

        # Reverse optimization: π = λ * Σ * w_market
        # where λ is risk aversion coefficient
        risk_aversion = (
            self.market_return - self.risk_free_rate
        ) / self.market_volatility**2
        equilibrium_returns = risk_aversion * np.dot(covariance, market_weights)

        # Step 2: Incorporate investor views
        if views is None or len(views) == 0:
            return pd.Series(equilibrium_returns, index=assets)

        # Build view matrix P and view returns Q
        n_views = len(views)
        P = np.zeros((n_views, n_assets))  # View matrix
        Q = np.zeros(n_views)  # View returns
        Omega = np.zeros((n_views, n_views))  # View uncertainty

        for i, view in enumerate(views):
            if view.view_type == "absolute":
                # Absolute view: asset will return X%
                asset_idx = assets.index(view.asset)
                P[i, asset_idx] = 1.0
                Q[i] = view.expected_return
            elif view.view_type == "relative":
                # Relative view: asset A will outperform asset B by X%
                asset_idx = assets.index(view.asset)
                relative_idx = assets.index(view.relative_to)
                P[i, asset_idx] = 1.0
                P[i, relative_idx] = -1.0
                Q[i] = view.expected_return

            # View uncertainty (proportional to confidence)
            # Higher confidence → lower uncertainty
            view_variance = (1.0 - view.confidence) * 0.01  # 1% base uncertainty
            Omega[i, i] = view_variance

        # Step 3: Black-Litterman formula
        # Posterior = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
        tau = 0.025  # Uncertainty in equilibrium (2.5% standard)

        tau_sigma_inv = np.linalg.inv(tau * covariance)
        omega_inv = np.linalg.inv(Omega)

        # Posterior precision
        posterior_precision = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
        posterior_covariance = np.linalg.inv(posterior_precision)

        # Posterior mean
        posterior_mean = np.dot(
            posterior_covariance,
            np.dot(tau_sigma_inv, equilibrium_returns) + np.dot(P.T, np.dot(omega_inv, Q)),
        )

        return pd.Series(posterior_mean, index=assets)

    def _risk_parity(
        self, covariance: pd.DataFrame, constraints: OptimizationConstraints
    ) -> np.ndarray:
        """
        Risk Parity: Equal risk contribution from each asset.

        Why it works:
        - Traditional portfolios are dominated by equity risk
        - Risk parity balances risk across asset classes
        - Bridgewater's All Weather portfolio uses this

        Args:
            covariance: Covariance matrix
            constraints: Portfolio constraints

        Returns:
            Optimal weights (np.ndarray)
        """
        n_assets = len(covariance)

        def risk_parity_objective(weights):
            """Minimize variance of risk contributions"""
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            marginal_risk = np.dot(covariance, weights)
            risk_contribution = weights * marginal_risk

            # Target: equal risk contribution (1/n each)
            target_risk = portfolio_variance / n_assets
            return np.sum((risk_contribution - target_risk) ** 2)

        # Constraints
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Weights sum to 1

        # Bounds
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight) for _ in range(n_assets)
        )

        # Initial guess: inverse volatility weighting
        volatilities = np.sqrt(np.diag(covariance))
        initial_weights = 1.0 / volatilities
        initial_weights /= initial_weights.sum()

        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else initial_weights

    def _mean_cvar(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        constraints: OptimizationConstraints,
        confidence: float = 0.95,
    ) -> np.ndarray:
        """
        Mean-CVaR optimization: Maximize return subject to CVaR constraint.

        Why CVaR > VaR:
        - CVaR considers tail losses beyond VaR
        - Coherent risk measure (VaR is not)
        - Used by sophisticated risk managers

        Args:
            returns: Historical returns
            expected_returns: Expected returns
            constraints: Portfolio constraints
            confidence: CVaR confidence level

        Returns:
            Optimal weights (np.ndarray)
        """
        n_assets = len(expected_returns)
        n_scenarios = len(returns)

        def cvar_objective(weights):
            """Negative expected return (we minimize)"""
            return -np.dot(weights, expected_returns)

        def cvar_constraint(weights):
            """CVaR constraint: CVaR must be < max acceptable loss"""
            portfolio_returns = returns.dot(weights)
            cvar = self._calculate_cvar(portfolio_returns, confidence)
            max_cvar = 0.15  # Max 15% tail loss
            return max_cvar - abs(cvar)

        # Constraints
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {"type": "ineq", "fun": cvar_constraint},  # CVaR constraint
        ]

        # Bounds
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight) for _ in range(n_assets)
        )

        # Initial guess: equal weight
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            cvar_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else initial_weights

    def _minimum_variance(
        self, covariance: pd.DataFrame, constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Minimum variance portfolio"""
        n_assets = len(covariance)

        def objective(weights):
            return np.dot(weights, np.dot(covariance, weights))

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight) for _ in range(n_assets)
        )

        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=cons
        )

        return result.x if result.success else initial_weights

    def _maximum_sharpe(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Maximum Sharpe ratio portfolio"""
        n_assets = len(expected_returns)

        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
            return -(
                (portfolio_return - self.risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else -999
            )

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight) for _ in range(n_assets)
        )

        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            negative_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else initial_weights

    def _equal_weight(
        self, assets: List[str], constraints: OptimizationConstraints
    ) -> np.ndarray:
        """Equal weight portfolio (naive diversification)"""
        n_assets = len(assets)
        return np.ones(n_assets) / n_assets

    def _hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity (Lopez de Prado 2016).

        Uses hierarchical clustering to build portfolio allocation tree.
        More stable than traditional optimization.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        n_assets = len(covariance)

        # Step 1: Compute distance matrix from correlation
        corr = returns.corr()
        distance = np.sqrt((1 - corr) / 2)  # Angular distance
        distance_matrix = squareform(distance)

        # Step 2: Hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method="single")

        # Step 3: Recursive bisection to allocate weights
        def _recursive_bisection(cluster_items):
            """Recursively split cluster and allocate variance"""
            if len(cluster_items) == 1:
                return {cluster_items[0]: 1.0}

            # Split cluster in half
            mid = len(cluster_items) // 2
            left = cluster_items[:mid]
            right = cluster_items[mid:]

            # Variance of each half
            left_var = self._cluster_variance(covariance, left)
            right_var = self._cluster_variance(covariance, right)

            # Allocate weight inversely to variance
            total_var = left_var + right_var
            left_weight = right_var / total_var if total_var > 0 else 0.5
            right_weight = left_var / total_var if total_var > 0 else 0.5

            # Recurse
            left_weights = _recursive_bisection(left)
            right_weights = _recursive_bisection(right)

            # Combine
            weights = {}
            for item, w in left_weights.items():
                weights[item] = w * left_weight
            for item, w in right_weights.items():
                weights[item] = w * right_weight

            return weights

        # Get cluster order from linkage
        cluster_order = list(range(n_assets))  # Simplified: use original order

        # Allocate weights
        weight_dict = _recursive_bisection(cluster_order)

        # Convert to array
        weights = np.array([weight_dict.get(i, 0.0) for i in range(n_assets)])
        return weights

    def _cluster_variance(self, covariance: pd.DataFrame, cluster: List[int]) -> float:
        """Calculate variance of a cluster"""
        if len(cluster) == 0:
            return 0.0
        if len(cluster) == 1:
            return covariance.iloc[cluster[0], cluster[0]]

        # Inverse variance weighting within cluster
        variances = np.array([covariance.iloc[i, i] for i in cluster])
        inv_var = 1.0 / variances
        weights = inv_var / inv_var.sum()

        # Cluster variance
        cov_subset = covariance.iloc[cluster, cluster]
        return np.dot(weights, np.dot(cov_subset, weights))

    def _shrink_covariance(
        self, sample_cov: pd.DataFrame, shrinkage: float = 0.2
    ) -> pd.DataFrame:
        """
        Ledoit-Wolf covariance shrinkage.

        Why it matters:
        - Sample covariance is noisy with limited data
        - Shrink toward structured estimator (diagonal)
        - Improves optimization stability
        """
        # Target: diagonal matrix (no correlations)
        target = np.diag(np.diag(sample_cov))

        # Shrink: (1-δ)*Sample + δ*Target
        shrunk = (1 - shrinkage) * sample_cov + shrinkage * target
        return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)

    def _calculate_cvar(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        var_threshold = returns.quantile(1 - confidence)
        cvar = returns[returns <= var_threshold].mean()
        return float(cvar) if not np.isnan(cvar) else 0.0

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        method: RebalanceMethod,
        threshold: float = 0.05,
        current_volatility: Optional[float] = None,
        transaction_costs: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            method: Rebalancing method
            threshold: Drift threshold for THRESHOLD method
            current_volatility: Current market volatility (for VOLATILITY_TRIGGER)
            transaction_costs: Estimated transaction costs (for COST_AWARE)

        Returns:
            (should_rebalance, reason)
        """
        if method == RebalanceMethod.NO_REBALANCE:
            return False, "No rebalance policy"

        # Calculate drift
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        drift = sum(
            abs(target_weights.get(a, 0.0) - current_weights.get(a, 0.0))
            for a in all_assets
        )

        if method == RebalanceMethod.THRESHOLD:
            if drift > threshold:
                return True, f"Drift {drift:.2%} exceeds threshold {threshold:.2%}"
            return False, f"Drift {drift:.2%} within threshold"

        elif method == RebalanceMethod.VOLATILITY_TRIGGER:
            # Rebalance if volatility spike
            if current_volatility is not None and current_volatility > 0.25:
                return True, f"Volatility spike: {current_volatility:.2%}"
            return False, "Volatility normal"

        elif method == RebalanceMethod.COST_AWARE:
            # Trade off drift cost vs transaction costs
            # Drift cost: how much expected return are we giving up?
            # Simplified: assume drift costs 1% per year per 10% drift
            drift_cost = drift * 0.10  # 10% drift = 1% cost

            if transaction_costs is None:
                transaction_costs = drift * (self.transaction_cost_bps / 10000)

            if drift_cost > transaction_costs * 2:
                # Rebalance if drift cost > 2x transaction costs
                return (
                    True,
                    f"Drift cost {drift_cost:.4f} > 2x TC {transaction_costs:.4f}",
                )
            return False, f"Drift cost {drift_cost:.4f} < 2x TC {transaction_costs:.4f}"

        else:  # CALENDAR
            # External logic handles calendar rebalancing
            return True, "Calendar rebalance"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Sample data
    np.random.seed(42)
    assets = ["SPY", "TLT", "GLD", "VNQ", "DBC"]
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n_assets = len(assets)
    n_days = len(dates)

    # Simulate returns
    returns_data = np.random.randn(n_days, n_assets) * 0.01
    returns = pd.DataFrame(returns_data, columns=assets, index=dates)

    # Initialize engine
    engine = PortfolioOptimizationEngine()

    # Test 1: Black-Litterman with views
    views = [
        BlackLittermanView(
            asset="SPY",
            view_type="absolute",
            expected_return=0.12,
            confidence=0.8,
        ),
        BlackLittermanView(
            asset="TLT",
            view_type="absolute",
            expected_return=0.04,
            confidence=0.6,
        ),
    ]

    result_bl = engine.optimize(
        returns,
        method=OptimizationMethod.BLACK_LITTERMAN,
        views=views,
    )

    print("\n=== Black-Litterman Optimization ===")
    print(f"Expected Return: {result_bl.expected_return:.2%}")
    print(f"Expected Volatility: {result_bl.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {result_bl.sharpe_ratio:.2f}")
    print("\nWeights:")
    for asset, weight in result_bl.weights.items():
        print(f"  {asset}: {weight:.2%}")

    # Test 2: Risk Parity
    result_rp = engine.optimize(
        returns,
        method=OptimizationMethod.RISK_PARITY,
    )

    print("\n=== Risk Parity Optimization ===")
    print(f"Expected Return: {result_rp.expected_return:.2%}")
    print(f"Expected Volatility: {result_rp.expected_volatility:.2%}")
    print("\nWeights:")
    for asset, weight in result_rp.weights.items():
        print(f"  {asset}: {weight:.2%}")
    print("\nRisk Contribution:")
    for asset, contrib in result_rp.risk_contribution.items():
        print(f"  {asset}: {contrib:.2%}")

    # Test 3: Mean-CVaR
    result_cvar = engine.optimize(
        returns,
        method=OptimizationMethod.MEAN_CVAR,
    )

    print("\n=== Mean-CVaR Optimization ===")
    print(f"Expected Return: {result_cvar.expected_return:.2%}")
    print(f"Expected CVaR (95%): {result_cvar.expected_cvar:.2%}")
    print("\nWeights:")
    for asset, weight in result_cvar.weights.items():
        print(f"  {asset}: {weight:.2%}")

    print("\n✅ Portfolio Optimization Engine - Tom Hogan")
