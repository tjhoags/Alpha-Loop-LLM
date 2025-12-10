
import numpy as np
import pandas as pd
from scipy.stats import norm


class QuantRiskEngine:
    """Advanced Risk Metrics:
    - Delta-Adjusted VaR
    - Expected Shortfall (CVaR)
    - Portfolio Convexity
    """

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.99, holding_period: int = 1) -> float:
        """Calculates Historical VaR.
        """
        if returns.empty:
            return 0.0
        return np.percentile(returns, 100 * (1 - confidence_level)) * np.sqrt(holding_period)

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.99) -> float:
        """Calculates Conditional VaR (Expected Shortfall).
        Average loss BEYOND the VaR cutoff.
        """
        if returns.empty:
            return 0.0
        var_cutoff = self.calculate_var(returns, confidence_level)
        return returns[returns <= var_cutoff].mean()

    def calculate_parametric_var(self, portfolio_value: float, vol: float, confidence_level: float = 0.99) -> float:
        """Parametric (Normal) VaR = Position * Vol * Z-score
        """
        z_score = norm.ppf(confidence_level)
        return portfolio_value * vol * z_score

    def delta_adjusted_var(self, portfolio_value: float, delta: float, underlying_vol: float, confidence_level: float = 0.99) -> float:
        """Delta-Normal VaR for derivatives portfolios.
        VaR ~ Delta * S * Vol * Z
        """
        z_score = norm.ppf(confidence_level)
        return abs(delta) * portfolio_value * underlying_vol * z_score

    def estimate_convexity(self, price_series: pd.Series) -> float:
        """Simple statistical convexity proxy: Kurtosis of return distribution.
        High kurtosis -> Fat tails -> Higher convexity risk.
        """
        return price_series.pct_change().kurtosis()


