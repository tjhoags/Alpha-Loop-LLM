"""
================================================================================
INSTITUTIONAL QUANT METRICS - Long/Short Hedge Fund Analytics
================================================================================
Advanced quantitative metrics for a long-short quant hedge fund including:

RISK METRICS:
- Value at Risk (VaR) - Parametric, Historical, Monte Carlo
- Delta-Adjusted VaR (for options/derivatives exposure)
- Conditional VaR (CVaR / Expected Shortfall)
- Component VaR (contribution by position)

FIXED INCOME / CONVEXITY:
- Duration (Modified, Macaulay, Effective)
- Convexity
- DV01 (Dollar Value of 01)

GREEKS (Options):
- Delta, Gamma, Vega, Theta, Rho
- Portfolio-level Greeks aggregation

ADVANCED VALUATION:
- Enterprise Value multiples
- DCF sensitivity metrics
- Quality factors (ROIC, FCF Yield, Earnings Quality)
- Growth-adjusted metrics (PEG, EV/EBITDA/Growth)

PORTFOLIO ANALYTICS:
- Sharpe, Sortino, Calmar, Information Ratio
- Beta, Alpha (Jensen's), R-squared
- Tracking Error, Active Share
- Factor exposures (Fama-French, Carhart)

================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptionPosition:
    """Represents an options position with Greeks."""
    symbol: str
    option_type: str  # "CALL" or "PUT"
    strike: float
    expiry_days: int
    quantity: int
    underlying_price: float
    implied_vol: float
    risk_free_rate: float = 0.05
    
    @property
    def delta(self) -> float:
        """Black-Scholes Delta."""
        d1 = self._d1()
        if self.option_type == "CALL":
            return norm.cdf(d1) * self.quantity
        else:
            return (norm.cdf(d1) - 1) * self.quantity
    
    @property
    def gamma(self) -> float:
        """Black-Scholes Gamma."""
        d1 = self._d1()
        t = self.expiry_days / 365
        return (norm.pdf(d1) / (self.underlying_price * self.implied_vol * np.sqrt(t))) * self.quantity
    
    @property
    def vega(self) -> float:
        """Black-Scholes Vega (per 1% vol change)."""
        d1 = self._d1()
        t = self.expiry_days / 365
        return (self.underlying_price * norm.pdf(d1) * np.sqrt(t) / 100) * self.quantity
    
    @property
    def theta(self) -> float:
        """Black-Scholes Theta (per day)."""
        d1, d2 = self._d1(), self._d2()
        t = self.expiry_days / 365
        S, K, r, sigma = self.underlying_price, self.strike, self.risk_free_rate, self.implied_vol
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))
        if self.option_type == "CALL":
            term2 = -r * K * np.exp(-r * t) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * t) * norm.cdf(-d2)
        
        return ((term1 + term2) / 365) * self.quantity
    
    @property
    def rho(self) -> float:
        """Black-Scholes Rho (per 1% rate change)."""
        d2 = self._d2()
        t = self.expiry_days / 365
        K = self.strike
        r = self.risk_free_rate
        
        if self.option_type == "CALL":
            return (K * t * np.exp(-r * t) * norm.cdf(d2) / 100) * self.quantity
        else:
            return (-K * t * np.exp(-r * t) * norm.cdf(-d2) / 100) * self.quantity
    
    def _d1(self) -> float:
        t = self.expiry_days / 365
        S, K = self.underlying_price, self.strike
        r, sigma = self.risk_free_rate, self.implied_vol
        return (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    
    def _d2(self) -> float:
        t = self.expiry_days / 365
        return self._d1() - self.implied_vol * np.sqrt(t)


@dataclass 
class BondPosition:
    """Represents a fixed income position."""
    symbol: str
    face_value: float
    coupon_rate: float
    yield_to_maturity: float
    years_to_maturity: float
    frequency: int = 2  # Semi-annual
    
    @property
    def macaulay_duration(self) -> float:
        """Macaulay Duration in years."""
        c = self.coupon_rate / self.frequency
        y = self.yield_to_maturity / self.frequency
        n = int(self.years_to_maturity * self.frequency)
        
        if y == 0:
            return self.years_to_maturity
        
        pv_weights = []
        for t in range(1, n + 1):
            cf = c * self.face_value if t < n else (c + 1) * self.face_value
            pv = cf / ((1 + y) ** t)
            pv_weights.append(pv * t / self.frequency)
        
        price = sum(cf / ((1 + y) ** t) for t, cf in enumerate(
            [c * self.face_value] * (n - 1) + [(c + 1) * self.face_value], 1
        ))
        
        return sum(pv_weights) / price if price > 0 else 0
    
    @property
    def modified_duration(self) -> float:
        """Modified Duration."""
        return self.macaulay_duration / (1 + self.yield_to_maturity / self.frequency)
    
    @property
    def convexity(self) -> float:
        """Bond Convexity."""
        c = self.coupon_rate / self.frequency
        y = self.yield_to_maturity / self.frequency
        n = int(self.years_to_maturity * self.frequency)
        
        if y == 0:
            return self.years_to_maturity ** 2
        
        conv_sum = 0
        price = 0
        for t in range(1, n + 1):
            cf = c * self.face_value if t < n else (c + 1) * self.face_value
            pv = cf / ((1 + y) ** t)
            price += pv
            conv_sum += pv * t * (t + 1)
        
        return conv_sum / (price * self.frequency**2 * (1 + y)**2) if price > 0 else 0
    
    @property
    def dv01(self) -> float:
        """Dollar Value of a Basis Point."""
        return self.modified_duration * self.face_value * 0.0001


# =============================================================================
# VALUE AT RISK (VaR) CALCULATIONS
# =============================================================================

class VaRCalculator:
    """
    Comprehensive Value at Risk calculations.
    """
    
    @staticmethod
    def parametric_var(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Parametric (Variance-Covariance) VaR.
        Assumes normal distribution of returns.
        """
        mu = returns.mean()
        sigma = returns.std()
        z_score = norm.ppf(1 - confidence)
        
        # Scale to horizon
        var_1d = portfolio_value * (mu - z_score * sigma)
        var_horizon = var_1d * np.sqrt(horizon_days)
        
        return abs(var_horizon)
    
    @staticmethod
    def historical_var(
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Historical Simulation VaR.
        Uses actual historical return distribution.
        """
        percentile = (1 - confidence) * 100
        var_pct = np.percentile(returns.dropna(), percentile)
        return abs(portfolio_value * var_pct)
    
    @staticmethod
    def monte_carlo_var(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
        portfolio_value: float = 1000000,
        simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR with GBM simulation.
        """
        mu = returns.mean()
        sigma = returns.std()
        
        # Simulate paths
        dt = 1 / 252  # Daily
        np.random.seed(42)
        
        simulated_returns = np.zeros(simulations)
        for i in range(simulations):
            path_return = 0
            for _ in range(horizon_days):
                z = np.random.standard_normal()
                daily_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
                path_return += daily_return
            simulated_returns[i] = path_return
        
        percentile = (1 - confidence) * 100
        var_pct = np.percentile(simulated_returns, percentile)
        return abs(portfolio_value * var_pct)
    
    @staticmethod
    def delta_adjusted_var(
        equity_returns: pd.Series,
        option_positions: List[OptionPosition],
        confidence: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Delta-Adjusted VaR for portfolios with options.
        Converts option exposure to delta-equivalent stock exposure.
        """
        # Calculate delta-equivalent exposure
        total_delta = sum(opt.delta * opt.underlying_price for opt in option_positions)
        
        # Standard VaR on equivalent exposure
        sigma = equity_returns.std()
        z_score = norm.ppf(1 - confidence)
        
        delta_var = abs(total_delta * z_score * sigma)
        return delta_var
    
    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Conditional VaR (CVaR) / Expected Shortfall.
        Average loss in the worst (1-confidence)% of cases.
        """
        var_threshold = np.percentile(returns.dropna(), (1 - confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return VaRCalculator.historical_var(returns, confidence, portfolio_value)
        
        expected_shortfall = tail_losses.mean()
        return abs(portfolio_value * expected_shortfall)
    
    @staticmethod
    def component_var(
        returns_matrix: pd.DataFrame,
        weights: np.ndarray,
        confidence: float = 0.95,
        portfolio_value: float = 1000000
    ) -> Dict[str, float]:
        """
        Component VaR - contribution of each position to total VaR.
        """
        # Covariance matrix
        cov_matrix = returns_matrix.cov()
        
        # Portfolio variance
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_sigma = np.sqrt(port_var)
        
        # Marginal VaR
        marginal_var = np.dot(cov_matrix, weights) / port_sigma
        
        # Component VaR
        z_score = norm.ppf(1 - confidence)
        component_vars = {}
        
        for i, col in enumerate(returns_matrix.columns):
            comp_var = weights[i] * marginal_var[i] * z_score * portfolio_value
            component_vars[col] = abs(comp_var)
        
        return component_vars


# =============================================================================
# ADVANCED VALUATION METRICS
# =============================================================================

class ValuationMetrics:
    """
    Comprehensive valuation metrics for equity analysis.
    Goes far beyond simple P/E ratios.
    """
    
    @staticmethod
    def enterprise_value(
        market_cap: float,
        total_debt: float,
        cash: float,
        minority_interest: float = 0,
        preferred_equity: float = 0
    ) -> float:
        """Enterprise Value calculation."""
        return market_cap + total_debt + minority_interest + preferred_equity - cash
    
    @staticmethod
    def ev_to_ebitda(enterprise_value: float, ebitda: float) -> float:
        """EV/EBITDA multiple."""
        return enterprise_value / ebitda if ebitda > 0 else np.nan
    
    @staticmethod
    def ev_to_ebit(enterprise_value: float, ebit: float) -> float:
        """EV/EBIT multiple."""
        return enterprise_value / ebit if ebit > 0 else np.nan
    
    @staticmethod
    def ev_to_revenue(enterprise_value: float, revenue: float) -> float:
        """EV/Revenue multiple."""
        return enterprise_value / revenue if revenue > 0 else np.nan
    
    @staticmethod
    def ev_to_fcf(enterprise_value: float, free_cash_flow: float) -> float:
        """EV/FCF multiple."""
        return enterprise_value / free_cash_flow if free_cash_flow > 0 else np.nan
    
    @staticmethod
    def peg_ratio(pe_ratio: float, earnings_growth_rate: float) -> float:
        """PEG Ratio (Price/Earnings to Growth)."""
        return pe_ratio / (earnings_growth_rate * 100) if earnings_growth_rate > 0 else np.nan
    
    @staticmethod
    def price_to_book(market_cap: float, book_value: float) -> float:
        """Price to Book Value."""
        return market_cap / book_value if book_value > 0 else np.nan
    
    @staticmethod
    def price_to_tangible_book(market_cap: float, tangible_book: float) -> float:
        """Price to Tangible Book Value."""
        return market_cap / tangible_book if tangible_book > 0 else np.nan
    
    @staticmethod
    def price_to_sales(market_cap: float, revenue: float) -> float:
        """Price to Sales."""
        return market_cap / revenue if revenue > 0 else np.nan
    
    @staticmethod
    def fcf_yield(free_cash_flow: float, market_cap: float) -> float:
        """Free Cash Flow Yield."""
        return free_cash_flow / market_cap if market_cap > 0 else np.nan
    
    @staticmethod
    def earnings_yield(earnings: float, market_cap: float) -> float:
        """Earnings Yield (inverse of P/E)."""
        return earnings / market_cap if market_cap > 0 else np.nan
    
    @staticmethod
    def dividend_yield(dividends: float, price: float) -> float:
        """Dividend Yield."""
        return dividends / price if price > 0 else np.nan
    
    @staticmethod
    def shareholder_yield(
        dividends: float,
        buybacks: float,
        debt_paydown: float,
        market_cap: float
    ) -> float:
        """
        Shareholder Yield = Dividend Yield + Buyback Yield + Debt Paydown Yield
        More comprehensive than dividend yield alone.
        """
        total_return = dividends + buybacks + debt_paydown
        return total_return / market_cap if market_cap > 0 else np.nan


# =============================================================================
# QUALITY METRICS
# =============================================================================

class QualityMetrics:
    """
    Quality factor metrics for stock selection.
    """
    
    @staticmethod
    def return_on_invested_capital(
        nopat: float,  # Net Operating Profit After Tax
        invested_capital: float
    ) -> float:
        """ROIC - Key quality metric."""
        return nopat / invested_capital if invested_capital > 0 else np.nan
    
    @staticmethod
    def return_on_equity(net_income: float, shareholders_equity: float) -> float:
        """ROE."""
        return net_income / shareholders_equity if shareholders_equity > 0 else np.nan
    
    @staticmethod
    def return_on_assets(net_income: float, total_assets: float) -> float:
        """ROA."""
        return net_income / total_assets if total_assets > 0 else np.nan
    
    @staticmethod
    def gross_margin(gross_profit: float, revenue: float) -> float:
        """Gross Margin."""
        return gross_profit / revenue if revenue > 0 else np.nan
    
    @staticmethod
    def operating_margin(operating_income: float, revenue: float) -> float:
        """Operating Margin."""
        return operating_income / revenue if revenue > 0 else np.nan
    
    @staticmethod
    def net_margin(net_income: float, revenue: float) -> float:
        """Net Profit Margin."""
        return net_income / revenue if revenue > 0 else np.nan
    
    @staticmethod
    def asset_turnover(revenue: float, avg_total_assets: float) -> float:
        """Asset Turnover Ratio."""
        return revenue / avg_total_assets if avg_total_assets > 0 else np.nan
    
    @staticmethod
    def inventory_turnover(cogs: float, avg_inventory: float) -> float:
        """Inventory Turnover."""
        return cogs / avg_inventory if avg_inventory > 0 else np.nan
    
    @staticmethod
    def accruals_ratio(
        net_income: float,
        operating_cash_flow: float,
        avg_total_assets: float
    ) -> float:
        """
        Accruals Ratio - Earnings quality indicator.
        Lower is better (more cash-based earnings).
        """
        accruals = net_income - operating_cash_flow
        return accruals / avg_total_assets if avg_total_assets > 0 else np.nan
    
    @staticmethod
    def earnings_quality(operating_cash_flow: float, net_income: float) -> float:
        """
        Earnings Quality = OCF / Net Income
        >1 means cash earnings exceed accounting earnings (good).
        """
        return operating_cash_flow / net_income if net_income > 0 else np.nan
    
    @staticmethod
    def altman_z_score(
        working_capital: float,
        retained_earnings: float,
        ebit: float,
        market_cap: float,
        total_liabilities: float,
        revenue: float,
        total_assets: float
    ) -> float:
        """
        Altman Z-Score - Bankruptcy prediction.
        Z > 2.99: Safe Zone
        1.81 < Z < 2.99: Grey Zone
        Z < 1.81: Distress Zone
        """
        if total_assets == 0:
            return np.nan
        
        A = working_capital / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liabilities if total_liabilities > 0 else 0
        E = revenue / total_assets
        
        return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    
    @staticmethod
    def piotroski_f_score(
        net_income: float,
        operating_cash_flow: float,
        roa_current: float,
        roa_prior: float,
        long_term_debt_current: float,
        long_term_debt_prior: float,
        current_ratio_current: float,
        current_ratio_prior: float,
        shares_current: float,
        shares_prior: float,
        gross_margin_current: float,
        gross_margin_prior: float,
        asset_turnover_current: float,
        asset_turnover_prior: float
    ) -> int:
        """
        Piotroski F-Score (0-9) - Value investing quality score.
        Higher is better.
        """
        score = 0
        
        # Profitability
        if net_income > 0: score += 1
        if operating_cash_flow > 0: score += 1
        if roa_current > roa_prior: score += 1
        if operating_cash_flow > net_income: score += 1  # Earnings quality
        
        # Leverage/Liquidity
        if long_term_debt_current < long_term_debt_prior: score += 1
        if current_ratio_current > current_ratio_prior: score += 1
        if shares_current <= shares_prior: score += 1  # No dilution
        
        # Operating Efficiency
        if gross_margin_current > gross_margin_prior: score += 1
        if asset_turnover_current > asset_turnover_prior: score += 1
        
        return score


# =============================================================================
# PORTFOLIO PERFORMANCE METRICS
# =============================================================================

class PerformanceMetrics:
    """
    Comprehensive portfolio performance analytics.
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252
    ) -> float:
        """Sharpe Ratio - Risk-adjusted return."""
        excess_returns = returns - risk_free_rate / periods_per_year
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252
    ) -> float:
        """Sortino Ratio - Downside risk-adjusted return."""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0
        
        downside_std = np.sqrt((downside_returns**2).mean())
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calmar Ratio = Annualized Return / Max Drawdown."""
        ann_return = returns.mean() * periods_per_year
        max_dd = PerformanceMetrics.max_drawdown(returns)
        return ann_return / abs(max_dd) if max_dd != 0 else np.inf
    
    @staticmethod
    def information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Information Ratio = Active Return / Tracking Error."""
        active_returns = returns - benchmark_returns
        if active_returns.std() == 0:
            return 0
        return np.sqrt(periods_per_year) * active_returns.mean() / active_returns.std()
    
    @staticmethod
    def tracking_error(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Tracking Error - Standard deviation of active returns."""
        active_returns = returns - benchmark_returns
        return active_returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Beta - Systematic risk measure."""
        cov = returns.cov(benchmark_returns)
        var = benchmark_returns.var()
        return cov / var if var > 0 else 0
    
    @staticmethod
    def alpha(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252
    ) -> float:
        """Jensen's Alpha - Risk-adjusted excess return."""
        beta = PerformanceMetrics.beta(returns, benchmark_returns)
        rf_daily = risk_free_rate / periods_per_year
        
        port_return = returns.mean() * periods_per_year
        bench_return = benchmark_returns.mean() * periods_per_year
        
        expected_return = rf_daily * periods_per_year + beta * (bench_return - rf_daily * periods_per_year)
        return port_return - expected_return
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Maximum Drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        return drawdowns.min()
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Percentage of positive returns."""
        return (returns > 0).mean()
    
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """Profit Factor = Gross Profits / Gross Losses."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else np.inf
    
    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        """
        Omega Ratio - Probability weighted ratio of gains vs losses.
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        sum_gains = gains.sum()
        sum_losses = losses.sum()
        
        return sum_gains / sum_losses if sum_losses > 0 else np.inf


# =============================================================================
# FACTOR ANALYSIS
# =============================================================================

class FactorAnalysis:
    """
    Multi-factor model analytics (Fama-French, Carhart).
    """
    
    @staticmethod
    def calculate_factor_exposures(
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio exposures to standard risk factors.
        
        factor_returns should have columns like:
        - MKT (Market excess return)
        - SMB (Small minus Big - Size)
        - HML (High minus Low - Value)
        - MOM (Momentum)
        - QMJ (Quality minus Junk)
        """
        import statsmodels.api as sm
        
        # Align data
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        if len(aligned) < 30:
            logger.warning("Insufficient data for factor analysis")
            return {}
        
        y = aligned.iloc[:, 0]
        X = sm.add_constant(aligned.iloc[:, 1:])
        
        try:
            model = sm.OLS(y, X).fit()
            exposures = dict(zip(X.columns[1:], model.params[1:]))
            exposures['alpha'] = model.params[0] * 252  # Annualized
            exposures['r_squared'] = model.rsquared
            return exposures
        except Exception as e:
            logger.error(f"Factor regression failed: {e}")
            return {}


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def calculate_portfolio_greeks(options: List[OptionPosition]) -> Dict[str, float]:
    """Aggregate Greeks across all option positions."""
    return {
        "total_delta": sum(opt.delta for opt in options),
        "total_gamma": sum(opt.gamma for opt in options),
        "total_vega": sum(opt.vega for opt in options),
        "total_theta": sum(opt.theta for opt in options),
        "total_rho": sum(opt.rho for opt in options),
    }


def calculate_portfolio_duration(bonds: List[BondPosition]) -> Dict[str, float]:
    """Aggregate duration metrics across bond positions."""
    total_value = sum(b.face_value for b in bonds)
    
    if total_value == 0:
        return {"weighted_duration": 0, "weighted_convexity": 0, "total_dv01": 0}
    
    weighted_duration = sum(
        b.modified_duration * b.face_value / total_value for b in bonds
    )
    weighted_convexity = sum(
        b.convexity * b.face_value / total_value for b in bonds
    )
    total_dv01 = sum(b.dv01 for b in bonds)
    
    return {
        "weighted_duration": weighted_duration,
        "weighted_convexity": weighted_convexity,
        "total_dv01": total_dv01,
    }


def comprehensive_risk_report(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_value: float = 1000000,
    options: Optional[List[OptionPosition]] = None,
    bonds: Optional[List[BondPosition]] = None
) -> Dict:
    """
    Generate comprehensive risk report for the portfolio.
    """
    report = {}
    
    # Performance
    report["performance"] = {
        "sharpe_ratio": PerformanceMetrics.sharpe_ratio(returns),
        "sortino_ratio": PerformanceMetrics.sortino_ratio(returns),
        "calmar_ratio": PerformanceMetrics.calmar_ratio(returns),
        "information_ratio": PerformanceMetrics.information_ratio(returns, benchmark_returns),
        "beta": PerformanceMetrics.beta(returns, benchmark_returns),
        "alpha": PerformanceMetrics.alpha(returns, benchmark_returns),
        "max_drawdown": PerformanceMetrics.max_drawdown(returns),
        "win_rate": PerformanceMetrics.win_rate(returns),
        "profit_factor": PerformanceMetrics.profit_factor(returns),
    }
    
    # VaR
    report["var"] = {
        "parametric_95": VaRCalculator.parametric_var(returns, 0.95, 1, portfolio_value),
        "parametric_99": VaRCalculator.parametric_var(returns, 0.99, 1, portfolio_value),
        "historical_95": VaRCalculator.historical_var(returns, 0.95, portfolio_value),
        "historical_99": VaRCalculator.historical_var(returns, 0.99, portfolio_value),
        "monte_carlo_95": VaRCalculator.monte_carlo_var(returns, 0.95, 1, portfolio_value),
        "cvar_95": VaRCalculator.conditional_var(returns, 0.95, portfolio_value),
    }
    
    # Options Greeks (if applicable)
    if options:
        report["greeks"] = calculate_portfolio_greeks(options)
        report["var"]["delta_adjusted_95"] = VaRCalculator.delta_adjusted_var(
            returns, options, 0.95, portfolio_value
        )
    
    # Fixed Income (if applicable)
    if bonds:
        report["fixed_income"] = calculate_portfolio_duration(bonds)
    
    return report


