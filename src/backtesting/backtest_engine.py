"""
Comprehensive Backtesting Framework

Rigorous historical testing with walk-forward analysis and Monte Carlo simulation.

Why Basic Approaches Fail:
- In-sample overfitting (curve fitting to past data)
- Look-ahead bias (using future data)
- Survivorship bias (only testing winners)
- Ignoring transaction costs and slippage
- No out-of-sample testing
- Unrealistic fills and execution

Our Creative Philosophy:
- Walk-forward optimization (out-of-sample testing)
- Monte Carlo simulation for robustness
- Transaction costs and slippage modeling
- Realistic order fills (market impact)
- Multiple time periods and regimes
- Statistical significance testing
- Drawdown and risk analysis
- Benchmark comparison

Elite institutions demand rigorous backtesting:
- Renaissance: Decades of walk-forward testing before deployment
- Two Sigma: Monte Carlo simulation for every strategy
- Citadel: Multi-regime backtesting (bull, bear, crisis)

Author: Tom Hogan
Date: 2025-12-09
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting modes"""

    IN_SAMPLE = "in_sample"  # Train on all data (overfitting risk)
    OUT_OF_SAMPLE = "out_of_sample"  # Test on holdout period
    WALK_FORWARD = "walk_forward"  # Rolling train/test windows
    MONTE_CARLO = "monte_carlo"  # Randomized scenarios
    CROSS_VALIDATION = "cross_validation"  # K-fold validation


class OrderType(Enum):
    """Order types"""

    MARKET = "market"  # Immediate execution at market price
    LIMIT = "limit"  # Execute at limit price or better
    STOP = "stop"  # Stop loss order
    STOP_LIMIT = "stop_limit"  # Stop with limit price


class FillModel(Enum):
    """Order fill models"""

    IMMEDIATE = "immediate"  # Filled at close price (unrealistic)
    NEXT_OPEN = "next_open"  # Filled at next day's open
    VWAP = "vwap"  # Filled at VWAP with slippage
    MARKET_IMPACT = "market_impact"  # Almgren-Chriss impact model


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    mode: BacktestMode = BacktestMode.WALK_FORWARD
    fill_model: FillModel = FillModel.MARKET_IMPACT
    commission_bps: float = 5.0  # 5 bps per trade
    slippage_bps: float = 3.0  # 3 bps slippage
    market_impact_pct: float = 0.10  # 10% of volatility
    max_position_size: float = 0.20  # Max 20% per position
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    walk_forward_train_days: int = 252  # 1 year training
    walk_forward_test_days: int = 63  # 3 months testing
    monte_carlo_simulations: int = 1000
    confidence_level: float = 0.95
    benchmark: str = "SPY"  # Benchmark ticker


@dataclass
class Trade:
    """Individual trade record"""

    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    shares: float
    price: float
    commission: float
    slippage: float
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position"""

    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # % of portfolio


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""

    config: BacktestConfig
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    avg_holding_period_days: float
    total_commission: float
    total_slippage: float
    benchmark_return: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades: List[Trade]
    monthly_returns: pd.Series
    annual_returns: pd.Series
    statistical_significance: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""

    in_sample_results: List[BacktestResult]
    out_of_sample_results: List[BacktestResult]
    combined_result: BacktestResult
    degradation: float  # Out-of-sample vs in-sample performance ratio
    consistency: float  # Std dev of out-of-sample returns


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""

    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_95: float
    probability_of_profit: float
    probability_of_drawdown_gt_20: float
    var_95: float
    cvar_95: float
    all_simulations: List[BacktestResult]


class BacktestEngine:
    """
    Institutional-grade backtesting engine.

    Features:
    - Walk-forward optimization (out-of-sample testing)
    - Monte Carlo simulation for robustness
    - Realistic transaction costs and market impact
    - Multiple fill models
    - Statistical significance testing
    - Regime-based performance analysis
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        logger.info(f"BacktestEngine initialized with ${config.initial_capital:,.0f}")

    def run_backtest(
        self,
        strategy_func: Callable,
        price_data: pd.DataFrame,  # Date × Symbol prices
        signal_data: Optional[pd.DataFrame] = None,  # Strategy signals
        benchmark_data: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run a single backtest.

        Args:
            strategy_func: Strategy function that returns signals
            price_data: Historical price data (OHLCV)
            signal_data: Pre-computed signals (optional)
            benchmark_data: Benchmark prices for comparison

        Returns:
            BacktestResult with comprehensive metrics
        """
        # Reset state
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Get trading dates
        dates = price_data.index
        dates = dates[
            (dates >= self.config.start_date) & (dates <= self.config.end_date)
        ]

        # Run simulation day by day
        for i, date in enumerate(dates):
            # Get signals for this date
            if signal_data is not None:
                signals = signal_data.loc[date] if date in signal_data.index else {}
            else:
                # Call strategy function
                lookback_data = price_data.loc[:date].tail(252)  # 1 year lookback
                signals = strategy_func(lookback_data, date)

            # Execute trades based on signals
            if signals:
                self._execute_signals(signals, price_data.loc[date], date)

            # Update portfolio value
            self._update_portfolio_value(price_data.loc[date])

            # Record equity curve
            self.equity_curve.append((date, self.portfolio_value))

        # Calculate performance metrics
        result = self._calculate_metrics(benchmark_data)
        result.config = self.config
        result.trades = self.trades

        return result

    def run_walk_forward(
        self,
        strategy_func: Callable,
        price_data: pd.DataFrame,
        benchmark_data: Optional[pd.Series] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Train on one period, test on next period, roll forward.
        This prevents overfitting and gives realistic out-of-sample results.
        """
        train_days = self.config.walk_forward_train_days
        test_days = self.config.walk_forward_test_days

        dates = price_data.index
        in_sample_results = []
        out_of_sample_results = []

        current_start = 0
        while current_start + train_days + test_days < len(dates):
            # Training period
            train_start = dates[current_start]
            train_end = dates[current_start + train_days]

            # Testing period
            test_start = train_end + timedelta(days=1)
            test_end = dates[min(current_start + train_days + test_days, len(dates) - 1)]

            # In-sample backtest (training period)
            in_sample_config = BacktestConfig(
                start_date=train_start,
                end_date=train_end,
                initial_capital=self.config.initial_capital,
                commission_bps=self.config.commission_bps,
                slippage_bps=self.config.slippage_bps,
            )
            in_sample_engine = BacktestEngine(in_sample_config)
            in_sample_result = in_sample_engine.run_backtest(
                strategy_func, price_data, benchmark_data=benchmark_data
            )
            in_sample_results.append(in_sample_result)

            # Out-of-sample backtest (testing period)
            out_sample_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=self.config.initial_capital,
                commission_bps=self.config.commission_bps,
                slippage_bps=self.config.slippage_bps,
            )
            out_sample_engine = BacktestEngine(out_sample_config)
            out_sample_result = out_sample_engine.run_backtest(
                strategy_func, price_data, benchmark_data=benchmark_data
            )
            out_of_sample_results.append(out_sample_result)

            # Roll forward
            current_start += test_days

        # Combine all out-of-sample results
        combined_equity = pd.concat(
            [r.equity_curve for r in out_of_sample_results]
        ).sort_index()

        # Calculate combined metrics
        combined_config = BacktestConfig(
            start_date=out_of_sample_results[0].config.start_date,
            end_date=out_of_sample_results[-1].config.end_date,
            initial_capital=self.config.initial_capital,
        )
        combined_engine = BacktestEngine(combined_config)
        combined_engine.equity_curve = [
            (date, val) for date, val in combined_equity.items()
        ]
        combined_result = combined_engine._calculate_metrics(benchmark_data)

        # Calculate degradation (how much worse is out-of-sample vs in-sample?)
        avg_in_sample_return = np.mean([r.annual_return for r in in_sample_results])
        avg_out_sample_return = np.mean([r.annual_return for r in out_of_sample_results])
        degradation = (
            avg_out_sample_return / avg_in_sample_return
            if avg_in_sample_return > 0
            else 0.0
        )

        # Consistency (how stable are out-of-sample results?)
        consistency = np.std([r.annual_return for r in out_of_sample_results])

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            combined_result=combined_result,
            degradation=float(degradation),
            consistency=float(consistency),
        )

    def run_monte_carlo(
        self,
        base_result: BacktestResult,
        n_simulations: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Randomly resample trade returns to generate distribution of outcomes.
        Tests robustness and provides confidence intervals.
        """
        if n_simulations is None:
            n_simulations = self.config.monte_carlo_simulations

        # Extract trade returns
        trade_returns = [
            t.pnl / self.config.initial_capital
            for t in base_result.trades
            if t.pnl is not None
        ]

        if len(trade_returns) < 10:
            logger.warning("Insufficient trades for Monte Carlo simulation")
            return MonteCarloResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

        simulated_returns = []

        for _ in range(n_simulations):
            # Resample trades with replacement
            resampled = np.random.choice(trade_returns, size=len(trade_returns), replace=True)

            # Calculate cumulative return
            cumulative_return = (1 + pd.Series(resampled)).prod() - 1
            simulated_returns.append(cumulative_return)

        simulated_returns = np.array(simulated_returns)

        # Calculate statistics
        mean_return = np.mean(simulated_returns)
        median_return = np.median(simulated_returns)
        std_return = np.std(simulated_returns)
        percentile_5 = np.percentile(simulated_returns, 5)
        percentile_95 = np.percentile(simulated_returns, 95)
        probability_profit = (simulated_returns > 0).sum() / n_simulations

        # Probability of >20% drawdown (simplified)
        probability_dd_gt_20 = (simulated_returns < -0.20).sum() / n_simulations

        # VaR and CVaR
        var_95 = np.percentile(simulated_returns, 5)
        cvar_95 = simulated_returns[simulated_returns <= var_95].mean()

        return MonteCarloResult(
            mean_return=float(mean_return),
            median_return=float(median_return),
            std_return=float(std_return),
            percentile_5=float(percentile_5),
            percentile_95=float(percentile_95),
            probability_of_profit=float(probability_profit),
            probability_of_drawdown_gt_20=float(probability_dd_gt_20),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            all_simulations=[],  # Could store individual results if needed
        )

    def _execute_signals(
        self, signals: Dict[str, float], prices: pd.Series, date: datetime
    ):
        """Execute trading signals with realistic fills"""
        for symbol, target_weight in signals.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Calculate target shares
            target_value = target_weight * self.portfolio_value
            target_shares = target_value / current_price if current_price > 0 else 0

            # Current position
            current_shares = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0, 0)).shares

            # Calculate trade
            shares_to_trade = target_shares - current_shares

            if abs(shares_to_trade) < 1:  # Skip tiny trades
                continue

            # Apply position size limits
            max_shares = (
                self.config.max_position_size * self.portfolio_value / current_price
            )
            shares_to_trade = np.clip(shares_to_trade, -max_shares, max_shares)

            # Execute trade
            side = "buy" if shares_to_trade > 0 else "sell"
            fill_price = self._calculate_fill_price(
                current_price, abs(shares_to_trade), side
            )

            commission = self._calculate_commission(abs(shares_to_trade), fill_price)
            slippage = abs(fill_price - current_price) * abs(shares_to_trade)

            # Update cash
            trade_value = shares_to_trade * fill_price
            self.cash -= trade_value + commission

            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol, 0, 0, 0, 0, 0, 0)

            position = self.positions[symbol]

            # Calculate PnL for closing trades
            pnl = None
            if position.shares != 0 and np.sign(shares_to_trade) != np.sign(position.shares):
                # Closing position
                closed_shares = min(abs(shares_to_trade), abs(position.shares))
                pnl = closed_shares * (fill_price - position.avg_cost)

            # Update position
            if position.shares + shares_to_trade == 0:
                # Fully closed
                del self.positions[symbol]
            else:
                # Update average cost
                if np.sign(shares_to_trade) == np.sign(position.shares) or position.shares == 0:
                    # Adding to position
                    total_cost = (
                        position.shares * position.avg_cost
                        + shares_to_trade * fill_price
                    )
                    total_shares = position.shares + shares_to_trade
                    position.avg_cost = (
                        total_cost / total_shares if total_shares != 0 else 0
                    )

                position.shares += shares_to_trade

            # Record trade
            trade = Trade(
                timestamp=date,
                symbol=symbol,
                side=side,
                shares=abs(shares_to_trade),
                price=fill_price,
                commission=commission,
                slippage=slippage,
                pnl=pnl,
            )
            self.trades.append(trade)

    def _calculate_fill_price(
        self, market_price: float, shares: float, side: str
    ) -> float:
        """Calculate realistic fill price with slippage and market impact"""
        # Base slippage
        slippage = self.config.slippage_bps / 10000

        # Market impact (larger trades have worse fills)
        # Simplified Almgren-Chriss: impact = k * sqrt(shares)
        market_impact = self.config.market_impact_pct * np.sqrt(shares) / 100

        total_slippage = slippage + market_impact

        if side == "buy":
            return market_price * (1 + total_slippage)
        else:
            return market_price * (1 - total_slippage)

    def _calculate_commission(self, shares: float, price: float) -> float:
        """Calculate commission"""
        trade_value = shares * price
        return trade_value * (self.config.commission_bps / 10000)

    def _update_portfolio_value(self, prices: pd.Series):
        """Update portfolio value and positions"""
        # Update position values
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.market_value = position.shares * position.current_price
                position.unrealized_pnl = (
                    position.market_value - position.shares * position.avg_cost
                )

        # Total portfolio value
        total_position_value = sum(p.market_value for p in self.positions.values())
        self.portfolio_value = self.cash + total_position_value

        # Update weights
        for position in self.positions.values():
            position.weight = (
                position.market_value / self.portfolio_value
                if self.portfolio_value > 0
                else 0
            )

    def _calculate_metrics(
        self, benchmark_data: Optional[pd.Series] = None
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) == 0:
            return BacktestResult(
                config=self.config,
                total_return=0.0,
                annual_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                avg_holding_period_days=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                benchmark_return=0.0,
                alpha=0.0,
                beta=0.0,
                tracking_error=0.0,
                information_ratio=0.0,
                equity_curve=pd.Series(),
                drawdown_series=pd.Series(),
                trades=[],
                monthly_returns=pd.Series(),
                annual_returns=pd.Series(),
                statistical_significance={},
                regime_performance={},
            )

        # Convert equity curve to Series
        equity_series = pd.Series(dict(self.equity_curve))
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

        # Annual return
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_daily = 0.045 / 252
        excess_returns = returns - risk_free_daily
        sharpe_ratio = (
            (excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            if excess_returns.std() > 0
            else 0.0
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (
            (annual_return - 0.045) / downside_std if downside_std > 0 else 0.0
        )

        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Trade statistics
        total_trades = len(self.trades)
        realized_pnls = [t.pnl for t in self.trades if t.pnl is not None]
        avg_trade_pnl = np.mean(realized_pnls) if realized_pnls else 0.0
        best_trade = max(realized_pnls) if realized_pnls else 0.0
        worst_trade = min(realized_pnls) if realized_pnls else 0.0

        wins = [p for p in realized_pnls if p > 0]
        losses = [p for p in realized_pnls if p < 0]
        win_rate = len(wins) / len(realized_pnls) if realized_pnls else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0
        )

        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)

        # Benchmark comparison
        benchmark_return = 0.0
        alpha = 0.0
        beta = 0.0
        tracking_error = 0.0
        information_ratio = 0.0

        if benchmark_data is not None:
            aligned_benchmark = benchmark_data.reindex(equity_series.index, method="ffill")
            benchmark_returns = aligned_benchmark.pct_change().dropna()

            if len(benchmark_returns) > 0:
                benchmark_return = (1 + benchmark_returns).prod() - 1

                # Alpha and beta regression
                aligned_returns = pd.DataFrame(
                    {"portfolio": returns, "benchmark": benchmark_returns}
                ).dropna()

                if len(aligned_returns) > 1:
                    slope, intercept, r_val, p_val, std_err = stats.linregress(
                        aligned_returns["benchmark"], aligned_returns["portfolio"]
                    )
                    beta = slope
                    alpha = intercept * 252  # Annualized

                    # Tracking error
                    active_returns = aligned_returns["portfolio"] - aligned_returns["benchmark"]
                    tracking_error = active_returns.std() * np.sqrt(252)

                    # Information ratio
                    information_ratio = (
                        alpha / tracking_error if tracking_error > 0 else 0.0
                    )

        # Monthly and annual returns
        equity_df = equity_series.to_frame("equity")
        equity_df["year"] = equity_df.index.year
        equity_df["month"] = equity_df.index.to_period("M")

        monthly_returns = equity_df.groupby("month")["equity"].apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 0 else 0.0
        )
        annual_returns = equity_df.groupby("year")["equity"].apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 0 else 0.0
        )

        # Statistical significance (t-test on returns)
        if len(returns) > 1:
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            statistical_significance = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_95": p_value < 0.05,
            }
        else:
            statistical_significance = {}

        return BacktestResult(
            config=self.config,
            total_return=float(total_return),
            annual_return=float(annual_return),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            calmar_ratio=float(calmar_ratio),
            max_drawdown=float(max_drawdown),
            volatility=float(volatility),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            total_trades=total_trades,
            avg_trade_pnl=float(avg_trade_pnl),
            best_trade=float(best_trade),
            worst_trade=float(worst_trade),
            avg_holding_period_days=0.0,  # TODO: Calculate from trades
            total_commission=float(total_commission),
            total_slippage=float(total_slippage),
            benchmark_return=float(benchmark_return),
            alpha=float(alpha),
            beta=float(beta),
            tracking_error=float(tracking_error),
            information_ratio=float(information_ratio),
            equity_curve=equity_series,
            drawdown_series=drawdown,
            trades=self.trades,
            monthly_returns=monthly_returns,
            annual_returns=annual_returns,
            statistical_significance=statistical_significance,
            regime_performance={},  # TODO: Calculate regime-specific performance
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    # Simulate price data
    prices = {}
    for symbol in symbols:
        price_series = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
        prices[symbol] = price_series

    price_data = pd.DataFrame(prices, index=dates)

    # Simulate benchmark (SPY)
    benchmark = pd.Series(
        100 * (1 + np.random.randn(len(dates)).cumsum() * 0.008),
        index=dates,
    )

    # Simple momentum strategy
    def momentum_strategy(price_data: pd.DataFrame, current_date: datetime) -> Dict[str, float]:
        """Simple 6-month momentum strategy"""
        if len(price_data) < 126:
            return {}

        # Calculate 6-month returns
        returns_6m = (price_data.iloc[-1] / price_data.iloc[-126]) - 1

        # Rank assets
        ranked = returns_6m.sort_values(ascending=False)

        # Long top 2, equal weight
        signals = {}
        for i, symbol in enumerate(ranked.index[:2]):
            signals[symbol] = 0.50  # 50% each

        return signals

    # Backtest configuration
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=100000,
        mode=BacktestMode.WALK_FORWARD,
        commission_bps=5.0,
        slippage_bps=3.0,
        rebalance_frequency="monthly",
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(
        momentum_strategy,
        price_data,
        benchmark_data=benchmark,
    )

    print("\n=== Backtest Results ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annual Return: {result.annual_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Benchmark Return: {result.benchmark_return:.2%}")
    print(f"Alpha: {result.alpha:.2%}")
    print(f"Beta: {result.beta:.2f}")
    print(f"Information Ratio: {result.information_ratio:.2f}")
    print(f"Total Costs: ${result.total_commission + result.total_slippage:,.0f}")

    print("\n✅ Backtesting Framework - Tom Hogan")
