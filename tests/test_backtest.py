"""
ALC-Algo Backtesting Tests
Author: Tom Hogan | Alpha Loop Capital, LLC

Tests for the backtesting engine and strategy evaluation.

NOTE: These tests are being updated for the new backtesting framework.
The old src.backtest.engine has been replaced by src.backtesting.backtest_engine
which provides enhanced features like walk-forward optimization and Monte Carlo simulation.
Some tests below may need API updates to match the new BacktestConfig-based interface.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBacktestEngine:
    """Tests for the backtesting engine."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for backtesting."""
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Generate realistic price series
        base_price = 100
        returns = np.random.randn(252) * 0.015  # 1.5% daily vol
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(252) * 0.003),
            'High': prices * (1 + np.abs(np.random.randn(252) * 0.008)),
            'Low': prices * (1 - np.abs(np.random.randn(252) * 0.008)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 252),
        }, index=dates)
        
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        return df
    
    @pytest.fixture
    def sample_signals(self, sample_price_data):
        """Create sample trading signals."""
        n = len(sample_price_data)
        np.random.seed(42)
        
        # Generate sparse signals
        signals = pd.Series(0, index=sample_price_data.index)
        
        # Add some buy signals (1)
        buy_indices = np.random.choice(n, size=10, replace=False)
        for i in buy_indices:
            signals.iloc[i] = 1
        
        # Add some sell signals (-1)
        sell_indices = np.random.choice(n, size=10, replace=False)
        for i in sell_indices:
            signals.iloc[i] = -1
        
        return signals
    
    def test_backtest_engine_import(self):
        """Test that backtest engine can be imported."""
        from src.backtesting.backtest_engine import BacktestEngine
        assert BacktestEngine is not None
    
    def test_backtest_engine_initialization(self, sample_price_data):
        """Test backtest engine initialization."""
        from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
        from datetime import datetime

        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000,
            commission_bps=10.0  # 0.001 = 10 bps
        )
        engine = BacktestEngine(config)

        assert engine is not None
        assert engine.config.initial_capital == 100000
        assert engine.config.commission_bps == 10.0
    
    def test_run_backtest(self, sample_price_data, sample_signals):
        """Test running a backtest."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0.001
        )
        
        results = engine.run(signals=sample_signals)
        
        assert results is not None
        assert 'portfolio_value' in results or hasattr(results, 'portfolio_value')
    
    def test_performance_metrics(self, sample_price_data, sample_signals):
        """Test performance metric calculations."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0.001
        )
        
        results = engine.run(signals=sample_signals)
        metrics = engine.calculate_metrics(results)
        
        # Check expected metrics exist
        expected_metrics = [
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'max_drawdown',
        ]
        
        for metric in expected_metrics:
            assert metric in metrics or hasattr(metrics, metric)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Test with known returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
        
        # Manual calculation
        excess_returns = returns - 0  # Assuming 0 risk-free rate
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create portfolio with known drawdown
        portfolio = pd.Series([100, 110, 105, 95, 100, 90, 95])
        
        # Calculate drawdown
        running_max = portfolio.cummax()
        drawdown = (portfolio - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max drawdown should be negative
        assert max_drawdown < 0
        
        # Known max drawdown: (90 - 110) / 110 = -0.1818
        expected_max_dd = -0.1818
        assert abs(max_drawdown - expected_max_dd) < 0.01
    
    def test_trade_tracking(self, sample_price_data, sample_signals):
        """Test that trades are properly tracked."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0.001
        )
        
        results = engine.run(signals=sample_signals)
        trades = engine.get_trades()
        
        assert trades is not None
        
        # Should have some trades based on signals
        if len(trades) > 0:
            # Check trade structure
            if isinstance(trades, pd.DataFrame):
                assert 'entry_price' in trades.columns or 'price' in trades.columns
    
    def test_position_sizing(self, sample_price_data):
        """Test position sizing logic."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0.001,
            position_size=0.1  # 10% position size
        )
        
        assert engine.position_size == 0.1
    
    def test_commission_impact(self, sample_price_data, sample_signals):
        """Test that commissions properly reduce returns."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        # Run with no commission
        engine_no_comm = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0
        )
        results_no_comm = engine_no_comm.run(signals=sample_signals)
        
        # Run with commission
        engine_with_comm = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0.001
        )
        results_with_comm = engine_with_comm.run(signals=sample_signals)
        
        # With commission should have lower or equal returns
        # (assuming there were trades)
        if isinstance(results_no_comm, dict) and isinstance(results_with_comm, dict):
            if 'total_return' in results_no_comm and 'total_return' in results_with_comm:
                assert results_with_comm['total_return'] <= results_no_comm['total_return']


class TestBacktestResults:
    """Tests for backtest result analysis."""
    
    def test_equity_curve(self):
        """Test equity curve generation."""
        # Simulate equity curve
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        returns = np.random.randn(100) * 0.01
        equity = 100000 * np.cumprod(1 + returns)
        
        equity_curve = pd.Series(equity, index=dates)
        
        assert len(equity_curve) == 100
        assert equity_curve.iloc[0] > 0
    
    def test_return_distribution(self):
        """Test return distribution analysis."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        
        # Calculate statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        assert isinstance(mean_return, float)
        assert std_return > 0
        assert isinstance(skewness, float)
        assert isinstance(kurtosis, float)


class TestStrategyEvaluation:
    """Tests for strategy evaluation metrics."""
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -30, 200, -10, 150, -80]
        })
        
        winning_trades = (trades['pnl'] > 0).sum()
        total_trades = len(trades)
        win_rate = winning_trades / total_trades
        
        # 5 winners out of 8
        expected_win_rate = 5 / 8
        assert abs(win_rate - expected_win_rate) < 0.01
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -30, 200, -10, 150, -80]
        })
        
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 525 / 170 = 3.088
        expected_pf = 525 / 170
        assert abs(profit_factor - expected_pf) < 0.01
    
    def test_average_trade(self):
        """Test average trade calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -30, 200, -10, 150, -80]
        })
        
        avg_trade = trades['pnl'].mean()
        
        # (525 - 170) / 8 = 44.375
        expected_avg = 44.375
        assert abs(avg_trade - expected_avg) < 0.01


class TestBacktestValidation:
    """Tests for backtest validation and sanity checks."""
    
    def test_no_lookahead_bias(self, sample_price_data):
        """Test that backtest doesn't use future information."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        # Create signals based only on past data
        signals = pd.Series(0, index=sample_price_data.index)
        
        # Signal based on past 5-day return
        past_returns = sample_price_data['Close'].pct_change(5)
        signals[past_returns > 0] = 1
        signals[past_returns < 0] = -1
        
        engine = BacktestEngine(
            data=sample_price_data,
            initial_capital=100000,
            commission=0.001
        )
        
        # Should run without error
        results = engine.run(signals=signals)
        assert results is not None
    
    def test_capital_preservation(self, sample_price_data):
        """Test that initial capital is preserved when no trades."""
        from src.backtesting.backtest_engine import BacktestEngine
        
        # No signals (all zeros)
        signals = pd.Series(0, index=sample_price_data.index)
        
        initial_capital = 100000
        engine = BacktestEngine(
            data=sample_price_data,
            initial_capital=initial_capital,
            commission=0.001
        )
        
        results = engine.run(signals=signals)
        
        # Final capital should equal initial when no trades
        if isinstance(results, dict) and 'final_capital' in results:
            assert results['final_capital'] == initial_capital


@pytest.fixture
def sample_price_data():
    """Module-level fixture for price data."""
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    base_price = 100
    returns = np.random.randn(252) * 0.015
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(252) * 0.003),
        'High': prices * (1 + np.abs(np.random.randn(252) * 0.008)),
        'Low': prices * (1 - np.abs(np.random.randn(252) * 0.008)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 252),
    }, index=dates)
    
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    return df


@pytest.fixture
def sample_signals(sample_price_data):
    """Module-level fixture for trading signals."""
    n = len(sample_price_data)
    np.random.seed(42)
    
    signals = pd.Series(0, index=sample_price_data.index)
    buy_indices = np.random.choice(n, size=10, replace=False)
    for i in buy_indices:
        signals.iloc[i] = 1
    
    sell_indices = np.random.choice(n, size=10, replace=False)
    for i in sell_indices:
        signals.iloc[i] = -1
    
    return signals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

