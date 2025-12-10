"""
ALC-Algo Feature Engineering Tests
Author: Tom Hogan | Alpha Loop Capital, LLC

Tests for technical indicators and feature generation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.randn(100) * 0.02  # 2% daily volatility
        close_prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'Open': close_prices * (1 + np.random.randn(100) * 0.005),
            'High': close_prices * (1 + np.abs(np.random.randn(100) * 0.01)),
            'Low': close_prices * (1 - np.abs(np.random.randn(100) * 0.01)),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 100),
        }, index=dates)
        
        # Ensure High >= Close >= Low
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        return df
    
    def test_technical_module_import(self):
        """Test that technical module can be imported."""
        from src.features.technical import TechnicalIndicators
        assert TechnicalIndicators is not None
    
    def test_sma_calculation(self, sample_ohlcv_data):
        """Test Simple Moving Average calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        sma = ti.sma(window=20)
        
        # Check output
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_ohlcv_data)
        
        # First 19 values should be NaN
        assert sma.iloc[:19].isna().all()
        
        # Non-NaN values should exist after warm-up
        assert not sma.iloc[19:].isna().all()
        
        # Manual verification of SMA calculation
        expected = sample_ohlcv_data['Close'].iloc[:20].mean()
        assert abs(sma.iloc[19] - expected) < 0.01
    
    def test_ema_calculation(self, sample_ohlcv_data):
        """Test Exponential Moving Average calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        ema = ti.ema(window=20)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_ohlcv_data)
        
        # EMA should have values after initial period
        assert not ema.iloc[20:].isna().all()
    
    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test Relative Strength Index calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        rsi = ti.rsi(window=14)
        
        assert isinstance(rsi, pd.Series)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        macd, signal, histogram = ti.macd()
        
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
        
        # Histogram should equal MACD - Signal
        valid_idx = ~(macd.isna() | signal.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx].values,
            (macd[valid_idx] - signal[valid_idx]).values,
            decimal=10
        )
    
    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        upper, middle, lower = ti.bollinger_bands(window=20, num_std=2)
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # Upper should be >= Middle >= Lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
    
    def test_atr_calculation(self, sample_ohlcv_data):
        """Test Average True Range calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        atr = ti.atr(window=14)
        
        assert isinstance(atr, pd.Series)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
    
    def test_volume_ratio(self, sample_ohlcv_data):
        """Test volume ratio calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        vol_ratio = ti.volume_ratio(window=20)
        
        assert isinstance(vol_ratio, pd.Series)
        
        # Volume ratio should be positive
        valid_ratio = vol_ratio.dropna()
        assert (valid_ratio > 0).all()
    
    def test_price_momentum(self, sample_ohlcv_data):
        """Test price momentum calculation."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        momentum = ti.momentum(window=10)
        
        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(sample_ohlcv_data)
    
    def test_add_all_indicators(self, sample_ohlcv_data):
        """Test adding all indicators to dataframe."""
        from src.features.technical import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_ohlcv_data)
        result = ti.add_all_indicators()
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have more columns than original
        assert len(result.columns) > len(sample_ohlcv_data.columns)
        
        # Check for expected indicator columns
        expected_columns = ['SMA_20', 'EMA_20', 'RSI_14']
        for col in expected_columns:
            assert col in result.columns or any(col.lower() in c.lower() for c in result.columns)


class TestFeatureGeneration:
    """Tests for feature generation utilities."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        return returns
    
    def test_returns_calculation(self, sample_returns):
        """Test return calculations."""
        # Simple return calculation
        prices = (1 + sample_returns).cumprod() * 100
        
        # Calculate returns from prices
        calculated_returns = prices.pct_change()
        
        # Should be close to original returns
        valid_idx = ~(calculated_returns.isna() | sample_returns.isna())
        np.testing.assert_array_almost_equal(
            calculated_returns[valid_idx].values[1:],  # Skip first (NaN)
            sample_returns[valid_idx].values[1:],
            decimal=2
        )
    
    def test_rolling_statistics(self, sample_returns):
        """Test rolling statistics calculation."""
        # Rolling mean
        rolling_mean = sample_returns.rolling(window=20).mean()
        assert not rolling_mean.iloc[19:].isna().all()
        
        # Rolling std
        rolling_std = sample_returns.rolling(window=20).std()
        assert (rolling_std.dropna() >= 0).all()
    
    def test_lag_features(self, sample_returns):
        """Test lag feature generation."""
        # Create lag features
        lags = [1, 5, 10]
        lag_features = pd.DataFrame()
        
        for lag in lags:
            lag_features[f'lag_{lag}'] = sample_returns.shift(lag)
        
        assert len(lag_features.columns) == len(lags)
        assert lag_features['lag_1'].iloc[1] == sample_returns.iloc[0]


class TestDataValidation:
    """Tests for data validation utilities."""
    
    def test_valid_ohlcv_data(self, sample_ohlcv_data):
        """Test validation of valid OHLCV data."""
        from src.features.technical import TechnicalIndicators
        
        # This should not raise
        ti = TechnicalIndicators(sample_ohlcv_data)
        assert ti is not None
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        incomplete_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200],
        })
        
        from src.features.technical import TechnicalIndicators
        
        # Should handle gracefully or raise appropriate error
        try:
            ti = TechnicalIndicators(incomplete_data)
            # If it doesn't raise, it should still work for close-only indicators
            sma = ti.sma(window=2)
            assert sma is not None
        except (ValueError, KeyError):
            # Expected behavior for missing columns
            pass


@pytest.fixture
def sample_ohlcv_data():
    """Module-level fixture for OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    base_price = 100
    returns = np.random.randn(100) * 0.02
    close_prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': close_prices * (1 + np.random.randn(100) * 0.005),
        'High': close_prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'Low': close_prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)
    
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    return df


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

