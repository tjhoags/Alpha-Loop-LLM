"""
ALC-Algo Technical Indicators
Author: Tom Hogan | Alpha Loop Capital, LLC

Technical analysis indicators for the ALC-Algo trading platform.
INSTITUTIONAL GRADE - Maximum compute, no shortcuts.
"""

from typing import Tuple, Optional, List
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Technical indicator calculations for market data.
    
    Built to institutional standards:
    - Efficient vectorized computations
    - Proper handling of edge cases
    - Configurable parameters
    """
    
    def __init__(self, data: pd.DataFrame, price_col: str = 'Close'):
        """
        Initialize with price data.
        
        Args:
            data: DataFrame with OHLCV columns
            price_col: Column to use for price-based indicators
        """
        self.data = data.copy()
        self.price_col = price_col
        
        # Validate required columns
        self._validate_data()
    
    def _validate_data(self):
        """Validate input data has required columns."""
        if self.price_col not in self.data.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in data")
    
    # =========================================================================
    # Moving Averages
    # =========================================================================
    
    def sma(self, window: int = 20, column: Optional[str] = None) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            window: Rolling window size
            column: Column to use (default: price_col)
        
        Returns:
            Series with SMA values
        """
        col = column or self.price_col
        return self.data[col].rolling(window=window).mean()
    
    def ema(self, window: int = 20, column: Optional[str] = None) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            window: Span for EMA
            column: Column to use (default: price_col)
        
        Returns:
            Series with EMA values
        """
        col = column or self.price_col
        return self.data[col].ewm(span=window, adjust=False).mean()
    
    def wma(self, window: int = 20, column: Optional[str] = None) -> pd.Series:
        """
        Weighted Moving Average.
        
        Args:
            window: Rolling window size
            column: Column to use (default: price_col)
        
        Returns:
            Series with WMA values
        """
        col = column or self.price_col
        weights = np.arange(1, window + 1)
        
        def weighted_average(x):
            return np.dot(x, weights) / weights.sum()
        
        return self.data[col].rolling(window=window).apply(weighted_average, raw=True)
    
    # =========================================================================
    # Momentum Indicators
    # =========================================================================
    
    def rsi(self, window: int = 14, column: Optional[str] = None) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            window: Period for RSI calculation
            column: Column to use (default: price_col)
        
        Returns:
            Series with RSI values (0-100)
        """
        col = column or self.price_col
        delta = self.data[col].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for undefined values
    
    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to use (default: price_col)
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        col = column or self.price_col
        
        fast_ema = self.data[col].ewm(span=fast, adjust=False).mean()
        slow_ema = self.data[col].ewm(span=slow, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def momentum(self, window: int = 10, column: Optional[str] = None) -> pd.Series:
        """
        Price Momentum.
        
        Args:
            window: Lookback period
            column: Column to use (default: price_col)
        
        Returns:
            Series with momentum values
        """
        col = column or self.price_col
        return self.data[col].diff(window)
    
    def roc(self, window: int = 10, column: Optional[str] = None) -> pd.Series:
        """
        Rate of Change.
        
        Args:
            window: Lookback period
            column: Column to use (default: price_col)
        
        Returns:
            Series with ROC values (percent)
        """
        col = column or self.price_col
        return self.data[col].pct_change(periods=window) * 100
    
    def stochastic(
        self,
        k_window: int = 14,
        d_window: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            k_window: Period for %K
            d_window: Period for %D (signal)
        
        Returns:
            Tuple of (%K, %D)
        """
        if 'High' not in self.data.columns or 'Low' not in self.data.columns:
            raise ValueError("Stochastic requires High and Low columns")
        
        low_min = self.data['Low'].rolling(window=k_window).min()
        high_max = self.data['High'].rolling(window=k_window).max()
        
        # Avoid division by zero
        range_hl = (high_max - low_min).replace(0, np.nan)
        
        k = ((self.data[self.price_col] - low_min) / range_hl) * 100
        d = k.rolling(window=d_window).mean()
        
        return k.fillna(50), d.fillna(50)
    
    # =========================================================================
    # Volatility Indicators
    # =========================================================================
    
    def bollinger_bands(
        self,
        window: int = 20,
        num_std: float = 2.0,
        column: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            window: Period for moving average
            num_std: Number of standard deviations
            column: Column to use (default: price_col)
        
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        col = column or self.price_col
        
        middle = self.data[col].rolling(window=window).mean()
        std = self.data[col].rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return upper, middle, lower
    
    def atr(self, window: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            window: Period for ATR
        
        Returns:
            Series with ATR values
        """
        if 'High' not in self.data.columns or 'Low' not in self.data.columns:
            raise ValueError("ATR requires High and Low columns")
        
        high = self.data['High']
        low = self.data['Low']
        close = self.data[self.price_col]
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return true_range.rolling(window=window).mean()
    
    def keltner_channels(
        self,
        ema_window: int = 20,
        atr_window: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        
        Args:
            ema_window: Period for EMA
            atr_window: Period for ATR
            multiplier: ATR multiplier
        
        Returns:
            Tuple of (Upper Channel, Middle Line, Lower Channel)
        """
        middle = self.ema(window=ema_window)
        atr_val = self.atr(window=atr_window)
        
        upper = middle + (atr_val * multiplier)
        lower = middle - (atr_val * multiplier)
        
        return upper, middle, lower
    
    # =========================================================================
    # Volume Indicators
    # =========================================================================
    
    def volume_ratio(self, window: int = 20) -> pd.Series:
        """
        Volume Ratio (current volume / average volume).
        
        Args:
            window: Period for volume average
        
        Returns:
            Series with volume ratio
        """
        if 'Volume' not in self.data.columns:
            raise ValueError("Volume ratio requires Volume column")
        
        avg_volume = self.data['Volume'].rolling(window=window).mean()
        return self.data['Volume'] / avg_volume
    
    def obv(self) -> pd.Series:
        """
        On-Balance Volume.
        
        Returns:
            Series with OBV values
        """
        if 'Volume' not in self.data.columns:
            raise ValueError("OBV requires Volume column")
        
        price_change = self.data[self.price_col].diff()
        
        obv = pd.Series(0, index=self.data.index, dtype=float)
        obv[price_change > 0] = self.data.loc[price_change > 0, 'Volume']
        obv[price_change < 0] = -self.data.loc[price_change < 0, 'Volume']
        
        return obv.cumsum()
    
    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Returns:
            Series with VWAP values
        """
        if 'Volume' not in self.data.columns:
            raise ValueError("VWAP requires Volume column")
        
        # Typical price
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            typical_price = (
                self.data['High'] + 
                self.data['Low'] + 
                self.data[self.price_col]
            ) / 3
        else:
            typical_price = self.data[self.price_col]
        
        cumulative_tp_vol = (typical_price * self.data['Volume']).cumsum()
        cumulative_vol = self.data['Volume'].cumsum()
        
        return cumulative_tp_vol / cumulative_vol
    
    # =========================================================================
    # Price Relative Indicators
    # =========================================================================
    
    def price_to_sma(self, window: int = 200) -> pd.Series:
        """
        Price relative to SMA (percent).
        
        Args:
            window: SMA period
        
        Returns:
            Series with price/SMA ratio as percent from SMA
        """
        sma = self.sma(window=window)
        return ((self.data[self.price_col] / sma) - 1) * 100
    
    def high_low_range(self, window: int = 20) -> pd.Series:
        """
        Price position within rolling high-low range.
        
        Args:
            window: Lookback period
        
        Returns:
            Series with values 0-1 (0 = at low, 1 = at high)
        """
        if 'High' not in self.data.columns or 'Low' not in self.data.columns:
            high = self.data[self.price_col].rolling(window=window).max()
            low = self.data[self.price_col].rolling(window=window).min()
        else:
            high = self.data['High'].rolling(window=window).max()
            low = self.data['Low'].rolling(window=window).min()
        
        range_hl = (high - low).replace(0, np.nan)
        return (self.data[self.price_col] - low) / range_hl
    
    # =========================================================================
    # Add All Indicators
    # =========================================================================
    
    def add_all_indicators(
        self,
        sma_windows: List[int] = [5, 10, 20, 50, 200],
        ema_windows: List[int] = [5, 10, 20, 50],
        rsi_window: int = 14,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Add all indicators to the dataframe.
        
        Args:
            sma_windows: List of SMA periods
            ema_windows: List of EMA periods
            rsi_window: RSI period
            include_volume: Include volume-based indicators
        
        Returns:
            DataFrame with all indicators added
        """
        result = self.data.copy()
        
        # SMAs
        for w in sma_windows:
            result[f'SMA_{w}'] = self.sma(window=w)
        
        # EMAs
        for w in ema_windows:
            result[f'EMA_{w}'] = self.ema(window=w)
        
        # Momentum
        result[f'RSI_{rsi_window}'] = self.rsi(window=rsi_window)
        
        macd_line, signal, histogram = self.macd()
        result['MACD'] = macd_line
        result['MACD_Signal'] = signal
        result['MACD_Hist'] = histogram
        
        result['Momentum_10'] = self.momentum(window=10)
        result['ROC_10'] = self.roc(window=10)
        
        # Volatility
        upper, middle, lower = self.bollinger_bands()
        result['BB_Upper'] = upper
        result['BB_Middle'] = middle
        result['BB_Lower'] = lower
        result['BB_Width'] = (upper - lower) / middle
        
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            result['ATR_14'] = self.atr(window=14)
            
            k, d = self.stochastic()
            result['Stoch_K'] = k
            result['Stoch_D'] = d
        
        # Volume
        if include_volume and 'Volume' in self.data.columns:
            result['Volume_Ratio'] = self.volume_ratio(window=20)
            result['OBV'] = self.obv()
            result['VWAP'] = self.vwap()
        
        # Price relative
        result['Price_to_SMA200'] = self.price_to_sma(window=200)
        result['HL_Position'] = self.high_low_range(window=20)
        
        # Returns
        result['Return_1d'] = self.data[self.price_col].pct_change(1)
        result['Return_5d'] = self.data[self.price_col].pct_change(5)
        result['Return_20d'] = self.data[self.price_col].pct_change(20)
        
        # Volatility (rolling std of returns)
        result['Volatility_20d'] = result['Return_1d'].rolling(window=20).std() * np.sqrt(252)
        
        return result
