"""
Unified Data Ingestion Module for Alpha Loop Capital.

Exposes:
- MarketDataCollector: Unified interface for Yahoo Finance/Alpha Vantage/IBKR
- DatasetBuilder: Builds ML-ready datasets (Prices + Fundamentals + Macro)
- PortfolioIngestion: Ingests and analyzes trade history
- AlphaVantageClient: Direct access to Alpha Vantage API
"""

from .collector import MarketDataCollector
from .dataset_builder import DatasetBuilder
from .portfolio import PortfolioIngestion, Trade, Portfolio
from .alpha_vantage import AlphaVantageClient

__all__ = [
    'MarketDataCollector',
    'DatasetBuilder',
    'PortfolioIngestion',
    'Trade',
    'Portfolio',
    'AlphaVantageClient'
]
