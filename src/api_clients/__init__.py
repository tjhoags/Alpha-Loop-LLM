"""
API Client Wrappers
===================

Unified interfaces for all external data sources.
"""

from .alpha_vantage_client import AlphaVantageClient
from .ibkr_client import IBKRClient
from .coinbase_client import CoinbaseClient
from .google_client import GoogleSheetsClient
from .slack_client import SlackNotifier
from .openai_client import OpenAIAnalyzer

__all__ = [
    'AlphaVantageClient',
    'IBKRClient',
    'CoinbaseClient',
    'GoogleSheetsClient',
    'SlackNotifier',
    'OpenAIAnalyzer',
]

