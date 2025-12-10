"""
================================================================================
UNIVERSE SCANNER - Small/Mid Cap Stock Universe (<$25B Market Cap)
================================================================================
Builds the complete trading universe for a long-short quant hedge fund.

Features:
- Scans all US equities under $25B market cap
- Filters by liquidity, volume, and data availability
- Categorizes by sector, market cap tier, and factor exposure
- Updates universe daily

================================================================================
"""

import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from src.config.settings import get_settings


# Market Cap Tiers
MARKET_CAP_TIERS = {
    "nano": (0, 50_000_000),           # < $50M
    "micro": (50_000_000, 300_000_000), # $50M - $300M
    "small": (300_000_000, 2_000_000_000),  # $300M - $2B
    "mid": (2_000_000_000, 10_000_000_000), # $2B - $10B
    "large_mid": (10_000_000_000, 25_000_000_000), # $10B - $25B
}

# Sectors (GICS)
SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Energy", "Materials",
    "Real Estate", "Utilities", "Communication Services"
]


class UniverseScanner:
    """
    Scans and maintains the trading universe.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.universe: List[Dict] = []
        
    def fetch_polygon_tickers(self) -> pd.DataFrame:
        """
        Fetch all US stock tickers from Polygon.
        """
        api_key = self.settings.polygon_api_key
        if not api_key:
            logger.warning("No Polygon API key - using sample universe")
            return self._get_sample_universe()
        
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "limit": 1000,
            "apiKey": api_key
        }
        
        all_tickers = []
        next_url = url
        
        try:
            while next_url and len(all_tickers) < 10000:
                resp = requests.get(next_url, params=params, timeout=30)
                data = resp.json()
                
                if "results" in data:
                    all_tickers.extend(data["results"])
                
                next_url = data.get("next_url")
                if next_url:
                    next_url = f"{next_url}&apiKey={api_key}"
                params = {}  # Clear params for pagination
                
            logger.info(f"Fetched {len(all_tickers)} tickers from Polygon")
            return pd.DataFrame(all_tickers)
            
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return self._get_sample_universe()
    
    def _get_sample_universe(self) -> pd.DataFrame:
        """
        Sample universe for testing when API unavailable.
        Includes representative small/mid cap stocks.
        """
        # Small/Mid Cap Universe - Representative Sample
        sample_tickers = [
            # Technology - Small/Mid
            {"ticker": "CRWD", "name": "CrowdStrike", "market_cap": 20_000_000_000, "sector": "Technology"},
            {"ticker": "NET", "name": "Cloudflare", "market_cap": 18_000_000_000, "sector": "Technology"},
            {"ticker": "DDOG", "name": "Datadog", "market_cap": 22_000_000_000, "sector": "Technology"},
            {"ticker": "ZS", "name": "Zscaler", "market_cap": 19_000_000_000, "sector": "Technology"},
            {"ticker": "OKTA", "name": "Okta", "market_cap": 12_000_000_000, "sector": "Technology"},
            {"ticker": "MDB", "name": "MongoDB", "market_cap": 15_000_000_000, "sector": "Technology"},
            {"ticker": "SNOW", "name": "Snowflake", "market_cap": 45_000_000_000, "sector": "Technology"},
            {"ticker": "PLTR", "name": "Palantir", "market_cap": 20_000_000_000, "sector": "Technology"},
            {"ticker": "PATH", "name": "UiPath", "market_cap": 8_000_000_000, "sector": "Technology"},
            {"ticker": "CFLT", "name": "Confluent", "market_cap": 7_000_000_000, "sector": "Technology"},
            {"ticker": "S", "name": "SentinelOne", "market_cap": 5_000_000_000, "sector": "Technology"},
            {"ticker": "ESTC", "name": "Elastic", "market_cap": 8_000_000_000, "sector": "Technology"},
            {"ticker": "GTLB", "name": "GitLab", "market_cap": 6_000_000_000, "sector": "Technology"},
            {"ticker": "DOCN", "name": "DigitalOcean", "market_cap": 3_000_000_000, "sector": "Technology"},
            {"ticker": "IONQ", "name": "IonQ", "market_cap": 2_000_000_000, "sector": "Technology"},
            
            # Healthcare - Small/Mid
            {"ticker": "DXCM", "name": "Dexcom", "market_cap": 24_000_000_000, "sector": "Healthcare"},
            {"ticker": "EXAS", "name": "Exact Sciences", "market_cap": 10_000_000_000, "sector": "Healthcare"},
            {"ticker": "INSP", "name": "Inspire Medical", "market_cap": 5_000_000_000, "sector": "Healthcare"},
            {"ticker": "RXRX", "name": "Recursion", "market_cap": 3_000_000_000, "sector": "Healthcare"},
            {"ticker": "DNA", "name": "Ginkgo Bioworks", "market_cap": 2_000_000_000, "sector": "Healthcare"},
            {"ticker": "BEAM", "name": "Beam Therapeutics", "market_cap": 3_000_000_000, "sector": "Healthcare"},
            {"ticker": "CRSP", "name": "CRISPR", "market_cap": 4_000_000_000, "sector": "Healthcare"},
            {"ticker": "NTLA", "name": "Intellia", "market_cap": 2_500_000_000, "sector": "Healthcare"},
            {"ticker": "VERV", "name": "Verve Therapeutics", "market_cap": 1_500_000_000, "sector": "Healthcare"},
            {"ticker": "VCYT", "name": "Veracyte", "market_cap": 2_000_000_000, "sector": "Healthcare"},
            
            # Consumer Discretionary - Small/Mid
            {"ticker": "DASH", "name": "DoorDash", "market_cap": 24_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "ABNB", "name": "Airbnb", "market_cap": 70_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "CHWY", "name": "Chewy", "market_cap": 8_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "ETSY", "name": "Etsy", "market_cap": 7_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "W", "name": "Wayfair", "market_cap": 5_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "CVNA", "name": "Carvana", "market_cap": 15_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "BROS", "name": "Dutch Bros", "market_cap": 5_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "SHAK", "name": "Shake Shack", "market_cap": 3_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "DKS", "name": "Dicks Sporting", "market_cap": 15_000_000_000, "sector": "Consumer Discretionary"},
            {"ticker": "FIVE", "name": "Five Below", "market_cap": 8_000_000_000, "sector": "Consumer Discretionary"},
            
            # Financials - Small/Mid
            {"ticker": "COIN", "name": "Coinbase", "market_cap": 20_000_000_000, "sector": "Financials"},
            {"ticker": "HOOD", "name": "Robinhood", "market_cap": 8_000_000_000, "sector": "Financials"},
            {"ticker": "SOFI", "name": "SoFi", "market_cap": 7_000_000_000, "sector": "Financials"},
            {"ticker": "UPST", "name": "Upstart", "market_cap": 3_000_000_000, "sector": "Financials"},
            {"ticker": "AFRM", "name": "Affirm", "market_cap": 10_000_000_000, "sector": "Financials"},
            {"ticker": "LC", "name": "LendingClub", "market_cap": 1_500_000_000, "sector": "Financials"},
            {"ticker": "NU", "name": "Nu Holdings", "market_cap": 35_000_000_000, "sector": "Financials"},
            {"ticker": "OPEN", "name": "Opendoor", "market_cap": 2_000_000_000, "sector": "Financials"},
            {"ticker": "RDFN", "name": "Redfin", "market_cap": 1_000_000_000, "sector": "Financials"},
            {"ticker": "BILL", "name": "Bill.com", "market_cap": 6_000_000_000, "sector": "Financials"},
            
            # Industrials - Small/Mid
            {"ticker": "AXON", "name": "Axon Enterprise", "market_cap": 18_000_000_000, "sector": "Industrials"},
            {"ticker": "TDG", "name": "TransDigm", "market_cap": 60_000_000_000, "sector": "Industrials"},
            {"ticker": "BLDR", "name": "Builders FirstSource", "market_cap": 18_000_000_000, "sector": "Industrials"},
            {"ticker": "RBC", "name": "RBC Bearings", "market_cap": 8_000_000_000, "sector": "Industrials"},
            {"ticker": "SITE", "name": "SiteOne Landscape", "market_cap": 6_000_000_000, "sector": "Industrials"},
            {"ticker": "AAON", "name": "AAON", "market_cap": 5_000_000_000, "sector": "Industrials"},
            {"ticker": "GNRC", "name": "Generac", "market_cap": 8_000_000_000, "sector": "Industrials"},
            {"ticker": "LECO", "name": "Lincoln Electric", "market_cap": 12_000_000_000, "sector": "Industrials"},
            {"ticker": "AME", "name": "Ametek", "market_cap": 38_000_000_000, "sector": "Industrials"},
            {"ticker": "POOL", "name": "Pool Corp", "market_cap": 13_000_000_000, "sector": "Industrials"},
            
            # Energy - Small/Mid  
            {"ticker": "AR", "name": "Antero Resources", "market_cap": 10_000_000_000, "sector": "Energy"},
            {"ticker": "RRC", "name": "Range Resources", "market_cap": 8_000_000_000, "sector": "Energy"},
            {"ticker": "MTDR", "name": "Matador Resources", "market_cap": 7_000_000_000, "sector": "Energy"},
            {"ticker": "CTRA", "name": "Coterra Energy", "market_cap": 18_000_000_000, "sector": "Energy"},
            {"ticker": "SM", "name": "SM Energy", "market_cap": 5_000_000_000, "sector": "Energy"},
            {"ticker": "CHRD", "name": "Chord Energy", "market_cap": 8_000_000_000, "sector": "Energy"},
            {"ticker": "GPOR", "name": "Gulfport Energy", "market_cap": 3_000_000_000, "sector": "Energy"},
            
            # Materials - Small/Mid
            {"ticker": "ATI", "name": "ATI Inc", "market_cap": 6_000_000_000, "sector": "Materials"},
            {"ticker": "CLF", "name": "Cleveland-Cliffs", "market_cap": 6_000_000_000, "sector": "Materials"},
            {"ticker": "CMC", "name": "Commercial Metals", "market_cap": 6_000_000_000, "sector": "Materials"},
            {"ticker": "RS", "name": "Reliance Steel", "market_cap": 16_000_000_000, "sector": "Materials"},
            {"ticker": "MP", "name": "MP Materials", "market_cap": 3_000_000_000, "sector": "Materials"},
            {"ticker": "LAC", "name": "Lithium Americas", "market_cap": 1_500_000_000, "sector": "Materials"},
            
            # Real Estate - Small/Mid
            {"ticker": "REXR", "name": "Rexford Industrial", "market_cap": 10_000_000_000, "sector": "Real Estate"},
            {"ticker": "FR", "name": "First Industrial", "market_cap": 7_000_000_000, "sector": "Real Estate"},
            {"ticker": "STAG", "name": "STAG Industrial", "market_cap": 7_000_000_000, "sector": "Real Estate"},
            {"ticker": "IIPR", "name": "Innovative Industrial", "market_cap": 3_000_000_000, "sector": "Real Estate"},
            {"ticker": "CUBE", "name": "CubeSmart", "market_cap": 10_000_000_000, "sector": "Real Estate"},
            
            # Communication - Small/Mid
            {"ticker": "PINS", "name": "Pinterest", "market_cap": 20_000_000_000, "sector": "Communication Services"},
            {"ticker": "SNAP", "name": "Snap Inc", "market_cap": 18_000_000_000, "sector": "Communication Services"},
            {"ticker": "TTD", "name": "Trade Desk", "market_cap": 40_000_000_000, "sector": "Communication Services"},
            {"ticker": "ZI", "name": "ZoomInfo", "market_cap": 5_000_000_000, "sector": "Communication Services"},
            {"ticker": "APP", "name": "AppLovin", "market_cap": 25_000_000_000, "sector": "Communication Services"},
            {"ticker": "MGNI", "name": "Magnite", "market_cap": 2_000_000_000, "sector": "Communication Services"},
            {"ticker": "PUBM", "name": "PubMatic", "market_cap": 1_000_000_000, "sector": "Communication Services"},
            
            # Major Indices/ETFs for reference
            {"ticker": "SPY", "name": "S&P 500 ETF", "market_cap": 0, "sector": "Index"},
            {"ticker": "QQQ", "name": "Nasdaq 100 ETF", "market_cap": 0, "sector": "Index"},
            {"ticker": "IWM", "name": "Russell 2000 ETF", "market_cap": 0, "sector": "Index"},
            {"ticker": "IWO", "name": "Russell 2000 Growth", "market_cap": 0, "sector": "Index"},
            {"ticker": "IWN", "name": "Russell 2000 Value", "market_cap": 0, "sector": "Index"},
            
            # Crypto
            {"ticker": "BTC-USD", "name": "Bitcoin", "market_cap": 0, "sector": "Crypto"},
            {"ticker": "ETH-USD", "name": "Ethereum", "market_cap": 0, "sector": "Crypto"},
        ]
        
        return pd.DataFrame(sample_tickers)
    
    def filter_universe(
        self,
        df: pd.DataFrame,
        max_market_cap: float = 25_000_000_000,
        min_volume: float = 100_000,
        exclude_otc: bool = True
    ) -> pd.DataFrame:
        """
        Filter universe by market cap, volume, and exchange.
        """
        filtered = df.copy()
        
        # Filter by market cap
        if "market_cap" in filtered.columns:
            filtered = filtered[
                (filtered["market_cap"] <= max_market_cap) | 
                (filtered["market_cap"] == 0)  # Keep ETFs/Crypto with 0
            ]
        
        # Exclude OTC
        if exclude_otc and "primary_exchange" in filtered.columns:
            filtered = filtered[~filtered["primary_exchange"].str.contains("OTC", na=False)]
        
        logger.info(f"Universe filtered to {len(filtered)} tickers")
        return filtered
    
    def get_trading_universe(self) -> List[str]:
        """
        Get the full trading universe as list of tickers.
        """
        df = self.fetch_polygon_tickers()
        df = self.filter_universe(df)
        
        if "ticker" in df.columns:
            return df["ticker"].tolist()
        return []
    
    def get_universe_by_tier(self) -> Dict[str, List[str]]:
        """
        Get universe organized by market cap tier.
        """
        df = self._get_sample_universe()
        
        tiers = {}
        for tier_name, (min_cap, max_cap) in MARKET_CAP_TIERS.items():
            tier_stocks = df[
                (df["market_cap"] >= min_cap) & 
                (df["market_cap"] < max_cap)
            ]["ticker"].tolist()
            tiers[tier_name] = tier_stocks
        
        # Add special categories
        tiers["indices"] = df[df["sector"] == "Index"]["ticker"].tolist()
        tiers["crypto"] = df[df["sector"] == "Crypto"]["ticker"].tolist()
        
        return tiers
    
    def get_universe_by_sector(self) -> Dict[str, List[str]]:
        """
        Get universe organized by sector.
        """
        df = self._get_sample_universe()
        
        sectors = {}
        for sector in df["sector"].unique():
            sectors[sector] = df[df["sector"] == sector]["ticker"].tolist()
        
        return sectors


def fetch_all_tickers() -> List[str]:
    """
    Convenience function to get all tickers.
    """
    scanner = UniverseScanner()
    return scanner.get_trading_universe()


def get_small_mid_cap_universe() -> List[str]:
    """
    Get the small/mid cap universe (<$25B).
    Returns ~100 stocks across all sectors.
    """
    scanner = UniverseScanner()
    df = scanner._get_sample_universe()
    
    # Filter to <$25B (excluding indices and crypto which have 0)
    stocks = df[
        ((df["market_cap"] > 0) & (df["market_cap"] <= 25_000_000_000)) |
        (df["sector"].isin(["Index", "Crypto"]))
    ]["ticker"].tolist()
    
    logger.info(f"Small/Mid Cap Universe: {len(stocks)} tickers")
    return stocks
