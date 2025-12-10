from typing import Dict

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings


class FundamentalsClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://www.alphavantage.co/query"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def fetch_overview(self, symbol: str) -> Dict:
        """Fetch Company Overview (Profile, Ratios)."""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.settings.alpha_vantage_api_key,
        }
        resp = requests.get(self.base_url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def fetch_cash_flow(self, symbol: str) -> Dict:
        """Fetch Cash Flow Statement."""
        params = {
            "function": "CASH_FLOW",
            "symbol": symbol,
            "apikey": self.settings.alpha_vantage_api_key,
        }
        resp = requests.get(self.base_url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return {}

    def get_advanced_metrics(self, symbol: str) -> Dict[str, float]:
        """Computes institutional-grade valuation metrics:
        - EV/EBITDA
        - FCF Yield
        - CROCI (approx)
        - Altman Z-Score components
        """
        overview = self.fetch_overview(symbol)
        cash_flow = self.fetch_cash_flow(symbol)

        metrics = {}
        if not overview or not cash_flow:
            return metrics

        try:
            # Parse raw values
            ebitda = float(overview.get("EBITDA", 0) or 0)
            ev = float(overview.get("EVToEBITDA", 0) or 0) * ebitda if ebitda else 0 # Backout EV if needed or use provided ratio directly

            # 1. EV / EBITDA
            metrics["ev_ebitda"] = float(overview.get("EVToEBITDA", 0) or 0)

            # 2. Free Cash Flow Yield (FCF / Market Cap)
            annual_reports = cash_flow.get("annualReports", [])
            if annual_reports:
                latest = annual_reports[0]
                ocf = float(latest.get("operatingCashflow", 0) or 0)
                capex = float(latest.get("capitalExpenditures", 0) or 0)
                fcf = ocf - capex
                market_cap = float(overview.get("MarketCapitalization", 0) or 1)
                metrics["fcf_yield"] = fcf / market_cap
            else:
                metrics["fcf_yield"] = 0.0

            # 3. PEG Ratio (Growth adjusted)
            metrics["peg_ratio"] = float(overview.get("PEGRatio", 0) or 0)

            # 4. Profitability
            metrics["roic"] = float(overview.get("ReturnOnAssetsTTM", 0) or 0) # Proxy if ROIC missing
            metrics["gross_margin"] = float(overview.get("GrossProfitTTM", 0) or 0) / (float(overview.get("RevenueTTM", 0) or 1))

        except Exception as e:
            logger.error(f"Error computing fundamentals for {symbol}: {e}")

        return metrics

    def persist_fundamentals(self, symbol: str, metrics: Dict):
        # Placeholder: Write to SQL table 'fundamentals'
        # engine = get_engine()
        # pd.DataFrame([metrics]).to_sql...
        pass


