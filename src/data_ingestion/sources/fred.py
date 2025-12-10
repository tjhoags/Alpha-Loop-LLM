from typing import Optional
import pandas as pd
from fredapi import Fred
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings


class FredClient:
    def __init__(self):
        self.settings = get_settings()
        if not self.settings.fred_api_key:
            logger.warning("FRED API Key missing. Macro data will be unavailable.")
            self.fred = None
        else:
            self.fred = Fred(api_key=self.settings.fred_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def fetch_series(self, series_id: str, observation_start: Optional[str] = None) -> pd.DataFrame:
        if not self.fred:
            return pd.DataFrame()
            
        try:
            logger.info(f"Fetching FRED series: {series_id}")
            data = self.fred.get_series(series_id, observation_start=observation_start)
            df = data.to_frame(name="value")
            df.index.name = "timestamp"
            df.reset_index(inplace=True)
            df["symbol"] = series_id
            df["source"] = "fred"
            return df
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            return pd.DataFrame()

    def fetch_core_macro_data(self) -> pd.DataFrame:
        """
        Fetches the 'Big 4' macro indicators + Central Bank Liquidity Components.
        
        Liquidity Formula = WALCL (Fed Assets) - WTREGEN (TGA) - RRPONTSYD (Reverse Repo)
        """
        series_ids = [
            "FEDFUNDS", "CPIAUCSL", "UNRATE", "VIXCLS", "DGS10", "T10Y2Y",
            "WALCL",     # Fed Total Assets
            "WTREGEN",   # Treasury General Account
            "RRPONTSYD"  # Overnight Reverse Repo
        ]
        frames = []
        start_date = "2000-01-01"
        
        for sid in series_ids:
            df = self.fetch_series(sid, observation_start=start_date)
            if not df.empty:
                frames.append(df)
                
        if not frames:
            return pd.DataFrame()
            
        combined = pd.concat(frames, ignore_index=True)
        
        # Calculate Net Liquidity if columns exist (requires pivoting or separate logic)
        # For simple ingestion, we store raw series. 
        # Feature Engineering (feature_engineering.py) will pivot and compute:
        # Net_Liq = WALCL - WTREGEN - RRPONTSYD
        
        return combined

