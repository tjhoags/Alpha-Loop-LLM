"""
Coinbase API Client
===================
Client for cryptocurrency data from Coinbase.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinbaseClient:
    """
    Coinbase API Client for cryptocurrency data.
    
    Note: Coinbase has deprecated their old API.
    Consider using Coinbase Advanced Trade API or alternative sources.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")
        self.available = False
        
        try:
            # Note: coinbase package may require updates for new API
            import requests
            self.session = requests.Session()
            self.base_url = "https://api.coinbase.com/v2"
            self.available = True
        except Exception as e:
            logger.warning(f"Coinbase client initialization failed: {e}")
    
    def get_spot_price(self, currency_pair: str = "BTC-USD") -> Optional[Dict]:
        """Get current spot price"""
        if not self.available:
            return None
        
        try:
            response = self.session.get(f"{self.base_url}/prices/{currency_pair}/spot")
            if response.ok:
                data = response.json()
                return {
                    "pair": currency_pair,
                    "price": float(data["data"]["amount"]),
                    "currency": data["data"]["currency"]
                }
        except Exception as e:
            logger.error(f"Error getting spot price: {e}")
        
        return None
    
    def get_exchange_rates(self, currency: str = "USD") -> Dict[str, float]:
        """Get exchange rates for all cryptocurrencies"""
        if not self.available:
            return {}
        
        try:
            response = self.session.get(f"{self.base_url}/exchange-rates?currency={currency}")
            if response.ok:
                data = response.json()
                return {k: float(v) for k, v in data["data"]["rates"].items()}
        except Exception as e:
            logger.error(f"Error getting exchange rates: {e}")
        
        return {}


class GoogleSheetsClient:
    """
    Google Sheets Client for spreadsheet integration.
    """
    
    def __init__(self, credentials_path: str = None):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        self.available = False
        self.client = None
        
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            
            if self.credentials_path and os.path.exists(self.credentials_path):
                scopes = [
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
                creds = Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=scopes
                )
                self.client = gspread.authorize(creds)
                self.available = True
                logger.info("Google Sheets client initialized")
        except Exception as e:
            logger.warning(f"Google Sheets not available: {e}")
    
    def read_sheet(self, spreadsheet_id: str, sheet_name: str = None) -> List[Dict]:
        """Read data from a Google Sheet"""
        if not self.available:
            return []
        
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            if sheet_name:
                worksheet = spreadsheet.worksheet(sheet_name)
            else:
                worksheet = spreadsheet.sheet1
            
            return worksheet.get_all_records()
        except Exception as e:
            logger.error(f"Error reading sheet: {e}")
            return []
    
    def write_to_sheet(
        self,
        spreadsheet_id: str,
        data: List[List],
        sheet_name: str = None
    ) -> bool:
        """Write data to a Google Sheet"""
        if not self.available:
            return False
        
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            if sheet_name:
                worksheet = spreadsheet.worksheet(sheet_name)
            else:
                worksheet = spreadsheet.sheet1
            
            worksheet.update('A1', data)
            return True
        except Exception as e:
            logger.error(f"Error writing to sheet: {e}")
            return False


if __name__ == "__main__":
    # Demo
    client = CoinbaseClient()
    
    if client.available:
        btc = client.get_spot_price("BTC-USD")
        if btc:
            print(f"BTC Price: ${btc['price']:,.2f}")
        
        eth = client.get_spot_price("ETH-USD")
        if eth:
            print(f"ETH Price: ${eth['price']:,.2f}")

