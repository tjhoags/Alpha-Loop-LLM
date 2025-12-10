"""
Interactive Brokers (IBKR) TWS API Client
==========================================
Wrapper for IBKR TWS/Gateway connection using ib_insync.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional import - may not be installed
try:
    from ib_insync import IB, Stock, Option, Contract, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. IBKR client will be unavailable.")


class IBKRClient:
    """
    Interactive Brokers TWS API Client
    
    Requires TWS or IB Gateway running with API enabled.
    
    Setup:
    1. Enable API in TWS: Configure > API > Settings
    2. Set socket port (default 7497 for TWS paper, 4001 for Gateway)
    3. Disable "Read-Only API" for trading
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        client_id: int = None
    ):
        if not IB_AVAILABLE:
            raise ImportError("ib_insync package required. Install with: pip install ib_insync")
        
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = client_id or int(os.getenv("IBKR_CLIENT_ID", "1"))
        
        self.ib = IB()
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from TWS/Gateway"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def get_account_summary(self) -> Dict:
        """Get account summary"""
        if not self.connected:
            self.connect()
        
        summary = self.ib.accountSummary()
        
        result = {}
        for item in summary:
            result[item.tag] = {
                "value": item.value,
                "currency": item.currency
            }
        
        return result
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.connected:
            self.connect()
        
        positions = self.ib.positions()
        
        result = []
        for pos in positions:
            result.append({
                "account": pos.account,
                "symbol": pos.contract.symbol,
                "sec_type": pos.contract.secType,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost
            })
        
        return result
    
    def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES"
    ) -> pd.DataFrame:
        """
        Get historical bar data.
        
        Parameters:
            symbol: Ticker symbol
            duration: "1 D", "1 W", "1 M", "1 Y"
            bar_size: "1 min", "5 mins", "1 hour", "1 day"
            what_to_show: "TRADES", "MIDPOINT", "BID", "ASK"
        """
        if not self.connected:
            self.connect()
        
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True
        )
        
        df = util.df(bars)
        if not df.empty:
            df.set_index("date", inplace=True)
        
        return df
    
    def get_option_chain(self, symbol: str) -> List[Dict]:
        """Get option chain for a symbol"""
        if not self.connected:
            self.connect()
        
        stock = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(stock)
        
        chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        
        result = []
        for chain in chains:
            result.append({
                "exchange": chain.exchange,
                "underlying_conid": chain.underlyingConId,
                "trading_class": chain.tradingClass,
                "multiplier": chain.multiplier,
                "expirations": list(chain.expirations),
                "strikes": list(chain.strikes)
            })
        
        return result
    
    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: float = None
    ) -> Dict:
        """
        Place an order.
        
        Parameters:
            symbol: Ticker symbol
            action: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MKT", "LMT", "STP", "STP_LMT"
            limit_price: Price for limit orders
        """
        if not self.connected:
            self.connect()
        
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        
        if order_type == "MKT":
            from ib_insync import MarketOrder
            order = MarketOrder(action, quantity)
        elif order_type == "LMT":
            from ib_insync import LimitOrder
            order = LimitOrder(action, quantity, limit_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        trade = self.ib.placeOrder(contract, order)
        
        return {
            "order_id": trade.order.orderId,
            "status": trade.orderStatus.status,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": order_type
        }
    
    def get_trades(self) -> List[Dict]:
        """Get today's trades"""
        if not self.connected:
            self.connect()
        
        trades = self.ib.trades()
        
        result = []
        for trade in trades:
            result.append({
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "avg_fill_price": trade.orderStatus.avgFillPrice
            })
        
        return result


if __name__ == "__main__":
    if IB_AVAILABLE:
        client = IBKRClient()
        if client.connect():
            # Get account summary
            summary = client.get_account_summary()
            print(f"Account Summary: {summary.get('NetLiquidation', 'N/A')}")
            
            # Get positions
            positions = client.get_positions()
            print(f"\nPositions: {len(positions)}")
            for pos in positions[:5]:
                print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_cost']:.2f}")
            
            client.disconnect()
    else:
        print("IBKR client unavailable - ib_insync not installed")

