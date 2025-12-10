import asyncio

from ib_insync import IB, Order, Stock
from loguru import logger

from src.config.settings_new import get_settings


class SmartExecution:
    """Advanced Execution Logic to minimize slippage and maximize speed.
    """

    def __init__(self, ib: IB):
        self.ib = ib
        self.settings = get_settings()

    async def execute_smart_order(self, symbol: str, action: str, quantity: float):
        """Tries to fill at Mid-Price first (Limit Order).
        If not filled in 'timelimit' seconds, crosses the spread (Market Order).
        """
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)

        # 1. Get Live Quote for Spread Check
        ticker = self.ib.reqMktData(contract, "", False, False)
        timeout = 0
        while ticker.bid < 0 or ticker.ask < 0:
            await asyncio.sleep(0.1)
            timeout += 1
            if timeout > 20: # 2 sec timeout for data
                break

        # Default to Market if no quote
        if ticker.bid <= 0 or ticker.ask <= 0:
            logger.warning(f"No quote for {symbol}. Sending MKT order immediately.")
            return self.place_market_order(contract, action, quantity)

        # 2. Calculate Mid-Price
        mid_price = (ticker.bid + ticker.ask) / 2
        spread_pct = (ticker.ask - ticker.bid) / mid_price

        logger.info(f"{symbol} Quote: {ticker.bid} / {ticker.ask} (Spread: {spread_pct:.4%})")

        # 3. Strategy Selection
        if spread_pct > 0.005: # > 0.5% Spread - Liquidty is thin, be careful
            logger.info(f"Wide spread detected on {symbol}. Working limit order at mid.")
            # Place Limit at Mid
            lmt_order = Order(action=action, totalQuantity=quantity, orderType="LMT", lmtPrice=round(mid_price, 2))
            trade = self.ib.placeOrder(contract, lmt_order)

            # 4. Monitor Fill (The "Chasing" Logic)
            wait_time = 0
            while not trade.isDone():
                await asyncio.sleep(1)
                wait_time += 1
                if wait_time > 10: # Wait 10 seconds
                    logger.info(f"Limit not filled for {symbol}. Aggressing to Market.")
                    # Cancel/Replace with Market
                    self.ib.cancelOrder(lmt_order)
                    return self.place_market_order(contract, action, quantity)
            return trade

        else:
            # Tight spread, just take it (MKT)
            return self.place_market_order(contract, action, quantity)

    def place_market_order(self, contract, action, quantity):
        order = Order(action=action, totalQuantity=quantity, orderType="MKT")
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Sent MARKET {action} for {quantity} {contract.symbol}")
        return trade


