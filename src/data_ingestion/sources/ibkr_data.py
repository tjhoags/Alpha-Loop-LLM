"""================================================================================
IBKR DATA SERVICE - Interactive Brokers Data Integration
================================================================================
Author: Chris Friedman | Alpha Loop Capital, LLC

Provides read-only access to IBKR account data for fund operations agents.
Used by SANTAS_HELPER and CPA for:
- Portfolio positions and valuations
- Account balances and cash
- Trade history and executions
- P&L tracking (realized/unrealized)
- NAV calculation inputs

This is a READ-ONLY service. Trade execution is handled by ExecutionAgent.

Port Reference:
- 7497 = Paper Trading
- 7496 = Live Trading
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# IBKR SDK availability
try:
    from ib_insync import IB
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logger.warning("ib_insync not installed. IBKR data will use simulation mode.")


class AssetClass(Enum):
    """Asset class types from IBKR"""
    EQUITY = "STK"
    OPTION = "OPT"
    FUTURE = "FUT"
    FOREX = "CASH"
    CRYPTO = "CRYPTO"
    BOND = "BOND"
    INDEX = "IND"


@dataclass
class IBKRPosition:
    """Position data from IBKR"""
    symbol: str
    asset_class: AssetClass
    quantity: Decimal
    avg_cost: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    account: str
    currency: str = "USD"

    @property
    def total_cost(self) -> Decimal:
        return self.quantity * self.avg_cost

    @property
    def pnl_percent(self) -> Decimal:
        if self.total_cost == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "quantity": float(self.quantity),
            "avg_cost": float(self.avg_cost),
            "market_price": float(self.market_price),
            "market_value": float(self.market_value),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "pnl_percent": float(self.pnl_percent),
            "account": self.account,
            "currency": self.currency,
        }


@dataclass
class IBKRAccountSummary:
    """Account summary from IBKR"""
    account_id: str
    net_liquidation: Decimal
    total_cash: Decimal
    settled_cash: Decimal
    buying_power: Decimal
    gross_position_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    maintenance_margin: Decimal
    available_funds: Decimal
    excess_liquidity: Decimal
    cushion: Decimal  # Excess liquidity as % of net liquidation
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "net_liquidation": float(self.net_liquidation),
            "total_cash": float(self.total_cash),
            "settled_cash": float(self.settled_cash),
            "buying_power": float(self.buying_power),
            "gross_position_value": float(self.gross_position_value),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "maintenance_margin": float(self.maintenance_margin),
            "available_funds": float(self.available_funds),
            "excess_liquidity": float(self.excess_liquidity),
            "cushion_pct": float(self.cushion * 100),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IBKRExecution:
    """Trade execution from IBKR"""
    exec_id: str
    symbol: str
    side: str  # BOT or SLD
    quantity: Decimal
    price: Decimal
    commission: Decimal
    account: str
    timestamp: datetime
    order_id: int
    exchange: str = ""

    @property
    def total_value(self) -> Decimal:
        return self.quantity * self.price

    @property
    def net_value(self) -> Decimal:
        """Net value after commission"""
        if self.side == "BOT":
            return -(self.total_value + self.commission)
        return self.total_value - self.commission

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "total_value": float(self.total_value),
            "commission": float(self.commission),
            "net_value": float(self.net_value),
            "account": self.account,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "exchange": self.exchange,
        }


class IBKRDataService:
    """
    Read-only IBKR data service for fund operations.

    Provides:
    - Real-time positions and valuations
    - Account balances and cash
    - Trade history
    - P&L calculations

    Used by SANTAS_HELPER for NAV calculations and CPA for tax reporting.
    """

    def __init__(self, simulation_mode: bool = None):
        """
        Initialize IBKR data service.

        Args:
            simulation_mode: Force simulation mode. If None, auto-detects.
        """
        self.settings = get_settings()
        self.simulation_mode = simulation_mode if simulation_mode is not None else not IBKR_AVAILABLE
        self.ib: Optional[IB] = None
        self._connected = False

        # Cache for reducing API calls
        self._position_cache: Dict[str, List[IBKRPosition]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

        logger.info(f"IBKRDataService initialized (simulation={self.simulation_mode})")

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def connect(self) -> bool:
        """Connect to IBKR (read-only data access)."""
        if self.simulation_mode:
            logger.info("IBKR Data Service in SIMULATION mode")
            self._connected = True
            return True

        if not IBKR_AVAILABLE:
            logger.warning("ib_insync not available. Using simulation mode.")
            self.simulation_mode = True
            self._connected = True
            return True

        try:
            self.ib = IB()
            # Use a different client ID to avoid conflicts with ExecutionEngine
            client_id = self.settings.ibkr_client_id + 100

            self.ib.connect(
                host=self.settings.ibkr_host,
                port=self.settings.ibkr_port,
                clientId=client_id,
                readonly=True  # Read-only connection
            )

            self._connected = self.ib.isConnected()

            if self._connected:
                accounts = self.ib.managedAccounts()
                logger.info(f"Connected to IBKR (read-only). Accounts: {accounts}")

            return self._connected

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            logger.info("Falling back to simulation mode")
            self.simulation_mode = True
            self._connected = True
            return True

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")
        self._connected = False

    def ensure_connected(self) -> bool:
        """Ensure we're connected, reconnect if needed."""
        if not self._connected:
            return self.connect()
        if self.ib and not self.ib.isConnected():
            return self.connect()
        return True

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def get_positions(self, account: str = None) -> List[IBKRPosition]:
        """
        Get all positions from IBKR.

        Args:
            account: Specific account (None for all)

        Returns:
            List of positions
        """
        if not self.ensure_connected():
            return []

        # Check cache
        cache_key = account or "ALL"
        if self._is_cache_valid() and cache_key in self._position_cache:
            return self._position_cache[cache_key]

        if self.simulation_mode:
            positions = self._get_simulated_positions()
        else:
            positions = self._get_live_positions(account)

        # Update cache
        self._position_cache[cache_key] = positions
        self._cache_timestamp = datetime.now()

        return positions

    def _get_live_positions(self, account: str = None) -> List[IBKRPosition]:
        """Get positions from live IBKR connection."""
        positions = []

        try:
            portfolio = self.ib.portfolio(account)

            for item in portfolio:
                pos = IBKRPosition(
                    symbol=item.contract.symbol,
                    asset_class=self._map_asset_class(item.contract.secType),
                    quantity=Decimal(str(item.position)),
                    avg_cost=Decimal(str(item.averageCost)),
                    market_price=Decimal(str(item.marketPrice)),
                    market_value=Decimal(str(item.marketValue)),
                    unrealized_pnl=Decimal(str(item.unrealizedPNL)),
                    realized_pnl=Decimal(str(item.realizedPNL)),
                    account=item.account,
                )
                positions.append(pos)

        except Exception as e:
            logger.error(f"Error getting IBKR positions: {e}")

        return positions

    def _get_simulated_positions(self) -> List[IBKRPosition]:
        """Get simulated positions for demo/testing."""
        # Simulated portfolio for training
        return [
            IBKRPosition(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                quantity=Decimal("500"),
                avg_cost=Decimal("175.50"),
                market_price=Decimal("185.25"),
                market_value=Decimal("92625"),
                unrealized_pnl=Decimal("4875"),
                realized_pnl=Decimal("2500"),
                account="PAPER",
            ),
            IBKRPosition(
                symbol="MSFT",
                asset_class=AssetClass.EQUITY,
                quantity=Decimal("300"),
                avg_cost=Decimal("380.00"),
                market_price=Decimal("412.50"),
                market_value=Decimal("123750"),
                unrealized_pnl=Decimal("9750"),
                realized_pnl=Decimal("5000"),
                account="PAPER",
            ),
            IBKRPosition(
                symbol="NVDA",
                asset_class=AssetClass.EQUITY,
                quantity=Decimal("200"),
                avg_cost=Decimal("450.00"),
                market_price=Decimal("520.00"),
                market_value=Decimal("104000"),
                unrealized_pnl=Decimal("14000"),
                realized_pnl=Decimal("8000"),
                account="PAPER",
            ),
            IBKRPosition(
                symbol="SPY",
                asset_class=AssetClass.EQUITY,
                quantity=Decimal("1000"),
                avg_cost=Decimal("480.00"),
                market_price=Decimal("505.00"),
                market_value=Decimal("505000"),
                unrealized_pnl=Decimal("25000"),
                realized_pnl=Decimal("15000"),
                account="PAPER",
            ),
            IBKRPosition(
                symbol="BTC",
                asset_class=AssetClass.CRYPTO,
                quantity=Decimal("2.5"),
                avg_cost=Decimal("42000"),
                market_price=Decimal("98500"),
                market_value=Decimal("246250"),
                unrealized_pnl=Decimal("141250"),
                realized_pnl=Decimal("25000"),
                account="PAPER",
            ),
        ]

    def _map_asset_class(self, sec_type: str) -> AssetClass:
        """Map IBKR security type to our AssetClass."""
        mapping = {
            "STK": AssetClass.EQUITY,
            "OPT": AssetClass.OPTION,
            "FUT": AssetClass.FUTURE,
            "CASH": AssetClass.FOREX,
            "CRYPTO": AssetClass.CRYPTO,
            "BOND": AssetClass.BOND,
            "IND": AssetClass.INDEX,
        }
        return mapping.get(sec_type, AssetClass.EQUITY)

    # =========================================================================
    # ACCOUNT SUMMARY
    # =========================================================================

    def get_account_summary(self, account: str = None) -> Optional[IBKRAccountSummary]:
        """
        Get account summary from IBKR.

        Args:
            account: Specific account (None for first account)

        Returns:
            Account summary
        """
        if not self.ensure_connected():
            return None

        if self.simulation_mode:
            return self._get_simulated_account_summary()

        return self._get_live_account_summary(account)

    def _get_live_account_summary(self, account: str = None) -> Optional[IBKRAccountSummary]:
        """Get account summary from live IBKR."""
        try:
            if not account:
                accounts = self.ib.managedAccounts()
                if not accounts:
                    return None
                account = accounts[0]

            # Request account values
            self.ib.reqAccountSummary()
            self.ib.sleep(2)

            # Parse account values
            values = {}
            for v in self.ib.accountSummary():
                if v.account == account:
                    values[v.tag] = Decimal(str(v.value)) if v.value else Decimal("0")

            return IBKRAccountSummary(
                account_id=account,
                net_liquidation=values.get("NetLiquidation", Decimal("0")),
                total_cash=values.get("TotalCashValue", Decimal("0")),
                settled_cash=values.get("SettledCash", Decimal("0")),
                buying_power=values.get("BuyingPower", Decimal("0")),
                gross_position_value=values.get("GrossPositionValue", Decimal("0")),
                unrealized_pnl=values.get("UnrealizedPnL", Decimal("0")),
                realized_pnl=values.get("RealizedPnL", Decimal("0")),
                maintenance_margin=values.get("MaintMarginReq", Decimal("0")),
                available_funds=values.get("AvailableFunds", Decimal("0")),
                excess_liquidity=values.get("ExcessLiquidity", Decimal("0")),
                cushion=values.get("Cushion", Decimal("0")),
            )

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None

    def _get_simulated_account_summary(self) -> IBKRAccountSummary:
        """Get simulated account summary."""
        # Based on simulated positions
        positions = self._get_simulated_positions()
        total_value = sum(p.market_value for p in positions)
        unrealized = sum(p.unrealized_pnl for p in positions)
        realized = sum(p.realized_pnl for p in positions)
        cash = Decimal("500000")  # Simulated cash

        net_liq = total_value + cash

        return IBKRAccountSummary(
            account_id="PAPER",
            net_liquidation=net_liq,
            total_cash=cash,
            settled_cash=cash,
            buying_power=cash * 4,  # 4x leverage
            gross_position_value=total_value,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            maintenance_margin=total_value * Decimal("0.25"),
            available_funds=cash,
            excess_liquidity=cash * Decimal("0.8"),
            cushion=Decimal("0.35"),
        )

    # =========================================================================
    # EXECUTIONS / TRADE HISTORY
    # =========================================================================

    def get_executions(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        account: str = None
    ) -> List[IBKRExecution]:
        """
        Get trade executions from IBKR.

        Args:
            start_date: Start date filter
            end_date: End date filter
            account: Specific account

        Returns:
            List of executions
        """
        if not self.ensure_connected():
            return []

        if self.simulation_mode:
            return self._get_simulated_executions(start_date, end_date)

        return self._get_live_executions(start_date, end_date, account)

    def _get_live_executions(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        account: str = None
    ) -> List[IBKRExecution]:
        """Get executions from live IBKR."""
        executions = []

        try:
            fills = self.ib.fills()

            for fill in fills:
                exec_time = fill.time

                # Date filter
                if start_date and exec_time < start_date:
                    continue
                if end_date and exec_time > end_date:
                    continue

                # Account filter
                if account and fill.execution.acctNumber != account:
                    continue

                exec_record = IBKRExecution(
                    exec_id=fill.execution.execId,
                    symbol=fill.contract.symbol,
                    side=fill.execution.side,
                    quantity=Decimal(str(fill.execution.shares)),
                    price=Decimal(str(fill.execution.price)),
                    commission=Decimal(str(fill.commissionReport.commission or 0)),
                    account=fill.execution.acctNumber,
                    timestamp=exec_time,
                    order_id=fill.execution.orderId,
                    exchange=fill.execution.exchange,
                )
                executions.append(exec_record)

        except Exception as e:
            logger.error(f"Error getting executions: {e}")

        return executions

    def _get_simulated_executions(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[IBKRExecution]:
        """Get simulated executions for training."""
        now = datetime.now()

        # Sample historical trades
        return [
            IBKRExecution(
                exec_id="SIM001",
                symbol="AAPL",
                side="BOT",
                quantity=Decimal("500"),
                price=Decimal("175.50"),
                commission=Decimal("1.00"),
                account="PAPER",
                timestamp=now - timedelta(days=30),
                order_id=1001,
                exchange="SMART",
            ),
            IBKRExecution(
                exec_id="SIM002",
                symbol="MSFT",
                side="BOT",
                quantity=Decimal("300"),
                price=Decimal("380.00"),
                commission=Decimal("1.00"),
                account="PAPER",
                timestamp=now - timedelta(days=25),
                order_id=1002,
                exchange="SMART",
            ),
            IBKRExecution(
                exec_id="SIM003",
                symbol="NVDA",
                side="BOT",
                quantity=Decimal("200"),
                price=Decimal("450.00"),
                commission=Decimal("1.00"),
                account="PAPER",
                timestamp=now - timedelta(days=20),
                order_id=1003,
                exchange="SMART",
            ),
            IBKRExecution(
                exec_id="SIM004",
                symbol="SPY",
                side="BOT",
                quantity=Decimal("1000"),
                price=Decimal("480.00"),
                commission=Decimal("1.00"),
                account="PAPER",
                timestamp=now - timedelta(days=15),
                order_id=1004,
                exchange="SMART",
            ),
            IBKRExecution(
                exec_id="SIM005",
                symbol="AAPL",
                side="SLD",
                quantity=Decimal("100"),
                price=Decimal("190.00"),
                commission=Decimal("1.00"),
                account="PAPER",
                timestamp=now - timedelta(days=5),
                order_id=1005,
                exchange="SMART",
            ),
        ]

    # =========================================================================
    # NAV CALCULATION HELPERS
    # =========================================================================

    def get_nav_components(self) -> Dict[str, Any]:
        """
        Get all components needed for NAV calculation.

        Returns dict with:
        - positions: List of positions
        - cash: Total cash balance
        - total_market_value: Sum of position values
        - unrealized_pnl: Total unrealized P&L
        - realized_pnl: Total realized P&L
        - net_liquidation: Net liquidation value (NAV proxy)
        """
        positions = self.get_positions()
        account = self.get_account_summary()

        if not account:
            # Fallback calculation from positions
            total_value = sum(p.market_value for p in positions)
            unrealized = sum(p.unrealized_pnl for p in positions)
            realized = sum(p.realized_pnl for p in positions)

            return {
                "positions": [p.to_dict() for p in positions],
                "position_count": len(positions),
                "cash": 0,
                "total_market_value": float(total_value),
                "unrealized_pnl": float(unrealized),
                "realized_pnl": float(realized),
                "net_liquidation": float(total_value),
                "source": "positions_only",
                "timestamp": datetime.now().isoformat(),
            }

        return {
            "positions": [p.to_dict() for p in positions],
            "position_count": len(positions),
            "cash": float(account.total_cash),
            "settled_cash": float(account.settled_cash),
            "total_market_value": float(account.gross_position_value),
            "unrealized_pnl": float(account.unrealized_pnl),
            "realized_pnl": float(account.realized_pnl),
            "net_liquidation": float(account.net_liquidation),
            "buying_power": float(account.buying_power),
            "margin_requirement": float(account.maintenance_margin),
            "excess_liquidity": float(account.excess_liquidity),
            "source": "ibkr_live" if not self.simulation_mode else "simulation",
            "timestamp": datetime.now().isoformat(),
        }

    def get_pnl_summary(self) -> Dict[str, Any]:
        """
        Get P&L summary for reporting.

        Used by SANTAS_HELPER for daily P&L and CPA for tax calculations.
        """
        positions = self.get_positions()

        # Group by asset class
        by_class = {}
        for pos in positions:
            cls = pos.asset_class.value
            if cls not in by_class:
                by_class[cls] = {
                    "positions": [],
                    "market_value": Decimal("0"),
                    "unrealized_pnl": Decimal("0"),
                    "realized_pnl": Decimal("0"),
                }
            by_class[cls]["positions"].append(pos.symbol)
            by_class[cls]["market_value"] += pos.market_value
            by_class[cls]["unrealized_pnl"] += pos.unrealized_pnl
            by_class[cls]["realized_pnl"] += pos.realized_pnl

        # Convert to floats
        for cls in by_class:
            by_class[cls]["market_value"] = float(by_class[cls]["market_value"])
            by_class[cls]["unrealized_pnl"] = float(by_class[cls]["unrealized_pnl"])
            by_class[cls]["realized_pnl"] = float(by_class[cls]["realized_pnl"])

        total_unrealized = sum(p.unrealized_pnl for p in positions)
        total_realized = sum(p.realized_pnl for p in positions)

        return {
            "total_unrealized_pnl": float(total_unrealized),
            "total_realized_pnl": float(total_realized),
            "total_pnl": float(total_unrealized + total_realized),
            "by_asset_class": by_class,
            "position_count": len(positions),
            "timestamp": datetime.now().isoformat(),
        }

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_ttl

    def invalidate_cache(self) -> None:
        """Invalidate the position cache."""
        self._position_cache.clear()
        self._cache_timestamp = None
        logger.debug("IBKR cache invalidated")

    def refresh_data(self) -> Dict[str, Any]:
        """Force refresh all data from IBKR."""
        self.invalidate_cache()
        return self.get_nav_components()


# Singleton instance
_ibkr_data_service: Optional[IBKRDataService] = None


def get_ibkr_data_service(simulation_mode: bool = None) -> IBKRDataService:
    """Get singleton IBKR data service instance."""
    global _ibkr_data_service
    if _ibkr_data_service is None:
        _ibkr_data_service = IBKRDataService(simulation_mode)
    return _ibkr_data_service


# ============================================================================
# CONVENIENCE FUNCTIONS FOR OPERATIONS AGENTS
# ============================================================================

def get_portfolio_for_nav() -> Dict[str, Any]:
    """
    Get portfolio data formatted for NAV calculation.

    Used by SANTAS_HELPER.generate_nav_pack()
    """
    service = get_ibkr_data_service()
    return service.get_nav_components()


def get_pnl_for_reporting() -> Dict[str, Any]:
    """
    Get P&L data formatted for reporting.

    Used by SANTAS_HELPER.calculate_pnl() and CPA tax calculations.
    """
    service = get_ibkr_data_service()
    return service.get_pnl_summary()


def get_executions_for_tax(year: int = None) -> List[Dict[str, Any]]:
    """
    Get executions for tax reporting.

    Used by CPA for K-1 preparation and wash sale tracking.
    """
    year = year or datetime.now().year
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59, 59)

    service = get_ibkr_data_service()
    executions = service.get_executions(start, end)

    return [e.to_dict() for e in executions]


def get_positions_for_audit() -> List[Dict[str, Any]]:
    """
    Get positions for audit verification.

    Used by CPA for audit PBC preparation.
    """
    service = get_ibkr_data_service()
    positions = service.get_positions()
    return [p.to_dict() for p in positions]


if __name__ == "__main__":
    # Demo / Test
    import json

    print("=" * 60)
    print("IBKR DATA SERVICE - Demo")
    print("=" * 60)

    service = get_ibkr_data_service(simulation_mode=True)

    print("\n1. NAV Components:")
    nav = service.get_nav_components()
    print(json.dumps(nav, indent=2, default=str))

    print("\n2. P&L Summary:")
    pnl = service.get_pnl_summary()
    print(json.dumps(pnl, indent=2, default=str))

    print("\n3. Account Summary:")
    acct = service.get_account_summary()
    if acct:
        print(json.dumps(acct.to_dict(), indent=2, default=str))

    print("\n4. Positions:")
    for pos in service.get_positions():
        print(f"  {pos.symbol}: {pos.quantity} @ ${pos.market_price} = ${pos.market_value}")

