"""
PORTFOLIO INGESTION MODULE
==========================
Alpha Loop Capital - Trade History Analysis

Purpose: Ingest historical trades and portfolio data from Alpha Loop Capital
         to analyze past performance and improve future trading decisions.

Supports multiple input formats:
- CSV files (from brokerage exports)
- Excel files
- JSON files
- Direct IBKR API integration
- Google Sheets

Author: Tom Hogan
Version: 1.0
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Types of trades"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_OPEN = "buy_to_open"      # Options
    SELL_TO_OPEN = "sell_to_open"    # Options
    BUY_TO_CLOSE = "buy_to_close"    # Options
    SELL_TO_CLOSE = "sell_to_close"  # Options
    DIVIDEND = "dividend"
    SPLIT = "split"
    TRANSFER = "transfer"


class AssetClass(Enum):
    """Asset classes"""
    EQUITY = "equity"
    OPTION = "option"
    ETF = "etf"
    CRYPTO = "crypto"
    FIXED_INCOME = "fixed_income"
    CASH = "cash"


@dataclass
class Trade:
    """Single trade record"""
    trade_id: str
    date: str
    ticker: str
    trade_type: TradeType
    asset_class: AssetClass
    quantity: float
    price: float
    total_value: float
    fees: float = 0.0
    
    # Options specific
    option_type: Optional[str] = None  # "call" or "put"
    strike: Optional[float] = None
    expiration: Optional[str] = None
    
    # Metadata
    account: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Calculated fields
    proceeds: float = 0.0
    cost_basis: float = 0.0
    realized_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "date": self.date,
            "ticker": self.ticker,
            "trade_type": self.trade_type.value,
            "asset_class": self.asset_class.value,
            "quantity": self.quantity,
            "price": self.price,
            "total_value": self.total_value,
            "fees": self.fees,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration,
            "account": self.account,
            "notes": self.notes,
            "tags": self.tags,
            "realized_pnl": self.realized_pnl
        }


@dataclass
class Position:
    """Current position in a security"""
    ticker: str
    asset_class: AssetClass
    quantity: float
    cost_basis: float
    avg_cost: float
    current_price: float = 0.0
    
    # Options specific
    option_type: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[str] = None
    
    # Calculated
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight_pct: float = 0.0
    
    # Tax lots
    tax_lots: List[Dict] = field(default_factory=list)
    
    def update_market_value(self, price: float, portfolio_value: float = 0.0) -> None:
        """Update market value and P&L"""
        self.current_price = price
        self.market_value = self.quantity * price * (100 if self.asset_class == AssetClass.OPTION else 1)
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.unrealized_pnl_pct = (self.unrealized_pnl / self.cost_basis * 100) if self.cost_basis > 0 else 0
        if portfolio_value > 0:
            self.weight_pct = (self.market_value / portfolio_value) * 100


@dataclass
class Portfolio:
    """Complete portfolio state"""
    name: str
    as_of_date: str
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    total_value: float = 0.0
    
    # Performance metrics
    total_cost_basis: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    
    def add_position(self, position: Position) -> None:
        """Add or update a position"""
        key = f"{position.ticker}_{position.option_type}_{position.strike}_{position.expiration}" if position.asset_class == AssetClass.OPTION else position.ticker
        self.positions[key] = position
        self._recalculate_totals()
    
    def _recalculate_totals(self) -> None:
        """Recalculate portfolio totals"""
        self.total_cost_basis = sum(p.cost_basis for p in self.positions.values())
        self.total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.total_value = sum(p.market_value for p in self.positions.values()) + self.cash


class PortfolioIngestion:
    """
    PORTFOLIO INGESTION ENGINE
    
    Loads historical trade data from various sources and
    provides analysis tools for performance improvement.
    
    Usage:
        ingestor = PortfolioIngestion()
        
        # Load from CSV
        trades = ingestor.load_from_csv("trades.csv")
        
        # Build portfolio from trades
        portfolio = ingestor.build_portfolio_from_trades(trades)
        
        # Analyze performance
        analysis = ingestor.analyze_performance(trades)
    """
    
    def __init__(self, data_dir: str = "data/portfolio_history"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades: List[Trade] = []
        self.portfolio: Optional[Portfolio] = None
        
        # Column mappings for different brokers
        self.column_mappings = {
            "ibkr": {
                "date": ["TradeDate", "Date/Time", "Date"],
                "ticker": ["Symbol", "Ticker", "SecurityDescription"],
                "quantity": ["Quantity", "Shares", "Amount"],
                "price": ["TradePrice", "Price", "T. Price"],
                "type": ["Buy/Sell", "Code", "Action"],
                "fees": ["CommFee", "Commission", "Fees"]
            },
            "schwab": {
                "date": ["Date", "Trade Date"],
                "ticker": ["Symbol", "Security"],
                "quantity": ["Quantity", "Qty"],
                "price": ["Price"],
                "type": ["Action"],
                "fees": ["Fees & Comm"]
            },
            "generic": {
                "date": ["date", "trade_date", "Date"],
                "ticker": ["ticker", "symbol", "Symbol"],
                "quantity": ["quantity", "shares", "Quantity"],
                "price": ["price", "Price"],
                "type": ["type", "action", "Type", "Action"],
                "fees": ["fees", "commission", "Fees"]
            }
        }
    
    def load_from_csv(
        self,
        filepath: str,
        broker: str = "generic",
        account: str = "default"
    ) -> List[Trade]:
        """
        Load trades from a CSV file.
        
        Parameters:
            filepath: Path to CSV file
            broker: Broker name for column mapping ("ibkr", "schwab", "generic")
            account: Account identifier
            
        Returns:
            List of Trade objects
        """
        logger.info(f"Loading trades from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
        
        # Normalize columns
        df = self._normalize_columns(df, broker)
        
        trades = []
        for idx, row in df.iterrows():
            try:
                trade = self._parse_row_to_trade(row, idx, account)
                if trade:
                    trades.append(trade)
            except Exception as e:
                logger.warning(f"Error parsing row {idx}: {e}")
        
        self.trades.extend(trades)
        logger.info(f"Loaded {len(trades)} trades from {filepath}")
        
        return trades
    
    def load_from_excel(
        self,
        filepath: str,
        sheet_name: str = None,
        broker: str = "generic",
        account: str = "default"
    ) -> List[Trade]:
        """Load trades from Excel file"""
        logger.info(f"Loading trades from Excel: {filepath}")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise
        
        df = self._normalize_columns(df, broker)
        
        trades = []
        for idx, row in df.iterrows():
            try:
                trade = self._parse_row_to_trade(row, idx, account)
                if trade:
                    trades.append(trade)
            except Exception as e:
                logger.warning(f"Error parsing row {idx}: {e}")
        
        self.trades.extend(trades)
        logger.info(f"Loaded {len(trades)} trades")
        
        return trades
    
    def load_from_json(self, filepath: str) -> List[Trade]:
        """Load trades from JSON file"""
        logger.info(f"Loading trades from JSON: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        trades = []
        for item in data.get("trades", data):
            trade = Trade(
                trade_id=item.get("trade_id", f"json_{len(trades)}"),
                date=item["date"],
                ticker=item["ticker"],
                trade_type=TradeType(item.get("trade_type", "buy")),
                asset_class=AssetClass(item.get("asset_class", "equity")),
                quantity=float(item["quantity"]),
                price=float(item["price"]),
                total_value=float(item.get("total_value", item["quantity"] * item["price"])),
                fees=float(item.get("fees", 0)),
                option_type=item.get("option_type"),
                strike=item.get("strike"),
                expiration=item.get("expiration"),
                account=item.get("account", ""),
                notes=item.get("notes", ""),
                tags=item.get("tags", [])
            )
            trades.append(trade)
        
        self.trades.extend(trades)
        logger.info(f"Loaded {len(trades)} trades from JSON")
        
        return trades
    
    def _normalize_columns(self, df: pd.DataFrame, broker: str) -> pd.DataFrame:
        """Normalize column names based on broker mapping"""
        mapping = self.column_mappings.get(broker, self.column_mappings["generic"])
        
        # Create standardized column mapping
        column_map = {}
        for standard_col, possible_names in mapping.items():
            for name in possible_names:
                if name in df.columns:
                    column_map[name] = standard_col
                    break
        
        df = df.rename(columns=column_map)
        return df
    
    def _parse_row_to_trade(
        self,
        row: pd.Series,
        idx: int,
        account: str
    ) -> Optional[Trade]:
        """Parse a dataframe row into a Trade object"""
        
        # Skip empty rows
        if pd.isna(row.get("ticker")):
            return None
        
        # Determine trade type
        trade_type_str = str(row.get("type", "buy")).lower()
        if "buy" in trade_type_str:
            trade_type = TradeType.BUY
        elif "sell" in trade_type_str:
            trade_type = TradeType.SELL
        elif "div" in trade_type_str:
            trade_type = TradeType.DIVIDEND
        else:
            trade_type = TradeType.BUY
        
        # Determine asset class
        ticker = str(row.get("ticker", "")).upper()
        if len(ticker) > 10 or row.get("option_type"):
            asset_class = AssetClass.OPTION
        elif ticker.endswith("USD") or ticker in ["BTC", "ETH"]:
            asset_class = AssetClass.CRYPTO
        else:
            asset_class = AssetClass.EQUITY
        
        quantity = abs(float(row.get("quantity", 0)))
        price = abs(float(row.get("price", 0)))
        fees = abs(float(row.get("fees", 0)))
        total_value = quantity * price
        
        # Handle negative quantities for sells
        if trade_type == TradeType.SELL:
            quantity = -quantity
        
        trade = Trade(
            trade_id=f"{account}_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            date=str(row.get("date", datetime.now().strftime("%Y-%m-%d"))),
            ticker=ticker,
            trade_type=trade_type,
            asset_class=asset_class,
            quantity=quantity,
            price=price,
            total_value=total_value,
            fees=fees,
            account=account,
            notes=str(row.get("notes", "")),
            option_type=row.get("option_type"),
            strike=row.get("strike"),
            expiration=row.get("expiration")
        )
        
        return trade
    
    def build_portfolio_from_trades(
        self,
        trades: List[Trade] = None,
        as_of_date: str = None
    ) -> Portfolio:
        """
        Build current portfolio state from trade history.
        
        Parameters:
            trades: List of trades (uses self.trades if None)
            as_of_date: Date to build portfolio as of (defaults to today)
            
        Returns:
            Portfolio object with current positions
        """
        trades = trades or self.trades
        as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
        
        # Filter trades up to as_of_date
        relevant_trades = [t for t in trades if t.date <= as_of_date]
        relevant_trades.sort(key=lambda x: x.date)
        
        portfolio = Portfolio(
            name="Alpha Loop Capital",
            as_of_date=as_of_date
        )
        
        # Track positions
        positions: Dict[str, Dict] = {}
        
        for trade in relevant_trades:
            key = trade.ticker
            if trade.asset_class == AssetClass.OPTION:
                key = f"{trade.ticker}_{trade.option_type}_{trade.strike}_{trade.expiration}"
            
            if key not in positions:
                positions[key] = {
                    "ticker": trade.ticker,
                    "asset_class": trade.asset_class,
                    "quantity": 0,
                    "cost_basis": 0,
                    "option_type": trade.option_type,
                    "strike": trade.strike,
                    "expiration": trade.expiration,
                    "tax_lots": []
                }
            
            pos = positions[key]
            
            if trade.trade_type in [TradeType.BUY, TradeType.BUY_TO_OPEN]:
                pos["quantity"] += abs(trade.quantity)
                pos["cost_basis"] += trade.total_value + trade.fees
                pos["tax_lots"].append({
                    "date": trade.date,
                    "quantity": abs(trade.quantity),
                    "cost_per_share": trade.price,
                    "total_cost": trade.total_value + trade.fees
                })
            elif trade.trade_type in [TradeType.SELL, TradeType.SELL_TO_CLOSE]:
                pos["quantity"] -= abs(trade.quantity)
                # FIFO cost basis reduction
                shares_to_remove = abs(trade.quantity)
                while shares_to_remove > 0 and pos["tax_lots"]:
                    lot = pos["tax_lots"][0]
                    if lot["quantity"] <= shares_to_remove:
                        shares_to_remove -= lot["quantity"]
                        pos["cost_basis"] -= lot["total_cost"]
                        pos["tax_lots"].pop(0)
                    else:
                        ratio = shares_to_remove / lot["quantity"]
                        pos["cost_basis"] -= lot["total_cost"] * ratio
                        lot["quantity"] -= shares_to_remove
                        lot["total_cost"] *= (1 - ratio)
                        shares_to_remove = 0
        
        # Create Position objects for non-zero positions
        for key, pos in positions.items():
            if abs(pos["quantity"]) > 0.001:
                position = Position(
                    ticker=pos["ticker"],
                    asset_class=pos["asset_class"],
                    quantity=pos["quantity"],
                    cost_basis=pos["cost_basis"],
                    avg_cost=pos["cost_basis"] / pos["quantity"] if pos["quantity"] > 0 else 0,
                    option_type=pos["option_type"],
                    strike=pos["strike"],
                    expiration=pos["expiration"],
                    tax_lots=pos["tax_lots"]
                )
                portfolio.add_position(position)
        
        self.portfolio = portfolio
        logger.info(f"Built portfolio with {len(portfolio.positions)} positions")
        
        return portfolio
    
    def analyze_performance(
        self,
        trades: List[Trade] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """
        Analyze trading performance.
        
        Returns comprehensive performance metrics including:
        - Win rate
        - Average win/loss
        - Sharpe ratio estimate
        - Best/worst trades
        - Performance by ticker
        - Performance by asset class
        """
        trades = trades or self.trades
        
        if start_date:
            trades = [t for t in trades if t.date >= start_date]
        if end_date:
            trades = [t for t in trades if t.date <= end_date]
        
        if not trades:
            return {"error": "No trades to analyze"}
        
        # Group trades by ticker for P&L calculation
        ticker_trades: Dict[str, List[Trade]] = {}
        for trade in trades:
            if trade.ticker not in ticker_trades:
                ticker_trades[trade.ticker] = []
            ticker_trades[trade.ticker].append(trade)
        
        # Calculate realized P&L per ticker
        realized_pnls = []
        ticker_stats = {}
        
        for ticker, ticker_trades_list in ticker_trades.items():
            buys = [t for t in ticker_trades_list if t.trade_type in [TradeType.BUY, TradeType.BUY_TO_OPEN]]
            sells = [t for t in ticker_trades_list if t.trade_type in [TradeType.SELL, TradeType.SELL_TO_CLOSE]]
            
            total_buy_value = sum(t.total_value + t.fees for t in buys)
            total_buy_qty = sum(abs(t.quantity) for t in buys)
            total_sell_value = sum(t.total_value - t.fees for t in sells)
            total_sell_qty = sum(abs(t.quantity) for t in sells)
            
            # Only count closed positions for realized P&L
            closed_qty = min(total_buy_qty, total_sell_qty)
            if closed_qty > 0 and total_buy_qty > 0:
                avg_buy = total_buy_value / total_buy_qty
                avg_sell = total_sell_value / total_sell_qty if total_sell_qty > 0 else 0
                realized_pnl = (avg_sell - avg_buy) * closed_qty
                realized_pnls.append({
                    "ticker": ticker,
                    "pnl": realized_pnl,
                    "pnl_pct": (realized_pnl / (avg_buy * closed_qty)) * 100 if avg_buy > 0 else 0,
                    "trade_count": len(ticker_trades_list)
                })
                
                ticker_stats[ticker] = {
                    "total_buys": len(buys),
                    "total_sells": len(sells),
                    "realized_pnl": realized_pnl,
                    "avg_buy_price": avg_buy / closed_qty if closed_qty > 0 else 0,
                    "avg_sell_price": avg_sell / total_sell_qty if total_sell_qty > 0 else 0
                }
        
        # Calculate aggregate stats
        winning_trades = [p for p in realized_pnls if p["pnl"] > 0]
        losing_trades = [p for p in realized_pnls if p["pnl"] < 0]
        
        total_pnl = sum(p["pnl"] for p in realized_pnls)
        win_rate = len(winning_trades) / len(realized_pnls) * 100 if realized_pnls else 0
        avg_win = np.mean([p["pnl"] for p in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([p["pnl"] for p in losing_trades]) if losing_trades else 0
        
        # Sort for best/worst
        realized_pnls.sort(key=lambda x: x["pnl"], reverse=True)
        
        return {
            "period": {
                "start": min(t.date for t in trades),
                "end": max(t.date for t in trades),
                "trade_count": len(trades)
            },
            "performance": {
                "total_realized_pnl": round(total_pnl, 2),
                "win_rate_pct": round(win_rate, 1),
                "average_win": round(avg_win, 2),
                "average_loss": round(avg_loss, 2),
                "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
                "winning_tickers": len(winning_trades),
                "losing_tickers": len(losing_trades)
            },
            "best_trades": realized_pnls[:5],
            "worst_trades": realized_pnls[-5:][::-1],
            "by_ticker": ticker_stats,
            "recommendations": self._generate_recommendations(realized_pnls, ticker_stats)
        }
    
    def _generate_recommendations(
        self,
        realized_pnls: List[Dict],
        ticker_stats: Dict
    ) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []
        
        # Win rate check
        wins = [p for p in realized_pnls if p["pnl"] > 0]
        win_rate = len(wins) / len(realized_pnls) * 100 if realized_pnls else 0
        
        if win_rate < 40:
            recommendations.append(
                "Win rate below 40% - Consider tightening entry criteria "
                "and waiting for better setups"
            )
        
        # Average win vs loss
        avg_win = np.mean([p["pnl"] for p in wins]) if wins else 0
        losses = [p for p in realized_pnls if p["pnl"] < 0]
        avg_loss = np.mean([abs(p["pnl"]) for p in losses]) if losses else 0
        
        if avg_loss > avg_win and avg_loss > 0:
            recommendations.append(
                f"Average loss (${avg_loss:.0f}) exceeds average win (${avg_win:.0f}) - "
                "Consider tighter stop losses"
            )
        
        # Concentration check
        if realized_pnls:
            top_ticker = realized_pnls[0]
            if top_ticker["pnl"] > sum(p["pnl"] for p in realized_pnls) * 0.5:
                recommendations.append(
                    f"Performance heavily dependent on {top_ticker['ticker']} - "
                    "Consider diversifying winners"
                )
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Performance looks solid! Keep following your process.")
        
        return recommendations
    
    def export_portfolio(self, filepath: str, format: str = "json") -> None:
        """Export current portfolio to file"""
        if not self.portfolio:
            raise ValueError("No portfolio to export. Build portfolio first.")
        
        data = {
            "name": self.portfolio.name,
            "as_of_date": self.portfolio.as_of_date,
            "total_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "positions": [
                {
                    "ticker": p.ticker,
                    "quantity": p.quantity,
                    "cost_basis": p.cost_basis,
                    "avg_cost": p.avg_cost,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "weight_pct": p.weight_pct
                }
                for p in self.portfolio.positions.values()
            ]
        }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            df = pd.DataFrame(data["positions"])
            df.to_csv(filepath, index=False)
        
        logger.info(f"Exported portfolio to {filepath}")
    
    def generate_report(self) -> str:
        """Generate human-readable portfolio report"""
        if not self.portfolio:
            return "No portfolio loaded. Use load_from_csv() first."
        
        analysis = self.analyze_performance()
        
        lines = [
            "=" * 60,
            "ALPHA LOOP CAPITAL - PORTFOLIO REPORT",
            f"As of: {self.portfolio.as_of_date}",
            "=" * 60,
            "",
            f"💰 TOTAL VALUE: ${self.portfolio.total_value:,.2f}",
            f"📊 POSITIONS: {len(self.portfolio.positions)}",
            f"💵 CASH: ${self.portfolio.cash:,.2f}",
            "",
            "📈 PERFORMANCE:",
            f"   Win Rate: {analysis['performance']['win_rate_pct']:.1f}%",
            f"   Total P&L: ${analysis['performance']['total_realized_pnl']:,.2f}",
            f"   Profit Factor: {analysis['performance']['profit_factor']:.2f}",
            "",
            "🏆 TOP POSITIONS:"
        ]
        
        positions = sorted(
            self.portfolio.positions.values(),
            key=lambda x: x.market_value,
            reverse=True
        )
        
        for pos in positions[:5]:
            pnl_emoji = "🟢" if pos.unrealized_pnl > 0 else "🔴"
            lines.append(
                f"   {pnl_emoji} {pos.ticker}: {pos.quantity:.0f} shares @ ${pos.avg_cost:.2f}"
            )
            lines.append(
                f"      Value: ${pos.market_value:,.0f} | P&L: ${pos.unrealized_pnl:,.0f} ({pos.unrealized_pnl_pct:.1f}%)"
            )
        
        lines.extend([
            "",
            "💡 RECOMMENDATIONS:"
        ])
        
        for rec in analysis.get("recommendations", []):
            lines.append(f"   • {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_alc_portfolio(filepath: str) -> Tuple[Portfolio, Dict]:
    """
    Quick function to load Alpha Loop Capital portfolio and analyze it.
    
    Usage:
        portfolio, analysis = load_alc_portfolio("my_trades.csv")
    """
    ingestor = PortfolioIngestion()
    ingestor.load_from_csv(filepath)
    portfolio = ingestor.build_portfolio_from_trades()
    analysis = ingestor.analyze_performance()
    
    print(ingestor.generate_report())
    
    return portfolio, analysis


if __name__ == "__main__":
    # Demo with sample data
    ingestor = PortfolioIngestion()
    
    # Create sample trades for demo
    sample_trades = [
        Trade(
            trade_id="1", date="2024-01-15", ticker="NVDA",
            trade_type=TradeType.BUY, asset_class=AssetClass.EQUITY,
            quantity=100, price=450.0, total_value=45000, fees=5
        ),
        Trade(
            trade_id="2", date="2024-03-20", ticker="NVDA",
            trade_type=TradeType.SELL, asset_class=AssetClass.EQUITY,
            quantity=-50, price=550.0, total_value=27500, fees=5
        ),
        Trade(
            trade_id="3", date="2024-02-01", ticker="CCJ",
            trade_type=TradeType.BUY, asset_class=AssetClass.EQUITY,
            quantity=500, price=35.0, total_value=17500, fees=5
        ),
    ]
    
    ingestor.trades = sample_trades
    portfolio = ingestor.build_portfolio_from_trades()
    
    # Update with current prices (in production, fetch from API)
    for ticker, pos in portfolio.positions.items():
        if ticker == "NVDA":
            pos.update_market_value(140.0, portfolio.total_value)
        elif ticker == "CCJ":
            pos.update_market_value(55.0, portfolio.total_value)
    
    portfolio._recalculate_totals()
    
    print(ingestor.generate_report())

