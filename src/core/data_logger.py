"""
Centralized Data Logger - Production Ready
Author: Tom Hogan | Alpha Loop Capital, LLC

Logs all trading activity to database, files, and cloud for ML training.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """Agent decision record"""
    timestamp: str
    agent_name: str
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    signal_strength: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class Trade:
    """Trade execution record"""
    timestamp: str
    symbol: str
    side: str  # buy, sell
    quantity: float
    price: float
    commission: float
    slippage: float
    agent_name: str
    strategy: str
    pnl: Optional[float] = None


@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot"""
    timestamp: str
    total_value: float
    cash: float
    positions: Dict[str, Dict[str, float]]  # symbol -> {shares, value, pnl}
    daily_pnl: float
    total_pnl: float


class DataLogger:
    """
    Production-ready data logger.

    Logs to:
    - PostgreSQL database (primary)
    - JSON files (backup)
    - Stdout (monitoring)
    """

    def __init__(self):
        self.db_conn = None
        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._connect_db()

        # Initialize tables
        self._init_tables()

        logger.info("DataLogger initialized")

    def _connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                logger.warning("DATABASE_URL not set - using local file logging only")
                return

            self.db_conn = psycopg2.connect(database_url)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.warning("Falling back to file-only logging")

    def _init_tables(self):
        """Create database tables if they don't exist"""
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()

            # Agent decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    agent_name VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    action VARCHAR(20) NOT NULL,
                    confidence FLOAT NOT NULL,
                    signal_strength FLOAT NOT NULL,
                    reasoning TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON agent_decisions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent_name);
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON agent_decisions(symbol);
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity FLOAT NOT NULL,
                    price FLOAT NOT NULL,
                    commission FLOAT NOT NULL,
                    slippage FLOAT NOT NULL,
                    agent_name VARCHAR(100),
                    strategy VARCHAR(100),
                    pnl FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            """)

            # Portfolio snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    total_value FLOAT NOT NULL,
                    cash FLOAT NOT NULL,
                    positions JSONB NOT NULL,
                    daily_pnl FLOAT NOT NULL,
                    total_pnl FLOAT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
            """)

            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume BIGINT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_data(timestamp);
                CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_data(symbol);
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    agent_name VARCHAR(100),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
            """)

            self.db_conn.commit()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            if self.db_conn:
                self.db_conn.rollback()

    def log_agent_decision(self, decision: AgentDecision):
        """Log agent decision"""
        # Log to database
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    INSERT INTO agent_decisions
                    (timestamp, agent_name, symbol, action, confidence, signal_strength, reasoning, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    decision.timestamp,
                    decision.agent_name,
                    decision.symbol,
                    decision.action,
                    decision.confidence,
                    decision.signal_strength,
                    decision.reasoning,
                    json.dumps(decision.metadata)
                ))
                self.db_conn.commit()
            except Exception as e:
                logger.error(f"Failed to log decision to database: {e}")
                self.db_conn.rollback()

        # Log to file
        self._log_to_file("agent_decisions", asdict(decision))

        # Log to stdout
        logger.info(f"[DECISION] {decision.agent_name} -> {decision.action} {decision.symbol} (conf: {decision.confidence:.2f})")

    def log_trade(self, trade: Trade):
        """Log trade execution"""
        # Log to database
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    INSERT INTO trades
                    (timestamp, symbol, side, quantity, price, commission, slippage, agent_name, strategy, pnl)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    trade.timestamp,
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.commission,
                    trade.slippage,
                    trade.agent_name,
                    trade.strategy,
                    trade.pnl
                ))
                self.db_conn.commit()
            except Exception as e:
                logger.error(f"Failed to log trade to database: {e}")
                self.db_conn.rollback()

        # Log to file
        self._log_to_file("trades", asdict(trade))

        # Log to stdout
        logger.info(f"[TRADE] {trade.side.upper()} {trade.quantity} {trade.symbol} @ ${trade.price:.2f}")

    def log_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """Log portfolio snapshot"""
        # Log to database
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    INSERT INTO portfolio_snapshots
                    (timestamp, total_value, cash, positions, daily_pnl, total_pnl)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    snapshot.timestamp,
                    snapshot.total_value,
                    snapshot.cash,
                    json.dumps(snapshot.positions),
                    snapshot.daily_pnl,
                    snapshot.total_pnl
                ))
                self.db_conn.commit()
            except Exception as e:
                logger.error(f"Failed to log snapshot to database: {e}")
                self.db_conn.rollback()

        # Log to file
        self._log_to_file("portfolio_snapshots", asdict(snapshot))

        # Log to stdout
        logger.info(f"[PORTFOLIO] Value: ${snapshot.total_value:,.2f} | P&L: ${snapshot.daily_pnl:,.2f}")

    def log_metric(self, metric_name: str, value: float, agent_name: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log performance metric"""
        timestamp = datetime.now().isoformat()

        # Log to database
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_metrics
                    (timestamp, metric_name, metric_value, agent_name, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    timestamp,
                    metric_name,
                    value,
                    agent_name,
                    json.dumps(metadata) if metadata else None
                ))
                self.db_conn.commit()
            except Exception as e:
                logger.error(f"Failed to log metric to database: {e}")
                self.db_conn.rollback()

        # Log to file
        self._log_to_file("metrics", {
            "timestamp": timestamp,
            "metric_name": metric_name,
            "value": value,
            "agent_name": agent_name,
            "metadata": metadata
        })

    def _log_to_file(self, log_type: str, data: Dict):
        """Log to JSON file (backup)"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"{log_type}_{today}.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def get_recent_decisions(self, agent_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent agent decisions"""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            if agent_name:
                cursor.execute("""
                    SELECT * FROM agent_decisions
                    WHERE agent_name = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (agent_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM agent_decisions
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get decisions: {e}")
            return []

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    def __del__(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()


# Global singleton
_data_logger = None

def get_data_logger() -> DataLogger:
    """Get global data logger instance"""
    global _data_logger
    if _data_logger is None:
        _data_logger = DataLogger()
    return _data_logger
