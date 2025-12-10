
from loguru import logger
from sqlalchemy import Column, DateTime, Float, MetaData, String, Table

from src.database.connection import get_engine


def create_tables():
    engine = get_engine()
    metadata = MetaData()

    # 1. Price Bars (OHLCV)
    price_bars = Table(
        "price_bars", metadata,
        Column("symbol", String(20), primary_key=True),
        Column("timestamp", DateTime, primary_key=True),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Float),
        Column("source", String(20)),
    )

    # 2. Fundamentals (Valuation)
    fundamentals = Table(
        "fundamentals", metadata,
        Column("symbol", String(20), primary_key=True),
        Column("timestamp", DateTime, primary_key=True),
        Column("ev_ebitda", Float),
        Column("fcf_yield", Float),
        Column("peg_ratio", Float),
        Column("roic", Float),
        Column("gross_margin", Float),
    )

    # 3. Risk Metrics (Greeks/VaR snapshot)
    risk_metrics = Table(
        "risk_metrics", metadata,
        Column("symbol", String(20), primary_key=True),
        Column("timestamp", DateTime, primary_key=True),
        Column("var_99", Float),
        Column("cvar_99", Float),
        Column("convexity", Float),
        Column("implied_vol", Float),
    )

    logger.info("Creating SQL tables if they don't exist...")
    metadata.create_all(engine)
    logger.success("Database schema initialized.")

if __name__ == "__main__":
    create_tables()


