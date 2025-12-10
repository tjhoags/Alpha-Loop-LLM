"""================================================================================
FULL UNIVERSE DATA INGESTION - All US Stocks and Options
================================================================================
Alpha Loop Capital, LLC

Ingests comprehensive market data:
1. ALL US Stocks (via Polygon ticker list)
2. Options Chains (for liquid underlyings)
3. ETFs (equity, bond, commodity, volatility)
4. Crypto (BTC, ETH, major coins)
5. Macro Indicators (FRED)

Data Sources:
- Polygon.io: Stocks, ETFs, Options, Crypto
- Alpha Vantage: Backup stock data
- FRED: Macro indicators
- Coinbase: Crypto prices

Usage:
    python scripts/ingest_full_universe.py --all
    python scripts/ingest_full_universe.py --stocks --options
    python scripts/ingest_full_universe.py --continuous --interval 60
================================================================================
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
from loguru import logger

from src.config.settings import get_settings
from src.database.connection import get_engine

# Rate limiting
POLYGON_RATE_LIMIT = 5  # calls per second for free tier
ALPHA_VANTAGE_DELAY = 12.5  # 5 calls/minute

# =============================================================================
# POLYGON API FUNCTIONS
# =============================================================================

def get_polygon_tickers(market: str = "stocks", active: bool = True, limit: int = 1000) -> List[Dict]:
    """Get all tickers from Polygon."""
    settings = get_settings()

    if not settings.polygon_api_key:
        logger.warning("Polygon API key not configured")
        return []

    all_tickers = []
    url = "https://api.polygon.io/v3/reference/tickers"

    params = {
        "market": market,
        "active": str(active).lower(),
        "limit": limit,
        "apiKey": settings.polygon_api_key,
    }

    try:
        while url:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            all_tickers.extend(results)

            # Pagination
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": settings.polygon_api_key}
            else:
                url = None

            logger.info(f"Fetched {len(all_tickers)} {market} tickers so far...")
            time.sleep(0.2)  # Rate limiting

    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")

    logger.info(f"Total {market} tickers: {len(all_tickers)}")
    return all_tickers


def fetch_polygon_bars(
    symbol: str,
    timespan: str = "minute",
    multiplier: int = 1,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Fetch OHLCV bars from Polygon."""
    settings = get_settings()

    if not settings.polygon_api_key:
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{now.strftime('%Y-%m-%d')}"

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": settings.polygon_api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=45)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        records = []
        for row in results:
            ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc)
            records.append({
                "symbol": symbol,
                "timestamp": ts,
                "open": row["o"],
                "high": row["h"],
                "low": row["l"],
                "close": row["c"],
                "volume": row["v"],
                "vwap": row.get("vw"),
                "transactions": row.get("n"),
                "source": "polygon",
            })

        return pd.DataFrame(records)

    except Exception as e:
        logger.debug(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_polygon_options_chain(underlying: str, expiration_date: str = None) -> pd.DataFrame:
    """Fetch options chain from Polygon."""
    settings = get_settings()

    if not settings.polygon_api_key:
        return pd.DataFrame()

    url = "https://api.polygon.io/v3/reference/options/contracts"

    params = {
        "underlying_ticker": underlying,
        "limit": 1000,
        "apiKey": settings.polygon_api_key,
    }

    if expiration_date:
        params["expiration_date"] = expiration_date

    all_contracts = []

    try:
        while url:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            all_contracts.extend(results)

            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": settings.polygon_api_key}
            else:
                url = None

            time.sleep(0.2)

    except Exception as e:
        logger.debug(f"Failed to fetch options for {underlying}: {e}")

    if not all_contracts:
        return pd.DataFrame()

    records = []
    for contract in all_contracts:
        records.append({
            "ticker": contract.get("ticker"),
            "underlying": contract.get("underlying_ticker"),
            "contract_type": contract.get("contract_type"),
            "strike": contract.get("strike_price"),
            "expiration_date": contract.get("expiration_date"),
            "exercise_style": contract.get("exercise_style"),
            "primary_exchange": contract.get("primary_exchange"),
            "shares_per_contract": contract.get("shares_per_contract", 100),
        })

    return pd.DataFrame(records)


def fetch_polygon_option_quotes(option_ticker: str) -> Dict:
    """Fetch latest quote for an option contract."""
    settings = get_settings()

    if not settings.polygon_api_key:
        return {}

    url = f"https://api.polygon.io/v3/quotes/{option_ticker}"

    params = {
        "limit": 1,
        "apiKey": settings.polygon_api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if results:
            quote = results[0]
            return {
                "bid": quote.get("bid_price"),
                "ask": quote.get("ask_price"),
                "bid_size": quote.get("bid_size"),
                "ask_size": quote.get("ask_size"),
                "timestamp": quote.get("sip_timestamp"),
            }

    except Exception as e:
        logger.debug(f"Failed to fetch quote for {option_ticker}: {e}")

    return {}


# =============================================================================
# ALPHA VANTAGE BACKUP
# =============================================================================

def fetch_alpha_vantage_intraday(symbol: str, interval: str = "5min") -> pd.DataFrame:
    """Fetch intraday data from Alpha Vantage (backup source)."""
    settings = get_settings()

    if not settings.alpha_vantage_api_key:
        return pd.DataFrame()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full",
        "apikey": settings.alpha_vantage_api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            return pd.DataFrame()

        records = []
        for timestamp_str, values in data[time_series_key].items():
            records.append({
                "symbol": symbol,
                "timestamp": pd.to_datetime(timestamp_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
                "source": "alpha_vantage",
            })

        return pd.DataFrame(records)

    except Exception as e:
        logger.debug(f"Alpha Vantage failed for {symbol}: {e}")
        return pd.DataFrame()


# =============================================================================
# FRED MACRO DATA
# =============================================================================

def fetch_fred_series(series_id: str, start_date: str = None) -> pd.DataFrame:
    """Fetch macroeconomic series from FRED."""
    settings = get_settings()

    if not settings.fred_api_key:
        return pd.DataFrame()

    url = "https://api.stlouisfed.org/fred/series/observations"

    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])

        records = []
        for obs in observations:
            try:
                value = float(obs["value"]) if obs["value"] != "." else None
                records.append({
                    "series_id": series_id,
                    "date": pd.to_datetime(obs["date"]),
                    "value": value,
                })
            except:
                pass

        return pd.DataFrame(records)

    except Exception as e:
        logger.debug(f"FRED failed for {series_id}: {e}")
        return pd.DataFrame()


FRED_SERIES = {
    # Interest Rates
    "DGS1": "1-Year Treasury",
    "DGS2": "2-Year Treasury",
    "DGS5": "5-Year Treasury",
    "DGS10": "10-Year Treasury",
    "DGS30": "30-Year Treasury",
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",

    # Credit Spreads
    "BAMLH0A0HYM2": "High Yield Spread",
    "BAMLC0A4CBBB": "BBB Spread",

    # Economic Indicators
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
    "PCEPI": "PCE Inflation",
    "GDPC1": "Real GDP",
    "INDPRO": "Industrial Production",

    # Volatility
    "VIXCLS": "VIX Index",

    # Money Supply
    "M2SL": "M2 Money Supply",

    # Housing
    "HOUST": "Housing Starts",
    "CSUSHPINSA": "Case-Shiller Home Price",
}


# =============================================================================
# COINBASE CRYPTO
# =============================================================================

def fetch_coinbase_candles(symbol: str, granularity: int = 300, lookback_hours: int = 24) -> pd.DataFrame:
    """Fetch crypto candles from Coinbase."""
    settings = get_settings()

    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": granularity,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for candle in data:
            # Coinbase format: [time, low, high, open, close, volume]
            ts = datetime.fromtimestamp(candle[0], tz=timezone.utc)
            records.append({
                "symbol": symbol,
                "timestamp": ts,
                "open": candle[3],
                "high": candle[2],
                "low": candle[1],
                "close": candle[4],
                "volume": candle[5],
                "source": "coinbase",
            })

        return pd.DataFrame(records)

    except Exception as e:
        logger.debug(f"Coinbase failed for {symbol}: {e}")
        return pd.DataFrame()


CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "MATIC-USD",
    "LINK-USD", "UNI-USD", "AAVE-USD", "CRV-USD", "MKR-USD",
]


# =============================================================================
# MASSIVE S3 DATA (Polygon Flat Files)
# =============================================================================

def ingest_massive_data(
    data_type: str = "equity/minute",
    date_str: str = None,
    max_files: int = 10,
) -> int:
    """Ingest data from Massive S3 (Polygon flat files).

    Data types available:
    - equity/minute: 1-minute stock bars
    - equity/daily: Daily stock bars
    - options/trades: Options trades
    - crypto/trades: Crypto trades
    """
    logger.info("=" * 70)
    logger.info(f"INGESTING MASSIVE S3 DATA: {data_type}")
    logger.info("=" * 70)

    try:
        from src.data_ingestion.sources.massive import MassiveClient
        client = MassiveClient()

        if client.s3 is None:
            logger.warning("Massive S3 credentials not configured")
            return 0

        # List available files
        prefix = f"{data_type}/"
        if date_str:
            prefix = f"{data_type}/{date_str}"

        files = client.list_files(prefix)

        if not files:
            logger.warning(f"No files found for prefix: {prefix}")
            return 0

        logger.info(f"Found {len(files)} files, processing up to {max_files}...")

        # Process files (most recent first)
        files = sorted(files, reverse=True)[:max_files]

        all_data = []
        for file_key in files:
            df = client.fetch_file(file_key)
            if df is not None and not df.empty:
                # Standardize columns
                if "ticker" in df.columns and "symbol" not in df.columns:
                    df["symbol"] = df["ticker"]
                all_data.append(df)
                logger.info(f"  Loaded {len(df)} rows from {file_key}")

        total_rows = 0
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            total_rows = persist_price_bars(combined)

        logger.info(f"Massive ingestion complete: {total_rows} rows")
        return total_rows

    except Exception as e:
        logger.error(f"Massive ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


# =============================================================================
# ALPHA VANTAGE PREMIUM
# =============================================================================

def ingest_alpha_vantage_premium(
    symbols: List[str] = None,
    include_fundamentals: bool = True,
    include_forex: bool = True,
) -> int:
    """Ingest data from Alpha Vantage Premium tier."""
    logger.info("=" * 70)
    logger.info("INGESTING ALPHA VANTAGE PREMIUM DATA")
    logger.info("=" * 70)

    try:
        from src.data_ingestion.sources.alpha_vantage_premium import get_av_premium
        client = get_av_premium()

        if not client.api_key:
            logger.warning("Alpha Vantage API key not configured")
            return 0

        if not symbols:
            symbols = [
                # Large cap stocks
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                # Financials
                "JPM", "BAC", "GS",
                # ETFs
                "SPY", "QQQ", "IWM",
            ]

        all_data = []
        fundamentals_data = []

        for symbol in symbols:
            logger.info(f"Processing {symbol}...")

            # Intraday data
            df = client.fetch_stock_intraday(symbol, interval="5min", outputsize="full")
            if not df.empty:
                all_data.append(df)

            # Daily data
            daily_df = client.fetch_stock_daily(symbol, outputsize="compact")
            if not daily_df.empty:
                all_data.append(daily_df)

            # Fundamentals
            if include_fundamentals:
                fund_data = client.fetch_fundamental_data(symbol)
                if fund_data:
                    fundamentals_data.append(fund_data)

        # Forex pairs
        if include_forex:
            forex_pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD")]
            for from_curr, to_curr in forex_pairs:
                logger.info(f"Processing forex {from_curr}/{to_curr}...")
                df = client.fetch_forex(from_curr, to_curr, interval="5min")
                if not df.empty:
                    all_data.append(df)

        # Persist price data
        total_rows = 0
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            total_rows = persist_price_bars(combined)

        # Persist fundamentals
        if fundamentals_data:
            fund_df = pd.DataFrame(fundamentals_data)
            persist_price_bars(fund_df, table="fundamentals")
            logger.info(f"Stored {len(fund_df)} fundamental records")

        logger.info(f"Alpha Vantage Premium complete: {total_rows} price rows")
        return total_rows

    except Exception as e:
        logger.error(f"Alpha Vantage Premium failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


# =============================================================================
# IBKR DATA INTEGRATION
# =============================================================================

def ingest_ibkr_data(
    symbols: List[str] = None,
    include_options: bool = True,
    duration: str = "1 D",
    bar_size: str = "1 min",
) -> int:
    """Ingest data from Interactive Brokers TWS/Gateway.

    Requires IBKR TWS or Gateway to be running.
    Paper trading port: 7497
    Live trading port: 7496
    """
    logger.info("=" * 70)
    logger.info("INGESTING IBKR DATA")
    logger.info("=" * 70)

    try:
        from ib_insync import IB, Stock, Option, Index, Forex, Crypto, Future
        from src.config.settings import get_settings

        settings = get_settings()

        # Connect to IBKR
        ib = IB()

        try:
            ib.connect(
                host=settings.ibkr_host,
                port=settings.ibkr_port,
                clientId=settings.ibkr_client_id + 100,  # Different client ID for data
                timeout=30,
            )
            logger.info(f"Connected to IBKR on port {settings.ibkr_port}")
        except Exception as e:
            logger.warning(f"Could not connect to IBKR: {e}")
            logger.warning("Make sure TWS or IB Gateway is running")
            return 0

        if not symbols:
            symbols = [
                "SPY", "QQQ", "IWM", "DIA",
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "JPM", "BAC", "GS",
                "XLF", "XLE", "XLK",
            ]

        all_data = []
        options_data = []

        for symbol in symbols:
            logger.info(f"Fetching IBKR data for {symbol}...")

            try:
                # Create contract
                contract = Stock(symbol, "SMART", "USD")
                ib.qualifyContracts(contract)

                # Request historical data
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )

                if bars:
                    records = []
                    for bar in bars:
                        records.append({
                            "symbol": symbol,
                            "timestamp": bar.date,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                            "vwap": bar.average,
                            "bar_count": bar.barCount,
                            "source": "ibkr",
                        })
                    df = pd.DataFrame(records)
                    all_data.append(df)
                    logger.info(f"  {symbol}: {len(df)} bars")

                # Options chain
                if include_options:
                    try:
                        chains = ib.reqSecDefOptParams(
                            contract.symbol, "", contract.secType, contract.conId
                        )

                        if chains:
                            chain = chains[0]
                            # Get strikes near ATM
                            ticker = ib.reqTickers(contract)[0]
                            if ticker.marketPrice():
                                atm = ticker.marketPrice()
                                strikes = [s for s in chain.strikes if abs(s - atm) / atm < 0.1]

                                # Get nearest expiry
                                expirations = sorted(chain.expirations)[:2]

                                for expiry in expirations:
                                    for strike in strikes[:10]:  # Limit strikes
                                        for right in ["C", "P"]:
                                            opt = Option(symbol, expiry, strike, right, "SMART")
                                            ib.qualifyContracts(opt)

                                            ticker = ib.reqTickers(opt)
                                            if ticker:
                                                t = ticker[0]
                                                options_data.append({
                                                    "underlying": symbol,
                                                    "strike": strike,
                                                    "expiry": expiry,
                                                    "right": right,
                                                    "bid": t.bid,
                                                    "ask": t.ask,
                                                    "last": t.last,
                                                    "volume": t.volume,
                                                    "open_interest": t.openInterest,
                                                    "iv": t.impliedVolatility,
                                                    "delta": t.modelGreeks.delta if t.modelGreeks else None,
                                                    "gamma": t.modelGreeks.gamma if t.modelGreeks else None,
                                                    "theta": t.modelGreeks.theta if t.modelGreeks else None,
                                                    "vega": t.modelGreeks.vega if t.modelGreeks else None,
                                                    "source": "ibkr_options",
                                                })

                    except Exception as e:
                        logger.debug(f"Options chain failed for {symbol}: {e}")

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

            time.sleep(0.5)  # Rate limiting

        # Disconnect
        ib.disconnect()

        # Persist data
        total_rows = 0
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            total_rows = persist_price_bars(combined)

        if options_data:
            opt_df = pd.DataFrame(options_data)
            persist_price_bars(opt_df, table="ibkr_options")
            logger.info(f"Stored {len(opt_df)} options quotes")

        logger.info(f"IBKR ingestion complete: {total_rows} price rows")
        return total_rows

    except ImportError:
        logger.error("ib_insync not installed. Run: pip install ib_insync")
        return 0
    except Exception as e:
        logger.error(f"IBKR ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


# =============================================================================
# DATABASE PERSISTENCE
# =============================================================================

def persist_price_bars(df: pd.DataFrame, table: str = "price_bars") -> int:
    """Persist price bars to database."""
    if df is None or df.empty:
        return 0

    try:
        engine = get_engine()

        # Clean up DataFrame
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        with engine.begin() as conn:
            df.to_sql(table, conn, if_exists="append", index=False, chunksize=1000)

        logger.info(f"Persisted {len(df)} rows to {table}")
        return len(df)

    except Exception as e:
        logger.error(f"Failed to persist to {table}: {e}")
        return 0


def persist_options_contracts(df: pd.DataFrame) -> int:
    """Persist options contracts to database."""
    return persist_price_bars(df, table="options_contracts")


def persist_macro_data(df: pd.DataFrame) -> int:
    """Persist macro indicators to database."""
    return persist_price_bars(df, table="macro_indicators")


# =============================================================================
# BATCH INGESTION FUNCTIONS
# =============================================================================

def ingest_all_stocks(
    lookback_days: int = 30,
    max_symbols: int = None,
    workers: int = 10,
) -> int:
    """Ingest all US stocks."""
    logger.info("=" * 70)
    logger.info("INGESTING ALL US STOCKS")
    logger.info("=" * 70)

    # Get all stock tickers
    tickers = get_polygon_tickers(market="stocks", active=True)

    if not tickers:
        logger.error("No tickers retrieved")
        return 0

    # Filter to US exchanges only
    us_exchanges = ["XNYS", "XNAS", "XASE", "ARCX", "BATS"]
    symbols = [
        t["ticker"] for t in tickers
        if t.get("primary_exchange") in us_exchanges
        and t.get("type") in ["CS", "ETF", "ADRC"]  # Common Stock, ETF, ADR
    ]

    if max_symbols:
        symbols = symbols[:max_symbols]

    logger.info(f"Ingesting {len(symbols)} US stocks...")

    total_rows = 0
    all_data = []

    def fetch_symbol(symbol: str) -> Tuple[str, pd.DataFrame]:
        df = fetch_polygon_bars(symbol, lookback_days=lookback_days)
        time.sleep(0.2)  # Rate limiting
        return symbol, df

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_symbol, s): s for s in symbols}

        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                _, df = future.result()
                if not df.empty:
                    all_data.append(df)
                    if (i + 1) % 100 == 0:
                        logger.info(f"Progress: {i + 1}/{len(symbols)} symbols")
            except Exception as e:
                logger.debug(f"Failed {symbol}: {e}")

    # Persist in batches
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        total_rows = persist_price_bars(combined)

    logger.info(f"Stock ingestion complete: {total_rows} rows")
    return total_rows


def ingest_etfs(lookback_days: int = 30, workers: int = 10) -> int:
    """Ingest all ETFs."""
    logger.info("=" * 70)
    logger.info("INGESTING ALL ETFs")
    logger.info("=" * 70)

    tickers = get_polygon_tickers(market="stocks", active=True)

    # Filter to ETFs
    etf_symbols = [t["ticker"] for t in tickers if t.get("type") == "ETF"]

    logger.info(f"Ingesting {len(etf_symbols)} ETFs...")

    all_data = []

    def fetch_etf(symbol: str) -> pd.DataFrame:
        df = fetch_polygon_bars(symbol, lookback_days=lookback_days)
        time.sleep(0.2)
        return df

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_etf, s): s for s in etf_symbols}

        for future in as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                pass

    total_rows = 0
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        total_rows = persist_price_bars(combined)

    logger.info(f"ETF ingestion complete: {total_rows} rows")
    return total_rows


def ingest_options(underlyings: List[str] = None, workers: int = 5) -> int:
    """Ingest options chains for specified underlyings."""
    logger.info("=" * 70)
    logger.info("INGESTING OPTIONS CHAINS")
    logger.info("=" * 70)

    if not underlyings:
        # Default to most liquid underlyings
        underlyings = [
            "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "AMD", "JPM", "BAC", "GS", "V", "MA",
            "GLD", "SLV", "USO", "TLT",
        ]

    logger.info(f"Fetching options for {len(underlyings)} underlyings...")

    all_contracts = []

    for underlying in underlyings:
        df = fetch_polygon_options_chain(underlying)
        if not df.empty:
            all_contracts.append(df)
            logger.info(f"  {underlying}: {len(df)} contracts")
        time.sleep(0.3)

    total_rows = 0
    if all_contracts:
        combined = pd.concat(all_contracts, ignore_index=True)
        total_rows = persist_options_contracts(combined)

    logger.info(f"Options ingestion complete: {total_rows} contracts")
    return total_rows


def ingest_crypto(lookback_hours: int = 168) -> int:
    """Ingest crypto data."""
    logger.info("=" * 70)
    logger.info("INGESTING CRYPTO DATA")
    logger.info("=" * 70)

    all_data = []

    for symbol in CRYPTO_SYMBOLS:
        df = fetch_coinbase_candles(symbol, lookback_hours=lookback_hours)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {symbol}: {len(df)} candles")
        time.sleep(0.5)

    total_rows = 0
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        total_rows = persist_price_bars(combined)

    logger.info(f"Crypto ingestion complete: {total_rows} rows")
    return total_rows


def ingest_macro() -> int:
    """Ingest FRED macro indicators."""
    logger.info("=" * 70)
    logger.info("INGESTING MACRO INDICATORS")
    logger.info("=" * 70)

    all_data = []

    for series_id, description in FRED_SERIES.items():
        df = fetch_fred_series(series_id)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {series_id} ({description}): {len(df)} observations")
        time.sleep(0.5)

    total_rows = 0
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        total_rows = persist_macro_data(combined)

    logger.info(f"Macro ingestion complete: {total_rows} rows")
    return total_rows


def ingest_all(
    lookback_days: int = 30,
    max_stocks: int = None,
    workers: int = 10,
    include_massive: bool = True,
    include_alpha_vantage: bool = True,
    include_ibkr: bool = True,
) -> Dict[str, int]:
    """Ingest everything from all sources."""
    results = {}

    logger.info("=" * 70)
    logger.info("FULL UNIVERSE DATA INGESTION")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # Polygon Stocks
    results["stocks"] = ingest_all_stocks(lookback_days, max_stocks, workers)

    # Polygon Options
    results["options"] = ingest_options()

    # Crypto
    results["crypto"] = ingest_crypto()

    # FRED Macro
    results["macro"] = ingest_macro()

    # Massive S3 (Polygon flat files)
    if include_massive:
        results["massive_equity"] = ingest_massive_data("equity/minute", max_files=10)
        results["massive_options"] = ingest_massive_data("options/trades", max_files=5)

    # Alpha Vantage Premium
    if include_alpha_vantage:
        results["alpha_vantage"] = ingest_alpha_vantage_premium()

    # IBKR (requires TWS/Gateway running)
    if include_ibkr:
        results["ibkr"] = ingest_ibkr_data()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 70)

    total = 0
    for source, count in results.items():
        logger.info(f"  {source}: {count:,} rows")
        total += count

    logger.info(f"\nTOTAL: {total:,} rows ingested")
    logger.info("=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Universe Data Ingestion")

    # What to ingest
    parser.add_argument("--all", action="store_true", help="Ingest everything from all sources")
    parser.add_argument("--stocks", action="store_true", help="Ingest US stocks (Polygon)")
    parser.add_argument("--etfs", action="store_true", help="Ingest ETFs (Polygon)")
    parser.add_argument("--options", action="store_true", help="Ingest options chains (Polygon)")
    parser.add_argument("--crypto", action="store_true", help="Ingest crypto (Coinbase)")
    parser.add_argument("--macro", action="store_true", help="Ingest FRED macro data")
    parser.add_argument("--massive", action="store_true", help="Ingest Massive S3 flat files")
    parser.add_argument("--alpha-vantage", action="store_true", help="Ingest Alpha Vantage premium")
    parser.add_argument("--ibkr", action="store_true", help="Ingest IBKR data (requires TWS)")

    # Options
    parser.add_argument("--lookback", type=int, default=30, help="Days of historical data")
    parser.add_argument("--max-stocks", type=int, help="Limit number of stocks")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Interval between runs (seconds)")

    # Massive options
    parser.add_argument("--massive-type", type=str, default="equity/minute",
                       help="Massive data type (equity/minute, equity/daily, options/trades)")
    parser.add_argument("--massive-files", type=int, default=10, help="Max Massive files to process")

    args = parser.parse_args()

    # Setup logging
    logger.add(
        "logs/data_ingestion_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )

    # Check API keys
    settings = get_settings()
    settings.log_configuration_status()

    def run_ingestion():
        results = {}

        if args.all:
            results = ingest_all(
                args.lookback,
                args.max_stocks,
                args.workers,
                include_massive=True,
                include_alpha_vantage=True,
                include_ibkr=True,
            )
        else:
            if args.stocks:
                results["stocks"] = ingest_all_stocks(args.lookback, args.max_stocks, args.workers)
            if args.etfs:
                results["etfs"] = ingest_etfs(args.lookback, args.workers)
            if args.options:
                results["options"] = ingest_options()
            if args.crypto:
                results["crypto"] = ingest_crypto()
            if args.macro:
                results["macro"] = ingest_macro()
            if args.massive:
                results["massive"] = ingest_massive_data(args.massive_type, max_files=args.massive_files)
            if getattr(args, 'alpha_vantage', False):
                results["alpha_vantage"] = ingest_alpha_vantage_premium()
            if args.ibkr:
                results["ibkr"] = ingest_ibkr_data()

        # If no specific flags, show help
        if not results:
            parser.print_help()
            return {}

        return results

    if args.continuous:
        logger.info(f"Running in continuous mode (interval: {args.interval}s)")

        while True:
            try:
                run_ingestion()
                logger.info(f"Next ingestion in {args.interval} seconds...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Ingestion stopped")
                break
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"Retrying in {args.interval} seconds...")
                time.sleep(args.interval)
    else:
        run_ingestion()


if __name__ == "__main__":
    main()
