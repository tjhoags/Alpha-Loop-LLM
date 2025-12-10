# ALC-Algo Data Sources

This document outlines the configured data sources for the Alpha Loop Capital algorithmic trading platform.

## Active Data Sources

### 1. Yahoo Finance (`yfinance`)
- **Type:** Free / Public
- **Usage:** 
  - Historical price data (OHLCV)
  - Basic company fundamentals (Sector, Industry, Market Cap)
  - Options chains
- **Implementation:** `src/ingest/collector.py`
- **Status:** ✅ Active

### 2. Alpha Vantage
- **Type:** Premium / API Key Required
- **Usage:**
  - Adjusted daily/weekly/monthly price data
  - Deep fundamentals (Income Statement, Balance Sheet, Cash Flow)
  - Technical indicators (computed server-side)
- **Implementation:** `src/ingest/alpha_vantage.py`
- **Status:** ✅ Active
- **Configuration:** Set `ALPHA_VANTAGE_API_KEY` in `.env`

### 3. FRED (Federal Reserve Economic Data)
- **Type:** Free / API Key Required
- **Usage:**
  - Macroeconomic indicators (GDP, Inflation/CPI, Interest Rates)
  - Employment data (Unemployment Rate)
  - Market stress indicators (VIX, Credit Spreads)
- **Implementation:** `src/ingest/dataset_builder.py`
- **Status:** ✅ Active
- **Configuration:** Set `FRED_API_KEY` in `.env`

### 4. Portfolio History
- **Type:** Internal Data
- **Usage:**
  - Ingesting past trades from Broker exports (CSV, Excel)
  - Analyzing performance (Win Rate, P&L, Drawdowns)
- **Implementation:** `src/ingest/portfolio.py`
- **Supported Formats:**
  - Interactive Brokers (CSV)
  - Schwab (CSV)
  - Generic (CSV/Excel/JSON)
- **Status:** ✅ Active

## Planned Data Sources

### 1. Interactive Brokers (IBKR)
- **Type:** Brokerage API
- **Usage:** Real-time market data and live execution.
- **Implementation:** `src/ingest/ibkr.py`
- **Status:** 🚧 Pending Implementation

### 2. SEC EDGAR
- **Type:** Regulatory Filings
- **Usage:** 10-K/10-Q parsing for sentiment and hidden risks.
- **Status:** 🚧 Planned

## Configuration

To enable these data sources, ensure your `.env` file contains:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
# IBKR_PORT=7497 (optional)
```

