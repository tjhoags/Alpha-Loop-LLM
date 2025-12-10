# ALC-Algo Data Sources - Complete Reference

**Author:** Tom Hogan | **Organization:** Alpha Loop Capital, LLC  
**Last Updated:** December 9, 2025

---

## 📊 Data Source Summary

| Category | Provider | Status | Priority | API Key Required |
|----------|----------|--------|----------|------------------|
| **Market Data** | Alpha Vantage | ✅ Configured | High | Yes |
| **Market Data** | Yahoo Finance | ✅ Configured | Medium | No |
| **Market Data** | IBKR | ✅ Configured | Critical | Yes (Account) |
| **Market Data** | Finviz | ✅ Configured | Medium | No (Free tier) |
| **Alternative** | Fiscal.ai | ✅ Configured | Medium | Yes |
| **Economic** | FRED | ✅ Configured | High | Yes |
| **Crypto** | Coinbase | ✅ Configured | Medium | Yes |
| **AI/ML** | OpenAI | ✅ Configured | Critical | Yes |
| **AI/ML** | Anthropic | ✅ Configured | Critical | Yes |
| **AI/ML** | Google (Gemini) | ✅ Configured | High | Yes |
| **AI/ML** | Perplexity | ✅ Configured | High | Yes |

---

## 🔑 Required API Keys (All in master_alc_env)

### Critical (Must Have)
```bash
# Interactive Brokers (Execution)
IBKR_ACCOUNT_ID=your_account_id
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper: 7497, Live: 7496

# AI/ML Protocols
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY_1=...
PERPLEXITY_API_KEY=pplx-...
```

### High Priority
```bash
# Market Data
ALPHA_VANTAGE_API_KEY=...
FRED_API_KEY=...

# Additional Google Keys (Backup/Rate Limiting)
GOOGLE_API_KEY_2=...
GOOGLE_API_KEY_3=...
```

### Medium Priority
```bash
# Alternative Data
FISCAL_AI_API_KEY=...
COINBASE_API_KEY=...
COINBASE_API_SECRET=...

# Collaboration
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
NOTION_API_KEY=secret_...
DROPBOX_ACCESS_TOKEN=...
```

---

## 📈 Market Data Sources

### 1. Interactive Brokers (IBKR)
**Role:** Primary execution and real-time data

| Feature | Details |
|---------|---------|
| Real-time quotes | ✅ |
| Historical data | ✅ Up to 20 years |
| Options chains | ✅ |
| Order execution | ✅ |
| Portfolio data | ✅ |

**Rate Limits:**
- 50 messages/second (single client)
- 100 historical data requests/10 minutes

**Used By Agents:**
- ExecutionAgent (order routing)
- DataAgent (real-time prices)
- OptionsAgent (chains, greeks)

### 2. Alpha Vantage
**Role:** Historical data, fundamentals, technical indicators

| Feature | Details |
|---------|---------|
| Daily OHLCV | ✅ 20+ years |
| Intraday (1min-60min) | ✅ Recent data |
| Fundamentals | ✅ Income, Balance, Cash Flow |
| Technical indicators | ✅ SMA, EMA, RSI, MACD, etc. |
| Earnings | ✅ Calendar and history |

**Rate Limits:**
- Free: 5 calls/minute, 500/day
- Premium: 75 calls/minute

**Used By Agents:**
- DataAgent (historical data)
- ResearchAgent (fundamentals)
- Strategy agents (technical signals)

### 3. Yahoo Finance (yfinance)
**Role:** Backup data, options chains

| Feature | Details |
|---------|---------|
| Historical OHLCV | ✅ |
| Options chains | ✅ |
| Company info | ✅ |
| Dividends/Splits | ✅ |

**Rate Limits:**
- ~2000 requests/hour (unofficial)

**Used By Agents:**
- DataAgent (backup/validation)
- OptionsAgent (chains)

### 4. Finviz
**Role:** Stock screening, news, heat maps

| Feature | Details |
|---------|---------|
| Stock screener | ✅ |
| News aggregation | ✅ |
| Technical patterns | ✅ |
| Sector performance | ✅ |

**Used By Agents:**
- ResearchAgent (screening)
- SentimentAgent (news)
- Sector agents (sector analysis)

---

## 📊 Alternative Data Sources

### 1. Fiscal.ai
**Role:** AI-powered financial analysis

| Feature | Details |
|---------|---------|
| Sentiment analysis | ✅ |
| Alternative data | ✅ |
| Macro indicators | ✅ |

### 2. FRED (Federal Reserve Economic Data)
**Role:** Economic indicators

| Series | Description |
|--------|-------------|
| GDP | Gross Domestic Product |
| UNRATE | Unemployment Rate |
| CPIAUCSL | Consumer Price Index |
| DFF | Fed Funds Rate |
| T10Y2Y | 10Y-2Y Treasury Spread |

**Used By Agents:**
- MacroAgent (economic regime)
- RiskAgent (systemic risk)
- StrategyAgent (macro-aware trading)

---

## 🪙 Crypto Data Sources

### 1. Coinbase
**Role:** Crypto execution and data

| Feature | Details |
|---------|---------|
| Real-time quotes | ✅ |
| Historical OHLCV | ✅ |
| Order execution | ✅ |
| Portfolio tracking | ✅ |

**Supported Assets:**
- BTC, ETH, SOL, and 100+ more

**Used By Agents:**
- CryptoAgent (analysis)
- ExecutionAgent (crypto trades)

---

## 🤖 AI/ML Protocol Sources

### 1. OpenAI (GPT-4)
**Role:** Complex reasoning, analysis

| Use Case | Model |
|----------|-------|
| Market analysis | gpt-4-turbo-preview |
| News summarization | gpt-4 |
| Strategy evaluation | gpt-4 |

**Rate Limits:**
- 10,000 TPM (tokens per minute) - varies by plan

### 2. Anthropic (Claude)
**Role:** Long-context analysis, strategy

| Use Case | Model |
|----------|-------|
| Strategy analysis | claude-3-opus |
| Document analysis | claude-3-opus |
| Code generation | claude-3-opus |

### 3. Google (Gemini)
**Role:** Real-time analysis, multi-modal

| Use Case | Model |
|----------|-------|
| Real-time research | gemini-pro |
| Chart analysis | gemini-pro-vision |

**Note:** Multiple API keys for rate limit handling

### 4. Perplexity
**Role:** Web-connected research

| Use Case | Model |
|----------|-------|
| Real-time research | pplx-70b-online |
| Market intelligence | pplx-70b-online |

**Advantage:** Live web access for current information

---

## 📁 Internal Data Sources

### 1. Historical Trades (Your Portfolio)
**Location:** Import via IBKR Flex Query

**Data Includes:**
- Trade date, symbol, quantity, price
- P&L per trade
- Position sizes over time

**Import Command:**
```bash
python scripts/ingest_portfolio.py "path/to/trades.csv"
```

### 2. Dropbox Sync
**Purpose:** Shared configuration and data

**Synced Paths:**
- `master_alc_env` - API keys
- Trade exports
- Configuration backups

---

## 🔄 Data Quality & Validation

### Validation Rules

1. **Price Data:**
   - No negative prices
   - Volume must be positive
   - OHLC relationship: Low ≤ Open, Close ≤ High

2. **Fundamental Data:**
   - Quarterly/annual frequency validated
   - Currency normalization applied

3. **Cross-Source Validation:**
   - Primary: IBKR
   - Backup: Alpha Vantage
   - Tertiary: Yahoo Finance
   - Discrepancies logged and flagged

### Data Freshness

| Data Type | Acceptable Delay |
|-----------|------------------|
| Real-time quotes | < 1 second |
| Historical daily | Same-day |
| Fundamentals | Quarterly |
| Economic data | Weekly |

---

## 📊 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  IBKR ──┐                                                            │
│  Alpha V ─┼──► DataAgent ──► Validation ──► Azure Storage            │
│  Yahoo ──┘                                                           │
│                                                                       │
│  FRED ────► MacroAgent ──► Economic Regime ──► All Strategy Agents  │
│                                                                       │
│  Fiscal.ai ─► SentimentAgent ──► Sentiment Scores ──► Trading       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚨 Data Source Failover

If primary source fails:

| Primary | Fallback 1 | Fallback 2 |
|---------|------------|------------|
| IBKR | Alpha Vantage | Yahoo |
| Alpha Vantage | Yahoo | IBKR |
| Coinbase | (no fallback - crypto only) | - |

---

## ✅ Checklist: All Data Sources Configured

- [ ] IBKR TWS/Gateway running
- [ ] Alpha Vantage API key in master_alc_env
- [ ] FRED API key in master_alc_env
- [ ] Coinbase API keys in master_alc_env
- [ ] OpenAI API key in master_alc_env
- [ ] Anthropic API key in master_alc_env
- [ ] Google API keys (x3) in master_alc_env
- [ ] Perplexity API key in master_alc_env
- [ ] Fiscal.ai API key in master_alc_env
- [ ] Slack webhook configured
- [ ] Notion API key configured
- [ ] Azure Storage configured for data caching

---

*Data Sources Reference*  
*Alpha Loop Capital, LLC*  
*Tom Hogan, Founder*

