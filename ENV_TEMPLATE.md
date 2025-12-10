# Environment Configuration Template

Copy these values to your `API - Dec 2025.env` file in Dropbox:
- Windows: `C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env`
- Mac: `~/Alphaloopcapital Dropbox/ALC Tech Agents/API - Dec 2025.env`

**NEVER commit actual API keys to git.**

## Database - Azure SQL Server

```
SQL_SERVER=alc-sql-server.database.windows.net
SQL_DB=alc_market_data
DB_USERNAME=CloudSAb3fcbb35
DB_PASSWORD=YOUR_PASSWORD_HERE
```

## Data APIs

```
# Alpha Vantage - Premium (75 calls/minute)
ALPHAVANTAGE_API_KEY=YOUR_KEY_HERE

# Polygon.io / Massive.com
PolygonIO_API_KEY=YOUR_KEY_HERE
MASSIVE_ACCESS_KEY=YOUR_KEY_HERE
MASSIVE_SECRET_KEY=YOUR_KEY_HERE
MASSIVE_ENDPOINT_URL=https://files.massive.com

# Coinbase
COINBASE_API_KEY=YOUR_KEY_HERE
COINBASE_API_SECRET=YOUR_SECRET_HERE

# FRED
FRED_DATA_API=YOUR_KEY_HERE
```

## AI Services

```
OPENAI_SECRET=YOUR_KEY_HERE
ANTHROPIC_API_KEY=YOUR_KEY_HERE
PERPLEXITY_API_KEY=YOUR_KEY_HERE
API_KEY=YOUR_GOOGLE_KEY_HERE
```

## IBKR Trading

```
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
```

Note: Port 7497 = Paper Trading (safe), Port 7496 = Live Trading (real money!)

## Current API Key Status

| Service | Status | Notes |
|---------|--------|-------|
| Alpha Vantage | Active | Premium tier |
| Polygon/Massive | Active | Unlimited |
| Coinbase | Active | Standard |
| FRED | Active | Free |
| OpenAI | Active | Tier 2 |
| Anthropic | Active | Tier 2 |
| Perplexity | Active | Updated Dec 2025 |
| Google/Gemini | Active | Free tier |
| Azure SQL | Active | Connected |

## Testing Your Configuration

After setting up your .env file, run:

```bash
# Windows
python scripts/test_all_apis.py

# Mac
python scripts/test_all_apis.py
```

All services should show SUCCESS or connected status.

