# TradingView Integration Module

**Branch:** `feature/tradingview-webscrape`

## Overview

This module scrapes advanced trading indicators, strategies, and Pine Scripts from TradingView and converts them to Python trading logic for the ALC-Algo system.

## Features

### 1. TradingView Scraper (`src/data_sources/tradingview_scraper.py`)

Scrapes TradingView for:
- **Top-rated indicators** (RSI, MACD, VWAP variants, etc.)
- **Popular strategies** (breakout, momentum, mean reversion)
- **Institutional indicators**:
  - Volume Profile
  - Order Flow / Footprint Charts
  - Market Profile
  - Anchored VWAP
  - Smart Money Concepts
  - Liquidity Levels
  - Cumulative Volume Delta (CVD)

**Usage:**
```python
from src.data_sources.tradingview_scraper import TradingViewScraper

scraper = TradingViewScraper(rate_limit=2.0)

# Get top indicators
top_indicators = scraper.get_top_indicators(limit=20)

# Get institutional-grade indicators
institutional = scraper.get_institutional_indicators()

# Search by keyword
vwap_scripts = scraper.search_by_keyword("Anchored VWAP", limit=10)

# Save to file
scraper.save_scripts_to_file(top_indicators, 'top_indicators.json')
```

### 2. Pine Script Parser (`src/data_sources/tradingview_pine_parser.py`)

Parses Pine Script code and converts to Python:
- **Extracts indicators** (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, etc.)
- **Extracts entry/exit conditions**
- **Extracts strategy parameters**
- **Converts to Python strategy functions**

**Supported Indicators:**
- Moving Averages (SMA, EMA)
- Momentum (RSI, Stochastic, MACD)
- Volatility (ATR, Bollinger Bands)
- Volume (VWAP, Volume Profile)
- Price Action (Highest, Lowest, Crossover, Crossunder)

**Usage:**
```python
from src.data_sources.tradingview_pine_parser import PineScriptParser

parser = PineScriptParser()

# Parse Pine Script
pine_code = """
//@version=5
strategy("My Strategy", overlay=true)

rsi_length = input.int(14, "RSI Length")
rsi_value = ta.rsi(close, rsi_length)

long_condition = ta.crossover(rsi_value, 30)
if long_condition
    strategy.entry("Long", strategy.long)
"""

strategy = parser.parse(pine_code)

# Convert to Python
python_code = parser.to_python_strategy(strategy)
print(python_code)
```

## Installation

```bash
pip install requests beautifulsoup4 lxml
```

## Running the Scraper

```bash
# Scrape top indicators and strategies
python src/data_sources/tradingview_scraper.py

# Parse Pine Script example
python src/data_sources/tradingview_pine_parser.py
```

## Output Files

Scraped data saved to:
```
data/tradingview/
├── scraped_scripts.json       # All scraped scripts
├── top_indicators.json        # Top-rated indicators
├── top_strategies.json        # Top-rated strategies
└── institutional.json         # Institutional-grade indicators
```

## Institutional Indicators to Scrape

### High Priority:
1. **Volume Profile** - Shows price levels with most volume
2. **Order Flow / Delta** - Buy vs sell volume imbalance
3. **Market Profile** - Time-based volume distribution
4. **Anchored VWAP** - VWAP from specific price levels
5. **Smart Money Concepts** - Order blocks, FVGs, liquidity grabs

### Medium Priority:
6. **Cumulative Volume Delta (CVD)** - Running buy/sell pressure
7. **Footprint Charts** - Bid/ask volume at each price level
8. **Liquidity Heatmaps** - Identify support/resistance zones
9. **Market Structure** - Higher highs, lower lows detection
10. **Volume Weighted Moving Average (VWMA)** - Volume-adjusted MA

### Advanced:
11. **VPOC (Volume Point of Control)** - Peak volume price
12. **Value Area** - 70% of volume range
13. **Volume Shelves** - Horizontal volume clusters
14. **Iceberg Orders** - Hidden large orders detection
15. **Tape Reading** - Time & Sales analysis

## Integration with ALC-Algo

### Step 1: Scrape Indicators
```python
scraper = TradingViewScraper()
institutional = scraper.get_institutional_indicators()
```

### Step 2: Parse Pine Scripts
```python
parser = PineScriptParser()
for script_url in script_urls:
    script = scraper.get_script_details(script_url)
    parsed = parser.parse(script.script_code)
    python_code = parser.to_python_strategy(parsed)
```

### Step 3: Create Agent Wrappers
```python
from src.agents.base_agent import BaseAgent

class VolumeProfileAgent(BaseAgent):
    """Agent using TradingView Volume Profile indicator"""

    def __init__(self):
        super().__init__("VolumeProfileAgent")
        self.indicator = load_tradingview_indicator("volume_profile")

    def decide(self, market_data: Dict) -> Dict:
        # Use parsed TradingView indicator
        vp_levels = self.indicator.calculate(market_data)

        if market_data['price'] < vp_levels['poc']:
            return {'action': 'buy', 'confidence': 0.8}
        else:
            return {'action': 'sell', 'confidence': 0.8}
```

### Step 4: Backtest New Strategies
```python
# Add TradingView strategies to backtesting
from backtest_2025_strategies import run_backtest_for_strategy

def tradingview_strategy_1(price_data, current_date):
    # Use converted Pine Script logic
    signals = parsed_strategy.calculate_signals(price_data)
    return signals
```

## Roadmap

### Phase 1: Basic Scraping ✅
- [x] TradingView scraper framework
- [x] Pine Script parser
- [x] Basic indicator extraction

### Phase 2: Advanced Indicators (Next)
- [ ] Volume Profile implementation
- [ ] Order Flow / Delta calculation
- [ ] Market Profile
- [ ] Anchored VWAP variants

### Phase 3: Integration
- [ ] Create agents for each indicator
- [ ] Backtest TradingView strategies
- [ ] Compare vs existing ALC-Algo strategies

### Phase 4: Live Trading
- [ ] Real-time indicator calculations
- [ ] TradingView webhook integration
- [ ] Alert system

## Notes

### Rate Limiting
- Respect TradingView's terms of service
- Default rate limit: 2 seconds between requests
- Use responsibly - don't hammer their servers

### Legal Considerations
- TradingView scripts may have licenses
- Only use open-source scripts or scripts with permission
- Do not redistribute proprietary indicators

### Data Quality
- TradingView structure may change (scraper may need updates)
- Verify parsed indicators before live trading
- Always backtest converted strategies

## Examples

### Example 1: Scrape Top 50 Indicators
```python
scraper = TradingViewScraper()
top_50 = scraper.get_top_indicators(limit=50)
scraper.save_scripts_to_file(top_50, 'top_50_indicators.json')
```

### Example 2: Find Smart Money Indicators
```python
smart_money = scraper.search_by_keyword("Smart Money Concepts", limit=20)
for script in smart_money:
    print(f"{script['title']} by {script['author']}")
    print(f"  Likes: {script['likes']}")
    print(f"  URL: {script['url']}")
```

### Example 3: Parse and Convert Strategy
```python
parser = PineScriptParser()
strategy = parser.parse(pine_script_code)

print(f"Strategy: {strategy.title}")
print(f"Indicators: {[i.name for i in strategy.indicators]}")
print(f"Entry conditions: {len(strategy.entry_conditions)}")

# Convert to Python
python_strategy = parser.to_python_strategy(strategy)
exec(python_strategy)  # Now callable as Python function
```

## Contributing

To add support for new Pine Script functions:
1. Add pattern to `INDICATOR_PATTERNS` in `PineScriptParser`
2. Add parsing logic in `_parse_indicators()`
3. Add Python conversion in `to_python_strategy()`
4. Test with example Pine Script

## Author

Tom Hogan | Alpha Loop Capital, LLC

## Branch Status

**Status:** 🚧 In Development
**Branch:** `feature/tradingview-webscrape`
**Ready for:** Scraping and parsing (needs testing with real TradingView URLs)

**Next Steps:**
1. Test scraper with live TradingView URLs
2. Fix any HTML parsing issues (TradingView structure)
3. Add more Pine Script function support
4. Create institutional indicator implementations
5. Integrate with existing ALC-Algo agents
