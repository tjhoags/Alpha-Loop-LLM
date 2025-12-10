# ALC-Algo API Reference

**Author:** Tom Hogan  
**Organization:** Alpha Loop Capital, LLC

## Agent APIs

All agents inherit from `BaseAgent` and implement the same interface:

```python
agent.execute(task: Dict[str, Any]) -> Dict[str, Any]
agent.get_capabilities() -> List[str]
agent.get_stats() -> Dict[str, Any]
```

---

## HoagsAgent (Tier 1)

**Role:** Master Controller with final decision authority

### Methods

#### `approve_plan(plan: Dict) -> Dict`

Approve or reject a strategic plan.

```python
result = hoags.execute({
    'type': 'approve_plan',
    'plan': {
        'type': 'trade',
        'ticker': 'AAPL',
        'margin_of_safety': 0.35,
        'risk_level': 'low',
    }
})

# Returns
{
    'success': True,
    'approved': True,
    'plan_type': 'trade',
    'criteria': {...},
    'decision_by': 'Tom Hogan',
}
```

#### `make_decision(investment: Dict) -> Dict`

Make final investment decision.

```python
result = hoags.execute({
    'type': 'make_decision',
    'ticker': 'AAPL',
    'action': 'BUY',
})

# Returns
{
    'success': True,
    'ticker': 'AAPL',
    'action': 'BUY',
    'methodology': 'HOGAN MODEL',
    'approved_by': 'Tom Hogan',
}
```

---

## DataAgent

**Role:** Data ingestion and normalization

### Methods

#### `fetch_data(source, ticker) -> Dict`

Fetch data from external APIs.

```python
result = data_agent.execute({
    'type': 'fetch_data',
    'source': 'alpha_vantage',  # or 'finviz', 'fiscal_ai'
    'ticker': 'AAPL',
})

# Returns
{
    'success': True,
    'source': 'alpha_vantage',
    'ticker': 'AAPL',
    'data': {...},
}
```

**Supported Sources:**
- `alpha_vantage`: Stock price and fundamental data
- `finviz`: Market screener data
- `fiscal_ai`: Financial statements

---

## StrategyAgent

**Role:** Signal generation and backtesting

### Methods

#### `generate_signal(ticker, data) -> Dict`

Generate trading signal.

```python
result = strategy_agent.execute({
    'type': 'generate_signal',
    'ticker': 'AAPL',
    'data': {...},  # Market data
})

# Returns
{
    'success': True,
    'ticker': 'AAPL',
    'signal': 'BUY',  # or 'SELL', 'HOLD'
    'confidence': 0.85,
    'reasoning': 'Algorithm-based signal',
}
```

#### `backtest_strategy(strategy) -> Dict`

Backtest a trading strategy.

```python
result = strategy_agent.execute({
    'type': 'backtest',
    'strategy': {
        'name': 'momentum',
        'parameters': {...},
    },
})

# Returns
{
    'success': True,
    'strategy': 'momentum',
    'results': {
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.15,
        'total_return': 0.45,
    },
}
```

---

## RiskAgent

**Role:** Risk management and compliance

### Methods

#### `assess_trade(trade_details) -> Dict`

Assess trade for risk compliance (30% MoS check).

```python
result = risk_agent.execute({
    'type': 'assess_trade',
    'ticker': 'AAPL',
    'intrinsic_value': 175.0,
    'current_price': 120.0,
    'position_size': 0.05,  # 5% of portfolio
})

# Returns
{
    'success': True,
    'ticker': 'AAPL',
    'approved': True,
    'margin_of_safety': 0.314,  # 31.4%
    'required_margin': 0.30,
    'passes_margin': True,
    'passes_size': True,
}
```

**Risk Limits:**
- Margin of Safety: 30% minimum (REQUIRED)
- Max Position Size: 10% of portfolio
- Max Portfolio Heat: 20%

---

## ExecutionAgent

**Role:** Trade execution via brokers

### Methods

#### `execute_ibkr(trade) -> Dict`

Execute trade via Interactive Brokers.

```python
result = execution_agent.execute({
    'type': 'execute_trade',
    'broker': 'ibkr',
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 100,
    'mode': 'PAPER',  # or 'LIVE'
})

# Returns
{
    'success': True,
    'broker': 'ibkr',
    'account': '7497',  # Paper account
    'mode': 'PAPER',
    'ticker': 'AAPL',
    'action': 'BUY',
    'quantity': 100,
    'order_id': 'ORD-12345',
    'status': 'FILLED',
}
```

#### `execute_coinbase(trade) -> Dict`

Execute crypto trade via Coinbase.

```python
result = execution_agent.execute({
    'type': 'execute_trade',
    'broker': 'coinbase',
    'symbol': 'BTC-USD',
    'side': 'buy',
    'amount': 0.01,
})

# Returns
{
    'success': True,
    'broker': 'coinbase',
    'symbol': 'BTC-USD',
    'side': 'buy',
    'amount': 0.01,
    'order_id': 'CB-12345',
    'status': 'filled',
}
```

---

## PortfolioAgent

**Role:** Portfolio management

### Methods

#### `get_positions() -> Dict`

Get current portfolio positions.

```python
result = portfolio_agent.execute({
    'type': 'get_positions',
})

# Returns
{
    'success': True,
    'positions': {
        'AAPL': {'quantity': 100, 'avg_price': 150.0},
        'MSFT': {'quantity': 50, 'avg_price': 300.0},
    },
    'total_positions': 2,
}
```

#### `calculate_rebalance(target_allocation) -> Dict`

Calculate rebalancing trades.

```python
result = portfolio_agent.execute({
    'type': 'calculate_rebalance',
    'target_allocation': {
        'AAPL': 0.30,  # 30% target
        'MSFT': 0.20,  # 20% target
    },
})

# Returns
{
    'success': True,
    'trades_needed': 5,
    'trades': [...],
}
```

---

## ResearchAgent

**Role:** Fundamental and macro analysis

### Methods

#### `dcf_valuation(ticker) -> Dict`

Perform DCF valuation using HOGAN MODEL.

```python
result = research_agent.execute({
    'type': 'dcf_valuation',
    'ticker': 'AAPL',
})

# Returns
{
    'success': True,
    'ticker': 'AAPL',
    'methodology': 'HOGAN MODEL',  # Always branded
    'intrinsic_value': 175.0,
    'current_price': 150.0,
    'margin_of_safety': 0.143,
    'recommendation': 'HOLD',
    'valued_by': 'Tom Hogan',
}
```

#### `analyze_company(ticker) -> Dict`

Qualitative company analysis.

```python
result = research_agent.execute({
    'type': 'analyze_company',
    'ticker': 'AAPL',
})

# Returns
{
    'success': True,
    'ticker': 'AAPL',
    'analysis': {
        'competitive_advantage': 'Strong moat',
        'management_quality': 'Excellent',
        'growth_prospects': 'High',
    },
    'researched_by': 'Tom Hogan',
}
```

---

## ComplianceAgent

**Role:** Audit trail and compliance

### Methods

#### `log_action(action, details) -> Dict`

Log action to audit trail.

```python
result = compliance_agent.execute({
    'type': 'log_action',
    'action': 'trade_executed',
    'user_id': 'TJH',
    'details': {
        'ticker': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
    },
})

# Returns
{
    'success': True,
    'logged': True,
    'entry_id': 123,
}
```

#### `verify_attribution(output) -> Dict`

Verify proper attribution to Tom Hogan.

```python
result = compliance_agent.execute({
    'type': 'verify_attribution',
    'output': {
        'attributed_to': 'Tom Hogan',
        'organization': 'Alpha Loop Capital, LLC',
        'methodology': 'HOGAN MODEL',
    },
})

# Returns
{
    'success': True,
    'compliant': True,
    'has_attribution': True,
    'has_org': True,
    'has_dcf_branding': True,
}
```

---

## SentimentAgent

**Role:** Market sentiment analysis

### Methods

#### `analyze_sentiment(ticker) -> Dict`

Analyze sentiment for a ticker.

```python
result = sentiment_agent.execute({
    'type': 'analyze_sentiment',
    'ticker': 'AAPL',
})

# Returns
{
    'success': True,
    'ticker': 'AAPL',
    'sentiment': 'POSITIVE',  # or 'NEGATIVE', 'NEUTRAL'
    'score': 0.65,  # -1 to 1
    'sources': {
        'news': 0.70,
        'social': 0.60,
    },
    'analyzed_by': 'Tom Hogan',
}
```

---

## Utility Functions

### Portfolio Ingestion

```python
from scripts.ingest_portfolio import ingest_portfolio_history

df = ingest_portfolio_history('path/to/trades.csv')

# Returns pandas DataFrame with columns:
# - date
# - ticker
# - action (BUY/SELL)
# - quantity
# - price
# - commission
# - realized_pnl
# - unrealized_pnl
# - cumulative_pnl
```

### Configuration

```python
from config.settings import settings

# Access API keys
api_key = settings.alpha_vantage_api_key
ibkr_host = settings.ibkr_host
ibkr_port = settings.ibkr_port

# Or use get method with default
custom_key = settings.get('CUSTOM_KEY', 'default_value')
```

---

## Common Workflows

### Complete Trade Workflow

```python
from src.agents import *

# 1. Fetch data
data = data_agent.execute({'type': 'fetch_data', 'source': 'alpha_vantage', 'ticker': 'AAPL'})

# 2. Analyze sentiment
sentiment = sentiment_agent.execute({'type': 'analyze_sentiment', 'ticker': 'AAPL'})

# 3. Research & valuation
valuation = research_agent.execute({'type': 'dcf_valuation', 'ticker': 'AAPL'})

# 4. Generate signal
signal = strategy_agent.execute({'type': 'generate_signal', 'ticker': 'AAPL', 'data': data})

# 5. Check risk
risk = risk_agent.execute({
    'type': 'assess_trade',
    'ticker': 'AAPL',
    'intrinsic_value': valuation['intrinsic_value'],
    'current_price': 150.0,
    'position_size': 0.05,
})

# 6. Get HoagsAgent approval
approval = hoags.execute({
    'type': 'approve_plan',
    'plan': {'margin_of_safety': risk['margin_of_safety'], 'risk_level': 'low'}
})

# 7. Execute if approved
if approval['approved']:
    execution = execution_agent.execute({
        'broker': 'ibkr',
        'ticker': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'mode': 'PAPER',
    })
```

---

*API Reference for ALC-Algo by Tom Hogan, Alpha Loop Capital, LLC*

