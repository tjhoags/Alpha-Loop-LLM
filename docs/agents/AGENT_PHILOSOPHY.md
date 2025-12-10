# ALC-Algo Agent Philosophy

## Core Principle: Basic Techniques Do NOT Work

**Author:** Tom Hogan | Alpha Loop Capital, LLC

---

## The Problem with Basic Approaches

Basic valuation and trading techniques have been arbitraged away. Every MBA knows P/E ratios. Every quant runs momentum screens. The market efficiently incorporates these signals.

**Basic approaches that don't provide edge:**
- Simple P/E comparisons
- Standard DCF with consensus estimates
- Trailing 12-month price momentum
- Moving average crossovers
- RSI overbought/oversold
- Simple technical patterns

---

## The ALC Solution: Creative, Adaptive Intelligence

Every agent in the ALC-Algo system is built with:

### 1. **Specialized, Non-Basic Analysis**

Each agent implements creative approaches that go beyond surface-level analysis:

| Agent Type | Basic Approach (Don't Use) | Creative Approach (We Use) |
|------------|---------------------------|---------------------------|
| **Value** | P/E, P/B ratios | Normalized earnings power, hidden assets, variant perception |
| **Momentum** | Price momentum | Earnings momentum, cross-asset signals, second derivatives |
| **Risk** | VaR calculations | Regime-adaptive risk, tail risk scenarios, correlation regime |
| **Sentiment** | Bullish/Bearish counts | Narrative lifecycle, positioning extremes, pain trade analysis |

### 2. **Continuous Learning**

Every agent learns from outcomes:

```python
# Agents automatically learn from every decision
outcome = agent.learn_from_outcome(
    prediction="Stock will outperform",
    actual="Underperformed by 5%",
    confidence=0.7,
    context={'regime': 'risk_off', 'thesis': 'value'}
)

# This updates:
# - Confidence calibration (was I overconfident?)
# - Regime-specific performance (does my approach work in this regime?)
# - Mistake patterns (am I repeating errors?)
# - Bayesian beliefs (update priors based on evidence)
```

### 3. **Multiple Thinking Modes**

Each agent employs creative thinking approaches:

| Mode | Description | Example |
|------|-------------|---------|
| **Contrarian** | What if consensus is wrong? | "Everyone is bearish on retail, but what if they're wrong?" |
| **Second-Order** | What does the market miss? | "EPS beat is priced in. What isn't priced in?" |
| **Regime-Aware** | What works in this regime? | "Momentum crashes in risk-off. Switch to quality." |
| **Behavioral** | What biases can we exploit? | "Recency bias causing overreaction to last quarter." |
| **Absence** | What's NOT happening that should? | "Market not reacting to guidance cut. Why?" |

### 4. **Regime Adaptation**

Markets change. What works in one regime doesn't work in another:

```python
# Agents detect and adapt to regime changes
regime = agent.detect_regime_change(market_data)

# Performance tracked by regime
agent._regime_performance = {
    'risk_on': {'accuracy': 0.72, 'trades': 150},
    'risk_off': {'accuracy': 0.45, 'trades': 80},  # Different strategy needed
    'crisis': {'accuracy': 0.30, 'trades': 20},    # Don't trade
}
```

---

## Agent Capabilities: What Makes Each Agent Specialized

### Value Agent

**NOT:** Simple P/E screening

**INSTEAD:**
- **Normalized Earnings Power** - Through-cycle earnings, not trailing
- **Hidden Assets** - Real estate at cost, NOLs, brand value
- **Variant Perception** - Why the market is wrong
- **Pre-Mortem Analysis** - What would kill this thesis?
- **Capital Allocation Quality** - Management track record

### Momentum Agent

**NOT:** Buy what's going up

**INSTEAD:**
- **Earnings Momentum** - EPS revisions, not price
- **Relative Momentum** - Outperforming alternatives
- **Cross-Asset Signals** - Credit, FX, commodities leading equities
- **Second Derivative** - Acceleration, not just direction
- **Reversal Detection** - Know when momentum is exhausting

### Risk Agent

**NOT:** VaR and position limits

**INSTEAD:**
- **Regime-Adaptive Risk** - Risk parameters change with regime
- **Tail Risk Scenarios** - Fat tails, not normal distributions
- **Correlation Regime** - Correlations spike in crisis
- **Liquidity Risk** - Can we get out?
- **Crowding Risk** - Is the trade too popular?

### Sentiment Agent

**NOT:** Count bullish vs bearish

**INSTEAD:**
- **Narrative Lifecycle** - Where are we in the narrative?
- **Positioning Extremes** - Who's already in the trade?
- **Pain Trade Analysis** - What would hurt most people?
- **Information Asymmetry** - Who knows what we don't?
- **Reflexivity** - How does sentiment affect fundamentals?

---

## Learning Methods

All agents employ multiple learning methods:

### 1. Reinforcement Learning
Learn from trade outcomes. Reward correct predictions, penalize mistakes.

### 2. Bayesian Updating
Update beliefs with each new piece of evidence. Prior → Evidence → Posterior.

### 3. Adversarial Learning
Specifically learn from mistakes. Track mistake patterns. Avoid repeating.

### 4. Meta-Learning
Learn which approaches work in which contexts. "When does value work? When does momentum work?"

### 5. Active Learning
Seek out informative experiences. Don't just wait for data.

---

## Confidence Calibration

Agents continuously calibrate confidence:

```python
# If agent says "70% confident", it should be right 70% of the time
# Track calibration and adjust:

agent._confidence_adjustment = 0.85  # We're overconfident, scale down

# Every prediction's confidence is adjusted:
calibrated_confidence = raw_confidence * agent._confidence_adjustment
```

---

## The 30% Margin of Safety Rule

**CRITICAL:** No investment recommendation without 30% margin of safety.

This is enforced by:
1. RiskAgent validates all trade proposals
2. HoagsAgent approves only with MoS ≥ 30%
3. ComplianceAgent logs all violations

```python
# Example check
intrinsic_value = 100
current_price = 65

margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
# margin_of_safety = 0.35 = 35% ✓ Passes

if margin_of_safety < 0.30:
    raise ValueError("Insufficient margin of safety")
```

---

## Agent Coordination Architecture (ACA)

Agents can detect capability gaps and propose new agents:

```python
# Agent detects it needs help
gap = agent.detect_capability_gap({
    'type': 'analyze_convertible_bond',
    'required_capabilities': ['convert_arb', 'credit_analysis', 'equity_analysis']
})

if gap:
    # Propose new specialized agent
    agent.request_aca_creation(
        gap_description="Need convertible bond analysis capability",
        suggested_capabilities=['convert_arb', 'bond_floor_analysis', 'delta_hedging'],
        suggested_name="ConvertArbitrageAgent"
    )
```

---

## Summary: The ALC Difference

| Dimension | Basic Approach | ALC Approach |
|-----------|----------------|--------------|
| **Analysis** | Surface metrics | Deep, creative analysis |
| **Learning** | Static models | Continuous adaptation |
| **Thinking** | Linear | Multi-modal, contrarian |
| **Regime** | One-size-fits-all | Regime-adaptive |
| **Confidence** | Uncalibrated | Continuously calibrated |
| **Mistakes** | Repeated | Analyzed and avoided |
| **Edge** | Crowded signals | Differentiated insights |

---

## Implementation Checklist for New Agents

When creating a new agent:

- [ ] Define specialized, non-basic capabilities
- [ ] Implement creative analysis methods
- [ ] Enable continuous learning (`learning_enabled=True`)
- [ ] Set appropriate thinking modes
- [ ] Implement regime detection
- [ ] Add confidence calibration
- [ ] Track and learn from mistakes
- [ ] Support ACA capability gap detection
- [ ] Attribute all outputs to Tom Hogan / Alpha Loop Capital, LLC

---

*Agent Philosophy Document - ALC-Algo*
*Author: Tom Hogan | Alpha Loop Capital, LLC*

