"""
================================================================================
VALUE AGENT - Creative Value Investing
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

PHILOSOPHY: Basic P/E and DCF DO NOT WORK.

This agent implements CREATIVE value approaches:
1. Normalized earnings power (through-cycle thinking)
2. Hidden assets and liabilities (off-balance sheet)
3. Sum-of-parts disconnects
4. Capital allocation quality (not just current returns)
5. Variant perception analysis (why is market wrong?)
6. Pre-mortem analysis (what would kill this thesis?)
7. Second-order effects of value realization
8. Behavioral edge exploitation (why others miss it)

Basic valuation = What the market already sees
Creative value = What the market DOESN'T see

Tier: STRATEGY (3)
Reports To: BOOKMAKER, HOAGS
Cluster: strategy

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train VALUE individually:
    python -m src.training.agent_training_utils --agent VALUE

    # Train with related strategy agents:
    python -m src.training.agent_training_utils --agents VALUE,MOMENTUM,BOOKMAKER

    # Cross-train: VALUE finds opportunities, AUTHOR documents:
    python -m src.training.agent_training_utils --cross-train "VALUE,RESEARCH_AGENT:AUTHOR:agent_trainer"

RUNNING THE AGENT:
    from src.agents.specialized.value_agent import ValueAgent

    value = ValueAgent()

    # Analyze variant perception
    result = value.process({
        "type": "analyze_variant",
        "ticker": "CCJ",
        "data": {...}
    })

    # Generate creative value thesis
    result = value.process({
        "type": "generate_thesis",
        "ticker": "CCJ",
        "data": {...}
    })

    # Pre-mortem analysis (kill box)
    result = value.process({
        "type": "pre_mortem",
        "ticker": "CCJ",
        "thesis": "Uranium supercycle thesis"
    })

================================================================================
"""

import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from dataclasses import dataclass
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier, ThinkingMode, LearningMethod


class ValueEdgeType(Enum):
    """Types of creative value edges - NOT basic metrics."""
    HIDDEN_ASSETS = "hidden_assets"              # Off-balance sheet value
    NORMALIZED_EARNINGS = "normalized_earnings"   # Through-cycle power
    SUM_OF_PARTS = "sum_of_parts"                # Conglomerate discount
    CAPITAL_ALLOCATION = "capital_allocation"     # Quality of reinvestment
    VARIANT_PERCEPTION = "variant_perception"     # Why market is wrong
    BEHAVIORAL = "behavioral"                     # Exploiting biases
    STRUCTURAL = "structural"                     # Forced sellers, etc.
    CATALYST = "catalyst"                         # What unlocks value


@dataclass
class ValueThesis:
    """A creative value thesis - NOT a simple DCF."""
    ticker: str
    thesis_type: ValueEdgeType
    variant_view: str                    # How we differ from consensus
    consensus_flaw: str                  # Why consensus is wrong
    hidden_value: Dict[str, float]       # Value not in the price
    kill_box: List[str]                  # What would kill this thesis
    catalysts: List[str]                 # What unlocks value
    conviction: float                    # 0-1
    margin_of_safety: float             # Required 30%+
    second_order_effects: List[str]     # Downstream implications
    behavioral_edge: str                 # Why others miss this


class ValueAgent(BaseAgent):
    """
    SPECIALIZED Value Agent - Creative Value Investing

    THIS IS NOT BASIC VALUE INVESTING.

    Basic value (what doesn't work):
    - Simple P/E comparisons
    - Trailing multiples
    - Standard DCF with consensus estimates
    - Screening for "cheap" stocks

    Creative value (what this agent does):

    1. NORMALIZED EARNINGS POWER
       - What can this business earn through a full cycle?
       - Strip out one-time items, COVID effects, cyclical peaks/troughs
       - Adjust for capital intensity changes

    2. HIDDEN ASSET ANALYSIS
       - Real estate carried at cost
       - NOLs and tax assets
       - Brand value not on books
       - Customer relationships
       - Strategic value to acquirer

    3. VARIANT PERCEPTION
       - What does the market believe?
       - Why might they be wrong?
       - What would change their mind?

    4. PRE-MORTEM / KILL BOX
       - Before investing, assume it failed
       - What went wrong?
       - Is this risk priced in?

    5. BEHAVIORAL EDGE
       - Why are others missing this?
       - Recency bias? Anchoring? Herding?
       - What's the pain trade?

    6. CAPITAL ALLOCATION QUALITY
       - Don't just look at current ROE
       - How has management allocated capital?
       - Track record of value creation
    """

    SUPPORTED_OPERATIONS = [
        "analyze_variant",           # Find variant perception
        "normalize_earnings",        # Through-cycle analysis
        "find_hidden_assets",        # Off-balance sheet value
        "sum_of_parts",              # Breakup value
        "pre_mortem",                # Kill box analysis
        "behavioral_edge",           # Why market is wrong
        "capital_allocation_score",  # Management quality
        "generate_thesis",           # Full creative thesis
    ]

    def __init__(self, user_id: str = "TJH"):
        """Initialize with creative value capabilities."""
        super().__init__(
            name="ValueAgent",
            tier=AgentTier.STRATEGY,
            capabilities=[
                # Creative value capabilities
                "normalized_earnings_analysis",
                "hidden_asset_detection",
                "variant_perception_analysis",
                "pre_mortem_analysis",
                "behavioral_edge_identification",
                "capital_allocation_scoring",
                "sum_of_parts_analysis",
                "catalyst_identification",
                # Learning capabilities
                "regime_adaptive_valuation",
                "contrarian_thesis_generation",
                "second_order_thinking",
            ],
            user_id=user_id,
            aca_enabled=True,
            learning_enabled=True,
            thinking_modes=[
                ThinkingMode.CONTRARIAN,        # Value is inherently contrarian
                ThinkingMode.SECOND_ORDER,      # What does market miss?
                ThinkingMode.BEHAVIORAL,        # Exploit biases
                ThinkingMode.STRUCTURAL,        # Find structural edges
                ThinkingMode.REGIME_AWARE,      # Value works differently in different regimes
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,   # Learn from outcomes
                LearningMethod.BAYESIAN,        # Update on new info
                LearningMethod.ADVERSARIAL,     # Learn from mistakes
                LearningMethod.META,            # Learn what value approaches work when
            ]
        )

        # Track thesis performance
        self._active_theses: Dict[str, ValueThesis] = {}
        self._thesis_outcomes: List[Dict[str, Any]] = []

        # Track what value approaches work in which regimes
        self._regime_value_performance: Dict[str, Dict[str, float]] = {}

        # Track behavioral patterns that create opportunities
        self._behavioral_patterns: Dict[str, int] = {
            'recency_bias': 0,
            'anchoring': 0,
            'herding': 0,
            'loss_aversion': 0,
            'overconfidence': 0,
        }

        self.logger.info("ValueAgent initialized - CREATIVE value investing, NOT basic P/E")

    @cached_property
    def _handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        """Cached handler dispatch table for O(1) lookup."""
        return {
            'analyze_variant': self._analyze_variant_perception,
            'normalize_earnings': self._normalize_earnings_power,
            'find_hidden_assets': self._find_hidden_assets,
            'sum_of_parts': self._sum_of_parts_analysis,
            'pre_mortem': self._pre_mortem_analysis,
            'behavioral_edge': self._identify_behavioral_edge,
            'capital_allocation_score': self._score_capital_allocation,
            'generate_thesis': self._generate_creative_thesis,
            'analyze': self._full_creative_analysis,
        }

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process value analysis task creatively."""
        task_type = task.get('type', 'unknown')
        handler = self._handlers.get(task_type, self._full_creative_analysis)
        return handler(task)

    def _analyze_variant_perception(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify variant perception - how do we differ from consensus?

        Key questions:
        - What does sell-side believe?
        - What does buy-side believe?
        - What do we believe differently?
        - WHY might we be right?
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})

        # Get consensus view
        consensus = self._extract_consensus(data)

        # Generate contrarian insight
        insight = self.think_contrarian(consensus, data)

        # Identify specific variant
        variant = {
            'consensus_view': consensus,
            'our_view': insight.contrarian_view,
            'key_differences': self._identify_perception_gaps(data),
            'why_we_could_be_right': self._support_variant(data),
            'what_would_prove_us_wrong': self._identify_disconfirmation(data),
            'catalysts_to_prove_thesis': self._identify_catalysts(data),
        }

        return {
            'success': True,
            'ticker': ticker,
            'variant_analysis': variant,
            'edge_type': ValueEdgeType.VARIANT_PERCEPTION.value,
            'confidence': insight.confidence,
            'methodology': 'HOGAN MODEL - Variant Perception',
        }

    def _normalize_earnings_power(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate normalized earnings power - NOT trailing P/E.

        This adjusts for:
        - Cycle position (peak/trough)
        - One-time items
        - COVID distortions
        - Capital intensity changes
        - Accounting choices
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})

        reported_eps = data.get('eps', 0)
        revenue = data.get('revenue', 0)

        # Adjustments (each one avoids basic trailing earnings)
        adjustments = {}

        # 1. Cycle adjustment
        cycle_position = data.get('cycle_position', 'mid')  # peak, mid, trough
        if cycle_position == 'peak':
            adjustments['cycle'] = -0.15  # Reduce by 15%
        elif cycle_position == 'trough':
            adjustments['cycle'] = 0.20  # Increase by 20%
        else:
            adjustments['cycle'] = 0

        # 2. One-time items
        one_time_items = data.get('one_time_items', 0)
        adjustments['one_time'] = -one_time_items / revenue if revenue > 0 else 0

        # 3. COVID distortion
        covid_impact = data.get('covid_benefit_headwind', 0)
        adjustments['covid'] = -covid_impact

        # 4. Capital intensity (maintenance vs growth capex)
        maintenance_capex = data.get('maintenance_capex', 0)
        reported_depreciation = data.get('depreciation', 0)
        adjustments['capex_intensity'] = (reported_depreciation - maintenance_capex) / revenue if revenue > 0 else 0

        # 5. Margin sustainability
        current_margin = data.get('operating_margin', 0)
        historical_avg_margin = data.get('historical_avg_margin', current_margin)
        adjustments['margin_sustainability'] = (historical_avg_margin - current_margin) * 0.5

        # Calculate normalized earnings
        total_adjustment = sum(adjustments.values())
        normalized_eps = reported_eps * (1 + total_adjustment)

        # Through-cycle earnings power
        earnings_power = {
            'reported_eps': reported_eps,
            'normalized_eps': normalized_eps,
            'adjustment_factor': 1 + total_adjustment,
            'adjustments': adjustments,
            'earnings_quality_score': self._score_earnings_quality(data),
            'cycle_position': cycle_position,
            'sustainable_eps_range': (normalized_eps * 0.9, normalized_eps * 1.1),
        }

        return {
            'success': True,
            'ticker': ticker,
            'normalized_earnings': earnings_power,
            'edge_type': ValueEdgeType.NORMALIZED_EARNINGS.value,
            'methodology': 'HOGAN MODEL - Normalized Earnings Power',
            'note': 'NOT simple trailing P/E - this is through-cycle earnings power',
        }

    def _find_hidden_assets(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find hidden assets not reflected in market price.

        Sources of hidden value:
        - Real estate at historical cost
        - NOLs and tax assets
        - Brand value
        - Customer relationships
        - Strategic value
        - Intellectual property
        - Investments at cost
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})

        market_cap = data.get('market_cap', 0)
        book_value = data.get('book_value', 0)

        hidden_assets = {}

        # 1. Real estate at cost
        re_book = data.get('real_estate_book', 0)
        re_market = data.get('real_estate_market_estimate', re_book * 1.5)
        if re_market > re_book:
            hidden_assets['real_estate'] = re_market - re_book

        # 2. NOLs
        nol_balance = data.get('nol_balance', 0)
        tax_rate = data.get('tax_rate', 0.21)
        hidden_assets['nol_value'] = nol_balance * tax_rate

        # 3. Investments at cost
        investments_book = data.get('investments_book', 0)
        investments_market = data.get('investments_market', investments_book)
        if investments_market > investments_book:
            hidden_assets['investments'] = investments_market - investments_book

        # 4. Brand value (estimate)
        brand_revenue = data.get('brand_premium_revenue', 0)
        hidden_assets['brand_value'] = brand_revenue * 3  # Rough multiple

        # 5. Customer relationships
        ltv = data.get('customer_ltv', 0)
        customer_count = data.get('customer_count', 0)
        hidden_assets['customer_value'] = ltv * customer_count * 0.5  # Haircut

        # 6. Strategic value premium
        strategic_premium = data.get('strategic_premium_estimate', 0)
        hidden_assets['strategic_value'] = strategic_premium

        total_hidden = sum(hidden_assets.values())

        return {
            'success': True,
            'ticker': ticker,
            'hidden_assets': hidden_assets,
            'total_hidden_value': total_hidden,
            'hidden_as_pct_market_cap': total_hidden / market_cap if market_cap > 0 else 0,
            'adjusted_nav': book_value + total_hidden,
            'edge_type': ValueEdgeType.HIDDEN_ASSETS.value,
            'methodology': 'HOGAN MODEL - Hidden Asset Analysis',
        }

    def _pre_mortem_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-mortem / Kill Box Analysis.

        Before investing, assume the investment FAILED.
        What went wrong?

        This prevents:
        - Confirmation bias
        - Overconfidence
        - Missing key risks
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        thesis = task.get('thesis', '')

        # Identify ways the thesis could fail
        kill_factors = []

        # Business risks
        if data.get('customer_concentration', 0) > 0.3:
            kill_factors.append("Customer concentration - loss of major customer")

        if data.get('debt_to_ebitda', 0) > 4:
            kill_factors.append("Leverage risk - covenant breach if EBITDA declines")

        if data.get('technology_disruption_risk', 'low') == 'high':
            kill_factors.append("Technology disruption - business model obsolescence")

        # Valuation risks
        if data.get('multiple_expansion_assumed', False):
            kill_factors.append("Multiple expansion didn't happen - value trap")

        # Management risks
        if data.get('management_quality_score', 5) < 5:
            kill_factors.append("Management destroyed value through poor capital allocation")

        # Macro risks
        kill_factors.append("Regime change - value underperforms in risk-off")
        kill_factors.append("Opportunity cost - better opportunities emerged")

        # Thesis-specific risks
        if thesis:
            thesis_risks = self._identify_thesis_specific_risks(thesis, data)
            kill_factors.extend(thesis_risks)

        # Score each risk
        risk_scores = {factor: self._score_risk(factor, data) for factor in kill_factors}

        return {
            'success': True,
            'ticker': ticker,
            'kill_box': kill_factors,
            'risk_scores': risk_scores,
            'highest_risk': max(risk_scores.items(), key=lambda x: x[1]) if risk_scores else None,
            'thesis_survival_probability': 1 - max(risk_scores.values()) if risk_scores else 0.5,
            'methodology': 'HOGAN MODEL - Pre-Mortem Analysis',
            'note': 'Assume failure first. Why did it fail?',
        }

    def _identify_behavioral_edge(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify behavioral edge - why are other investors wrong?

        Behavioral biases that create value opportunities:
        - Recency bias (overweight recent events)
        - Anchoring (stuck on old price)
        - Herding (following crowd)
        - Loss aversion (selling winners too early)
        - Overconfidence (in growth estimates)
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})

        behavioral_edges = {}

        # Recency bias detection
        recent_news_sentiment = data.get('recent_news_sentiment', 'neutral')
        fundamental_trend = data.get('fundamental_trend', 'neutral')
        if recent_news_sentiment != fundamental_trend:
            behavioral_edges['recency_bias'] = {
                'description': f"Market focused on {recent_news_sentiment} news, missing {fundamental_trend} fundamentals",
                'edge_strength': 0.6
            }
            self._behavioral_patterns['recency_bias'] += 1

        # Anchoring detection
        current_price = data.get('price', 0)
        year_ago_price = data.get('price_1y_ago', current_price)
        intrinsic_value = data.get('intrinsic_value', current_price)
        if abs(current_price - year_ago_price) / year_ago_price < 0.1:  # Price hasn't moved
            if abs(intrinsic_value - current_price) / current_price > 0.2:
                behavioral_edges['anchoring'] = {
                    'description': "Market anchored to old price despite changed fundamentals",
                    'edge_strength': 0.5
                }
                self._behavioral_patterns['anchoring'] += 1

        # Herding detection
        analyst_rating_consensus = data.get('analyst_consensus', 'hold')
        institutional_ownership_change = data.get('inst_ownership_change', 0)
        if analyst_rating_consensus == 'sell' and institutional_ownership_change < -0.05:
            behavioral_edges['herding'] = {
                'description': "Institutions herding out, may create opportunity",
                'edge_strength': 0.55
            }
            self._behavioral_patterns['herding'] += 1

        # Pain trade detection
        short_interest = data.get('short_interest', 0)
        if short_interest > 0.15:  # >15% short
            behavioral_edges['pain_trade'] = {
                'description': f"High short interest ({short_interest:.0%}) - pain trade potential",
                'edge_strength': 0.4
            }

        return {
            'success': True,
            'ticker': ticker,
            'behavioral_edges': behavioral_edges,
            'total_edge_score': sum(e['edge_strength'] for e in behavioral_edges.values()),
            'dominant_bias': max(behavioral_edges.items(), key=lambda x: x[1]['edge_strength'])[0] if behavioral_edges else None,
            'edge_type': ValueEdgeType.BEHAVIORAL.value,
            'methodology': 'HOGAN MODEL - Behavioral Edge Analysis',
        }

    def _generate_creative_thesis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete creative value thesis.

        NOT a simple "it's cheap" thesis.
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})

        # Run all creative analyses
        variant = self._analyze_variant_perception({'ticker': ticker, 'data': data})
        normalized = self._normalize_earnings_power({'ticker': ticker, 'data': data})
        hidden = self._find_hidden_assets({'ticker': ticker, 'data': data})
        kill_box = self._pre_mortem_analysis({'ticker': ticker, 'data': data})
        behavioral = self._identify_behavioral_edge({'ticker': ticker, 'data': data})

        # Calculate margin of safety using normalized earnings
        price = data.get('price', 0)
        normalized_eps = normalized['normalized_earnings']['normalized_eps']
        fair_multiple = data.get('fair_pe_multiple', 15)
        intrinsic_value = normalized_eps * fair_multiple

        # Add hidden assets
        shares = data.get('shares_outstanding', 1)
        intrinsic_value += hidden['total_hidden_value'] / shares

        margin_of_safety = (intrinsic_value - price) / intrinsic_value if intrinsic_value > 0 else 0

        # Build thesis
        thesis = ValueThesis(
            ticker=ticker,
            thesis_type=self._determine_primary_edge([variant, normalized, hidden, behavioral]),
            variant_view=variant['variant_analysis']['our_view'],
            consensus_flaw=variant['variant_analysis']['key_differences'][0] if variant['variant_analysis']['key_differences'] else "Market extrapolating current state",
            hidden_value=hidden['hidden_assets'],
            kill_box=kill_box['kill_box'][:5],
            catalysts=variant['variant_analysis'].get('catalysts_to_prove_thesis', []),
            conviction=self._calculate_conviction([variant, normalized, kill_box, behavioral]),
            margin_of_safety=margin_of_safety,
            second_order_effects=self._identify_second_order_effects(ticker, data),
            behavioral_edge=behavioral['dominant_bias'] or "No dominant behavioral edge"
        )

        # Store thesis
        self._active_theses[ticker] = thesis

        # Learn from thesis generation
        self.learn_from_outcome(
            prediction=f"Thesis generated for {ticker}",
            actual="pending",
            confidence=thesis.conviction,
            context={'ticker': ticker, 'margin_of_safety': margin_of_safety}
        )

        return {
            'success': True,
            'ticker': ticker,
            'thesis': {
                'type': thesis.thesis_type.value,
                'variant_view': thesis.variant_view,
                'consensus_flaw': thesis.consensus_flaw,
                'hidden_value': thesis.hidden_value,
                'kill_box': thesis.kill_box,
                'catalysts': thesis.catalysts,
                'conviction': thesis.conviction,
                'margin_of_safety': thesis.margin_of_safety,
                'behavioral_edge': thesis.behavioral_edge,
                'second_order_effects': thesis.second_order_effects,
            },
            'intrinsic_value': intrinsic_value,
            'current_price': price,
            'upside': (intrinsic_value - price) / price if price > 0 else 0,
            'meets_30pct_mos': margin_of_safety >= 0.30,
            'methodology': 'HOGAN MODEL - Creative Value Thesis',
            'note': 'This is NOT basic value investing',
        }

    def _full_creative_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Full creative value analysis."""
        return self._generate_creative_thesis(task)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_consensus(self, data: Dict[str, Any]) -> str:
        """Extract consensus view from data."""
        rating = data.get('analyst_consensus', 'hold')
        pt = data.get('average_price_target', 0)
        return f"Consensus: {rating.upper()}, PT: ${pt:.2f}"

    def _identify_perception_gaps(self, data: Dict[str, Any]) -> List[str]:
        """Identify gaps in market perception."""
        gaps = []
        if data.get('earnings_quality_concerns', False):
            gaps.append("Market may not see earnings quality issues")
        if data.get('hidden_assets_significant', False):
            gaps.append("Hidden assets not in consensus valuation")
        if data.get('cycle_position', 'mid') in ['peak', 'trough']:
            gaps.append("Market may be extrapolating current cycle position")
        if not gaps:
            gaps.append("Market focused on near-term, missing long-term")
        return gaps

    def _support_variant(self, data: Dict[str, Any]) -> List[str]:
        """Find support for variant view."""
        support = []
        if data.get('insider_buying', False):
            support.append("Insiders buying")
        if data.get('historical_accuracy', 0) > 0.6:
            support.append("Our historical analysis accuracy supports confidence")
        support.append("Differentiated research process")
        return support

    def _identify_disconfirmation(self, data: Dict[str, Any]) -> List[str]:
        """Identify what would prove us wrong."""
        disconfirm = []
        disconfirm.append("Fundamentals deteriorate for 2+ quarters")
        disconfirm.append("Management capital allocation worsens")
        disconfirm.append("Catalyst doesn't materialize in expected timeframe")
        return disconfirm

    def _identify_catalysts(self, data: Dict[str, Any]) -> List[str]:
        """Identify potential catalysts."""
        catalysts = []
        if data.get('strategic_review_possible', False):
            catalysts.append("Strategic review/sale")
        if data.get('management_change_likely', False):
            catalysts.append("Management change")
        catalysts.append("Earnings beat demonstrating thesis")
        catalysts.append("Multiple expansion as perception shifts")
        return catalysts

    def _score_earnings_quality(self, data: Dict[str, Any]) -> float:
        """Score earnings quality 0-1."""
        score = 0.5
        if data.get('cash_conversion', 0) > 0.9:
            score += 0.2
        if data.get('accruals_ratio', 0) < 0.05:
            score += 0.1
        if data.get('revenue_quality', 'medium') == 'high':
            score += 0.1
        return min(score, 1.0)

    def _identify_thesis_specific_risks(self, thesis: str, data: Dict[str, Any]) -> List[str]:
        """Identify risks specific to the thesis."""
        risks = []
        if 'turnaround' in thesis.lower():
            risks.append("Turnaround takes longer than expected")
        if 'growth' in thesis.lower():
            risks.append("Growth fails to materialize")
        if 'margin' in thesis.lower():
            risks.append("Margins don't expand as expected")
        return risks

    def _score_risk(self, risk: str, data: Dict[str, Any]) -> float:
        """Score a risk factor 0-1."""
        base_score = 0.3
        if 'leverage' in risk.lower() and data.get('debt_to_ebitda', 0) > 3:
            return 0.7
        if 'customer' in risk.lower() and data.get('customer_concentration', 0) > 0.4:
            return 0.6
        return base_score

    def _determine_primary_edge(self, analyses: List[Dict]) -> ValueEdgeType:
        """Determine primary value edge type."""
        # Simple heuristic - can be made more sophisticated
        return ValueEdgeType.VARIANT_PERCEPTION

    def _calculate_conviction(self, analyses: List[Dict]) -> float:
        """Calculate overall conviction."""
        # Aggregate confidence from analyses
        confidences = []
        for analysis in analyses:
            if 'confidence' in analysis:
                confidences.append(analysis['confidence'])

        if not confidences:
            return 0.5

        base_conviction = sum(confidences) / len(confidences)
        return self._calibrated_confidence(base_conviction)

    def _identify_second_order_effects(self, ticker: str, data: Dict[str, Any]) -> List[str]:
        """Identify second-order effects of value realization."""
        effects = []
        effects.append("If thesis works, similar companies may re-rate")
        effects.append("Success validates creative value approach")
        effects.append("May attract acquirer attention")
        return effects

    def _score_capital_allocation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Score management's capital allocation."""
        data = task.get('data', {})

        scores = {
            'historical_roic': data.get('historical_roic', 0.10),
            'acquisition_track_record': data.get('acquisition_success_rate', 0.5),
            'dividend_consistency': data.get('dividend_growth_years', 0) / 25,
            'buyback_timing': data.get('buyback_timing_score', 0.5),
            'reinvestment_rate': data.get('reinvestment_roic', 0.10),
        }

        overall = sum(scores.values()) / len(scores)

        return {
            'success': True,
            'capital_allocation_scores': scores,
            'overall_score': overall,
            'grade': 'A' if overall > 0.7 else 'B' if overall > 0.5 else 'C' if overall > 0.3 else 'F',
            'methodology': 'HOGAN MODEL - Capital Allocation Analysis',
        }

    def _sum_of_parts_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Sum of parts valuation for conglomerates."""
        ticker = task.get('ticker', '')
        data = task.get('data', {})

        segments = data.get('segments', {})
        segment_values = {}

        for segment_name, segment_data in segments.items():
            ebitda = segment_data.get('ebitda', 0)
            appropriate_multiple = segment_data.get('peer_multiple', 10)
            segment_values[segment_name] = ebitda * appropriate_multiple

        total_sotp_value = sum(segment_values.values())
        net_debt = data.get('net_debt', 0)
        equity_value = total_sotp_value - net_debt

        market_cap = data.get('market_cap', equity_value)
        discount = (equity_value - market_cap) / equity_value if equity_value > 0 else 0

        return {
            'success': True,
            'ticker': ticker,
            'segment_values': segment_values,
            'total_enterprise_value': total_sotp_value,
            'net_debt': net_debt,
            'equity_value': equity_value,
            'market_cap': market_cap,
            'conglomerate_discount': discount,
            'edge_type': ValueEdgeType.SUM_OF_PARTS.value,
            'methodology': 'HOGAN MODEL - Sum of Parts Analysis',
        }

    def get_capabilities(self) -> List[str]:
        """Return specialized capabilities."""
        return self.capabilities

