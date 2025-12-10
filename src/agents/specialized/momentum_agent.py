"""
Momentum Agent - SPECIALIZED Creative Momentum Strategies
Author: Tom Hogan | Alpha Loop Capital, LLC

PHILOSOPHY: Basic price momentum DOESN'T WORK anymore - too crowded.

This agent implements CREATIVE momentum approaches:
1. EARNINGS MOMENTUM (the real signal, not price)
2. RELATIVE MOMENTUM (rotate to strongest, short weakest)
3. CROSS-ASSET MOMENTUM (signals from other markets)
4. VOLATILITY-ADJUSTED MOMENTUM (risk-normalize)
5. REGIME-ADAPTIVE MOMENTUM (what works changes)
6. REVERSAL DETECTION (momentum decay signals)
7. SECOND-DERIVATIVE (rate of change of change)
8. SENTIMENT MOMENTUM (institutional flow, options flow)

Basic momentum = Following price trends
Creative momentum = Understanding WHY prices are moving
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier, ThinkingMode, LearningMethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import math


class MomentumEdgeType(Enum):
    """Types of creative momentum edges - NOT basic trend following."""
    EARNINGS_MOMENTUM = "earnings_momentum"          # EPS revisions > price
    RELATIVE_MOMENTUM = "relative_momentum"          # Strongest vs weakest
    CROSS_ASSET = "cross_asset"                      # Signal from other markets
    VOLATILITY_ADJUSTED = "volatility_adjusted"      # Risk-normalized
    SECOND_DERIVATIVE = "second_derivative"          # Acceleration/deceleration
    SENTIMENT_FLOW = "sentiment_flow"                # Smart money flow
    REVERSAL_WARNING = "reversal_warning"            # Momentum decay
    REGIME_ADAPTIVE = "regime_adaptive"              # Adapts to market regime


@dataclass
class MomentumSignal:
    """A creative momentum signal - NOT just price trend."""
    ticker: str
    signal_type: MomentumEdgeType
    direction: str                        # "long" or "short"
    strength: float                       # -1 to +1
    confidence: float                     # 0 to 1
    supporting_factors: List[str]
    warning_signs: List[str]
    reversal_probability: float
    optimal_holding_period: str           # "days", "weeks", "months"
    volatility_adjusted: bool
    regime_alignment: float               # How well does this fit current regime


class MomentumAgent(BaseAgent):
    """
    SPECIALIZED Momentum Agent - Creative Momentum Strategies
    
    THIS IS NOT BASIC TREND FOLLOWING.
    
    Basic momentum (what doesn't work):
    - Simple 12-month price return
    - Moving average crossovers
    - RSI overbought/oversold
    - Following the trend blindly
    
    Creative momentum (what this agent does):
    
    1. EARNINGS MOMENTUM
       - EPS revision breadth and magnitude
       - Estimate dispersion changes
       - Surprise patterns
       - Quality of beats (revenue vs cost cuts)
    
    2. RELATIVE MOMENTUM
       - Not just "is it going up"
       - Is it going up MORE than alternatives?
       - Sector rotation signals
       - Style rotation signals
    
    3. CROSS-ASSET SIGNALS
       - What are credit spreads saying?
       - Commodity signals for equities
       - Currency signals
       - Volatility surface information
    
    4. SECOND DERIVATIVE
       - Not just momentum, but acceleration
       - Rate of change of rate of change
       - Inflection point detection
    
    5. MOMENTUM DECAY DETECTION
       - Recognize when momentum is exhausting
       - Volume divergences
       - Breadth divergences
       - Internal decay before price breaks
    
    6. REGIME ADAPTATION
       - Momentum works differently in different regimes
       - Risk-on: ride momentum
       - Risk-off: momentum crashes
       - Transition periods: most dangerous
    """
    
    def __init__(self, user_id: str = "TJH"):
        """Initialize with creative momentum capabilities."""
        super().__init__(
            name="MomentumAgent",
            tier=AgentTier.STRATEGY,
            capabilities=[
                # Creative momentum capabilities
                "earnings_momentum_analysis",
                "relative_strength_ranking",
                "cross_asset_signals",
                "volatility_adjusted_momentum",
                "second_derivative_analysis",
                "reversal_detection",
                "momentum_decay_detection",
                "regime_adaptive_momentum",
                # Flow analysis
                "institutional_flow_analysis",
                "options_flow_analysis",
                # Learning
                "regime_aware_positioning",
                "momentum_crash_prediction",
            ],
            user_id=user_id,
            aca_enabled=True,
            learning_enabled=True,
            thinking_modes=[
                ThinkingMode.REGIME_AWARE,      # Critical for momentum
                ThinkingMode.SECOND_ORDER,      # What's driving the momentum?
                ThinkingMode.CONTRARIAN,        # Know when to fade
                ThinkingMode.STRUCTURAL,        # Find structural momentum
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,   # Learn from trade outcomes
                LearningMethod.BAYESIAN,        # Update regime beliefs
                LearningMethod.ADVERSARIAL,     # Learn from momentum crashes
                LearningMethod.META,            # Learn which momentum type works when
            ]
        )
        
        # Track momentum type performance by regime
        self._regime_momentum_performance: Dict[str, Dict[str, float]] = {
            'risk_on': {'earnings': 0.6, 'price': 0.5, 'relative': 0.7},
            'risk_off': {'earnings': 0.3, 'price': 0.2, 'relative': 0.4},
            'normal': {'earnings': 0.5, 'price': 0.4, 'relative': 0.5},
            'crisis': {'earnings': 0.2, 'price': 0.1, 'relative': 0.3},
        }
        
        # Track reversal signals
        self._reversal_warnings: Dict[str, List[Dict[str, Any]]] = {}
        
        # Track momentum crashes for learning
        self._momentum_crash_history: List[Dict[str, Any]] = []
        
        self.logger.info("MomentumAgent initialized - CREATIVE momentum, NOT basic trend following")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process momentum analysis task creatively."""
        task_type = task.get('type', 'unknown')
        
        handlers = {
            'earnings_momentum': self._analyze_earnings_momentum,
            'relative_momentum': self._analyze_relative_momentum,
            'cross_asset': self._analyze_cross_asset_signals,
            'second_derivative': self._analyze_second_derivative,
            'reversal_check': self._check_reversal_signals,
            'regime_momentum': self._regime_adapted_momentum,
            'full_analysis': self._full_creative_analysis,
            'generate_signal': self._generate_creative_signal,
        }
        
        handler = handlers.get(task_type, self._full_creative_analysis)
        return handler(task)
    
    def _analyze_earnings_momentum(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Earnings momentum - the REAL momentum signal.
        
        Price momentum is just a shadow of earnings momentum.
        This is what institutional investors actually trade.
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        
        # EPS revision analysis
        eps_revisions = data.get('eps_revisions', {})
        current_eps = data.get('current_eps_estimate', 0)
        eps_3m_ago = data.get('eps_estimate_3m_ago', current_eps)
        eps_revision = (current_eps - eps_3m_ago) / abs(eps_3m_ago) if eps_3m_ago else 0
        
        # Revision breadth (how many analysts revising up vs down)
        revisions_up = eps_revisions.get('up', 0)
        revisions_down = eps_revisions.get('down', 0)
        total_revisions = revisions_up + revisions_down
        revision_breadth = (revisions_up - revisions_down) / total_revisions if total_revisions > 0 else 0
        
        # Surprise pattern
        last_4_surprises = data.get('last_4_surprises', [0, 0, 0, 0])
        beat_rate = sum(1 for s in last_4_surprises if s > 0) / len(last_4_surprises)
        avg_surprise_magnitude = sum(last_4_surprises) / len(last_4_surprises)
        
        # Quality of beats (revenue-driven vs cost-driven)
        revenue_beats = data.get('revenue_beat_rate', 0.5)
        quality_score = 0.7 * revenue_beats + 0.3 * beat_rate
        
        # Estimate dispersion (conviction of analysts)
        estimate_dispersion = data.get('eps_estimate_std', 0) / abs(current_eps) if current_eps else 0
        conviction_score = 1 - min(estimate_dispersion, 1)  # Lower dispersion = higher conviction
        
        # Calculate earnings momentum score
        earnings_momentum_score = (
            0.35 * eps_revision +
            0.25 * revision_breadth +
            0.20 * avg_surprise_magnitude +
            0.10 * quality_score +
            0.10 * conviction_score
        )
        
        signal_strength = max(-1, min(1, earnings_momentum_score * 5))  # Scale to -1 to 1
        
        return {
            'success': True,
            'ticker': ticker,
            'earnings_momentum': {
                'eps_revision': eps_revision,
                'revision_breadth': revision_breadth,
                'beat_rate': beat_rate,
                'avg_surprise': avg_surprise_magnitude,
                'quality_score': quality_score,
                'analyst_conviction': conviction_score,
                'composite_score': earnings_momentum_score,
            },
            'signal_strength': signal_strength,
            'direction': 'long' if signal_strength > 0.2 else 'short' if signal_strength < -0.2 else 'neutral',
            'edge_type': MomentumEdgeType.EARNINGS_MOMENTUM.value,
            'methodology': 'HOGAN MODEL - Earnings Momentum',
            'note': 'This is fundamental momentum, not price momentum',
        }
    
    def _analyze_relative_momentum(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Relative momentum - it's not enough to be going up.
        You need to be going up MORE than alternatives.
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        universe = task.get('universe', [])
        
        # Calculate relative strength
        ticker_return = data.get('return_6m', 0)
        sector_return = data.get('sector_return_6m', 0)
        market_return = data.get('market_return_6m', 0)
        
        # Relative to sector
        sector_relative = ticker_return - sector_return
        
        # Relative to market
        market_relative = ticker_return - market_return
        
        # Rank within universe
        universe_returns = {t: d.get('return_6m', 0) for t, d in universe}
        if universe_returns:
            sorted_returns = sorted(universe_returns.values(), reverse=True)
            rank = sorted_returns.index(ticker_return) + 1 if ticker_return in sorted_returns else len(sorted_returns) // 2
            percentile_rank = 1 - (rank / len(sorted_returns))
        else:
            percentile_rank = 0.5
        
        # Calculate relative momentum score
        rel_momentum_score = (
            0.30 * (percentile_rank - 0.5) * 2 +  # Scale to -1 to 1
            0.35 * sector_relative * 10 +          # Scale
            0.35 * market_relative * 10            # Scale
        )
        
        signal_strength = max(-1, min(1, rel_momentum_score))
        
        # Determine if this is a rotation candidate
        is_rotation_candidate = percentile_rank > 0.8 or percentile_rank < 0.2
        
        return {
            'success': True,
            'ticker': ticker,
            'relative_momentum': {
                'absolute_return': ticker_return,
                'sector_relative': sector_relative,
                'market_relative': market_relative,
                'universe_percentile': percentile_rank,
                'composite_score': rel_momentum_score,
            },
            'signal_strength': signal_strength,
            'direction': 'long' if signal_strength > 0.3 else 'short' if signal_strength < -0.3 else 'neutral',
            'rotation_candidate': is_rotation_candidate,
            'rotation_type': 'buy' if percentile_rank > 0.8 else 'sell' if percentile_rank < 0.2 else 'hold',
            'edge_type': MomentumEdgeType.RELATIVE_MOMENTUM.value,
            'methodology': 'HOGAN MODEL - Relative Momentum',
        }
    
    def _analyze_cross_asset_signals(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-asset signals - what are OTHER markets telling us?
        
        Markets are connected. Credit, FX, commodities, and vol
        often lead equity moves.
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        
        cross_asset_signals = {}
        
        # Credit spread signal
        credit_spread = data.get('credit_spread', 1.5)  # %
        credit_spread_change = data.get('credit_spread_change_3m', 0)
        if credit_spread_change < -0.2:
            cross_asset_signals['credit'] = {'direction': 'bullish', 'strength': 0.6, 'desc': 'Credit spreads tightening'}
        elif credit_spread_change > 0.3:
            cross_asset_signals['credit'] = {'direction': 'bearish', 'strength': 0.7, 'desc': 'Credit spreads widening - risk off'}
        
        # VIX signal
        vix = data.get('vix', 20)
        vix_change = data.get('vix_change_1m', 0)
        if vix < 15 and vix_change < -2:
            cross_asset_signals['volatility'] = {'direction': 'bullish', 'strength': 0.5, 'desc': 'Low and falling vol'}
        elif vix > 25:
            cross_asset_signals['volatility'] = {'direction': 'bearish', 'strength': 0.6, 'desc': 'Elevated volatility'}
        
        # Sector-specific commodity signals
        sector = data.get('sector', '')
        if sector == 'energy':
            oil_momentum = data.get('oil_momentum_3m', 0)
            cross_asset_signals['commodity'] = {
                'direction': 'bullish' if oil_momentum > 0.1 else 'bearish' if oil_momentum < -0.1 else 'neutral',
                'strength': abs(oil_momentum) * 5,
                'desc': f"Oil momentum: {oil_momentum:.1%}"
            }
        elif sector == 'materials':
            copper_momentum = data.get('copper_momentum_3m', 0)
            cross_asset_signals['commodity'] = {
                'direction': 'bullish' if copper_momentum > 0.1 else 'bearish' if copper_momentum < -0.1 else 'neutral',
                'strength': abs(copper_momentum) * 5,
                'desc': f"Copper momentum: {copper_momentum:.1%}"
            }
        
        # FX signal (for multinationals)
        if data.get('fx_sensitive', False):
            dxy_change = data.get('dxy_change_3m', 0)
            fx_impact = 'bullish' if dxy_change < -0.02 else 'bearish' if dxy_change > 0.02 else 'neutral'
            cross_asset_signals['fx'] = {
                'direction': fx_impact,
                'strength': abs(dxy_change) * 10,
                'desc': f"DXY change: {dxy_change:.1%}"
            }
        
        # Yield curve signal
        yield_curve = data.get('2s10s_spread', 0)
        if yield_curve < 0:
            cross_asset_signals['rates'] = {
                'direction': 'bearish',
                'strength': 0.5,
                'desc': 'Inverted yield curve - recession risk'
            }
        
        # Calculate composite signal
        bullish_count = sum(1 for s in cross_asset_signals.values() if s['direction'] == 'bullish')
        bearish_count = sum(1 for s in cross_asset_signals.values() if s['direction'] == 'bearish')
        total_signals = len(cross_asset_signals)
        
        if total_signals > 0:
            composite_direction = 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'mixed'
            composite_strength = abs(bullish_count - bearish_count) / total_signals
        else:
            composite_direction = 'neutral'
            composite_strength = 0
        
        return {
            'success': True,
            'ticker': ticker,
            'cross_asset_signals': cross_asset_signals,
            'composite_direction': composite_direction,
            'composite_strength': composite_strength,
            'signals_aligned': bullish_count == total_signals or bearish_count == total_signals,
            'edge_type': MomentumEdgeType.CROSS_ASSET.value,
            'methodology': 'HOGAN MODEL - Cross-Asset Momentum',
        }
    
    def _analyze_second_derivative(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Second derivative analysis - rate of change of rate of change.
        
        This catches inflection points BEFORE they become obvious.
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        
        # Price momentum (first derivative)
        return_1m = data.get('return_1m', 0)
        return_2m = data.get('return_2m', 0)
        return_3m = data.get('return_3m', 0)
        
        # Second derivative (acceleration/deceleration)
        acceleration_1m = return_1m - return_2m
        acceleration_2m = return_2m - return_3m
        
        # Third derivative (jerk - change in acceleration)
        jerk = acceleration_1m - acceleration_2m
        
        # Volume second derivative
        volume_1m = data.get('volume_change_1m', 0)
        volume_2m = data.get('volume_change_2m', 0)
        volume_acceleration = volume_1m - volume_2m
        
        # Breadth second derivative
        breadth_1w = data.get('breadth_change_1w', 0)
        breadth_2w = data.get('breadth_change_2w', 0)
        breadth_acceleration = breadth_1w - breadth_2w
        
        # Determine signal type
        if acceleration_1m > 0 and jerk > 0:
            signal_type = "accelerating_up"
            signal_strength = min(1, (acceleration_1m + jerk) * 10)
        elif acceleration_1m < 0 and jerk < 0:
            signal_type = "accelerating_down"
            signal_strength = max(-1, (acceleration_1m + jerk) * 10)
        elif acceleration_1m > 0 and jerk < 0:
            signal_type = "decelerating_up"  # Warning: momentum slowing
            signal_strength = acceleration_1m * 5  # Weaker signal
        elif acceleration_1m < 0 and jerk > 0:
            signal_type = "decelerating_down"  # Potential bottom
            signal_strength = acceleration_1m * 5
        else:
            signal_type = "neutral"
            signal_strength = 0
        
        # Inflection point detection
        is_inflection = (acceleration_1m * acceleration_2m < 0)  # Sign change
        
        return {
            'success': True,
            'ticker': ticker,
            'second_derivative': {
                'first_derivative': return_1m,
                'acceleration': acceleration_1m,
                'jerk': jerk,
                'volume_acceleration': volume_acceleration,
                'breadth_acceleration': breadth_acceleration,
            },
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'is_inflection_point': is_inflection,
            'direction': 'long' if signal_strength > 0.2 else 'short' if signal_strength < -0.2 else 'neutral',
            'edge_type': MomentumEdgeType.SECOND_DERIVATIVE.value,
            'methodology': 'HOGAN MODEL - Second Derivative Momentum',
            'note': 'Catches inflection points before they become obvious',
        }
    
    def _check_reversal_signals(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect momentum exhaustion / reversal signals.
        
        Momentum doesn't last forever. Detect when it's failing.
        """
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        
        reversal_warnings = []
        reversal_score = 0
        
        # 1. Volume divergence
        price_trend = data.get('price_trend', 'up')  # 'up' or 'down'
        volume_trend = data.get('volume_trend', 'up')
        if price_trend == 'up' and volume_trend == 'down':
            reversal_warnings.append("Volume divergence - price up on declining volume")
            reversal_score += 0.2
        
        # 2. Breadth divergence
        index_making_highs = data.get('index_making_highs', False)
        breadth_confirming = data.get('breadth_confirming', True)
        if index_making_highs and not breadth_confirming:
            reversal_warnings.append("Breadth divergence - highs not confirmed by breadth")
            reversal_score += 0.25
        
        # 3. Overbought/oversold extremes
        rsi = data.get('rsi', 50)
        if rsi > 80:
            reversal_warnings.append(f"RSI extreme overbought: {rsi}")
            reversal_score += 0.15
        elif rsi < 20:
            reversal_warnings.append(f"RSI extreme oversold: {rsi}")
            reversal_score += 0.15
        
        # 4. Sentiment extremes
        sentiment = data.get('sentiment_score', 0.5)
        if sentiment > 0.9:
            reversal_warnings.append("Sentiment extreme bullish - contrarian warning")
            reversal_score += 0.2
        elif sentiment < 0.1:
            reversal_warnings.append("Sentiment extreme bearish - potential bottom")
            reversal_score += 0.2
        
        # 5. Momentum decay (from second derivative)
        acceleration = data.get('momentum_acceleration', 0)
        if data.get('in_uptrend', False) and acceleration < -0.05:
            reversal_warnings.append("Momentum decaying in uptrend")
            reversal_score += 0.15
        
        # 6. Sector rotation away
        sector_flow = data.get('sector_fund_flow', 0)
        if sector_flow < -0.05:
            reversal_warnings.append("Money rotating out of sector")
            reversal_score += 0.1
        
        # Store warnings
        if ticker not in self._reversal_warnings:
            self._reversal_warnings[ticker] = []
        self._reversal_warnings[ticker].append({
            'timestamp': datetime.now(),
            'warnings': reversal_warnings,
            'score': reversal_score
        })
        
        return {
            'success': True,
            'ticker': ticker,
            'reversal_warnings': reversal_warnings,
            'reversal_probability': min(reversal_score, 1.0),
            'action_suggested': 'reduce_exposure' if reversal_score > 0.5 else 'monitor' if reversal_score > 0.25 else 'hold',
            'edge_type': MomentumEdgeType.REVERSAL_WARNING.value,
            'methodology': 'HOGAN MODEL - Reversal Detection',
        }
    
    def _regime_adapted_momentum(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt momentum strategy to current market regime.
        
        Momentum behaves VERY differently in different regimes.
        """
        data = task.get('data', {})
        
        # Detect current regime
        regime, regime_confidence = self.detect_regime_change(data.get('market_data', {}))
        
        # Get regime-specific recommendations
        regime_config = {
            'risk_on': {
                'momentum_weight': 0.8,
                'preferred_type': 'relative',
                'holding_period': 'weeks',
                'position_size': 'full',
                'note': 'Momentum works well - ride trends'
            },
            'risk_off': {
                'momentum_weight': 0.2,
                'preferred_type': 'earnings',
                'holding_period': 'days',
                'position_size': 'reduced',
                'note': 'Momentum dangerous - crashes happen'
            },
            'crisis': {
                'momentum_weight': 0.0,
                'preferred_type': 'reversal',
                'holding_period': 'none',
                'position_size': 'zero',
                'note': 'Do NOT trade momentum in crisis'
            },
            'normal': {
                'momentum_weight': 0.5,
                'preferred_type': 'earnings',
                'holding_period': 'weeks',
                'position_size': 'moderate',
                'note': 'Selective momentum - focus on quality'
            },
            'correlated': {
                'momentum_weight': 0.3,
                'preferred_type': 'cross_asset',
                'holding_period': 'days',
                'position_size': 'reduced',
                'note': 'Diversification doesn\'t work - be cautious'
            }
        }
        
        config = regime_config.get(regime, regime_config['normal'])
        
        # Check historical performance of momentum types in this regime
        historical_perf = self._regime_momentum_performance.get(regime, {})
        
        return {
            'success': True,
            'current_regime': regime,
            'regime_confidence': regime_confidence,
            'recommendation': config,
            'historical_performance': historical_perf,
            'should_trade_momentum': config['momentum_weight'] > 0.3,
            'edge_type': MomentumEdgeType.REGIME_ADAPTIVE.value,
            'methodology': 'HOGAN MODEL - Regime-Adaptive Momentum',
        }
    
    def _generate_creative_signal(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive creative momentum signal."""
        ticker = task.get('ticker', '')
        data = task.get('data', {})
        
        # Run all analyses
        earnings = self._analyze_earnings_momentum({'ticker': ticker, 'data': data})
        relative = self._analyze_relative_momentum({'ticker': ticker, 'data': data, 'universe': task.get('universe', {})})
        cross_asset = self._analyze_cross_asset_signals({'ticker': ticker, 'data': data})
        second_deriv = self._analyze_second_derivative({'ticker': ticker, 'data': data})
        reversal = self._check_reversal_signals({'ticker': ticker, 'data': data})
        regime = self._regime_adapted_momentum({'data': data})
        
        # Weight signals by regime
        regime_weight = regime['recommendation']['momentum_weight']
        
        # Composite signal
        composite_score = (
            0.30 * earnings['signal_strength'] +
            0.25 * relative['signal_strength'] +
            0.20 * (1 if cross_asset['composite_direction'] == 'bullish' else -1 if cross_asset['composite_direction'] == 'bearish' else 0) * cross_asset['composite_strength'] +
            0.15 * second_deriv['signal_strength'] +
            0.10 * (1 - reversal['reversal_probability']) * (1 if earnings['signal_strength'] > 0 else -1)
        )
        
        # Apply regime adjustment
        adjusted_score = composite_score * regime_weight
        
        # Determine final signal
        if adjusted_score > 0.3:
            direction = 'long'
        elif adjusted_score < -0.3:
            direction = 'short'
        else:
            direction = 'neutral'
        
        # Build signal object
        signal = MomentumSignal(
            ticker=ticker,
            signal_type=MomentumEdgeType[regime['recommendation']['preferred_type'].upper()] if regime['recommendation']['preferred_type'].upper() in [e.name for e in MomentumEdgeType] else MomentumEdgeType.EARNINGS_MOMENTUM,
            direction=direction,
            strength=adjusted_score,
            confidence=self._calibrated_confidence(abs(adjusted_score)),
            supporting_factors=[
                f"Earnings momentum: {earnings['signal_strength']:.2f}",
                f"Relative strength: {relative['signal_strength']:.2f}",
                f"Cross-asset: {cross_asset['composite_direction']}",
            ],
            warning_signs=reversal['reversal_warnings'],
            reversal_probability=reversal['reversal_probability'],
            optimal_holding_period=regime['recommendation']['holding_period'],
            volatility_adjusted=True,
            regime_alignment=regime_weight
        )
        
        # Learn from signal generation
        self.learn_from_outcome(
            prediction=f"Momentum signal for {ticker}: {direction}",
            actual="pending",
            confidence=signal.confidence,
            context={'ticker': ticker, 'regime': regime['current_regime']}
        )
        
        return {
            'success': True,
            'ticker': ticker,
            'signal': {
                'direction': signal.direction,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'type': signal.signal_type.value,
                'supporting_factors': signal.supporting_factors,
                'warning_signs': signal.warning_signs,
                'reversal_probability': signal.reversal_probability,
                'holding_period': signal.optimal_holding_period,
                'regime_alignment': signal.regime_alignment,
            },
            'component_signals': {
                'earnings': earnings['signal_strength'],
                'relative': relative['signal_strength'],
                'cross_asset': cross_asset['composite_direction'],
                'second_derivative': second_deriv['signal_type'],
            },
            'regime': regime['current_regime'],
            'should_trade': regime['should_trade_momentum'] and reversal['reversal_probability'] < 0.5,
            'methodology': 'HOGAN MODEL - Creative Momentum',
        }
    
    def _full_creative_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Full creative momentum analysis."""
        return self._generate_creative_signal(task)
    
    def get_capabilities(self) -> List[str]:
        """Return specialized capabilities."""
        return self.capabilities

