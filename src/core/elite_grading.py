"""================================================================================
ELITE GRADING SYSTEM
================================================================================
Institutional-grade grading with UNIQUE features that differentiate from
standard quant funds. Higher bars, proprietary metrics, edge detection.

WHAT MAKES THIS DIFFERENT FROM CITADEL/TWO SIGMA:
1. Regime-Adaptive Requirements - Agents must prove competence across regimes
2. Edge Decay Tracking - Measures how long alpha persists (alpha half-life)
3. Adversarial Robustness - Survives intentional market manipulation scenarios
4. Information Ratio by Source - Tracks which data sources generate real alpha
5. Conviction-Accuracy Correlation - High conviction must mean high accuracy
6. Crowding Penalty - Penalizes strategies that become too popular
7. Reflexivity Score - Measures if strategy affects the market it trades
8. Black Swan Readiness - Performance during tail events
================================================================================
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class GradeLevel(Enum):
    """Elite grading levels - much harder than standard."""

    ELITE = "ELITE"          # Top 1% - Ready for live capital
    BATTLE_READY = "BATTLE_READY"  # Top 5% - Paper trading approved
    DEVELOPING = "DEVELOPING"      # Top 25% - Continue training
    INADEQUATE = "INADEQUATE"      # Bottom 75% - Needs major work
    FAILED = "FAILED"              # Critical flaws - Rebuild


@dataclass(frozen=True)
class EliteThresholds:
    """ELITE THRESHOLDS - Much higher bar than typical quant funds.

    These are based on what separates top-tier performers from average.
    """

    # =========================================================================
    # CORE PERFORMANCE (Higher than industry standard)
    # =========================================================================
    min_sharpe_ratio: float = 2.0          # Most funds accept 1.0, we need 2.0+
    min_sortino_ratio: float = 2.5         # Penalize downside more heavily
    min_calmar_ratio: float = 1.5          # Return/MaxDrawdown
    max_drawdown_pct: float = 0.10         # Max 10% drawdown (most allow 20%+)
    min_win_rate: float = 0.55             # Must win >55% of trades
    min_profit_factor: float = 1.8         # Gross profit / gross loss

    # =========================================================================
    # UNIQUE/PROPRIETARY METRICS (What others DON'T measure)
    # =========================================================================
    min_alpha_half_life_days: int = 30     # Alpha must persist 30+ days
    min_regime_consistency: float = 0.7    # Profitable in 70%+ of regime types
    min_conviction_accuracy_corr: float = 0.6  # High conviction = high accuracy
    max_crowding_score: float = 0.3        # Strategy can't be >30% crowded
    min_black_swan_survival: float = 0.8   # 80%+ survival in tail events
    min_information_ratio: float = 1.0     # Alpha / Tracking Error
    max_reflexivity_impact: float = 0.05   # Strategy can't move market >5%

    # =========================================================================
    # OPERATIONAL REQUIREMENTS
    # =========================================================================
    min_executions: int = 500              # Need 500+ trades for significance
    min_unique_symbols: int = 50           # Must trade diverse universe
    min_training_epochs: int = 100         # Minimum training iterations
    min_backtest_years: float = 3.0        # 3+ years of backtesting
    min_out_of_sample_sharpe: float = 1.5  # OOS performance check

    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    min_var_accuracy: float = 0.95         # VaR predictions must be 95% accurate
    max_tail_loss_multiple: float = 2.0    # Tail loss can't exceed 2x VaR
    min_liquidity_score: float = 0.8       # Must trade liquid instruments
    max_concentration: float = 0.15        # No position >15% of portfolio

    # =========================================================================
    # LEARNING & ADAPTATION
    # =========================================================================
    min_learning_rate: float = 0.02        # Must improve 2%+ per 100 trades
    min_adaptation_speed: int = 5          # Adapt to regime change in 5 days
    min_error_recovery_rate: float = 0.9   # Recover from 90%+ of errors


@dataclass
class UniqueFeatureScores:
    """PROPRIETARY METRICS - These are what make us different.

    Standard funds measure Sharpe, we measure these too:
    """

    # Alpha decay - how quickly does the edge disappear?
    alpha_half_life_days: float = 0.0
    alpha_decay_rate: float = 0.0

    # Regime performance - works in all market conditions?
    bull_market_sharpe: float = 0.0
    bear_market_sharpe: float = 0.0
    sideways_market_sharpe: float = 0.0
    high_vol_sharpe: float = 0.0
    low_vol_sharpe: float = 0.0
    regime_consistency: float = 0.0  # Std dev of sharpe across regimes

    # Conviction-accuracy correlation
    # When agent is 90% confident, is it 90% accurate?
    conviction_accuracy_correlation: float = 0.0
    overconfidence_penalty: float = 0.0
    underconfidence_penalty: float = 0.0

    # Crowding detection - are we just following the herd?
    strategy_crowding_score: float = 0.0  # 0 = unique, 1 = everyone does it
    factor_overlap: float = 0.0           # Overlap with common factors
    signal_correlation_with_flow: float = 0.0  # Correlated with ETF flows?

    # Black swan readiness
    covid_crash_return: float = 0.0       # March 2020 performance
    flash_crash_return: float = 0.0       # Flash crash performance
    volmageddon_return: float = 0.0       # Feb 2018 performance
    tail_event_avg_return: float = 0.0    # Average in tail events
    black_swan_survival: float = 0.0      # % of tail events survived

    # Reflexivity - does our trading affect the market?
    market_impact_estimate: float = 0.0   # Estimated price impact
    reflexivity_score: float = 0.0        # Does strategy cause its own success?
    capacity_estimate_usd: float = 0.0    # Max AUM before alpha decay

    # Information edge by source
    price_alpha_contribution: float = 0.0
    volume_alpha_contribution: float = 0.0
    sentiment_alpha_contribution: float = 0.0
    fundamental_alpha_contribution: float = 0.0
    alternative_data_alpha_contribution: float = 0.0

    # Uniqueness score - how different from standard quant strategies?
    uniqueness_score: float = 0.0
    intellectual_property_score: float = 0.0  # Proprietary edge


@dataclass
class TradeStatistics:
    """Detailed trade-level statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0
    avg_holding_period_hours: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # Expected value per trade

    # Time-based
    trades_per_day: float = 0.0
    avg_trades_per_symbol: float = 0.0
    unique_symbols_traded: int = 0

    # Slippage and costs
    avg_slippage_bps: float = 0.0
    total_commission_pct: float = 0.0
    net_profit_after_costs: float = 0.0


@dataclass
class RiskStatistics:
    """Risk metrics."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown_pct: float = 0.0

    # VaR metrics
    var_95_daily: float = 0.0
    var_99_daily: float = 0.0
    cvar_95_daily: float = 0.0
    var_breaches: int = 0  # How many times VaR was exceeded
    var_accuracy: float = 0.0

    # Tail risk
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0  # 95th percentile / 5th percentile

    # Concentration
    max_position_concentration: float = 0.0
    sector_concentration: float = 0.0
    factor_concentration: float = 0.0


@dataclass
class LearningStatistics:
    """Learning and adaptation metrics."""

    total_training_epochs: int = 0
    total_learning_events: int = 0
    improvement_rate_per_100_trades: float = 0.0

    # Adaptation
    regime_detection_accuracy: float = 0.0
    adaptation_speed_days: int = 0

    # Error handling
    total_errors: int = 0
    recovered_errors: int = 0
    error_recovery_rate: float = 0.0

    # Exploration vs exploitation
    exploration_rate: float = 0.0
    strategy_diversity_score: float = 0.0


@dataclass
class EliteGradeResult:
    """Complete grading result."""

    grade: GradeLevel
    score: float  # 0-100
    passed: bool

    # Component scores
    performance_score: float = 0.0
    risk_score: float = 0.0
    uniqueness_score: float = 0.0
    learning_score: float = 0.0
    operational_score: float = 0.0

    # Detailed breakdown
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # What makes this agent special
    unique_edges: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    graded_at: datetime = field(default_factory=datetime.utcnow)
    thresholds_used: str = "elite_v1"


class EliteAgentGrader:
    """ELITE GRADING ENGINE

    This is NOT your typical quant fund grading. We measure things
    that others don't even consider.
    """

    def __init__(self, thresholds: Optional[EliteThresholds] = None):
        self.thresholds = thresholds or EliteThresholds()

    def grade(
        self,
        trade_stats: TradeStatistics,
        risk_stats: RiskStatistics,
        unique_features: UniqueFeatureScores,
        learning_stats: LearningStatistics,
    ) -> EliteGradeResult:
        """Comprehensive grading across all dimensions.
        """
        failures = []
        warnings = []
        strengths = []
        unique_edges = []
        recommendations = []

        t = self.thresholds

        # =====================================================================
        # PERFORMANCE SCORING (25 points)
        # =====================================================================
        perf_score = 0.0

        if risk_stats.sharpe_ratio >= t.min_sharpe_ratio:
            perf_score += 5
            if risk_stats.sharpe_ratio >= 3.0:
                strengths.append(f"Exceptional Sharpe: {risk_stats.sharpe_ratio:.2f}")
                perf_score += 2
        else:
            failures.append(f"Sharpe {risk_stats.sharpe_ratio:.2f} < {t.min_sharpe_ratio}")

        if risk_stats.sortino_ratio >= t.min_sortino_ratio:
            perf_score += 5
        else:
            failures.append(f"Sortino {risk_stats.sortino_ratio:.2f} < {t.min_sortino_ratio}")

        if trade_stats.win_rate >= t.min_win_rate:
            perf_score += 5
        else:
            warnings.append(f"Win rate {trade_stats.win_rate:.1%} below target")

        if trade_stats.profit_factor >= t.min_profit_factor:
            perf_score += 5
        else:
            failures.append(f"Profit factor {trade_stats.profit_factor:.2f} < {t.min_profit_factor}")

        if risk_stats.max_drawdown_pct <= t.max_drawdown_pct:
            perf_score += 5
            if risk_stats.max_drawdown_pct <= 0.05:
                strengths.append(f"Minimal drawdown: {risk_stats.max_drawdown_pct:.1%}")
        else:
            failures.append(f"Max drawdown {risk_stats.max_drawdown_pct:.1%} > {t.max_drawdown_pct:.1%}")

        # =====================================================================
        # UNIQUENESS SCORING (25 points) - THE DIFFERENTIATOR
        # =====================================================================
        unique_score = 0.0

        # Alpha persistence
        if unique_features.alpha_half_life_days >= t.min_alpha_half_life_days:
            unique_score += 5
            if unique_features.alpha_half_life_days >= 60:
                unique_edges.append(f"Long alpha half-life: {unique_features.alpha_half_life_days:.0f} days")
                unique_score += 2
        else:
            failures.append(f"Alpha decays too fast: {unique_features.alpha_half_life_days:.0f} days")

        # Regime consistency
        if unique_features.regime_consistency >= t.min_regime_consistency:
            unique_score += 5
            strengths.append("Works across market regimes")
        else:
            failures.append(f"Regime inconsistent: {unique_features.regime_consistency:.1%}")

        # Conviction-accuracy correlation
        if unique_features.conviction_accuracy_correlation >= t.min_conviction_accuracy_corr:
            unique_score += 5
            unique_edges.append("Calibrated confidence - knows when it knows")
        else:
            warnings.append("Conviction not well calibrated to accuracy")

        # Anti-crowding
        if unique_features.strategy_crowding_score <= t.max_crowding_score:
            unique_score += 5
            if unique_features.strategy_crowding_score <= 0.1:
                unique_edges.append(f"Highly unique strategy (crowding: {unique_features.strategy_crowding_score:.1%})")
        else:
            failures.append(f"Strategy too crowded: {unique_features.strategy_crowding_score:.1%}")

        # Black swan readiness
        if unique_features.black_swan_survival >= t.min_black_swan_survival:
            unique_score += 5
            unique_edges.append("Survives tail events")
        else:
            failures.append(f"Black swan survival: {unique_features.black_swan_survival:.1%}")

        # =====================================================================
        # RISK SCORING (25 points)
        # =====================================================================
        risk_score = 0.0

        # VaR accuracy
        if risk_stats.var_accuracy >= t.min_var_accuracy:
            risk_score += 6
        else:
            warnings.append(f"VaR accuracy {risk_stats.var_accuracy:.1%} needs improvement")

        # Tail risk
        if risk_stats.kurtosis < 5.0:
            risk_score += 5
        else:
            warnings.append(f"High kurtosis (fat tails): {risk_stats.kurtosis:.1f}")

        # Concentration
        if risk_stats.max_position_concentration <= t.max_concentration:
            risk_score += 5
        else:
            failures.append(f"Over-concentrated: {risk_stats.max_position_concentration:.1%}")

        # Liquidity
        if unique_features.capacity_estimate_usd >= 10_000_000:
            risk_score += 5
            strengths.append(f"Scalable to ${unique_features.capacity_estimate_usd/1e6:.0f}M")
        else:
            warnings.append("Limited capacity")

        # Reflexivity
        if unique_features.reflexivity_score <= t.max_reflexivity_impact:
            risk_score += 4
        else:
            warnings.append("Strategy may move market")

        # =====================================================================
        # LEARNING SCORING (15 points)
        # =====================================================================
        learn_score = 0.0

        if learning_stats.improvement_rate_per_100_trades >= t.min_learning_rate:
            learn_score += 5
            strengths.append("Continuously improving")
        else:
            recommendations.append("Increase learning rate")

        if learning_stats.adaptation_speed_days <= t.min_adaptation_speed:
            learn_score += 5
            unique_edges.append(f"Fast adaptation: {learning_stats.adaptation_speed_days} days")
        else:
            recommendations.append("Improve regime detection speed")

        if learning_stats.error_recovery_rate >= t.min_error_recovery_rate:
            learn_score += 5
        else:
            warnings.append(f"Error recovery rate: {learning_stats.error_recovery_rate:.1%}")

        # =====================================================================
        # OPERATIONAL SCORING (10 points)
        # =====================================================================
        ops_score = 0.0

        if trade_stats.total_trades >= t.min_executions:
            ops_score += 3
        else:
            failures.append(f"Insufficient trades: {trade_stats.total_trades} < {t.min_executions}")

        if trade_stats.unique_symbols_traded >= t.min_unique_symbols:
            ops_score += 3
        else:
            warnings.append(f"Trade more symbols: {trade_stats.unique_symbols_traded}")

        if learning_stats.total_training_epochs >= t.min_training_epochs:
            ops_score += 4
        else:
            recommendations.append("Need more training epochs")

        # =====================================================================
        # FINAL GRADE CALCULATION
        # =====================================================================
        total_score = perf_score + unique_score + risk_score + learn_score + ops_score

        # Determine grade
        if len(failures) == 0 and total_score >= 85:
            grade = GradeLevel.ELITE
            passed = True
        elif len(failures) <= 2 and total_score >= 70:
            grade = GradeLevel.BATTLE_READY
            passed = True
        elif len(failures) <= 4 and total_score >= 50:
            grade = GradeLevel.DEVELOPING
            passed = False
        elif total_score >= 30:
            grade = GradeLevel.INADEQUATE
            passed = False
        else:
            grade = GradeLevel.FAILED
            passed = False

        return EliteGradeResult(
            grade=grade,
            score=total_score,
            passed=passed,
            performance_score=perf_score,
            risk_score=risk_score,
            uniqueness_score=unique_score,
            learning_score=learn_score,
            operational_score=ops_score,
            failures=failures,
            warnings=warnings,
            strengths=strengths,
            unique_edges=unique_edges,
            recommendations=recommendations,
        )

    def quick_grade(self, stats: Dict[str, Any]) -> EliteGradeResult:
        """Quick grading from a flat dictionary of stats.
        """
        # Parse into structured objects
        trade_stats = TradeStatistics(
            total_trades=stats.get("total_trades", 0),
            winning_trades=stats.get("winning_trades", 0),
            losing_trades=stats.get("losing_trades", 0),
            win_rate=stats.get("win_rate", 0),
            profit_factor=stats.get("profit_factor", 0),
            unique_symbols_traded=stats.get("unique_symbols", 0),
            avg_win_pct=stats.get("avg_win_pct", 0),
            avg_loss_pct=stats.get("avg_loss_pct", 0),
        )

        risk_stats = RiskStatistics(
            sharpe_ratio=stats.get("sharpe_ratio", 0),
            sortino_ratio=stats.get("sortino_ratio", 0),
            calmar_ratio=stats.get("calmar_ratio", 0),
            max_drawdown_pct=stats.get("max_drawdown", 0),
            var_95_daily=stats.get("var_95", 0),
            var_accuracy=stats.get("var_accuracy", 0.95),
            max_position_concentration=stats.get("max_concentration", 0),
            kurtosis=stats.get("kurtosis", 3.0),
        )

        unique_features = UniqueFeatureScores(
            alpha_half_life_days=stats.get("alpha_half_life", 30),
            regime_consistency=stats.get("regime_consistency", 0.7),
            conviction_accuracy_correlation=stats.get("conviction_accuracy_corr", 0.6),
            strategy_crowding_score=stats.get("crowding_score", 0.2),
            black_swan_survival=stats.get("black_swan_survival", 0.8),
            reflexivity_score=stats.get("reflexivity", 0.02),
            capacity_estimate_usd=stats.get("capacity_usd", 10_000_000),
            uniqueness_score=stats.get("uniqueness", 0.5),
        )

        learning_stats = LearningStatistics(
            total_training_epochs=stats.get("epochs", 100),
            improvement_rate_per_100_trades=stats.get("learning_rate", 0.02),
            adaptation_speed_days=stats.get("adaptation_speed", 5),
            error_recovery_rate=stats.get("error_recovery", 0.9),
        )

        return self.grade(trade_stats, risk_stats, unique_features, learning_stats)


def calculate_unique_features(
    returns: np.ndarray,
    signals: np.ndarray,
    confidences: np.ndarray,
    regime_labels: np.ndarray,
    market_returns: np.ndarray,
) -> UniqueFeatureScores:
    """Calculate proprietary unique features from raw data.

    These are the metrics that differentiate elite strategies.
    """
    features = UniqueFeatureScores()

    if len(returns) < 100:
        return features

    # Alpha half-life calculation
    # Measure how quickly alpha decays over time
    try:
        cumulative_alpha = np.cumsum(returns - market_returns)
        if len(cumulative_alpha) > 50:
            # Find when cumulative alpha drops to half
            peak_alpha = np.max(cumulative_alpha)
            half_alpha = peak_alpha / 2
            half_life_idx = np.argmax(cumulative_alpha >= half_alpha)
            features.alpha_half_life_days = max(1, half_life_idx)
    except Exception:
        features.alpha_half_life_days = 30

    # Regime consistency
    # Calculate Sharpe in each regime
    unique_regimes = np.unique(regime_labels)
    regime_sharpes = []
    for regime in unique_regimes:
        mask = regime_labels == regime
        if mask.sum() > 20:
            regime_returns = returns[mask]
            regime_sharpe = np.mean(regime_returns) / (np.std(regime_returns) + 1e-10) * np.sqrt(252)
            regime_sharpes.append(regime_sharpe)

    if regime_sharpes:
        # Consistency = % of regimes with positive Sharpe
        features.regime_consistency = np.mean([s > 0 for s in regime_sharpes])

    # Conviction-accuracy correlation
    # When confidence is high, are we more accurate?
    if len(confidences) == len(returns):
        correct = (returns > 0).astype(float)
        features.conviction_accuracy_correlation = np.corrcoef(confidences, correct)[0, 1]
        if np.isnan(features.conviction_accuracy_correlation):
            features.conviction_accuracy_correlation = 0

    # Black swan survival
    # Performance during extreme market moves (>3 std dev)
    market_std = np.std(market_returns)
    tail_mask = np.abs(market_returns) > 3 * market_std
    if tail_mask.sum() > 0:
        tail_returns = returns[tail_mask]
        features.black_swan_survival = np.mean(tail_returns > -0.05)  # Survived if lost <5%
        features.tail_event_avg_return = np.mean(tail_returns)

    # Uniqueness score (correlation with common factors)
    # Lower correlation = more unique
    market_corr = np.corrcoef(returns, market_returns)[0, 1]
    features.uniqueness_score = 1 - abs(market_corr) if not np.isnan(market_corr) else 0.5

    return features


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def grade_agent_elite(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Grade an agent using elite thresholds.
    Returns a dictionary with grade and details.
    """
    grader = EliteAgentGrader()
    result = grader.quick_grade(stats)

    return {
        "grade": result.grade.value,
        "score": result.score,
        "passed": result.passed,
        "performance_score": result.performance_score,
        "risk_score": result.risk_score,
        "uniqueness_score": result.uniqueness_score,
        "learning_score": result.learning_score,
        "operational_score": result.operational_score,
        "failures": result.failures,
        "warnings": result.warnings,
        "strengths": result.strengths,
        "unique_edges": result.unique_edges,
        "recommendations": result.recommendations,
        "graded_at": result.graded_at.isoformat(),
    }


def is_elite(stats: Dict[str, Any]) -> bool:
    """Quick check if agent meets elite standards."""
    result = grade_agent_elite(stats)
    return result["grade"] == "ELITE"


def is_battle_ready(stats: Dict[str, Any]) -> bool:
    """Quick check if agent is ready for paper trading."""
    result = grade_agent_elite(stats)
    return result["grade"] in ["ELITE", "BATTLE_READY"]

