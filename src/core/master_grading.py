"""================================================================================
MASTER GRADING SYSTEM - Objectively Hard Standards
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

GOAL: Compete with Citadel, Goldman Sachs, Renaissance, Two Sigma

This grading system is INTENTIONALLY BRUTAL. The bar is set at institutional
hedge fund level because that's who you're competing with.

PHILOSOPHY:
- If the system gives A's easily, the bar is too low
- If your best models can't beat a B, that's the point
- An A here should mean you could work at Renaissance
- A C here might be profitable, but you're not competitive

GRADING TIERS:
- S (95-100): Renaissance/Medallion level - nearly impossible
- A (85-94): Elite hedge fund level - exceptionally rare
- B (70-84): Competent quant fund level - this is good
- C (55-69): Average fund level - needs improvement
- D (40-54): Below average - significant work needed
- F (<40): Not production ready

WHAT WE GRADE:
1. AGENTS - Individual agent performance
2. MODELS - ML model quality
3. SIGNALS - Signal generation
4. EXECUTION - Trade execution quality
5. RISK - Risk management effectiveness
6. INTEGRATION - How well everything works together

BENCHMARK: Citadel, Renaissance, Two Sigma, Bridgewater
================================================================================
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GradeTier(Enum):
    """Grade tiers with institutional standards."""

    S = "S"   # 95-100: Renaissance level
    A = "A"   # 85-94: Elite hedge fund
    B = "B"   # 70-84: Competent quant
    C = "C"   # 55-69: Average fund
    D = "D"   # 40-54: Below average
    F = "F"   # <40: Not production ready


class ComponentType(Enum):
    """Components being graded."""

    AGENT = "agent"
    MODEL = "model"
    SIGNAL = "signal"
    EXECUTION = "execution"
    RISK = "risk"
    INTEGRATION = "integration"
    OVERALL = "overall"


@dataclass
class GradeReport:
    """A detailed grade report."""

    component_id: str
    component_type: ComponentType
    component_name: str

    timestamp: datetime

    # Scores
    raw_score: float  # 0-100
    adjusted_score: float  # After difficulty adjustment
    grade: GradeTier

    # Breakdown
    metrics: Dict[str, float]
    weights: Dict[str, float]

    # Context
    benchmark: str  # "citadel", "renaissance", etc.
    percentile: float  # Where this falls in distribution

    # Feedback
    strengths: List[str]
    weaknesses: List[str]
    action_items: List[str]

    # Comparison
    vs_benchmark_gap: float  # How far below/above benchmark
    improvement_since_last: float  # Change from last grading

    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "component_name": self.component_name,
            "raw_score": self.raw_score,
            "adjusted_score": self.adjusted_score,
            "grade": self.grade.value,
            "metrics": self.metrics,
            "benchmark": self.benchmark,
            "percentile": self.percentile,
            "vs_benchmark_gap": self.vs_benchmark_gap,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "action_items": self.action_items,
        }


@dataclass
class InstitutionalBenchmark:
    """Benchmark metrics from elite institutions."""

    name: str

    # Returns
    annual_return: float  # Target annual return
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown: float  # Maximum drawdown
    win_rate: float  # % of winning trades

    # Model quality
    information_ratio: float
    hit_rate: float
    model_stability: float

    # Execution
    slippage_bps: float  # Execution slippage
    fill_rate: float  # Order fill rate

    # Risk
    var_accuracy: float  # VaR model accuracy
    stress_test_pass_rate: float


class MasterGradingSystem:
    """Master Grading System - Institutional Standards

    INTENTIONALLY BRUTAL because you're competing with:
    - Citadel: $55B AUM, 26% annual returns, 100+ PhDs
    - Renaissance: $130B AUM, 66% Medallion returns, best in history
    - Two Sigma: $60B AUM, massive alt data, 1000+ engineers
    - Bridgewater: $150B AUM, systematic macro pioneer

    Your system needs to be graded against THESE standards, not
    against "average" quant funds or retail traders.

    GRADING PHILOSOPHY:
    1. If it's easy to get an A, the system is broken
    2. A "B" should be genuinely impressive
    3. "C" is average - and average loses to the above
    4. The goal is to make the grading so hard that passing means something
    """

    # Institutional benchmarks - THE BAR TO BEAT
    BENCHMARKS = {
        "renaissance": InstitutionalBenchmark(
            name="Renaissance Medallion",
            annual_return=0.66,  # 66% gross returns
            sharpe_ratio=4.0,  # Exceptional
            max_drawdown=0.05,  # 5% max DD
            win_rate=0.55,  # Just above 50% but consistent
            information_ratio=3.0,
            hit_rate=0.52,
            model_stability=0.95,
            slippage_bps=0.5,
            fill_rate=0.99,
            var_accuracy=0.97,
            stress_test_pass_rate=0.95,
        ),
        "citadel": InstitutionalBenchmark(
            name="Citadel Wellington",
            annual_return=0.26,  # 26% returns
            sharpe_ratio=2.5,
            max_drawdown=0.08,
            win_rate=0.54,
            information_ratio=2.2,
            hit_rate=0.51,
            model_stability=0.90,
            slippage_bps=0.3,  # Best in class execution
            fill_rate=0.995,
            var_accuracy=0.95,
            stress_test_pass_rate=0.92,
        ),
        "two_sigma": InstitutionalBenchmark(
            name="Two Sigma",
            annual_return=0.20,
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.53,
            information_ratio=1.8,
            hit_rate=0.50,
            model_stability=0.88,
            slippage_bps=1.0,
            fill_rate=0.98,
            var_accuracy=0.93,
            stress_test_pass_rate=0.90,
        ),
        "bridgewater": InstitutionalBenchmark(
            name="Bridgewater Pure Alpha",
            annual_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.52,
            information_ratio=1.5,
            hit_rate=0.50,
            model_stability=0.85,
            slippage_bps=2.0,  # Larger positions, more impact
            fill_rate=0.97,
            var_accuracy=0.92,
            stress_test_pass_rate=0.88,
        ),
    }

    # Default benchmark - aim for Citadel level
    DEFAULT_BENCHMARK = "citadel"

    # Grade thresholds (HARD)
    GRADE_THRESHOLDS = {
        GradeTier.S: 95,  # Nearly impossible
        GradeTier.A: 85,  # Exceptionally rare
        GradeTier.B: 70,  # Good
        GradeTier.C: 55,  # Average
        GradeTier.D: 40,  # Below average
        GradeTier.F: 0,   # Failing
    }

    # Difficulty multipliers by component (lower = harder to score)
    DIFFICULTY_MULTIPLIERS = {
        ComponentType.AGENT: 0.85,      # Agents are hard to perfect
        ComponentType.MODEL: 0.80,       # Models are harder
        ComponentType.SIGNAL: 0.75,      # Signals are very hard
        ComponentType.EXECUTION: 0.90,   # Execution is more measurable
        ComponentType.RISK: 0.85,        # Risk is hard to grade
        ComponentType.INTEGRATION: 0.70, # Integration is hardest
        ComponentType.OVERALL: 0.80,     # Overall average
    }

    def __init__(self, benchmark: str = None):
        """Initialize master grading system.

        Args:
        ----
            benchmark: Which institution to benchmark against
        """
        self.benchmark_name = benchmark or self.DEFAULT_BENCHMARK
        self.benchmark = self.BENCHMARKS.get(self.benchmark_name, self.BENCHMARKS["citadel"])

        # Grade history
        self.grade_history: List[GradeReport] = []

        # Component trackers
        self.agent_grades: Dict[str, GradeReport] = {}
        self.model_grades: Dict[str, GradeReport] = {}
        self.signal_grades: Dict[str, GradeReport] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Master Grading System initialized - Benchmark: {self.benchmark.name}")

    # =========================================================================
    # AGENT GRADING
    # =========================================================================

    def grade_agent(
        self,
        agent_id: str,
        agent_name: str,
        metrics: Dict[str, Any],
    ) -> GradeReport:
        """Grade an individual agent.

        AGENT GRADING CRITERIA:
        - Prediction accuracy
        - Signal quality
        - Response time
        - Uptime/reliability
        - Collaboration effectiveness
        - Learning rate

        Args:
        ----
            agent_id: Unique agent identifier
            agent_name: Human readable name
            metrics: Performance metrics

        Returns:
        -------
            GradeReport with grade and feedback
        """
        # Define weights for agent scoring
        weights = {
            "prediction_accuracy": 0.25,
            "signal_quality": 0.20,
            "response_time": 0.10,
            "reliability": 0.15,
            "collaboration": 0.15,
            "learning_rate": 0.15,
        }

        scores = {}

        # 1. Prediction accuracy (vs random = 50%)
        accuracy = metrics.get("prediction_accuracy", 0.50)
        # Scale: 50% = 0, 55% = 50, 60% = 75, 65%+ = 100
        if accuracy <= 0.50:
            scores["prediction_accuracy"] = 0
        elif accuracy <= 0.55:
            scores["prediction_accuracy"] = (accuracy - 0.50) / 0.05 * 50
        elif accuracy <= 0.60:
            scores["prediction_accuracy"] = 50 + (accuracy - 0.55) / 0.05 * 25
        else:
            scores["prediction_accuracy"] = min(100, 75 + (accuracy - 0.60) / 0.05 * 25)

        # 2. Signal quality (Sharpe ratio of signals)
        signal_sharpe = metrics.get("signal_sharpe", 0)
        # Scale: 0 = 0, 1.0 = 50, 2.0 = 80, 3.0+ = 100
        if signal_sharpe <= 0:
            scores["signal_quality"] = 0
        elif signal_sharpe <= 1.0:
            scores["signal_quality"] = signal_sharpe * 50
        elif signal_sharpe <= 2.0:
            scores["signal_quality"] = 50 + (signal_sharpe - 1.0) * 30
        else:
            scores["signal_quality"] = min(100, 80 + (signal_sharpe - 2.0) * 20)

        # 3. Response time (milliseconds)
        response_ms = metrics.get("response_time_ms", 1000)
        # Scale: <10ms = 100, 10-100ms = 75-100, 100-1000ms = 50-75, >1000ms = 0-50
        if response_ms <= 10:
            scores["response_time"] = 100
        elif response_ms <= 100:
            scores["response_time"] = 75 + (100 - response_ms) / 90 * 25
        elif response_ms <= 1000:
            scores["response_time"] = 50 + (1000 - response_ms) / 900 * 25
        else:
            scores["response_time"] = max(0, 50 - (response_ms - 1000) / 1000 * 50)

        # 4. Reliability (uptime percentage)
        uptime = metrics.get("uptime_pct", 0.99)
        # Scale: 99.99% = 100, 99.9% = 90, 99% = 70, 95% = 50, <90% = 0
        if uptime >= 0.9999:
            scores["reliability"] = 100
        elif uptime >= 0.999:
            scores["reliability"] = 90 + (uptime - 0.999) / 0.0009 * 10
        elif uptime >= 0.99:
            scores["reliability"] = 70 + (uptime - 0.99) / 0.009 * 20
        elif uptime >= 0.95:
            scores["reliability"] = 50 + (uptime - 0.95) / 0.04 * 20
        else:
            scores["reliability"] = max(0, uptime / 0.95 * 50)

        # 5. Collaboration (inter-agent effectiveness)
        collab_score = metrics.get("collaboration_score", 0.5)
        scores["collaboration"] = collab_score * 100

        # 6. Learning rate (improvement over time)
        learning_rate = metrics.get("learning_rate", 0)
        # Scale: negative = 0, 0% = 50, 5%+ improvement = 100
        if learning_rate < 0:
            scores["learning_rate"] = max(0, 50 + learning_rate * 100)  # Penalize regression
        else:
            scores["learning_rate"] = min(100, 50 + learning_rate / 0.05 * 50)

        # Calculate raw score
        raw_score = sum(scores[k] * weights[k] for k in weights.keys())

        # Apply difficulty adjustment (makes it harder)
        difficulty = self.DIFFICULTY_MULTIPLIERS[ComponentType.AGENT]
        adjusted_score = raw_score * difficulty

        # Determine grade
        grade = self._score_to_grade(adjusted_score)

        # Generate feedback
        strengths, weaknesses, actions = self._generate_agent_feedback(scores, metrics)

        # Calculate benchmark gap
        vs_benchmark = adjusted_score - 75  # 75 is "Citadel level" for agents

        report = GradeReport(
            component_id=agent_id,
            component_type=ComponentType.AGENT,
            component_name=agent_name,
            timestamp=datetime.now(),
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            grade=grade,
            metrics=scores,
            weights=weights,
            benchmark=self.benchmark.name,
            percentile=self._score_to_percentile(adjusted_score),
            strengths=strengths,
            weaknesses=weaknesses,
            action_items=actions,
            vs_benchmark_gap=vs_benchmark,
            improvement_since_last=self._get_improvement(agent_id, adjusted_score),
        )

        self.agent_grades[agent_id] = report
        self.grade_history.append(report)

        return report

    def _generate_agent_feedback(
        self,
        scores: Dict[str, float],
        metrics: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate actionable feedback for agent."""
        strengths = []
        weaknesses = []
        actions = []

        # Analyze each dimension
        if scores["prediction_accuracy"] >= 70:
            strengths.append(f"Strong prediction accuracy ({scores['prediction_accuracy']:.0f}/100)")
        elif scores["prediction_accuracy"] < 50:
            weaknesses.append(f"Prediction accuracy needs work ({scores['prediction_accuracy']:.0f}/100)")
            actions.append("Review feature engineering and model architecture")

        if scores["signal_quality"] >= 70:
            strengths.append("High quality signals (Sharpe contribution)")
        elif scores["signal_quality"] < 50:
            weaknesses.append("Signal quality below threshold")
            actions.append("Add noise filtering, improve signal confidence scoring")

        if scores["response_time"] >= 70:
            strengths.append("Fast response times")
        elif scores["response_time"] < 50:
            weaknesses.append("Response time too slow for real-time")
            actions.append("Optimize computation, consider caching, profile bottlenecks")

        if scores["reliability"] >= 70:
            strengths.append("High reliability/uptime")
        elif scores["reliability"] < 50:
            weaknesses.append("Reliability issues detected")
            actions.append("Add monitoring, improve error handling, implement retries")

        if scores["learning_rate"] >= 70:
            strengths.append("Strong improvement trajectory")
        elif scores["learning_rate"] < 50:
            weaknesses.append("Learning/improvement stalled")
            actions.append("Review training pipeline, check for data issues, consider architecture changes")

        return strengths, weaknesses, actions

    # =========================================================================
    # MODEL GRADING
    # =========================================================================

    def grade_model(
        self,
        model_id: str,
        model_name: str,
        metrics: Dict[str, Any],
    ) -> GradeReport:
        """Grade a ML model with institutional standards.

        MODEL GRADING CRITERIA (VERY HARD):
        - Out-of-sample returns
        - Sharpe ratio
        - Information ratio
        - Maximum drawdown
        - Win rate
        - Profit factor
        - Model stability
        - Overfitting score

        BENCHMARK: Renaissance Medallion has Sharpe ~4.0
        Citadel Wellington has Sharpe ~2.5
        """
        weights = {
            "sharpe_ratio": 0.20,
            "returns": 0.15,
            "information_ratio": 0.15,
            "max_drawdown": 0.15,
            "win_rate": 0.10,
            "profit_factor": 0.10,
            "stability": 0.10,
            "overfitting": 0.05,
        }

        scores = {}

        # 1. Sharpe ratio (HARD - 2.0 is Citadel level)
        sharpe = metrics.get("sharpe_ratio", 0)
        # Scale: 0 = 0, 1.0 = 40, 2.0 = 70, 2.5 = 85, 4.0+ = 100
        if sharpe <= 0:
            scores["sharpe_ratio"] = 0
        elif sharpe <= 1.0:
            scores["sharpe_ratio"] = sharpe * 40
        elif sharpe <= 2.0:
            scores["sharpe_ratio"] = 40 + (sharpe - 1.0) * 30
        elif sharpe <= 2.5:
            scores["sharpe_ratio"] = 70 + (sharpe - 2.0) / 0.5 * 15
        else:
            scores["sharpe_ratio"] = min(100, 85 + (sharpe - 2.5) / 1.5 * 15)

        # 2. Annual returns (vs benchmark)
        annual_return = metrics.get("annual_return", 0)
        benchmark_return = self.benchmark.annual_return
        # Score based on % of benchmark achieved
        if annual_return <= 0:
            scores["returns"] = 0
        else:
            ratio = annual_return / benchmark_return
            scores["returns"] = min(100, ratio * 70)  # 100% of benchmark = 70

        # 3. Information ratio
        ir = metrics.get("information_ratio", 0)
        # Scale: 0 = 0, 1.0 = 50, 2.0 = 80, 3.0+ = 100
        if ir <= 0:
            scores["information_ratio"] = 0
        elif ir <= 1.0:
            scores["information_ratio"] = ir * 50
        elif ir <= 2.0:
            scores["information_ratio"] = 50 + (ir - 1.0) * 30
        else:
            scores["information_ratio"] = min(100, 80 + (ir - 2.0) * 20)

        # 4. Max drawdown (LOWER IS BETTER)
        max_dd = metrics.get("max_drawdown", 0.5)
        benchmark_dd = self.benchmark.max_drawdown
        # Scale: <= benchmark = 100, 2x benchmark = 50, 4x+ = 0
        if max_dd <= benchmark_dd:
            scores["max_drawdown"] = 100
        elif max_dd <= benchmark_dd * 2:
            scores["max_drawdown"] = 100 - (max_dd - benchmark_dd) / benchmark_dd * 50
        elif max_dd <= benchmark_dd * 4:
            scores["max_drawdown"] = 50 - (max_dd - benchmark_dd * 2) / (benchmark_dd * 2) * 50
        else:
            scores["max_drawdown"] = 0

        # 5. Win rate
        win_rate = metrics.get("win_rate", 0.50)
        # Scale: 50% = 30, 52% = 50, 55% = 80, 60%+ = 100
        if win_rate <= 0.50:
            scores["win_rate"] = win_rate / 0.50 * 30
        elif win_rate <= 0.52:
            scores["win_rate"] = 30 + (win_rate - 0.50) / 0.02 * 20
        elif win_rate <= 0.55:
            scores["win_rate"] = 50 + (win_rate - 0.52) / 0.03 * 30
        else:
            scores["win_rate"] = min(100, 80 + (win_rate - 0.55) / 0.05 * 20)

        # 6. Profit factor (total profits / total losses)
        pf = metrics.get("profit_factor", 1.0)
        # Scale: 1.0 = 0, 1.5 = 50, 2.0 = 75, 3.0+ = 100
        if pf <= 1.0:
            scores["profit_factor"] = 0
        elif pf <= 1.5:
            scores["profit_factor"] = (pf - 1.0) / 0.5 * 50
        elif pf <= 2.0:
            scores["profit_factor"] = 50 + (pf - 1.5) / 0.5 * 25
        else:
            scores["profit_factor"] = min(100, 75 + (pf - 2.0) * 25)

        # 7. Model stability (consistent performance across time)
        stability = metrics.get("stability", 0.5)
        scores["stability"] = stability * 100

        # 8. Overfitting score (LOWER IS BETTER - measures train vs test gap)
        overfit = metrics.get("overfitting_score", 0.5)
        # Scale: 0 = 100, 0.2 = 70, 0.5 = 40, 1.0 = 0
        scores["overfitting"] = max(0, 100 - overfit * 100)

        # Calculate raw score
        raw_score = sum(scores[k] * weights[k] for k in weights.keys())

        # Apply HARD difficulty adjustment
        difficulty = self.DIFFICULTY_MULTIPLIERS[ComponentType.MODEL]
        adjusted_score = raw_score * difficulty

        # Determine grade
        grade = self._score_to_grade(adjusted_score)

        # Generate feedback
        strengths, weaknesses, actions = self._generate_model_feedback(scores, metrics)

        report = GradeReport(
            component_id=model_id,
            component_type=ComponentType.MODEL,
            component_name=model_name,
            timestamp=datetime.now(),
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            grade=grade,
            metrics=scores,
            weights=weights,
            benchmark=self.benchmark.name,
            percentile=self._score_to_percentile(adjusted_score),
            strengths=strengths,
            weaknesses=weaknesses,
            action_items=actions,
            vs_benchmark_gap=adjusted_score - 75,
            improvement_since_last=self._get_improvement(model_id, adjusted_score),
        )

        self.model_grades[model_id] = report
        self.grade_history.append(report)

        return report

    def _generate_model_feedback(
        self,
        scores: Dict[str, float],
        metrics: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate actionable feedback for model."""
        strengths = []
        weaknesses = []
        actions = []

        if scores["sharpe_ratio"] >= 70:
            strengths.append(f"Strong risk-adjusted returns (Sharpe {metrics.get('sharpe_ratio', 0):.2f})")
        elif scores["sharpe_ratio"] < 50:
            weaknesses.append("Sharpe ratio below institutional threshold")
            actions.append("Improve signal quality or reduce position sizing volatility")

        if scores["max_drawdown"] >= 70:
            strengths.append("Well-controlled drawdowns")
        elif scores["max_drawdown"] < 50:
            weaknesses.append("Drawdowns exceed acceptable limits")
            actions.append("Implement tighter stop losses, reduce position concentration")

        if scores["stability"] >= 70:
            strengths.append("Consistent performance across time periods")
        elif scores["stability"] < 50:
            weaknesses.append("Performance unstable - regime dependent")
            actions.append("Add regime detection, diversify across strategies")

        if scores["overfitting"] >= 70:
            strengths.append("Good generalization (low overfitting)")
        elif scores["overfitting"] < 50:
            weaknesses.append("CRITICAL: Model appears overfit")
            actions.append("Simplify model, increase regularization, use walk-forward validation")

        return strengths, weaknesses, actions

    # =========================================================================
    # SIGNAL GRADING
    # =========================================================================

    def grade_signal(
        self,
        signal_id: str,
        signal_name: str,
        metrics: Dict[str, Any],
    ) -> GradeReport:
        """Grade a signal/alpha source.

        SIGNAL GRADING (HARDEST):
        - Information coefficient (IC)
        - IC decay
        - Capacity
        - Uniqueness
        - Crowding

        NOTE: Most signals score poorly here. That's intentional.
        If your signal scores a B, it's genuinely good.
        """
        weights = {
            "information_coefficient": 0.30,
            "ic_decay": 0.20,
            "capacity": 0.15,
            "uniqueness": 0.20,
            "crowding": 0.15,
        }

        scores = {}

        # 1. Information Coefficient (correlation with future returns)
        ic = metrics.get("information_coefficient", 0)
        # Scale: 0 = 0, 0.02 = 30, 0.05 = 60, 0.10+ = 100
        # NOTE: IC of 0.05 is GOOD. IC of 0.10 is EXCEPTIONAL.
        if ic <= 0:
            scores["information_coefficient"] = 0
        elif ic <= 0.02:
            scores["information_coefficient"] = ic / 0.02 * 30
        elif ic <= 0.05:
            scores["information_coefficient"] = 30 + (ic - 0.02) / 0.03 * 30
        else:
            scores["information_coefficient"] = min(100, 60 + (ic - 0.05) / 0.05 * 40)

        # 2. IC decay (how fast does signal lose predictive power)
        ic_halflife_days = metrics.get("ic_halflife_days", 30)
        # Scale: <7 days = 0, 7-30 days = 50, 30-90 days = 80, 90+ days = 100
        if ic_halflife_days < 7:
            scores["ic_decay"] = 0
        elif ic_halflife_days < 30:
            scores["ic_decay"] = 50 * (ic_halflife_days - 7) / 23
        elif ic_halflife_days < 90:
            scores["ic_decay"] = 50 + 30 * (ic_halflife_days - 30) / 60
        else:
            scores["ic_decay"] = min(100, 80 + 20 * (ic_halflife_days - 90) / 90)

        # 3. Capacity (how much capital can trade this signal)
        capacity_millions = metrics.get("capacity_millions", 10)
        # Scale: <$1M = 0, $1-10M = 50, $10-100M = 80, $100M+ = 100
        if capacity_millions < 1:
            scores["capacity"] = capacity_millions * 50
        elif capacity_millions < 10:
            scores["capacity"] = 50 + (capacity_millions - 1) / 9 * 30
        elif capacity_millions < 100:
            scores["capacity"] = 80 + (capacity_millions - 10) / 90 * 20
        else:
            scores["capacity"] = 100

        # 4. Uniqueness (how differentiated is this signal)
        uniqueness = metrics.get("uniqueness_score", 50)
        scores["uniqueness"] = uniqueness

        # 5. Crowding (inverse - higher crowding = lower score)
        crowding = metrics.get("crowding_score", 50)
        scores["crowding"] = 100 - crowding

        # Calculate raw score
        raw_score = sum(scores[k] * weights[k] for k in weights.keys())

        # Apply HARDEST difficulty adjustment
        difficulty = self.DIFFICULTY_MULTIPLIERS[ComponentType.SIGNAL]
        adjusted_score = raw_score * difficulty

        grade = self._score_to_grade(adjusted_score)

        report = GradeReport(
            component_id=signal_id,
            component_type=ComponentType.SIGNAL,
            component_name=signal_name,
            timestamp=datetime.now(),
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            grade=grade,
            metrics=scores,
            weights=weights,
            benchmark=self.benchmark.name,
            percentile=self._score_to_percentile(adjusted_score),
            strengths=[],
            weaknesses=[],
            action_items=[],
            vs_benchmark_gap=adjusted_score - 75,
            improvement_since_last=self._get_improvement(signal_id, adjusted_score),
        )

        self.signal_grades[signal_id] = report
        self.grade_history.append(report)

        return report

    # =========================================================================
    # OVERALL SYSTEM GRADING
    # =========================================================================

    def grade_overall_system(self) -> GradeReport:
        """Grade the entire system.

        Combines all component grades with weighting.
        """
        if not self.agent_grades and not self.model_grades and not self.signal_grades:
            return GradeReport(
                component_id="system",
                component_type=ComponentType.OVERALL,
                component_name="Alpha Loop System",
                timestamp=datetime.now(),
                raw_score=0,
                adjusted_score=0,
                grade=GradeTier.F,
                metrics={},
                weights={},
                benchmark=self.benchmark.name,
                percentile=0,
                strengths=[],
                weaknesses=["No components graded yet"],
                action_items=["Run grading on agents, models, and signals"],
                vs_benchmark_gap=-75,
                improvement_since_last=0,
            )

        # Aggregate scores
        component_scores = []

        for grade in self.agent_grades.values():
            component_scores.append(("agent", grade.adjusted_score))

        for grade in self.model_grades.values():
            component_scores.append(("model", grade.adjusted_score))

        for grade in self.signal_grades.values():
            component_scores.append(("signal", grade.adjusted_score))

        # Weight by component type
        type_weights = {
            "agent": 0.25,
            "model": 0.40,
            "signal": 0.35,
        }

        type_sums = {"agent": 0, "model": 0, "signal": 0}
        type_counts = {"agent": 0, "model": 0, "signal": 0}

        for comp_type, score in component_scores:
            type_sums[comp_type] += score
            type_counts[comp_type] += 1

        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0

        for comp_type in ["agent", "model", "signal"]:
            if type_counts[comp_type] > 0:
                avg = type_sums[comp_type] / type_counts[comp_type]
                weighted_sum += avg * type_weights[comp_type]
                total_weight += type_weights[comp_type]

        if total_weight > 0:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 0

        # Apply overall difficulty
        difficulty = self.DIFFICULTY_MULTIPLIERS[ComponentType.OVERALL]
        adjusted_score = raw_score * difficulty

        grade = self._score_to_grade(adjusted_score)

        # Generate overall feedback
        strengths = []
        weaknesses = []
        actions = []

        if type_counts["model"] > 0 and type_sums["model"] / type_counts["model"] >= 70:
            strengths.append("Strong model performance")
        elif type_counts["model"] > 0:
            weaknesses.append("Model performance needs improvement")
            actions.append("Focus on model architecture and training")

        if type_counts["signal"] > 0 and type_sums["signal"] / type_counts["signal"] >= 70:
            strengths.append("High quality signals")
        elif type_counts["signal"] > 0:
            weaknesses.append("Signal quality below institutional standards")
            actions.append("Develop unique data sources, improve signal research")

        return GradeReport(
            component_id="system",
            component_type=ComponentType.OVERALL,
            component_name="Alpha Loop System",
            timestamp=datetime.now(),
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            grade=grade,
            metrics={
                "agents_graded": type_counts["agent"],
                "models_graded": type_counts["model"],
                "signals_graded": type_counts["signal"],
                "agent_avg": type_sums["agent"] / max(1, type_counts["agent"]),
                "model_avg": type_sums["model"] / max(1, type_counts["model"]),
                "signal_avg": type_sums["signal"] / max(1, type_counts["signal"]),
            },
            weights=type_weights,
            benchmark=self.benchmark.name,
            percentile=self._score_to_percentile(adjusted_score),
            strengths=strengths,
            weaknesses=weaknesses,
            action_items=actions,
            vs_benchmark_gap=adjusted_score - 75,
            improvement_since_last=0,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _score_to_grade(self, score: float) -> GradeTier:
        """Convert numeric score to grade tier."""
        if score >= self.GRADE_THRESHOLDS[GradeTier.S]:
            return GradeTier.S
        elif score >= self.GRADE_THRESHOLDS[GradeTier.A]:
            return GradeTier.A
        elif score >= self.GRADE_THRESHOLDS[GradeTier.B]:
            return GradeTier.B
        elif score >= self.GRADE_THRESHOLDS[GradeTier.C]:
            return GradeTier.C
        elif score >= self.GRADE_THRESHOLDS[GradeTier.D]:
            return GradeTier.D
        else:
            return GradeTier.F

    def _score_to_percentile(self, score: float) -> float:
        """Convert score to percentile (among quant funds)."""
        # Assume normal distribution with mean 50, std 20
        # This is generous - most funds fail
        z_score = (score - 50) / 20
        # Approximate percentile from z-score
        percentile = 50 * (1 + math.erf(z_score / math.sqrt(2)))
        return round(percentile, 1)

    def _get_improvement(self, component_id: str, new_score: float) -> float:
        """Get improvement since last grading."""
        # Find previous grade for this component
        previous_grades = [
            g for g in self.grade_history
            if g.component_id == component_id
        ]

        if len(previous_grades) < 2:
            return 0.0

        # Get second-to-last grade
        previous = previous_grades[-2]
        return new_score - previous.adjusted_score

    def get_grade_summary(self) -> Dict[str, Any]:
        """Get summary of all grades."""
        return {
            "benchmark": self.benchmark.name,
            "agents": {
                "count": len(self.agent_grades),
                "grades": {k: v.grade.value for k, v in self.agent_grades.items()},
            },
            "models": {
                "count": len(self.model_grades),
                "grades": {k: v.grade.value for k, v in self.model_grades.items()},
            },
            "signals": {
                "count": len(self.signal_grades),
                "grades": {k: v.grade.value for k, v in self.signal_grades.items()},
            },
            "overall": self.grade_overall_system().to_dict(),
        }

    def print_report_card(self):
        """Print a formatted report card."""
        overall = self.grade_overall_system()

        print("\n" + "=" * 70)
        print("ALPHA LOOP CAPITAL - REPORT CARD")
        print(f"Benchmark: {self.benchmark.name}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)

        print(f"\n{'OVERALL GRADE:':<30} {overall.grade.value}")
        print(f"{'Score:':<30} {overall.adjusted_score:.1f}/100")
        print(f"{'Percentile:':<30} {overall.percentile:.1f}%")
        print(f"{'vs Benchmark:':<30} {overall.vs_benchmark_gap:+.1f} points")

        print("\n" + "-" * 70)
        print("COMPONENT GRADES")
        print("-" * 70)

        print(f"\n{'AGENTS (' + str(len(self.agent_grades)) + '):'}")
        for agent_id, grade in self.agent_grades.items():
            print(f"  {grade.component_name:<30} {grade.grade.value} ({grade.adjusted_score:.1f})")

        print(f"\n{'MODELS (' + str(len(self.model_grades)) + '):'}")
        for model_id, grade in self.model_grades.items():
            print(f"  {grade.component_name:<30} {grade.grade.value} ({grade.adjusted_score:.1f})")

        print(f"\n{'SIGNALS (' + str(len(self.signal_grades)) + '):'}")
        for signal_id, grade in self.signal_grades.items():
            print(f"  {grade.component_name:<30} {grade.grade.value} ({grade.adjusted_score:.1f})")

        print("\n" + "-" * 70)
        print("STRENGTHS")
        print("-" * 70)
        for s in overall.strengths:
            print(f"  ✓ {s}")

        print("\n" + "-" * 70)
        print("AREAS FOR IMPROVEMENT")
        print("-" * 70)
        for w in overall.weaknesses:
            print(f"  ✗ {w}")

        print("\n" + "-" * 70)
        print("ACTION ITEMS")
        print("-" * 70)
        for a in overall.action_items:
            print(f"  → {a}")

        print("\n" + "=" * 70)
        print("NOTE: Grades are benchmarked against institutional hedge funds.")
        print("A 'C' here might be profitable, but won't compete with Citadel.")
        print("=" * 70 + "\n")


# =============================================================================
# SINGLETON
# =============================================================================

_grading_system: Optional[MasterGradingSystem] = None


def get_master_grading_system(benchmark: str = None) -> MasterGradingSystem:
    """Get master grading system singleton."""
    global _grading_system
    if _grading_system is None:
        _grading_system = MasterGradingSystem(benchmark)
    return _grading_system

