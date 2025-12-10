"""================================================================================
INSTITUTIONAL GRADING SYSTEM - CITADEL/GOLDMAN COMPETITION LEVEL
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC
Version: 3.0 | December 2024

This is NOT a participation trophy system. This is RUTHLESS grading designed
to compete with the world's best hedge funds.

Philosophy:
- If it can't beat random by a SIGNIFICANT margin, TERMINATE IT
- If it can't survive stress, TERMINATE IT
- If it doesn't learn, TERMINATE IT
- If it has EGO (overconfidence), TERMINATE IT
- Only the ELITE survive

Grade Scale:
- S+ : Citadel Elite (top 0.1% of all agents ever created)
- S  : Goldman/Renaissance level (top 1%)
- A+ : Institutional grade - PROMOTED TO LIVE
- A  : Institutional grade - PROMOTED TO PAPER
- B+ : Promising - Continue training with priority
- B  : Acceptable - Continue training
- C  : Below standard - PROBATION (24hr to improve or terminate)
- D  : Unacceptable - IMMEDIATE TERMINATION
- F  : Dangerous - BLACKLIST (never recreate this configuration)

"By end of 2026, they will know Alpha Loop Capital."
================================================================================
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple


class GradeLevel(Enum):
    """Institutional grade levels."""

    S_PLUS = "S+"    # Citadel Elite
    S = "S"          # Goldman/Renaissance
    A_PLUS = "A+"    # Institutional - Live
    A = "A"          # Institutional - Paper
    B_PLUS = "B+"    # Promising
    B = "B"          # Acceptable
    C = "C"          # Probation
    D = "D"          # Terminate
    F = "F"          # Blacklist


class AgentCategory(Enum):
    """Agent categories with different standards."""

    MASTER = "master"          # HOAGS, GHOST
    SENIOR = "senior"          # SCOUT, HUNTER, BOOKMAKER, etc.
    OPERATIONAL = "operational" # Data, Execution, Risk, etc.
    STRATEGY = "strategy"      # 34 strategy agents
    SECTOR = "sector"          # 11 sector agents
    SWARM = "swarm"           # Dynamically created


@dataclass(frozen=True)
class InstitutionalThresholds:
    """RUTHLESS thresholds for institutional-grade trading.

    These are based on what Citadel, Renaissance, Goldman actually require.
    If you can't meet these, you don't belong in production.
    """

    # =========================================================================
    # PREDICTION QUALITY (The Core)
    # =========================================================================

    # AUC (Area Under Curve) - Most Important Metric
    min_auc_live: float = 0.58          # Must beat random by 16%+ for live
    min_auc_paper: float = 0.55         # Must beat random by 10%+ for paper
    min_auc_training: float = 0.52      # Must beat random by 4%+ to continue
    max_auc_suspicious: float = 0.70    # Above this = likely overfitting

    # Accuracy
    min_accuracy_live: float = 0.56     # 56%+ for live trading
    min_accuracy_paper: float = 0.54    # 54%+ for paper trading
    min_accuracy_training: float = 0.52 # 52%+ to continue training

    # Precision (When we say BUY, we better be right)
    min_precision: float = 0.55         # 55%+ precision on signals

    # =========================================================================
    # RISK METRICS (The Guardrails)
    # =========================================================================

    # Sharpe Ratio (Risk-Adjusted Returns)
    min_sharpe_live: float = 2.0        # Sharpe 2.0+ for live (top 5% hedge funds)
    min_sharpe_paper: float = 1.5       # Sharpe 1.5+ for paper
    min_sharpe_training: float = 1.0    # Sharpe 1.0+ to continue

    # Sortino Ratio (Downside Risk)
    min_sortino: float = 2.5            # Must have asymmetric returns

    # Calmar Ratio (Return/Max Drawdown)
    min_calmar: float = 1.5             # Return must exceed drawdown

    # Maximum Drawdown
    max_drawdown_live: float = 0.05     # 5% max drawdown for live
    max_drawdown_paper: float = 0.08    # 8% max drawdown for paper
    max_drawdown_training: float = 0.12 # 12% max during training

    # Value at Risk (95th percentile)
    max_var_95: float = 0.02            # 2% VaR limit

    # =========================================================================
    # TRADING METRICS (The Execution)
    # =========================================================================

    # Win Rate
    min_win_rate: float = 0.52          # Must win more than lose

    # Profit Factor (Gross Profits / Gross Losses)
    min_profit_factor: float = 1.5      # Must make 50% more than you lose

    # Average Win / Average Loss
    min_win_loss_ratio: float = 1.2     # Avg win must exceed avg loss by 20%

    # Trade Count (Proof of concept)
    min_trades_live: int = 500          # 500+ trades proven for live
    min_trades_paper: int = 200         # 200+ trades for paper
    min_trades_training: int = 50       # 50+ trades to evaluate

    # =========================================================================
    # EXPERIENCE METRICS (The Battle Scars)
    # =========================================================================

    # Execution Count
    min_executions_live: int = 1000     # 1000+ successful executions
    min_executions_paper: int = 500     # 500+ for paper
    min_executions_training: int = 100  # 100+ for continued training

    # Learning Events
    min_learning_events: int = 1000     # Must have learned from 1000+ outcomes

    # Battle Stats (Stress Survival)
    min_crashes_survived: int = 5       # Must survive 5+ system crashes
    min_drawdowns_navigated: int = 10   # Must navigate 10+ drawdown periods
    min_regime_changes: int = 3         # Must adapt to 3+ regime changes
    min_black_swans: int = 1            # Must survive at least 1 black swan

    # =========================================================================
    # CAPABILITY METRICS (The Skillset)
    # =========================================================================

    # Capabilities
    min_capabilities_master: int = 15   # Masters need 15+ capabilities
    min_capabilities_senior: int = 10   # Seniors need 10+
    min_capabilities_standard: int = 5  # Standard need 5+

    # =========================================================================
    # CALIBRATION METRICS (The Humility)
    # =========================================================================

    # Confidence Calibration (Agents that are overconfident get penalized)
    max_overconfidence: float = 0.10    # Can't be >10% overconfident
    min_confidence_calibration: float = 0.85  # Must be 85%+ calibrated

    # =========================================================================
    # CONSISTENCY METRICS (The Reliability)
    # =========================================================================

    # Cross-Validation Consistency
    max_cv_std: float = 0.05            # CV scores can't vary more than 5%

    # Out-of-Sample Degradation
    max_oos_degradation: float = 0.10   # OOS can't be >10% worse than training

    # Rolling Window Stability
    min_rolling_consistency: float = 0.80  # 80%+ consistent across time windows


@dataclass
class AgentGradeReport:
    """Comprehensive grade report for an agent."""

    agent_id: str
    agent_name: str
    category: AgentCategory

    # Final Grade
    grade: GradeLevel
    grade_score: float  # 0-100 numerical score

    # Verdict
    promoted: bool
    promotion_level: str  # "LIVE", "PAPER", "TRAINING", "TERMINATED", "BLACKLISTED"

    # Metric Scores (each 0-100)
    prediction_score: float
    risk_score: float
    trading_score: float
    experience_score: float
    capability_score: float
    calibration_score: float
    consistency_score: float

    # Detailed Metrics
    metrics: Dict[str, Any]

    # Issues Found
    critical_failures: List[str]
    warnings: List[str]
    strengths: List[str]

    # Recommendations
    action_required: str
    improvement_areas: List[str]

    # Metadata
    graded_at: datetime = field(default_factory=datetime.now)
    graded_by: str = "InstitutionalGrader"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "category": self.category.value,
            "grade": self.grade.value,
            "grade_score": self.grade_score,
            "promoted": self.promoted,
            "promotion_level": self.promotion_level,
            "scores": {
                "prediction": self.prediction_score,
                "risk": self.risk_score,
                "trading": self.trading_score,
                "experience": self.experience_score,
                "capability": self.capability_score,
                "calibration": self.calibration_score,
                "consistency": self.consistency_score,
            },
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "strengths": self.strengths,
            "action_required": self.action_required,
            "graded_at": self.graded_at.isoformat(),
        }


class InstitutionalGrader:
    """RUTHLESS institutional-grade agent grader.

    This grader has ONE job: ensure only ELITE agents make it to production.

    It evaluates 7 dimensions:
    1. Prediction Quality (AUC, Accuracy, Precision)
    2. Risk Management (Sharpe, Sortino, Drawdown, VaR)
    3. Trading Performance (Win Rate, Profit Factor)
    4. Battle Experience (Crashes, Drawdowns, Black Swans)
    5. Capabilities (Skills, Methods)
    6. Calibration (Confidence accuracy)
    7. Consistency (Stability over time)

    Grading Philosophy:
    - One critical failure = IMMEDIATE TERMINATION
    - Multiple warnings = PROBATION
    - Mediocrity is not tolerated
    - Only the best survive
    """

    def __init__(self, thresholds: InstitutionalThresholds = None):
        self.thresholds = thresholds or InstitutionalThresholds()
        self.grading_history: List[AgentGradeReport] = []

        # Weight multipliers for final score
        self.weights = {
            "prediction": 0.25,    # 25% - Core value
            "risk": 0.25,          # 25% - Critical for survival
            "trading": 0.20,       # 20% - Execution matters
            "experience": 0.10,    # 10% - Battle-tested
            "capability": 0.08,    # 8% - Skillset
            "calibration": 0.07,   # 7% - Humility
            "consistency": 0.05,   # 5% - Reliability
        }

    def grade_agent(
        self,
        agent_id: str,
        agent_name: str,
        category: AgentCategory,
        stats: Dict[str, Any],
    ) -> AgentGradeReport:
        """Grade an agent against institutional standards.

        Args:
        ----
            agent_id: Unique agent identifier
            agent_name: Human-readable name
            category: Agent category (affects standards)
            stats: Dictionary of all agent statistics

        Returns:
        -------
            Comprehensive AgentGradeReport
        """
        t = self.thresholds

        critical_failures = []
        warnings = []
        strengths = []

        # =====================================================================
        # DIMENSION 1: PREDICTION QUALITY (25%)
        # =====================================================================
        prediction_score, pred_failures, pred_warnings, pred_strengths = \
            self._grade_prediction(stats, t)
        critical_failures.extend(pred_failures)
        warnings.extend(pred_warnings)
        strengths.extend(pred_strengths)

        # =====================================================================
        # DIMENSION 2: RISK MANAGEMENT (25%)
        # =====================================================================
        risk_score, risk_failures, risk_warnings, risk_strengths = \
            self._grade_risk(stats, t)
        critical_failures.extend(risk_failures)
        warnings.extend(risk_warnings)
        strengths.extend(risk_strengths)

        # =====================================================================
        # DIMENSION 3: TRADING PERFORMANCE (20%)
        # =====================================================================
        trading_score, trade_failures, trade_warnings, trade_strengths = \
            self._grade_trading(stats, t)
        critical_failures.extend(trade_failures)
        warnings.extend(trade_warnings)
        strengths.extend(trade_strengths)

        # =====================================================================
        # DIMENSION 4: BATTLE EXPERIENCE (10%)
        # =====================================================================
        experience_score, exp_failures, exp_warnings, exp_strengths = \
            self._grade_experience(stats, t)
        critical_failures.extend(exp_failures)
        warnings.extend(exp_warnings)
        strengths.extend(exp_strengths)

        # =====================================================================
        # DIMENSION 5: CAPABILITIES (8%)
        # =====================================================================
        capability_score, cap_failures, cap_warnings, cap_strengths = \
            self._grade_capabilities(stats, t, category)
        critical_failures.extend(cap_failures)
        warnings.extend(cap_warnings)
        strengths.extend(cap_strengths)

        # =====================================================================
        # DIMENSION 6: CALIBRATION (7%)
        # =====================================================================
        calibration_score, cal_failures, cal_warnings, cal_strengths = \
            self._grade_calibration(stats, t)
        critical_failures.extend(cal_failures)
        warnings.extend(cal_warnings)
        strengths.extend(cal_strengths)

        # =====================================================================
        # DIMENSION 7: CONSISTENCY (5%)
        # =====================================================================
        consistency_score, con_failures, con_warnings, con_strengths = \
            self._grade_consistency(stats, t)
        critical_failures.extend(con_failures)
        warnings.extend(con_warnings)
        strengths.extend(con_strengths)

        # =====================================================================
        # CALCULATE FINAL SCORE
        # =====================================================================
        final_score = (
            prediction_score * self.weights["prediction"] +
            risk_score * self.weights["risk"] +
            trading_score * self.weights["trading"] +
            experience_score * self.weights["experience"] +
            capability_score * self.weights["capability"] +
            calibration_score * self.weights["calibration"] +
            consistency_score * self.weights["consistency"]
        )

        # =====================================================================
        # DETERMINE GRADE
        # =====================================================================
        grade, promoted, promotion_level, action = self._determine_grade(
            final_score, critical_failures, warnings, category,
        )

        # =====================================================================
        # GENERATE IMPROVEMENT AREAS
        # =====================================================================
        improvement_areas = self._identify_improvements(
            prediction_score, risk_score, trading_score,
            experience_score, capability_score,
            calibration_score, consistency_score,
        )

        # =====================================================================
        # CREATE REPORT
        # =====================================================================
        report = AgentGradeReport(
            agent_id=agent_id,
            agent_name=agent_name,
            category=category,
            grade=grade,
            grade_score=final_score,
            promoted=promoted,
            promotion_level=promotion_level,
            prediction_score=prediction_score,
            risk_score=risk_score,
            trading_score=trading_score,
            experience_score=experience_score,
            capability_score=capability_score,
            calibration_score=calibration_score,
            consistency_score=consistency_score,
            metrics=stats,
            critical_failures=critical_failures,
            warnings=warnings,
            strengths=strengths,
            action_required=action,
            improvement_areas=improvement_areas,
        )

        self.grading_history.append(report)

        return report

    def _grade_prediction(
        self, stats: Dict, t: InstitutionalThresholds,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade prediction quality."""
        failures, warnings, strengths = [], [], []
        scores = []

        # AUC
        auc = stats.get("auc", 0.5)
        if auc < t.min_auc_training:
            failures.append(f"CRITICAL: AUC {auc:.3f} < {t.min_auc_training} (random)")
        elif auc > t.max_auc_suspicious:
            warnings.append(f"SUSPICIOUS: AUC {auc:.3f} > {t.max_auc_suspicious} (overfitting?)")
        elif auc >= t.min_auc_live:
            strengths.append(f"EXCELLENT AUC: {auc:.3f} (live-ready)")

        auc_score = min(100, max(0, (auc - 0.5) / (0.65 - 0.5) * 100))
        scores.append(auc_score)

        # Accuracy
        accuracy = stats.get("accuracy", 0.5)
        if accuracy < t.min_accuracy_training:
            failures.append(f"CRITICAL: Accuracy {accuracy:.1%} < {t.min_accuracy_training:.0%}")
        elif accuracy >= t.min_accuracy_live:
            strengths.append(f"EXCELLENT Accuracy: {accuracy:.1%}")

        acc_score = min(100, max(0, (accuracy - 0.5) / (0.60 - 0.5) * 100))
        scores.append(acc_score)

        # Precision
        precision = stats.get("precision", 0.5)
        if precision < t.min_precision:
            warnings.append(f"LOW Precision: {precision:.1%} < {t.min_precision:.0%}")
        elif precision >= 0.60:
            strengths.append(f"HIGH Precision: {precision:.1%}")

        prec_score = min(100, max(0, (precision - 0.5) / (0.65 - 0.5) * 100))
        scores.append(prec_score)

        return sum(scores) / len(scores), failures, warnings, strengths

    def _grade_risk(
        self, stats: Dict, t: InstitutionalThresholds,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade risk management."""
        failures, warnings, strengths = [], [], []
        scores = []

        # Sharpe Ratio
        sharpe = stats.get("sharpe_ratio", stats.get("sharpe", 0))
        if sharpe < t.min_sharpe_training:
            failures.append(f"CRITICAL: Sharpe {sharpe:.2f} < {t.min_sharpe_training}")
        elif sharpe >= t.min_sharpe_live:
            strengths.append(f"EXCELLENT Sharpe: {sharpe:.2f} (live-ready)")

        sharpe_score = min(100, max(0, sharpe / 2.5 * 100))
        scores.append(sharpe_score)

        # Max Drawdown
        drawdown = stats.get("max_drawdown", 1.0)
        if drawdown > t.max_drawdown_training:
            failures.append(f"CRITICAL: Drawdown {drawdown:.1%} > {t.max_drawdown_training:.0%}")
        elif drawdown <= t.max_drawdown_live:
            strengths.append(f"LOW Drawdown: {drawdown:.1%}")

        dd_score = min(100, max(0, (1 - drawdown / 0.20) * 100))
        scores.append(dd_score)

        # Sortino Ratio
        sortino = stats.get("sortino_ratio", stats.get("sortino", 0))
        if sortino >= t.min_sortino:
            strengths.append(f"EXCELLENT Sortino: {sortino:.2f}")

        sortino_score = min(100, max(0, sortino / 3.0 * 100))
        scores.append(sortino_score)

        # VaR
        var = stats.get("var_95", stats.get("var", 0.05))
        if var > t.max_var_95:
            warnings.append(f"HIGH VaR: {var:.1%} > {t.max_var_95:.0%}")

        var_score = min(100, max(0, (1 - var / 0.05) * 100))
        scores.append(var_score)

        return sum(scores) / len(scores), failures, warnings, strengths

    def _grade_trading(
        self, stats: Dict, t: InstitutionalThresholds,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade trading performance."""
        failures, warnings, strengths = [], [], []
        scores = []

        # Win Rate
        win_rate = stats.get("win_rate", 0.5)
        if win_rate < t.min_win_rate:
            warnings.append(f"LOW Win Rate: {win_rate:.1%} < {t.min_win_rate:.0%}")
        elif win_rate >= 0.55:
            strengths.append(f"HIGH Win Rate: {win_rate:.1%}")

        wr_score = min(100, max(0, (win_rate - 0.45) / (0.60 - 0.45) * 100))
        scores.append(wr_score)

        # Profit Factor
        pf = stats.get("profit_factor", 1.0)
        if pf < t.min_profit_factor:
            warnings.append(f"LOW Profit Factor: {pf:.2f} < {t.min_profit_factor}")
        elif pf >= 2.0:
            strengths.append(f"EXCELLENT Profit Factor: {pf:.2f}")

        pf_score = min(100, max(0, (pf - 1.0) / (2.5 - 1.0) * 100))
        scores.append(pf_score)

        # Trade Count
        trades = stats.get("total_trades", stats.get("execution_count", 0))
        if trades < t.min_trades_training:
            failures.append(f"INSUFFICIENT TRADES: {trades} < {t.min_trades_training}")
        elif trades >= t.min_trades_live:
            strengths.append(f"WELL-TESTED: {trades} trades")

        trade_score = min(100, max(0, trades / t.min_trades_live * 100))
        scores.append(trade_score)

        return sum(scores) / len(scores), failures, warnings, strengths

    def _grade_experience(
        self, stats: Dict, t: InstitutionalThresholds,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade battle experience."""
        failures, warnings, strengths = [], [], []
        scores = []

        battle_stats = stats.get("battle_stats", {})

        # Crashes Survived
        crashes = battle_stats.get("crashes_survived", 0)
        if crashes >= t.min_crashes_survived:
            strengths.append(f"RESILIENT: Survived {crashes} crashes")
        else:
            warnings.append(f"UNTESTED: Only {crashes} crashes (need {t.min_crashes_survived})")
        crash_score = min(100, crashes / t.min_crashes_survived * 100)
        scores.append(crash_score)

        # Drawdowns Navigated
        drawdowns = battle_stats.get("drawdowns_navigated", 0)
        if drawdowns >= t.min_drawdowns_navigated:
            strengths.append(f"BATTLE-HARDENED: Navigated {drawdowns} drawdowns")
        else:
            warnings.append(f"INEXPERIENCED: Only {drawdowns} drawdowns")
        dd_score = min(100, drawdowns / t.min_drawdowns_navigated * 100)
        scores.append(dd_score)

        # Regime Changes
        regimes = battle_stats.get("regime_changes_adapted", 0)
        if regimes >= t.min_regime_changes:
            strengths.append(f"ADAPTIVE: Handled {regimes} regime changes")
        regime_score = min(100, regimes / t.min_regime_changes * 100)
        scores.append(regime_score)

        # Black Swans
        swans = battle_stats.get("black_swans_handled", 0)
        if swans >= t.min_black_swans:
            strengths.append(f"ANTIFRAGILE: Survived {swans} black swans")
        else:
            warnings.append("FRAGILE: No black swan experience")
        swan_score = min(100, swans / t.min_black_swans * 100) if t.min_black_swans > 0 else 50
        scores.append(swan_score)

        # Learning Events
        learning = stats.get("learning", {})
        events = learning.get("total_outcomes", 0)
        if events < t.min_learning_events:
            warnings.append(f"INSUFFICIENT LEARNING: {events} < {t.min_learning_events}")
        learning_score = min(100, events / t.min_learning_events * 100)
        scores.append(learning_score)

        return sum(scores) / len(scores) if scores else 50, failures, warnings, strengths

    def _grade_capabilities(
        self, stats: Dict, t: InstitutionalThresholds, category: AgentCategory,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade capabilities."""
        failures, warnings, strengths = [], [], []

        capabilities = stats.get("capabilities", [])
        cap_count = len(capabilities) if isinstance(capabilities, list) else 0

        # Category-specific thresholds
        if category == AgentCategory.MASTER:
            required = t.min_capabilities_master
        elif category == AgentCategory.SENIOR:
            required = t.min_capabilities_senior
        else:
            required = t.min_capabilities_standard

        if cap_count < required:
            warnings.append(f"UNDERPOWERED: {cap_count} capabilities < {required}")
        elif cap_count >= required * 1.5:
            strengths.append(f"VERSATILE: {cap_count} capabilities")

        score = min(100, cap_count / required * 100)

        return score, failures, warnings, strengths

    def _grade_calibration(
        self, stats: Dict, t: InstitutionalThresholds,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade confidence calibration."""
        failures, warnings, strengths = [], [], []

        calibration = stats.get("confidence_calibration", stats.get("calibration", 0.8))
        overconfidence = stats.get("overconfidence", 0.1)

        if calibration < t.min_confidence_calibration:
            warnings.append(f"POORLY CALIBRATED: {calibration:.0%}")
        elif calibration >= 0.90:
            strengths.append(f"WELL CALIBRATED: {calibration:.0%}")

        if overconfidence > t.max_overconfidence:
            failures.append(f"OVERCONFIDENT: {overconfidence:.0%} > {t.max_overconfidence:.0%}")

        score = min(100, calibration * 100)

        return score, failures, warnings, strengths

    def _grade_consistency(
        self, stats: Dict, t: InstitutionalThresholds,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """Grade consistency over time."""
        failures, warnings, strengths = [], [], []
        scores = []

        # CV Standard Deviation
        cv_std = stats.get("cv_std", stats.get("cv_auc_std", 0.03))
        if cv_std > t.max_cv_std:
            warnings.append(f"INCONSISTENT: CV std {cv_std:.2%} > {t.max_cv_std:.0%}")
        elif cv_std <= 0.03:
            strengths.append(f"STABLE: CV std {cv_std:.2%}")
        cv_score = min(100, max(0, (1 - cv_std / 0.10) * 100))
        scores.append(cv_score)

        # Out-of-Sample Degradation
        oos_deg = stats.get("oos_degradation", 0.05)
        if oos_deg > t.max_oos_degradation:
            failures.append(f"OVERFITTING: OOS {oos_deg:.1%} worse than train")
        oos_score = min(100, max(0, (1 - oos_deg / 0.15) * 100))
        scores.append(oos_score)

        return sum(scores) / len(scores) if scores else 75, failures, warnings, strengths

    def _determine_grade(
        self,
        score: float,
        critical_failures: List[str],
        warnings: List[str],
        category: AgentCategory,
    ) -> Tuple[GradeLevel, bool, str, str]:
        """Determine final grade and promotion status."""
        # ANY critical failure = immediate action
        if critical_failures:
            if len(critical_failures) >= 3:
                return GradeLevel.F, False, "BLACKLISTED", \
                    f"BLACKLIST: {len(critical_failures)} critical failures"
            elif len(critical_failures) >= 1:
                return GradeLevel.D, False, "TERMINATED", \
                    f"TERMINATE: {critical_failures[0]}"

        # Many warnings = probation
        if len(warnings) >= 5:
            return GradeLevel.C, False, "PROBATION", \
                f"PROBATION: {len(warnings)} warnings - 24hr to improve"

        # Score-based grading
        if score >= 95:
            return GradeLevel.S_PLUS, True, "LIVE", \
                "CITADEL ELITE: Immediate live deployment"
        elif score >= 90:
            return GradeLevel.S, True, "LIVE", \
                "GOLDMAN TIER: Ready for live trading"
        elif score >= 85:
            return GradeLevel.A_PLUS, True, "LIVE", \
                "INSTITUTIONAL: Approved for live trading"
        elif score >= 80:
            return GradeLevel.A, True, "PAPER", \
                "INSTITUTIONAL: Approved for paper trading"
        elif score >= 70:
            return GradeLevel.B_PLUS, False, "TRAINING", \
                "PROMISING: Continue training with priority"
        elif score >= 60:
            return GradeLevel.B, False, "TRAINING", \
                "ACCEPTABLE: Continue standard training"
        else:
            return GradeLevel.C, False, "PROBATION", \
                f"BELOW STANDARD: Score {score:.0f}/100 needs work"

    def _identify_improvements(
        self, pred: float, risk: float, trade: float,
        exp: float, cap: float, cal: float, con: float,
    ) -> List[str]:
        """Identify areas needing improvement."""
        improvements = []

        scores = [
            ("Prediction Quality", pred),
            ("Risk Management", risk),
            ("Trading Performance", trade),
            ("Battle Experience", exp),
            ("Capabilities", cap),
            ("Calibration", cal),
            ("Consistency", con),
        ]

        # Sort by lowest score
        scores.sort(key=lambda x: x[1])

        for name, score in scores[:3]:  # Top 3 weakest areas
            if score < 70:
                improvements.append(f"CRITICAL: {name} ({score:.0f}/100)")
            elif score < 80:
                improvements.append(f"IMPROVE: {name} ({score:.0f}/100)")

        return improvements


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def grade_agent_institutional(
    agent_id: str,
    agent_name: str,
    category: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Convenience wrapper for institutional grading.

    Usage:
        from src.core.institutional_grading import grade_agent_institutional

        result = grade_agent_institutional(
            agent_id="ghost_001",
            agent_name="GhostAgent",
            category="master",
            stats=agent.get_stats()
        )

        print(f"Grade: {result['grade']}")
        print(f"Promoted: {result['promoted']}")
        print(f"Level: {result['promotion_level']}")
    """
    grader = InstitutionalGrader()

    cat = AgentCategory(category.lower())

    report = grader.grade_agent(
        agent_id=agent_id,
        agent_name=agent_name,
        category=cat,
        stats=stats,
    )

    return report.to_dict()


def get_grade_requirements() -> Dict[str, Any]:
    """Get all grading requirements for documentation."""
    t = InstitutionalThresholds()

    return {
        "prediction": {
            "auc_live": t.min_auc_live,
            "auc_paper": t.min_auc_paper,
            "auc_training": t.min_auc_training,
            "accuracy_live": t.min_accuracy_live,
            "accuracy_paper": t.min_accuracy_paper,
            "precision": t.min_precision,
        },
        "risk": {
            "sharpe_live": t.min_sharpe_live,
            "sharpe_paper": t.min_sharpe_paper,
            "sortino": t.min_sortino,
            "calmar": t.min_calmar,
            "max_drawdown_live": t.max_drawdown_live,
            "max_drawdown_paper": t.max_drawdown_paper,
            "var_95": t.max_var_95,
        },
        "trading": {
            "win_rate": t.min_win_rate,
            "profit_factor": t.min_profit_factor,
            "trades_live": t.min_trades_live,
            "trades_paper": t.min_trades_paper,
        },
        "experience": {
            "crashes_survived": t.min_crashes_survived,
            "drawdowns_navigated": t.min_drawdowns_navigated,
            "regime_changes": t.min_regime_changes,
            "black_swans": t.min_black_swans,
            "learning_events": t.min_learning_events,
        },
        "grade_scale": {
            "S+": "95+ (Citadel Elite - Top 0.1%)",
            "S": "90-94 (Goldman/Renaissance - Top 1%)",
            "A+": "85-89 (Institutional - Live Trading)",
            "A": "80-84 (Institutional - Paper Trading)",
            "B+": "70-79 (Promising - Priority Training)",
            "B": "60-69 (Acceptable - Standard Training)",
            "C": "50-59 (Probation - 24hr to improve)",
            "D": "<50 (Terminated)",
            "F": "Critical Failures (Blacklisted)",
        },
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("INSTITUTIONAL GRADING SYSTEM - REQUIREMENTS")
    print("=" * 70)

    reqs = get_grade_requirements()

    print("\n📊 PREDICTION REQUIREMENTS:")
    for k, v in reqs["prediction"].items():
        print(f"  {k}: {v}")

    print("\n⚠️ RISK REQUIREMENTS:")
    for k, v in reqs["risk"].items():
        print(f"  {k}: {v}")

    print("\n💹 TRADING REQUIREMENTS:")
    for k, v in reqs["trading"].items():
        print(f"  {k}: {v}")

    print("\n🎖️ EXPERIENCE REQUIREMENTS:")
    for k, v in reqs["experience"].items():
        print(f"  {k}: {v}")

    print("\n📋 GRADE SCALE:")
    for grade, desc in reqs["grade_scale"].items():
        print(f"  {grade}: {desc}")

    print("\n" + "=" * 70)
    print("Only the ELITE survive. No participation trophies.")
    print("=" * 70)


