"""
================================================================================
INSTITUTIONAL-GRADE AGENT GRADING SYSTEM
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

This grading system is built to compete with Citadel, Goldman Sachs, Two Sigma,
and Renaissance Technologies. If you're not getting A grades, you're not ready.

NO EASY PASSES. NO PARTICIPATION TROPHIES.

Grading Philosophy:
- An A grade means you can DESTROY the competition
- A B grade means you might survive
- A C grade means you need serious work
- A D/F grade means start over

By end of 2026, they will know Alpha Loop Capital.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from enum import Enum


class GradeTier(Enum):
    """Grade tiers - each has specific requirements."""
    A_PLUS = "A+"   # Elite - Top 1% hedge fund quality
    A = "A"         # Excellent - Institutional quality
    B_PLUS = "B+"   # Good - Needs polish
    B = "B"         # Acceptable - Significant gaps
    C = "C"         # Below standard - Major issues
    D = "D"         # Failing - Not production ready
    F = "F"         # Failed completely


@dataclass(frozen=True)
class InstitutionalThresholds:
    """
    INSTITUTIONAL-GRADE THRESHOLDS - CITADEL/GOLDMAN LEVEL
    
    These are HARD. If you can't meet them, you can't compete.
    """
    # =========================================================================
    # MODEL PERFORMANCE (ML Agents)
    # =========================================================================
    min_auc: float = 0.54              # Must beat random SIGNIFICANTLY
    elite_auc: float = 0.58            # Elite threshold
    min_accuracy: float = 0.53         # Consistent edge required
    elite_accuracy: float = 0.57       # Elite threshold
    min_sharpe: float = 1.5            # Risk-adjusted minimum
    elite_sharpe: float = 2.5          # Elite Sharpe ratio
    max_drawdown: float = 0.08         # 8% max - STRICT risk control
    elite_drawdown: float = 0.05       # Elite - 5% max drawdown
    min_win_rate: float = 0.52         # Must win more than lose
    elite_win_rate: float = 0.58       # Elite win rate
    min_profit_factor: float = 1.2     # Profits > Losses
    elite_profit_factor: float = 1.8   # Elite profit factor
    
    # =========================================================================
    # AGENT EXECUTION (Operational Agents)
    # =========================================================================
    min_success_rate: float = 0.90     # 90% success required
    elite_success_rate: float = 0.98   # Elite - near perfect
    min_executions: int = 100          # Need statistical significance
    elite_executions: int = 1000       # Elite - battle-tested
    min_capabilities: int = 5          # Minimum skill breadth
    elite_capabilities: int = 15       # Elite - comprehensive skills
    
    # =========================================================================
    # LEARNING & ADAPTATION
    # =========================================================================
    min_learning_events: int = 100     # Must learn from experience
    elite_learning_events: int = 1000  # Elite - extensive learning
    min_adaptation_rate: float = 0.7   # How fast agent adapts
    elite_adaptation_rate: float = 0.9 # Elite adaptation
    min_regime_accuracy: float = 0.6   # Regime detection accuracy
    elite_regime_accuracy: float = 0.8 # Elite regime detection
    
    # =========================================================================
    # BATTLE STATS (Toughness Metrics)
    # =========================================================================
    min_crashes_survived: int = 5      # Must handle failures
    elite_crashes_survived: int = 50   # Elite - battle-hardened
    min_drawdowns_navigated: int = 3   # Experience with adversity
    elite_drawdowns_navigated: int = 20
    min_regime_changes: int = 2        # Adapted to regime changes
    elite_regime_changes: int = 10     # Elite - fully adaptive
    min_black_swans: int = 0           # Handled black swan events
    elite_black_swans: int = 3         # Elite - crisis-tested
    
    # =========================================================================
    # UNIQUE ALPHA (What Makes You Different)
    # =========================================================================
    min_unique_insights: int = 10      # Original thinking required
    elite_unique_insights: int = 100   # Elite - innovation machine
    min_contrarian_wins: int = 5       # Being right when others wrong
    elite_contrarian_wins: int = 50    # Elite contrarian
    min_information_edges: int = 3     # Proprietary information
    elite_information_edges: int = 30  # Elite - information advantage
    
    # =========================================================================
    # COMPETITIVE BENCHMARKS (vs Industry)
    # =========================================================================
    spy_outperformance_min: float = 0.02   # Must beat SPY by 2%
    spy_outperformance_elite: float = 0.10 # Elite - 10% over SPY
    peer_percentile_min: float = 0.60      # Top 40% vs peers
    peer_percentile_elite: float = 0.95    # Elite - Top 5%


@dataclass
class AgentGradeReport:
    """Comprehensive grade report for an agent."""
    agent_name: str
    grade: GradeTier
    grade_letter: str
    numeric_score: float  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Category scores (0-100 each)
    performance_score: float = 0.0
    execution_score: float = 0.0
    learning_score: float = 0.0
    battle_score: float = 0.0
    alpha_score: float = 0.0
    competitive_score: float = 0.0
    
    # Details
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    improvement_actions: List[str] = field(default_factory=list)
    
    # Competitive analysis
    vs_citadel: str = "UNKNOWN"
    vs_goldman: str = "UNKNOWN"
    vs_twosigma: str = "UNKNOWN"
    
    # Production readiness
    production_ready: bool = False
    paper_trading_ready: bool = False
    needs_retraining: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "grade": self.grade_letter,
            "grade_tier": self.grade.value,
            "numeric_score": self.numeric_score,
            "timestamp": self.timestamp.isoformat(),
            "category_scores": {
                "performance": self.performance_score,
                "execution": self.execution_score,
                "learning": self.learning_score,
                "battle": self.battle_score,
                "alpha": self.alpha_score,
                "competitive": self.competitive_score,
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "critical_failures": self.critical_failures,
            "improvement_actions": self.improvement_actions,
            "competitive_analysis": {
                "vs_citadel": self.vs_citadel,
                "vs_goldman": self.vs_goldman,
                "vs_twosigma": self.vs_twosigma,
            },
            "production_ready": self.production_ready,
            "paper_trading_ready": self.paper_trading_ready,
            "needs_retraining": self.needs_retraining,
        }


class InstitutionalGrader:
    """
    INSTITUTIONAL-GRADE AGENT GRADING SYSTEM
    
    This grader is HARSH. It evaluates agents against hedge fund standards.
    If you're not getting A grades, you're not competing with the big boys.
    
    Grading Categories (equally weighted):
    1. PERFORMANCE - Raw predictive ability
    2. EXECUTION - Operational excellence
    3. LEARNING - Continuous improvement
    4. BATTLE - Resilience under pressure
    5. ALPHA - Unique insights and edge
    6. COMPETITIVE - How you stack vs industry
    """
    
    def __init__(self, thresholds: InstitutionalThresholds = None):
        self.thresholds = thresholds or InstitutionalThresholds()
        
    def grade_agent(self, stats: Dict[str, Any]) -> AgentGradeReport:
        """
        Comprehensively grade an agent.
        
        Args:
            stats: Agent statistics dictionary
            
        Returns:
            AgentGradeReport with detailed grading
        """
        agent_name = stats.get("name", "UnknownAgent")
        
        # Calculate category scores
        perf_score = self._grade_performance(stats)
        exec_score = self._grade_execution(stats)
        learn_score = self._grade_learning(stats)
        battle_score = self._grade_battle(stats)
        alpha_score = self._grade_alpha(stats)
        comp_score = self._grade_competitive(stats)
        
        # Calculate overall numeric score (0-100)
        numeric_score = (
            perf_score * 0.25 +      # Performance is critical
            exec_score * 0.15 +      # Execution matters
            learn_score * 0.15 +     # Learning is key
            battle_score * 0.15 +    # Resilience required
            alpha_score * 0.20 +     # Unique edge is gold
            comp_score * 0.10        # Must beat competition
        )
        
        # Determine grade tier
        grade = self._numeric_to_grade(numeric_score)
        
        # Get critical failures
        critical = self._identify_critical_failures(stats)
        
        # If ANY critical failures, cap grade at C
        if critical:
            if grade in [GradeTier.A_PLUS, GradeTier.A, GradeTier.B_PLUS, GradeTier.B]:
                grade = GradeTier.C
                numeric_score = min(numeric_score, 69)
        
        # Build report
        report = AgentGradeReport(
            agent_name=agent_name,
            grade=grade,
            grade_letter=grade.value,
            numeric_score=round(numeric_score, 1),
            performance_score=round(perf_score, 1),
            execution_score=round(exec_score, 1),
            learning_score=round(learn_score, 1),
            battle_score=round(battle_score, 1),
            alpha_score=round(alpha_score, 1),
            competitive_score=round(comp_score, 1),
            strengths=self._identify_strengths(stats),
            weaknesses=self._identify_weaknesses(stats),
            critical_failures=critical,
            improvement_actions=self._generate_improvements(stats),
            vs_citadel=self._compare_to_citadel(numeric_score),
            vs_goldman=self._compare_to_goldman(numeric_score),
            vs_twosigma=self._compare_to_twosigma(numeric_score),
            production_ready=(grade in [GradeTier.A_PLUS, GradeTier.A] and not critical),
            paper_trading_ready=(grade in [GradeTier.A_PLUS, GradeTier.A, GradeTier.B_PLUS, GradeTier.B] and not critical),
            needs_retraining=(grade in [GradeTier.C, GradeTier.D, GradeTier.F] or bool(critical)),
        )
        
        return report
    
    def _grade_performance(self, stats: Dict[str, Any]) -> float:
        """Grade model/agent performance (0-100)."""
        t = self.thresholds
        score = 50  # Start at baseline
        
        # AUC scoring
        auc = self._parse_rate(stats.get("auc", stats.get("metrics", {}).get("auc")))
        if auc >= t.elite_auc:
            score += 15
        elif auc >= t.min_auc:
            score += 10 * ((auc - t.min_auc) / (t.elite_auc - t.min_auc))
        else:
            score -= 20  # Penalty for below minimum
        
        # Accuracy scoring
        acc = self._parse_rate(stats.get("accuracy", stats.get("metrics", {}).get("accuracy")))
        if acc >= t.elite_accuracy:
            score += 15
        elif acc >= t.min_accuracy:
            score += 10 * ((acc - t.min_accuracy) / (t.elite_accuracy - t.min_accuracy))
        else:
            score -= 20
        
        # Sharpe scoring
        sharpe = float(stats.get("sharpe", stats.get("metrics", {}).get("sharpe", 0)) or 0)
        if sharpe >= t.elite_sharpe:
            score += 15
        elif sharpe >= t.min_sharpe:
            score += 10 * ((sharpe - t.min_sharpe) / (t.elite_sharpe - t.min_sharpe))
        else:
            score -= 15
        
        # Max drawdown scoring (lower is better)
        dd = float(stats.get("max_drawdown", stats.get("metrics", {}).get("max_drawdown", 0.1)) or 0.1)
        if dd <= t.elite_drawdown:
            score += 10
        elif dd <= t.max_drawdown:
            score += 5 * ((t.max_drawdown - dd) / (t.max_drawdown - t.elite_drawdown))
        else:
            score -= 15  # Penalty for excessive drawdown
        
        return max(0, min(100, score))
    
    def _grade_execution(self, stats: Dict[str, Any]) -> float:
        """Grade execution quality (0-100)."""
        t = self.thresholds
        score = 50
        
        # Success rate
        success_rate = self._parse_rate(stats.get("success_rate"))
        if success_rate >= t.elite_success_rate:
            score += 20
        elif success_rate >= t.min_success_rate:
            score += 15 * ((success_rate - t.min_success_rate) / (t.elite_success_rate - t.min_success_rate))
        else:
            score -= 25
        
        # Execution count
        exec_count = int(stats.get("execution_count", 0) or 0)
        if exec_count >= t.elite_executions:
            score += 15
        elif exec_count >= t.min_executions:
            score += 10 * ((exec_count - t.min_executions) / (t.elite_executions - t.min_executions))
        else:
            score -= 15
        
        # Capabilities breadth
        caps = len(stats.get("capabilities", []))
        if caps >= t.elite_capabilities:
            score += 15
        elif caps >= t.min_capabilities:
            score += 10 * ((caps - t.min_capabilities) / (t.elite_capabilities - t.min_capabilities))
        else:
            score -= 10
        
        return max(0, min(100, score))
    
    def _grade_learning(self, stats: Dict[str, Any]) -> float:
        """Grade learning and adaptation (0-100)."""
        t = self.thresholds
        score = 50
        
        learning = stats.get("learning", {})
        
        # Learning events
        events = int(learning.get("total_outcomes", 0) or 0)
        if events >= t.elite_learning_events:
            score += 20
        elif events >= t.min_learning_events:
            score += 15 * ((events - t.min_learning_events) / (t.elite_learning_events - t.min_learning_events))
        else:
            score -= 20
        
        # Recent accuracy (learning effectiveness)
        recent_acc = float(learning.get("recent_accuracy", 0.5) or 0.5)
        if recent_acc >= 0.7:
            score += 15
        elif recent_acc >= 0.55:
            score += 10
        elif recent_acc >= 0.5:
            score += 0
        else:
            score -= 15  # Degrading performance
        
        # Adaptations made
        adaptations = int(learning.get("adaptations", 0) or 0)
        if adaptations >= 10:
            score += 15
        elif adaptations >= 3:
            score += 10
        elif adaptations >= 1:
            score += 5
        else:
            score -= 5  # No adaptation
        
        return max(0, min(100, score))
    
    def _grade_battle(self, stats: Dict[str, Any]) -> float:
        """Grade battle-hardiness and resilience (0-100)."""
        t = self.thresholds
        score = 50
        
        battle = stats.get("battle_stats", {})
        
        # Crashes survived
        crashes = int(battle.get("crashes_survived", 0) or 0)
        if crashes >= t.elite_crashes_survived:
            score += 15
        elif crashes >= t.min_crashes_survived:
            score += 10
        else:
            score -= 5  # Untested under pressure
        
        # Drawdowns navigated
        dds = int(battle.get("drawdowns_navigated", 0) or 0)
        if dds >= t.elite_drawdowns_navigated:
            score += 15
        elif dds >= t.min_drawdowns_navigated:
            score += 10
        else:
            score -= 5
        
        # Regime changes adapted
        regimes = int(battle.get("regime_changes_adapted", 0) or 0)
        if regimes >= t.elite_regime_changes:
            score += 15
        elif regimes >= t.min_regime_changes:
            score += 10
        else:
            score -= 10  # Not regime-adaptive
        
        # Black swans handled
        swans = int(battle.get("black_swans_handled", 0) or 0)
        if swans >= t.elite_black_swans:
            score += 10
        elif swans >= t.min_black_swans:
            score += 5
        
        return max(0, min(100, score))
    
    def _grade_alpha(self, stats: Dict[str, Any]) -> float:
        """Grade unique alpha generation (0-100) - THIS IS WHAT MAKES YOU DIFFERENT."""
        t = self.thresholds
        score = 50
        
        battle = stats.get("battle_stats", {})
        
        # Unique insights generated
        insights = int(battle.get("unique_insights", 0) or 0)
        if insights >= t.elite_unique_insights:
            score += 25  # Major bonus for innovation
        elif insights >= t.min_unique_insights:
            score += 15
        else:
            score -= 20  # Not innovative enough
        
        # Contrarian wins (being right when others wrong)
        contrarian = int(stats.get("contrarian_wins", 0) or 0)
        if contrarian >= t.elite_contrarian_wins:
            score += 20
        elif contrarian >= t.min_contrarian_wins:
            score += 10
        else:
            score -= 10  # Following the herd
        
        # Information edges exploited
        edges = int(battle.get("information_edges", 0) or 0)
        if edges >= t.elite_information_edges:
            score += 15
        elif edges >= t.min_information_edges:
            score += 10
        else:
            score -= 5
        
        return max(0, min(100, score))
    
    def _grade_competitive(self, stats: Dict[str, Any]) -> float:
        """Grade competitive positioning vs industry (0-100)."""
        t = self.thresholds
        score = 50
        
        # SPY outperformance
        spy_beat = float(stats.get("spy_outperformance", 0) or 0)
        if spy_beat >= t.spy_outperformance_elite:
            score += 25
        elif spy_beat >= t.spy_outperformance_min:
            score += 15
        elif spy_beat > 0:
            score += 5
        else:
            score -= 20  # Can't beat index? Why exist?
        
        # Peer percentile
        percentile = float(stats.get("peer_percentile", 0.5) or 0.5)
        if percentile >= t.peer_percentile_elite:
            score += 25
        elif percentile >= t.peer_percentile_min:
            score += 15
        else:
            score -= 10
        
        return max(0, min(100, score))
    
    def _numeric_to_grade(self, score: float) -> GradeTier:
        """Convert numeric score to grade tier."""
        if score >= 95:
            return GradeTier.A_PLUS
        elif score >= 85:
            return GradeTier.A
        elif score >= 80:
            return GradeTier.B_PLUS
        elif score >= 70:
            return GradeTier.B
        elif score >= 60:
            return GradeTier.C
        elif score >= 50:
            return GradeTier.D
        else:
            return GradeTier.F
    
    def _identify_critical_failures(self, stats: Dict[str, Any]) -> List[str]:
        """Identify critical failures that disqualify for production."""
        failures = []
        t = self.thresholds
        
        # AUC below absolute minimum
        auc = self._parse_rate(stats.get("auc", stats.get("metrics", {}).get("auc")))
        if auc < 0.51:
            failures.append(f"AUC {auc:.3f} is below absolute minimum (0.51) - no predictive ability")
        
        # Excessive drawdown
        dd = float(stats.get("max_drawdown", stats.get("metrics", {}).get("max_drawdown", 0)) or 0)
        if dd > 0.15:
            failures.append(f"Max drawdown {dd:.1%} exceeds critical limit (15%) - unacceptable risk")
        
        # Success rate critical failure
        sr = self._parse_rate(stats.get("success_rate"))
        if sr < 0.80 and stats.get("execution_count", 0) > 50:
            failures.append(f"Success rate {sr:.1%} is critically low (<80%)")
        
        # No learning
        learning = stats.get("learning", {})
        if learning.get("total_outcomes", 0) < 10 and stats.get("execution_count", 0) > 100:
            failures.append("Agent is not learning from outcomes - critical flaw")
        
        # Negative Sharpe
        sharpe = float(stats.get("sharpe", stats.get("metrics", {}).get("sharpe", 0)) or 0)
        if sharpe < 0:
            failures.append(f"Negative Sharpe ratio ({sharpe:.2f}) - losing money risk-adjusted")
        
        return failures
    
    def _identify_strengths(self, stats: Dict[str, Any]) -> List[str]:
        """Identify agent strengths."""
        strengths = []
        t = self.thresholds
        
        auc = self._parse_rate(stats.get("auc", stats.get("metrics", {}).get("auc")))
        if auc >= t.elite_auc:
            strengths.append(f"Elite AUC ({auc:.3f}) - exceptional predictive ability")
        
        sharpe = float(stats.get("sharpe", stats.get("metrics", {}).get("sharpe", 0)) or 0)
        if sharpe >= t.elite_sharpe:
            strengths.append(f"Elite Sharpe ratio ({sharpe:.2f}) - excellent risk-adjusted returns")
        
        insights = stats.get("battle_stats", {}).get("unique_insights", 0)
        if insights >= t.elite_unique_insights:
            strengths.append(f"High innovation ({insights} unique insights) - differentiated thinking")
        
        crashes = stats.get("battle_stats", {}).get("crashes_survived", 0)
        if crashes >= t.elite_crashes_survived:
            strengths.append(f"Battle-hardened ({crashes} crashes survived) - proven resilience")
        
        return strengths
    
    def _identify_weaknesses(self, stats: Dict[str, Any]) -> List[str]:
        """Identify agent weaknesses."""
        weaknesses = []
        t = self.thresholds
        
        auc = self._parse_rate(stats.get("auc", stats.get("metrics", {}).get("auc")))
        if auc < t.min_auc:
            weaknesses.append(f"AUC ({auc:.3f}) below minimum - improve model")
        
        exec_count = int(stats.get("execution_count", 0) or 0)
        if exec_count < t.min_executions:
            weaknesses.append(f"Only {exec_count} executions - needs more testing")
        
        insights = stats.get("battle_stats", {}).get("unique_insights", 0)
        if insights < t.min_unique_insights:
            weaknesses.append(f"Only {insights} unique insights - not differentiated enough")
        
        dd = float(stats.get("max_drawdown", stats.get("metrics", {}).get("max_drawdown", 0)) or 0)
        if dd > t.max_drawdown:
            weaknesses.append(f"Drawdown ({dd:.1%}) too high - tighten risk controls")
        
        return weaknesses
    
    def _generate_improvements(self, stats: Dict[str, Any]) -> List[str]:
        """Generate specific improvement actions."""
        actions = []
        
        auc = self._parse_rate(stats.get("auc", stats.get("metrics", {}).get("auc")))
        if auc < 0.54:
            actions.append("Add behavioral features (sentiment, fear/greed, herding)")
            actions.append("Increase training data - more history improves patterns")
        
        insights = stats.get("battle_stats", {}).get("unique_insights", 0)
        if insights < 10:
            actions.append("Enable all thinking modes (contrarian, second-order, regime-aware)")
            actions.append("Integrate cross-agent learning for novel insights")
        
        crashes = stats.get("battle_stats", {}).get("crashes_survived", 0)
        if crashes < 5:
            actions.append("Run stress tests and fault injection via NOBUS agent")
        
        regimes = stats.get("battle_stats", {}).get("regime_changes_adapted", 0)
        if regimes < 2:
            actions.append("Test across multiple market regimes (crisis, risk-on, risk-off)")
        
        return actions if actions else ["Continue current training - on track"]
    
    def _compare_to_citadel(self, score: float) -> str:
        """Compare to Citadel benchmark."""
        if score >= 90:
            return "COMPETITIVE - Can compete at Citadel level"
        elif score >= 75:
            return "APPROACHING - Close but not there yet"
        elif score >= 60:
            return "BEHIND - Significant gap to close"
        else:
            return "NOT COMPETITIVE - Major overhaul needed"
    
    def _compare_to_goldman(self, score: float) -> str:
        """Compare to Goldman benchmark."""
        if score >= 85:
            return "COMPETITIVE - Institutional quality"
        elif score >= 70:
            return "APPROACHING - Getting there"
        else:
            return "BEHIND - More work needed"
    
    def _compare_to_twosigma(self, score: float) -> str:
        """Compare to Two Sigma benchmark."""
        if score >= 92:
            return "COMPETITIVE - Quant fund quality"
        elif score >= 80:
            return "APPROACHING - Strong but not elite"
        else:
            return "BEHIND - Focus on unique alpha"
    
    @staticmethod
    def _parse_rate(rate_value: Any) -> float:
        """Convert '85.0%' or numeric into float 0-1."""
        if rate_value is None:
            return 0.0
        if isinstance(rate_value, str) and rate_value.endswith("%"):
            try:
                return float(rate_value.strip("%")) / 100.0
            except ValueError:
                return 0.0
        try:
            val = float(rate_value)
            # If value > 1, assume it's a percentage
            return val / 100.0 if val > 1 else val
        except (TypeError, ValueError):
            return 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def grade_agent_institutional(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper for institutional grading.
    
    Returns full grade report as dictionary.
    """
    grader = InstitutionalGrader()
    report = grader.grade_agent(stats)
    return report.to_dict()


def is_production_ready(stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Quick check: Is this agent production ready?
    
    Returns:
        (ready: bool, reason: str)
    """
    grader = InstitutionalGrader()
    report = grader.grade_agent(stats)
    
    if report.production_ready:
        return True, f"Grade {report.grade_letter} - Production ready"
    else:
        reasons = report.critical_failures if report.critical_failures else report.weaknesses[:2]
        return False, f"Grade {report.grade_letter} - Issues: {'; '.join(reasons)}"


def get_grade_summary(stats: Dict[str, Any]) -> str:
    """
    Get a one-line grade summary.
    """
    grader = InstitutionalGrader()
    report = grader.grade_agent(stats)
    
    return (
        f"{report.agent_name}: {report.grade_letter} ({report.numeric_score}/100) | "
        f"Perf:{report.performance_score:.0f} Exec:{report.execution_score:.0f} "
        f"Learn:{report.learning_score:.0f} Battle:{report.battle_score:.0f} "
        f"Alpha:{report.alpha_score:.0f} | "
        f"{'PRODUCTION READY' if report.production_ready else 'NOT READY'}"
    )


# Legacy compatibility
class GradeThresholds:
    """Legacy alias for backwards compatibility."""
    min_success_rate: float = 0.90
    min_executions: int = 100
    min_capabilities: int = 5
    min_learning_events: int = 100
    min_battle_stats: int = 5


class AgentGrader(InstitutionalGrader):
    """Legacy alias for backwards compatibility."""
    pass


def grade_agent_stats(stats: Dict[str, Any], thresholds=None) -> Dict[str, Any]:
    """Legacy wrapper for backwards compatibility."""
    return grade_agent_institutional(stats)
