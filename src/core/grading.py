"""
Agent grading utilities to enforce objective thresholds for institutional readiness.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class GradeThresholds:
    """Configurable thresholds for readiness grading."""

    min_success_rate: float = 0.8
    min_executions: int = 10
    min_capabilities: int = 3
    min_learning_events: int = 10
    min_battle_stats: int = 0


class AgentGrader:
    """Scores agents using hard thresholds, returning letter grades and reasons."""

    def __init__(self, thresholds: GradeThresholds | None = None):
        self.thresholds = thresholds or GradeThresholds()

    def grade(self, stats: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Compute a grade and metadata based on provided stats."""
        t = self.thresholds
        reasons = []

        # Normalize inputs
        success_rate = self._parse_rate(stats.get("success_rate"))
        executions = int(stats.get("execution_count", 0) or 0)
        capabilities = len(stats.get("capabilities", []))
        learning = stats.get("learning", {})
        learning_events = int(learning.get("total_outcomes", 0) or 0)
        battle = stats.get("battle_stats", {})
        battle_signal = sum(int(v or 0) for v in battle.values())

        # Evaluate thresholds
        if success_rate < t.min_success_rate:
            reasons.append(f"success_rate {success_rate:.1%} < {t.min_success_rate:.0%}")
        if executions < t.min_executions:
            reasons.append(f"executions {executions} < {t.min_executions}")
        if capabilities < t.min_capabilities:
            reasons.append(f"capabilities {capabilities} < {t.min_capabilities}")
        if learning_events < t.min_learning_events:
            reasons.append(f"learning_events {learning_events} < {t.min_learning_events}")
        if battle_signal < t.min_battle_stats:
            reasons.append("battle stats insufficient")

        # Grade mapping
        if not reasons:
            grade = "A"
        elif len(reasons) <= 2:
            grade = "B"
        elif len(reasons) <= 4:
            grade = "C"
        else:
            grade = "D"

        return grade, {
            "success_rate": success_rate,
            "executions": executions,
            "capabilities": capabilities,
            "learning_events": learning_events,
            "battle_signal": battle_signal,
            "issues": reasons,
        }

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
            return float(rate_value)
        except (TypeError, ValueError):
            return 0.0


def grade_agent_stats(stats: Dict[str, Any], thresholds: GradeThresholds | None = None) -> Dict[str, Any]:
    """Convenience wrapper."""
    grader = AgentGrader(thresholds)
    grade, meta = grader.grade(stats)
    return {"grade": grade, **meta}

