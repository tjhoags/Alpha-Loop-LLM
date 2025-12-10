"""================================================================================
ALC-ALGO LEARNING OPTIMIZER - MAXIMIZE LEARNING SPEED AND EFFICIENCY
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

This module provides advanced learning optimization for all agents:
- Prioritized experience replay
- Adaptive learning rates
- Meta-learning coordination
- Cross-machine learning synchronization
- Learning efficiency tracking

PHILOSOPHY: MAXIMUM LEARNING, NO LIMITS ON COMPUTE
================================================================================
"""

import logging
import platform
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ALC.LearningOptimizer")


@dataclass
class LearningExperience:
    """A single learning experience with priority scoring."""

    experience_id: str
    timestamp: datetime
    agent_name: str
    machine_id: str
    experience_type: str  # trade, prediction, regime_change, mistake
    data: Dict[str, Any]
    outcome: Optional[Any] = None
    priority: float = 1.0
    times_replayed: int = 0
    td_error: float = 0.0  # Temporal difference error


@dataclass
class LearningSession:
    """Track a learning session across machines."""

    session_id: str
    start_time: datetime
    machine_id: str
    agent_name: str
    experiences_processed: int = 0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    learning_rate: float = 0.01
    adaptations_made: int = 0


class PrioritizedExperienceReplay:
    """Prioritized Experience Replay (PER) for agent learning.

    Higher priority experiences (surprising outcomes, mistakes) are
    replayed more frequently to accelerate learning.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """Initialize PER buffer.

        Args:
        ----
            capacity: Maximum experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Increase beta over time

        self.experiences: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, experience: LearningExperience, priority: Optional[float] = None):
        """Add experience with priority."""
        if priority is None:
            priority = self.max_priority

        self.experiences.append(experience)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size: int) -> Tuple[List[LearningExperience], List[float]]:
        """Sample batch with prioritized sampling.

        Returns experiences and importance sampling weights.
        """
        if len(self.experiences) < batch_size:
            batch_size = len(self.experiences)

        # Convert to probabilities
        total_priority = sum(self.priorities)
        probabilities = [p / total_priority for p in self.priorities]

        # Sample indices based on priorities
        indices = random.choices(range(len(self.experiences)), weights=probabilities, k=batch_size)

        # Calculate importance sampling weights
        min_prob = min(probabilities)
        max_weight = (min_prob * len(self.experiences)) ** (-self.beta)

        experiences = []
        weights = []

        for idx in indices:
            exp = self.experiences[idx]
            exp.times_replayed += 1
            experiences.append(exp)

            # Importance sampling weight
            weight = (probabilities[idx] * len(self.experiences)) ** (-self.beta)
            weights.append(weight / max_weight)

        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, weights

    def update_priority(self, experience_id: str, new_priority: float):
        """Update priority for an experience."""
        for i, exp in enumerate(self.experiences):
            if exp.experience_id == experience_id:
                self.priorities[i] = new_priority ** self.alpha
                self.max_priority = max(self.max_priority, new_priority)
                break


class AdaptiveLearningRate:
    """Adaptive learning rate scheduler that responds to:
    - Recent accuracy trends
    - Regime changes
    - Error magnitude
    - Learning stability
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        min_lr: float = 0.0001,
        max_lr: float = 0.1,
        patience: int = 10,
        factor: float = 0.5,
    ):
        self.lr = initial_lr
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor

        self.accuracy_history: List[float] = []
        self.lr_history: List[Tuple[datetime, float]] = []
        self.wait = 0
        self.best_accuracy = 0.0

    def update(self, current_accuracy: float, error_magnitude: float = 0.0) -> float:
        """Update learning rate based on recent performance.

        Args:
        ----
            current_accuracy: Current accuracy metric
            error_magnitude: Average error magnitude (for regression tasks)

        Returns:
        -------
            Updated learning rate
        """
        self.accuracy_history.append(current_accuracy)

        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                # Reduce learning rate
                new_lr = self.lr * self.factor
                self.lr = max(self.min_lr, new_lr)
                self.wait = 0
                logger.info(f"Reduced learning rate to {self.lr:.6f}")

        # Increase LR if consistently improving
        if len(self.accuracy_history) >= 5:
            recent = self.accuracy_history[-5:]
            if all(recent[i] < recent[i+1] for i in range(4)):
                new_lr = self.lr * 1.1
                self.lr = min(self.max_lr, new_lr)
                logger.info(f"Increased learning rate to {self.lr:.6f}")

        self.lr_history.append((datetime.now(), self.lr))
        return self.lr

    def on_regime_change(self, new_regime: str):
        """Reset learning rate on regime change to allow faster adaptation."""
        self.lr = self.initial_lr
        self.wait = 0
        logger.info(f"Reset learning rate to {self.lr:.6f} due to regime change: {new_regime}")


class MetaLearningCoordinator:
    """Coordinates meta-learning across all agents.

    Tracks which learning strategies work best in different:
    - Market regimes
    - Time periods
    - Asset classes
    - Agent types
    """

    def __init__(self):
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.regime_strategies: Dict[str, List[str]] = {}
        self.agent_strategies: Dict[str, str] = {}

    def record_strategy_outcome(
        self,
        strategy: str,
        regime: str,
        success: bool,
        magnitude: float = 1.0,
    ):
        """Record outcome for a learning strategy."""
        key = f"{strategy}_{regime}"
        if key not in self.strategy_performance:
            self.strategy_performance[key] = {"success": 0, "failure": 0, "magnitude": 0}

        if success:
            self.strategy_performance[key]["success"] += 1
            self.strategy_performance[key]["magnitude"] += magnitude
        else:
            self.strategy_performance[key]["failure"] += 1

    def get_best_strategy(self, regime: str) -> str:
        """Get best learning strategy for current regime."""
        strategies = ["reinforcement", "bayesian", "adversarial", "ensemble", "meta"]
        best_strategy = "ensemble"  # Default
        best_score = 0.0

        for strategy in strategies:
            key = f"{strategy}_{regime}"
            if key in self.strategy_performance:
                perf = self.strategy_performance[key]
                total = perf["success"] + perf["failure"]
                if total > 10:  # Minimum samples
                    score = perf["success"] / total
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy

        return best_strategy


class CrossMachineLearningSync:
    """Synchronizes learning across multiple machines (Lenovo + MacBook).

    Features:
    - Federated learning-style aggregation
    - Conflict resolution
    - Experience sharing
    - Model averaging
    """

    def __init__(self):
        self.machine_id = platform.node().replace(" ", "_").replace(".", "_")
        self.sync_log: List[Dict[str, Any]] = []

    def prepare_sync_package(
        self,
        learning_outcomes: List[Any],
        mistake_patterns: Dict[str, int],
        beliefs: Dict[str, float],
        model_weights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare a sync package for upload.

        Returns a package that can be merged with other machines.
        """
        return {
            "machine_id": self.machine_id,
            "timestamp": datetime.now().isoformat(),
            "learning_outcomes_count": len(learning_outcomes),
            "learning_outcomes": list(learning_outcomes)[-1000],  # Last 1000
            "mistake_patterns": mistake_patterns,
            "beliefs": beliefs,
            "model_weights": model_weights,
        }

    def merge_sync_packages(
        self,
        packages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge sync packages from multiple machines.

        Implements federated averaging for compatible fields.
        """
        if not packages:
            return {}

        merged = {
            "sources": [p["machine_id"] for p in packages],
            "merge_timestamp": datetime.now().isoformat(),
            "learning_outcomes": [],
            "mistake_patterns": {},
            "beliefs": {},
        }

        # Merge learning outcomes (dedupe by experience_id if available)
        seen_ids = set()
        for pkg in packages:
            for outcome in pkg.get("learning_outcomes", []):
                if hasattr(outcome, "outcome_id"):
                    if outcome.outcome_id not in seen_ids:
                        seen_ids.add(outcome.outcome_id)
                        merged["learning_outcomes"].append(outcome)
                else:
                    merged["learning_outcomes"].append(outcome)

        # Sum mistake patterns
        for pkg in packages:
            for pattern, count in pkg.get("mistake_patterns", {}).items():
                merged["mistake_patterns"][pattern] = (
                    merged["mistake_patterns"].get(pattern, 0) + count
                )

        # Average beliefs (weighted by recency if timestamps available)
        belief_sums: Dict[str, float] = {}
        belief_counts: Dict[str, int] = {}
        for pkg in packages:
            for key, value in pkg.get("beliefs", {}).items():
                belief_sums[key] = belief_sums.get(key, 0) + value
                belief_counts[key] = belief_counts.get(key, 0) + 1
        merged["beliefs"] = {
            k: v / belief_counts[k] for k, v in belief_sums.items()
        }

        self.sync_log.append({
            "timestamp": datetime.now(),
            "machines_merged": len(packages),
            "outcomes_merged": len(merged["learning_outcomes"]),
        })

        return merged


class LearningOptimizer:
    """Master learning optimizer that coordinates all learning optimization.

    Use this class to maximize learning speed and efficiency across
    all agents and machines.
    """

    def __init__(self):
        self.experience_replay = PrioritizedExperienceReplay()
        self.adaptive_lr = AdaptiveLearningRate()
        self.meta_coordinator = MetaLearningCoordinator()
        self.cross_machine_sync = CrossMachineLearningSync()

        # Learning efficiency metrics
        self.total_experiences = 0
        self.useful_experiences = 0  # Led to correct predictions later
        self.learning_efficiency = 0.0

        # Session tracking
        self.current_session: Optional[LearningSession] = None
        self.session_history: List[LearningSession] = []

        logger.info(f"LearningOptimizer initialized on machine: {self.cross_machine_sync.machine_id}")

    def start_session(self, agent_name: str, initial_accuracy: float = 0.0) -> LearningSession:
        """Start a new learning session."""
        import uuid

        session = LearningSession(
            session_id=str(uuid.uuid4())[:8],
            start_time=datetime.now(),
            machine_id=self.cross_machine_sync.machine_id,
            agent_name=agent_name,
            accuracy_before=initial_accuracy,
            learning_rate=self.adaptive_lr.lr,
        )
        self.current_session = session
        return session

    def end_session(self, final_accuracy: float) -> Dict[str, Any]:
        """End learning session and return metrics."""
        if not self.current_session:
            return {}

        session = self.current_session
        session.accuracy_after = final_accuracy

        improvement = final_accuracy - session.accuracy_before
        duration = (datetime.now() - session.start_time).total_seconds()

        metrics = {
            "session_id": session.session_id,
            "duration_seconds": duration,
            "experiences_processed": session.experiences_processed,
            "accuracy_improvement": improvement,
            "learning_rate_final": self.adaptive_lr.lr,
            "adaptations_made": session.adaptations_made,
            "machine": session.machine_id,
        }

        self.session_history.append(session)
        self.current_session = None

        logger.info(f"Session complete: {improvement:.2%} improvement in {duration:.1f}s")
        return metrics

    def add_experience(
        self,
        agent_name: str,
        experience_type: str,
        data: Dict[str, Any],
        outcome: Optional[Any] = None,
        error_magnitude: float = 0.0,
    ) -> str:
        """Add a learning experience with automatic priority calculation.

        Higher priorities for:
        - Mistakes (we learn more from failures)
        - Surprising outcomes (high error)
        - Regime change events
        - Novel situations
        """
        import uuid

        exp_id = str(uuid.uuid4())[:8]

        # Calculate priority
        priority = 1.0

        # Mistakes get higher priority
        if experience_type == "mistake":
            priority *= 2.0

        # Higher error = higher priority
        priority *= (1.0 + error_magnitude)

        # Regime changes are important
        if experience_type == "regime_change":
            priority *= 1.5

        experience = LearningExperience(
            experience_id=exp_id,
            timestamp=datetime.now(),
            agent_name=agent_name,
            machine_id=self.cross_machine_sync.machine_id,
            experience_type=experience_type,
            data=data,
            outcome=outcome,
            priority=priority,
            td_error=error_magnitude,
        )

        self.experience_replay.add(experience, priority)
        self.total_experiences += 1

        if self.current_session:
            self.current_session.experiences_processed += 1

        return exp_id

    def get_learning_batch(self, batch_size: int = 32) -> Tuple[List[LearningExperience], List[float]]:
        """Get a prioritized batch of experiences for learning."""
        return self.experience_replay.sample(batch_size)

    def update_learning_rate(self, current_accuracy: float) -> float:
        """Update adaptive learning rate based on performance."""
        return self.adaptive_lr.update(current_accuracy)

    def on_regime_change(self, new_regime: str):
        """Handle regime change - reset learning rate for faster adaptation."""
        self.adaptive_lr.on_regime_change(new_regime)

        if self.current_session:
            self.current_session.adaptations_made += 1

    def get_optimal_strategy(self, regime: str) -> str:
        """Get the best learning strategy for current regime."""
        return self.meta_coordinator.get_best_strategy(regime)

    def record_strategy_outcome(self, strategy: str, regime: str, success: bool):
        """Record outcome of a learning strategy."""
        self.meta_coordinator.record_strategy_outcome(strategy, regime, success)

        if success:
            self.useful_experiences += 1

        self.learning_efficiency = (
            self.useful_experiences / self.total_experiences
            if self.total_experiences > 0 else 0.0
        )

    def prepare_for_sync(
        self,
        learning_outcomes: List[Any],
        mistake_patterns: Dict[str, int],
        beliefs: Dict[str, float],
    ) -> Dict[str, Any]:
        """Prepare learning state for cross-machine sync."""
        return self.cross_machine_sync.prepare_sync_package(
            learning_outcomes, mistake_patterns, beliefs,
        )

    def merge_from_other_machines(self, packages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge learning state from other machines."""
        return self.cross_machine_sync.merge_sync_packages(packages)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning optimizer stats."""
        return {
            "total_experiences": self.total_experiences,
            "useful_experiences": self.useful_experiences,
            "learning_efficiency": f"{self.learning_efficiency:.1%}",
            "current_learning_rate": self.adaptive_lr.lr,
            "experience_buffer_size": len(self.experience_replay.experiences),
            "beta_annealing": self.experience_replay.beta,
            "sessions_completed": len(self.session_history),
            "machine_id": self.cross_machine_sync.machine_id,
            "strategy_performance": self.meta_coordinator.strategy_performance,
        }


# Global instance for easy access
learning_optimizer = LearningOptimizer()

