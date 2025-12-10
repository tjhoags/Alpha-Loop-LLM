"""================================================================================
LEARNING PATH ENHANCEMENT ENGINE
================================================================================
Advanced self-improvement architecture for institutional-grade agents.

These features go beyond basic ML training to create TRUE learning systems
that improve over time through sophisticated feedback loops.

PART II: LEARNING PATH ENHANCEMENT FUNCTIONS
================================================================================
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""

    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    EUPHORIA = "euphoria"
    CAPITULATION = "capitulation"


class SignalType(Enum):
    """Types of trading signals."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    INSIDER_ACTIVITY = "insider_activity"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    FLOW = "flow"
    EVENT = "event"
    MACRO = "macro"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CROSS_ASSET = "cross_asset"


@dataclass
class PredictionRecord:
    """Record of a single prediction for calibration tracking."""

    prediction_id: str
    agent_name: str
    timestamp: datetime
    symbol: str
    direction: int  # 1 = long, -1 = short
    confidence: float  # 0-100%
    signal_type: SignalType
    regime: MarketRegime
    thesis: str  # Written thesis BEFORE the event
    target_price: Optional[float] = None
    target_date: Optional[datetime] = None

    # Filled in after resolution
    outcome: Optional[bool] = None  # True = correct, False = wrong
    actual_return: Optional[float] = None
    resolution_date: Optional[datetime] = None
    post_mortem: Optional[str] = None  # Analysis after the fact


@dataclass
class SignalPerformance:
    """Performance metrics for a signal type."""

    signal_type: SignalType
    total_signals: int = 0
    correct_signals: int = 0
    total_pnl: float = 0.0
    avg_return: float = 0.0
    win_rate: float = 0.0

    # Half-life metrics
    alpha_by_holding_period: Dict[int, float] = field(default_factory=dict)
    optimal_holding_days: int = 0
    half_life_days: float = 0.0

    # Regime-specific
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    best_regime: Optional[str] = None
    worst_regime: Optional[str] = None


@dataclass
class ThesisRecord:
    """Investment thesis tracking."""

    thesis_id: str
    agent_name: str
    created_at: datetime
    symbol: str
    thesis_type: str  # "secular", "cyclical", "event", "tactical"
    thesis_text: str
    expected_duration_days: int
    target_return: float

    # Validation
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    still_valid: bool = True
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    actual_return: Optional[float] = None


# =============================================================================
# PREDICTION CALIBRATION TRACKER
# =============================================================================

class PredictionCalibrationTracker:
    """Track prediction confidence vs actual outcomes.

    When an agent says "80% confident", are they right 80% of the time?
    Perfect calibration = confidence equals accuracy at each level.

    USAGE:
        tracker = PredictionCalibrationTracker()
        tracker.record_prediction(agent, symbol, confidence=0.8, direction=1)
        # ... later ...
        tracker.resolve_prediction(prediction_id, outcome=True, actual_return=0.05)
        calibration = tracker.get_calibration_score("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.predictions: Dict[str, PredictionRecord] = {}
        self.storage_path = storage_path or Path("data/calibration")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load historical predictions."""
        history_file = self.storage_path / "predictions.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for p in data:
                    pred = PredictionRecord(
                        prediction_id=p["prediction_id"],
                        agent_name=p["agent_name"],
                        timestamp=datetime.fromisoformat(p["timestamp"]),
                        symbol=p["symbol"],
                        direction=p["direction"],
                        confidence=p["confidence"],
                        signal_type=SignalType(p["signal_type"]),
                        regime=MarketRegime(p["regime"]),
                        thesis=p["thesis"],
                        outcome=p.get("outcome"),
                        actual_return=p.get("actual_return"),
                    )
                    self.predictions[pred.prediction_id] = pred
                logger.info(f"Loaded {len(self.predictions)} historical predictions")
            except Exception as e:
                logger.warning(f"Could not load prediction history: {e}")

    def _save_history(self):
        """Save predictions to disk."""
        history_file = self.storage_path / "predictions.json"
        data = []
        for p in self.predictions.values():
            data.append({
                "prediction_id": p.prediction_id,
                "agent_name": p.agent_name,
                "timestamp": p.timestamp.isoformat(),
                "symbol": p.symbol,
                "direction": p.direction,
                "confidence": p.confidence,
                "signal_type": p.signal_type.value,
                "regime": p.regime.value,
                "thesis": p.thesis,
                "outcome": p.outcome,
                "actual_return": p.actual_return,
            })
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_prediction(
        self,
        agent_name: str,
        symbol: str,
        direction: int,
        confidence: float,
        signal_type: SignalType,
        regime: MarketRegime,
        thesis: str,
        target_price: Optional[float] = None,
        target_date: Optional[datetime] = None,
    ) -> str:
        """Record a new prediction.

        Args:
        ----
            agent_name: Name of the predicting agent
            symbol: Ticker symbol
            direction: 1 for long, -1 for short
            confidence: Confidence level 0-100%
            signal_type: Type of signal
            regime: Current market regime
            thesis: Written thesis explaining the prediction

        Returns:
        -------
            prediction_id for later resolution
        """
        prediction_id = f"{agent_name}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        prediction = PredictionRecord(
            prediction_id=prediction_id,
            agent_name=agent_name,
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            signal_type=signal_type,
            regime=regime,
            thesis=thesis,
            target_price=target_price,
            target_date=target_date,
        )

        self.predictions[prediction_id] = prediction
        self._save_history()

        logger.info(f"Recorded prediction {prediction_id}: {agent_name} {direction} {symbol} @ {confidence:.0%} confidence")
        return prediction_id

    def resolve_prediction(
        self,
        prediction_id: str,
        outcome: bool,
        actual_return: float,
        post_mortem: Optional[str] = None,
    ):
        """Resolve a prediction with actual outcome.

        Args:
        ----
            prediction_id: ID from record_prediction
            outcome: True if prediction was correct
            actual_return: Actual return achieved
            post_mortem: Analysis of what happened
        """
        if prediction_id not in self.predictions:
            logger.warning(f"Unknown prediction: {prediction_id}")
            return

        pred = self.predictions[prediction_id]
        pred.outcome = outcome
        pred.actual_return = actual_return
        pred.resolution_date = datetime.now()
        pred.post_mortem = post_mortem

        self._save_history()
        logger.info(f"Resolved {prediction_id}: {'CORRECT' if outcome else 'WRONG'} ({actual_return:.2%})")

    def get_calibration_score(self, agent_name: str) -> Dict[str, Any]:
        """Calculate calibration score for an agent.

        Perfect calibration = when agent says 80%, they're right 80% of time.

        Returns dict with:
            - calibration_error: Mean absolute difference from perfect calibration
            - confidence_buckets: Accuracy at each confidence level
            - overconfidence_score: How much agent overestimates
            - underconfidence_score: How much agent underestimates
        """
        agent_preds = [
            p for p in self.predictions.values()
            if p.agent_name == agent_name and p.outcome is not None
        ]

        if len(agent_preds) < 20:
            return {"error": "Insufficient predictions (need 20+)"}

        # Bucket by confidence level (10% buckets)
        buckets = defaultdict(list)
        for p in agent_preds:
            bucket = int(p.confidence * 10) * 10  # 0-10, 10-20, etc.
            buckets[bucket].append(1 if p.outcome else 0)

        # Calculate accuracy per bucket
        confidence_buckets = {}
        calibration_errors = []
        overconfidence = []
        underconfidence = []

        for bucket, outcomes in buckets.items():
            expected_accuracy = bucket / 100  # e.g., 80% bucket expects 80% accuracy
            actual_accuracy = np.mean(outcomes)
            confidence_buckets[bucket] = {
                "expected": expected_accuracy,
                "actual": actual_accuracy,
                "n_predictions": len(outcomes),
            }

            error = actual_accuracy - expected_accuracy
            calibration_errors.append(abs(error))

            if error < 0:
                overconfidence.append(abs(error))
            else:
                underconfidence.append(error)

        return {
            "calibration_error": np.mean(calibration_errors) if calibration_errors else 0,
            "confidence_buckets": confidence_buckets,
            "overconfidence_score": np.mean(overconfidence) if overconfidence else 0,
            "underconfidence_score": np.mean(underconfidence) if underconfidence else 0,
            "total_predictions": len(agent_preds),
            "overall_accuracy": np.mean([1 if p.outcome else 0 for p in agent_preds]),
        }

    def get_calibration_penalty(self, agent_name: str) -> float:
        """Calculate penalty for poor calibration.

        Returns 0-1 where 0 = perfect calibration, 1 = terrible calibration.
        """
        calibration = self.get_calibration_score(agent_name)
        if "error" in calibration:
            return 0.5  # Default penalty for insufficient data

        # Combine calibration error + overconfidence (overconfidence is worse)
        base_error = calibration["calibration_error"]
        overconf_penalty = calibration["overconfidence_score"] * 1.5  # Penalize overconfidence more

        return min(1.0, base_error + overconf_penalty)


# =============================================================================
# SIGNAL HALF-LIFE ANALYZER
# =============================================================================

class SignalHalfLifeAnalyzer:
    """Analyze how long each signal type generates alpha before decay.

    Different signals have different optimal holding periods:
    - Insider buying: 30-day half-life (information diffuses slowly)
    - Momentum: 3-5 day half-life (mean reversion kicks in)
    - Earnings surprise: 1-day half-life (immediate price adjustment)

    Use this to adjust position sizing by signal freshness.

    USAGE:
        analyzer = SignalHalfLifeAnalyzer()
        analyzer.record_signal_return(signal_type, holding_days=5, return_pct=0.02)
        half_life = analyzer.get_half_life(SignalType.MOMENTUM)
        size_adj = analyzer.get_position_size_adjustment(SignalType.MOMENTUM, days_since_signal=3)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/signal_analysis")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # signal_type -> holding_days -> list of returns
        self.returns_by_period: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._load_history()

    def _load_history(self):
        """Load historical signal returns."""
        history_file = self.storage_path / "signal_returns.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for signal_type, periods in data.items():
                    for period, returns in periods.items():
                        self.returns_by_period[signal_type][int(period)] = returns
            except Exception as e:
                logger.warning(f"Could not load signal history: {e}")

    def _save_history(self):
        """Save to disk."""
        history_file = self.storage_path / "signal_returns.json"
        data = {
            signal_type: {str(period): returns for period, returns in periods.items()}
            for signal_type, periods in self.returns_by_period.items()
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_signal_return(
        self,
        signal_type: SignalType,
        holding_days: int,
        return_pct: float,
    ):
        """Record return for a signal at a specific holding period."""
        self.returns_by_period[signal_type.value][holding_days].append(return_pct)
        self._save_history()

    def get_alpha_curve(self, signal_type: SignalType) -> Dict[int, float]:
        """Get average alpha at each holding period.

        Returns dict of holding_days -> avg_return
        """
        signal_returns = self.returns_by_period.get(signal_type.value, {})

        return {
            period: np.mean(returns) if returns else 0
            for period, returns in sorted(signal_returns.items())
        }

    def get_half_life(self, signal_type: SignalType) -> float:
        """Calculate signal half-life (days until alpha drops to 50%).

        Returns number of days, or -1 if insufficient data.
        """
        alpha_curve = self.get_alpha_curve(signal_type)

        if not alpha_curve or len(alpha_curve) < 3:
            # Default half-lives by signal type
            defaults = {
                SignalType.MOMENTUM: 5,
                SignalType.MEAN_REVERSION: 3,
                SignalType.INSIDER_ACTIVITY: 30,
                SignalType.SENTIMENT: 7,
                SignalType.FUNDAMENTAL: 60,
                SignalType.TECHNICAL: 3,
                SignalType.FLOW: 5,
                SignalType.EVENT: 1,
                SignalType.MACRO: 30,
                SignalType.VOLATILITY: 5,
            }
            return defaults.get(signal_type, 10)

        # Find peak alpha
        periods = sorted(alpha_curve.keys())
        alphas = [alpha_curve[p] for p in periods]
        peak_alpha = max(alphas)
        peak_idx = alphas.index(peak_alpha)

        if peak_alpha <= 0:
            return -1

        # Find when alpha drops to 50% of peak
        half_alpha = peak_alpha / 2

        for i in range(peak_idx, len(alphas)):
            if alphas[i] <= half_alpha:
                return periods[i]

        return periods[-1]  # If never decays to 50%, return longest period

    def get_optimal_holding_period(self, signal_type: SignalType) -> int:
        """Get optimal holding period (max risk-adjusted return)."""
        alpha_curve = self.get_alpha_curve(signal_type)

        if not alpha_curve:
            return 5  # Default

        # Find period with best alpha (simple version)
        # Could enhance with Sharpe-ratio-like calculation
        best_period = max(alpha_curve.keys(), key=lambda p: alpha_curve[p])
        return best_period

    def get_position_size_adjustment(
        self,
        signal_type: SignalType,
        days_since_signal: int,
    ) -> float:
        """Get position size adjustment based on signal freshness.

        Returns multiplier 0-1 where:
        - 1.0 = signal is fresh, use full size
        - 0.5 = signal at half-life, use half size
        - 0.0 = signal fully decayed, don't trade
        """
        half_life = self.get_half_life(signal_type)

        if half_life <= 0:
            return 0.5

        # Exponential decay
        decay = np.exp(-0.693 * days_since_signal / half_life)  # 0.693 = ln(2)
        return max(0, min(1, decay))


# =============================================================================
# REGIME CONTEXT TAGGER
# =============================================================================

class RegimeContextTagger:
    """Tag every prediction with market regime and learn regime-specific accuracy.

    "This insider signal has 74% accuracy in bull regimes but only 41% in bear regimes"

    USAGE:
        tagger = RegimeContextTagger()
        regime = tagger.classify_current_regime(vix=15, spy_50d_return=0.05)
        tagger.record_signal_outcome(signal_type, regime, outcome=True)
        accuracy = tagger.get_regime_accuracy(SignalType.MOMENTUM, MarketRegime.BULL_LOW_VOL)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/regime_analysis")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # signal_type -> regime -> list of outcomes (True/False)
        self.outcomes: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
        self._load_history()

    def _load_history(self):
        """Load historical regime outcomes."""
        history_file = self.storage_path / "regime_outcomes.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for signal_type, regimes in data.items():
                    for regime, outcomes in regimes.items():
                        self.outcomes[signal_type][regime] = outcomes
            except Exception as e:
                logger.warning(f"Could not load regime history: {e}")

    def _save_history(self):
        """Save to disk."""
        history_file = self.storage_path / "regime_outcomes.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.outcomes), f, indent=2)

    def classify_current_regime(
        self,
        spy_return_20d: float,
        vix: float,
        spy_vs_200ma: float = 0,
        credit_spread: float = 0,
    ) -> MarketRegime:
        """Classify current market regime.

        Args:
        ----
            spy_return_20d: S&P 500 20-day return
            vix: VIX level
            spy_vs_200ma: SPY price vs 200-day MA (%)
            credit_spread: HY credit spread (optional)
        """
        high_vol = vix > 25
        very_high_vol = vix > 35

        # Crisis detection
        if very_high_vol and spy_return_20d < -0.10:
            return MarketRegime.CAPITULATION

        if very_high_vol and credit_spread > 5:
            return MarketRegime.CRISIS

        # Euphoria detection
        if vix < 12 and spy_return_20d > 0.10:
            return MarketRegime.EUPHORIA

        # Recovery
        if spy_vs_200ma < 0 and spy_return_20d > 0.05:
            return MarketRegime.RECOVERY

        # Standard regimes
        if spy_return_20d > 0.02:
            return MarketRegime.BULL_HIGH_VOL if high_vol else MarketRegime.BULL_LOW_VOL
        elif spy_return_20d < -0.02:
            return MarketRegime.BEAR_HIGH_VOL if high_vol else MarketRegime.BEAR_LOW_VOL
        else:
            return MarketRegime.SIDEWAYS_HIGH_VOL if high_vol else MarketRegime.SIDEWAYS_LOW_VOL

    def record_signal_outcome(
        self,
        signal_type: SignalType,
        regime: MarketRegime,
        outcome: bool,
    ):
        """Record signal outcome in a specific regime."""
        self.outcomes[signal_type.value][regime.value].append(outcome)
        self._save_history()

    def get_regime_accuracy(
        self,
        signal_type: SignalType,
        regime: MarketRegime,
    ) -> Optional[float]:
        """Get accuracy of signal type in specific regime."""
        outcomes = self.outcomes.get(signal_type.value, {}).get(regime.value, [])

        if len(outcomes) < 10:
            return None  # Insufficient data

        return sum(outcomes) / len(outcomes)

    def get_full_regime_analysis(self, signal_type: SignalType) -> Dict[str, Any]:
        """Get complete regime analysis for a signal type."""
        signal_outcomes = self.outcomes.get(signal_type.value, {})

        analysis = {}
        for regime in MarketRegime:
            outcomes = signal_outcomes.get(regime.value, [])
            if outcomes:
                analysis[regime.value] = {
                    "accuracy": sum(outcomes) / len(outcomes),
                    "n_signals": len(outcomes),
                    "profitable": sum(outcomes),
                }

        if analysis:
            # Find best and worst regimes
            by_accuracy = sorted(
                [(k, v["accuracy"]) for k, v in analysis.items() if v["n_signals"] >= 5],
                key=lambda x: x[1],
                reverse=True,
            )

            if by_accuracy:
                analysis["best_regime"] = by_accuracy[0][0]
                analysis["worst_regime"] = by_accuracy[-1][0]
                analysis["regime_spread"] = by_accuracy[0][1] - by_accuracy[-1][1]

        return analysis

    def should_trade_signal(
        self,
        signal_type: SignalType,
        current_regime: MarketRegime,
        min_accuracy: float = 0.5,
    ) -> Tuple[bool, str]:
        """Determine if a signal should be traded in current regime.

        Returns (should_trade, reason)
        """
        accuracy = self.get_regime_accuracy(signal_type, current_regime)

        if accuracy is None:
            return True, "Insufficient regime data, proceeding with caution"

        if accuracy < min_accuracy:
            return False, f"{signal_type.value} has only {accuracy:.0%} accuracy in {current_regime.value}"

        return True, f"{signal_type.value} has {accuracy:.0%} accuracy in {current_regime.value}"


# =============================================================================
# CROSS-AGENT SIGNAL CORRELATION MONITOR
# =============================================================================

class CrossAgentCorrelationMonitor:
    """Track when multiple agents agree and learn which combinations matter.

    When Agent A and Agent B both say "buy AAPL" = higher conviction
    But WHICH combinations of agreement actually predict success?
    Learn optimal ensemble weighting dynamically.

    USAGE:
        monitor = CrossAgentCorrelationMonitor()
        monitor.record_multi_agent_signal(
            symbol="AAPL",
            direction=1,
            agents_agreeing=["MOMENTUM", "SENTIMENT"],
            outcome=True
        )
        weight = monitor.get_combination_weight(["MOMENTUM", "SENTIMENT"])
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/correlation_analysis")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # frozenset of agent names -> list of outcomes
        self.combination_outcomes: Dict[str, List[bool]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "combinations.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.combination_outcomes = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load correlation history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "combinations.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.combination_outcomes), f, indent=2)

    def _combo_key(self, agents: List[str]) -> str:
        """Create consistent key for agent combination."""
        return "|".join(sorted(agents))

    def record_multi_agent_signal(
        self,
        symbol: str,
        direction: int,
        agents_agreeing: List[str],
        outcome: bool,
        return_pct: Optional[float] = None,
    ):
        """Record outcome when multiple agents agree."""
        if len(agents_agreeing) < 2:
            return  # Need at least 2 agents agreeing

        key = self._combo_key(agents_agreeing)
        self.combination_outcomes[key].append(outcome)
        self._save_history()

        logger.debug(f"Recorded {key} agreement on {symbol}: {'WIN' if outcome else 'LOSS'}")

    def get_combination_accuracy(self, agents: List[str]) -> Optional[float]:
        """Get historical accuracy when these agents agree."""
        key = self._combo_key(agents)
        outcomes = self.combination_outcomes.get(key, [])

        if len(outcomes) < 5:
            return None

        return sum(outcomes) / len(outcomes)

    def get_combination_weight(self, agents: List[str]) -> float:
        """Get conviction weight multiplier for agent combination.

        Returns 0.5-2.0 where:
        - 0.5 = this combo historically underperforms, reduce size
        - 1.0 = neutral
        - 2.0 = this combo is highly predictive, increase size
        """
        accuracy = self.get_combination_accuracy(agents)

        if accuracy is None:
            return 1.0  # Neutral if insufficient data

        # Map accuracy to weight
        # 50% accuracy = 1.0 weight
        # 75% accuracy = 1.5 weight
        # 90% accuracy = 2.0 weight
        # 30% accuracy = 0.5 weight

        weight = 0.5 + (accuracy - 0.3) * 2.5
        return max(0.5, min(2.0, weight))

    def get_best_combinations(self, min_signals: int = 10) -> List[Tuple[str, float, int]]:
        """Get best-performing agent combinations."""
        results = []

        for combo, outcomes in self.combination_outcomes.items():
            if len(outcomes) >= min_signals:
                accuracy = sum(outcomes) / len(outcomes)
                results.append((combo, accuracy, len(outcomes)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_agreement_level(self, signals: Dict[str, int]) -> Tuple[float, List[str]]:
        """Calculate agreement level from agent signals.

        Args:
        ----
            signals: Dict of agent_name -> direction (1, -1, or 0)

        Returns:
        -------
            (agreement_score, list of agreeing agents)
        """
        longs = [a for a, d in signals.items() if d > 0]
        shorts = [a for a, d in signals.items() if d < 0]

        total_agents = len([a for a, d in signals.items() if d != 0])

        if total_agents == 0:
            return 0, []

        if len(longs) > len(shorts):
            agreement = len(longs) / total_agents
            agreeing = longs
        else:
            agreement = len(shorts) / total_agents
            agreeing = shorts

        return agreement, agreeing


# =============================================================================
# CONTRARIAN TRIGGER DETECTOR
# =============================================================================

class ContrarianTriggerDetector:
    """Learn to recognize when consensus forms = potential contrarian opportunity.

    Very high agreement often = crowded trade = potential reversal.
    Track agent agreement levels and identify contrarian opportunities.

    USAGE:
        detector = ContrarianTriggerDetector()
        trigger = detector.check_contrarian_trigger(
            agreement_level=0.9,  # 90% of agents agree
            current_regime=MarketRegime.EUPHORIA
        )
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/contrarian_analysis")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Track outcomes when consensus was high/low
        self.consensus_outcomes: Dict[str, List[Dict]] = {
            "high_consensus": [],  # >80% agreement
            "medium_consensus": [],  # 50-80% agreement
            "low_consensus": [],  # <50% agreement
        }
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "contrarian.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.consensus_outcomes = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load contrarian history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "contrarian.json"
        with open(history_file, "w") as f:
            json.dump(self.consensus_outcomes, f, indent=2)

    def record_consensus_outcome(
        self,
        agreement_level: float,
        consensus_direction: int,  # What consensus predicted
        actual_return: float,  # What actually happened
        regime: MarketRegime,
    ):
        """Record outcome when there was high/medium/low consensus."""
        if agreement_level >= 0.8:
            bucket = "high_consensus"
        elif agreement_level >= 0.5:
            bucket = "medium_consensus"
        else:
            bucket = "low_consensus"

        # Did consensus get it right?
        consensus_correct = (
            (consensus_direction > 0 and actual_return > 0) or
            (consensus_direction < 0 and actual_return < 0)
        )

        self.consensus_outcomes[bucket].append({
            "agreement": agreement_level,
            "direction": consensus_direction,
            "actual_return": actual_return,
            "correct": consensus_correct,
            "regime": regime.value,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_history()

    def check_contrarian_trigger(
        self,
        agreement_level: float,
        current_regime: MarketRegime,
        consensus_direction: int,
    ) -> Dict[str, Any]:
        """Check if contrarian trade is warranted.

        Returns analysis including:
        - is_contrarian_opportunity: bool
        - contrarian_score: 0-1 confidence in contrarian trade
        - historical_consensus_accuracy: how often consensus is right at this level
        - recommendation: "follow_consensus" | "go_contrarian" | "stay_neutral"
        """
        high_consensus = self.consensus_outcomes.get("high_consensus", [])

        # Calculate historical accuracy of high consensus
        if high_consensus:
            consensus_accuracy = sum(1 for o in high_consensus if o["correct"]) / len(high_consensus)
        else:
            consensus_accuracy = 0.5  # Assume 50% if no data

        # Check regime-specific consensus accuracy
        regime_outcomes = [o for o in high_consensus if o.get("regime") == current_regime.value]
        if len(regime_outcomes) >= 5:
            regime_consensus_accuracy = sum(1 for o in regime_outcomes if o["correct"]) / len(regime_outcomes)
        else:
            regime_consensus_accuracy = consensus_accuracy

        # Contrarian score
        # High agreement + euphoric regime + historically poor consensus = strong contrarian signal
        contrarian_score = 0.0

        if agreement_level >= 0.9:
            contrarian_score += 0.3
        elif agreement_level >= 0.8:
            contrarian_score += 0.2

        if current_regime in [MarketRegime.EUPHORIA, MarketRegime.CAPITULATION]:
            contrarian_score += 0.3

        if regime_consensus_accuracy < 0.5:
            contrarian_score += 0.2

        if consensus_accuracy < 0.5:
            contrarian_score += 0.2

        # Recommendation
        if contrarian_score >= 0.6:
            recommendation = "go_contrarian"
        elif contrarian_score <= 0.2:
            recommendation = "follow_consensus"
        else:
            recommendation = "stay_neutral"

        return {
            "is_contrarian_opportunity": contrarian_score >= 0.5,
            "contrarian_score": contrarian_score,
            "agreement_level": agreement_level,
            "historical_consensus_accuracy": consensus_accuracy,
            "regime_consensus_accuracy": regime_consensus_accuracy,
            "current_regime": current_regime.value,
            "recommendation": recommendation,
        }


# =============================================================================
# ATTRIBUTION FEEDBACK LOOP
# =============================================================================

class AttributionFeedbackLoop:
    """P&L attribution to individual signals.

    Signals that generated P&L get reinforced.
    Signals that lost money get weighted down.

    USAGE:
        loop = AttributionFeedbackLoop()
        loop.record_signal_pnl(signal_type, signal_id, pnl=1000)
        weight = loop.get_signal_weight(SignalType.MOMENTUM)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/attribution")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # signal_type -> list of pnl values
        self.signal_pnl: Dict[str, List[float]] = defaultdict(list)

        # Individual signal tracking
        self.signal_history: List[Dict] = []

        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "attribution.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                self.signal_pnl = defaultdict(list, data.get("signal_pnl", {}))
                self.signal_history = data.get("signal_history", [])
            except Exception as e:
                logger.warning(f"Could not load attribution history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "attribution.json"
        with open(history_file, "w") as f:
            json.dump({
                "signal_pnl": dict(self.signal_pnl),
                "signal_history": self.signal_history[-10000:],  # Keep last 10k
            }, f, indent=2)

    def record_signal_pnl(
        self,
        signal_type: SignalType,
        signal_id: str,
        pnl: float,
        symbol: str = "",
        agent_name: str = "",
    ):
        """Record P&L attributed to a signal."""
        self.signal_pnl[signal_type.value].append(pnl)

        self.signal_history.append({
            "signal_type": signal_type.value,
            "signal_id": signal_id,
            "pnl": pnl,
            "symbol": symbol,
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_history()

        logger.debug(f"Attributed ${pnl:,.0f} to {signal_type.value}")

    def get_signal_weight(self, signal_type: SignalType) -> float:
        """Get weight for signal type based on historical P&L.

        Returns 0.5-2.0 where:
        - 0.5 = signal historically loses money, reduce exposure
        - 1.0 = neutral
        - 2.0 = signal historically makes money, increase exposure
        """
        pnl_history = self.signal_pnl.get(signal_type.value, [])

        if len(pnl_history) < 20:
            return 1.0  # Neutral if insufficient data

        # Use recent history (last 100 signals)
        recent = pnl_history[-100:]

        total_pnl = sum(recent)
        win_rate = sum(1 for p in recent if p > 0) / len(recent)

        # Calculate Sharpe-like metric
        if len(recent) > 1 and np.std(recent) > 0:
            sharpe = np.mean(recent) / np.std(recent)
        else:
            sharpe = 0

        # Map to weight
        # Positive Sharpe = increase weight
        # Negative Sharpe = decrease weight
        weight = 1.0 + np.tanh(sharpe) * 0.5

        return max(0.5, min(2.0, weight))

    def get_attribution_report(self) -> Dict[str, Any]:
        """Get full attribution report."""
        report = {}

        for signal_type, pnl_list in self.signal_pnl.items():
            if pnl_list:
                report[signal_type] = {
                    "total_pnl": sum(pnl_list),
                    "avg_pnl": np.mean(pnl_list),
                    "win_rate": sum(1 for p in pnl_list if p > 0) / len(pnl_list),
                    "n_signals": len(pnl_list),
                    "current_weight": self.get_signal_weight(SignalType(signal_type)),
                }

        # Sort by total P&L
        report = dict(sorted(
            report.items(),
            key=lambda x: x[1]["total_pnl"],
            reverse=True,
        ))

        return report


# =============================================================================
# THESIS HALF-LIFE TRACKER
# =============================================================================

class ThesisHalfLifeTracker:
    """Track how long investment theses remain valid.

    Some theses are multi-year (secular growth trends)
    Some are multi-week (event-driven)
    Learn to distinguish and size accordingly.

    USAGE:
        tracker = ThesisHalfLifeTracker()
        thesis_id = tracker.create_thesis(
            agent="VALUE",
            symbol="AAPL",
            thesis_type="secular",
            thesis_text="iPhone installed base drives services growth",
            expected_duration_days=365
        )
        tracker.add_checkpoint(thesis_id, still_valid=True, notes="Q3 services beat")
        tracker.invalidate_thesis(thesis_id, reason="Services growth slowing")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/thesis_tracking")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.theses: Dict[str, ThesisRecord] = {}
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "theses.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for t in data:
                    thesis = ThesisRecord(
                        thesis_id=t["thesis_id"],
                        agent_name=t["agent_name"],
                        created_at=datetime.fromisoformat(t["created_at"]),
                        symbol=t["symbol"],
                        thesis_type=t["thesis_type"],
                        thesis_text=t["thesis_text"],
                        expected_duration_days=t["expected_duration_days"],
                        target_return=t["target_return"],
                        checkpoints=t.get("checkpoints", []),
                        still_valid=t.get("still_valid", True),
                        invalidated_at=datetime.fromisoformat(t["invalidated_at"]) if t.get("invalidated_at") else None,
                        invalidation_reason=t.get("invalidation_reason"),
                        actual_return=t.get("actual_return"),
                    )
                    self.theses[thesis.thesis_id] = thesis
            except Exception as e:
                logger.warning(f"Could not load thesis history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "theses.json"
        data = []
        for t in self.theses.values():
            data.append({
                "thesis_id": t.thesis_id,
                "agent_name": t.agent_name,
                "created_at": t.created_at.isoformat(),
                "symbol": t.symbol,
                "thesis_type": t.thesis_type,
                "thesis_text": t.thesis_text,
                "expected_duration_days": t.expected_duration_days,
                "target_return": t.target_return,
                "checkpoints": t.checkpoints,
                "still_valid": t.still_valid,
                "invalidated_at": t.invalidated_at.isoformat() if t.invalidated_at else None,
                "invalidation_reason": t.invalidation_reason,
                "actual_return": t.actual_return,
            })
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_thesis(
        self,
        agent_name: str,
        symbol: str,
        thesis_type: str,  # "secular", "cyclical", "event", "tactical"
        thesis_text: str,
        expected_duration_days: int,
        target_return: float,
    ) -> str:
        """Create a new investment thesis."""
        thesis_id = f"{agent_name}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        thesis = ThesisRecord(
            thesis_id=thesis_id,
            agent_name=agent_name,
            created_at=datetime.now(),
            symbol=symbol,
            thesis_type=thesis_type,
            thesis_text=thesis_text,
            expected_duration_days=expected_duration_days,
            target_return=target_return,
        )

        self.theses[thesis_id] = thesis
        self._save_history()

        logger.info(f"Created thesis {thesis_id}: {thesis_type} on {symbol}")
        return thesis_id

    def add_checkpoint(
        self,
        thesis_id: str,
        still_valid: bool,
        notes: str,
        current_return: Optional[float] = None,
    ):
        """Add a checkpoint to track thesis validity over time."""
        if thesis_id not in self.theses:
            logger.warning(f"Unknown thesis: {thesis_id}")
            return

        thesis = self.theses[thesis_id]
        thesis.checkpoints.append({
            "timestamp": datetime.now().isoformat(),
            "still_valid": still_valid,
            "notes": notes,
            "current_return": current_return,
        })

        if not still_valid:
            thesis.still_valid = False
            thesis.invalidated_at = datetime.now()
            thesis.invalidation_reason = notes

        self._save_history()

    def invalidate_thesis(self, thesis_id: str, reason: str, actual_return: float = 0):
        """Mark a thesis as invalidated."""
        if thesis_id not in self.theses:
            return

        thesis = self.theses[thesis_id]
        thesis.still_valid = False
        thesis.invalidated_at = datetime.now()
        thesis.invalidation_reason = reason
        thesis.actual_return = actual_return

        self._save_history()

        # Calculate actual duration vs expected
        actual_days = (thesis.invalidated_at - thesis.created_at).days
        logger.info(
            f"Thesis {thesis_id} invalidated after {actual_days} days "
            f"(expected {thesis.expected_duration_days}): {reason}",
        )

    def get_thesis_accuracy_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Analyze thesis accuracy by type."""
        by_type: Dict[str, List[ThesisRecord]] = defaultdict(list)

        for thesis in self.theses.values():
            if thesis.actual_return is not None:
                by_type[thesis.thesis_type].append(thesis)

        results = {}
        for thesis_type, theses in by_type.items():
            profitable = [t for t in theses if t.actual_return > 0]
            avg_return = np.mean([t.actual_return for t in theses])

            # Calculate average duration accuracy
            duration_ratios = [
                (t.invalidated_at - t.created_at).days / t.expected_duration_days
                for t in theses if t.invalidated_at
            ]

            results[thesis_type] = {
                "win_rate": len(profitable) / len(theses) if theses else 0,
                "avg_return": avg_return,
                "n_theses": len(theses),
                "avg_duration_ratio": np.mean(duration_ratios) if duration_ratios else 1.0,
            }

        return results


# =============================================================================
# PREDICTION JOURNAL QA
# =============================================================================

class PredictionJournalQA:
    """Force agents to write thesis BEFORE event.
    After event, compare prediction to reality.
    Store as training data for continuous improvement.

    This is the "intellectual honesty" module - no hindsight bias.

    USAGE:
        journal = PredictionJournalQA()
        entry_id = journal.create_pre_event_entry(
            agent="HUNTER",
            event_type="earnings",
            symbol="AAPL",
            prediction="Beat on EPS, miss on guidance",
            expected_impact="Stock drops 5% on weak guidance"
        )
        journal.record_post_event(
            entry_id,
            actual_outcome="Beat on EPS, raised guidance",
            actual_impact="Stock up 7%",
            lessons_learned="Underestimated services momentum"
        )
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/journal")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.entries: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "journal.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.entries = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load journal: {e}")

    def _save_history(self):
        history_file = self.storage_path / "journal.json"
        with open(history_file, "w") as f:
            json.dump(self.entries, f, indent=2)

    def create_pre_event_entry(
        self,
        agent_name: str,
        event_type: str,  # "earnings", "fed_meeting", "product_launch", etc.
        symbol: str,
        event_date: datetime,
        prediction: str,
        expected_impact: str,
        confidence: float,
        key_factors: List[str],
    ) -> str:
        """Create a pre-event prediction entry.

        This MUST be created BEFORE the event happens.
        """
        entry_id = f"{agent_name}_{symbol}_{event_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        entry = {
            "entry_id": entry_id,
            "agent_name": agent_name,
            "event_type": event_type,
            "symbol": symbol,
            "event_date": event_date.isoformat(),
            "created_at": datetime.now().isoformat(),

            # Pre-event prediction (cannot be modified after event)
            "prediction": prediction,
            "expected_impact": expected_impact,
            "confidence": confidence,
            "key_factors": key_factors,

            # Post-event analysis (filled in later)
            "actual_outcome": None,
            "actual_impact": None,
            "prediction_correct": None,
            "impact_correct": None,
            "lessons_learned": None,
            "resolved_at": None,
        }

        self.entries.append(entry)
        self._save_history()

        logger.info(f"Journal entry created: {entry_id}")
        return entry_id

    def record_post_event(
        self,
        entry_id: str,
        actual_outcome: str,
        actual_impact: str,
        prediction_correct: bool,
        impact_correct: bool,
        lessons_learned: str,
    ):
        """Record what actually happened after the event.

        This is the critical learning step - comparing prediction to reality.
        """
        for entry in self.entries:
            if entry["entry_id"] == entry_id:
                entry["actual_outcome"] = actual_outcome
                entry["actual_impact"] = actual_impact
                entry["prediction_correct"] = prediction_correct
                entry["impact_correct"] = impact_correct
                entry["lessons_learned"] = lessons_learned
                entry["resolved_at"] = datetime.now().isoformat()

                self._save_history()

                logger.info(
                    f"Journal entry resolved: {entry_id} - "
                    f"Prediction: {'CORRECT' if prediction_correct else 'WRONG'}, "
                    f"Impact: {'CORRECT' if impact_correct else 'WRONG'}",
                )
                return

        logger.warning(f"Journal entry not found: {entry_id}")

    def get_agent_accuracy(self, agent_name: str) -> Dict[str, Any]:
        """Get prediction accuracy for an agent."""
        agent_entries = [
            e for e in self.entries
            if e["agent_name"] == agent_name and e["resolved_at"] is not None
        ]

        if not agent_entries:
            return {"error": "No resolved entries"}

        prediction_correct = sum(1 for e in agent_entries if e["prediction_correct"])
        impact_correct = sum(1 for e in agent_entries if e["impact_correct"])

        # Accuracy by event type
        by_event_type: Dict[str, List] = defaultdict(list)
        for e in agent_entries:
            by_event_type[e["event_type"]].append(e["prediction_correct"])

        event_accuracy = {
            event_type: sum(outcomes) / len(outcomes)
            for event_type, outcomes in by_event_type.items()
        }

        return {
            "total_predictions": len(agent_entries),
            "prediction_accuracy": prediction_correct / len(agent_entries),
            "impact_accuracy": impact_correct / len(agent_entries),
            "accuracy_by_event_type": event_accuracy,
            "common_lessons": self._extract_common_lessons(agent_entries),
        }

    def _extract_common_lessons(self, entries: List[Dict]) -> List[str]:
        """Extract common themes from lessons learned."""
        lessons = [e.get("lessons_learned", "") for e in entries if e.get("lessons_learned")]

        # Simple word frequency (could enhance with NLP)
        word_freq: Dict[str, int] = defaultdict(int)
        for lesson in lessons:
            for word in lesson.lower().split():
                if len(word) > 4:
                    word_freq[word] += 1

        # Return top themes
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [w[0] for w in top_words]

    def get_training_data(self) -> List[Dict]:
        """Export journal as training data.

        Returns list of (prediction, outcome, correct) tuples for ML training.
        """
        return [
            {
                "input": {
                    "event_type": e["event_type"],
                    "symbol": e["symbol"],
                    "prediction": e["prediction"],
                    "expected_impact": e["expected_impact"],
                    "confidence": e["confidence"],
                    "key_factors": e["key_factors"],
                },
                "output": {
                    "actual_outcome": e["actual_outcome"],
                    "actual_impact": e["actual_impact"],
                },
                "label": e["prediction_correct"],
            }
            for e in self.entries
            if e["resolved_at"] is not None
        ]


# =============================================================================
# MASTER LEARNING ENGINE
# =============================================================================

class LearningEngine:
    """Master learning engine that coordinates all learning components.

    USAGE:
        engine = LearningEngine()

        # Record a prediction
        pred_id = engine.record_prediction(
            agent="MOMENTUM",
            symbol="AAPL",
            direction=1,
            confidence=0.75,
            signal_type=SignalType.MOMENTUM,
            thesis="Strong momentum breakout pattern"
        )

        # After trade closes
        engine.resolve_trade(
            prediction_id=pred_id,
            outcome=True,
            actual_return=0.03,
            holding_days=5
        )

        # Get agent's learning score
        score = engine.get_learning_score("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize all components
        self.calibration = PredictionCalibrationTracker(self.storage_path / "calibration")
        self.signal_analyzer = SignalHalfLifeAnalyzer(self.storage_path / "signals")
        self.regime_tagger = RegimeContextTagger(self.storage_path / "regimes")
        self.correlation_monitor = CrossAgentCorrelationMonitor(self.storage_path / "correlations")
        self.contrarian_detector = ContrarianTriggerDetector(self.storage_path / "contrarian")
        self.attribution = AttributionFeedbackLoop(self.storage_path / "attribution")
        self.thesis_tracker = ThesisHalfLifeTracker(self.storage_path / "theses")
        self.journal = PredictionJournalQA(self.storage_path / "journal")

        logger.info("Learning Engine initialized with all components")

    def record_prediction(
        self,
        agent_name: str,
        symbol: str,
        direction: int,
        confidence: float,
        signal_type: SignalType,
        thesis: str,
        regime: Optional[MarketRegime] = None,
    ) -> str:
        """Record a new prediction across all tracking systems."""
        if regime is None:
            regime = MarketRegime.SIDEWAYS_LOW_VOL  # Default

        return self.calibration.record_prediction(
            agent_name=agent_name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            signal_type=signal_type,
            regime=regime,
            thesis=thesis,
        )

    def resolve_trade(
        self,
        prediction_id: str,
        outcome: bool,
        actual_return: float,
        holding_days: int,
        pnl_dollars: float = 0,
    ):
        """Resolve a trade and update all learning systems."""
        # Get original prediction
        pred = self.calibration.predictions.get(prediction_id)
        if not pred:
            logger.warning(f"Prediction not found: {prediction_id}")
            return

        # Update calibration
        self.calibration.resolve_prediction(prediction_id, outcome, actual_return)

        # Update signal half-life
        self.signal_analyzer.record_signal_return(
            pred.signal_type, holding_days, actual_return,
        )

        # Update regime analysis
        self.regime_tagger.record_signal_outcome(
            pred.signal_type, pred.regime, outcome,
        )

        # Update P&L attribution
        if pnl_dollars != 0:
            self.attribution.record_signal_pnl(
                pred.signal_type,
                prediction_id,
                pnl_dollars,
                pred.symbol,
                pred.agent_name,
            )

    def get_learning_score(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive learning score for an agent.

        Combines all learning metrics into a single score.
        """
        calibration = self.calibration.get_calibration_score(agent_name)
        journal_accuracy = self.journal.get_agent_accuracy(agent_name)

        # Calculate composite score
        scores = []

        # Calibration score (lower error = better)
        if "calibration_error" in calibration:
            cal_score = max(0, 1 - calibration["calibration_error"] * 2)
            scores.append(("calibration", cal_score))

        # Prediction accuracy
        if "prediction_accuracy" in journal_accuracy:
            scores.append(("prediction", journal_accuracy["prediction_accuracy"]))

        # Calculate overall
        if scores:
            overall = np.mean([s[1] for s in scores])
        else:
            overall = 0.5

        return {
            "overall_learning_score": overall,
            "calibration": calibration,
            "journal_accuracy": journal_accuracy,
            "component_scores": dict(scores),
        }

    def get_signal_recommendation(
        self,
        signal_type: SignalType,
        current_regime: MarketRegime,
        days_since_signal: int,
        agreeing_agents: List[str],
    ) -> Dict[str, Any]:
        """Get comprehensive recommendation for a signal.

        Combines all learning insights to recommend position size.
        """
        # Signal freshness
        freshness_adj = self.signal_analyzer.get_position_size_adjustment(
            signal_type, days_since_signal,
        )

        # Regime appropriateness
        should_trade, regime_reason = self.regime_tagger.should_trade_signal(
            signal_type, current_regime,
        )

        # P&L attribution weight
        pnl_weight = self.attribution.get_signal_weight(signal_type)

        # Multi-agent agreement
        if len(agreeing_agents) >= 2:
            combo_weight = self.correlation_monitor.get_combination_weight(agreeing_agents)
        else:
            combo_weight = 1.0

        # Check for contrarian trigger
        agreement_level = len(agreeing_agents) / 10  # Assume 10 agents
        contrarian = self.contrarian_detector.check_contrarian_trigger(
            agreement_level, current_regime, 1,
        )

        # Calculate final size multiplier
        if not should_trade:
            size_multiplier = 0.25  # Reduce but don't eliminate
        else:
            size_multiplier = freshness_adj * pnl_weight * combo_weight

        # Cap at 2x
        size_multiplier = min(2.0, size_multiplier)

        return {
            "should_trade": should_trade,
            "size_multiplier": size_multiplier,
            "freshness_adjustment": freshness_adj,
            "regime_reason": regime_reason,
            "pnl_weight": pnl_weight,
            "combo_weight": combo_weight,
            "contrarian_analysis": contrarian,
            "recommendation": "TRADE" if should_trade and size_multiplier > 0.5 else "SKIP",
        }


# =============================================================================
# PART III: ADVANCED SELF-IMPROVEMENT ARCHITECTURE
# =============================================================================
# These go BEYOND basic learning - they create TRUE intelligence

# =============================================================================
# COGNITIVE BIAS DETECTOR
# =============================================================================

class CognitiveBias(Enum):
    """Types of cognitive biases that affect trading."""

    RECENCY_BIAS = "recency_bias"           # Over-weighting recent data
    ANCHORING = "anchoring"                  # Stuck on old price targets
    CONFIRMATION_BIAS = "confirmation_bias"  # Ignoring contradicting signals
    LOSS_AVERSION = "loss_aversion"          # Holding losers too long
    OVERCONFIDENCE = "overconfidence"        # Excessive confidence after wins
    GAMBLER_FALLACY = "gambler_fallacy"      # "Due" for a win after losses
    HERD_MENTALITY = "herd_mentality"        # Following crowd blindly
    SUNK_COST = "sunk_cost"                  # Can't let go of bad positions
    HINDSIGHT_BIAS = "hindsight_bias"        # "I knew it all along"
    AVAILABILITY_BIAS = "availability_bias"  # Over-weighting memorable events


@dataclass
class BiasIncident:
    """Record of a detected bias incident."""

    bias_type: CognitiveBias
    agent_name: str
    timestamp: datetime
    symbol: str
    description: str
    severity: float  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)
    corrected: bool = False


class CognitiveBiasDetector:
    """Detect and correct cognitive biases in trading behavior.

    HUMANS have these biases. AGENTS can too if not monitored.
    This catches biases BEFORE they cost money.

    USAGE:
        detector = CognitiveBiasDetector()
        biases = detector.analyze_agent_behavior(
            agent_name="MOMENTUM",
            recent_signals=signals,
            recent_trades=trades
        )
        detector.apply_bias_correction(agent_weights)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/bias_detection")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.incidents: List[BiasIncident] = []
        self.agent_bias_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "bias_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for inc in data.get("incidents", []):
                    self.incidents.append(BiasIncident(
                        bias_type=CognitiveBias(inc["bias_type"]),
                        agent_name=inc["agent_name"],
                        timestamp=datetime.fromisoformat(inc["timestamp"]),
                        symbol=inc["symbol"],
                        description=inc["description"],
                        severity=inc["severity"],
                        evidence=inc.get("evidence", {}),
                        corrected=inc.get("corrected", False),
                    ))
                self.agent_bias_scores = defaultdict(lambda: defaultdict(float), data.get("scores", {}))
            except Exception as e:
                logger.warning(f"Could not load bias history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "bias_history.json"
        data = {
            "incidents": [
                {
                    "bias_type": inc.bias_type.value,
                    "agent_name": inc.agent_name,
                    "timestamp": inc.timestamp.isoformat(),
                    "symbol": inc.symbol,
                    "description": inc.description,
                    "severity": inc.severity,
                    "evidence": inc.evidence,
                    "corrected": inc.corrected,
                }
                for inc in self.incidents[-1000:]  # Keep last 1000
            ],
            "scores": dict(self.agent_bias_scores),
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def detect_recency_bias(
        self,
        agent_name: str,
        signal_weights: List[float],
        time_indices: List[int],
    ) -> Optional[BiasIncident]:
        """Detect if agent over-weights recent data.

        If last 10% of data has 50%+ influence = recency bias.
        """
        if len(signal_weights) < 50:
            return None

        n = len(signal_weights)
        recent_cutoff = int(n * 0.9)

        total_weight = sum(abs(w) for w in signal_weights)
        recent_weight = sum(abs(w) for i, w in enumerate(signal_weights) if i >= recent_cutoff)

        recent_influence = recent_weight / (total_weight + 1e-10)

        if recent_influence > 0.5:  # Recent 10% has >50% influence
            incident = BiasIncident(
                bias_type=CognitiveBias.RECENCY_BIAS,
                agent_name=agent_name,
                timestamp=datetime.now(),
                symbol="*",
                description=f"Recent 10% of data has {recent_influence:.1%} influence",
                severity=min(1.0, (recent_influence - 0.5) * 2),
                evidence={"recent_influence": recent_influence},
            )
            self.incidents.append(incident)
            self.agent_bias_scores[agent_name]["recency_bias"] += incident.severity
            self._save_history()
            logger.warning(f"Recency bias detected in {agent_name}: {recent_influence:.1%}")
            return incident

        return None

    def detect_loss_aversion(
        self,
        agent_name: str,
        trades: List[Dict],
    ) -> Optional[BiasIncident]:
        """Detect if agent holds losers too long vs winners.

        Classic loss aversion = holding losers 2x longer than winners.
        """
        if len(trades) < 20:
            return None

        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers = [t for t in trades if t.get("pnl", 0) < 0]

        if not winners or not losers:
            return None

        avg_winner_duration = np.mean([t.get("duration_hours", 0) for t in winners])
        avg_loser_duration = np.mean([t.get("duration_hours", 0) for t in losers])

        duration_ratio = avg_loser_duration / (avg_winner_duration + 1e-10)

        if duration_ratio > 2.0:  # Holding losers 2x longer
            incident = BiasIncident(
                bias_type=CognitiveBias.LOSS_AVERSION,
                agent_name=agent_name,
                timestamp=datetime.now(),
                symbol="*",
                description=f"Holding losers {duration_ratio:.1f}x longer than winners",
                severity=min(1.0, (duration_ratio - 2.0) / 3.0),
                evidence={
                    "avg_winner_duration": avg_winner_duration,
                    "avg_loser_duration": avg_loser_duration,
                    "ratio": duration_ratio,
                },
            )
            self.incidents.append(incident)
            self.agent_bias_scores[agent_name]["loss_aversion"] += incident.severity
            self._save_history()
            logger.warning(f"Loss aversion detected in {agent_name}")
            return incident

        return None

    def detect_overconfidence(
        self,
        agent_name: str,
        recent_win_rate: float,
        recent_confidence: float,
        historical_win_rate: float,
    ) -> Optional[BiasIncident]:
        """Detect if winning streak leads to overconfidence.
        """
        # After a winning streak, confidence shouldn't spike unreasonably
        expected_confidence = historical_win_rate + 0.1  # Allow some boost

        if recent_confidence > expected_confidence and recent_win_rate > 0.7:
            overconfidence_gap = recent_confidence - expected_confidence

            incident = BiasIncident(
                bias_type=CognitiveBias.OVERCONFIDENCE,
                agent_name=agent_name,
                timestamp=datetime.now(),
                symbol="*",
                description=f"Confidence {recent_confidence:.0%} after {recent_win_rate:.0%} win rate",
                severity=min(1.0, overconfidence_gap * 5),
                evidence={
                    "recent_win_rate": recent_win_rate,
                    "recent_confidence": recent_confidence,
                    "historical_win_rate": historical_win_rate,
                },
            )
            self.incidents.append(incident)
            self.agent_bias_scores[agent_name]["overconfidence"] += incident.severity
            self._save_history()
            return incident

        return None

    def detect_herd_mentality(
        self,
        agent_name: str,
        agent_signals: List[int],
        consensus_signals: List[int],
    ) -> Optional[BiasIncident]:
        """Detect if agent is just following the crowd.

        >90% agreement with consensus = potential herd behavior.
        """
        if len(agent_signals) < 20:
            return None

        agreement = np.mean([
            1 if a == c else 0
            for a, c in zip(agent_signals, consensus_signals)
        ])

        if agreement > 0.9:
            incident = BiasIncident(
                bias_type=CognitiveBias.HERD_MENTALITY,
                agent_name=agent_name,
                timestamp=datetime.now(),
                symbol="*",
                description=f"{agreement:.0%} agreement with consensus",
                severity=min(1.0, (agreement - 0.9) * 10),
                evidence={"consensus_agreement": agreement},
            )
            self.incidents.append(incident)
            self.agent_bias_scores[agent_name]["herd_mentality"] += incident.severity
            self._save_history()
            return incident

        return None

    def get_bias_report(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive bias report for an agent."""
        agent_incidents = [i for i in self.incidents if i.agent_name == agent_name]

        bias_counts = defaultdict(int)
        bias_severity = defaultdict(float)

        for inc in agent_incidents:
            bias_counts[inc.bias_type.value] += 1
            bias_severity[inc.bias_type.value] += inc.severity

        return {
            "total_incidents": len(agent_incidents),
            "bias_counts": dict(bias_counts),
            "bias_severity": dict(bias_severity),
            "top_bias": max(bias_counts.keys(), key=lambda k: bias_counts[k]) if bias_counts else None,
            "cumulative_scores": dict(self.agent_bias_scores.get(agent_name, {})),
            "recent_incidents": [
                {
                    "type": i.bias_type.value,
                    "description": i.description,
                    "severity": i.severity,
                    "timestamp": i.timestamp.isoformat(),
                }
                for i in agent_incidents[-10:]
            ],
        }

    def get_bias_correction_factor(self, agent_name: str, bias_type: CognitiveBias) -> float:
        """Get correction factor to counter a specific bias.

        Returns multiplier 0.5-1.5 to adjust for bias.
        """
        scores = self.agent_bias_scores.get(agent_name, {})
        bias_score = scores.get(bias_type.value, 0)

        # Higher bias score = stronger correction
        correction = 1.0 - (bias_score * 0.1)  # Each point reduces by 10%
        return max(0.5, min(1.5, correction))


# =============================================================================
# MISTAKE TAXONOMY ANALYZER
# =============================================================================

class MistakeCategory(Enum):
    """Categories of trading mistakes."""

    PREMATURE_EXIT = "premature_exit"         # Exited too early
    LATE_EXIT = "late_exit"                   # Held too long
    WRONG_SIZING = "wrong_sizing"             # Position too big/small
    BAD_TIMING = "bad_timing"                 # Right direction, wrong time
    WRONG_DIRECTION = "wrong_direction"       # Completely wrong
    MISSED_OPPORTUNITY = "missed_opportunity" # Didn't trade when should have
    CHASING = "chasing"                       # Entered after move
    FIGHTING_TREND = "fighting_trend"         # Counter-trend in strong trend
    OVER_TRADING = "over_trading"             # Too many trades
    UNDER_TRADING = "under_trading"           # Too few trades when opportunities exist


@dataclass
class MistakeRecord:
    """Record of a trading mistake."""

    mistake_id: str
    category: MistakeCategory
    agent_name: str
    timestamp: datetime
    symbol: str
    description: str
    cost_estimate: float  # Estimated P&L impact
    root_cause: str
    correction_applied: bool = False


class MistakeTaxonomyAnalyzer:
    """Classify and analyze trading mistakes systematically.

    Different mistakes need different corrections:
    - Premature exits -> Adjust trailing stops
    - Wrong sizing -> Recalibrate Kelly criterion
    - Bad timing -> Improve entry logic

    USAGE:
        analyzer = MistakeTaxonomyAnalyzer()
        analyzer.record_mistake(
            category=MistakeCategory.PREMATURE_EXIT,
            agent_name="MOMENTUM",
            symbol="AAPL",
            description="Exited at +2% when trend continued to +8%",
            cost_estimate=6000
        )
        weaknesses = analyzer.get_weakness_profile("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/mistakes")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.mistakes: Dict[str, MistakeRecord] = {}
        self.correction_strategies: Dict[MistakeCategory, str] = {
            MistakeCategory.PREMATURE_EXIT: "Widen trailing stops, use ATR-based exits",
            MistakeCategory.LATE_EXIT: "Tighten stops, use time-based exits",
            MistakeCategory.WRONG_SIZING: "Recalibrate position sizing model",
            MistakeCategory.BAD_TIMING: "Add time-of-day/week filters",
            MistakeCategory.WRONG_DIRECTION: "Review signal logic, add confirmation filters",
            MistakeCategory.MISSED_OPPORTUNITY: "Lower entry threshold, reduce filters",
            MistakeCategory.CHASING: "Add momentum exhaustion detection",
            MistakeCategory.FIGHTING_TREND: "Add trend strength filter",
            MistakeCategory.OVER_TRADING: "Increase min confidence threshold",
            MistakeCategory.UNDER_TRADING: "Reduce signal requirements",
        }
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "mistakes.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for m in data:
                    mistake = MistakeRecord(
                        mistake_id=m["mistake_id"],
                        category=MistakeCategory(m["category"]),
                        agent_name=m["agent_name"],
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        symbol=m["symbol"],
                        description=m["description"],
                        cost_estimate=m["cost_estimate"],
                        root_cause=m["root_cause"],
                        correction_applied=m.get("correction_applied", False),
                    )
                    self.mistakes[mistake.mistake_id] = mistake
            except Exception as e:
                logger.warning(f"Could not load mistake history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "mistakes.json"
        data = [
            {
                "mistake_id": m.mistake_id,
                "category": m.category.value,
                "agent_name": m.agent_name,
                "timestamp": m.timestamp.isoformat(),
                "symbol": m.symbol,
                "description": m.description,
                "cost_estimate": m.cost_estimate,
                "root_cause": m.root_cause,
                "correction_applied": m.correction_applied,
            }
            for m in self.mistakes.values()
        ]
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_mistake(
        self,
        category: MistakeCategory,
        agent_name: str,
        symbol: str,
        description: str,
        cost_estimate: float,
        root_cause: str = "",
    ) -> str:
        """Record a trading mistake."""
        mistake_id = f"{agent_name}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        mistake = MistakeRecord(
            mistake_id=mistake_id,
            category=category,
            agent_name=agent_name,
            timestamp=datetime.now(),
            symbol=symbol,
            description=description,
            cost_estimate=cost_estimate,
            root_cause=root_cause or self.correction_strategies.get(category, "Unknown"),
        )

        self.mistakes[mistake_id] = mistake
        self._save_history()

        logger.info(f"Recorded mistake {category.value} for {agent_name}: ${cost_estimate:,.0f}")
        return mistake_id

    def auto_classify_mistake(
        self,
        agent_name: str,
        symbol: str,
        entry_price: float,
        exit_price: float,
        actual_low: float,
        actual_high: float,
        holding_period_hours: float,
        signal_direction: int,  # 1 = long, -1 = short
    ) -> Optional[MistakeCategory]:
        """Automatically classify a trade mistake based on outcomes.
        """
        trade_return = (exit_price - entry_price) / entry_price * signal_direction
        max_possible = (actual_high - entry_price) / entry_price * signal_direction
        max_adverse = (actual_low - entry_price) / entry_price * signal_direction

        # Lost money
        if trade_return < 0:
            # Check if we were fighting the trend
            if signal_direction > 0 and actual_high < entry_price:
                return MistakeCategory.FIGHTING_TREND
            elif signal_direction < 0 and actual_low > entry_price:
                return MistakeCategory.FIGHTING_TREND

            # Wrong direction entirely
            if max_possible < -0.01:
                return MistakeCategory.WRONG_DIRECTION

            # Just bad timing
            return MistakeCategory.BAD_TIMING

        # Made money but could have made more
        if trade_return > 0:
            missed_upside = max_possible - trade_return

            # Exited too early
            if missed_upside > trade_return:  # Left more on table than captured
                return MistakeCategory.PREMATURE_EXIT

            # Position too small (good trade but under-sized)
            # This would need position sizing info to detect

        return None

    def get_weakness_profile(self, agent_name: str) -> Dict[str, Any]:
        """Get profile of agent's weaknesses."""
        agent_mistakes = [m for m in self.mistakes.values() if m.agent_name == agent_name]

        if not agent_mistakes:
            return {"status": "No mistakes recorded"}

        category_counts = defaultdict(int)
        category_costs = defaultdict(float)

        for m in agent_mistakes:
            category_counts[m.category.value] += 1
            category_costs[m.category.value] += m.cost_estimate

        # Find top weakness
        top_weakness = max(category_counts.keys(), key=lambda k: category_costs[k])

        return {
            "total_mistakes": len(agent_mistakes),
            "total_cost": sum(m.cost_estimate for m in agent_mistakes),
            "category_counts": dict(category_counts),
            "category_costs": dict(category_costs),
            "top_weakness": top_weakness,
            "correction_strategy": self.correction_strategies.get(
                MistakeCategory(top_weakness), "Unknown",
            ),
            "recent_mistakes": [
                {
                    "category": m.category.value,
                    "symbol": m.symbol,
                    "cost": m.cost_estimate,
                    "description": m.description,
                }
                for m in sorted(agent_mistakes, key=lambda x: x.timestamp, reverse=True)[:5]
            ],
        }


# =============================================================================
# ANTI-FRAGILITY SCORER
# =============================================================================

class AntifragilityScorer:
    """Measure if agents get BETTER after stress.

    Fragile = breaks under stress
    Robust = survives stress
    Antifragile = IMPROVES under stress

    We want antifragile agents.

    USAGE:
        scorer = AntifragilityScorer()
        scorer.record_stress_event(
            agent_name="MOMENTUM",
            stress_type="drawdown",
            pre_stress_sharpe=1.5,
            post_stress_sharpe=1.8
        )
        score = scorer.get_antifragility_score("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/antifragility")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # agent_name -> list of (pre_stress_metric, post_stress_metric)
        self.stress_responses: Dict[str, List[Dict]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "stress_responses.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.stress_responses = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load stress history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "stress_responses.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.stress_responses), f, indent=2)

    def record_stress_event(
        self,
        agent_name: str,
        stress_type: str,  # "drawdown", "high_vol", "black_swan", "regime_change"
        pre_stress_sharpe: float,
        post_stress_sharpe: float,
        stress_magnitude: float = 0.0,
        recovery_time_days: int = 0,
    ):
        """Record how agent responded to stress."""
        self.stress_responses[agent_name].append({
            "stress_type": stress_type,
            "pre_stress_sharpe": pre_stress_sharpe,
            "post_stress_sharpe": post_stress_sharpe,
            "improvement": post_stress_sharpe - pre_stress_sharpe,
            "stress_magnitude": stress_magnitude,
            "recovery_time_days": recovery_time_days,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_history()

    def get_antifragility_score(self, agent_name: str) -> Dict[str, Any]:
        """Calculate antifragility score.

        Score > 0 = antifragile (improves under stress)
        Score = 0 = robust (survives stress)
        Score < 0 = fragile (breaks under stress)
        """
        responses = self.stress_responses.get(agent_name, [])

        if len(responses) < 3:
            return {"score": 0, "status": "Insufficient stress events"}

        improvements = [r["improvement"] for r in responses]

        # Calculate antifragility metrics
        avg_improvement = np.mean(improvements)
        improvement_in_severe = [
            r["improvement"] for r in responses
            if r.get("stress_magnitude", 0) > 0.5
        ]

        # Recovery speed
        recovery_times = [r.get("recovery_time_days", 30) for r in responses]
        avg_recovery = np.mean(recovery_times)

        # Score calculation
        # Positive if improving after stress, negative if degrading
        antifragility_score = avg_improvement * 10  # Scale to -10 to +10

        # Bonus for fast recovery
        if avg_recovery < 5:
            antifragility_score += 1

        # Bonus for improvement in severe stress
        if improvement_in_severe and np.mean(improvement_in_severe) > 0:
            antifragility_score += 2

        return {
            "score": antifragility_score,
            "avg_improvement": avg_improvement,
            "avg_recovery_days": avg_recovery,
            "stress_events": len(responses),
            "classification": (
                "ANTIFRAGILE" if antifragility_score > 1 else
                "ROBUST" if antifragility_score > -1 else
                "FRAGILE"
            ),
            "by_stress_type": self._group_by_stress_type(responses),
        }

    def _group_by_stress_type(self, responses: List[Dict]) -> Dict[str, float]:
        """Group responses by stress type."""
        by_type = defaultdict(list)
        for r in responses:
            by_type[r["stress_type"]].append(r["improvement"])

        return {k: np.mean(v) for k, v in by_type.items()}


# =============================================================================
# PATTERN FATIGUE MONITOR
# =============================================================================

class PatternFatigueMonitor:
    """Detect when trading patterns stop working.

    Patterns have lifecycles:
    1. Discovery - High alpha, low usage
    2. Growth - Spreading adoption, declining alpha
    3. Maturity - Widespread use, minimal alpha
    4. Decay - Negative alpha as pattern fails

    USAGE:
        monitor = PatternFatigueMonitor()
        monitor.record_pattern_performance(
            pattern_name="momentum_breakout",
            timestamp=datetime.now(),
            win_rate=0.55,
            avg_return=0.008
        )
        status = monitor.get_pattern_lifecycle_status("momentum_breakout")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/pattern_fatigue")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # pattern_name -> time series of performance
        self.pattern_history: Dict[str, List[Dict]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "patterns.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.pattern_history = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load pattern history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "patterns.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.pattern_history), f, indent=2)

    def record_pattern_performance(
        self,
        pattern_name: str,
        win_rate: float,
        avg_return: float,
        n_signals: int = 0,
    ):
        """Record pattern performance at a point in time."""
        self.pattern_history[pattern_name].append({
            "timestamp": datetime.now().isoformat(),
            "win_rate": win_rate,
            "avg_return": avg_return,
            "n_signals": n_signals,
        })
        self._save_history()

    def get_pattern_lifecycle_status(self, pattern_name: str) -> Dict[str, Any]:
        """Determine where pattern is in its lifecycle.
        """
        history = self.pattern_history.get(pattern_name, [])

        if len(history) < 5:
            return {"status": "UNKNOWN", "message": "Insufficient history"}

        # Get recent vs historical performance
        recent = history[-10:]
        historical = history[:-10] if len(history) > 10 else history

        recent_avg_return = np.mean([h["avg_return"] for h in recent])
        historical_avg_return = np.mean([h["avg_return"] for h in historical])

        recent_win_rate = np.mean([h["win_rate"] for h in recent])
        historical_win_rate = np.mean([h["win_rate"] for h in historical])

        # Calculate degradation
        return_degradation = (historical_avg_return - recent_avg_return) / (abs(historical_avg_return) + 1e-10)
        win_rate_degradation = historical_win_rate - recent_win_rate

        # Determine lifecycle stage
        if recent_avg_return > historical_avg_return and recent_avg_return > 0.01:
            status = "DISCOVERY"
            recommendation = "Increase allocation, pattern is working well"
        elif return_degradation < 0.2 and recent_avg_return > 0.005:
            status = "GROWTH"
            recommendation = "Maintain allocation, pattern still has alpha"
        elif return_degradation < 0.5 and recent_avg_return > 0:
            status = "MATURITY"
            recommendation = "Reduce allocation, alpha is declining"
        else:
            status = "DECAY"
            recommendation = "STOP using this pattern, consider opposite"

        return {
            "status": status,
            "recommendation": recommendation,
            "recent_avg_return": recent_avg_return,
            "historical_avg_return": historical_avg_return,
            "return_degradation": return_degradation,
            "recent_win_rate": recent_win_rate,
            "win_rate_degradation": win_rate_degradation,
            "data_points": len(history),
        }

    def get_all_pattern_statuses(self) -> Dict[str, Dict]:
        """Get lifecycle status for all tracked patterns."""
        return {
            pattern: self.get_pattern_lifecycle_status(pattern)
            for pattern in self.pattern_history.keys()
        }


# =============================================================================
# OPPORTUNITY COST TRACKER
# =============================================================================

class OpportunityCostTracker:
    """Track trades NOT taken and learn from missed opportunities.

    Sometimes the best learning is from what you DIDN'T do.

    USAGE:
        tracker = OpportunityCostTracker()
        tracker.record_missed_opportunity(
            agent_name="MOMENTUM",
            symbol="NVDA",
            signal_generated=True,
            reason_not_traded="Confidence below threshold",
            what_would_have_happened=0.15  # 15% gain
        )
        cost = tracker.get_opportunity_cost_report("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/opportunity_cost")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.missed_opportunities: List[Dict] = []
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "missed.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.missed_opportunities = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load opportunity history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "missed.json"
        with open(history_file, "w") as f:
            json.dump(self.missed_opportunities[-5000:], f, indent=2)

    def record_missed_opportunity(
        self,
        agent_name: str,
        symbol: str,
        signal_generated: bool,
        reason_not_traded: str,
        hypothetical_return: float,
        confidence_at_time: float = 0,
        regime_at_time: str = "",
    ):
        """Record a missed trading opportunity."""
        self.missed_opportunities.append({
            "agent_name": agent_name,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal_generated": signal_generated,
            "reason": reason_not_traded,
            "hypothetical_return": hypothetical_return,
            "confidence": confidence_at_time,
            "regime": regime_at_time,
        })
        self._save_history()

        if hypothetical_return > 0.05:
            logger.info(f"Missed opportunity in {symbol}: {hypothetical_return:.1%} ({reason_not_traded})")

    def get_opportunity_cost_report(self, agent_name: str) -> Dict[str, Any]:
        """Get report of missed opportunities."""
        agent_misses = [
            m for m in self.missed_opportunities
            if m["agent_name"] == agent_name
        ]

        if not agent_misses:
            return {"status": "No missed opportunities recorded"}

        profitable_misses = [m for m in agent_misses if m["hypothetical_return"] > 0]

        # Group by reason
        by_reason = defaultdict(list)
        for m in agent_misses:
            by_reason[m["reason"]].append(m["hypothetical_return"])

        reason_analysis = {
            reason: {
                "count": len(returns),
                "avg_missed_return": np.mean(returns),
                "total_opportunity_cost": sum(max(0, r) for r in returns),
            }
            for reason, returns in by_reason.items()
        }

        # Find the most costly reason
        worst_reason = max(
            reason_analysis.keys(),
            key=lambda k: reason_analysis[k]["total_opportunity_cost"],
        ) if reason_analysis else None

        return {
            "total_missed": len(agent_misses),
            "profitable_missed": len(profitable_misses),
            "total_opportunity_cost": sum(m["hypothetical_return"] for m in profitable_misses),
            "avg_missed_return": np.mean([m["hypothetical_return"] for m in agent_misses]),
            "by_reason": reason_analysis,
            "worst_filter": worst_reason,
            "recommendation": f"Review '{worst_reason}' filter - may be too strict" if worst_reason else None,
            "top_missed": sorted(
                agent_misses,
                key=lambda x: x["hypothetical_return"],
                reverse=True,
            )[:10],
        }


# =============================================================================
# META-LEARNING OPTIMIZER
# =============================================================================

class MetaLearningOptimizer:
    """Learn HOW to learn better.

    Optimize the learning process itself:
    - Which learning rate works best?
    - How much data should we use?
    - When should we retrain?

    USAGE:
        optimizer = MetaLearningOptimizer()
        optimizer.record_learning_experiment(
            agent_name="MOMENTUM",
            hyperparameters={"learning_rate": 0.01, "batch_size": 32},
            pre_experiment_sharpe=1.2,
            post_experiment_sharpe=1.5,
            experiment_duration_days=30
        )
        optimal = optimizer.get_optimal_learning_config("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/meta_learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # agent_name -> list of learning experiments
        self.experiments: Dict[str, List[Dict]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "experiments.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.experiments = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load meta-learning history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "experiments.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.experiments), f, indent=2)

    def record_learning_experiment(
        self,
        agent_name: str,
        hyperparameters: Dict[str, Any],
        pre_experiment_sharpe: float,
        post_experiment_sharpe: float,
        experiment_duration_days: int,
        data_size: int = 0,
        retrain_frequency_days: int = 0,
    ):
        """Record results of a learning experiment."""
        improvement = post_experiment_sharpe - pre_experiment_sharpe
        efficiency = improvement / (experiment_duration_days + 1)  # Improvement per day

        self.experiments[agent_name].append({
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "pre_sharpe": pre_experiment_sharpe,
            "post_sharpe": post_experiment_sharpe,
            "improvement": improvement,
            "efficiency": efficiency,
            "duration_days": experiment_duration_days,
            "data_size": data_size,
            "retrain_frequency": retrain_frequency_days,
        })
        self._save_history()

    def get_optimal_learning_config(self, agent_name: str) -> Dict[str, Any]:
        """Determine optimal learning configuration based on experiments.
        """
        agent_experiments = self.experiments.get(agent_name, [])

        if len(agent_experiments) < 5:
            return {
                "status": "Insufficient experiments",
                "default_config": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "retrain_frequency_days": 7,
                    "lookback_days": 365,
                },
            }

        # Find best experiment by improvement
        best_experiment = max(agent_experiments, key=lambda x: x["improvement"])

        # Find most efficient experiment (improvement per day)
        most_efficient = max(agent_experiments, key=lambda x: x["efficiency"])

        # Aggregate hyperparameter performance
        hp_performance: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

        for exp in agent_experiments:
            for hp_name, hp_value in exp["hyperparameters"].items():
                hp_performance[hp_name][str(hp_value)].append(exp["improvement"])

        # Find best value for each hyperparameter
        optimal_hps = {}
        for hp_name, value_results in hp_performance.items():
            best_value = max(value_results.keys(), key=lambda v: np.mean(value_results[v]))
            optimal_hps[hp_name] = best_value

        return {
            "optimal_config": optimal_hps,
            "best_experiment": best_experiment["hyperparameters"],
            "best_improvement": best_experiment["improvement"],
            "most_efficient": most_efficient["hyperparameters"],
            "best_efficiency": most_efficient["efficiency"],
            "total_experiments": len(agent_experiments),
            "recommendation": (
                "Try more experiments" if len(agent_experiments) < 20
                else "Use optimal_config for best results"
            ),
        }


# =============================================================================
# SKILL ATTRIBUTION MATRIX
# =============================================================================

class SkillAttributionMatrix:
    """Decompose returns into skill vs luck.

    Was profit from:
    - Timing skill?
    - Security selection?
    - Position sizing?
    - Risk management?
    - Pure luck?

    USAGE:
        matrix = SkillAttributionMatrix()
        matrix.record_trade_attribution(
            agent_name="MOMENTUM",
            trade_id="123",
            timing_contribution=0.02,
            selection_contribution=0.03,
            sizing_contribution=0.01,
            risk_mgmt_contribution=0.005
        )
        attribution = matrix.get_skill_breakdown("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/skill_attribution")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # agent_name -> list of attributions
        self.attributions: Dict[str, List[Dict]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "attributions.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.attributions = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load attribution history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "attributions.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.attributions), f, indent=2)

    def record_trade_attribution(
        self,
        agent_name: str,
        trade_id: str,
        total_return: float,
        timing_contribution: float,      # Entry/exit timing
        selection_contribution: float,   # Picking the right security
        sizing_contribution: float,      # Position sizing
        risk_mgmt_contribution: float,   # Stop loss / take profit
        market_contribution: float = 0,  # Beta exposure
    ):
        """Record skill attribution for a trade."""
        # Residual is luck
        skill_sum = (timing_contribution + selection_contribution +
                     sizing_contribution + risk_mgmt_contribution + market_contribution)
        luck_contribution = total_return - skill_sum

        self.attributions[agent_name].append({
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "total_return": total_return,
            "timing": timing_contribution,
            "selection": selection_contribution,
            "sizing": sizing_contribution,
            "risk_mgmt": risk_mgmt_contribution,
            "market": market_contribution,
            "luck": luck_contribution,
        })
        self._save_history()

    def get_skill_breakdown(self, agent_name: str) -> Dict[str, Any]:
        """Get breakdown of where returns come from."""
        agent_attributions = self.attributions.get(agent_name, [])

        if not agent_attributions:
            return {"status": "No attributions recorded"}

        df = pd.DataFrame(agent_attributions)

        # Calculate contribution percentages
        total_return = df["total_return"].sum()

        if abs(total_return) < 0.001:
            return {"status": "Insufficient returns for attribution"}

        breakdown = {
            "timing_pct": df["timing"].sum() / total_return * 100,
            "selection_pct": df["selection"].sum() / total_return * 100,
            "sizing_pct": df["sizing"].sum() / total_return * 100,
            "risk_mgmt_pct": df["risk_mgmt"].sum() / total_return * 100,
            "market_pct": df["market"].sum() / total_return * 100,
            "luck_pct": df["luck"].sum() / total_return * 100,
        }

        # Identify strongest skill
        skill_only = {k: v for k, v in breakdown.items() if k not in ["market_pct", "luck_pct"]}
        strongest_skill = max(skill_only.keys(), key=lambda k: skill_only[k])
        weakest_skill = min(skill_only.keys(), key=lambda k: skill_only[k])

        # Skill vs luck ratio
        total_skill = sum(v for k, v in breakdown.items() if k not in ["luck_pct", "market_pct"])
        skill_luck_ratio = total_skill / (abs(breakdown["luck_pct"]) + 1)

        return {
            "breakdown": breakdown,
            "strongest_skill": strongest_skill.replace("_pct", ""),
            "weakest_skill": weakest_skill.replace("_pct", ""),
            "skill_luck_ratio": skill_luck_ratio,
            "total_trades_analyzed": len(agent_attributions),
            "recommendation": (
                f"Focus on improving {weakest_skill.replace('_pct', '')}"
                if skill_luck_ratio > 1 else
                "Returns may be luck-driven, need more skill development"
            ),
        }


# =============================================================================
# ENHANCED MASTER LEARNING ENGINE
# =============================================================================

class EnhancedLearningEngine(LearningEngine):
    """ENHANCED Learning Engine with all Part III components.

    This is the MASTER system that orchestrates all learning.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__(storage_path)

        # Part III components
        self.bias_detector = CognitiveBiasDetector(self.storage_path / "biases")
        self.mistake_analyzer = MistakeTaxonomyAnalyzer(self.storage_path / "mistakes")
        self.antifragility_scorer = AntifragilityScorer(self.storage_path / "antifragility")
        self.pattern_monitor = PatternFatigueMonitor(self.storage_path / "patterns")
        self.opportunity_tracker = OpportunityCostTracker(self.storage_path / "opportunities")
        self.meta_optimizer = MetaLearningOptimizer(self.storage_path / "meta")
        self.skill_matrix = SkillAttributionMatrix(self.storage_path / "skills")

        logger.info("Enhanced Learning Engine initialized with all Part III components")

    def get_comprehensive_agent_report(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive learning report for an agent.

        Includes ALL learning metrics across all systems.
        """
        return {
            "agent_name": agent_name,
            "generated_at": datetime.now().isoformat(),

            # Part II metrics
            "learning_score": self.get_learning_score(agent_name),

            # Part III metrics
            "cognitive_biases": self.bias_detector.get_bias_report(agent_name),
            "weakness_profile": self.mistake_analyzer.get_weakness_profile(agent_name),
            "antifragility": self.antifragility_scorer.get_antifragility_score(agent_name),
            "opportunity_cost": self.opportunity_tracker.get_opportunity_cost_report(agent_name),
            "optimal_learning_config": self.meta_optimizer.get_optimal_learning_config(agent_name),
            "skill_attribution": self.skill_matrix.get_skill_breakdown(agent_name),

            # Pattern statuses (all patterns)
            "pattern_statuses": self.pattern_monitor.get_all_pattern_statuses(),
        }

    def get_agent_improvement_plan(self, agent_name: str) -> Dict[str, Any]:
        """Generate actionable improvement plan for an agent.
        """
        report = self.get_comprehensive_agent_report(agent_name)

        priorities = []

        # Check biases
        bias_report = report["cognitive_biases"]
        if bias_report.get("total_incidents", 0) > 5:
            priorities.append({
                "area": "Cognitive Bias",
                "issue": f"Detected {bias_report['total_incidents']} bias incidents",
                "action": f"Focus on correcting {bias_report.get('top_bias', 'unknown')} bias",
                "priority": "HIGH",
            })

        # Check weaknesses
        weakness = report["weakness_profile"]
        if weakness.get("total_cost", 0) > 1000:
            priorities.append({
                "area": "Trading Mistakes",
                "issue": f"${weakness['total_cost']:,.0f} in preventable losses",
                "action": weakness.get("correction_strategy", "Review trading logic"),
                "priority": "HIGH",
            })

        # Check antifragility
        antifragility = report["antifragility"]
        if antifragility.get("classification") == "FRAGILE":
            priorities.append({
                "area": "Stress Resilience",
                "issue": "Agent is FRAGILE under stress",
                "action": "Add adaptive mechanisms, improve drawdown recovery",
                "priority": "CRITICAL",
            })

        # Check opportunity cost
        opportunity = report["opportunity_cost"]
        if opportunity.get("total_opportunity_cost", 0) > 0.1:
            priorities.append({
                "area": "Opportunity Cost",
                "issue": f"Missed {opportunity['total_opportunity_cost']:.1%} in returns",
                "action": opportunity.get("recommendation", "Review trade filters"),
                "priority": "MEDIUM",
            })

        # Check skill vs luck
        skill = report["skill_attribution"]
        if skill.get("skill_luck_ratio", 0) < 1:
            priorities.append({
                "area": "Skill Development",
                "issue": "Returns may be luck-driven",
                "action": skill.get("recommendation", "Develop core skills"),
                "priority": "HIGH",
            })

        return {
            "agent_name": agent_name,
            "priorities": sorted(priorities, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x["priority"]]),
            "top_3_actions": [p["action"] for p in priorities[:3]],
            "overall_health": (
                "CRITICAL" if any(p["priority"] == "CRITICAL" for p in priorities)
                else "NEEDS_WORK" if any(p["priority"] == "HIGH" for p in priorities)
                else "GOOD" if priorities
                else "EXCELLENT"
            ),
        }


# =============================================================================
# PART IV: EDGE INTELLIGENCE SYSTEMS
# =============================================================================
# The most advanced features - TRUE competitive advantage

# =============================================================================
# ADVERSARIAL ROBUSTNESS TESTER
# =============================================================================

class AdversarialScenario(Enum):
    """Types of adversarial market scenarios."""

    FRONT_RUNNING = "front_running"           # Someone front-runs our orders
    SPOOFING = "spoofing"                     # Fake orders manipulate price
    STOP_HUNTING = "stop_hunting"             # Market moves to hit stops
    FLASH_CRASH = "flash_crash"               # Sudden price collapse
    LIQUIDITY_TRAP = "liquidity_trap"         # Can't exit position
    CORRELATION_BREAK = "correlation_break"   # Normal correlations fail
    NEWS_MANIPULATION = "news_manipulation"   # Fake news moves market
    WASH_TRADING = "wash_trading"             # Fake volume
    REGIME_SHIFT = "regime_shift"             # Sudden market regime change


class AdversarialRobustnessTester:
    """Test agents against market manipulation and adversarial conditions.

    In real markets, others are trying to exploit you.
    This tests how robust your strategies are.

    USAGE:
        tester = AdversarialRobustnessTester()
        result = tester.simulate_front_running(
            agent=momentum_agent,
            scenario_intensity=0.5
        )
        robustness = tester.get_robustness_score("MOMENTUM")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/adversarial")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # agent_name -> scenario -> list of results
        self.test_results: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "adversarial_tests.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for agent, scenarios in data.items():
                    for scenario, results in scenarios.items():
                        self.test_results[agent][scenario] = results
            except Exception as e:
                logger.warning(f"Could not load adversarial history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "adversarial_tests.json"
        with open(history_file, "w") as f:
            json.dump({
                agent: dict(scenarios)
                for agent, scenarios in self.test_results.items()
            }, f, indent=2)

    def simulate_front_running(
        self,
        agent_name: str,
        original_signals: List[Dict],
        front_run_delay_ms: int = 100,
        front_run_size_multiplier: float = 10.0,
    ) -> Dict[str, Any]:
        """Simulate front-running attack.

        Tests: If someone sees your orders 100ms before execution
        and trades 10x your size, what happens?
        """
        impacted_returns = []
        original_returns = []

        for signal in original_signals:
            original_return = signal.get("return", 0)
            original_returns.append(original_return)

            # Front-running causes slippage
            # Impact = proportional to trade size and front-runner size
            slippage = 0.001 * front_run_size_multiplier  # 0.1% per 10x multiplier

            if signal.get("direction") == "BUY":
                impacted_return = original_return - slippage
            else:
                impacted_return = original_return - slippage

            impacted_returns.append(impacted_return)

        original_sharpe = np.mean(original_returns) / (np.std(original_returns) + 1e-10) * np.sqrt(252)
        impacted_sharpe = np.mean(impacted_returns) / (np.std(impacted_returns) + 1e-10) * np.sqrt(252)

        survival_rate = impacted_sharpe / (original_sharpe + 1e-10) if original_sharpe > 0 else 0

        result = {
            "scenario": AdversarialScenario.FRONT_RUNNING.value,
            "timestamp": datetime.now().isoformat(),
            "original_sharpe": original_sharpe,
            "impacted_sharpe": impacted_sharpe,
            "sharpe_degradation": original_sharpe - impacted_sharpe,
            "survival_rate": survival_rate,
            "passed": survival_rate > 0.7,  # Still retain 70% of Sharpe
            "parameters": {
                "delay_ms": front_run_delay_ms,
                "size_multiplier": front_run_size_multiplier,
            },
        }

        self.test_results[agent_name][AdversarialScenario.FRONT_RUNNING.value].append(result)
        self._save_history()

        return result

    def simulate_stop_hunting(
        self,
        agent_name: str,
        trades: List[Dict],
        stop_levels: List[float],
        hunt_probability: float = 0.3,
    ) -> Dict[str, Any]:
        """Simulate stop-loss hunting.

        Market "knows" where stops are and hunts them.
        """
        stopped_out = 0
        false_stops = 0  # Stops that got hit but price recovered

        for trade, stop in zip(trades, stop_levels):
            # Simulate if market would have hunted this stop
            if np.random.random() < hunt_probability:
                stopped_out += 1
                # 50% of hunted stops would have recovered
                if np.random.random() < 0.5:
                    false_stops += 1

        false_stop_rate = false_stops / (stopped_out + 1)

        result = {
            "scenario": AdversarialScenario.STOP_HUNTING.value,
            "timestamp": datetime.now().isoformat(),
            "total_trades": len(trades),
            "stopped_out": stopped_out,
            "false_stops": false_stops,
            "false_stop_rate": false_stop_rate,
            "passed": false_stop_rate < 0.3,  # Less than 30% false stops
        }

        self.test_results[agent_name][AdversarialScenario.STOP_HUNTING.value].append(result)
        self._save_history()

        return result

    def simulate_liquidity_trap(
        self,
        agent_name: str,
        position_sizes: List[float],
        avg_daily_volumes: List[float],
    ) -> Dict[str, Any]:
        """Test if positions can actually be exited.

        Large position + low liquidity = trapped.
        """
        trap_risk_scores = []

        for size, volume in zip(position_sizes, avg_daily_volumes):
            # Days to exit at 10% of daily volume
            days_to_exit = size / (volume * 0.1)

            # Trap risk increases with exit time
            trap_risk = min(1.0, days_to_exit / 10)  # 10+ days = max risk
            trap_risk_scores.append(trap_risk)

        avg_trap_risk = np.mean(trap_risk_scores)
        max_trap_risk = max(trap_risk_scores)

        result = {
            "scenario": AdversarialScenario.LIQUIDITY_TRAP.value,
            "timestamp": datetime.now().isoformat(),
            "avg_trap_risk": avg_trap_risk,
            "max_trap_risk": max_trap_risk,
            "positions_at_risk": sum(1 for t in trap_risk_scores if t > 0.5),
            "passed": max_trap_risk < 0.5,
        }

        self.test_results[agent_name][AdversarialScenario.LIQUIDITY_TRAP.value].append(result)
        self._save_history()

        return result

    def get_robustness_score(self, agent_name: str) -> Dict[str, Any]:
        """Get overall adversarial robustness score."""
        agent_results = self.test_results.get(agent_name, {})

        if not agent_results:
            return {"status": "No adversarial tests run"}

        scenario_scores = {}

        for scenario, results in agent_results.items():
            if results:
                passed = sum(1 for r in results if r.get("passed", False))
                scenario_scores[scenario] = {
                    "tests_run": len(results),
                    "passed": passed,
                    "pass_rate": passed / len(results),
                }

        overall_pass_rate = np.mean([
            s["pass_rate"] for s in scenario_scores.values()
        ]) if scenario_scores else 0

        return {
            "overall_robustness": overall_pass_rate,
            "classification": (
                "HIGHLY_ROBUST" if overall_pass_rate > 0.9 else
                "ROBUST" if overall_pass_rate > 0.7 else
                "VULNERABLE" if overall_pass_rate > 0.5 else
                "HIGHLY_VULNERABLE"
            ),
            "by_scenario": scenario_scores,
            "weakest_area": min(
                scenario_scores.keys(),
                key=lambda k: scenario_scores[k]["pass_rate"],
            ) if scenario_scores else None,
        }


# =============================================================================
# INFORMATION EDGE QUANTIFIER
# =============================================================================

class InformationSource(Enum):
    """Types of information sources."""

    PRICE_ACTION = "price_action"
    VOLUME = "volume"
    ORDER_FLOW = "order_flow"
    SENTIMENT = "sentiment"
    FUNDAMENTALS = "fundamentals"
    MACRO = "macro"
    ALTERNATIVE_DATA = "alternative_data"
    INSIDER_ACTIVITY = "insider_activity"
    OPTIONS_FLOW = "options_flow"
    TECHNICAL_INDICATORS = "technical_indicators"


class InformationEdgeQuantifier:
    """Quantify which information sources generate REAL alpha.

    Not all data is equal. This measures which sources
    actually predict returns vs which are noise.

    USAGE:
        quantifier = InformationEdgeQuantifier()
        quantifier.record_signal_source(
            source=InformationSource.OPTIONS_FLOW,
            signal_direction=1,
            actual_return=0.05,
            confidence=0.8
        )
        edge = quantifier.get_information_edge(InformationSource.OPTIONS_FLOW)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/information_edge")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # source -> list of (signal, actual_return)
        self.source_signals: Dict[str, List[Dict]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "info_edge.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.source_signals = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load information edge history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "info_edge.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.source_signals), f, indent=2)

    def record_signal_source(
        self,
        source: InformationSource,
        signal_direction: int,
        actual_return: float,
        confidence: float,
        time_to_realization_hours: float = 0,
    ):
        """Record a signal and its outcome from a specific source."""
        self.source_signals[source.value].append({
            "timestamp": datetime.now().isoformat(),
            "direction": signal_direction,
            "actual_return": actual_return,
            "confidence": confidence,
            "correct": (actual_return > 0 and signal_direction > 0) or (actual_return < 0 and signal_direction < 0),
            "time_to_realization": time_to_realization_hours,
        })
        self._save_history()

    def get_information_edge(self, source: InformationSource) -> Dict[str, Any]:
        """Quantify the edge from a specific information source."""
        signals = self.source_signals.get(source.value, [])

        if len(signals) < 30:
            return {"status": "Insufficient signals", "n_signals": len(signals)}

        returns = [s["actual_return"] * s["direction"] for s in signals]
        correct = [s["correct"] for s in signals]
        confidences = [s["confidence"] for s in signals]

        # Information coefficient (IC)
        # Correlation between signal and actual return
        ic = np.corrcoef(
            [s["direction"] * s["confidence"] for s in signals],
            [s["actual_return"] for s in signals],
        )[0, 1]

        if np.isnan(ic):
            ic = 0

        # Sharpe of this signal source
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

        # Time value - does the edge decay?
        time_values = [s.get("time_to_realization", 24) for s in signals]
        early_signals = [s for s in signals if s.get("time_to_realization", 24) < 24]
        late_signals = [s for s in signals if s.get("time_to_realization", 24) >= 24]

        early_accuracy = np.mean([s["correct"] for s in early_signals]) if early_signals else 0
        late_accuracy = np.mean([s["correct"] for s in late_signals]) if late_signals else 0

        return {
            "source": source.value,
            "n_signals": len(signals),
            "hit_rate": np.mean(correct),
            "avg_return": np.mean(returns),
            "sharpe": sharpe,
            "information_coefficient": ic,
            "early_accuracy": early_accuracy,
            "late_accuracy": late_accuracy,
            "time_decay": early_accuracy - late_accuracy,  # Positive = edge decays
            "edge_classification": (
                "STRONG_EDGE" if ic > 0.1 and sharpe > 1.0 else
                "MODERATE_EDGE" if ic > 0.05 and sharpe > 0.5 else
                "WEAK_EDGE" if ic > 0 and sharpe > 0 else
                "NO_EDGE"
            ),
        }

    def get_information_ranking(self) -> List[Dict]:
        """Rank all information sources by edge quality."""
        rankings = []

        for source in InformationSource:
            edge = self.get_information_edge(source)
            if edge.get("status") != "Insufficient signals":
                rankings.append({
                    "source": source.value,
                    "ic": edge.get("information_coefficient", 0),
                    "sharpe": edge.get("sharpe", 0),
                    "hit_rate": edge.get("hit_rate", 0),
                    "edge_class": edge.get("edge_classification", "UNKNOWN"),
                })

        return sorted(rankings, key=lambda x: x["ic"], reverse=True)


# =============================================================================
# SECOND-ORDER EFFECT TRACKER
# =============================================================================

class SecondOrderEffectTracker:
    """Track cascading/second-order effects in markets.

    First-order: "AAPL is down"
    Second-order: "AAPL suppliers are down because AAPL is down"
    Third-order: "Logistics stocks are down because AAPL suppliers are down"

    USAGE:
        tracker = SecondOrderEffectTracker()
        tracker.record_cascade(
            trigger_symbol="AAPL",
            trigger_event="earnings_miss",
            affected_symbols=["AVGO", "TSM", "HON"],
            lags_hours=[1, 2, 4],
            impacts=[0.02, 0.015, 0.01]
        )
        prediction = tracker.predict_cascade("NVDA", "guidance_cut")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/second_order")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # trigger_symbol -> event_type -> list of cascades
        self.cascades: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "cascades.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for symbol, events in data.items():
                    for event, cascades in events.items():
                        self.cascades[symbol][event] = cascades
            except Exception as e:
                logger.warning(f"Could not load cascade history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "cascades.json"
        with open(history_file, "w") as f:
            json.dump({
                symbol: dict(events)
                for symbol, events in self.cascades.items()
            }, f, indent=2)

    def record_cascade(
        self,
        trigger_symbol: str,
        trigger_event: str,
        trigger_return: float,
        affected_symbols: List[str],
        lags_hours: List[float],
        impacts: List[float],
    ):
        """Record a cascade event."""
        cascade = {
            "timestamp": datetime.now().isoformat(),
            "trigger_return": trigger_return,
            "effects": [
                {
                    "symbol": s,
                    "lag_hours": l,
                    "impact": i,
                }
                for s, l, i in zip(affected_symbols, lags_hours, impacts)
            ],
        }

        self.cascades[trigger_symbol][trigger_event].append(cascade)
        self._save_history()

    def predict_cascade(
        self,
        trigger_symbol: str,
        trigger_event: str,
        trigger_magnitude: float = 0.05,
    ) -> Dict[str, Any]:
        """Predict likely cascade effects from a trigger event.
        """
        historical = self.cascades.get(trigger_symbol, {}).get(trigger_event, [])

        if len(historical) < 3:
            return {"status": "Insufficient cascade history"}

        # Aggregate effects
        symbol_effects: Dict[str, List[Dict]] = defaultdict(list)

        for cascade in historical:
            scale = trigger_magnitude / (cascade.get("trigger_return", 0.05) + 1e-10)

            for effect in cascade.get("effects", []):
                symbol_effects[effect["symbol"]].append({
                    "scaled_impact": effect["impact"] * scale,
                    "lag": effect["lag_hours"],
                })

        predictions = []
        for symbol, effects in symbol_effects.items():
            avg_impact = np.mean([e["scaled_impact"] for e in effects])
            avg_lag = np.mean([e["lag"] for e in effects])
            confidence = min(1.0, len(effects) / 10)  # More history = more confidence

            predictions.append({
                "symbol": symbol,
                "expected_impact": avg_impact,
                "expected_lag_hours": avg_lag,
                "confidence": confidence,
                "historical_occurrences": len(effects),
            })

        return {
            "trigger": f"{trigger_symbol} {trigger_event}",
            "predictions": sorted(predictions, key=lambda x: abs(x["expected_impact"]), reverse=True),
            "tradeable_opportunities": [
                p for p in predictions
                if p["confidence"] > 0.5 and abs(p["expected_impact"]) > 0.01
            ],
        }


# =============================================================================
# CONVICTION SCALING ENGINE
# =============================================================================

class ConvictionScalingEngine:
    """Dynamically scale position sizes based on conviction quality.

    Not just "how confident" but "how RELIABLY confident."
    An agent with 80% confidence that's right 80% of the time
    should size bigger than one that's right 60% of the time.

    USAGE:
        engine = ConvictionScalingEngine()
        engine.record_conviction_outcome(
            agent_name="MOMENTUM",
            stated_confidence=0.8,
            actual_outcome=True
        )
        scale = engine.get_size_scalar("MOMENTUM", stated_confidence=0.8)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/conviction")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # agent_name -> confidence_bucket -> list of outcomes
        self.conviction_history: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "conviction.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for agent, buckets in data.items():
                    for bucket, outcomes in buckets.items():
                        self.conviction_history[agent][bucket] = outcomes
            except Exception as e:
                logger.warning(f"Could not load conviction history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "conviction.json"
        with open(history_file, "w") as f:
            json.dump({
                agent: dict(buckets)
                for agent, buckets in self.conviction_history.items()
            }, f, indent=2)

    def _confidence_bucket(self, confidence: float) -> str:
        """Convert confidence to bucket (0.5-0.6, 0.6-0.7, etc.)"""
        bucket = int(confidence * 10) / 10
        return f"{bucket:.1f}"

    def record_conviction_outcome(
        self,
        agent_name: str,
        stated_confidence: float,
        actual_outcome: bool,
    ):
        """Record conviction vs actual outcome."""
        bucket = self._confidence_bucket(stated_confidence)
        self.conviction_history[agent_name][bucket].append(actual_outcome)
        self._save_history()

    def get_calibration_matrix(self, agent_name: str) -> Dict[str, Any]:
        """Get calibration matrix showing stated vs actual accuracy."""
        agent_history = self.conviction_history.get(agent_name, {})

        if not agent_history:
            return {"status": "No conviction history"}

        matrix = {}
        for bucket, outcomes in agent_history.items():
            if len(outcomes) >= 5:
                stated = float(bucket)
                actual = np.mean(outcomes)
                matrix[bucket] = {
                    "stated_confidence": stated,
                    "actual_accuracy": actual,
                    "calibration_error": abs(stated - actual),
                    "n_samples": len(outcomes),
                    "overconfident": stated > actual,
                }

        return {
            "matrix": matrix,
            "avg_calibration_error": np.mean([m["calibration_error"] for m in matrix.values()]) if matrix else 0,
            "consistently_overconfident": np.mean([m["overconfident"] for m in matrix.values()]) > 0.5 if matrix else False,
        }

    def get_size_scalar(self, agent_name: str, stated_confidence: float) -> float:
        """Get position size scalar based on conviction reliability.

        Returns 0.5-2.0 where:
        - 0.5 = This confidence level is unreliable, halve size
        - 1.0 = Confidence is accurate, normal size
        - 2.0 = Confidence is actually UNDER-stated, double size
        """
        bucket = self._confidence_bucket(stated_confidence)
        agent_history = self.conviction_history.get(agent_name, {})
        outcomes = agent_history.get(bucket, [])

        if len(outcomes) < 10:
            return 1.0  # Neutral if insufficient data

        actual_accuracy = np.mean(outcomes)

        # Scale based on relationship between stated and actual
        if actual_accuracy >= stated_confidence:
            # Underconfident - scale up
            scalar = 1.0 + (actual_accuracy - stated_confidence)
        else:
            # Overconfident - scale down
            scalar = 1.0 - (stated_confidence - actual_accuracy) * 2  # Penalize overconfidence more

        return max(0.5, min(2.0, scalar))


# =============================================================================
# MARKET MICROSTRUCTURE LEARNER
# =============================================================================

class MarketMicrostructureLearner:
    """Learn optimal execution from market microstructure.

    - Best time of day to trade
    - Best day of week
    - Optimal order types
    - When to be patient vs aggressive

    USAGE:
        learner = MarketMicrostructureLearner()
        learner.record_execution(
            symbol="AAPL",
            time_of_day=14.5,  # 2:30 PM
            day_of_week=2,     # Wednesday
            order_type="limit",
            slippage=0.0003,
            fill_rate=0.95
        )
        optimal = learner.get_optimal_execution_params("AAPL")
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/microstructure")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # symbol -> list of executions
        self.executions: Dict[str, List[Dict]] = defaultdict(list)
        self._load_history()

    def _load_history(self):
        history_file = self.storage_path / "executions.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.executions = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Could not load execution history: {e}")

    def _save_history(self):
        history_file = self.storage_path / "executions.json"
        with open(history_file, "w") as f:
            json.dump(dict(self.executions), f, indent=2)

    def record_execution(
        self,
        symbol: str,
        time_of_day: float,  # Hours from midnight (e.g., 14.5 = 2:30 PM)
        day_of_week: int,    # 0 = Monday, 4 = Friday
        order_type: str,     # "market", "limit", "stop"
        slippage: float,     # Actual vs expected price
        fill_rate: float,    # % of order filled
        urgency: str = "normal",  # "low", "normal", "high"
        spread_at_time: float = 0,
    ):
        """Record an execution for learning."""
        self.executions[symbol].append({
            "timestamp": datetime.now().isoformat(),
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "order_type": order_type,
            "slippage": slippage,
            "fill_rate": fill_rate,
            "urgency": urgency,
            "spread": spread_at_time,
        })
        self._save_history()

    def get_optimal_execution_params(self, symbol: str) -> Dict[str, Any]:
        """Get optimal execution parameters for a symbol."""
        history = self.executions.get(symbol, [])

        if len(history) < 20:
            return {
                "status": "Insufficient history",
                "default": {
                    "best_hours": [10, 11, 14, 15],  # Avoid open/close
                    "best_days": [1, 2, 3],          # Tue-Thu
                    "order_type": "limit",
                },
            }

        df = pd.DataFrame(history)

        # Best time of day (lowest slippage)
        hourly_slippage = df.groupby(df["time_of_day"].astype(int))["slippage"].mean()
        best_hours = hourly_slippage.nsmallest(3).index.tolist()
        worst_hours = hourly_slippage.nlargest(2).index.tolist()

        # Best day of week
        daily_slippage = df.groupby("day_of_week")["slippage"].mean()
        best_days = daily_slippage.nsmallest(3).index.tolist()

        # Best order type
        order_performance = df.groupby("order_type").agg({
            "slippage": "mean",
            "fill_rate": "mean",
        })

        # Score = low slippage + high fill rate
        order_performance["score"] = order_performance["fill_rate"] - order_performance["slippage"] * 100
        best_order_type = order_performance["score"].idxmax()

        return {
            "symbol": symbol,
            "n_executions": len(history),
            "best_hours": best_hours,
            "worst_hours": worst_hours,
            "best_days": best_days,
            "best_order_type": best_order_type,
            "avg_slippage": df["slippage"].mean(),
            "avg_fill_rate": df["fill_rate"].mean(),
            "recommendation": f"Trade {symbol} during hours {best_hours} using {best_order_type} orders",
        }


# =============================================================================
# ULTIMATE LEARNING ORCHESTRATOR
# =============================================================================

class UltimateLearningOrchestrator:
    """THE MASTER SYSTEM - Orchestrates ALL learning components.

    This is the brain that coordinates:
    - Part II: Core Learning Functions
    - Part III: Advanced Self-Improvement
    - Part IV: Edge Intelligence

    USAGE:
        orchestrator = UltimateLearningOrchestrator()
        report = orchestrator.get_complete_agent_analysis("MOMENTUM")
        orchestrator.run_daily_learning_cycle()
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Part II - Core
        self.calibration = PredictionCalibrationTracker(self.storage_path / "calibration")
        self.signal_analyzer = SignalHalfLifeAnalyzer(self.storage_path / "signals")
        self.regime_tagger = RegimeContextTagger(self.storage_path / "regimes")
        self.correlation_monitor = CrossAgentCorrelationMonitor(self.storage_path / "correlations")
        self.contrarian_detector = ContrarianTriggerDetector(self.storage_path / "contrarian")
        self.attribution = AttributionFeedbackLoop(self.storage_path / "attribution")
        self.thesis_tracker = ThesisHalfLifeTracker(self.storage_path / "theses")
        self.journal = PredictionJournalQA(self.storage_path / "journal")

        # Part III - Advanced Self-Improvement
        self.bias_detector = CognitiveBiasDetector(self.storage_path / "biases")
        self.mistake_analyzer = MistakeTaxonomyAnalyzer(self.storage_path / "mistakes")
        self.antifragility_scorer = AntifragilityScorer(self.storage_path / "antifragility")
        self.pattern_monitor = PatternFatigueMonitor(self.storage_path / "patterns")
        self.opportunity_tracker = OpportunityCostTracker(self.storage_path / "opportunities")
        self.meta_optimizer = MetaLearningOptimizer(self.storage_path / "meta")
        self.skill_matrix = SkillAttributionMatrix(self.storage_path / "skills")

        # Part IV - Edge Intelligence
        self.adversarial_tester = AdversarialRobustnessTester(self.storage_path / "adversarial")
        self.info_edge = InformationEdgeQuantifier(self.storage_path / "info_edge")
        self.cascade_tracker = SecondOrderEffectTracker(self.storage_path / "cascades")
        self.conviction_engine = ConvictionScalingEngine(self.storage_path / "conviction")
        self.microstructure = MarketMicrostructureLearner(self.storage_path / "microstructure")

        logger.info("Ultimate Learning Orchestrator initialized with ALL components")

    def get_complete_agent_analysis(self, agent_name: str) -> Dict[str, Any]:
        """COMPREHENSIVE analysis across ALL learning systems.

        This is the ultimate report on an agent's learning health.
        """
        return {
            "agent_name": agent_name,
            "generated_at": datetime.now().isoformat(),

            # === PART II: CORE LEARNING ===
            "calibration": self.calibration.get_calibration_score(agent_name),
            "thesis_accuracy": self.thesis_tracker.get_thesis_accuracy_by_type(),
            "journal_accuracy": self.journal.get_agent_accuracy(agent_name),
            "signal_attribution": self.attribution.get_attribution_report(),

            # === PART III: SELF-IMPROVEMENT ===
            "cognitive_biases": self.bias_detector.get_bias_report(agent_name),
            "weakness_profile": self.mistake_analyzer.get_weakness_profile(agent_name),
            "antifragility": self.antifragility_scorer.get_antifragility_score(agent_name),
            "opportunity_cost": self.opportunity_tracker.get_opportunity_cost_report(agent_name),
            "optimal_learning": self.meta_optimizer.get_optimal_learning_config(agent_name),
            "skill_breakdown": self.skill_matrix.get_skill_breakdown(agent_name),

            # === PART IV: EDGE INTELLIGENCE ===
            "adversarial_robustness": self.adversarial_tester.get_robustness_score(agent_name),
            "information_ranking": self.info_edge.get_information_ranking(),
            "conviction_calibration": self.conviction_engine.get_calibration_matrix(agent_name),

            # === PATTERN HEALTH ===
            "pattern_statuses": self.pattern_monitor.get_all_pattern_statuses(),
        }

    def get_trading_decision_enhancement(
        self,
        agent_name: str,
        symbol: str,
        signal_direction: int,
        stated_confidence: float,
        signal_type: SignalType,
        current_regime: MarketRegime,
        agreeing_agents: List[str],
    ) -> Dict[str, Any]:
        """Enhance a trading decision with ALL learning insights.

        Takes a raw signal and enhances it with everything we've learned.
        """
        # Base signal freshness
        half_life = self.signal_analyzer.get_half_life(signal_type)

        # Regime appropriateness
        should_trade, regime_reason = self.regime_tagger.should_trade_signal(
            signal_type, current_regime,
        )

        # P&L attribution weight
        pnl_weight = self.attribution.get_signal_weight(signal_type)

        # Multi-agent agreement
        combo_weight = self.correlation_monitor.get_combination_weight(agreeing_agents) if len(agreeing_agents) >= 2 else 1.0

        # Contrarian check
        agreement_level = len(agreeing_agents) / 10
        contrarian = self.contrarian_detector.check_contrarian_trigger(
            agreement_level, current_regime, signal_direction,
        )

        # Bias correction
        recency_correction = self.bias_detector.get_bias_correction_factor(
            agent_name, CognitiveBias.RECENCY_BIAS,
        )

        # Conviction calibration
        conviction_scalar = self.conviction_engine.get_size_scalar(agent_name, stated_confidence)

        # Execution optimization
        execution = self.microstructure.get_optimal_execution_params(symbol)

        # Calculate final enhanced size
        base_multiplier = pnl_weight * combo_weight * recency_correction * conviction_scalar

        if contrarian.get("recommendation") == "go_contrarian":
            final_direction = -signal_direction
            final_multiplier = base_multiplier * 0.5  # Smaller size for contrarian
        else:
            final_direction = signal_direction
            final_multiplier = base_multiplier

        if not should_trade:
            final_multiplier *= 0.25  # Heavily reduce if regime is wrong

        return {
            "original_signal": {
                "direction": signal_direction,
                "confidence": stated_confidence,
            },
            "enhanced_signal": {
                "direction": final_direction,
                "size_multiplier": min(2.0, final_multiplier),
                "should_trade": should_trade and final_multiplier > 0.3,
            },
            "adjustments": {
                "pnl_weight": pnl_weight,
                "combo_weight": combo_weight,
                "bias_correction": recency_correction,
                "conviction_scalar": conviction_scalar,
                "regime_appropriate": should_trade,
                "contrarian_trigger": contrarian.get("is_contrarian_opportunity", False),
            },
            "execution_guidance": {
                "best_hours": execution.get("best_hours", []),
                "best_order_type": execution.get("best_order_type", "limit"),
            },
            "regime_context": regime_reason,
            "signal_half_life_days": half_life,
        }

    def run_daily_learning_cycle(self, agents: List[str]) -> Dict[str, Any]:
        """Run the daily learning cycle for all agents.

        This should be called at end of each trading day.
        """
        results = {
            "cycle_timestamp": datetime.now().isoformat(),
            "agents_processed": [],
            "patterns_updated": 0,
            "biases_detected": 0,
            "improvement_opportunities": [],
        }

        for agent_name in agents:
            # Get comprehensive analysis
            analysis = self.get_complete_agent_analysis(agent_name)

            # Count issues
            biases = analysis.get("cognitive_biases", {})
            if biases.get("total_incidents", 0) > 0:
                results["biases_detected"] += biases["total_incidents"]

            # Check antifragility
            antifragility = analysis.get("antifragility", {})
            if antifragility.get("classification") == "FRAGILE":
                results["improvement_opportunities"].append({
                    "agent": agent_name,
                    "issue": "FRAGILE under stress",
                    "priority": "HIGH",
                })

            # Check skill attribution
            skills = analysis.get("skill_breakdown", {})
            if skills.get("skill_luck_ratio", 1) < 1:
                results["improvement_opportunities"].append({
                    "agent": agent_name,
                    "issue": "Returns may be luck-driven",
                    "priority": "MEDIUM",
                })

            results["agents_processed"].append(agent_name)

        # Update pattern statuses
        pattern_statuses = self.pattern_monitor.get_all_pattern_statuses()
        decaying_patterns = [
            p for p, s in pattern_statuses.items()
            if s.get("status") in ["MATURITY", "DECAY"]
        ]
        results["patterns_updated"] = len(pattern_statuses)
        results["decaying_patterns"] = decaying_patterns

        logger.info(f"Daily learning cycle complete: {len(agents)} agents, {results['biases_detected']} biases detected")

        return results

    def get_system_health_dashboard(self) -> Dict[str, Any]:
        """Get overall system health across all learning components.
        """
        return {
            "generated_at": datetime.now().isoformat(),

            # Data health
            "calibration_records": len(self.calibration.predictions),
            "thesis_records": len(self.thesis_tracker.theses),
            "journal_entries": len(self.journal.entries),
            "mistake_records": len(self.mistake_analyzer.mistakes),

            # Pattern health
            "patterns_tracked": len(self.pattern_monitor.pattern_history),
            "pattern_statuses": self.pattern_monitor.get_all_pattern_statuses(),

            # Information edge
            "info_sources_ranked": self.info_edge.get_information_ranking(),

            # Recommendations
            "system_recommendations": [
                "Run daily learning cycle after market close",
                "Review decaying patterns monthly",
                "Audit cognitive biases weekly",
                "Update adversarial tests quarterly",
            ],
        }
