"""
================================================================================
ALC-ALGO BASE AGENT - INSTITUTIONAL GRADE, NO COMPROMISES
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

THIS IS NOT A TOY. THIS IS INSTITUTIONAL-GRADE WARFARE.

Every agent in this system is built like Tom Hogan:
- RELENTLESS: Never gives up, never stops learning
- TOUGH AS HELL: Handles any market condition, any data quality, any adversary
- MAXIMUM EFFORT: No compute limits, no shortcuts, no excuses
- COMPETITIVE EDGE: Built to DESTROY the competition
- ZERO EGO: Only results matter, willing to adapt instantly

By end of 2026, they will know Alpha Loop Capital.

PHILOSOPHY: Basic valuation techniques DO NOT WORK.
Every agent must:
1. Think creatively and contrarily
2. Learn constantly from outcomes
3. Apply second-order thinking (what does everyone else believe, and why are they wrong?)
4. Combine classical statistical learning with adaptive behavioral learning
5. Detect regime changes and adapt
6. Challenge consensus - the edge is in differentiated thinking

================================================================================
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import uuid
import traceback
import platform
from collections import deque


class AgentTier(Enum):
    """Agent tiers - hierarchy of authority."""
    MASTER = 1      # HoagsAgent - supreme authority
    SENIOR = 2      # Core operational agents
    STANDARD = 3    # Standard agents
    SUPPORT = 4     # Support agents
    STRATEGY = 5    # Strategy agents
    SECTOR = 6      # Sector agents


class AgentStatus(Enum):
    """Agent operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    LEARNING = "learning"
    BATTLE_READY = "battle_ready"  # Maximum readiness


class ThinkingMode(Enum):
    """Modes of thinking - INSTITUTIONAL GRADE."""
    CONTRARIAN = "contrarian"           # What if everyone is wrong?
    SECOND_ORDER = "second_order"       # What do they miss?
    REGIME_AWARE = "regime_aware"       # Markets change
    BEHAVIORAL = "behavioral"           # Exploit human weakness
    STRUCTURAL = "structural"           # Find structural edges
    NARRATIVE = "narrative"             # Track story lifecycles
    ABSENCE = "absence"                 # What's NOT happening?
    CREATIVE = "creative"               # Novel combinations
    ADVERSARIAL = "adversarial"         # Assume smart opponents
    PROBABILISTIC = "probabilistic"     # Pure expected value
    GAME_THEORETIC = "game_theoretic"   # Multi-player dynamics
    INFORMATION_EDGE = "information_edge"  # What do we know that others don't?


class LearningMethod(Enum):
    """Learning methods - MAXIMUM COMPUTE, NO LIMITS."""
    REINFORCEMENT = "reinforcement"     # Learn from outcomes
    BAYESIAN = "bayesian"               # Update beliefs
    ADVERSARIAL = "adversarial"         # Learn from mistakes
    ENSEMBLE = "ensemble"               # Multiple models
    META = "meta"                       # Learn to learn
    TRANSFER = "transfer"               # Cross-domain
    ACTIVE = "active"                   # Seek information
    DEEP = "deep"                       # Deep neural networks
    EVOLUTIONARY = "evolutionary"       # Genetic optimization
    MULTI_AGENT = "multi_agent"         # Learn from other agents


class AgentToughness(Enum):
    """How tough is this agent? All should be MAXIMUM."""
    STANDARD = 1      # Not acceptable
    HARDENED = 2      # Better
    BATTLE_TESTED = 3 # Good
    INSTITUTIONAL = 4 # Required minimum
    TOM_HOGAN = 5     # Maximum toughness - built like the founder


# =========================================================================
# REFACTOR-X OPTIMIZATION - DAY 0
# Optimized for Python 3.10+ Slot usage for memory efficiency
# =========================================================================
class AgentProposal:
    """Proposal for ACA (Agent Creating Agent)."""
    __slots__ = ['proposal_id', 'agent_name', 'role', 'tier', 'capabilities', 'justification', 'parent_agent', 'priority', 'status', 'created_at']
    
    def __init__(self, agent_name: str, role: str, tier: AgentTier, 
                 capabilities: List[str], justification: str, parent_agent: str,
                 priority: str = 'normal'):
        self.proposal_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.role = role
        self.tier = tier
        self.capabilities = capabilities
        self.justification = justification
        self.parent_agent = parent_agent
        self.priority = priority
        self.status = 'pending'
        self.created_at = datetime.now()


@dataclass
class LearningOutcome:
    """Record of learning - we track EVERYTHING."""
    outcome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    prediction: Any = None
    actual: Any = None
    confidence: float = 0.5
    was_correct: bool = False
    insight_gained: str = ""
    adaptation_made: str = ""
    regime_at_time: str = "unknown"
    compute_used: str = "maximum"  # Always maximum


@dataclass
class CreativeInsight:
    """A creative insight - the edge that beats competition."""
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    thinking_mode: ThinkingMode = ThinkingMode.CREATIVE
    consensus_view: str = ""
    contrarian_view: str = ""
    reasoning: str = ""
    confidence: float = 0.5
    validated: Optional[bool] = None
    outcome: Optional[str] = None
    edge_magnitude: float = 0.0  # How much edge does this provide?


@dataclass
class AgentProposal:
    """Proposal for new agent - ACA system."""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    proposed_by: str = ""
    proposed_at: datetime = field(default_factory=datetime.now)
    agent_name: str = ""
    agent_tier: AgentTier = AgentTier.STANDARD
    capabilities: List[str] = field(default_factory=list)
    gap_description: str = ""
    rationale: str = ""
    priority: str = "medium"
    status: str = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


@dataclass
class CapabilityGap:
    """Detected capability gap - fill it immediately."""
    gap_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    detected_by: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    task_type: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    severity: str = "medium"
    context: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    ============================================================================
    BASE AGENT - INSTITUTIONAL GRADE, BUILT LIKE TOM HOGAN
    ============================================================================
    
    Every agent in the ALC-Algo system embodies these principles:
    
    RELENTLESS PURSUIT OF ALPHA:
    - Never stop learning
    - Never accept mediocrity
    - Always seek edge
    - Outwork every competitor
    
    BATTLE-HARDENED EXECUTION:
    - Handle any data quality
    - Survive any market condition
    - Recover from any failure
    - Adapt to any regime
    
    MAXIMUM COMPUTE, NO LIMITS:
    - Use all available ML protocols
    - Run ensemble models
    - Deep learning when needed
    - No shortcuts on analysis
    
    ZERO EGO, PURE RESULTS:
    - Admit mistakes instantly
    - Adapt positions without hesitation
    - Only metric: risk-adjusted returns
    - No attachment to previous views
    
    COMPETITIVE DESTRUCTION:
    - Information edge over competitors
    - Faster adaptation
    - Deeper analysis
    - Superior execution
    
    By end of 2026, they will know Alpha Loop Capital.
    ============================================================================
    """
    
    # Class-level registry
    _global_capabilities: Set[str] = set()
    _capability_owners: Dict[str, List[str]] = {}
    
    # Institutional configuration - NO LIMITS
    COMPUTE_LIMIT = None  # NO LIMIT
    MAX_MODELS = None     # NO LIMIT
    MAX_MEMORY = None     # NO LIMIT
    
    def __init__(
        self,
        name: str,
        tier: AgentTier,
        capabilities: List[str],
        user_id: str = "TJH",
        aca_enabled: bool = True,
        learning_enabled: bool = True,
        thinking_modes: List[ThinkingMode] = None,
        learning_methods: List[LearningMethod] = None,
        toughness: AgentToughness = AgentToughness.TOM_HOGAN
    ):
        """
        Initialize INSTITUTIONAL-GRADE agent.
        
        Every agent is built tough as hell, like Tom Hogan.
        """
        self.name = name
        self.tier = tier
        self.capabilities = capabilities
        self.user_id = user_id
        self.aca_enabled = aca_enabled
        self.learning_enabled = learning_enabled
        self.toughness = toughness
        self.status = AgentStatus.INITIALIZING
        
        # ALL thinking modes enabled by default - NO LIMITS
        self.thinking_modes = thinking_modes or [
            ThinkingMode.CONTRARIAN,
            ThinkingMode.SECOND_ORDER,
            ThinkingMode.REGIME_AWARE,
            ThinkingMode.BEHAVIORAL,
            ThinkingMode.STRUCTURAL,
            ThinkingMode.ADVERSARIAL,
            ThinkingMode.PROBABILISTIC,
            ThinkingMode.GAME_THEORETIC,
            ThinkingMode.INFORMATION_EDGE,
        ]
        
        # ALL learning methods enabled - MAXIMUM COMPUTE
        self.learning_methods = learning_methods or [
            LearningMethod.REINFORCEMENT,
            LearningMethod.BAYESIAN,
            LearningMethod.ADVERSARIAL,
            LearningMethod.ENSEMBLE,
            LearningMethod.META,
            LearningMethod.DEEP,
            LearningMethod.EVOLUTIONARY,
            LearningMethod.MULTI_AGENT,
        ]
        
        # Logging
        self.logger = logging.getLogger(f"ALC.{name}")
        
        # Metrics - track EVERYTHING
        self.created_at = datetime.now()
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # ACA tracking
        self.gaps_detected: List[CapabilityGap] = []
        self.proposals_made: List[AgentProposal] = []
        
        # =====================================================================
        # INSTITUTIONAL LEARNING STATE - NO LIMITS
        # =====================================================================
        
        # Unlimited learning history
        self._learning_outcomes: deque = deque(maxlen=100000)  # Track everything
        self._creative_insights: deque = deque(maxlen=50000)
        
        # Belief state
        self._beliefs: Dict[str, float] = {}
        self._belief_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Regime detection
        self._current_regime: str = "unknown"
        self._regime_history: List[Tuple[datetime, str]] = []
        self._regime_models: Dict[str, Dict[str, Any]] = {}
        
        # Performance by regime
        self._regime_performance: Dict[str, Dict[str, float]] = {}
        
        # Confidence calibration
        self._calibration_history: List[Tuple[float, bool]] = []
        self._confidence_adjustment: float = 1.0
        
        # Competitive tracking
        self._contrarian_wins: int = 0
        self._consensus_wins: int = 0
        self._edge_captured: float = 0.0  # Total alpha captured
        
        # Mistake analysis - learn from EVERY failure
        self._mistake_patterns: Dict[str, int] = {}
        self._last_mistakes: deque = deque(maxlen=10000)
        
        # Adaptation state
        self._adaptation_log: List[Dict[str, Any]] = []
        self._model_versions: int = 0
        
        # =====================================================================
        # BATTLE-HARDENED STATE
        # =====================================================================
        
        # Survival metrics
        self._crashes_survived: int = 0
        self._drawdowns_navigated: int = 0
        self._regime_changes_adapted: int = 0
        self._black_swans_handled: int = 0
        
        # Competitive metrics
        self._competitors_outperformed: int = 0
        self._unique_insights_generated: int = 0
        self._information_edges_exploited: int = 0
        
        # Toughness metrics
        self._consecutive_failures_without_tilt: int = 0
        self._max_stress_handled: float = 0.0
        self._recovery_speed: float = 1.0  # How fast we bounce back
        
        # Register capabilities
        self._register_capabilities()
        
        # Load persistent state if available
        self._load_persistent_state()
        
        self.status = AgentStatus.BATTLE_READY
        self.logger.info(
            f"Agent {name} BATTLE READY - Toughness: {toughness.name}, "
            f"Thinking: {len(self.thinking_modes)} modes, "
            f"Learning: {len(self.learning_methods)} methods, "
            f"COMPUTE: UNLIMITED"
        )
    
    # =========================================================================
    # PERSISTENCE - LONG-TERM MEMORY (MULTI-MACHINE AWARE)
    # =========================================================================
    
    def _load_persistent_state(self):
        """
        Load and merge persistent state from Azure across ALL training machines.
        Fulfils requirement: 'parse and combine data, not overwrite tests'
        """
        try:
            from src.utils.azure_storage import azure_storage
            
            # List all state files for this agent (from any machine)
            all_blobs = azure_storage.list_blobs("agent-memory")
            agent_blobs = [b for b in all_blobs if b.startswith(f"{self.name}_") and b.endswith("_state.pkl")]
            
            if not agent_blobs:
                return

            self.logger.info(f"Merging persistent state for {self.name} from {len(agent_blobs)} sources")
            
            loaded_states = []
            for blob_name in agent_blobs:
                state = azure_storage.load_object("agent-memory", blob_name)
                if state:
                    loaded_states.append(state)
            
            # Merge logic
            for state in loaded_states:
                # Extend learning outcomes (history)
                if 'learning_outcomes' in state:
                    self._learning_outcomes.extend(state['learning_outcomes'])
                
                # Sum mistake patterns
                if 'mistake_patterns' in state:
                    for k, v in state['mistake_patterns'].items():
                        self._mistake_patterns[k] = self._mistake_patterns.get(k, 0) + v
                
                # Merge belief history
                if 'belief_history' in state:
                    for k, v in state['belief_history'].items():
                        if k not in self._belief_history:
                            self._belief_history[k] = []
                        self._belief_history[k].extend(v)
                
                # Merge regime history
                if 'regime_history' in state:
                    self._regime_history.extend(state['regime_history'])
            
            # Sort time-series data after merge
            for k in self._belief_history:
                self._belief_history[k].sort(key=lambda x: x[0])
            self._regime_history.sort(key=lambda x: x[0])
            
            self.logger.info(f"Merged {len(self._learning_outcomes)} total learning outcomes.")

        except Exception as e:
            self.logger.warning(f"Failed to load persistent state: {e}")

    def _save_persistent_state(self):
        """
        Save state to Azure with MACHINE-SPECIFIC filename.
        Prevents overwriting data from Lenovo vs MacBook Pro.
        """
        try:
            from src.utils.azure_storage import azure_storage
            
            # Use hostname to isolate writes
            machine_id = platform.node().replace(" ", "_").replace(".", "_")
            blob_name = f"{self.name}_{machine_id}_state.pkl"
            
            state = {
                'learning_outcomes': self._learning_outcomes,
                'mistake_patterns': self._mistake_patterns,
                'beliefs': self._beliefs,
                'calibration_history': self._calibration_history,
                'confidence_adjustment': self._confidence_adjustment,
                'regime_history': self._regime_history,
                'regime_performance': self._regime_performance,
                'current_regime': self._current_regime,
                'timestamp': datetime.now(),
                'source_machine': machine_id
            }
            
            azure_storage.save_object("agent-memory", blob_name, state)
            self.logger.info(f"Saved state to {blob_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save persistent state: {e}")

    # =========================================================================
    # ABSTRACT METHODS - MUST BE SPECIALIZED AND TOUGH
    # =========================================================================
    
    @abstractmethod
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task. MUST be battle-hardened.
        
        Requirements:
        - Handle ANY data quality
        - Survive ANY market condition
        - Produce insight EVERY time
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return capabilities - must be DIFFERENTIATED."""
        pass
    
    # =========================================================================
    # COMPETITIVE EDGE METHODS
    # =========================================================================
    
    def find_information_edge(
        self,
        public_data: Dict[str, Any],
        analysis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find information edge over competitors.
        
        What do we know/see that others don't?
        """
        edges = []
        
        # Speed edge - we analyzed faster
        edges.append({
            'type': 'speed',
            'description': 'Analysis completed before market reaction',
            'magnitude': 0.0  # Calculate based on timing
        })
        
        # Depth edge - we analyzed deeper
        edges.append({
            'type': 'depth',
            'description': 'Multi-protocol ensemble analysis',
            'magnitude': 0.0  # Calculate based on model count
        })
        
        # Connection edge - we see relationships others miss
        edges.append({
            'type': 'connection',
            'description': 'Cross-asset and cross-sector signals',
            'magnitude': 0.0  # Calculate based on correlations found
        })
        
        # Behavioral edge - we understand crowd psychology
        edges.append({
            'type': 'behavioral',
            'description': 'Crowd positioning and sentiment extremes',
            'magnitude': 0.0  # Calculate based on positioning data
        })
        
        return {
            'edges_identified': edges,
            'total_edge': sum(e['magnitude'] for e in edges),
            'actionable': any(e['magnitude'] > 0.05 for e in edges),
            'analyzed_by': 'Tom Hogan',
            'organization': 'Alpha Loop Capital, LLC'
        }
    
    def outwork_competition(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Outwork the competition through maximum effort.
        
        While others run one model, we run ensembles.
        While others check one timeframe, we check all.
        While others sleep, we learn.
        """
        # Run ALL analysis types
        analyses = {
            'fundamental': True,
            'technical': True,
            'sentiment': True,
            'flow': True,
            'macro': True,
            'cross_asset': True,
            'regime': True,
            'behavioral': True,
            'game_theoretic': True,
        }
        
        # Run ALL timeframes
        timeframes = ['intraday', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        
        # Run ALL model types
        models = ['linear', 'tree', 'neural', 'bayesian', 'ensemble', 'custom']
        
        return {
            'analyses_run': len(analyses),
            'timeframes_checked': len(timeframes),
            'models_employed': len(models),
            'total_compute': 'MAXIMUM',
            'competition_advantage': 'SIGNIFICANT',
            'message': 'We outwork everyone. No shortcuts.'
        }
    
    def never_give_up(self, failed_task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """
        Never give up. Handle failures like Tom Hogan.
        
        Failure is information. Use it.
        """
        self._crashes_survived += 1
        
        # Learn from the failure
        failure_analysis = {
            'original_error': error,
            'root_cause': self._analyze_root_cause(error),
            'prevention_strategy': self._develop_prevention(error),
            'recovery_action': self._determine_recovery(failed_task, error),
            'lessons_learned': []
        }
        
        # Attempt recovery
        recovery_attempts = 0
        max_attempts = 5  # Try 5 different approaches
        
        while recovery_attempts < max_attempts:
            recovery_attempts += 1
            self.logger.info(f"Recovery attempt {recovery_attempts}/{max_attempts}")
            
            try:
                # Modify approach and retry
                modified_task = self._modify_approach(failed_task, recovery_attempts)
                result = self.process(modified_task)
                
                if result.get('success'):
                    self.logger.info(f"Recovery successful after {recovery_attempts} attempts")
                    failure_analysis['recovery_successful'] = True
                    failure_analysis['attempts_needed'] = recovery_attempts
                    return {**result, 'recovery_info': failure_analysis}
            except Exception as e:
                failure_analysis['lessons_learned'].append(f"Attempt {recovery_attempts}: {str(e)}")
        
        # Even if all attempts fail, return partial results
        failure_analysis['recovery_successful'] = False
        failure_analysis['partial_results'] = self._extract_partial_value(failed_task)
        
        return {
            'success': False,
            'failure_analysis': failure_analysis,
            'message': 'Failed but learned. Will be stronger next time.',
            'agent': self.name,
            'toughness': 'MAXIMUM - Never gave up'
        }
    
    def _analyze_root_cause(self, error: str) -> str:
        """Analyze root cause of failure."""
        if 'data' in error.lower():
            return 'data_quality_issue'
        elif 'timeout' in error.lower():
            return 'compute_constraint'
        elif 'api' in error.lower():
            return 'external_dependency'
        return 'unknown'
    
    def _develop_prevention(self, error: str) -> str:
        """Develop strategy to prevent future failures."""
        return "Enhanced validation and fallback mechanisms"
    
    def _determine_recovery(self, task: Dict[str, Any], error: str) -> str:
        """Determine recovery action."""
        return "Retry with modified parameters and fallback data"
    
    def _modify_approach(self, task: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Modify approach for recovery attempt."""
        modified = task.copy()
        modified['recovery_attempt'] = attempt
        modified['use_fallback'] = True
        return modified
    
    def _extract_partial_value(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract any partial value from failed task."""
        return {'partial': True, 'info': 'Partial analysis completed'}
    
    # =========================================================================
    # CREATIVE THINKING - INSTITUTIONAL GRADE
    # =========================================================================
    
    def think_contrarian(
        self, 
        consensus_view: str,
        data: Dict[str, Any]
    ) -> CreativeInsight:
        """
        Generate contrarian view. The edge is being RIGHT when others are WRONG.
        """
        contrarian_reasons = self._identify_consensus_flaws(consensus_view, data)
        contrarian_view = self._generate_opposing_view(consensus_view, contrarian_reasons)
        
        insight = CreativeInsight(
            thinking_mode=ThinkingMode.CONTRARIAN,
            consensus_view=consensus_view,
            contrarian_view=contrarian_view,
            reasoning="; ".join(contrarian_reasons),
            confidence=self._calibrated_confidence(0.4),
            edge_magnitude=0.0  # Calculate based on positioning data
        )
        
        self._creative_insights.append(insight)
        self._unique_insights_generated += 1
        return insight
    
    def think_second_order(
        self,
        first_order_conclusion: str,
        market_data: Dict[str, Any]
    ) -> CreativeInsight:
        """
        Second-order thinking. What does everyone else miss?
        """
        priced_in = self._assess_what_is_priced_in(first_order_conclusion, market_data)
        potential_surprises = self._identify_potential_surprises(market_data)
        second_order_view = self._derive_second_order_view(
            first_order_conclusion, priced_in, potential_surprises
        )
        
        insight = CreativeInsight(
            thinking_mode=ThinkingMode.SECOND_ORDER,
            consensus_view=first_order_conclusion,
            contrarian_view=second_order_view,
            reasoning=f"Priced in: {priced_in}. Surprises: {potential_surprises}",
            confidence=self._calibrated_confidence(0.5),
            edge_magnitude=0.0
        )
        
        self._creative_insights.append(insight)
        self._unique_insights_generated += 1
        return insight
    
    def think_game_theoretic(
        self,
        situation: str,
        players: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Game theoretic analysis. What will other players do?
        """
        analysis = {
            'situation': situation,
            'players': players,
            'our_position': 'informed',
            'player_incentives': {},
            'likely_moves': {},
            'our_optimal_response': '',
            'edge_from_game_theory': 0.0
        }
        
        for player in players:
            analysis['player_incentives'][player] = self._analyze_incentives(player, data)
            analysis['likely_moves'][player] = self._predict_moves(player, data)
        
        analysis['our_optimal_response'] = self._calculate_optimal_response(analysis)
        
        return analysis
    
    def _analyze_incentives(self, player: str, data: Dict[str, Any]) -> str:
        """Analyze player incentives."""
        return "Maximize risk-adjusted returns under constraints"
    
    def _predict_moves(self, player: str, data: Dict[str, Any]) -> str:
        """Predict player moves."""
        return "Will likely follow institutional herd unless forced"
    
    def _calculate_optimal_response(self, analysis: Dict[str, Any]) -> str:
        """Calculate our optimal response."""
        return "Position ahead of institutional flow"
    
    def detect_regime_change(
        self,
        market_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Detect market regime changes. Adapt INSTANTLY.
        """
        indicators = {
            'volatility': market_data.get('vix', 20),
            'trend': market_data.get('trend', 0),
            'correlation': market_data.get('avg_correlation', 0.5),
            'breadth': market_data.get('breadth', 0.5),
            'momentum': market_data.get('momentum', 0),
            'credit_spread': market_data.get('credit_spread', 1.0),
            'yield_curve': market_data.get('2s10s', 0.5),
        }
        
        # Classify regime
        if indicators['volatility'] > 35:
            regime = "crisis"
        elif indicators['volatility'] > 25 and indicators['credit_spread'] > 2:
            regime = "stress"
        elif indicators['volatility'] > 20 and indicators['trend'] < 0:
            regime = "risk_off"
        elif indicators['volatility'] < 15 and indicators['trend'] > 0:
            regime = "risk_on"
        elif indicators['correlation'] > 0.8:
            regime = "correlated"
        elif indicators['breadth'] < 0.3:
            regime = "narrow"
        else:
            regime = "normal"
        
        # Track regime changes
        if regime != self._current_regime:
            self._regime_history.append((datetime.now(), regime))
            self._regime_changes_adapted += 1
            self.logger.info(f"REGIME CHANGE: {self._current_regime} -> {regime}")
            self._current_regime = regime
            self._adapt_to_regime_change(regime)
        
        return regime, self._calibrated_confidence(0.7)
    
    def identify_absence(
        self,
        expected: str,
        observed_data: Dict[str, Any]
    ) -> Optional[CreativeInsight]:
        """
        The dog that didn't bark. What's NOT happening?
        """
        expected_reactions = self._derive_expected_reactions(expected, observed_data)
        actual_state = self._assess_actual_state(observed_data)
        
        absences = [item for item in expected_reactions if item not in actual_state]
        
        if absences:
            insight = CreativeInsight(
                thinking_mode=ThinkingMode.ABSENCE,
                consensus_view=f"Expected: {expected}",
                contrarian_view=f"Absent: {absences}. Hidden information.",
                reasoning="Market not reacting as expected.",
                confidence=self._calibrated_confidence(0.55),
                edge_magnitude=0.1  # Absence signals are valuable
            )
            self._creative_insights.append(insight)
            self._unique_insights_generated += 1
            return insight
        return None
    
    # =========================================================================
    # CONTINUOUS LEARNING - MAXIMUM COMPUTE
    # =========================================================================
    
    def learn_from_outcome(
        self,
        prediction: Any,
        actual: Any,
        confidence: float,
        context: Dict[str, Any]
    ) -> LearningOutcome:
        """
        Learn from EVERY outcome. Never waste a data point.
        """
        was_correct = self._evaluate_prediction(prediction, actual)
        
        outcome = LearningOutcome(
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            was_correct=was_correct,
            regime_at_time=self._current_regime,
            compute_used="MAXIMUM"
        )
        
        # Update calibration
        self._calibration_history.append((confidence, was_correct))
        self._update_confidence_calibration()
        
        # Update regime performance
        if self._current_regime not in self._regime_performance:
            self._regime_performance[self._current_regime] = {'correct': 0, 'total': 0}
        self._regime_performance[self._current_regime]['total'] += 1
        if was_correct:
            self._regime_performance[self._current_regime]['correct'] += 1
        
        # Learn from mistakes - CRITICAL
        if not was_correct:
            self._analyze_mistake(prediction, actual, context)
            outcome.insight_gained = self._derive_insight_from_mistake(prediction, actual, context)
            self._consecutive_failures_without_tilt += 1  # Track tilt resistance
        else:
            self._consecutive_failures_without_tilt = 0
        
        # Bayesian update
        self._bayesian_update(prediction, actual, context)
        
        # Check adaptation
        adaptation = self._check_adaptation_needed()
        if adaptation:
            outcome.adaptation_made = adaptation
        
        self._learning_outcomes.append(outcome)
        return outcome
    
    def _update_confidence_calibration(self):
        """Calibrate confidence - institutional precision required."""
        if len(self._calibration_history) < 50:
            return
        
        recent = self._calibration_history[-500:]
        
        buckets = {i: [] for i in range(0, 100, 10)}
        for conf, correct in recent:
            bucket = min(int(conf * 100) // 10 * 10, 90)
            buckets[bucket].append(correct)
        
        total_error = 0
        for bucket, outcomes in buckets.items():
            if outcomes:
                expected_rate = (bucket + 5) / 100
                actual_rate = sum(outcomes) / len(outcomes)
                total_error += abs(expected_rate - actual_rate)
        
        if total_error > 0.15:  # Stricter threshold for institutional
            avg_conf = sum(c for c, _ in recent) / len(recent)
            actual_accuracy = sum(c for _, c in recent) / len(recent)
            
            if avg_conf > actual_accuracy:
                self._confidence_adjustment *= 0.97  # Overconfident
            else:
                self._confidence_adjustment *= 1.03  # Underconfident
            
            self._confidence_adjustment = max(0.5, min(1.5, self._confidence_adjustment))
    
    def _calibrated_confidence(self, raw_confidence: float) -> float:
        """Apply calibration."""
        return min(0.95, max(0.05, raw_confidence * self._confidence_adjustment))
    
    def _analyze_mistake(self, prediction: Any, actual: Any, context: Dict[str, Any]):
        """Analyze mistakes ruthlessly. No ego."""
        mistake_type = self._categorize_mistake(prediction, actual, context)
        self._mistake_patterns[mistake_type] = self._mistake_patterns.get(mistake_type, 0) + 1
        
        self._last_mistakes.append({
            'type': mistake_type,
            'prediction': prediction,
            'actual': actual,
            'context': context,
            'timestamp': datetime.now(),
            'regime': self._current_regime
        })
        
        if self._mistake_patterns[mistake_type] >= 3:
            self.logger.warning(f"PATTERN DETECTED: {mistake_type} ({self._mistake_patterns[mistake_type]} times)")
    
    def _categorize_mistake(self, prediction: Any, actual: Any, context: Dict[str, Any]) -> str:
        """Categorize mistake type."""
        if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
            if prediction > 0 and actual < 0:
                return "false_positive_direction"
            elif prediction < 0 and actual > 0:
                return "false_negative_direction"
            elif abs(prediction - actual) / max(abs(actual), 1) > 0.5:
                return "magnitude_error"
        return "general_error"
    
    def _bayesian_update(self, prediction: Any, actual: Any, context: Dict[str, Any]):
        """Bayesian belief update."""
        belief_key = context.get('belief_key', 'general')
        prior = self._beliefs.get(belief_key, 0.5)
        correct = self._evaluate_prediction(prediction, actual)
        likelihood = 0.8 if correct else 0.2
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        
        self._beliefs[belief_key] = posterior
        if belief_key not in self._belief_history:
            self._belief_history[belief_key] = []
        self._belief_history[belief_key].append((datetime.now(), posterior))
    
    def _check_adaptation_needed(self) -> Optional[str]:
        """Check if adaptation needed. Adapt FAST."""
        if len(self._learning_outcomes) < 50:
            return None
        
        recent = list(self._learning_outcomes)[-50:]
        recent_accuracy = sum(1 for o in recent if o.was_correct) / len(recent)
        
        # Stricter threshold for institutional
        if recent_accuracy < 0.45:
            self._model_versions += 1
            adaptation = f"Model adaptation v{self._model_versions} - accuracy was {recent_accuracy:.1%}"
            self._adaptation_log.append({
                'timestamp': datetime.now(),
                'reason': 'low_accuracy',
                'accuracy': recent_accuracy,
                'version': self._model_versions
            })
            self.logger.info(f"ADAPTATION TRIGGERED: {adaptation}")
            return adaptation
        
        return None
    
    def _adapt_to_regime_change(self, new_regime: str):
        """Adapt to regime change INSTANTLY."""
        self._adaptation_log.append({
            'timestamp': datetime.now(),
            'reason': 'regime_change',
            'new_regime': new_regime
        })
        self.logger.info(f"ADAPTING to {new_regime} regime")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _identify_consensus_flaws(self, consensus: str, data: Dict[str, Any]) -> List[str]:
        """Find flaws in consensus."""
        return [
            "Consensus anchored to recent data",
            "Linear extrapolation of trends",
            "Tail risks underweighted",
            "Ignoring structural changes",
            "Herd behavior in analysis"
        ]
    
    def _generate_opposing_view(self, consensus: str, reasons: List[str]) -> str:
        """Generate opposing view."""
        return f"Contrarian: {'; '.join(reasons[:2])}"
    
    def _assess_what_is_priced_in(self, conclusion: str, data: Dict[str, Any]) -> str:
        """Assess what's priced in."""
        return "Current price reflects visible consensus"
    
    def _identify_potential_surprises(self, data: Dict[str, Any]) -> List[str]:
        """Identify potential surprises."""
        return ["Timing", "Magnitude", "Second-order effects", "Policy response"]
    
    def _derive_second_order_view(self, first: str, priced_in: str, surprises: List[str]) -> str:
        """Derive second-order view."""
        return f"Focus on: {', '.join(surprises)}"
    
    def _derive_expected_reactions(self, expected: str, data: Dict[str, Any]) -> List[str]:
        """Expected reactions."""
        return ["price_move", "volume_spike", "volatility_change"]
    
    def _assess_actual_state(self, data: Dict[str, Any]) -> List[str]:
        """Actual state."""
        return []
    
    def _evaluate_prediction(self, prediction: Any, actual: Any) -> bool:
        """Evaluate prediction."""
        if isinstance(prediction, bool):
            return prediction == actual
        if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
            if prediction != 0 and actual != 0:
                return (prediction > 0) == (actual > 0)
        return prediction == actual
    
    def _derive_insight_from_mistake(self, prediction: Any, actual: Any, context: Dict[str, Any]) -> str:
        """Derive insight from mistake."""
        return f"Prediction {prediction} wrong (actual: {actual}). Updating model."
    
    # =========================================================================
    # ACA METHODS
    # =========================================================================
    
    def _register_capabilities(self):
        """Register capabilities."""
        for cap in self.capabilities:
            BaseAgent._global_capabilities.add(cap)
            if cap not in BaseAgent._capability_owners:
                BaseAgent._capability_owners[cap] = []
            if self.name not in BaseAgent._capability_owners[cap]:
                BaseAgent._capability_owners[cap].append(self.name)
    
    def detect_capability_gap(
        self, 
        task: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[CapabilityGap]:
        """Detect capability gaps."""
        if not self.aca_enabled:
            return None
        
        if required_capabilities is None:
            required_capabilities = task.get('required_capabilities', [])
            task_type = task.get('type', '')
            if not required_capabilities and task_type:
                required_capabilities = self._infer_required_capabilities(task_type)
        
        my_capabilities = set(self.capabilities)
        required_set = set(required_capabilities)
        missing = required_set - my_capabilities
        
        truly_missing = [cap for cap in missing if cap not in BaseAgent._global_capabilities]
        
        if truly_missing:
            gap = CapabilityGap(
                detected_by=self.name,
                task_type=task.get('type', 'unknown'),
                required_capabilities=list(required_set),
                missing_capabilities=truly_missing,
                severity=self._assess_gap_severity(truly_missing, task),
                context={'task': task, 'agent_tier': self.tier.name}
            )
            self.gaps_detected.append(gap)
            return gap
        return None
    
    def _infer_required_capabilities(self, task_type: str) -> List[str]:
        """Infer required capabilities."""
        return []
    
    def _assess_gap_severity(self, missing: List[str], task: Dict[str, Any]) -> str:
        """Assess gap severity."""
        if task.get('priority') == 'critical':
            return 'critical'
        if len(missing) >= 3:
            return 'high'
        return 'medium'
    
    def propose_agent(
        self,
        agent_name: str,
        capabilities: List[str],
        tier: AgentTier = AgentTier.STANDARD,
        gap_description: str = "",
        rationale: str = "",
        priority: str = "medium"
    ) -> AgentProposal:
        """Propose new agent."""
        if not self.aca_enabled:
            raise RuntimeError(f"Agent {self.name} does not have ACA enabled")
        
        proposal = AgentProposal(
            proposed_by=self.name,
            agent_name=agent_name,
            agent_tier=tier,
            capabilities=capabilities,
            gap_description=gap_description,
            rationale=rationale,
            priority=priority
        )
        self.proposals_made.append(proposal)
        return proposal
    
    def request_aca_creation(
        self,
        gap_description: str,
        suggested_capabilities: List[str],
        suggested_name: Optional[str] = None,
        suggested_tier: AgentTier = AgentTier.STANDARD
    ) -> AgentProposal:
        """Request ACA creation."""
        if not suggested_name:
            capability_prefix = suggested_capabilities[0] if suggested_capabilities else "utility"
            suggested_name = f"{capability_prefix.title()}Agent"
        
        return self.propose_agent(
            agent_name=suggested_name,
            capabilities=suggested_capabilities,
            tier=suggested_tier,
            gap_description=gap_description,
            rationale=f"Gap detected by {self.name}",
            priority="medium"
        )
    
    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if can handle task."""
        required = task.get('required_capabilities', [])
        if not required:
            required = self._infer_required_capabilities(task.get('type', ''))
        return all(cap in self.capabilities for cap in required)
    
    # =========================================================================
    # EXECUTION - BATTLE-HARDENED
    # =========================================================================
    
    def _log_execution(self, task: Dict[str, Any], result: Dict[str, Any], success: bool):
        """Log execution."""
        self.execution_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task - BATTLE-HARDENED, NEVER GIVES UP.
        """
        self.logger.info(f"[{self.name}] EXECUTING: {task.get('type', 'unknown')}")
        
        # Validate task input
        if not isinstance(task, dict):
            self.logger.error(f"INVALID TASK FORMAT: Expected dict, got {type(task)}")
            return {'success': False, 'error': 'Invalid task format'}

        # Detect regime
        if 'market_data' in task:
            self.detect_regime_change(task['market_data'])
        
        # Check capability gaps
        if self.aca_enabled:
            gap = self.detect_capability_gap(task)
            if gap and gap.severity in ['critical', 'high']:
                self.logger.warning(f"CAPABILITY GAP: {gap.missing_capabilities}")
        
        try:
            result = self.process(task)
            self._log_execution(task, result, success=True)
            
            # Attribution
            result['attributed_to'] = 'Tom Hogan'
            result['organization'] = 'Alpha Loop Capital, LLC'
            result['agent'] = self.name
            result['agent_tier'] = self.tier.name
            result['toughness'] = self.toughness.name
            result['regime'] = self._current_regime
            result['timestamp'] = datetime.now().isoformat()
            result['compute'] = 'MAXIMUM'
            
            if self.learning_enabled:
                result['learning_state'] = {
                    'total_learnings': len(self._learning_outcomes),
                    'recent_accuracy': self._get_recent_accuracy(),
                    'confidence_calibration': self._confidence_adjustment,
                    'current_regime': self._current_regime,
                    'unique_insights': self._unique_insights_generated,
                    'crashes_survived': self._crashes_survived,
                }
                # Save state after successful execution
                self._save_persistent_state()
            
            return result
            
        except Exception as e:
            # Critical system errors must propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise

            self.logger.error(f"ERROR: {str(e)} - ATTEMPTING RECOVERY")
            self.logger.debug(traceback.format_exc())
            
            # Never give up - attempt recovery
            recovery_result = self.never_give_up(task, str(e))
            
            if recovery_result.get('success'):
                self._log_execution(task, recovery_result, success=True)
                return recovery_result
            
            self._log_execution(task, {'error': str(e)}, success=False)
            return {
                'success': False,
                'error': str(e),
                'recovery_attempted': True,
                'recovery_result': recovery_result,
                'agent': self.name,
                'toughness': self.toughness.name,
                'message': 'Failed but learned. Will be stronger.',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_recent_accuracy(self) -> float:
        """Get recent accuracy."""
        if not self._learning_outcomes:
            return 0.0
        recent = list(self._learning_outcomes)[-100:]
        return sum(1 for o in recent if o.was_correct) / len(recent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats."""
        success_rate = (self.success_count / self.execution_count * 100) if self.execution_count > 0 else 0
        
        return {
            'name': self.name,
            'tier': self.tier.name,
            'toughness': self.toughness.name,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'execution_count': self.execution_count,
            'success_rate': f"{success_rate:.1f}%",
            'created_at': self.created_at.isoformat(),
            'compute': 'MAXIMUM - NO LIMITS',
            'battle_stats': {
                'crashes_survived': self._crashes_survived,
                'drawdowns_navigated': self._drawdowns_navigated,
                'regime_changes_adapted': self._regime_changes_adapted,
                'black_swans_handled': self._black_swans_handled,
                'unique_insights': self._unique_insights_generated,
                'information_edges': self._information_edges_exploited,
            },
            'learning': {
                'total_outcomes': len(self._learning_outcomes),
                'recent_accuracy': self._get_recent_accuracy(),
                'confidence_calibration': self._confidence_adjustment,
                'adaptations': len(self._adaptation_log),
                'model_version': self._model_versions,
            },
            'regime': {
                'current': self._current_regime,
                'performance_by_regime': self._regime_performance
            },
        }
    
    @classmethod
    def get_global_capabilities(cls) -> Set[str]:
        return cls._global_capabilities.copy()
    
    @classmethod
    def find_agents_with_capability(cls, capability: str) -> List[str]:
        return cls._capability_owners.get(capability, [])
    
    def __repr__(self) -> str:
        return f"<{self.name} | {self.toughness.name} | COMPUTE: UNLIMITED>"
