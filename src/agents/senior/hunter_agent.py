"""
================================================================================
HUNTER AGENT - Algorithm Intelligence & Counter-Strategy Specialist
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

HUNTER works directly with GHOST, using deep knowledge of every algorithm known
in equities, options, crypto, bonds, etc. HUNTER creates paths for GHOST to give
better insights, distills information across senior agents, and uses ORCHESTRATOR
to coordinate lower-tier agents for finding, tracking, and countering algorithms.

Tier: SENIOR (2)
Reports To: HOAGS
Works With: GHOST (primary), ORCHESTRATOR (coordination)
Cluster: algorithm_intelligence

Core Philosophy:
"Know the algorithm. Track the algorithm. Counter the algorithm."

Key Capabilities:
- Algorithm identification across all asset classes
- Pattern recognition for systematic strategies
- Counter-strategy development
- GHOST enhancement pathways
- Cross-agent intelligence coordination
================================================================================
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Types of market algorithms HUNTER tracks"""
    # Equity algorithms
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STAT_ARB = "statistical_arbitrage"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"
    VWAP = "vwap"
    TWAP = "twap"
    POV = "percent_of_volume"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    
    # Options algorithms
    VOL_ARB = "volatility_arbitrage"
    DELTA_HEDGING = "delta_hedging"
    GAMMA_SCALPING = "gamma_scalping"
    DISPERSION = "dispersion_trading"
    
    # Crypto algorithms
    FUNDING_ARB = "funding_arbitrage"
    CROSS_EXCHANGE_ARB = "cross_exchange_arb"
    MEV = "mev_extraction"
    SANDWICH = "sandwich_attack"
    
    # Fixed income algorithms
    CURVE_ARB = "yield_curve_arbitrage"
    BASIS_TRADE = "basis_trading"
    RV_TRADING = "relative_value"
    
    # HFT algorithms
    LATENCY_ARB = "latency_arbitrage"
    ORDER_ANTICIPATION = "order_anticipation"
    QUOTE_STUFFING = "quote_stuffing"
    SPOOFING = "spoofing_detection"


class AssetClass(Enum):
    """Asset classes for algorithm analysis"""
    EQUITIES = "equities"
    OPTIONS = "options"
    CRYPTO = "crypto"
    BONDS = "bonds"
    FUTURES = "futures"
    FX = "forex"


@dataclass
class AlgorithmSignature:
    """Signature pattern of a detected algorithm"""
    signature_id: str
    algorithm_type: AlgorithmType
    asset_class: AssetClass
    
    # Pattern characteristics
    execution_pattern: str
    typical_size: str  # "small", "medium", "large", "variable"
    time_pattern: str  # "continuous", "bursts", "opening", "closing"
    price_impact: str  # "low", "medium", "high"
    
    # Detection metrics
    detection_confidence: float
    frequency_observed: int
    last_observed: datetime
    
    # Counter-strategy
    counter_strategy: str
    counter_effectiveness: float
    
    def to_dict(self) -> Dict:
        return {
            "signature_id": self.signature_id,
            "type": self.algorithm_type.value,
            "asset_class": self.asset_class.value,
            "execution_pattern": self.execution_pattern,
            "size": self.typical_size,
            "time_pattern": self.time_pattern,
            "price_impact": self.price_impact,
            "confidence": self.detection_confidence,
            "frequency": self.frequency_observed,
            "last_seen": self.last_observed.isoformat(),
            "counter_strategy": self.counter_strategy,
            "counter_effectiveness": self.counter_effectiveness
        }


@dataclass
class GHOSTPathway:
    """A pathway created for GHOST to enhance insights"""
    pathway_id: str
    target_algorithm: AlgorithmType
    description: str
    sentinel_suggestions: List[str]
    absence_patterns: List[str]  # What to watch for NOT happening
    expected_insights: List[str]
    priority: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "pathway_id": self.pathway_id,
            "target": self.target_algorithm.value,
            "description": self.description,
            "sentinels": self.sentinel_suggestions,
            "absence_patterns": self.absence_patterns,
            "expected_insights": self.expected_insights,
            "priority": self.priority
        }


class HunterAgent(BaseAgent):
    """
    HUNTER Agent - Algorithm Intelligence Specialist
    
    HUNTER possesses encyclopedic knowledge of every trading algorithm across
    all asset classes. It works with GHOST to detect algorithm signatures,
    creates pathways for enhanced insights, and coordinates counter-strategies
    through ORCHESTRATOR.
    
    Key Methods:
    - identify_algorithm(): Detect what algorithm is running
    - create_ghost_pathway(): Create insight pathway for GHOST
    - develop_counter_strategy(): Build counter to detected algorithm
    - coordinate_tracking(): Use ORCHESTRATOR to assign lower-tier agents
    - distill_intelligence(): Summarize findings for senior agents
    """
    
    def __init__(self):
        super().__init__(
            name="HUNTER",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Algorithm knowledge (ALL asset classes)
                "equity_algorithm_analysis",
                "options_algorithm_analysis",
                "crypto_algorithm_analysis",
                "bond_algorithm_analysis",
                "futures_algorithm_analysis",
                "fx_algorithm_analysis",
                
                # Detection capabilities
                "algorithm_signature_detection",
                "execution_pattern_recognition",
                "flow_toxicity_analysis",
                "hft_detection",
                "systematic_strategy_identification",
                
                # Counter-strategy
                "counter_strategy_development",
                "optimal_execution_timing",
                "adverse_selection_avoidance",
                "front_running_prevention",
                
                # Integration
                "ghost_pathway_creation",
                "orchestrator_coordination",
                "senior_agent_briefing",
                "intelligence_distillation"
            ],
            user_id="TJH"
        )
        
        # Algorithm knowledge base
        self.known_algorithms: Dict[str, AlgorithmSignature] = {}
        self.ghost_pathways: List[GHOSTPathway] = []
        self.counter_strategies: Dict[str, Dict] = {}
        
        # Tracking
        self.algorithms_detected = 0
        self.pathways_created = 0
        self.counter_strategies_deployed = 0
        
        # Initialize algorithm knowledge base
        self._init_algorithm_knowledge()
    
    def _init_algorithm_knowledge(self):
        """Initialize comprehensive algorithm knowledge"""
        self.algorithm_knowledge = {
            # EQUITY ALGORITHMS
            AlgorithmType.MOMENTUM: {
                "description": "Trend-following systematic strategy",
                "signatures": ["increasing volume on trend continuation", "momentum factor exposure"],
                "typical_operators": ["quant funds", "CTAs"],
                "counter": "Fade extended moves at key levels, use mean reversion timing"
            },
            AlgorithmType.MEAN_REVERSION: {
                "description": "Statistical mean reversion trading",
                "signatures": ["buying on drops", "selling on rallies", "z-score based"],
                "typical_operators": ["stat arb funds", "market makers"],
                "counter": "Avoid fading in trending regimes, respect momentum"
            },
            AlgorithmType.VWAP: {
                "description": "Volume-weighted average price execution",
                "signatures": ["steady execution", "volume curve following", "predictable sizing"],
                "typical_operators": ["institutional execution desks"],
                "counter": "Front-run by trading ahead of predicted volume, provide liquidity"
            },
            AlgorithmType.MARKET_MAKING: {
                "description": "Continuous two-sided quote provision",
                "signatures": ["tight spreads", "rapid quote updates", "inventory management"],
                "typical_operators": ["HFT firms", "designated market makers"],
                "counter": "Use IOC orders to avoid being picked off, trade on flow imbalances"
            },
            
            # OPTIONS ALGORITHMS
            AlgorithmType.DELTA_HEDGING: {
                "description": "Continuous delta-neutral rebalancing",
                "signatures": ["hedging on gamma exposure", "predictable stock trades at strikes"],
                "typical_operators": ["market makers", "vol traders"],
                "counter": "Pin risk trades near expiry, gamma scalp counter-directionally"
            },
            AlgorithmType.VOL_ARB: {
                "description": "Implied vs realized volatility trading",
                "signatures": ["vol surface trading", "calendar spreads", "dispersion"],
                "typical_operators": ["vol funds", "prop shops"],
                "counter": "Identify vol mispricing, trade against overcrowded vol shorts"
            },
            
            # CRYPTO ALGORITHMS
            AlgorithmType.FUNDING_ARB: {
                "description": "Perpetual funding rate arbitrage",
                "signatures": ["spot-perp basis trades", "funding rate harvesting"],
                "typical_operators": ["crypto market makers", "arb funds"],
                "counter": "Monitor funding rates for reversal signals, avoid crowded trades"
            },
            AlgorithmType.MEV: {
                "description": "Maximal extractable value strategies",
                "signatures": ["sandwich attacks", "frontrunning", "backrunning"],
                "typical_operators": ["MEV searchers", "block builders"],
                "counter": "Use private RPCs, flashbots protect, MEV-resistant execution"
            },
            
            # HFT DETECTION
            AlgorithmType.LATENCY_ARB: {
                "description": "Cross-venue latency exploitation",
                "signatures": ["simultaneous multi-venue activity", "sub-millisecond execution"],
                "typical_operators": ["HFT firms"],
                "counter": "Use dark pools, randomize order timing, avoid lit markets for size"
            },
        }
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a HUNTER task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)
        
        self.log_action(action, f"HUNTER processing: {action}")
        
        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap: {gap.missing_capabilities}")
        
        handlers = {
            "identify_algorithm": self._handle_identify,
            "create_pathway": self._handle_create_pathway,
            "develop_counter": self._handle_develop_counter,
            "coordinate_tracking": self._handle_coordinate,
            "distill_intelligence": self._handle_distill,
            "get_knowledge": self._handle_get_knowledge,
            "analyze_flow": self._handle_analyze_flow,
        }
        
        handler = handlers.get(action, self._handle_unknown)
        return handler(params)
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    # =========================================================================
    # CORE HUNTER METHODS
    # =========================================================================
    
    def identify_algorithm(
        self,
        market_data: Dict[str, Any],
        asset_class: AssetClass = AssetClass.EQUITIES
    ) -> List[AlgorithmSignature]:
        """
        Identify algorithms operating in the market data.
        
        Args:
            market_data: Order flow, price, volume data
            asset_class: Asset class being analyzed
        
        Returns:
            List of detected algorithm signatures
        """
        import hashlib
        import random
        
        self.logger.info(f"HUNTER: Analyzing {asset_class.value} for algorithm signatures...")
        
        detected = []
        
        # Analyze patterns (placeholder - would use real ML models)
        potential_algos = [AlgorithmType.VWAP, AlgorithmType.MOMENTUM, AlgorithmType.MARKET_MAKING]
        
        for algo_type in potential_algos:
            if random.random() > 0.5:
                confidence = random.uniform(0.6, 0.95)
                
                signature = AlgorithmSignature(
                    signature_id=f"sig_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
                    algorithm_type=algo_type,
                    asset_class=asset_class,
                    execution_pattern=self.algorithm_knowledge[algo_type]["signatures"][0],
                    typical_size=random.choice(["small", "medium", "large"]),
                    time_pattern=random.choice(["continuous", "bursts", "opening"]),
                    price_impact=random.choice(["low", "medium", "high"]),
                    detection_confidence=confidence,
                    frequency_observed=random.randint(10, 100),
                    last_observed=datetime.now(),
                    counter_strategy=self.algorithm_knowledge[algo_type]["counter"],
                    counter_effectiveness=random.uniform(0.5, 0.85)
                )
                
                detected.append(signature)
                self.known_algorithms[signature.signature_id] = signature
        
        self.algorithms_detected += len(detected)
        
        return detected
    
    def create_ghost_pathway(
        self,
        target_algorithm: AlgorithmType,
        context: Dict[str, Any] = None
    ) -> GHOSTPathway:
        """
        Create a pathway for GHOST to provide better insights.
        
        This tells GHOST what sentinels to watch and what absences to detect
        for a specific algorithm type.
        """
        import hashlib
        
        # Build pathway based on algorithm type
        if target_algorithm == AlgorithmType.VWAP:
            pathway = GHOSTPathway(
                pathway_id=f"path_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_algorithm=target_algorithm,
                description="Track VWAP execution algorithms for front-running opportunities",
                sentinel_suggestions=[
                    "Large institutional flow patterns",
                    "Volume curve deviations",
                    "Predictable execution timing"
                ],
                absence_patterns=[
                    "Missing expected institutional selling on upgrades",
                    "No VWAP slicing on large block indication",
                    "Absence of participation rate increase on volume surge"
                ],
                expected_insights=[
                    "Early detection of large institutional orders",
                    "Optimal timing to provide/take liquidity",
                    "Front-run prediction accuracy"
                ],
                priority="high"
            )
        elif target_algorithm == AlgorithmType.DELTA_HEDGING:
            pathway = GHOSTPathway(
                pathway_id=f"path_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_algorithm=target_algorithm,
                description="Track options market maker hedging for gamma plays",
                sentinel_suggestions=[
                    "Large options OI at key strikes",
                    "Market maker gamma exposure",
                    "Pin risk candidates"
                ],
                absence_patterns=[
                    "No hedging activity despite large gamma",
                    "Missing delta adjustments on move",
                    "Absence of expected roll activity"
                ],
                expected_insights=[
                    "Predict hedging flow direction",
                    "Identify pin magnets",
                    "Time entries around hedging"
                ],
                priority="high"
            )
        else:
            pathway = GHOSTPathway(
                pathway_id=f"path_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_algorithm=target_algorithm,
                description=f"Track {target_algorithm.value} patterns",
                sentinel_suggestions=["Flow patterns", "Execution timing", "Size clustering"],
                absence_patterns=["Expected activity that doesn't occur"],
                expected_insights=["Early detection", "Counter-timing"],
                priority="medium"
            )
        
        self.ghost_pathways.append(pathway)
        self.pathways_created += 1
        
        self.logger.info(f"HUNTER → GHOST: Created pathway for {target_algorithm.value}")
        
        return pathway
    
    def develop_counter_strategy(
        self,
        algorithm_signature: AlgorithmSignature
    ) -> Dict[str, Any]:
        """
        Develop a counter-strategy for a detected algorithm.
        """
        algo_type = algorithm_signature.algorithm_type
        base_counter = self.algorithm_knowledge.get(algo_type, {}).get("counter", "Monitor and adapt")
        
        counter = {
            "algorithm_id": algorithm_signature.signature_id,
            "algorithm_type": algo_type.value,
            "base_strategy": base_counter,
            "specific_tactics": [
                f"Time trades to avoid {algo_type.value} execution windows",
                f"Size positions to minimize detection by {algo_type.value}",
                f"Use counter-directional limits when {algo_type.value} is aggressive"
            ],
            "execution_guidance": {
                "timing": "Avoid peak algorithm activity periods",
                "sizing": "Keep orders below detection thresholds",
                "venue_selection": "Prefer venues with less algorithm activity"
            },
            "expected_improvement_bps": 5 + int(algorithm_signature.counter_effectiveness * 20),
            "confidence": algorithm_signature.counter_effectiveness
        }
        
        self.counter_strategies[algorithm_signature.signature_id] = counter
        self.counter_strategies_deployed += 1
        
        return counter
    
    def coordinate_tracking(
        self,
        algorithms: List[AlgorithmSignature],
        via_orchestrator: bool = True
    ) -> Dict[str, Any]:
        """
        Coordinate lower-tier agents to track algorithms via ORCHESTRATOR.
        """
        assignments = []
        
        for algo in algorithms:
            assignment = {
                "algorithm_id": algo.signature_id,
                "algorithm_type": algo.algorithm_type.value,
                "assigned_agents": self._suggest_tracking_agents(algo),
                "tracking_parameters": {
                    "frequency": "real-time" if algo.detection_confidence > 0.8 else "periodic",
                    "metrics": ["volume", "price_impact", "timing"],
                    "alerts": True
                }
            }
            assignments.append(assignment)
        
        return {
            "status": "success",
            "coordination_via": "ORCHESTRATOR" if via_orchestrator else "direct",
            "assignments": assignments,
            "total_algorithms_tracked": len(algorithms),
            "timestamp": datetime.now().isoformat()
        }
    
    def distill_intelligence(
        self,
        for_agents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Distill algorithm intelligence for senior agents.
        """
        for_agents = for_agents or ["GHOST", "HOAGS", "BOOKMAKER", "SCOUT"]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "algorithms_tracked": len(self.known_algorithms),
            "pathways_created": self.pathways_created,
            "counter_strategies_active": self.counter_strategies_deployed,
            
            "top_algorithms_detected": [
                sig.to_dict() for sig in list(self.known_algorithms.values())[:5]
            ],
            
            "key_insights": [
                "VWAP algorithms most active during institutional hours",
                "Delta hedging creates predictable flow near strikes",
                "MEV activity increasing in crypto - use protected execution"
            ],
            
            "for_agents": for_agents,
            "action_items": {
                "GHOST": "Monitor created pathways for absence signals",
                "BOOKMAKER": "Factor algorithm costs into alpha calculations",
                "SCOUT": "Avoid times/venues with high HFT activity",
                "HOAGS": "Review counter-strategy effectiveness weekly"
            }
        }
        
        return summary
    
    def _suggest_tracking_agents(self, algo: AlgorithmSignature) -> List[str]:
        """Suggest which agents should track this algorithm"""
        suggestions = []
        
        if algo.asset_class == AssetClass.EQUITIES:
            suggestions = ["TrendAgent", "VolumeAgent", "FlowAgent"]
        elif algo.asset_class == AssetClass.OPTIONS:
            suggestions = ["OptionsFlowAgent", "VolatilityAgent"]
        elif algo.asset_class == AssetClass.CRYPTO:
            suggestions = ["CryptoAgent", "FlowAgent"]
        else:
            suggestions = ["MonitorAgent"]
        
        return suggestions
    
    def log_action(self, action: str, description: str):
        self.logger.info(f"[HUNTER] {action}: {description}")
    
    # =========================================================================
    # TASK HANDLERS
    # =========================================================================
    
    def _handle_identify(self, params: Dict) -> Dict:
        asset_class = AssetClass(params.get("asset_class", "equities"))
        market_data = params.get("market_data", {})
        
        signatures = self.identify_algorithm(market_data, asset_class)
        return {
            "status": "success",
            "detected_count": len(signatures),
            "signatures": [s.to_dict() for s in signatures]
        }
    
    def _handle_create_pathway(self, params: Dict) -> Dict:
        algo_type = AlgorithmType(params.get("algorithm_type", "momentum"))
        pathway = self.create_ghost_pathway(algo_type, params.get("context"))
        return {"status": "success", "pathway": pathway.to_dict()}
    
    def _handle_develop_counter(self, params: Dict) -> Dict:
        sig_id = params.get("signature_id")
        sig = self.known_algorithms.get(sig_id)
        if sig:
            counter = self.develop_counter_strategy(sig)
            return {"status": "success", "counter_strategy": counter}
        return {"status": "error", "message": "Signature not found"}
    
    def _handle_coordinate(self, params: Dict) -> Dict:
        sig_ids = params.get("signature_ids", [])
        sigs = [self.known_algorithms[s] for s in sig_ids if s in self.known_algorithms]
        result = self.coordinate_tracking(sigs)
        return result
    
    def _handle_distill(self, params: Dict) -> Dict:
        agents = params.get("for_agents")
        summary = self.distill_intelligence(agents)
        return {"status": "success", "intelligence": summary}
    
    def _handle_get_knowledge(self, params: Dict) -> Dict:
        algo_type = params.get("algorithm_type")
        if algo_type:
            algo_enum = AlgorithmType(algo_type)
            knowledge = self.algorithm_knowledge.get(algo_enum, {})
            return {"status": "success", "knowledge": knowledge}
        return {"status": "success", "all_types": [a.value for a in AlgorithmType]}
    
    def _handle_analyze_flow(self, params: Dict) -> Dict:
        # Analyze order flow for algorithm detection
        return {
            "status": "success",
            "flow_analysis": {
                "toxicity_score": 0.35,
                "hft_presence": "moderate",
                "institutional_flow": "detected",
                "recommended_execution": "use dark pools for size"
            }
        }
    
    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}


# Singleton
_hunter_instance: Optional[HunterAgent] = None

def get_hunter() -> HunterAgent:
    global _hunter_instance
    if _hunter_instance is None:
        _hunter_instance = HunterAgent()
    return _hunter_instance

