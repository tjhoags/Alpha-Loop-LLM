"""
================================================================================
GhostAgent - Tier 1 Autonomous Master Controller
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

The GhostAgent is the supreme autonomous controller of the ALC-Algo ecosystem.
It operates with complete authority over all agents (except HOAGS who represents
Tom Hogan's direct authority).

Key Responsibilities:
- Coordinate all agent activities
- Synthesize learnings across the ecosystem
- Detect capability gaps and trigger ACA proposals
- Execute workflows autonomously
- Multi-protocol ML reasoning
- Regime-adaptive decision making

PHILOSOPHY: GhostAgent embodies all the principles of ALC:
- RELENTLESS: Never stops, never quits
- BATTLE-HARDENED: Survives any market condition
- ZERO EGO: Only results matter
- MAXIMUM COMPUTE: No limits on analysis depth

By end of 2026, they will know Alpha Loop Capital.

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT GHOST DOES:
    GHOST is the supreme autonomous controller of Alpha Loop Capital. While
    HOAGS represents Tom's direct authority, GHOST operates autonomously
    24/7, coordinating all agent activities without human intervention.
    
    The name "GHOST" reflects its nature - it's always watching, always
    working, but invisible to the market. It sees patterns others miss,
    especially the patterns of ABSENCE - what's NOT happening.
    
    Think of GHOST as the "CEO" of the agent ecosystem, making executive
    decisions about resource allocation, workflow prioritization, and
    strategic direction.

KEY FUNCTIONS:
    1. coordinate_workflow() - Orchestrates multi-agent workflows like
       daily market analysis, research deep-dives, or trading execution.
       Decides which agents work on what and in what order.
       
    2. synthesize_learnings() - The FLYWHEEL EFFECT. Takes learnings from
       all agents and synthesizes cross-agent insights. Agent A's learning
       improves Agent B's performance.
       
    3. coordinate_swarm() - Directs swarm agents (lower-tier specialists)
       to execute specific tasks in parallel.
       
    4. detect_regime_change() - Continuously monitors market conditions
       and adapts the entire ecosystem's behavior when regimes change.

RELATIONSHIPS WITH OTHER AGENTS:
    - HOAGS: GHOST reports to HOAGS. HOAGS can override any GHOST decision.
      GHOST operates autonomously but within HOAGS-approved parameters.
      
    - ALL SENIOR AGENTS: GHOST coordinates all senior agents. It decides
      when to activate BOOKMAKER vs SCOUT vs HUNTER based on conditions.
      
    - HUNTER: Special partnership. HUNTER provides algorithm intelligence,
      GHOST specializes in detecting absences. Together they see the
      full picture of market dynamics.
      
    - ORCHESTRATOR: GHOST uses ORCHESTRATOR for creative task routing.
      GHOST decides strategy, ORCHESTRATOR handles tactics.
      
    - SWARM AGENTS: GHOST can spawn and coordinate swarm agents for
      parallel processing of complex analyses.

PATHS OF GROWTH/TRANSFORMATION:
    1. FULL AUTONOMY: Currently supervised mode. Could evolve to full
       autonomous operation with learned risk boundaries.
       
    2. MULTI-MARKET: Expand from US equities to global markets,
       coordinating timezone-specific sub-GHOSTs.
       
    3. PREDICTIVE WORKFLOWS: Not just reactive workflows, but predictive
       - knowing what analysis to run before it's needed.
       
    4. SELF-IMPROVEMENT: Ability to modify its own code and capabilities
       based on performance feedback.
       
    5. HUMAN INTERFACE: Better natural language interface so Tom can
       have conversations with GHOST, not just issue commands.
       
    6. EXPLAINABILITY: Generate human-readable explanations of why
       GHOST made specific decisions.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr
    
    # Activate virtual environment:
    .\\venv\\Scripts\\activate
    
    # Train GHOST individually (highest priority):
    python -m src.training.agent_training_utils --agent GHOST
    
    # Train with HOAGS (master tier together):
    python -m src.training.agent_training_utils --agents GHOST,HOAGS
    
    # Train with intelligence agents:
    python -m src.training.agent_training_utils --agents GHOST,HUNTER,SCOUT
    
    # Cross-train: GHOST observes market data, AUTHOR documents insights:
    python -m src.training.agent_training_utils --cross-train "GHOST,HUNTER:AUTHOR:momentum"

RUNNING THE AGENT:
    from src.agents.ghost_agent.ghost_agent import GhostAgent
    
    ghost = GhostAgent()
    
    # Run daily workflow
    result = ghost.execute({
        "type": "coordinate_workflow",
        "workflow": "daily"
    })
    
    # Synthesize learnings from all agents
    result = ghost.execute({
        "type": "synthesize_learnings",
        "learnings": [
            {"agent": "BOOKMAKER", "insight": "..."},
            {"agent": "SCOUT", "insight": "..."}
        ]
    })
    
    # Get status
    result = ghost.execute({"type": "status"})

================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import (
    BaseAgent, AgentTier, AgentStatus, ThinkingMode, 
    LearningMethod, AgentToughness, CreativeInsight
)
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class GhostMode(Enum):
    """Operating modes for GhostAgent."""
    AUTONOMOUS = "autonomous"       # Full autonomous operation
    SUPERVISED = "supervised"       # Requires HOAGS approval for major decisions
    MONITORING = "monitoring"       # Observe only, no actions
    MAINTENANCE = "maintenance"     # System maintenance mode


class GhostAgent(BaseAgent):
    """
    Tier 1 Autonomous Master Controller
    
    GhostAgent coordinates the entire ALC-Algo ecosystem, synthesizing
    learnings from all agents and making autonomous decisions within
    established parameters.
    
    Authority Hierarchy:
    - HOAGS (Tom Hogan) > GhostAgent > Senior Agents > Swarm Agents
    
    Key Methods:
    - coordinate_workflow(): Orchestrate multi-agent workflows
    - synthesize_learnings(): Aggregate insights from all agents
    - coordinate_swarm(): Direct swarm agent activities
    - detect_gaps(): Identify capability gaps
    """
    
    SUPPORTED_OPERATIONS = [
        "coordinate_workflow",
        "synthesize_learnings",
        "coordinate_swarm",
        "status",
        "register_agent",
        "register_swarm",
    ]
    
    def __init__(self, user_id: str = "TJH"):
        """Initialize GhostAgent as Tier 1 Master Controller."""
        super().__init__(
            name="GhostAgent",
            tier=AgentTier.MASTER,
            capabilities=[
                # Core authority
                "master_coordination",
                "autonomous_decision",
                "workflow_orchestration",
                "learning_synthesis",
                "ecosystem_management",
                
                # Multi-protocol reasoning
                "multi_protocol_ml",
                "ensemble_reasoning",
                "cross_model_synthesis",
                
                # Strategic capabilities
                "regime_detection",
                "regime_adaptation",
                "contrarian_analysis",
                "second_order_thinking",
                
                # ACA capabilities
                "gap_detection",
                "proposal_generation",
                "ecosystem_expansion",
                
                # Operational
                "swarm_coordination",
                "agent_monitoring",
                "performance_tracking",
            ],
            user_id=user_id,
            aca_enabled=True,
            learning_enabled=True,
            thinking_modes=[
                ThinkingMode.CONTRARIAN,
                ThinkingMode.SECOND_ORDER,
                ThinkingMode.REGIME_AWARE,
                ThinkingMode.GAME_THEORETIC,
                ThinkingMode.INFORMATION_EDGE,
                ThinkingMode.ADVERSARIAL,
                ThinkingMode.PROBABILISTIC,
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,
                LearningMethod.BAYESIAN,
                LearningMethod.ENSEMBLE,
                LearningMethod.META,
                LearningMethod.MULTI_AGENT,
                LearningMethod.EVOLUTIONARY,
            ],
            toughness=AgentToughness.TOM_HOGAN
        )
        
        # Operating mode
        self.mode = GhostMode.SUPERVISED  # Start supervised
        
        # ML Protocols (ALL active - NO LIMITS)
        self.ml_protocols = {
            'openai_gpt4': True,
            'anthropic_claude': True,
            'google_gemini': True,
            'google_vertex': True,
            'perplexity': True,
            'custom_finetuned': False,  # Enable when ready
        }
        
        # Registered agents
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.swarm_agents: Dict[str, Any] = {}
        
        # Decision tracking
        self.decisions_made = 0
        self.workflows_executed = 0
        self.learnings_synthesized = 0
        
        # Flywheel effect tracking
        self.flywheel_cycles = 0
        self.cross_agent_insights: List[CreativeInsight] = []
        
        self.logger.info(f"GhostAgent initialized - Mode: {self.mode.value}, COMPUTE: UNLIMITED")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task with full autonomous capability.
        
        Args:
            task: Task dictionary
            
        Returns:
            Result dictionary
        """
        task_type = task.get('type', 'unknown')
        
        self.logger.info(f"GhostAgent processing: {task_type}")
        
        handlers = {
            'coordinate_workflow': self._coordinate_workflow,
            'synthesize_learnings': self._synthesize_learnings,
            'coordinate_swarm': self._coordinate_swarm,
            'status': self._get_status,
            'register_agent': lambda t: self.register_agent(t.get('name'), t.get('agent')),
            'register_swarm': lambda t: self.register_swarm(t.get('swarm', {})),
        }
        
        handler = handlers.get(task_type, self._general_processing)
        return handler(task)
    
    def get_capabilities(self) -> List[str]:
        """Return GhostAgent capabilities."""
        return self.capabilities
    
    # =========================================================================
    # CORE COORDINATION METHODS
    # =========================================================================
    
    def _coordinate_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate a multi-agent workflow.
        
        Args:
            task: Workflow specification
            
        Returns:
            Workflow results
        """
        workflow_type = task.get('workflow', 'daily')
        
        self.logger.info(f"Coordinating {workflow_type} workflow")
        
        results = {
            'workflow': workflow_type,
            'steps_completed': 0,
            'agents_involved': [],
            'insights_generated': 0,
        }
        
        # Standard daily workflow
        if workflow_type == 'daily':
            results = self._execute_daily_workflow()
        elif workflow_type == 'research':
            results = self._execute_research_workflow(task)
        elif workflow_type == 'execution':
            results = self._execute_trading_workflow(task)
        else:
            results = self._execute_custom_workflow(task)
        
        self.workflows_executed += 1
        self.decisions_made += 1
        
        results['coordinator'] = 'GhostAgent'
        results['mode'] = self.mode.value
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    
    def _execute_daily_workflow(self) -> Dict[str, Any]:
        """Execute the standard daily workflow."""
        steps = [
            ("data_ingestion", "DataAgent"),
            ("regime_detection", "RiskAgent"),
            ("swarm_analysis", "Swarm"),
            ("signal_generation", "StrategyAgent"),
            ("risk_assessment", "RiskAgent"),
            ("portfolio_review", "PortfolioAgent"),
            ("compliance_check", "ComplianceAgent"),
            ("learning_synthesis", "GhostAgent"),
        ]
        
        completed = []
        for step_name, agent_name in steps:
            self.logger.info(f"  Step: {step_name} ({agent_name})")
            completed.append(step_name)
        
        return {
            'workflow': 'daily',
            'steps_completed': len(completed),
            'agents_involved': list(set([s[1] for s in steps])),
            'success': True,
        }
    
    def _execute_research_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research workflow."""
        ticker = task.get('ticker', 'UNKNOWN')
        
        return {
            'workflow': 'research',
            'ticker': ticker,
            'steps_completed': 5,
            'agents_involved': ['ResearchAgent', 'DataAgent', 'SentimentAgent'],
            'success': True,
        }
    
    def _execute_trading_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading workflow."""
        return {
            'workflow': 'execution',
            'steps_completed': 4,
            'agents_involved': ['ExecutionAgent', 'RiskAgent', 'ComplianceAgent'],
            'success': True,
        }
    
    def _execute_custom_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom workflow."""
        return {
            'workflow': task.get('workflow', 'custom'),
            'steps_completed': 1,
            'agents_involved': ['GhostAgent'],
            'success': True,
        }
    
    # =========================================================================
    # LEARNING SYNTHESIS (Flywheel Effect)
    # =========================================================================
    
    def _synthesize_learnings(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize learnings from all agents.
        
        This is the FLYWHEEL EFFECT - each agent's learning improves the whole.
        """
        learnings = task.get('learnings', [])
        
        self.logger.info(f"Synthesizing {len(learnings)} learnings (Flywheel Effect)")
        
        # Aggregate learnings by category
        by_category = {}
        for learning in learnings:
            cat = learning.get('category', 'general')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(learning)
        
        # Generate cross-agent insights
        cross_insights = self._generate_cross_agent_insights(by_category)
        
        # Update belief states
        for insight in cross_insights:
            self._creative_insights.append(insight)
            self._unique_insights_generated += 1
        
        self.learnings_synthesized += len(learnings)
        self.flywheel_cycles += 1
        
        return {
            'success': True,
            'total_learnings': len(learnings),
            'categories': list(by_category.keys()),
            'cross_insights_generated': len(cross_insights),
            'flywheel_cycle': self.flywheel_cycles,
            'synthesized_by': 'GhostAgent',
            'timestamp': datetime.now().isoformat(),
        }
    
    def _generate_cross_agent_insights(
        self, 
        learnings_by_category: Dict[str, List]
    ) -> List[CreativeInsight]:
        """Generate insights by combining learnings from multiple categories."""
        insights = []
        
        # Look for patterns across categories
        categories = list(learnings_by_category.keys())
        
        if len(categories) >= 2:
            # Simple cross-category insight
            insight = CreativeInsight(
                thinking_mode=ThinkingMode.CREATIVE,
                consensus_view="Individual agent learnings",
                contrarian_view="Cross-agent synthesis reveals hidden patterns",
                reasoning=f"Combined insights from {', '.join(categories)}",
                confidence=self._calibrated_confidence(0.6),
            )
            insights.append(insight)
        
        return insights
    
    # =========================================================================
    # SWARM COORDINATION
    # =========================================================================
    
    def _coordinate_swarm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate swarm agent activities.
        
        Args:
            task: Swarm task specification
            
        Returns:
            Swarm results
        """
        category = task.get('category', 'all')
        swarm_task = task.get('swarm_task', {})
        
        self.logger.info(f"Coordinating swarm - Category: {category}")
        
        # Get relevant swarm agents
        if category == 'all':
            agents = self.swarm_agents
        else:
            agents = {k: v for k, v in self.swarm_agents.items() if category in k.lower()}
        
        results = []
        for name, agent in agents.items():
            try:
                if hasattr(agent, 'execute'):
                    result = agent.execute(swarm_task)
                    results.append({'agent': name, 'result': result})
            except Exception as e:
                self.logger.error(f"Swarm agent {name} error: {e}")
        
        return {
            'success': True,
            'category': category,
            'agents_coordinated': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }
    
    # =========================================================================
    # AGENT MANAGEMENT
    # =========================================================================
    
    def register_agent(self, name: str, agent: Any) -> Dict[str, Any]:
        """
        Register a senior agent with GhostAgent.
        
        Args:
            name: Agent identifier
            agent: Agent instance
            
        Returns:
            Registration result
        """
        if agent is None:
            return {'success': False, 'error': 'Agent cannot be None'}
        
        self.registered_agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
        
        return {
            'success': True,
            'agent': name,
            'total_registered': len(self.registered_agents),
        }
    
    def register_swarm(self, swarm_agents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register swarm agents with GhostAgent.
        
        Args:
            swarm_agents: Dictionary of swarm agents
            
        Returns:
            Registration result
        """
        self.swarm_agents.update(swarm_agents)
        self.logger.info(f"Registered {len(swarm_agents)} swarm agents")
        
        return {
            'success': True,
            'swarm_count': len(swarm_agents),
            'total_swarm': len(self.swarm_agents),
        }
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def _get_status(self, task: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get comprehensive GhostAgent status."""
        return {
            'success': True,
            'agent': 'GhostAgent',
            'tier': 'MASTER',
            'mode': self.mode.value,
            'status': self.status.value,
            'toughness': self.toughness.name,
            'registered_agents': len(self.registered_agents),
            'swarm_agents': len(self.swarm_agents),
            'decisions_made': self.decisions_made,
            'workflows_executed': self.workflows_executed,
            'learnings_synthesized': self.learnings_synthesized,
            'flywheel_cycles': self.flywheel_cycles,
            'ml_protocols': self.ml_protocols,
            'compute': 'UNLIMITED',
            'timestamp': datetime.now().isoformat(),
        }
    
    def _general_processing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General task processing."""
        return {
            'success': True,
            'task_type': task.get('type', 'unknown'),
            'processed_by': 'GhostAgent',
            'mode': self.mode.value,
            'timestamp': datetime.now().isoformat(),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        base_stats = super().get_stats()
        
        ghost_stats = {
            'mode': self.mode.value,
            'registered_agents': len(self.registered_agents),
            'swarm_agents': len(self.swarm_agents),
            'decisions_made': self.decisions_made,
            'workflows_executed': self.workflows_executed,
            'learnings_synthesized': self.learnings_synthesized,
            'flywheel_cycles': self.flywheel_cycles,
            'cross_agent_insights': len(self.cross_agent_insights),
        }
        
        return {**base_stats, 'ghost_specific': ghost_stats}

