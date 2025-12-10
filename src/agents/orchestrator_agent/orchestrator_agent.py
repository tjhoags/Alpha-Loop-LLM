"""
================================================================================
ORCHESTRATOR AGENT - Creative Task Coordination & Agent Improvement
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

ORCHESTRATOR coordinates all agents, routes tasks optimally, and continuously
improves agents with new skillsets. Uses creative, out-of-the-box thinking
from psychology, sociology, and other disciplines to enhance agent capabilities.

Tier: SENIOR (2)
Reports To: HOAGS
Cluster: coordination

CREATIVE THINKING MANDATE:
- Apply psychology (behavioral economics, cognitive biases, game theory)
- Apply sociology (crowd behavior, network effects, social proof)
- Apply unconventional disciplines (military strategy, evolutionary biology)
- Generate novel approaches no one else is considering
- Articulate improvements clearly to THE_AUTHOR for documentation

Core Philosophy:
"Think different. Improve constantly. Orchestrate brilliantly."
================================================================================
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier

logger = logging.getLogger(__name__)


class CreativeFramework(Enum):
    """Creative thinking frameworks ORCHESTRATOR applies"""
    PSYCHOLOGY = "psychology"
    SOCIOLOGY = "sociology"
    BEHAVIORAL_ECONOMICS = "behavioral_economics"
    GAME_THEORY = "game_theory"
    MILITARY_STRATEGY = "military_strategy"
    EVOLUTIONARY = "evolutionary_biology"
    SYSTEMS_THINKING = "systems_thinking"
    DESIGN_THINKING = "design_thinking"
    FIRST_PRINCIPLES = "first_principles"
    INVERSION = "inversion"


@dataclass
class AgentImprovement:
    """An improvement proposed for an agent"""
    improvement_id: str
    target_agent: str
    improvement_type: str  # "new_skill", "enhanced_capability", "novel_approach"
    description: str
    creative_framework: CreativeFramework
    rationale: str
    expected_benefit: str
    implementation_notes: str
    status: str = "proposed"  # proposed, approved, implemented
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "improvement_id": self.improvement_id,
            "target_agent": self.target_agent,
            "type": self.improvement_type,
            "description": self.description,
            "framework": self.creative_framework.value,
            "rationale": self.rationale,
            "expected_benefit": self.expected_benefit,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TaskAssignment:
    """Assignment of a task to an agent"""
    task_id: str
    task_type: str
    assigned_to: str
    priority: int
    resources_allocated: List[str]
    deadline: Optional[datetime]
    creative_approach: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "type": self.task_type,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "resources": self.resources_allocated,
            "creative_approach": self.creative_approach
        }


class OrchestratorAgent(BaseAgent):
    """
    ORCHESTRATOR Agent - Creative Coordinator & Agent Improver
    
    ORCHESTRATOR uses creative, out-of-the-box thinking from psychology,
    sociology, and unconventional disciplines to:
    1. Optimally route tasks to the right agents
    2. Continuously improve agents with new skillsets
    3. Generate novel approaches others aren't considering
    4. Articulate improvements clearly to THE_AUTHOR
    
    Key Methods:
    - orchestrate(): Route tasks optimally
    - improve_agent(): Propose new skills/capabilities
    - apply_creative_framework(): Generate novel approaches
    - coordinate_resources(): Allocate agent resources
    - brief_author(): Prepare documentation for THE_AUTHOR
    """
    
    # Creative thinking frameworks with their applications
    CREATIVE_APPLICATIONS = {
        CreativeFramework.PSYCHOLOGY: {
            "disciplines": ["cognitive biases", "decision making", "motivation", "perception"],
            "market_applications": [
                "Exploit anchoring bias in analyst estimates",
                "Recognize confirmation bias in consensus views",
                "Use loss aversion for risk management framing",
                "Apply prospect theory to position sizing"
            ]
        },
        CreativeFramework.SOCIOLOGY: {
            "disciplines": ["crowd behavior", "social proof", "network effects", "institutional behavior"],
            "market_applications": [
                "Detect herding behavior before trend exhaustion",
                "Identify social proof cascade triggers",
                "Map institutional network effects",
                "Predict retail capitulation points"
            ]
        },
        CreativeFramework.BEHAVIORAL_ECONOMICS: {
            "disciplines": ["bounded rationality", "heuristics", "mental accounting", "time inconsistency"],
            "market_applications": [
                "Exploit mental accounting in sector rotation",
                "Identify bounded rationality in option pricing",
                "Trade against time-inconsistent behavior",
                "Find mispricings from heuristic shortcuts"
            ]
        },
        CreativeFramework.GAME_THEORY: {
            "disciplines": ["nash equilibrium", "mechanism design", "signaling", "repeated games"],
            "market_applications": [
                "Model activist investor game trees",
                "Analyze management incentive alignment",
                "Decode signaling in corporate actions",
                "Predict competitor responses"
            ]
        },
        CreativeFramework.MILITARY_STRATEGY: {
            "disciplines": ["OODA loop", "center of gravity", "flanking", "concentration of force"],
            "market_applications": [
                "Apply OODA loop to trading decisions",
                "Identify opponent's center of gravity",
                "Flanking positions on crowded trades",
                "Concentrate capital at decisive moments"
            ]
        },
        CreativeFramework.INVERSION: {
            "disciplines": ["avoiding failure", "Charlie Munger approach", "second-order effects"],
            "market_applications": [
                "Invert: How do we lose money? Then avoid it",
                "What would make this trade fail?",
                "Who loses if we're right?",
                "What are the unintended consequences?"
            ]
        }
    }
    
    def __init__(self):
        super().__init__(
            name="ORCHESTRATOR",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Core orchestration
                "task_routing",
                "resource_allocation",
                "agent_coordination",
                "priority_management",
                "workflow_optimization",
                
                # Creative thinking (NEW)
                "creative_thinking",
                "psychological_analysis",
                "sociological_insights",
                "behavioral_economics",
                "game_theory_application",
                "unconventional_approaches",
                "first_principles_reasoning",
                "inversion_thinking",
                
                # Agent improvement (NEW)
                "agent_skill_enhancement",
                "capability_gap_filling",
                "novel_approach_generation",
                "continuous_improvement",
                
                # Communication
                "author_briefing",
                "improvement_articulation",
                "documentation_preparation"
            ],
            user_id="TJH"
        )
        
        # Improvement tracking
        self.improvements_proposed: List[AgentImprovement] = []
        self.improvements_implemented: List[AgentImprovement] = []
        
        # Task tracking
        self.active_assignments: Dict[str, TaskAssignment] = {}
        self.completed_tasks = 0
        
        # Agent capability registry
        self.agent_capabilities: Dict[str, List[str]] = {}
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an ORCHESTRATOR task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)
        
        self.log_action(action, f"ORCHESTRATOR processing: {action}")
        
        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap: {gap.missing_capabilities}")
        
        handlers = {
            "orchestrate": self._handle_orchestrate,
            "improve_agent": self._handle_improve_agent,
            "apply_framework": self._handle_apply_framework,
            "coordinate": self._handle_coordinate,
            "brief_author": self._handle_brief_author,
            "get_creative_ideas": self._handle_get_creative,
            "route_task": self._handle_route_task,
            "get_improvements": self._handle_get_improvements,
        }
        
        handler = handlers.get(action, self._handle_unknown)
        return handler(params)
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    # =========================================================================
    # CORE ORCHESTRATOR METHODS
    # =========================================================================
    
    def orchestrate(
        self,
        task: Dict[str, Any],
        available_agents: List[str] = None
    ) -> TaskAssignment:
        """
        Orchestrate a task to the optimal agent(s).
        
        Considers:
        - Agent capabilities
        - Current workload
        - Task requirements
        - Creative approaches
        """
        import hashlib
        
        task_type = task.get("type", "general")
        priority = task.get("priority", 5)
        
        # Determine best agent(s)
        best_agent = self._select_best_agent(task_type, available_agents)
        
        # Generate creative approach if applicable
        creative_approach = self._generate_creative_approach(task)
        
        assignment = TaskAssignment(
            task_id=f"task_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            task_type=task_type,
            assigned_to=best_agent,
            priority=priority,
            resources_allocated=self._allocate_resources(task),
            deadline=task.get("deadline"),
            creative_approach=creative_approach
        )
        
        self.active_assignments[assignment.task_id] = assignment
        
        self.logger.info(f"ORCHESTRATOR: Assigned {task_type} to {best_agent}")
        
        return assignment
    
    def improve_agent(
        self,
        target_agent: str,
        framework: CreativeFramework = None,
        context: Dict[str, Any] = None
    ) -> AgentImprovement:
        """
        Propose an improvement for an agent using creative thinking.
        """
        import hashlib
        
        framework = framework or self._select_best_framework(target_agent, context)
        
        # Generate improvement using creative framework
        improvement_idea = self._generate_improvement(target_agent, framework, context)
        
        improvement = AgentImprovement(
            improvement_id=f"imp_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            target_agent=target_agent,
            improvement_type=improvement_idea["type"],
            description=improvement_idea["description"],
            creative_framework=framework,
            rationale=improvement_idea["rationale"],
            expected_benefit=improvement_idea["expected_benefit"],
            implementation_notes=improvement_idea["implementation"]
        )
        
        self.improvements_proposed.append(improvement)
        
        # Notify THE_AUTHOR
        self._notify_author(improvement)
        
        self.logger.info(f"ORCHESTRATOR: Proposed improvement for {target_agent} using {framework.value}")
        
        return improvement
    
    def apply_creative_framework(
        self,
        framework: CreativeFramework,
        problem: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply a creative thinking framework to generate novel solutions.
        """
        framework_data = self.CREATIVE_APPLICATIONS.get(framework, {})
        
        # Generate ideas using the framework
        ideas = []
        applications = framework_data.get("market_applications", [])
        
        for app in applications:
            ideas.append({
                "approach": app,
                "applied_to": problem,
                "novelty_score": self._assess_novelty(app, problem),
                "feasibility": "high" if "detect" in app.lower() or "identify" in app.lower() else "medium"
            })
        
        # Add custom generated ideas
        custom_ideas = self._brainstorm_custom(framework, problem, context)
        ideas.extend(custom_ideas)
        
        return {
            "framework": framework.value,
            "problem": problem,
            "ideas_generated": len(ideas),
            "ideas": ideas,
            "top_recommendation": ideas[0] if ideas else None,
            "disciplines_applied": framework_data.get("disciplines", [])
        }
    
    def coordinate_resources(
        self,
        task: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """
        Coordinate resources across multiple agents for a complex task.
        """
        coordination = {
            "task_type": task.get("type"),
            "agents_involved": agents,
            "resource_allocation": {},
            "communication_protocol": "async_message_passing",
            "creative_synergies": []
        }
        
        for agent in agents:
            coordination["resource_allocation"][agent] = {
                "cpu_priority": "normal",
                "data_access": "full",
                "output_channel": "shared_bus"
            }
        
        # Identify creative synergies
        coordination["creative_synergies"] = self._identify_synergies(agents)
        
        return coordination
    
    def brief_author(
        self,
        improvements: List[AgentImprovement] = None
    ) -> Dict[str, Any]:
        """
        Prepare a brief for THE_AUTHOR documenting improvements.
        """
        improvements = improvements or self.improvements_proposed[-10:]
        
        brief = {
            "timestamp": datetime.now().isoformat(),
            "total_improvements": len(improvements),
            "improvements": [imp.to_dict() for imp in improvements],
            "summary": self._generate_summary(improvements),
            "for_publication": True,
            "tone_guidance": "analytical with dry humor (Tom's style)"
        }
        
        return brief
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _select_best_agent(self, task_type: str, available: List[str] = None) -> str:
        """Select the best agent for a task type"""
        agent_mappings = {
            "alpha_generation": "BOOKMAKER",
            "arbitrage": "SCOUT",
            "algorithm_tracking": "HUNTER",
            "absence_detection": "GHOST",
            "writing": "THE_AUTHOR",
            "weight_optimization": "STRINGS",
            "risk": "RiskAgent",
            "execution": "ExecutionAgent",
            "research": "ResearchAgent",
        }
        return agent_mappings.get(task_type, "HoagsAgent")
    
    def _generate_creative_approach(self, task: Dict) -> Optional[str]:
        """Generate a creative approach for the task"""
        task_type = task.get("type", "")
        
        approaches = {
            "alpha_generation": "Apply inversion: What would make us lose money? Avoid that.",
            "risk": "Use OODA loop: Observe market state, Orient to regime, Decide on hedges, Act quickly",
            "execution": "Apply military flanking: Don't compete on price, compete on timing/venue",
        }
        return approaches.get(task_type)
    
    def _allocate_resources(self, task: Dict) -> List[str]:
        """Allocate resources for a task"""
        priority = task.get("priority", 5)
        if priority <= 2:
            return ["full_compute", "priority_data", "real_time_feeds"]
        elif priority <= 5:
            return ["standard_compute", "batch_data"]
        return ["minimal_compute"]
    
    def _select_best_framework(self, agent: str, context: Dict = None) -> CreativeFramework:
        """Select best creative framework for an agent"""
        agent_frameworks = {
            "GHOST": CreativeFramework.PSYCHOLOGY,  # Absence = behavioral bias
            "BOOKMAKER": CreativeFramework.GAME_THEORY,
            "SCOUT": CreativeFramework.BEHAVIORAL_ECONOMICS,
            "HUNTER": CreativeFramework.MILITARY_STRATEGY,
        }
        return agent_frameworks.get(agent, CreativeFramework.FIRST_PRINCIPLES)
    
    def _generate_improvement(
        self,
        agent: str,
        framework: CreativeFramework,
        context: Dict = None
    ) -> Dict:
        """Generate an improvement idea"""
        import random
        
        framework_ideas = {
            CreativeFramework.PSYCHOLOGY: {
                "type": "new_skill",
                "description": f"Add cognitive bias detection to {agent}",
                "rationale": "Markets are driven by human psychology; detecting biases provides edge",
                "expected_benefit": "Earlier detection of sentiment shifts",
                "implementation": "Add bias scoring module with trained classifiers"
            },
            CreativeFramework.GAME_THEORY: {
                "type": "enhanced_capability",
                "description": f"Add Nash equilibrium analysis to {agent}",
                "rationale": "Understanding opponent strategies reveals optimal responses",
                "expected_benefit": "Better prediction of institutional behavior",
                "implementation": "Implement payoff matrix analysis for key market participants"
            },
            CreativeFramework.INVERSION: {
                "type": "novel_approach",
                "description": f"Add 'failure mode analysis' to {agent}",
                "rationale": "Knowing how we fail helps us avoid failure",
                "expected_benefit": "Reduced drawdowns, better risk management",
                "implementation": "Pre-mortem analysis before every major decision"
            }
        }
        
        return framework_ideas.get(framework, {
            "type": "enhancement",
            "description": f"General improvement for {agent}",
            "rationale": "Continuous improvement",
            "expected_benefit": "Better performance",
            "implementation": "Incremental updates"
        })
    
    def _assess_novelty(self, approach: str, problem: str) -> float:
        """Assess how novel an approach is"""
        import random
        return random.uniform(0.5, 0.95)
    
    def _brainstorm_custom(
        self,
        framework: CreativeFramework,
        problem: str,
        context: Dict = None
    ) -> List[Dict]:
        """Brainstorm custom ideas"""
        return [{
            "approach": f"Custom {framework.value} application to {problem}",
            "applied_to": problem,
            "novelty_score": 0.8,
            "feasibility": "medium"
        }]
    
    def _identify_synergies(self, agents: List[str]) -> List[str]:
        """Identify creative synergies between agents"""
        synergies = []
        if "GHOST" in agents and "HUNTER" in agents:
            synergies.append("GHOST absence detection + HUNTER algorithm knowledge = predictive edge")
        if "BOOKMAKER" in agents and "SCOUT" in agents:
            synergies.append("BOOKMAKER alpha + SCOUT arbitrage = execution optimization")
        return synergies
    
    def _generate_summary(self, improvements: List[AgentImprovement]) -> str:
        """Generate summary for THE_AUTHOR"""
        return f"ORCHESTRATOR proposed {len(improvements)} improvements using creative frameworks including psychology, game theory, and inversion thinking."
    
    def _notify_author(self, improvement: AgentImprovement):
        """Notify THE_AUTHOR of a new improvement"""
        self.logger.info(f"ORCHESTRATOR → THE_AUTHOR: New improvement for {improvement.target_agent}")
    
    def log_action(self, action: str, description: str):
        self.logger.info(f"[ORCHESTRATOR] {action}: {description}")
    
    # =========================================================================
    # TASK HANDLERS
    # =========================================================================
    
    def _handle_orchestrate(self, params: Dict) -> Dict:
        task = params.get("task", params)
        agents = params.get("available_agents")
        assignment = self.orchestrate(task, agents)
        return {"status": "success", "assignment": assignment.to_dict()}
    
    def _handle_improve_agent(self, params: Dict) -> Dict:
        agent = params.get("target_agent", "")
        framework = CreativeFramework(params.get("framework", "first_principles")) if params.get("framework") else None
        improvement = self.improve_agent(agent, framework, params.get("context"))
        return {"status": "success", "improvement": improvement.to_dict()}
    
    def _handle_apply_framework(self, params: Dict) -> Dict:
        framework = CreativeFramework(params.get("framework", "inversion"))
        problem = params.get("problem", "")
        result = self.apply_creative_framework(framework, problem, params.get("context"))
        return {"status": "success", "result": result}
    
    def _handle_coordinate(self, params: Dict) -> Dict:
        task = params.get("task", {})
        agents = params.get("agents", [])
        coordination = self.coordinate_resources(task, agents)
        return {"status": "success", "coordination": coordination}
    
    def _handle_brief_author(self, params: Dict) -> Dict:
        brief = self.brief_author()
        return {"status": "success", "brief": brief}
    
    def _handle_get_creative(self, params: Dict) -> Dict:
        frameworks = [f.value for f in CreativeFramework]
        return {
            "status": "success",
            "frameworks": frameworks,
            "applications": {f.value: self.CREATIVE_APPLICATIONS.get(f, {}) for f in CreativeFramework}
        }
    
    def _handle_route_task(self, params: Dict) -> Dict:
        task_type = params.get("task_type", "")
        best_agent = self._select_best_agent(task_type)
        return {"status": "success", "recommended_agent": best_agent}
    
    def _handle_get_improvements(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "proposed": len(self.improvements_proposed),
            "implemented": len(self.improvements_implemented),
            "recent": [i.to_dict() for i in self.improvements_proposed[-10:]]
        }
    
    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}


# Singleton
_orchestrator_instance: Optional[OrchestratorAgent] = None

def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance

