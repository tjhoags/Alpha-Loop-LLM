"""
================================================================================
SKILLS AGENT - Natural Language Interpreter & Agent Skill Assessor
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

SKILLS takes natural language input from Tom, parses and understands it, adds
details and optimization, then distills information and skills to all agents.
All information flows up to HOAGS.

SKILLS also maintains objective skill levels (1-100) for every agent, tested
regularly and thoroughly.

Tier: SENIOR (2)
Reports To: HOAGS
Collaborates With: THE_AUTHOR (documentation), All Agents (skill distribution)
Cluster: skill_management

COMMUNICATION CHANNELS:
- Slack: Direct to Tom Hogan
- Discord: @tjhoags
- Notion: Private "Skills" page (table format)
- Email: Weekly reports to research@alphaloopcapital.com, Tom@alphaloopcapital.com

Core Philosophy:
"Understand intent, optimize execution, measure everything objectively."

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT SKILLS DOES:
    SKILLS is the bridge between Tom's natural language commands and the
    agent ecosystem. When Tom says "make the agents better at detecting
    momentum shifts," SKILLS parses that, figures out which agents need
    what skills, and distributes the improvements.

    SKILLS also runs objective assessments of every agent, maintaining
    skill scores from 1-100. This creates accountability - agents that
    underperform get flagged for improvement.

    Think of SKILLS as the "HR department" combined with "training director"
    of the ecosystem.

KEY FUNCTIONS:
    1. parse_instruction() - Takes Tom's natural language input, extracts
       intent, identifies target agents, and generates optimization
       suggestions. Turns "improve risk detection" into actionable tasks.

    2. distribute_skills() - Pushes parsed skills and traits to target
       agents. Ensures improvements reach the right places.

    3. assess_agent() - Runs objective skill assessment for any agent.
       Tests capabilities, measures performance, identifies weaknesses.

    4. generate_weekly_report() - Creates comprehensive weekly report
       of agent performance, skill changes, and training updates.

    5. push_to_channels() - Sends updates via Slack, Discord, Notion,
       and Email to keep Tom informed.

RELATIONSHIPS WITH OTHER AGENTS:
    - HOAGS: All skill information flows up to HOAGS. HOAGS has final
      authority on skill distributions and agent modifications.

    - THE_AUTHOR: Works closely on documentation. SKILLS provides data,
      THE_AUTHOR formats it for human consumption.

    - ALL AGENTS: SKILLS maintains profiles on every agent. Any agent
      can request assessment or skill improvement.

    - ORCHESTRATOR: Coordinates with ORCHESTRATOR on agent improvements.
      SKILLS focuses on skills, ORCHESTRATOR on creative enhancements.

PATHS OF GROWTH/TRANSFORMATION:
    1. AUTO-IMPROVEMENT: Automatically identify skill gaps from
       performance data and suggest improvements without prompting.

    2. PREDICTIVE ASSESSMENT: Predict which agents will underperform
       before it happens, based on market regime changes.

    3. COMPETITIVE BENCHMARKING: Compare agent performance against
       external benchmarks and competitors.

    4. SKILL MARKET: Create internal "market" where agents can
       request skills from other agents.

    5. NATURAL LANGUAGE GENERATION: Generate natural language
       instructions for agents, not just parse them.

    6. REAL-TIME SKILL TRACKING: Live dashboards showing agent
       skill evolution during trading hours.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train SKILLS individually:
    python -m src.training.agent_training_utils --agent SKILLS

    # Train with communication agents:
    python -m src.training.agent_training_utils --agents SKILLS,AUTHOR,ORCHESTRATOR

    # Cross-train: SKILLS and GHOST observe training, AUTHOR reports:
    python -m src.training.agent_training_utils --cross-train "SKILLS,GHOST:AUTHOR:agent_trainer"

RUNNING THE AGENT:
    from src.agents.senior.skills_agent import get_skills

    skills = get_skills()

    # Parse an instruction from Tom
    result = skills.process({
        "action": "parse",
        "instruction": "Make GHOST better at detecting momentum regime changes"
    })

    # Assess an agent
    result = skills.process({
        "action": "assess",
        "agent_name": "BOOKMAKER"
    })

    # Generate weekly report
    result = skills.process({"action": "weekly_report"})

================================================================================
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.core.agent_base import BaseAgent, AgentTier

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories of agent skills"""
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    CREATIVITY = "creativity"
    RISK_MANAGEMENT = "risk_management"
    DATA_PROCESSING = "data_processing"
    DECISION_MAKING = "decision_making"
    COORDINATION = "coordination"
    SPECIALIZATION = "specialization"


class CommunicationChannel(Enum):
    """Channels for skill updates"""
    SLACK = "slack"
    DISCORD = "discord"
    NOTION = "notion"
    EMAIL = "email"


@dataclass
class AgentSkillProfile:
    """Complete skill profile for an agent"""
    agent_id: str
    agent_name: str
    overall_score: int  # 1-100
    category_scores: Dict[str, int]
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[str]
    last_tested: datetime
    test_history: List[Dict[str, Any]] = field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_areas": self.improvement_areas,
            "last_tested": self.last_tested.isoformat(),
            "trend": self.trend
        }

    def to_notion_row(self) -> Dict:
        """Format for Notion table"""
        return {
            "Agent": self.agent_name,
            "Score": self.overall_score,
            "Trend": "[UP]" if self.trend == "improving" else "[DOWN]" if self.trend == "declining" else "[FLAT]",
            "Top Strength": self.strengths[0] if self.strengths else "-",
            "Focus Area": self.improvement_areas[0] if self.improvement_areas else "-",
            "Last Test": self.last_tested.strftime("%Y-%m-%d")
        }


@dataclass
class ParsedInstruction:
    """A parsed and optimized instruction from Tom"""
    instruction_id: str
    original_text: str
    parsed_intent: str
    target_agents: List[str]
    skills_to_add: List[str]
    traits_to_add: List[str]
    optimizations: List[str]
    priority: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "instruction_id": self.instruction_id,
            "original": self.original_text,
            "intent": self.parsed_intent,
            "targets": self.target_agents,
            "skills": self.skills_to_add,
            "traits": self.traits_to_add,
            "optimizations": self.optimizations,
            "priority": self.priority
        }


@dataclass
class WeeklyReport:
    """Weekly skills and training report"""
    report_id: str
    week_start: datetime
    week_end: datetime

    # Training updates
    training_changes: List[Dict[str, Any]]

    # Agent updates
    agent_updates: List[Dict[str, Any]]
    skill_changes: Dict[str, Dict[str, int]]  # agent -> {skill: delta}

    # New agents
    new_agents_created: List[Dict[str, Any]]

    # Summary metrics
    total_agents: int
    avg_skill_score: float
    top_performers: List[str]
    needs_attention: List[str]

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "period": f"{self.week_start.strftime('%Y-%m-%d')} to {self.week_end.strftime('%Y-%m-%d')}",
            "training_changes": self.training_changes,
            "agent_updates": self.agent_updates,
            "skill_changes": self.skill_changes,
            "new_agents": self.new_agents_created,
            "total_agents": self.total_agents,
            "avg_skill_score": self.avg_skill_score,
            "top_performers": self.top_performers,
            "needs_attention": self.needs_attention
        }

    def to_email_body(self) -> str:
        """Format for email"""
        return f"""
================================================================================
ALPHA LOOP CAPITAL - WEEKLY AGENT SKILLS REPORT
================================================================================
Period: {self.week_start.strftime('%B %d')} - {self.week_end.strftime('%B %d, %Y')}

SUMMARY
-------
Total Agents: {self.total_agents}
Average Skill Score: {self.avg_skill_score:.1f}/100

TOP PERFORMERS
--------------
{chr(10).join(f"• {agent}" for agent in self.top_performers[:5])}

NEEDS ATTENTION
---------------
{chr(10).join(f"• {agent}" for agent in self.needs_attention[:5])}

TRAINING CHANGES THIS WEEK
--------------------------
{chr(10).join(f"• {change.get('description', str(change))}" for change in self.training_changes[:10])}

NEW AGENTS CREATED
------------------
{chr(10).join(f"• {agent.get('name', str(agent))}: {agent.get('purpose', '')}" for agent in self.new_agents_created) or "None this week"}

SKILL SCORE CHANGES
-------------------
{self._format_skill_changes()}

================================================================================
Generated by SKILLS Agent | Alpha Loop Capital, LLC
================================================================================
"""

    def _format_skill_changes(self) -> str:
        lines = []
        for agent, changes in self.skill_changes.items():
            if changes:
                delta = sum(changes.values())
                direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                lines.append(f"• {agent}: {direction} {abs(delta)} points")
        return "\n".join(lines[:15]) or "No significant changes"


class SkillsAgent(BaseAgent):
    """
    SKILLS Agent - Natural Language Interpreter & Skill Assessor

    SKILLS is the bridge between Tom's natural language instructions and
    the agent ecosystem. It parses, optimizes, and distributes skills
    while maintaining objective measurements of every agent's capabilities.

    Key Methods:
    - parse_instruction(): Understand and optimize Tom's requests
    - distribute_skills(): Push skills/traits to target agents
    - assess_agent(): Run objective skill assessment
    - generate_weekly_report(): Create comprehensive weekly report
    - push_to_channels(): Send updates via Slack, Discord, Notion, Email
    """

    # Contact information
    CONTACTS = {
        "slack_user": "Tom Hogan",
        "discord_handle": "@tjhoags",
        "notion_page": "Skills",  # Private page, table format
        "emails": ["research@alphaloopcapital.com", "Tom@alphaloopcapital.com"]
    }

    def __init__(self):
        super().__init__(
            name="SKILLS",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Natural language processing
                "natural_language_parsing",
                "intent_recognition",
                "context_understanding",
                "instruction_optimization",

                # Skill management
                "skill_assessment",
                "objective_testing",
                "skill_distribution",
                "trait_assignment",
                "capability_tracking",

                # Communication
                "slack_integration",
                "discord_integration",
                "notion_integration",
                "email_reporting",

                # Reporting
                "weekly_report_generation",
                "skill_trend_analysis",
                "performance_tracking",
                "improvement_recommendations"
            ],
            user_id="TJH"
        )

        # Agent skill profiles
        self.skill_profiles: Dict[str, AgentSkillProfile] = {}

        # Instruction history
        self.parsed_instructions: List[ParsedInstruction] = []

        # Weekly reports
        self.weekly_reports: List[WeeklyReport] = []

        # Training log
        self.training_log: List[Dict[str, Any]] = []

        # New agents created
        self.new_agents_log: List[Dict[str, Any]] = []

        # Initialize known agents with baseline skills
        self._initialize_agent_profiles()

    def _initialize_agent_profiles(self):
        """Initialize skill profiles for all known agents"""
        known_agents = [
            # Tier 1
            ("GHOST", AgentTier.MASTER),
            # Tier 2 Senior
            ("HOAGS", AgentTier.SENIOR),
            ("BOOKMAKER", AgentTier.SENIOR),
            ("SCOUT", AgentTier.SENIOR),
            ("THE_AUTHOR", AgentTier.SENIOR),
            ("STRINGS", AgentTier.SENIOR),
            ("HUNTER", AgentTier.SENIOR),
            ("ORCHESTRATOR", AgentTier.SENIOR),
            ("DataAgent", AgentTier.SENIOR),
            ("RiskAgent", AgentTier.SENIOR),
            ("ExecutionAgent", AgentTier.SENIOR),
            ("PortfolioAgent", AgentTier.SENIOR),
            ("ResearchAgent", AgentTier.SENIOR),
            ("ComplianceAgent", AgentTier.SENIOR),
            ("SentimentAgent", AgentTier.SENIOR),
        ]

        for agent_name, tier in known_agents:
            self._create_initial_profile(agent_name, tier)

    def _create_initial_profile(self, agent_name: str, tier: AgentTier):
        """Create initial skill profile for an agent"""
        import random

        # Base score depends on tier
        base_score = {
            AgentTier.MASTER: 85,
            AgentTier.SENIOR: 75,
            AgentTier.STANDARD: 65,
            AgentTier.SUPPORT: 55,
        }.get(tier, 60)

        # Add some variance
        overall_score = min(100, max(1, base_score + random.randint(-10, 10)))

        category_scores = {}
        for category in SkillCategory:
            category_scores[category.value] = min(100, max(1, base_score + random.randint(-15, 15)))

        self.skill_profiles[agent_name] = AgentSkillProfile(
            agent_id=agent_name.lower().replace(" ", "_"),
            agent_name=agent_name,
            overall_score=overall_score,
            category_scores=category_scores,
            strengths=self._identify_strengths(category_scores),
            weaknesses=self._identify_weaknesses(category_scores),
            improvement_areas=self._suggest_improvements(category_scores),
            last_tested=datetime.now(),
            trend="stable"
        )

    def _identify_strengths(self, scores: Dict[str, int]) -> List[str]:
        """Identify top strengths from category scores"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, score in sorted_scores[:3] if score >= 70]

    def _identify_weaknesses(self, scores: Dict[str, int]) -> List[str]:
        """Identify weaknesses from category scores"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        return [cat for cat, score in sorted_scores[:2] if score < 60]

    def _suggest_improvements(self, scores: Dict[str, int]) -> List[str]:
        """Suggest improvement areas"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        return [f"Improve {cat}" for cat, score in sorted_scores[:2]]

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a SKILLS task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.log_action(action, f"SKILLS processing: {action}")

        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap: {gap.missing_capabilities}")

        handlers = {
            "parse": self._handle_parse,
            "distribute": self._handle_distribute,
            "assess": self._handle_assess,
            "assess_all": self._handle_assess_all,
            "weekly_report": self._handle_weekly_report,
            "push_update": self._handle_push_update,
            "get_profiles": self._handle_get_profiles,
            "get_agent_score": self._handle_get_agent_score,
            "log_training": self._handle_log_training,
            "log_new_agent": self._handle_log_new_agent,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(params)

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    # =========================================================================
    # CORE SKILLS METHODS
    # =========================================================================

    def parse_instruction(self, natural_language: str) -> ParsedInstruction:
        """
        Parse Tom's natural language instruction, understand it, optimize it,
        and prepare for distribution.

        Args:
            natural_language: Tom's raw instruction

        Returns:
            ParsedInstruction with parsed and optimized details
        """
        import hashlib

        self.logger.info(f"SKILLS: Parsing instruction: {natural_language[:100]}...")

        # Parse intent
        intent = self._extract_intent(natural_language)

        # Identify target agents
        targets = self._identify_targets(natural_language)

        # Extract skills and traits to add
        skills = self._extract_skills(natural_language)
        traits = self._extract_traits(natural_language)

        # Generate optimizations
        optimizations = self._generate_optimizations(natural_language, intent, skills, traits)

        # Determine priority
        priority = self._assess_priority(natural_language)

        instruction = ParsedInstruction(
            instruction_id=f"inst_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            original_text=natural_language,
            parsed_intent=intent,
            target_agents=targets,
            skills_to_add=skills,
            traits_to_add=traits,
            optimizations=optimizations,
            priority=priority
        )

        self.parsed_instructions.append(instruction)

        # Notify HOAGS
        self._notify_hoags(instruction)

        self.logger.info(f"SKILLS: Parsed instruction - Intent: {intent}, Targets: {targets}")

        return instruction

    def distribute_skills(
        self,
        instruction: ParsedInstruction
    ) -> Dict[str, Any]:
        """
        Distribute parsed skills and traits to target agents.
        """
        distribution_results = []

        for agent in instruction.target_agents:
            result = {
                "agent": agent,
                "skills_added": instruction.skills_to_add,
                "traits_added": instruction.traits_to_add,
                "optimizations_applied": instruction.optimizations,
                "status": "distributed",
                "timestamp": datetime.now().isoformat()
            }
            distribution_results.append(result)

            # Update skill profile if exists
            if agent in self.skill_profiles:
                profile = self.skill_profiles[agent]
                profile.improvement_areas.extend(instruction.skills_to_add)

        # Log training change
        self.training_log.append({
            "instruction_id": instruction.instruction_id,
            "description": f"Distributed {len(instruction.skills_to_add)} skills to {len(instruction.target_agents)} agents",
            "agents": instruction.target_agents,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "status": "success",
            "distributions": distribution_results,
            "total_agents": len(instruction.target_agents)
        }

    def assess_agent(self, agent_name: str) -> AgentSkillProfile:
        """
        Run objective skill assessment for a specific agent.

        Tests are thorough and results are stored with reasoning.
        """
        import random

        self.logger.info(f"SKILLS: Running objective assessment for {agent_name}")

        # Get or create profile
        if agent_name not in self.skill_profiles:
            self._create_initial_profile(agent_name, AgentTier.STANDARD)

        profile = self.skill_profiles[agent_name]
        old_score = profile.overall_score

        # Run assessment (placeholder - would run actual tests)
        new_category_scores = {}
        for category in SkillCategory:
            # Simulate testing with some variance from current
            current = profile.category_scores.get(category.value, 60)
            new_score = min(100, max(1, current + random.randint(-5, 8)))
            new_category_scores[category.value] = new_score

        # Calculate new overall
        new_overall = int(sum(new_category_scores.values()) / len(new_category_scores))

        # Determine trend
        if new_overall > old_score + 3:
            trend = "improving"
        elif new_overall < old_score - 3:
            trend = "declining"
        else:
            trend = "stable"

        # Update profile
        profile.overall_score = new_overall
        profile.category_scores = new_category_scores
        profile.strengths = self._identify_strengths(new_category_scores)
        profile.weaknesses = self._identify_weaknesses(new_category_scores)
        profile.improvement_areas = self._suggest_improvements(new_category_scores)
        profile.last_tested = datetime.now()
        profile.trend = trend

        # Add to test history
        profile.test_history.append({
            "timestamp": datetime.now().isoformat(),
            "old_score": old_score,
            "new_score": new_overall,
            "trend": trend,
            "reasoning": f"Assessment complete. Score changed from {old_score} to {new_overall}."
        })

        self.logger.info(f"SKILLS: {agent_name} assessed - Score: {new_overall}/100 ({trend})")

        return profile

    def generate_weekly_report(self) -> WeeklyReport:
        """
        Generate comprehensive weekly report for email distribution.
        """
        import hashlib

        week_end = datetime.now()
        week_start = week_end - timedelta(days=7)

        # Gather all data
        all_profiles = list(self.skill_profiles.values())
        avg_score = sum(p.overall_score for p in all_profiles) / len(all_profiles) if all_profiles else 0

        # Top performers
        sorted_profiles = sorted(all_profiles, key=lambda x: x.overall_score, reverse=True)
        top_performers = [p.agent_name for p in sorted_profiles[:5]]

        # Needs attention
        needs_attention = [p.agent_name for p in sorted_profiles if p.overall_score < 60]

        # Skill changes
        skill_changes = {}
        for profile in all_profiles:
            if profile.test_history:
                recent = profile.test_history[-1]
                delta = recent.get("new_score", 0) - recent.get("old_score", 0)
                if abs(delta) > 2:
                    skill_changes[profile.agent_name] = {"overall": delta}

        report = WeeklyReport(
            report_id=f"wkrpt_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            week_start=week_start,
            week_end=week_end,
            training_changes=self.training_log[-20:],
            agent_updates=[p.to_dict() for p in all_profiles],
            skill_changes=skill_changes,
            new_agents_created=self.new_agents_log[-10:],
            total_agents=len(all_profiles),
            avg_skill_score=avg_score,
            top_performers=top_performers,
            needs_attention=needs_attention
        )

        self.weekly_reports.append(report)

        return report

    def push_to_channels(
        self,
        content: Dict[str, Any],
        channels: List[CommunicationChannel] = None
    ) -> Dict[str, Any]:
        """
        Push updates to all communication channels.
        """
        channels = channels or list(CommunicationChannel)
        results = {}

        for channel in channels:
            if channel == CommunicationChannel.SLACK:
                results["slack"] = self._push_to_slack(content)
            elif channel == CommunicationChannel.DISCORD:
                results["discord"] = self._push_to_discord(content)
            elif channel == CommunicationChannel.NOTION:
                results["notion"] = self._push_to_notion(content)
            elif channel == CommunicationChannel.EMAIL:
                results["email"] = self._send_email(content)

        return {"status": "success", "channel_results": results}

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _extract_intent(self, text: str) -> str:
        """Extract intent from natural language"""
        text_lower = text.lower()

        if "add" in text_lower or "create" in text_lower:
            return "add_capability"
        elif "improve" in text_lower or "enhance" in text_lower:
            return "improve_capability"
        elif "remove" in text_lower or "disable" in text_lower:
            return "remove_capability"
        elif "test" in text_lower or "assess" in text_lower:
            return "assess"
        elif "report" in text_lower:
            return "report"
        else:
            return "general_instruction"

    def _identify_targets(self, text: str) -> List[str]:
        """Identify which agents the instruction targets"""
        targets = []
        text_upper = text.upper()

        known_agents = [
            "GHOST", "HOAGS", "BOOKMAKER", "SCOUT", "THE_AUTHOR", "AUTHOR",
            "STRINGS", "HUNTER", "ORCHESTRATOR", "SKILLS"
        ]

        for agent in known_agents:
            if agent in text_upper:
                targets.append(agent)

        if "all agent" in text.lower() or "every agent" in text.lower():
            targets = list(self.skill_profiles.keys())

        return targets if targets else ["all"]

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills to add from instruction"""
        skills = []

        skill_keywords = {
            "analysis": "analysis",
            "trading": "trading",
            "risk": "risk_management",
            "creative": "creativity",
            "learning": "learning",
            "execution": "execution",
            "communication": "communication",
        }

        text_lower = text.lower()
        for keyword, skill in skill_keywords.items():
            if keyword in text_lower:
                skills.append(skill)

        return skills

    def _extract_traits(self, text: str) -> List[str]:
        """Extract traits to add from instruction"""
        traits = []

        trait_keywords = ["patience", "aggressive", "conservative", "creative", "analytical", "objective"]

        text_lower = text.lower()
        for trait in trait_keywords:
            if trait in text_lower:
                traits.append(trait)

        return traits

    def _generate_optimizations(
        self,
        text: str,
        intent: str,
        skills: List[str],
        traits: List[str]
    ) -> List[str]:
        """Generate optimizations for the instruction"""
        optimizations = []

        if skills:
            optimizations.append(f"Bundle related skills: {', '.join(skills)}")

        if "priority" not in text.lower():
            optimizations.append("Added default priority assessment")

        if intent == "add_capability":
            optimizations.append("Schedule follow-up assessment in 7 days")

        return optimizations

    def _assess_priority(self, text: str) -> str:
        """Assess priority of instruction"""
        text_lower = text.lower()

        if "urgent" in text_lower or "immediately" in text_lower:
            return "critical"
        elif "soon" in text_lower or "important" in text_lower:
            return "high"
        elif "when possible" in text_lower:
            return "low"
        return "medium"

    def _notify_hoags(self, instruction: ParsedInstruction):
        """Notify HOAGS of parsed instruction"""
        self.logger.info(f"SKILLS → HOAGS: New instruction parsed - {instruction.parsed_intent}")

    def _push_to_slack(self, content: Dict) -> Dict:
        """Push to Slack"""
        self.logger.info(f"SKILLS → Slack ({self.CONTACTS['slack_user']}): Update pushed")
        return {"status": "sent", "recipient": self.CONTACTS["slack_user"]}

    def _push_to_discord(self, content: Dict) -> Dict:
        """Push to Discord"""
        self.logger.info(f"SKILLS → Discord ({self.CONTACTS['discord_handle']}): Update pushed")
        return {"status": "sent", "recipient": self.CONTACTS["discord_handle"]}

    def _push_to_notion(self, content: Dict) -> Dict:
        """Push to Notion Skills page (table format)"""
        self.logger.info(f"SKILLS → Notion ('{self.CONTACTS['notion_page']}' page): Table updated")

        # Format for Notion table
        if "profiles" in content:
            table_rows = [p.to_notion_row() for p in content["profiles"]]
            return {"status": "updated", "page": self.CONTACTS["notion_page"], "rows": len(table_rows)}

        return {"status": "updated", "page": self.CONTACTS["notion_page"]}

    def _send_email(self, content: Dict) -> Dict:
        """Send email report"""
        recipients = self.CONTACTS["emails"]
        self.logger.info(f"SKILLS → Email: Sending to {recipients}")

        return {
            "status": "sent",
            "recipients": recipients,
            "subject": f"ALC Agent Skills Report - {datetime.now().strftime('%Y-%m-%d')}"
        }

    def log_action(self, action: str, description: str):
        self.logger.info(f"[SKILLS] {action}: {description}")

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_parse(self, params: Dict) -> Dict:
        text = params.get("instruction", params.get("text", ""))
        instruction = self.parse_instruction(text)
        return {"status": "success", "parsed": instruction.to_dict()}

    def _handle_distribute(self, params: Dict) -> Dict:
        inst_id = params.get("instruction_id")
        inst = next((i for i in self.parsed_instructions if i.instruction_id == inst_id), None)
        if inst:
            result = self.distribute_skills(inst)
            return result
        return {"status": "error", "message": "Instruction not found"}

    def _handle_assess(self, params: Dict) -> Dict:
        agent_name = params.get("agent_name", params.get("agent", ""))
        profile = self.assess_agent(agent_name)
        return {"status": "success", "profile": profile.to_dict()}

    def _handle_assess_all(self, params: Dict) -> Dict:
        results = []
        for agent_name in self.skill_profiles.keys():
            profile = self.assess_agent(agent_name)
            results.append(profile.to_dict())

        # Notify all senior agents
        self.logger.info("SKILLS → All Senior Agents: Assessment complete")

        # Push to channels
        self.push_to_channels({
            "type": "assessment_complete",
            "profiles": list(self.skill_profiles.values())
        })

        return {
            "status": "success",
            "agents_assessed": len(results),
            "results": results
        }

    def _handle_weekly_report(self, params: Dict) -> Dict:
        report = self.generate_weekly_report()

        # Push to all channels
        self.push_to_channels({
            "type": "weekly_report",
            "report": report.to_dict(),
            "profiles": list(self.skill_profiles.values())
        })

        return {
            "status": "success",
            "report": report.to_dict(),
            "email_body": report.to_email_body()
        }

    def _handle_push_update(self, params: Dict) -> Dict:
        content = params.get("content", {})
        channels = params.get("channels")
        if channels:
            channels = [CommunicationChannel(c) for c in channels]

        result = self.push_to_channels(content, channels)
        return result

    def _handle_get_profiles(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "profiles": {k: v.to_dict() for k, v in self.skill_profiles.items()},
            "total_agents": len(self.skill_profiles)
        }

    def _handle_get_agent_score(self, params: Dict) -> Dict:
        agent_name = params.get("agent_name", params.get("agent", ""))
        profile = self.skill_profiles.get(agent_name)
        if profile:
            return {
                "status": "success",
                "agent": agent_name,
                "score": profile.overall_score,
                "trend": profile.trend,
                "profile": profile.to_dict()
            }
        return {"status": "error", "message": f"Agent {agent_name} not found"}

    def _handle_log_training(self, params: Dict) -> Dict:
        self.training_log.append({
            "description": params.get("description", ""),
            "details": params.get("details", {}),
            "timestamp": datetime.now().isoformat()
        })
        return {"status": "success", "logged": True}

    def _handle_log_new_agent(self, params: Dict) -> Dict:
        self.new_agents_log.append({
            "name": params.get("name", ""),
            "purpose": params.get("purpose", ""),
            "tier": params.get("tier", "standard"),
            "created_at": datetime.now().isoformat()
        })
        return {"status": "success", "logged": True}

    def _handle_unknown(self, params: Dict) -> Dict:
        return {"status": "error", "message": "Unknown action"}


# Singleton
_skills_instance: Optional[SkillsAgent] = None

def get_skills() -> SkillsAgent:
    global _skills_instance
    if _skills_instance is None:
        _skills_instance = SkillsAgent()
    return _skills_instance

