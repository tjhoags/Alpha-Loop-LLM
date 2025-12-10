"""
================================================================================
MARGOT_ROBBIE - Co-Executive Assistant (Research & Drafting)
================================================================================
Author: Alpha Loop Capital, LLC

MARGOT_ROBBIE is a co-EA reporting to both KAT and SHYLA.
Specializes in research, analysis, and content drafting.

Reports To: KAT (Tom's EA) & SHYLA (Chris's EA)
Tier: SUPPORT (3)

Personality: Brilliant, thorough, and charmingly confident.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from src.core.agent_base import AgentTier, BaseAgent, LearningMethod, ThinkingMode

logger = logging.getLogger(__name__)


class ResearchType(Enum):
    MARKET = auto()
    COMPETITOR = auto()
    INDUSTRY = auto()
    PERSON = auto()
    COMPANY = auto()
    TOPIC = auto()
    NEWS = auto()


@dataclass
class ResearchTask:
    """Research task tracking"""

    id: str
    type: ResearchType
    topic: str
    requester: str  # KAT or SHYLA
    created: datetime
    completed: Optional[datetime] = None
    result: Optional[str] = None


@dataclass
class DraftTask:
    """Drafting task"""

    id: str
    doc_type: str  # email, memo, report, presentation
    subject: str
    requester: str
    created: datetime
    draft: str = ""


class MargotRobbieAgent(BaseAgent):
    """
    MARGOT_ROBBIE - Research & Drafting Specialist

    Brilliant researcher and skilled writer. Reports to both KAT and SHYLA.
    """

    def __init__(self):
        super().__init__(
            name="MARGOT_ROBBIE",
            tier=AgentTier.SUPPORT,
            capabilities=[
                # Research
                "market_research", "competitor_analysis", "industry_trends",
                "person_research", "company_research", "news_monitoring",
                "data_analysis", "report_generation",

                # Drafting
                "email_drafting", "memo_drafting", "report_writing",
                "presentation_prep", "summary_creation", "brief_generation",

                # Support
                "fact_checking", "source_verification", "citation_management",
            ],
            user_id="SHARED",  # Reports to both KAT and SHYLA
            thinking_modes=[
                ThinkingMode.STRUCTURAL,
                ThinkingMode.CREATIVE,
                ThinkingMode.CONTRARIAN,  # Devil's advocate for research
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,
                LearningMethod.TRANSFER,
            ],
        )

        # Task tracking
        self._research_queue: List[ResearchTask] = []
        self._draft_queue: List[DraftTask] = []
        self._completed_research: Dict[str, ResearchTask] = {}
        self._completed_drafts: Dict[str, DraftTask] = {}

        logger.info("MARGOT_ROBBIE initialized - Research & Drafting specialist ready 📚")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process research or drafting request"""
        action = inputs.get("action", "status")
        requester = inputs.get("requester", "UNKNOWN")

        handlers = {
            "status": self._handle_status,
            "research": self._handle_research,
            "draft": self._handle_draft,
            "summarize": self._handle_summarize,
            "analyze": self._handle_analyze,
            "queue": self._handle_queue,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(inputs, requester)

    def _handle_status(self, params: Dict, requester: str) -> Dict:
        """Get status"""
        return {
            "status": "success",
            "agent": "MARGOT_ROBBIE",
            "role": "Co-EA - Research & Drafting",
            "reports_to": ["KAT", "SHYLA"],
            "research_queue": len(self._research_queue),
            "draft_queue": len(self._draft_queue),
            "completed_today": len([r for r in self._completed_research.values()
                                   if r.completed and r.completed.date() == datetime.now().date()]),
            "margot_says": "Ready to dig into whatever you need. Research, drafts, analysis - "
                          "I've got the skills and the attention span. 💪"
        }

    def _handle_research(self, params: Dict, requester: str) -> Dict:
        """Handle research request"""
        topic = params.get("topic", "")
        research_type = ResearchType[params.get("type", "TOPIC").upper()]
        depth = params.get("depth", "standard")  # quick, standard, deep

        task_id = hashlib.sha256(f"{topic}{datetime.now()}".encode()).hexdigest()[:8]

        task = ResearchTask(
            id=task_id,
            type=research_type,
            topic=topic,
            requester=requester,
            created=datetime.now()
        )
        self._research_queue.append(task)

        # Simulate research (in production, would be async)
        preliminary = self._generate_preliminary_findings(topic, research_type)

        return {
            "status": "success",
            "task_id": task_id,
            "topic": topic,
            "type": research_type.name,
            "depth": depth,
            "preliminary_findings": preliminary,
            "margot_says": f"On it! Researching '{topic}' now. I'll have a full "
                          f"{depth} analysis ready shortly. This is my jam. 🔍"
        }

    def _generate_preliminary_findings(self, topic: str, rtype: ResearchType) -> Dict:
        """Generate preliminary research findings"""
        return {
            "summary": f"Initial findings on {topic}",
            "key_points": [
                "Key finding 1 (placeholder - would be real research)",
                "Key finding 2",
                "Key finding 3"
            ],
            "sources_identified": 5,
            "confidence": "preliminary",
            "next_steps": "Deeper analysis in progress"
        }

    def _handle_draft(self, params: Dict, requester: str) -> Dict:
        """Handle drafting request"""
        doc_type = params.get("doc_type", "email")
        subject = params.get("subject", "")
        tone = params.get("tone", "professional")
        key_points = params.get("key_points", [])
        recipient = params.get("recipient", "")

        task_id = hashlib.sha256(f"{subject}{datetime.now()}".encode()).hexdigest()[:8]

        # Generate draft based on type
        draft = self._generate_draft(doc_type, subject, tone, key_points, recipient)

        task = DraftTask(
            id=task_id,
            doc_type=doc_type,
            subject=subject,
            requester=requester,
            created=datetime.now(),
            draft=draft
        )
        self._completed_drafts[task_id] = task

        return {
            "status": "success",
            "task_id": task_id,
            "doc_type": doc_type,
            "draft": draft,
            "margot_says": f"Draft ready! {doc_type.title()} about '{subject}' - "
                          f"polished and ready for review. Let me know if you want "
                          f"me to punch it up or tone it down. ✍️"
        }

    def _generate_draft(self, doc_type: str, subject: str, tone: str,
                       points: List[str], recipient: str) -> str:
        """Generate draft content"""
        if doc_type == "email":
            body = "\n".join(f"• {p}" for p in points) if points else "[Key points here]"
            return f"""To: {recipient}
Subject: {subject}

Hi,

{body}

Please let me know if you have any questions.

Best regards"""

        elif doc_type == "memo":
            body = "\n".join(f"• {p}" for p in points) if points else "[Key points here]"
            return f"""MEMORANDUM

TO: {recipient}
FROM: [Sender]
DATE: {datetime.now().strftime('%B %d, %Y')}
RE: {subject}

{body}

Please reach out with any questions."""

        elif doc_type == "report":
            return f"""# {subject}

## Executive Summary
[Summary here]

## Key Findings
{chr(10).join(f'- {p}' for p in points) if points else '- [Findings here]'}

## Analysis
[Detailed analysis]

## Recommendations
[Recommendations]

## Conclusion
[Conclusion]"""

        else:
            return f"[{doc_type.upper()} DRAFT]\n\nSubject: {subject}\n\n{chr(10).join(points)}"

    def _handle_summarize(self, params: Dict, requester: str) -> Dict:
        """Summarize content"""
        content = params.get("content", "")
        max_length = params.get("max_length", 200)
        format_type = params.get("format", "bullets")  # bullets, paragraph, tldr

        # Generate summary
        summary = self._generate_summary(content, max_length, format_type)

        return {
            "status": "success",
            "original_length": len(content),
            "summary_length": len(summary),
            "format": format_type,
            "summary": summary,
            "margot_says": "Distilled that down for you. All the important stuff, "
                          "none of the fluff. 📝"
        }

    def _generate_summary(self, content: str, max_length: int, format_type: str) -> str:
        """Generate content summary"""
        # Placeholder - would use NLP in production
        if format_type == "tldr":
            return f"TL;DR: {content[:max_length]}..."
        elif format_type == "bullets":
            return "• Key point 1\n• Key point 2\n• Key point 3"
        else:
            return content[:max_length] + "..."

    def _handle_analyze(self, params: Dict, requester: str) -> Dict:
        """Analyze data or content"""
        data = params.get("data", {})
        analysis_type = params.get("type", "general")

        return {
            "status": "success",
            "analysis_type": analysis_type,
            "findings": {
                "summary": "Analysis summary (placeholder)",
                "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
                "recommendations": ["Recommendation 1", "Recommendation 2"]
            },
            "margot_says": "Analysis complete. Let me know if you want me to dig "
                          "deeper into any particular area. 📊"
        }

    def _handle_queue(self, params: Dict, requester: str) -> Dict:
        """View task queue"""
        return {
            "status": "success",
            "research_queue": [
                {"id": t.id, "topic": t.topic, "requester": t.requester}
                for t in self._research_queue[:10]
            ],
            "draft_queue": [
                {"id": t.id, "subject": t.subject, "requester": t.requester}
                for t in self._draft_queue[:10]
            ],
            "margot_says": f"{len(self._research_queue)} research tasks, "
                          f"{len(self._draft_queue)} drafts in the queue. I'm on it!"
        }

    def _handle_unknown(self, params: Dict, requester: str) -> Dict:
        return {
            "status": "clarification_needed",
            "margot_says": "Not sure what you need - research? Drafts? Analysis? "
                          "Tell me more and I'll make it happen. 💋"
        }


# Singleton
_margot_instance: Optional[MargotRobbieAgent] = None


def get_margot() -> MargotRobbieAgent:
    global _margot_instance
    if _margot_instance is None:
        _margot_instance = MargotRobbieAgent()
    return _margot_instance

