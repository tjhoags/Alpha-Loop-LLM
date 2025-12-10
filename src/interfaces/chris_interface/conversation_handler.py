"""
================================================================================
CONVERSATION HANDLER - Natural Language Processing for Agent Communication
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

This module provides intelligent conversation handling capabilities:
- Intent recognition
- Context management
- Multi-turn dialogue support
- Proactive suggestions

================================================================================
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(Enum):
    """User intents that the system can recognize"""
    GET_STATUS = "get_status"
    GET_NAV = "get_nav"
    GET_REPORT = "get_report"
    GET_TAX_STATUS = "get_tax_status"
    GET_AUDIT_STATUS = "get_audit_status"
    REQUEST_ACTION = "request_action"
    ASK_QUESTION = "ask_question"
    PROVIDE_FEEDBACK = "provide_feedback"
    SCHEDULE_MEETING = "schedule_meeting"
    ESCALATE_ISSUE = "escalate_issue"
    GREETING = "greeting"
    FAREWELL = "farewell"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Result of intent parsing"""
    intent: Intent
    confidence: float
    entities: Dict[str, Any]
    suggested_agent: str
    suggested_action: str


class ConversationHandler:
    """
    Handles natural language understanding and conversation flow
    for interactions with fund operations agents.
    """

    # Intent patterns
    INTENT_PATTERNS = {
        Intent.GREETING: [
            r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
            r"\bhow are you\b",
        ],
        Intent.FAREWELL: [
            r"\b(bye|goodbye|see you|talk later|thanks|thank you)\b",
        ],
        Intent.GET_STATUS: [
            r"\b(status|update|how are things|what's happening|overview)\b",
            r"\bgive me (a |an )?(status|update)\b",
        ],
        Intent.GET_NAV: [
            r"\b(nav|net asset value|fund value|aum)\b",
            r"\bhow much (is|are) (the fund|we) worth\b",
        ],
        Intent.GET_REPORT: [
            r"\b(report|statement|summary)\b",
            r"\b(send|email|generate|prepare|give) me\b",
        ],
        Intent.GET_TAX_STATUS: [
            r"\b(tax|k-1|k1|irs|filing)\b",
            r"\b(estimated tax|quarterly tax)\b",
        ],
        Intent.GET_AUDIT_STATUS: [
            r"\b(audit|auditor|pbc|fieldwork)\b",
        ],
        Intent.REQUEST_ACTION: [
            r"\b(calculate|compute|run|execute|process)\b",
            r"\b(please|can you|could you)\b.*\b(do|make|create|prepare)\b",
        ],
        Intent.ASK_QUESTION: [
            r"\b(what|when|where|who|why|how|which)\b",
            r"\?$",
        ],
        Intent.ESCALATE_ISSUE: [
            r"\b(urgent|emergency|critical|important|asap|immediately)\b",
            r"\b(problem|issue|error|wrong|mistake)\b",
        ],
    }

    # Entity patterns
    ENTITY_PATTERNS = {
        "fund_id": r"(ALC_\w+|fund\s*\d+)",
        "investor_id": r"(LP\d+|investor\s*\d+)",
        "date": r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|today|yesterday|this month|last month|q[1-4]|Q[1-4])",
        "amount": r"\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d+)?\s*(million|m|thousand|k)?",
        "period": r"(q[1-4]|Q[1-4]|monthly|quarterly|annual|ytd|mtd)\s*-?\s*\d{4}?",
        "report_type": r"(nav pack|capital account|performance|financial statement|k-1|k1|form pf|adv|13f)",
    }

    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.conversation_turns: List[Dict] = []

    def parse_message(self, message: str) -> ParsedIntent:
        """
        Parse a user message to extract intent and entities.

        Args:
            message: The user's message

        Returns:
            ParsedIntent with recognized intent, entities, and suggestions
        """
        message_lower = message.lower().strip()

        # Recognize intent
        intent, confidence = self._recognize_intent(message_lower)

        # Extract entities
        entities = self._extract_entities(message)

        # Determine suggested agent and action
        agent, action = self._suggest_routing(intent, entities, message_lower)

        parsed = ParsedIntent(
            intent=intent,
            confidence=confidence,
            entities=entities,
            suggested_agent=agent,
            suggested_action=action
        )

        # Track conversation turn
        self.conversation_turns.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "parsed": {
                "intent": intent.value,
                "confidence": confidence,
                "entities": entities,
                "agent": agent,
                "action": action
            }
        })

        return parsed

    def _recognize_intent(self, message: str) -> Tuple[Intent, float]:
        """Recognize the intent from message"""
        scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[intent] = score

        if not scores:
            return Intent.UNKNOWN, 0.3

        # Get highest scoring intent
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] * 0.4, 0.95)

        return best_intent, confidence

    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message"""
        entities = {}

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches

        return entities

    def _suggest_routing(
        self,
        intent: Intent,
        entities: Dict[str, Any],
        message: str
    ) -> Tuple[str, str]:
        """Suggest which agent and action to route to"""

        # CPA indicators
        cpa_keywords = ["tax", "k-1", "k1", "audit", "form pf", "adv", "13f",
                       "regulatory", "compliance", "firm p&l", "pnl", "deadline"]

        is_cpa = any(kw in message for kw in cpa_keywords)

        # Default agent
        agent = "CPA" if is_cpa else "SANTAS_HELPER"

        # Determine action based on intent
        action_map = {
            Intent.GET_STATUS: "report_to_chris",
            Intent.GET_NAV: "calculate_nav",
            Intent.GET_REPORT: "generate_lp_report" if not is_cpa else "report_to_chris",
            Intent.GET_TAX_STATUS: "get_deadlines",
            Intent.GET_AUDIT_STATUS: "coordinate_audit",
            Intent.REQUEST_ACTION: "run_daily_operations" if not is_cpa else "get_status",
            Intent.ASK_QUESTION: "report_to_chris",
            Intent.ESCALATE_ISSUE: "report_to_chris",
            Intent.GREETING: "report_to_chris",
            Intent.FAREWELL: "get_status",
            Intent.UNKNOWN: "report_to_chris"
        }

        action = action_map.get(intent, "report_to_chris")

        # Refine based on entities
        if "report_type" in entities:
            report = entities["report_type"]
            if isinstance(report, str):
                report = report.lower()
                if "nav" in report:
                    agent = "SANTAS_HELPER"
                    action = "generate_nav_pack"
                elif "k-1" in report or "k1" in report:
                    agent = "CPA"
                    action = "generate_k1s"
                elif "form pf" in report:
                    agent = "CPA"
                    action = "file_form_pf"

        return agent, action

    def generate_proactive_suggestions(self) -> List[str]:
        """Generate proactive suggestions based on context and time"""
        suggestions = []
        now = datetime.now()

        # Time-based suggestions
        if now.hour < 9:
            suggestions.append("Would you like a morning status update?")

        # Month-end suggestions
        if now.day >= 28:
            suggestions.append("Month-end is approaching. Want a NAV preview?")

        # Quarter-end suggestions
        if now.month in [3, 6, 9, 12] and now.day >= 20:
            suggestions.append("Quarter-end is near. Review performance fees?")

        # Tax season suggestions (Jan-March)
        if now.month in [1, 2, 3]:
            suggestions.append("K-1 season - check preparation status?")

        return suggestions

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of conversation context"""
        return {
            "total_turns": len(self.conversation_turns),
            "recent_intents": [t["parsed"]["intent"] for t in self.conversation_turns[-5:]],
            "entities_discussed": self._aggregate_entities(),
            "suggestions": self.generate_proactive_suggestions()
        }

    def _aggregate_entities(self) -> Dict[str, List]:
        """Aggregate entities from conversation"""
        aggregated = {}
        for turn in self.conversation_turns:
            for key, value in turn["parsed"]["entities"].items():
                if key not in aggregated:
                    aggregated[key] = []
                if isinstance(value, list):
                    aggregated[key].extend(value)
                else:
                    aggregated[key].append(value)

        # Deduplicate
        for key in aggregated:
            aggregated[key] = list(set(aggregated[key]))

        return aggregated

