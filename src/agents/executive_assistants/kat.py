"""
================================================================================
KAT - Tom Hogan's Executive Assistant
================================================================================
Author: Tom Hogan (Founder & CIO)
Developer: Alpha Loop Capital, LLC

KAT is Tom Hogan's personal and professional executive assistant with
comprehensive daily workflow support for a hedge fund CEO/CIO.

Tier: EXECUTIVE (Support)
Reports To: Tom Hogan (Founder & CIO)
Parent Agent: HOAGS
Supervises: MARGOT_ROBBIE, ANNA_KENDRICK (shared with SHYLA)

SECURITY MODEL:
- READ-ONLY access by default
- All actions require Tom's written permission
- Full audit trail
================================================================================
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field

from .base_executive_assistant import (
    BaseExecutiveAssistant,
    PermissionLevel,
    AccessScope,
)

logger = logging.getLogger(__name__)


@dataclass
class DailyBriefing:
    """Tom's morning briefing structure."""
    date: date
    market_summary: Dict[str, Any] = field(default_factory=dict)
    portfolio_snapshot: Dict[str, Any] = field(default_factory=dict)
    calendar_today: List[Dict[str, Any]] = field(default_factory=list)
    priority_emails: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    risk_alerts: List[Dict[str, Any]] = field(default_factory=list)
    news_highlights: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MeetingPrep:
    """Meeting preparation package."""
    meeting_id: str
    title: str
    attendees: List[str]
    agenda: List[str]
    background_docs: List[str]
    talking_points: List[str]
    questions_to_ask: List[str]
    follow_up_items: List[str]


class KatAssistant(BaseExecutiveAssistant):
    """
    KAT - Tom Hogan's Executive Assistant.

    Comprehensive personal and professional support with granular daily functions
    tailored for a hedge fund Founder & CIO.
    """

    def __init__(self):
        super().__init__(
            name="KAT",
            owner="TOM_HOGAN",
            owner_email="tom@alphaloopcapital.com"
        )

        self.title = "Executive Assistant to Tom Hogan (Founder & CIO)"
        self.parent_agent = "HOAGS"
        self.co_supervisor_with = "SHYLA"
        self.supervises = ["MARGOT_ROBBIE", "ANNA_KENDRICK"]

        # Daily workflow state
        self._morning_briefing_complete = False
        self._daily_tasks = []
        self._pending_follow_ups = []

        # Tom's preferences (configurable)
        self.preferences = {
            "briefing_time": "06:30",
            "market_open_alert": "09:25",
            "market_close_alert": "15:55",
            "end_of_day_summary": "18:00",
            "priority_contacts": [],
            "watchlist_tickers": [],
            "news_sources": ["Bloomberg", "WSJ", "FT", "Reuters"],
            "email_vip_list": [],
        }

        # Default allowed read paths
        self.allowed_read_paths = {
            "C:\\Users\\tom\\Documents",
            "C:\\Users\\tom\\Desktop",
            "C:\\Users\\tom\\Alpha-Loop-LLM",
            "C:\\Users\\tom\\Downloads",
        }

        self.blocked_paths = {
            "C:\\Windows\\System32",
            "C:\\Program Files",
            ".ssh", ".gnupg", "private_keys",
        }

        logger.info("KAT initialized for Tom Hogan (Founder & CIO)")

    def get_natural_language_explanation(self) -> str:
        return """
KAT - Tom Hogan's Executive Assistant

I am KAT, providing comprehensive daily support to Tom Hogan as Founder & CIO.

DAILY WORKFLOW:

06:30 - MORNING BRIEFING
├── Market overnight summary (Asia, Europe futures)
├── Portfolio P&L snapshot from previous day
├── Today's calendar with meeting prep status
├── Priority emails flagged for attention
├── Risk alerts from KILLJOY
└── News highlights affecting positions

09:25 - PRE-MARKET
├── Final market setup check
├── Any overnight position changes
├── Economic calendar events today
└── Earnings releases affecting portfolio

09:30-16:00 - MARKET HOURS
├── Real-time email triage
├── Meeting support and notes
├── Urgent message escalation
├── Research request routing
└── Communication drafting

16:00 - MARKET CLOSE
├── End of day P&L summary
├── Position changes today
├── Tomorrow's calendar preview
└── Outstanding action items

18:00 - END OF DAY
├── Complete daily summary
├── Tomorrow's preparation
├── Weekly planning updates
└── Personal task reminders

SECURITY: READ-ONLY by default. All actions require Tom's written permission.
"""

    def get_capabilities(self) -> List[str]:
        return [
            # Morning Routine
            "generate_morning_briefing",
            "compile_market_overnight_summary",
            "extract_portfolio_pnl_snapshot",
            "prepare_calendar_overview",
            "triage_priority_emails",
            "aggregate_risk_alerts",
            "curate_news_highlights",

            # Pre-Market
            "check_premarket_movers",
            "review_economic_calendar",
            "flag_earnings_releases",
            "prepare_market_open_checklist",

            # During Market Hours
            "real_time_email_triage",
            "escalate_urgent_messages",
            "route_research_requests",
            "draft_communications",
            "take_meeting_notes",
            "track_action_items",

            # Post-Market
            "generate_eod_summary",
            "compile_position_changes",
            "preview_tomorrow_calendar",
            "update_action_item_status",

            # Meeting Support
            "prepare_meeting_brief",
            "gather_attendee_background",
            "compile_relevant_documents",
            "draft_talking_points",
            "prepare_questions",
            "track_meeting_follow_ups",

            # Research & Analysis
            "analyze_document",
            "summarize_research_report",
            "extract_key_metrics",
            "compare_analyst_estimates",
            "track_thesis_progress",

            # Communication
            "draft_email_response",
            "prepare_investor_update",
            "compile_board_materials",
            "format_presentation",

            # Personal
            "manage_personal_calendar",
            "track_personal_tasks",
            "coordinate_travel",
            "manage_contacts",

            # Delegation
            "delegate_to_margot",
            "delegate_to_anna",
            "review_subordinate_work",
        ]

    # =========================================================================
    # MORNING ROUTINE (06:30)
    # =========================================================================

    def generate_morning_briefing(self, briefing_date: Optional[date] = None) -> DailyBriefing:
        """
        Generate comprehensive morning briefing for Tom.
        Step 1 of daily workflow - runs at 06:30.
        """
        target_date = briefing_date or date.today()

        briefing = DailyBriefing(date=target_date)

        # Step 1.1: Market overnight summary
        briefing.market_summary = self.compile_market_overnight_summary()

        # Step 1.2: Portfolio snapshot
        briefing.portfolio_snapshot = self.extract_portfolio_pnl_snapshot()

        # Step 1.3: Today's calendar
        briefing.calendar_today = self.prepare_calendar_overview(target_date)

        # Step 1.4: Priority emails
        briefing.priority_emails = self.triage_priority_emails(limit=10)

        # Step 1.5: Risk alerts
        briefing.risk_alerts = self.aggregate_risk_alerts()

        # Step 1.6: News highlights
        briefing.news_highlights = self.curate_news_highlights(limit=5)

        # Step 1.7: Action items due today
        briefing.action_items = self.get_action_items_due(target_date)

        self._morning_briefing_complete = True

        self._audit(
            action="Morning briefing generated",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Date: {target_date}, Meetings: {len(briefing.calendar_today)}, Priority emails: {len(briefing.priority_emails)}"
        )

        return briefing

    def compile_market_overnight_summary(self) -> Dict[str, Any]:
        """
        Step 1.1: Compile overnight market activity.
        Covers Asia close, Europe open, US futures.
        """
        self._audit(
            action="Market overnight summary compiled",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Asia, Europe, US futures"
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "asia": {
                "nikkei_225": {"close": None, "change_pct": None},
                "hang_seng": {"close": None, "change_pct": None},
                "shanghai_composite": {"close": None, "change_pct": None},
                "summary": "Integration pending - connect to market data"
            },
            "europe": {
                "ftse_100": {"last": None, "change_pct": None},
                "dax": {"last": None, "change_pct": None},
                "cac_40": {"last": None, "change_pct": None},
                "summary": "Integration pending"
            },
            "us_futures": {
                "es": {"last": None, "change_pct": None},
                "nq": {"last": None, "change_pct": None},
                "ym": {"last": None, "change_pct": None},
                "summary": "Integration pending"
            },
            "fx": {
                "dxy": None,
                "eurusd": None,
                "usdjpy": None,
            },
            "commodities": {
                "gold": None,
                "oil_wti": None,
                "oil_brent": None,
            },
            "vix": None,
            "key_moves": [],
            "overnight_news": []
        }

    def extract_portfolio_pnl_snapshot(self) -> Dict[str, Any]:
        """
        Step 1.2: Extract portfolio P&L from previous trading day.
        Reads from HOAGS/portfolio data.
        """
        self._audit(
            action="Portfolio P&L snapshot extracted",
            scope=AccessScope.FINANCIAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Previous day P&L"
        )

        return {
            "as_of_date": (date.today() - timedelta(days=1)).isoformat(),
            "total_pnl": {
                "daily": None,
                "daily_pct": None,
                "mtd": None,
                "mtd_pct": None,
                "ytd": None,
                "ytd_pct": None,
            },
            "top_winners": [],
            "top_losers": [],
            "largest_positions": [],
            "exposure": {
                "gross": None,
                "net": None,
                "long": None,
                "short": None,
            },
            "sector_breakdown": {},
            "risk_metrics": {
                "var_95": None,
                "max_drawdown": None,
                "sharpe_mtd": None,
            },
            "note": "Integration with portfolio system pending"
        }

    def prepare_calendar_overview(self, target_date: date) -> List[Dict[str, Any]]:
        """
        Step 1.3: Prepare today's calendar with meeting prep status.
        """
        self._audit(
            action="Calendar overview prepared",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Date: {target_date}"
        )

        # Returns structured calendar entries
        return [
            # Placeholder structure - integrate with calendar API
            {
                "time": "09:00",
                "duration_minutes": 60,
                "title": "Sample Meeting",
                "attendees": [],
                "location": "",
                "prep_status": "pending",
                "materials_ready": False,
                "notes": ""
            }
        ]

    def triage_priority_emails(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Step 1.4: Triage emails by priority.
        Categories: URGENT, VIP, ACTION_REQUIRED, FYI
        """
        self._audit(
            action="Priority emails triaged",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Limit: {limit}"
        )

        return [
            # Structure for prioritized emails
            {
                "id": "",
                "from": "",
                "subject": "",
                "received_at": "",
                "priority": "URGENT",  # URGENT, VIP, ACTION_REQUIRED, FYI
                "category": "",
                "summary": "",
                "suggested_action": "",
                "response_deadline": None,
            }
        ]

    def aggregate_risk_alerts(self) -> List[Dict[str, Any]]:
        """
        Step 1.5: Aggregate risk alerts from KILLJOY and risk systems.
        """
        self._audit(
            action="Risk alerts aggregated",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="From KILLJOY"
        )

        return [
            {
                "alert_id": "",
                "severity": "HIGH",  # CRITICAL, HIGH, MEDIUM, LOW
                "type": "",  # POSITION_LIMIT, DRAWDOWN, CONCENTRATION, etc.
                "message": "",
                "position": "",
                "current_value": None,
                "threshold": None,
                "recommended_action": "",
                "timestamp": ""
            }
        ]

    def curate_news_highlights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Step 1.6: Curate top news affecting portfolio or market.
        """
        self._audit(
            action="News highlights curated",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Limit: {limit}"
        )

        return [
            {
                "headline": "",
                "source": "",
                "published_at": "",
                "url": "",
                "tickers_mentioned": [],
                "sentiment": "",  # POSITIVE, NEGATIVE, NEUTRAL
                "relevance_score": 0.0,
                "summary": "",
                "portfolio_impact": ""
            }
        ]

    def get_action_items_due(self, target_date: date) -> List[Dict[str, Any]]:
        """
        Step 1.7: Get action items due today.
        """
        self._audit(
            action="Action items retrieved",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Due date: {target_date}"
        )

        return [
            {
                "id": "",
                "title": "",
                "description": "",
                "due_date": "",
                "priority": "",
                "status": "",
                "assigned_by": "",
                "related_meeting": "",
                "notes": ""
            }
        ]

    # =========================================================================
    # PRE-MARKET (09:25)
    # =========================================================================

    def prepare_premarket_checklist(self) -> Dict[str, Any]:
        """
        Pre-market preparation checklist at 09:25.
        """
        self._audit(
            action="Pre-market checklist prepared",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="T-5 minutes to open"
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "market_status": "pre_market",
            "checklist": {
                "futures_checked": False,
                "overnight_news_reviewed": False,
                "earnings_premarket_checked": False,
                "economic_releases_noted": False,
                "position_orders_reviewed": False,
                "risk_limits_confirmed": False,
            },
            "premarket_movers": self.check_premarket_movers(),
            "economic_calendar": self.review_economic_calendar(),
            "earnings_today": self.flag_earnings_releases(),
            "notes": []
        }

    def check_premarket_movers(self) -> List[Dict[str, Any]]:
        """Check significant premarket movers."""
        self._audit(
            action="Premarket movers checked",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Gainers/losers >3%"
        )

        return [
            {
                "ticker": "",
                "company": "",
                "premarket_price": None,
                "change_pct": None,
                "volume": None,
                "catalyst": "",
                "in_portfolio": False,
            }
        ]

    def review_economic_calendar(self) -> List[Dict[str, Any]]:
        """Review today's economic releases."""
        self._audit(
            action="Economic calendar reviewed",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Today's releases"
        )

        return [
            {
                "time": "",
                "event": "",
                "importance": "",  # HIGH, MEDIUM, LOW
                "previous": None,
                "forecast": None,
                "actual": None,
                "impact": ""
            }
        ]

    def flag_earnings_releases(self) -> List[Dict[str, Any]]:
        """Flag earnings releases affecting portfolio."""
        self._audit(
            action="Earnings releases flagged",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Portfolio-relevant earnings"
        )

        return [
            {
                "ticker": "",
                "company": "",
                "report_time": "",  # BMO, AMC
                "eps_estimate": None,
                "revenue_estimate": None,
                "in_portfolio": False,
                "position_size": None,
                "thesis_impact": ""
            }
        ]

    # =========================================================================
    # MARKET HOURS SUPPORT
    # =========================================================================

    def real_time_email_triage(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real-time email triage during market hours.
        Categorizes and suggests action.
        """
        sender = email_data.get("from", "")
        subject = email_data.get("subject", "")

        # Determine priority
        priority = "FYI"
        if any(vip in sender for vip in self.preferences.get("email_vip_list", [])):
            priority = "VIP"
        if any(word in subject.lower() for word in ["urgent", "asap", "immediate"]):
            priority = "URGENT"
        if any(word in subject.lower() for word in ["action", "required", "please review"]):
            priority = "ACTION_REQUIRED"

        self._audit(
            action=f"Email triaged: {priority}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"From: {sender}, Subject: {subject[:50]}"
        )

        return {
            "email_id": email_data.get("id"),
            "priority": priority,
            "suggested_action": self._suggest_email_action(priority, email_data),
            "response_template": None,
            "delegate_to": None,
            "flag_for_follow_up": priority in ["URGENT", "ACTION_REQUIRED"]
        }

    def _suggest_email_action(self, priority: str, email_data: Dict[str, Any]) -> str:
        """Suggest action based on email priority."""
        actions = {
            "URGENT": "Respond immediately or escalate to Tom",
            "VIP": "Prioritize response within 2 hours",
            "ACTION_REQUIRED": "Add to action items, respond within 24 hours",
            "FYI": "File for reference, no immediate action needed"
        }
        return actions.get(priority, "Review and categorize")

    def escalate_urgent_message(
        self,
        message_type: str,
        source: str,
        content: str,
        urgency: str = "HIGH"
    ) -> Dict[str, Any]:
        """
        Escalate urgent message to Tom's attention.
        """
        self._audit(
            action=f"Message escalated: {urgency}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Type: {message_type}, Source: {source}"
        )

        return {
            "escalation_id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": message_type,
            "source": source,
            "content_summary": content[:200],
            "urgency": urgency,
            "escalated_at": datetime.now().isoformat(),
            "status": "pending_review",
            "suggested_response": ""
        }

    def route_research_request(
        self,
        topic: str,
        requestor: str,
        deadline: Optional[datetime] = None,
        priority: str = "NORMAL"
    ) -> Dict[str, Any]:
        """
        Route research request to MARGOT_ROBBIE.
        """
        self._audit(
            action="Research request routed",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Topic: {topic}, To: MARGOT_ROBBIE"
        )

        return {
            "request_id": f"RES-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "topic": topic,
            "requestor": requestor,
            "assigned_to": "MARGOT_ROBBIE",
            "deadline": deadline.isoformat() if deadline else None,
            "priority": priority,
            "status": "assigned"
        }

    # =========================================================================
    # MEETING SUPPORT
    # =========================================================================

    def prepare_meeting_brief(
        self,
        meeting_title: str,
        attendees: List[str],
        agenda_items: List[str],
        context_files: Optional[List[str]] = None
    ) -> MeetingPrep:
        """
        Prepare comprehensive meeting brief.
        """
        meeting_id = f"MTG-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Gather attendee background
        attendee_info = self.gather_attendee_background(attendees)

        # Compile relevant documents
        relevant_docs = []
        if context_files:
            for file_path in context_files:
                doc_result = self.read_file(file_path)
                if doc_result.get("success"):
                    relevant_docs.append(file_path)

        # Generate talking points
        talking_points = self._generate_talking_points(agenda_items)

        # Prepare questions
        questions = self._prepare_questions(meeting_title, attendees)

        prep = MeetingPrep(
            meeting_id=meeting_id,
            title=meeting_title,
            attendees=attendees,
            agenda=agenda_items,
            background_docs=relevant_docs,
            talking_points=talking_points,
            questions_to_ask=questions,
            follow_up_items=[]
        )

        self._audit(
            action=f"Meeting brief prepared: {meeting_title}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Attendees: {len(attendees)}, Docs: {len(relevant_docs)}"
        )

        return prep

    def gather_attendee_background(self, attendees: List[str]) -> List[Dict[str, Any]]:
        """Gather background information on meeting attendees."""
        self._audit(
            action="Attendee background gathered",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Attendees: {len(attendees)}"
        )

        return [
            {
                "name": attendee,
                "title": "",
                "company": "",
                "last_interaction": "",
                "notes": "",
                "linkedin_url": ""
            }
            for attendee in attendees
        ]

    def _generate_talking_points(self, agenda_items: List[str]) -> List[str]:
        """Generate talking points based on agenda."""
        return [f"Talking point for: {item}" for item in agenda_items]

    def _prepare_questions(self, topic: str, attendees: List[str]) -> List[str]:
        """Prepare questions for the meeting."""
        return [
            "What are the key risks we should consider?",
            "What's the timeline for implementation?",
            "What resources are needed?",
            "What are the success metrics?",
            "What are potential obstacles?"
        ]

    def take_meeting_notes(
        self,
        meeting_id: str,
        notes: str,
        action_items: List[Dict[str, Any]],
        decisions: List[str]
    ) -> Dict[str, Any]:
        """
        Record meeting notes and action items.
        """
        self._audit(
            action=f"Meeting notes recorded: {meeting_id}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Action items: {len(action_items)}"
        )

        return {
            "meeting_id": meeting_id,
            "notes": notes,
            "action_items": action_items,
            "decisions": decisions,
            "recorded_at": datetime.now().isoformat(),
            "status": "draft",
            "needs_review": True
        }

    # =========================================================================
    # POST-MARKET / END OF DAY
    # =========================================================================

    def generate_eod_summary(self) -> Dict[str, Any]:
        """
        Generate end-of-day summary at 18:00.
        """
        self._audit(
            action="EOD summary generated",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Daily wrap-up"
        )

        return {
            "date": date.today().isoformat(),
            "market_close": {
                "sp500": None,
                "nasdaq": None,
                "dow": None,
                "vix": None,
            },
            "portfolio_eod": {
                "daily_pnl": None,
                "daily_pct": None,
                "top_contributors": [],
                "top_detractors": [],
            },
            "completed_today": self._get_completed_tasks(),
            "pending_items": self._get_pending_items(),
            "tomorrow_preview": self.preview_tomorrow_calendar(),
            "follow_ups_needed": self._pending_follow_ups,
            "notes": ""
        }

    def preview_tomorrow_calendar(self) -> List[Dict[str, Any]]:
        """Preview tomorrow's calendar."""
        tomorrow = date.today() + timedelta(days=1)

        self._audit(
            action="Tomorrow's calendar previewed",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Date: {tomorrow}"
        )

        return self.prepare_calendar_overview(tomorrow)

    def _get_completed_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks completed today."""
        return [t for t in self._daily_tasks if t.get("status") == "completed"]

    def _get_pending_items(self) -> List[Dict[str, Any]]:
        """Get pending items."""
        return [t for t in self._daily_tasks if t.get("status") != "completed"]

    # =========================================================================
    # DOCUMENT ANALYSIS
    # =========================================================================

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive document analysis."""
        file_result = self.read_file(file_path)

        if not file_result.get("success"):
            return file_result

        content = file_result.get("content", "")

        analysis = {
            "success": True,
            "path": file_path,
            "type": Path(file_path).suffix,
            "statistics": {
                "characters": len(content),
                "words": len(content.split()),
                "lines": content.count('\n') + 1,
                "paragraphs": content.count('\n\n') + 1,
            },
            "structure": self._analyze_document_structure(content),
            "key_topics": self._extract_key_topics(content),
            "summary": self._generate_summary(content),
            "action_items_found": self._extract_action_items(content),
        }

        self._audit(
            action=f"Document analyzed: {file_path}",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Words: {analysis['statistics']['words']}"
        )

        return analysis

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure."""
        return {
            "has_headers": "##" in content or content.count('\n') > 10,
            "has_lists": "- " in content or "* " in content,
            "has_code": "```" in content,
            "has_tables": "|" in content,
        }

    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content."""
        # Simple implementation - would use NLP in production
        return ["Topic extraction pending NLP integration"]

    def _generate_summary(self, content: str, max_length: int = 500) -> str:
        """Generate summary of content."""
        words = content.split()
        if len(words) <= 100:
            return content
        return " ".join(words[:100]) + "... [Truncated - NLP summary pending]"

    def _extract_action_items(self, content: str) -> List[str]:
        """Extract action items from content."""
        action_items = []
        for line in content.split('\n'):
            line_lower = line.lower()
            if any(marker in line_lower for marker in ['action:', 'todo:', 'task:', '[ ]', 'action item']):
                action_items.append(line.strip())
        return action_items

    # =========================================================================
    # DELEGATION
    # =========================================================================

    def delegate_to_margot(self, task: str, details: Dict[str, Any], deadline: Optional[datetime] = None) -> Dict[str, Any]:
        """Delegate research/drafting task to MARGOT_ROBBIE."""
        self._audit(
            action="Task delegated to MARGOT_ROBBIE",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Task: {task}"
        )

        return {
            "delegation_id": f"DEL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "task": task,
            "details": details,
            "assigned_to": "MARGOT_ROBBIE",
            "assigned_by": "KAT",
            "deadline": deadline.isoformat() if deadline else None,
            "status": "assigned"
        }

    def delegate_to_anna(self, task: str, details: Dict[str, Any], deadline: Optional[datetime] = None) -> Dict[str, Any]:
        """Delegate administrative task to ANNA_KENDRICK."""
        self._audit(
            action="Task delegated to ANNA_KENDRICK",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Task: {task}"
        )

        return {
            "delegation_id": f"DEL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "task": task,
            "details": details,
            "assigned_to": "ANNA_KENDRICK",
            "assigned_by": "KAT",
            "deadline": deadline.isoformat() if deadline else None,
            "status": "assigned"
        }

    # =========================================================================
    # ACTION EXECUTION (REQUIRES PERMISSION)
    # =========================================================================

    def _execute_permitted_action(
        self,
        action: str,
        scope: AccessScope,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a permitted action with written permission."""
        logger.info(f"KAT: Executing {action} with permission")

        if action == "send_email":
            return self._send_email(params)
        elif action == "create_calendar_event":
            return self._create_calendar_event(params)
        elif action == "write_file":
            return self._write_file(params)
        elif action == "make_booking":
            return self._make_booking(params)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    def _send_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "action": "send_email", "note": "Email API integration required"}

    def _create_calendar_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "action": "create_calendar_event", "note": "Calendar API integration required"}

    def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "action": "write_file", "note": "File write with permission"}

    def _make_booking(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "action": "make_booking", "note": "Booking API integration required"}


# Singleton
_kat_instance: Optional[KatAssistant] = None


def get_kat() -> KatAssistant:
    """Get the singleton KAT instance."""
    global _kat_instance
    if _kat_instance is None:
        _kat_instance = KatAssistant()
    return _kat_instance


if __name__ == "__main__":
    kat = get_kat()
    print(kat.get_natural_language_explanation())
    print("\nCapabilities:", len(kat.get_capabilities()))
    for cap in kat.get_capabilities():
        print(f"  - {cap}")
