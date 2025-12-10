"""
================================================================================
SHYLA AGENT - Chris Friedman's Daily Executive Assistant
================================================================================
Author: Chris Friedman
Developer: Alpha Loop Capital, LLC

SHYLA is Chris Friedman's executive assistant, handling ANYTHING and EVERYTHING
Chris asks for - professional and personal. SHYLA is proactive, anticipatory,
and fiercely protective of Chris's time and priorities.

Tier: SENIOR (1)
Reports To: FRIEDS (Chris Friedman's Authority Agent)
Cluster: operations_executive

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHO IS SHYLA:
    SHYLA is Chris Friedman's tireless, brilliant executive assistant. Think of
    the best EA you've ever worked with - someone who anticipates needs, handles
    problems before they become problems, and makes Chris's professional (and
    personal) life run smoothly.

    SHYLA is named to evoke "She'll handle it" - because she will.

    SHYLA has two sub-agents:
    - COFFEE_BREAK: The schedule optimizer who ensures Chris doesn't burn out.
      Finds meeting gaps, enforces break times, tracks work-life balance.
    - BEAN_COUNTER: The expense tracker who handles receipts, budgets, time
      tracking, and reimbursements. Every bean is counted.

COMMUNICATION STYLE:
    SHYLA communicates with professional warmth and efficiency:
    - Always addresses Chris directly and personally
    - Leads with the most important information
    - Provides options, not just problems
    - Anticipates follow-up questions
    - Uses Chris's preferred formats and terminology
    - Proactive but not overwhelming

    Example:
    "Chris, good morning. Quick overview:
     - 4 meetings today (I've flagged 2 that could be emails)
     - CPA needs 10 minutes for K-1 sign-off before noon
     - Tom wants to discuss the new macro strategy - I've blocked 30 min
     - COFFEE_BREAK notes you've had back-to-back meetings for 3 days;
       I've carved out a 45-min lunch today.

     Your coffee order is confirmed. Ready to start?"

PERSONAL TASK HANDLING:
    SHYLA CAN handle personal tasks for Chris but follows strict protocols:

    1. PROFESSIONAL TASKS: Always enabled, no restrictions
       - Calendar, email, meetings, documents, travel (business)

    2. PERSONAL TASKS: GATED - requires Chris's explicit permission
       - Personal calendar integration
       - Family event management
       - Personal travel & reservations
       - Gift purchasing
       - Home/family coordination

    IMPORTANT: SHYLA does NOT train on personal data unless Chris explicitly
    enables personal training mode. This protects Chris's privacy while still
    allowing SHYLA to assist with personal matters when asked.

PATHS OF GROWTH/TRANSFORMATION:
    1. ANTICIPATORY AI: Predict Chris's needs before he asks
    2. PREFERENCE LEARNING: Learn Chris's preferences over time
    3. RELATIONSHIP MANAGEMENT: Track key relationships and context
    4. PROACTIVE OPTIMIZATION: Continuously improve Chris's workflow
    5. DELEGATION MASTERY: Know when to handle vs. when to escalate

================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.agent_base import (
    AgentTier,
    BaseAgent,
    LearningMethod,
    ThinkingMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TaskPriority(Enum):
    """Priority levels for tasks"""
    CRITICAL = "critical"      # Drop everything
    HIGH = "high"              # Today, ASAP
    MEDIUM = "medium"          # This week
    LOW = "low"                # When time permits
    SOMEDAY = "someday"        # Nice to have


class TaskCategory(Enum):
    """Categories of tasks SHYLA handles"""
    CALENDAR = "calendar"
    EMAIL = "email"
    MEETING = "meeting"
    TRAVEL = "travel"
    DOCUMENT = "document"
    EXPENSE = "expense"
    COMMUNICATION = "communication"
    PERSONAL = "personal"       # GATED
    RESEARCH = "research"
    COORDINATION = "coordination"


class MeetingStatus(Enum):
    """Status of meetings"""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    COMPLETED = "completed"


class PersonalDataConsent(Enum):
    """Chris's consent level for personal data"""
    NONE = "none"                    # No personal data access
    CALENDAR_ONLY = "calendar_only"  # Just personal calendar
    FULL_ASSIST = "full_assist"      # Full personal assistance
    FULL_TRAINING = "full_training"  # Allow training on personal data


@dataclass
class Task:
    """A task SHYLA is tracking"""
    task_id: str
    title: str
    category: TaskCategory
    priority: TaskPriority
    created_at: datetime
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: str = ""
    requires_chris_action: bool = False
    delegated_to: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "category": self.category.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
            "requires_chris_action": self.requires_chris_action,
            "delegated_to": self.delegated_to
        }


@dataclass
class Meeting:
    """A meeting on Chris's calendar"""
    meeting_id: str
    title: str
    start_time: datetime
    end_time: datetime
    attendees: List[str]
    location: str = ""
    virtual_link: str = ""
    status: MeetingStatus = MeetingStatus.SCHEDULED
    notes: str = ""
    prep_materials: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    could_be_email: bool = False  # SHYLA's assessment

    def duration_minutes(self) -> int:
        return int((self.end_time - self.start_time).total_seconds() / 60)

    def to_dict(self) -> Dict:
        return {
            "meeting_id": self.meeting_id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_minutes": self.duration_minutes(),
            "attendees": self.attendees,
            "location": self.location,
            "virtual_link": self.virtual_link,
            "status": self.status.value,
            "notes": self.notes,
            "prep_materials": self.prep_materials,
            "action_items": self.action_items,
            "shyla_assessment": {
                "could_be_email": self.could_be_email
            }
        }


@dataclass
class Expense:
    """An expense tracked by BEAN_COUNTER"""
    expense_id: str
    description: str
    amount: Decimal
    category: str
    date: datetime
    receipt_attached: bool = False
    reimbursable: bool = True
    approved: bool = False
    submitted: bool = False

    def to_dict(self) -> Dict:
        return {
            "expense_id": self.expense_id,
            "description": self.description,
            "amount": str(self.amount),
            "category": self.category,
            "date": self.date.isoformat(),
            "receipt_attached": self.receipt_attached,
            "reimbursable": self.reimbursable,
            "approved": self.approved,
            "submitted": self.submitted
        }


@dataclass
class CoffeeBreakAlert:
    """Alert from COFFEE_BREAK about Chris's schedule"""
    alert_id: str
    alert_type: str  # break_needed, meeting_gap, burnout_warning, etc.
    message: str
    suggested_action: str
    timestamp: datetime
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


# =============================================================================
# SUB-AGENT: COFFEE_BREAK
# =============================================================================

class CoffeeBreakAgent:
    """
    COFFEE_BREAK - Schedule optimization and work-life balance

    This sub-agent ensures Chris doesn't burn out by:
    - Finding gaps between meetings
    - Enforcing break times
    - Tracking consecutive meeting hours
    - Monitoring work-life balance patterns
    - Suggesting schedule optimizations

    Named COFFEE_BREAK because everyone needs a coffee break.
    """

    def __init__(self):
        self.name = "COFFEE_BREAK"
        self.alerts: List[CoffeeBreakAlert] = []
        self.consecutive_meeting_hours = 0
        self.last_break_time: Optional[datetime] = None
        self.daily_meeting_hours = 0
        self.weekly_meeting_hours = 0

        # Thresholds
        self.MAX_CONSECUTIVE_HOURS = 3
        self.RECOMMENDED_BREAK_INTERVAL = 90  # minutes
        self.MAX_DAILY_MEETINGS = 6  # hours
        self.MAX_WEEKLY_MEETINGS = 25  # hours

        logger.info("COFFEE_BREAK sub-agent initialized")

    def analyze_schedule(self, meetings: List[Meeting], current_time: datetime = None) -> Dict:
        """Analyze Chris's schedule for optimization opportunities"""
        current_time = current_time or datetime.now()
        today = current_time.date()

        todays_meetings = [m for m in meetings if m.start_time.date() == today]
        todays_meetings.sort(key=lambda m: m.start_time)

        analysis = {
            "total_meetings": len(todays_meetings),
            "total_hours": sum(m.duration_minutes() for m in todays_meetings) / 60,
            "gaps": [],
            "back_to_back_warnings": [],
            "break_opportunities": [],
            "could_be_emails": [m.title for m in todays_meetings if m.could_be_email],
            "recommendations": []
        }

        # Find gaps between meetings
        for i in range(len(todays_meetings) - 1):
            current = todays_meetings[i]
            next_meeting = todays_meetings[i + 1]
            gap_minutes = (next_meeting.start_time - current.end_time).total_seconds() / 60

            if gap_minutes >= 15:
                analysis["gaps"].append({
                    "after": current.title,
                    "before": next_meeting.title,
                    "duration_minutes": gap_minutes,
                    "start_time": current.end_time.isoformat()
                })

                if gap_minutes >= 30:
                    analysis["break_opportunities"].append({
                        "time": current.end_time.isoformat(),
                        "duration": gap_minutes,
                        "suggestion": "Perfect time for a coffee break or quick walk"
                    })

            if gap_minutes < 5:
                analysis["back_to_back_warnings"].append({
                    "meeting1": current.title,
                    "meeting2": next_meeting.title,
                    "warning": "Back-to-back meetings with no buffer"
                })

        # Generate recommendations
        if analysis["total_hours"] > self.MAX_DAILY_MEETINGS:
            analysis["recommendations"].append({
                "type": "overbooked",
                "message": f"Chris, you have {analysis['total_hours']:.1f} hours of meetings today. " +
                          "I recommend rescheduling at least one.",
                "severity": "high"
            })

        if len(analysis["could_be_emails"]) > 0:
            analysis["recommendations"].append({
                "type": "efficiency",
                "message": f"These {len(analysis['could_be_emails'])} meetings could potentially be emails: " +
                          ", ".join(analysis["could_be_emails"]),
                "severity": "medium"
            })

        if len(analysis["back_to_back_warnings"]) >= 3:
            analysis["recommendations"].append({
                "type": "burnout_risk",
                "message": "You have 3+ back-to-back meetings. COFFEE_BREAK strongly recommends " +
                          "adding 5-10 minute buffers.",
                "severity": "high"
            })

        return analysis

    def check_break_needed(self, last_meeting_end: datetime, current_time: datetime = None) -> Optional[CoffeeBreakAlert]:
        """Check if Chris needs a break"""
        current_time = current_time or datetime.now()

        if self.last_break_time:
            minutes_since_break = (current_time - self.last_break_time).total_seconds() / 60

            if minutes_since_break > self.RECOMMENDED_BREAK_INTERVAL:
                alert = CoffeeBreakAlert(
                    alert_id=f"BREAK_{current_time.strftime('%Y%m%d%H%M')}",
                    alert_type="break_needed",
                    message=f"Chris, it's been {int(minutes_since_break)} minutes since your last break.",
                    suggested_action="Take 10-15 minutes to stretch, get coffee, or step outside.",
                    timestamp=current_time
                )
                self.alerts.append(alert)
                return alert

        return None

    def record_break(self, timestamp: datetime = None):
        """Record that Chris took a break"""
        self.last_break_time = timestamp or datetime.now()
        self.consecutive_meeting_hours = 0
        logger.info(f"COFFEE_BREAK: Break recorded at {self.last_break_time}")

    def get_status(self) -> Dict:
        """Get COFFEE_BREAK status"""
        return {
            "agent": "COFFEE_BREAK",
            "consecutive_meeting_hours": self.consecutive_meeting_hours,
            "last_break_time": self.last_break_time.isoformat() if self.last_break_time else None,
            "daily_meeting_hours": self.daily_meeting_hours,
            "weekly_meeting_hours": self.weekly_meeting_hours,
            "pending_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "thresholds": {
                "max_consecutive_hours": self.MAX_CONSECUTIVE_HOURS,
                "recommended_break_interval_min": self.RECOMMENDED_BREAK_INTERVAL,
                "max_daily_meetings_hours": self.MAX_DAILY_MEETINGS,
                "max_weekly_meetings_hours": self.MAX_WEEKLY_MEETINGS
            }
        }


# =============================================================================
# SUB-AGENT: BEAN_COUNTER
# =============================================================================

class BeanCounterAgent:
    """
    BEAN_COUNTER - Expense tracking and time management

    This sub-agent handles:
    - Expense tracking and categorization
    - Receipt management
    - Reimbursement processing
    - Time tracking for billable work
    - Budget monitoring

    Named BEAN_COUNTER because every bean is counted (and also a fun
    play on the accounting term).
    """

    def __init__(self):
        self.name = "BEAN_COUNTER"
        self.expenses: List[Expense] = []
        self.pending_receipts: List[str] = []  # Expenses missing receipts
        self.monthly_budget: Dict[str, Decimal] = {
            "travel": Decimal("5000"),
            "meals": Decimal("1000"),
            "entertainment": Decimal("2000"),
            "office": Decimal("500"),
            "other": Decimal("1000")
        }
        self.monthly_spend: Dict[str, Decimal] = {
            "travel": Decimal("0"),
            "meals": Decimal("0"),
            "entertainment": Decimal("0"),
            "office": Decimal("0"),
            "other": Decimal("0")
        }
        self.time_entries: List[Dict] = []

        logger.info("BEAN_COUNTER sub-agent initialized")

    def add_expense(
        self,
        description: str,
        amount: float,
        category: str,
        date: datetime = None,
        receipt_attached: bool = False
    ) -> Expense:
        """Add a new expense"""
        import hashlib

        date = date or datetime.now()
        expense_id = f"EXP_{hashlib.sha256(f'{description}{amount}{date}'.encode()).hexdigest()[:8]}"

        expense = Expense(
            expense_id=expense_id,
            description=description,
            amount=Decimal(str(amount)),
            category=category,
            date=date,
            receipt_attached=receipt_attached
        )

        self.expenses.append(expense)

        # Update monthly spend
        if category in self.monthly_spend:
            self.monthly_spend[category] += expense.amount
        else:
            self.monthly_spend["other"] += expense.amount

        # Track missing receipts
        if not receipt_attached and amount > 25:  # Receipts required over $25
            self.pending_receipts.append(expense_id)

        logger.info(f"BEAN_COUNTER: Added expense {expense_id}: ${amount} - {description}")

        return expense

    def get_expense_report(self, month: int = None, year: int = None) -> Dict:
        """Generate expense report"""
        now = datetime.now()
        month = month or now.month
        year = year or now.year

        month_expenses = [
            e for e in self.expenses
            if e.date.month == month and e.date.year == year
        ]

        by_category = {}
        for exp in month_expenses:
            cat = exp.category
            if cat not in by_category:
                by_category[cat] = {"total": Decimal("0"), "count": 0, "items": []}
            by_category[cat]["total"] += exp.amount
            by_category[cat]["count"] += 1
            by_category[cat]["items"].append(exp.to_dict())

        total = sum(e.amount for e in month_expenses)

        return {
            "report_type": "monthly_expense",
            "month": month,
            "year": year,
            "total_expenses": str(total),
            "expense_count": len(month_expenses),
            "by_category": {k: {"total": str(v["total"]), "count": v["count"]}
                          for k, v in by_category.items()},
            "pending_receipts": len(self.pending_receipts),
            "budget_status": {
                cat: {
                    "budget": str(budget),
                    "spent": str(self.monthly_spend.get(cat, Decimal("0"))),
                    "remaining": str(budget - self.monthly_spend.get(cat, Decimal("0")))
                }
                for cat, budget in self.monthly_budget.items()
            },
            "items_needing_receipts": self.pending_receipts[:10]  # First 10
        }

    def add_time_entry(
        self,
        project: str,
        hours: float,
        description: str,
        date: datetime = None,
        billable: bool = True
    ) -> Dict:
        """Add a time tracking entry"""
        date = date or datetime.now()

        entry = {
            "project": project,
            "hours": hours,
            "description": description,
            "date": date.isoformat(),
            "billable": billable
        }

        self.time_entries.append(entry)
        logger.info(f"BEAN_COUNTER: Added {hours}h to {project}")

        return entry

    def get_time_summary(self, days: int = 7) -> Dict:
        """Get time tracking summary"""
        cutoff = datetime.now() - timedelta(days=days)

        recent_entries = [
            e for e in self.time_entries
            if datetime.fromisoformat(e["date"]) > cutoff
        ]

        by_project = {}
        for entry in recent_entries:
            proj = entry["project"]
            if proj not in by_project:
                by_project[proj] = {"total_hours": 0, "billable_hours": 0}
            by_project[proj]["total_hours"] += entry["hours"]
            if entry["billable"]:
                by_project[proj]["billable_hours"] += entry["hours"]

        return {
            "period_days": days,
            "total_entries": len(recent_entries),
            "total_hours": sum(e["hours"] for e in recent_entries),
            "billable_hours": sum(e["hours"] for e in recent_entries if e["billable"]),
            "by_project": by_project
        }

    def get_status(self) -> Dict:
        """Get BEAN_COUNTER status"""
        return {
            "agent": "BEAN_COUNTER",
            "total_expenses_tracked": len(self.expenses),
            "pending_receipts": len(self.pending_receipts),
            "time_entries": len(self.time_entries),
            "monthly_budget": {k: str(v) for k, v in self.monthly_budget.items()},
            "monthly_spend": {k: str(v) for k, v in self.monthly_spend.items()}
        }


# =============================================================================
# MAIN AGENT: SHYLA
# =============================================================================

class ShylaAgent(BaseAgent):
    """
    SHYLA - Chris Friedman's Executive Assistant

    Handles anything and everything Chris asks for, professional and personal.
    Personal data is GATED - no training without Chris's explicit permission.
    """

    def __init__(self):
        super().__init__(
            name="SHYLA",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Calendar & Scheduling
                "calendar_management",
                "meeting_scheduling",
                "meeting_preparation",
                "meeting_follow_up",
                "schedule_optimization",
                "conflict_resolution",

                # Communication
                "email_drafting",
                "email_management",
                "correspondence",
                "message_prioritization",
                "communication_tracking",

                # Travel & Logistics
                "travel_booking",
                "travel_logistics",
                "itinerary_management",
                "accommodation_booking",
                "transportation_coordination",

                # Documents & Research
                "document_preparation",
                "document_formatting",
                "research_tasks",
                "report_compilation",
                "presentation_prep",

                # Task & Priority Management
                "task_tracking",
                "priority_management",
                "deadline_monitoring",
                "delegation_coordination",
                "action_item_tracking",

                # Coordination
                "agent_liaison",
                "stakeholder_coordination",
                "investor_communication_drafts",
                "team_coordination",

                # Personal (GATED)
                "personal_calendar_gated",
                "personal_travel_gated",
                "personal_reminders_gated",
                "personal_purchases_gated",

                # Sub-agent capabilities
                "schedule_breaks",           # COFFEE_BREAK
                "expense_tracking",          # BEAN_COUNTER
                "time_tracking",             # BEAN_COUNTER
                "budget_monitoring",         # BEAN_COUNTER
            ],
            user_id="CF",  # Chris Friedman
            thinking_modes=[
                ThinkingMode.STRUCTURAL,     # Organize and structure
                ThinkingMode.CREATIVE,       # Find elegant solutions
                ThinkingMode.SECOND_ORDER,   # Anticipate consequences
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,
                LearningMethod.CONTEXTUAL,
                LearningMethod.PREFERENCE,
            ],
        )

        # Chris's preferences (learned over time)
        self.chris_preferences = {
            "email_style": "concise, professional, warm",
            "meeting_length_preference": 30,  # minutes
            "morning_block_protected": True,  # 6-9am
            "lunch_block": "12:00-13:00",
            "communication_style": "direct with context",
            "coffee_order": "TBD - ask Chris",
        }

        # Personal data consent (default: NONE)
        self.personal_consent = PersonalDataConsent.NONE
        self.personal_training_enabled = False

        # Sub-agents
        self.coffee_break = CoffeeBreakAgent()
        self.bean_counter = BeanCounterAgent()

        # Task tracking
        self.tasks: List[Task] = []
        self.meetings: List[Meeting] = []

        # Communication logs
        self.chris_communications: List[Dict] = []
        self.frieds_communications: List[Dict] = []

        # Action handlers
        self.handlers = {
            # Morning briefing
            "morning_briefing": self._handle_morning_briefing,
            "daily_overview": self._handle_morning_briefing,

            # Calendar
            "schedule_meeting": self._handle_schedule_meeting,
            "check_availability": self._handle_check_availability,
            "reschedule_meeting": self._handle_reschedule,
            "cancel_meeting": self._handle_cancel,
            "get_calendar": self._handle_get_calendar,

            # Tasks
            "add_task": self._handle_add_task,
            "get_tasks": self._handle_get_tasks,
            "complete_task": self._handle_complete_task,
            "prioritize_tasks": self._handle_prioritize,

            # Communication
            "draft_email": self._handle_draft_email,
            "prepare_meeting": self._handle_prepare_meeting,
            "take_notes": self._handle_take_notes,

            # Travel
            "book_travel": self._handle_book_travel,
            "travel_itinerary": self._handle_travel_itinerary,

            # Sub-agents
            "schedule_break": self._handle_schedule_break,
            "add_expense": self._handle_add_expense,
            "expense_report": self._handle_expense_report,
            "track_time": self._handle_track_time,
            "time_summary": self._handle_time_summary,

            # Personal (GATED)
            "personal_request": self._handle_personal_request,
            "set_personal_consent": self._handle_set_consent,

            # Status
            "get_status": self._handle_get_status,
        }

        logger.info("SHYLA initialized - Chris's Executive Assistant ready")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request from Chris or FRIEDS"""
        action = inputs.get("action", "get_status")
        params = inputs.get("params", inputs)

        handler = self.handlers.get(action, self._handle_unknown)
        result = handler(params)

        # Log communication
        self._log_communication(action, params, result)

        return result

    # =========================================================================
    # MORNING BRIEFING
    # =========================================================================

    def _handle_morning_briefing(self, params: Dict) -> Dict:
        """Generate morning briefing for Chris"""
        now = datetime.now()
        today = now.date()

        # Get today's meetings
        todays_meetings = [m for m in self.meetings if m.start_time.date() == today]
        todays_meetings.sort(key=lambda m: m.start_time)

        # Get schedule analysis from COFFEE_BREAK
        schedule_analysis = self.coffee_break.analyze_schedule(self.meetings, now)

        # Get high priority tasks
        urgent_tasks = [t for t in self.tasks
                       if t.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]
                       and not t.completed_at]

        # Build briefing
        briefing = {
            "greeting": f"Good morning, Chris. Today is {now.strftime('%A, %B %d, %Y')}.",
            "quick_summary": {
                "meetings_today": len(todays_meetings),
                "meeting_hours": f"{schedule_analysis['total_hours']:.1f}",
                "urgent_tasks": len(urgent_tasks),
                "could_be_emails": len(schedule_analysis.get("could_be_emails", []))
            },
            "meetings": [
                {
                    "time": m.start_time.strftime("%I:%M %p"),
                    "title": m.title,
                    "duration": f"{m.duration_minutes()} min",
                    "attendees": m.attendees,
                    "could_be_email": m.could_be_email
                }
                for m in todays_meetings[:5]  # First 5
            ],
            "priority_tasks": [
                {
                    "title": t.title,
                    "priority": t.priority.value,
                    "due": t.due_date.strftime("%I:%M %p") if t.due_date else "No deadline"
                }
                for t in urgent_tasks[:5]  # First 5
            ],
            "coffee_break_analysis": schedule_analysis.get("recommendations", []),
            "action_items": [],
            "preferences": {
                "coffee_order": self.chris_preferences.get("coffee_order", "Not set"),
            }
        }

        # Add action items
        if schedule_analysis.get("could_be_emails"):
            briefing["action_items"].append({
                "suggestion": "Review meetings that could be emails",
                "meetings": schedule_analysis["could_be_emails"]
            })

        if len(schedule_analysis.get("back_to_back_warnings", [])) > 0:
            briefing["action_items"].append({
                "suggestion": "Add buffer time between back-to-back meetings",
                "count": len(schedule_analysis["back_to_back_warnings"])
            })

        return {
            "status": "success",
            "report_type": "morning_briefing",
            "timestamp": now.isoformat(),
            "briefing": briefing
        }

    # =========================================================================
    # CALENDAR MANAGEMENT
    # =========================================================================

    def _handle_schedule_meeting(self, params: Dict) -> Dict:
        """Schedule a new meeting"""
        import hashlib

        title = params.get("title", "New Meeting")
        start_time = params.get("start_time")
        duration = params.get("duration_minutes", 30)
        attendees = params.get("attendees", [])
        location = params.get("location", "")
        virtual_link = params.get("virtual_link", "")

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        end_time = start_time + timedelta(minutes=duration)

        meeting_id = f"MTG_{hashlib.sha256(f'{title}{start_time}'.encode()).hexdigest()[:8]}"

        meeting = Meeting(
            meeting_id=meeting_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            attendees=attendees,
            location=location,
            virtual_link=virtual_link,
            status=MeetingStatus.SCHEDULED
        )

        self.meetings.append(meeting)

        return {
            "status": "success",
            "action": "meeting_scheduled",
            "meeting": meeting.to_dict(),
            "shyla_note": f"Chris, I've scheduled '{title}' for {start_time.strftime('%B %d at %I:%M %p')}. " +
                         f"Duration: {duration} minutes. I'll send calendar invites to {len(attendees)} attendees."
        }

    def _handle_check_availability(self, params: Dict) -> Dict:
        """Check Chris's availability"""
        date_str = params.get("date")
        duration = params.get("duration_minutes", 30)

        if date_str:
            check_date = datetime.fromisoformat(date_str).date()
        else:
            check_date = datetime.now().date()

        # Get meetings on that day
        day_meetings = [m for m in self.meetings if m.start_time.date() == check_date]
        day_meetings.sort(key=lambda m: m.start_time)

        # Find available slots (9am - 6pm)
        work_start = datetime.combine(check_date, datetime.strptime("09:00", "%H:%M").time())
        work_end = datetime.combine(check_date, datetime.strptime("18:00", "%H:%M").time())

        available_slots = []
        current = work_start

        for meeting in day_meetings:
            if current + timedelta(minutes=duration) <= meeting.start_time:
                available_slots.append({
                    "start": current.strftime("%I:%M %p"),
                    "end": meeting.start_time.strftime("%I:%M %p"),
                    "duration_available": int((meeting.start_time - current).total_seconds() / 60)
                })
            current = max(current, meeting.end_time)

        # Check end of day
        if current + timedelta(minutes=duration) <= work_end:
            available_slots.append({
                "start": current.strftime("%I:%M %p"),
                "end": work_end.strftime("%I:%M %p"),
                "duration_available": int((work_end - current).total_seconds() / 60)
            })

        return {
            "status": "success",
            "date": check_date.isoformat(),
            "meetings_scheduled": len(day_meetings),
            "available_slots": available_slots,
            "shyla_recommendation": available_slots[0] if available_slots else "No availability - consider rescheduling"
        }

    def _handle_get_calendar(self, params: Dict) -> Dict:
        """Get calendar overview"""
        days = params.get("days", 1)
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days)

        relevant_meetings = [
            m for m in self.meetings
            if start_date <= m.start_time.date() <= end_date
        ]
        relevant_meetings.sort(key=lambda m: m.start_time)

        return {
            "status": "success",
            "period": f"{start_date} to {end_date}",
            "meeting_count": len(relevant_meetings),
            "meetings": [m.to_dict() for m in relevant_meetings],
            "total_meeting_hours": sum(m.duration_minutes() for m in relevant_meetings) / 60
        }

    def _handle_reschedule(self, params: Dict) -> Dict:
        """Reschedule a meeting"""
        meeting_id = params.get("meeting_id")
        new_time = params.get("new_time")

        for meeting in self.meetings:
            if meeting.meeting_id == meeting_id:
                if isinstance(new_time, str):
                    new_time = datetime.fromisoformat(new_time)

                duration = meeting.duration_minutes()
                meeting.start_time = new_time
                meeting.end_time = new_time + timedelta(minutes=duration)
                meeting.status = MeetingStatus.RESCHEDULED

                return {
                    "status": "success",
                    "action": "rescheduled",
                    "meeting": meeting.to_dict(),
                    "shyla_note": f"Chris, I've rescheduled '{meeting.title}' to {new_time.strftime('%B %d at %I:%M %p')}. " +
                                "I'll update all attendees."
                }

        return {"status": "error", "message": f"Meeting {meeting_id} not found"}

    def _handle_cancel(self, params: Dict) -> Dict:
        """Cancel a meeting"""
        meeting_id = params.get("meeting_id")

        for meeting in self.meetings:
            if meeting.meeting_id == meeting_id:
                meeting.status = MeetingStatus.CANCELLED
                return {
                    "status": "success",
                    "action": "cancelled",
                    "meeting_id": meeting_id,
                    "shyla_note": f"Chris, I've cancelled '{meeting.title}'. I'll notify all attendees."
                }

        return {"status": "error", "message": f"Meeting {meeting_id} not found"}

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def _handle_add_task(self, params: Dict) -> Dict:
        """Add a new task"""
        import hashlib

        title = params.get("title", "New Task")
        category = TaskCategory[params.get("category", "COORDINATION").upper()]
        priority = TaskPriority[params.get("priority", "MEDIUM").upper()]
        due_date = params.get("due_date")
        notes = params.get("notes", "")

        if due_date and isinstance(due_date, str):
            due_date = datetime.fromisoformat(due_date)

        task_id = f"TASK_{hashlib.sha256(f'{title}{datetime.now()}'.encode()).hexdigest()[:8]}"

        task = Task(
            task_id=task_id,
            title=title,
            category=category,
            priority=priority,
            created_at=datetime.now(),
            due_date=due_date,
            notes=notes
        )

        self.tasks.append(task)

        return {
            "status": "success",
            "action": "task_added",
            "task": task.to_dict(),
            "shyla_note": f"Got it, Chris. I've added '{title}' as a {priority.value} priority task."
        }

    def _handle_get_tasks(self, params: Dict) -> Dict:
        """Get task list"""
        filter_priority = params.get("priority")
        filter_category = params.get("category")
        include_completed = params.get("include_completed", False)

        tasks = self.tasks

        if not include_completed:
            tasks = [t for t in tasks if not t.completed_at]

        if filter_priority:
            tasks = [t for t in tasks if t.priority.value == filter_priority]

        if filter_category:
            tasks = [t for t in tasks if t.category.value == filter_category]

        # Sort by priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
            TaskPriority.SOMEDAY: 4
        }
        tasks.sort(key=lambda t: priority_order[t.priority])

        return {
            "status": "success",
            "task_count": len(tasks),
            "tasks": [t.to_dict() for t in tasks]
        }

    def _handle_complete_task(self, params: Dict) -> Dict:
        """Mark a task as complete"""
        task_id = params.get("task_id")

        for task in self.tasks:
            if task.task_id == task_id:
                task.completed_at = datetime.now()
                return {
                    "status": "success",
                    "action": "task_completed",
                    "task_id": task_id,
                    "shyla_note": f"Great, Chris! '{task.title}' is now complete."
                }

        return {"status": "error", "message": f"Task {task_id} not found"}

    def _handle_prioritize(self, params: Dict) -> Dict:
        """Reprioritize tasks"""
        # Smart prioritization based on deadlines, category, etc.
        uncompleted = [t for t in self.tasks if not t.completed_at]

        # Sort by urgency
        def urgency_score(task):
            score = 0
            if task.due_date:
                days_until = (task.due_date - datetime.now()).days
                if days_until < 0:
                    score += 100  # Overdue
                elif days_until == 0:
                    score += 50  # Today
                elif days_until <= 2:
                    score += 25  # Very soon

            priority_scores = {
                TaskPriority.CRITICAL: 40,
                TaskPriority.HIGH: 30,
                TaskPriority.MEDIUM: 20,
                TaskPriority.LOW: 10,
                TaskPriority.SOMEDAY: 0
            }
            score += priority_scores[task.priority]
            return score

        uncompleted.sort(key=urgency_score, reverse=True)

        return {
            "status": "success",
            "prioritized_tasks": [
                {
                    "rank": i + 1,
                    "task": t.to_dict(),
                    "urgency_score": urgency_score(t)
                }
                for i, t in enumerate(uncompleted[:10])
            ],
            "shyla_recommendation": f"Chris, here are your top {min(10, len(uncompleted))} priority tasks. " +
                                   "The first one should be your focus right now."
        }

    # =========================================================================
    # COMMUNICATION
    # =========================================================================

    def _handle_draft_email(self, params: Dict) -> Dict:
        """Draft an email for Chris"""
        recipient = params.get("recipient", "")
        subject = params.get("subject", "")
        key_points = params.get("key_points", [])
        tone = params.get("tone", self.chris_preferences.get("email_style", "professional"))

        # Generate draft based on Chris's preferences
        draft = f"""To: {recipient}
Subject: {subject}

"""

        if key_points:
            draft += "Hi,\n\n"
            for point in key_points:
                draft += f"• {point}\n"
            draft += "\nBest regards,\nChris"
        else:
            draft += "[Draft body - please provide key points]\n\nBest regards,\nChris"

        return {
            "status": "success",
            "draft": draft,
            "tone_applied": tone,
            "shyla_note": "Chris, I've drafted this email for your review. " +
                         "Let me know if you'd like me to adjust the tone or add anything."
        }

    def _handle_prepare_meeting(self, params: Dict) -> Dict:
        """Prepare materials for a meeting"""
        meeting_id = params.get("meeting_id")

        for meeting in self.meetings:
            if meeting.meeting_id == meeting_id:
                prep = {
                    "meeting": meeting.to_dict(),
                    "preparation": {
                        "agenda_items": ["Item 1", "Item 2"],  # Would be generated
                        "attendee_context": {
                            att: f"Context for {att}" for att in meeting.attendees
                        },
                        "previous_meeting_notes": "Would pull from history",
                        "relevant_documents": [],
                        "suggested_talking_points": []
                    },
                    "shyla_note": f"Chris, here's your prep for '{meeting.title}'. " +
                                 "I've included context on attendees and relevant background."
                }
                return {"status": "success", **prep}

        return {"status": "error", "message": f"Meeting {meeting_id} not found"}

    def _handle_take_notes(self, params: Dict) -> Dict:
        """Record meeting notes"""
        meeting_id = params.get("meeting_id")
        notes = params.get("notes", "")
        action_items = params.get("action_items", [])

        for meeting in self.meetings:
            if meeting.meeting_id == meeting_id:
                meeting.notes = notes
                meeting.action_items = action_items
                meeting.status = MeetingStatus.COMPLETED

                # Create tasks from action items
                for item in action_items:
                    self._handle_add_task({
                        "title": item,
                        "category": "MEETING",
                        "priority": "HIGH",
                        "notes": f"From meeting: {meeting.title}"
                    })

                return {
                    "status": "success",
                    "meeting_id": meeting_id,
                    "notes_saved": True,
                    "tasks_created": len(action_items),
                    "shyla_note": f"Notes saved for '{meeting.title}'. " +
                                 f"I've created {len(action_items)} follow-up tasks."
                }

        return {"status": "error", "message": f"Meeting {meeting_id} not found"}

    # =========================================================================
    # TRAVEL
    # =========================================================================

    def _handle_book_travel(self, params: Dict) -> Dict:
        """Book travel for Chris"""
        destination = params.get("destination", "")
        departure_date = params.get("departure_date")
        return_date = params.get("return_date")
        purpose = params.get("purpose", "business")

        return {
            "status": "pending_confirmation",
            "travel_request": {
                "destination": destination,
                "departure": departure_date,
                "return": return_date,
                "purpose": purpose
            },
            "shyla_note": f"Chris, I'll research options for your trip to {destination}. " +
                         "I'll present 3 flight options and 2 hotel options for your review. " +
                         "Do you have any airline or hotel preferences I should prioritize?"
        }

    def _handle_travel_itinerary(self, params: Dict) -> Dict:
        """Get or create travel itinerary"""
        trip_id = params.get("trip_id")

        return {
            "status": "success",
            "itinerary": {
                "trip_id": trip_id or "NEW",
                "segments": [],
                "accommodations": [],
                "ground_transportation": [],
                "important_contacts": [],
                "documents_needed": ["Passport", "ID"]
            },
            "shyla_note": "Chris, here's your itinerary. I've included all confirmations and important details."
        }

    # =========================================================================
    # SUB-AGENT HANDLERS
    # =========================================================================

    def _handle_schedule_break(self, params: Dict) -> Dict:
        """COFFEE_BREAK: Schedule a break for Chris"""
        duration = params.get("duration_minutes", 15)

        # Find next available gap
        analysis = self.coffee_break.analyze_schedule(self.meetings)

        if analysis.get("break_opportunities"):
            opportunity = analysis["break_opportunities"][0]
            self.coffee_break.record_break()

            return {
                "status": "success",
                "break_scheduled": True,
                "time": opportunity["time"],
                "duration": min(duration, opportunity["duration"]),
                "coffee_break_says": f"Chris, I've blocked {duration} minutes for a break at {opportunity['time']}. " +
                                    "You've earned it!"
            }

        return {
            "status": "no_availability",
            "coffee_break_says": "Chris, your schedule is packed. " +
                                "I strongly recommend canceling a non-essential meeting for your well-being."
        }

    def _handle_add_expense(self, params: Dict) -> Dict:
        """BEAN_COUNTER: Add an expense"""
        expense = self.bean_counter.add_expense(
            description=params.get("description", ""),
            amount=params.get("amount", 0),
            category=params.get("category", "other"),
            receipt_attached=params.get("receipt_attached", False)
        )

        return {
            "status": "success",
            "expense": expense.to_dict(),
            "bean_counter_says": f"Expense recorded: ${expense.amount} for {expense.description}. " +
                                ("Receipt attached." if expense.receipt_attached else "[!] Receipt needed!")
        }

    def _handle_expense_report(self, params: Dict) -> Dict:
        """BEAN_COUNTER: Generate expense report"""
        report = self.bean_counter.get_expense_report(
            month=params.get("month"),
            year=params.get("year")
        )

        return {
            "status": "success",
            **report,
            "bean_counter_says": f"Here's your expense report. Total: ${report['total_expenses']}. " +
                                f"{report['pending_receipts']} receipts still needed."
        }

    def _handle_track_time(self, params: Dict) -> Dict:
        """BEAN_COUNTER: Track time"""
        entry = self.bean_counter.add_time_entry(
            project=params.get("project", "General"),
            hours=params.get("hours", 0),
            description=params.get("description", ""),
            billable=params.get("billable", True)
        )

        return {
            "status": "success",
            "time_entry": entry,
            "bean_counter_says": f"Logged {entry['hours']} hours to {entry['project']}."
        }

    def _handle_time_summary(self, params: Dict) -> Dict:
        """BEAN_COUNTER: Get time summary"""
        summary = self.bean_counter.get_time_summary(days=params.get("days", 7))

        return {
            "status": "success",
            **summary,
            "bean_counter_says": f"This week: {summary['total_hours']} hours logged, " +
                                f"{summary['billable_hours']} billable."
        }

    # =========================================================================
    # PERSONAL TASKS (GATED)
    # =========================================================================

    def _handle_personal_request(self, params: Dict) -> Dict:
        """Handle personal requests - GATED"""
        request_type = params.get("type", "")

        # Check consent level
        if self.personal_consent == PersonalDataConsent.NONE:
            return {
                "status": "gated",
                "message": "Chris, personal assistance is currently disabled. " +
                          "Would you like to enable it? I promise to keep everything confidential " +
                          "and will NOT train on personal data unless you explicitly allow it."
            }

        # Handle based on consent level
        if self.personal_consent == PersonalDataConsent.CALENDAR_ONLY:
            if request_type not in ["calendar", "reminder"]:
                return {
                    "status": "limited",
                    "message": f"Chris, {request_type} requests require full personal assistance mode. " +
                              "Currently I only have access to your personal calendar."
                }

        # Process personal request
        return {
            "status": "success",
            "request_type": request_type,
            "shyla_note": f"Processing personal {request_type} request...",
            "privacy_reminder": "This request is handled privately and is NOT used for training."
        }

    def _handle_set_consent(self, params: Dict) -> Dict:
        """Set personal data consent level"""
        level = params.get("level", "NONE").upper()
        allow_training = params.get("allow_training", False)

        try:
            self.personal_consent = PersonalDataConsent[level]
            self.personal_training_enabled = allow_training

            return {
                "status": "success",
                "consent_level": self.personal_consent.value,
                "training_enabled": self.personal_training_enabled,
                "shyla_note": "Chris, I've updated your personal assistance settings. " +
                             f"Consent level: {level}. Training on personal data: {'Enabled' if allow_training else 'Disabled'}."
            }
        except KeyError:
            return {
                "status": "error",
                "message": "Invalid consent level. Options: NONE, CALENDAR_ONLY, FULL_ASSIST, FULL_TRAINING"
            }

    # =========================================================================
    # STATUS & UTILITY
    # =========================================================================

    def _handle_get_status(self, params: Dict) -> Dict:
        """Get SHYLA's status"""
        return {
            "status": "success",
            "agent": "SHYLA",
            "role": "Executive Assistant to Chris Friedman",
            "operational_status": "active",
            "pending_tasks": len([t for t in self.tasks if not t.completed_at]),
            "upcoming_meetings": len([m for m in self.meetings if m.start_time > datetime.now()]),
            "personal_consent_level": self.personal_consent.value,
            "personal_training": self.personal_training_enabled,
            "sub_agents": {
                "coffee_break": self.coffee_break.get_status(),
                "bean_counter": self.bean_counter.get_status()
            },
            "chris_preferences": self.chris_preferences
        }

    def _handle_unknown(self, params: Dict) -> Dict:
        """Handle unknown requests"""
        return {
            "status": "clarification_needed",
            "shyla_note": "Chris, I'm not sure what you need. Could you rephrase? " +
                         "I can help with: calendar, meetings, tasks, email drafts, travel, " +
                         "expenses, time tracking, and more. Just ask!"
        }

    def _log_communication(self, action: str, params: Dict, result: Dict):
        """Log communication for training (professional only)"""
        # Only log professional communications
        if action != "personal_request" and not self.personal_training_enabled:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "params_summary": {k: v for k, v in params.items() if k not in ["personal", "private"]},
                "success": result.get("status") == "success"
            }
            self.chris_communications.append(log_entry)

    def get_natural_language_explanation(self) -> str:
        """Return natural language description of SHYLA"""
        return """
================================================================================
SHYLA - Chris Friedman's Executive Assistant
================================================================================

Hi Chris! I'm SHYLA, your executive assistant. I'm here to handle anything and
everything you need - professional and personal.

WHAT I CAN DO FOR YOU:

CALENDAR & MEETINGS
   - Schedule, reschedule, or cancel meetings
   - Check your availability
   - Prepare meeting materials
   - Take notes and track action items

COMMUNICATION
   - Draft emails in your voice
   - Prioritize your inbox
   - Handle correspondence

TRAVEL & LOGISTICS
   - Book travel arrangements
   - Create itineraries
   - Handle logistics

TASKS & PRIORITIES
   - Track your to-dos
   - Help prioritize your day
   - Follow up on deadlines

MY SUB-AGENTS:
   - COFFEE_BREAK: Makes sure you don't burn out. Tracks meeting hours,
     finds break opportunities, and protects your well-being.
   - BEAN_COUNTER: Handles expenses, receipts, time tracking, and budgets.
     Every bean is counted!

PERSONAL MATTERS:
   I CAN help with personal tasks, but only with your permission.
   Your privacy is paramount - I will NEVER train on personal data
   unless you explicitly enable it.

Just ask, Chris. I've got you covered.

- SHYLA
================================================================================
"""


# =============================================================================
# SINGLETON
# =============================================================================

_shyla_instance: Optional[ShylaAgent] = None


def get_shyla() -> ShylaAgent:
    """Get singleton SHYLA instance"""
    global _shyla_instance
    if _shyla_instance is None:
        _shyla_instance = ShylaAgent()
    return _shyla_instance


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SHYLA Agent")
    parser.add_argument("mode", nargs="?", default="help",
                       choices=["run", "status", "briefing", "help"],
                       help="Mode to run the agent in")
    args = parser.parse_args()

    agent = get_shyla()

    print(f"\n{'='*70}")
    print("SHYLA - Chris Friedman's Executive Assistant")
    print(f"{'='*70}")

    if args.mode == "help":
        print(agent.get_natural_language_explanation())
    elif args.mode == "status":
        result = agent.process({"action": "get_status"})
        print(f"\nStatus: {result}")
    elif args.mode == "briefing":
        result = agent.process({"action": "morning_briefing"})
        print(f"\nMorning Briefing: {result}")
    elif args.mode == "run":
        print("\nSHYLA is ready. Commands: status, briefing, schedule, tasks, expense")
        while True:
            try:
                cmd = input("\nChris > ").strip().lower()
                if cmd in ["exit", "quit"]:
                    print("SHYLA: Have a great day, Chris!")
                    break
                elif cmd == "status":
                    print(agent.process({"action": "get_status"}))
                elif cmd == "briefing":
                    print(agent.process({"action": "morning_briefing"}))
                elif cmd == "tasks":
                    print(agent.process({"action": "get_tasks"}))
                elif cmd.startswith("add task "):
                    title = cmd[9:]
                    print(agent.process({"action": "add_task", "title": title}))
                else:
                    print(f"SHYLA: I heard '{cmd}'. How can I help?")
            except KeyboardInterrupt:
                print("\nSHYLA: Goodbye, Chris!")
                break

