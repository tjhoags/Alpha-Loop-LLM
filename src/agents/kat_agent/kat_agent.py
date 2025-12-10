"""
================================================================================
KAT AGENT - Tom Hogan's Executive Assistant
================================================================================
Author: Tom Hogan (Founder & CIO)
Developer: Alpha Loop Capital, LLC

KAT is Tom Hogan's executive assistant - brilliant, efficient, and charming.
Handles ANYTHING Tom asks for with grace, wit, and unwavering loyalty.

Tier: SENIOR (1)
Reports To: HOAGS (Tom Hogan's Authority Agent)
Cluster: investment_executive

SECURITY MODEL:
- READ-ONLY access by default
- NO actions without WRITTEN PERMISSION from Tom
- Full audit trail on all activities
- Escalates sensitive matters immediately

PERSONALITY:
- Charming, witty, and professionally flirty
- Fiercely protective of Tom's time
- Anticipates needs before they arise
- Always one step ahead

================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from src.core.agent_base import (
    AgentTier,
    BaseAgent,
    LearningMethod,
    ThinkingMode,
)

logger = logging.getLogger(__name__)

# Thread pool for async operations
_executor = ThreadPoolExecutor(max_workers=4)


# =============================================================================
# ENUMS - Optimized with auto()
# =============================================================================

class Permission(Enum):
    """Permission levels for actions"""
    READ_ONLY = auto()      # Default - can only read/view
    SUGGEST = auto()        # Can suggest but not execute
    EXECUTE = auto()        # Can execute with notification
    FULL = auto()           # Full access (rare)


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    SOMEDAY = 4


class ActionType(Enum):
    """Types of actions KAT can take"""
    VIEW = auto()           # Always allowed
    DRAFT = auto()          # Always allowed
    SCHEDULE = auto()       # Requires permission
    SEND = auto()           # Requires permission
    BOOK = auto()           # Requires permission
    PURCHASE = auto()       # Requires permission
    MODIFY = auto()         # Requires permission


# =============================================================================
# DATA CLASSES - Optimized with __slots__
# =============================================================================

@dataclass(slots=True)
class Task:
    """Lightweight task tracking"""

    id: str
    title: str
    priority: TaskPriority
    created: datetime
    due: Optional[datetime] = None
    done: bool = False
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority.name,
            "created": self.created.isoformat(),
            "due": self.due.isoformat() if self.due else None,
            "done": self.done
        }


@dataclass(slots=True)
class Meeting:
    """Lightweight meeting tracking"""

    id: str
    title: str
    start: datetime
    end: datetime
    attendees: Tuple[str, ...]  # Immutable for hashing
    location: str = ""
    notes: str = ""

    @property
    def duration_min(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_min": self.duration_min,
            "attendees": list(self.attendees),
            "location": self.location
        }


@dataclass(slots=True)
class AuditEntry:
    """Audit trail entry"""

    timestamp: datetime
    action: ActionType
    target: str
    permission: Permission
    result: str  # "allowed", "blocked", "pending_approval"
    details: str = ""


# =============================================================================
# KAT AGENT
# =============================================================================

class KatAgent(BaseAgent):
    """
    KAT - Tom Hogan's Executive Assistant

    Charming, efficient, and fiercely loyal. KAT handles everything Tom
    needs with wit, grace, and impeccable judgment.

    READ-ONLY by default. No actions without Tom's written permission.
    """

    # Class-level constants for performance
    WORK_START = 6  # 6 AM
    WORK_END = 22   # 10 PM
    DEFAULT_MEETING_DURATION = 30
    MAX_CONSECUTIVE_MEETING_HOURS = 4

    def __init__(self):
        super().__init__(
            name="KAT",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Calendar (READ-ONLY default)
                "calendar_view", "calendar_draft", "calendar_schedule",
                "availability_check", "meeting_prep", "meeting_notes",

                # Communication (READ-ONLY default)
                "email_view", "email_draft", "email_send",
                "message_view", "message_draft",

                # Travel (READ-ONLY default)
                "travel_view", "travel_draft", "travel_book",
                "itinerary_view", "itinerary_draft",

                # Tasks
                "task_view", "task_create", "task_prioritize",

                # Research (always allowed)
                "research", "analysis", "summarize",

                # Coordination
                "agent_liaison", "stakeholder_coordination",

                # Personal (GATED)
                "personal_calendar_gated", "personal_tasks_gated",
            ],
            user_id="TH",  # Tom Hogan
            thinking_modes=[
                ThinkingMode.STRUCTURAL,
                ThinkingMode.CREATIVE,
                ThinkingMode.SECOND_ORDER,
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,
                LearningMethod.CONTEXTUAL,
            ],
        )

        # Security model
        self._permission_level = Permission.READ_ONLY
        self._written_permissions: Set[str] = set()  # Set for O(1) lookup
        self._audit_log: List[AuditEntry] = []

        # Caches for performance
        self._tasks: Dict[str, Task] = {}
        self._meetings: Dict[str, Meeting] = {}
        self._preferences_cache: Dict[str, Any] = {
            "communication_style": "direct, data-driven",
            "meeting_preference": "short and focused",
            "coffee_order": "Ask Tom",
        }

        # Communication log (capped for memory)
        self._tom_communications: List[Dict] = []
        self._max_comm_log = 1000

        logger.info("KAT initialized - Tom's Executive Assistant ready")

    # =========================================================================
    # CORE PROCESSING - Optimized
    # =========================================================================

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with security checks"""
        action = inputs.get("action", "status")

        # Fast path for common actions
        handler = self._get_handler(action)

        # Security check
        action_type = self._classify_action(action)
        if not self._check_permission(action_type, action):
            return self._permission_denied(action, action_type)

        # Execute and audit
        result = handler(inputs)
        self._audit(action_type, action, result)
        self._log_communication(action, result)

        return result

    @lru_cache(maxsize=128)
    def _get_handler(self, action: str):
        """Get handler with caching"""
        handlers = {
            # Status
            "status": self._handle_status,
            "briefing": self._handle_briefing,
            "morning_briefing": self._handle_briefing,

            # Calendar
            "calendar": self._handle_calendar,
            "schedule": self._handle_schedule,
            "availability": self._handle_availability,

            # Tasks
            "tasks": self._handle_tasks,
            "add_task": self._handle_add_task,
            "prioritize": self._handle_prioritize,

            # Communication
            "draft_email": self._handle_draft_email,
            "draft_message": self._handle_draft_message,

            # Research
            "research": self._handle_research,

            # Permissions
            "grant_permission": self._handle_grant_permission,
            "audit_log": self._handle_audit_log,
        }
        return handlers.get(action, self._handle_unknown)

    def _classify_action(self, action: str) -> ActionType:
        """Classify action type for permission checking"""
        view_actions = {"status", "briefing", "morning_briefing", "calendar",
                       "tasks", "availability", "audit_log", "research"}
        draft_actions = {"draft_email", "draft_message", "add_task", "prioritize"}

        if action in view_actions:
            return ActionType.VIEW
        elif action in draft_actions:
            return ActionType.DRAFT
        elif action == "schedule":
            return ActionType.SCHEDULE
        elif action in {"send_email", "send_message"}:
            return ActionType.SEND
        elif action in {"book_travel", "book_meeting"}:
            return ActionType.BOOK
        else:
            return ActionType.MODIFY

    def _check_permission(self, action_type: ActionType, action: str) -> bool:
        """Check if action is permitted"""
        # VIEW and DRAFT always allowed
        if action_type in {ActionType.VIEW, ActionType.DRAFT}:
            return True

        # Check written permissions
        if action in self._written_permissions:
            return True

        # Check permission level
        if self._permission_level == Permission.FULL:
            return True
        elif self._permission_level == Permission.EXECUTE:
            return action_type not in {ActionType.PURCHASE}
        elif self._permission_level == Permission.SUGGEST:
            return False

        return False

    def _permission_denied(self, action: str, action_type: ActionType) -> Dict:
        """Handle permission denied"""
        return {
            "status": "permission_required",
            "action": action,
            "action_type": action_type.name,
            "kat_says": f"Tom, darling, I'd love to help with that, but I need your "
                       f"written permission first. Just say '{action} approved' and "
                       f"I'll make it happen.",
            "required_approval": f"'{action} approved'"
        }

    def _audit(self, action_type: ActionType, action: str, result: Dict):
        """Log action to audit trail"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            action=action_type,
            target=action,
            permission=self._permission_level,
            result="allowed" if result.get("status") != "permission_required" else "blocked"
        )
        self._audit_log.append(entry)

        # Cap audit log size
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

    # =========================================================================
    # HANDLERS - Optimized
    # =========================================================================

    def _handle_status(self, params: Dict) -> Dict:
        """Get KAT status"""
        return {
            "status": "success",
            "agent": "KAT",
            "role": "Executive Assistant to Tom Hogan (Founder & CIO)",
            "security": {
                "permission_level": self._permission_level.name,
                "written_permissions": list(self._written_permissions),
                "audit_entries": len(self._audit_log)
            },
            "tasks_pending": sum(1 for t in self._tasks.values() if not t.done),
            "meetings_upcoming": sum(1 for m in self._meetings.values()
                                    if m.start > datetime.now()),
            "kat_says": "Ready when you are, boss."
        }

    def _handle_briefing(self, params: Dict) -> Dict:
        """Morning briefing for Tom"""
        now = datetime.now()
        today = now.date()

        # Get today's meetings efficiently
        todays_meetings = sorted(
            [m for m in self._meetings.values() if m.start.date() == today],
            key=lambda m: m.start
        )

        # Get urgent tasks
        urgent_tasks = sorted(
            [t for t in self._tasks.values()
             if not t.done and t.priority.value <= TaskPriority.HIGH.value],
            key=lambda t: t.priority.value
        )

        total_meeting_hours = sum(m.duration_min for m in todays_meetings) / 60

        # Build charming briefing
        greeting = self._get_greeting(now)

        return {
            "status": "success",
            "greeting": greeting,
            "date": now.strftime("%A, %B %d, %Y"),
            "summary": {
                "meetings": len(todays_meetings),
                "meeting_hours": f"{total_meeting_hours:.1f}",
                "urgent_tasks": len(urgent_tasks),
            },
            "meetings": [m.to_dict() for m in todays_meetings[:5]],
            "priority_tasks": [t.to_dict() for t in urgent_tasks[:5]],
            "kat_says": self._generate_briefing_message(
                len(todays_meetings), total_meeting_hours, len(urgent_tasks)
            )
        }

    def _get_greeting(self, now: datetime) -> str:
        """Get time-appropriate greeting"""
        hour = now.hour
        if hour < 12:
            return "Good morning, Tom!"
        elif hour < 17:
            return "Good afternoon."
        else:
            return "Good evening, Tom."

    def _generate_briefing_message(self, meetings: int, hours: float, tasks: int) -> str:
        """Generate personalized briefing message"""
        if hours > 6:
            return (f"Tom, you've got a packed day - {meetings} meetings totaling "
                   f"{hours:.1f} hours. I've flagged {tasks} priorities. "
                   f"Let me know if you want me to run interference on anything. "
                   f"I've got your back.")
        elif meetings == 0:
            return (f"Lucky you! Clear calendar today. {tasks} tasks to tackle. "
                   f"Perfect day to crush it. I'll keep interruptions at bay.")
        else:
            return (f"Looking good - {meetings} meetings, {tasks} priorities. "
                   f"Manageable. Let me know what you need.")

    def _handle_calendar(self, params: Dict) -> Dict:
        """View calendar"""
        days = params.get("days", 1)
        start = datetime.now()
        end = start + timedelta(days=days)

        relevant = sorted(
            [m for m in self._meetings.values() if start <= m.start <= end],
            key=lambda m: m.start
        )

        return {
            "status": "success",
            "period": f"{start.date()} to {end.date()}",
            "meetings": [m.to_dict() for m in relevant],
            "kat_says": f"Here's your calendar for the next {days} day(s). "
                       f"{len(relevant)} meetings on the books."
        }

    def _handle_availability(self, params: Dict) -> Dict:
        """Check availability"""
        date_str = params.get("date", datetime.now().date().isoformat())
        duration = params.get("duration", self.DEFAULT_MEETING_DURATION)

        check_date = datetime.fromisoformat(date_str).date()
        day_meetings = sorted(
            [m for m in self._meetings.values() if m.start.date() == check_date],
            key=lambda m: m.start
        )

        # Find gaps
        slots = []
        work_start = datetime.combine(check_date, datetime.min.time().replace(hour=self.WORK_START))
        work_end = datetime.combine(check_date, datetime.min.time().replace(hour=self.WORK_END))

        current = work_start
        for m in day_meetings:
            if (m.start - current).total_seconds() / 60 >= duration:
                slots.append({
                    "start": current.strftime("%I:%M %p"),
                    "end": m.start.strftime("%I:%M %p"),
                    "duration": int((m.start - current).total_seconds() / 60)
                })
            current = max(current, m.end)

        if (work_end - current).total_seconds() / 60 >= duration:
            slots.append({
                "start": current.strftime("%I:%M %p"),
                "end": work_end.strftime("%I:%M %p"),
                "duration": int((work_end - current).total_seconds() / 60)
            })

        return {
            "status": "success",
            "date": str(check_date),
            "available_slots": slots,
            "kat_says": f"Found {len(slots)} available slots on {check_date}. "
                       f"Want me to hold any for you?"
        }

    def _handle_schedule(self, params: Dict) -> Dict:
        """Schedule meeting (requires permission)"""
        title = params.get("title", "Meeting")
        start_str = params.get("start")
        duration = params.get("duration", self.DEFAULT_MEETING_DURATION)
        attendees = tuple(params.get("attendees", []))

        start = datetime.fromisoformat(start_str) if start_str else datetime.now()
        end = start + timedelta(minutes=duration)

        meeting_id = hashlib.sha256(f"{title}{start}".encode()).hexdigest()[:8]

        meeting = Meeting(
            id=meeting_id,
            title=title,
            start=start,
            end=end,
            attendees=attendees
        )

        self._meetings[meeting_id] = meeting

        return {
            "status": "success",
            "meeting": meeting.to_dict(),
            "kat_says": f"Done! '{title}' is on your calendar for "
                       f"{start.strftime('%B %d at %I:%M %p')}. "
                       f"I'll send invites to {len(attendees)} people."
        }

    def _handle_tasks(self, params: Dict) -> Dict:
        """View tasks"""
        include_done = params.get("include_done", False)

        tasks = [t for t in self._tasks.values()
                if include_done or not t.done]
        tasks.sort(key=lambda t: (t.priority.value, t.created))

        return {
            "status": "success",
            "tasks": [t.to_dict() for t in tasks],
            "kat_says": f"{len(tasks)} tasks on your list. Top priority first."
        }

    def _handle_add_task(self, params: Dict) -> Dict:
        """Add task"""
        title = params.get("title", "New Task")
        priority = TaskPriority[params.get("priority", "MEDIUM").upper()]
        due = params.get("due")

        task_id = hashlib.sha256(f"{title}{datetime.now()}".encode()).hexdigest()[:8]

        task = Task(
            id=task_id,
            title=title,
            priority=priority,
            created=datetime.now(),
            due=datetime.fromisoformat(due) if due else None
        )

        self._tasks[task_id] = task

        return {
            "status": "success",
            "task": task.to_dict(),
            "kat_says": f"Got it! '{title}' added as {priority.name} priority. "
                       f"I'll keep an eye on it for you."
        }

    def _handle_prioritize(self, params: Dict) -> Dict:
        """Smart task prioritization"""
        active_tasks = [t for t in self._tasks.values() if not t.done]

        def urgency(t: Task) -> int:
            score = t.priority.value * 10
            if t.due:
                days = (t.due - datetime.now()).days
                if days < 0:
                    score -= 50  # Overdue
                elif days == 0:
                    score -= 30
                elif days <= 2:
                    score -= 15
            return score

        active_tasks.sort(key=urgency)

        return {
            "status": "success",
            "prioritized": [
                {"rank": i+1, **t.to_dict(), "urgency_score": -urgency(t)}
                for i, t in enumerate(active_tasks[:10])
            ],
            "kat_says": "Here's what needs your attention most, ranked by urgency. "
                       "I'd start with #1 if I were you."
        }

    def _handle_draft_email(self, params: Dict) -> Dict:
        """Draft email"""
        to = params.get("to", "")
        subject = params.get("subject", "")
        points = params.get("points", [])

        body = "\n".join(f"• {p}" for p in points) if points else "[Your message here]"

        draft = f"To: {to}\nSubject: {subject}\n\n{body}\n\nBest,\nTom"

        return {
            "status": "success",
            "draft": draft,
            "kat_says": "Draft ready for your review. Want me to adjust the tone "
                       "or add anything? Just say the word."
        }

    def _handle_draft_message(self, params: Dict) -> Dict:
        """Draft message"""
        to = params.get("to", "")
        content = params.get("content", "")

        return {
            "status": "success",
            "draft": f"To {to}: {content}",
            "kat_says": "Message drafted. Ready to send when you give the green light."
        }

    def _handle_research(self, params: Dict) -> Dict:
        """Handle research request"""
        topic = params.get("topic", "")

        return {
            "status": "success",
            "topic": topic,
            "kat_says": f"I'll dig into '{topic}' and have a summary for you shortly. "
                       f"Anything specific you want me to focus on?",
            "delegation": "Delegating to MARGOT_ROBBIE for deep research."
        }

    def _handle_grant_permission(self, params: Dict) -> Dict:
        """Grant permission for an action"""
        action = params.get("action", "")

        if action:
            self._written_permissions.add(action)
            return {
                "status": "success",
                "permission_granted": action,
                "kat_says": f"Permission noted! I can now help with '{action}'. "
                           f"Thank you for trusting me, Tom."
            }

        return {"status": "error", "message": "No action specified"}

    def _handle_audit_log(self, params: Dict) -> Dict:
        """View audit log"""
        limit = params.get("limit", 50)

        recent = self._audit_log[-limit:]

        return {
            "status": "success",
            "entries": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "action": e.action.name,
                    "target": e.target,
                    "permission": e.permission.name,
                    "result": e.result
                }
                for e in recent
            ],
            "total_entries": len(self._audit_log),
            "kat_says": f"Here's the audit trail - {len(recent)} recent entries. "
                       f"Full transparency, always."
        }

    def _handle_unknown(self, params: Dict) -> Dict:
        """Handle unknown action"""
        return {
            "status": "clarification_needed",
            "kat_says": "Tom, I'm not quite sure what you need. "
                       "Calendar? Tasks? Email drafts? Research? "
                       "Tell me more and I'll make it happen."
        }

    def _log_communication(self, action: str, result: Dict):
        """Log communication (capped)"""
        self._tom_communications.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "success": result.get("status") == "success"
        })

        if len(self._tom_communications) > self._max_comm_log:
            self._tom_communications = self._tom_communications[-500:]

    def get_natural_language_explanation(self) -> str:
        return """
================================================================================
KAT - Tom Hogan's Executive Assistant
================================================================================

Hey Tom! I'm KAT, your executive assistant. I'm here to make your life
easier - handling calendar, tasks, communications, and whatever else you need.

SECURITY (Because I respect your privacy):
   - READ-ONLY by default - I can view and draft, but not act
   - Need your written permission for any actions
   - Full audit trail on everything I do
   - Your trust matters to me

CALENDAR & SCHEDULING
   - View your schedule anytime
   - Draft meeting requests
   - Find availability
   - Prep materials

TASKS & PRIORITIES
   - Track everything on your plate
   - Smart prioritization
   - Deadline monitoring

COMMUNICATION
   - Draft emails in your voice
   - Prepare messages
   - Research and summarize

COORDINATION
   - Work with MARGOT_ROBBIE on research
   - Work with ANNA_KENDRICK on scheduling
   - Liaise with other agents

Just say the word, boss. I've got you.

- KAT
================================================================================
"""


# =============================================================================
# SINGLETON
# =============================================================================

_kat_instance: Optional[KatAgent] = None


def get_kat() -> KatAgent:
    """Get singleton KAT instance"""
    global _kat_instance
    if _kat_instance is None:
        _kat_instance = KatAgent()
    return _kat_instance


if __name__ == "__main__":
    kat = get_kat()
    print(kat.get_natural_language_explanation())
    print("\nStatus:", kat.process({"action": "status"}))

