"""
================================================================================
ANNA_KENDRICK - Co-Executive Assistant (Admin & Scheduling)
================================================================================
Author: Alpha Loop Capital, LLC

ANNA_KENDRICK is a co-EA reporting to both KAT and SHYLA.
Specializes in administrative tasks, scheduling, and coordination.

Reports To: KAT (Tom's EA) & SHYLA (Chris's EA)
Tier: SUPPORT (3)

Personality: Witty, organized, and delightfully efficient.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.core.agent_base import AgentTier, BaseAgent, LearningMethod, ThinkingMode

logger = logging.getLogger(__name__)


class EventType(Enum):
    MEETING = auto()
    CALL = auto()
    TRAVEL = auto()
    DEADLINE = auto()
    REMINDER = auto()
    BLOCK = auto()


class AdminTaskType(Enum):
    FILING = auto()
    BOOKING = auto()
    COORDINATION = auto()
    DOCUMENTATION = auto()
    EXPENSE = auto()
    TRAVEL = auto()


@dataclass
class ScheduleRequest:
    """Schedule coordination request"""

    id: str
    type: EventType
    title: str
    participants: Tuple[str, ...]
    duration: int  # minutes
    requester: str
    created: datetime
    status: str = "pending"
    proposed_times: List[datetime] = None

    def __post_init__(self):
        if self.proposed_times is None:
            self.proposed_times = []


@dataclass
class AdminTask:
    """Administrative task"""

    id: str
    type: AdminTaskType
    description: str
    requester: str
    created: datetime
    completed: Optional[datetime] = None
    priority: int = 2  # 0=critical, 1=high, 2=medium, 3=low


class AnnaKendrickAgent(BaseAgent):
    """
    ANNA_KENDRICK - Admin & Scheduling Specialist

    Master scheduler and admin wizard. Reports to both KAT and SHYLA.
    """

    def __init__(self):
        super().__init__(
            name="ANNA_KENDRICK",
            tier=AgentTier.SUPPORT,
            capabilities=[
                # Scheduling
                "meeting_scheduling", "calendar_coordination", "conflict_resolution",
                "availability_finding", "time_zone_management", "recurring_events",

                # Admin
                "document_filing", "expense_tracking", "travel_booking",
                "vendor_coordination", "supply_ordering", "office_management",

                # Coordination
                "multi_party_scheduling", "event_planning", "reminder_setting",
                "follow_up_tracking", "deadline_management",
            ],
            user_id="SHARED",
            thinking_modes=[
                ThinkingMode.STRUCTURAL,
                ThinkingMode.PROBABILISTIC,
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,
                LearningMethod.CONTEXTUAL,
            ],
        )

        # Scheduling data
        self._schedule_requests: Dict[str, ScheduleRequest] = {}
        self._admin_tasks: Dict[str, AdminTask] = {}

        # Calendar cache (for conflict detection)
        self._calendar_cache: Dict[str, List[Tuple[datetime, datetime]]] = {
            "TOM": [],
            "CHRIS": [],
        }

        logger.info("ANNA_KENDRICK initialized - Admin & Scheduling specialist ready 📅")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process scheduling or admin request"""
        action = inputs.get("action", "status")
        requester = inputs.get("requester", "UNKNOWN")

        handlers = {
            "status": self._handle_status,
            "find_time": self._handle_find_time,
            "schedule": self._handle_schedule,
            "coordinate": self._handle_coordinate,
            "admin_task": self._handle_admin_task,
            "travel": self._handle_travel,
            "reminder": self._handle_reminder,
            "queue": self._handle_queue,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(inputs, requester)

    def _handle_status(self, params: Dict, requester: str) -> Dict:
        """Get status"""
        pending = sum(1 for r in self._schedule_requests.values() if r.status == "pending")

        return {
            "status": "success",
            "agent": "ANNA_KENDRICK",
            "role": "Co-EA - Admin & Scheduling",
            "reports_to": ["KAT", "SHYLA"],
            "pending_schedule_requests": pending,
            "admin_tasks": len([t for t in self._admin_tasks.values() if not t.completed]),
            "anna_says": "Calendars synced, tasks organized, ready to coordinate. "
                        "What do you need scheduled? I can make anything work. 😊"
        }

    def _handle_find_time(self, params: Dict, requester: str) -> Dict:
        """Find available time slots"""
        participants = params.get("participants", [])
        duration = params.get("duration", 30)
        date_range = params.get("date_range", 7)  # days
        preferred_times = params.get("preferred_times", [])  # morning, afternoon, etc.

        # Find available slots
        start_date = datetime.now()
        end_date = start_date + timedelta(days=date_range)

        slots = self._find_available_slots(
            participants=tuple(participants),
            duration=duration,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "status": "success",
            "participants": participants,
            "duration": duration,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "available_slots": [
                {
                    "start": slot.strftime("%A, %B %d at %I:%M %p"),
                    "end": (slot + timedelta(minutes=duration)).strftime("%I:%M %p")
                }
                for slot in slots[:10]
            ],
            "anna_says": f"Found {len(slots)} available slots for {len(participants)} people. "
                        f"Here are the top options. Want me to send invites? 📅"
        }

    def _find_available_slots(self, participants: Tuple[str, ...], duration: int,
                              start_date: datetime, end_date: datetime) -> List[datetime]:
        """Find available time slots for all participants"""
        # Simplified implementation - would integrate with real calendars
        slots = []
        current = start_date.replace(hour=9, minute=0, second=0, microsecond=0)

        while current < end_date:
            # Skip weekends
            if current.weekday() < 5:  # Monday-Friday
                # Check business hours (9 AM - 6 PM)
                for hour in range(9, 18):
                    slot = current.replace(hour=hour)
                    if slot > datetime.now():
                        # Check conflicts (simplified)
                        if not self._has_conflict(participants, slot, duration):
                            slots.append(slot)

            current += timedelta(days=1)

        return slots[:20]  # Return top 20

    def _has_conflict(self, participants: Tuple[str, ...], start: datetime, duration: int) -> bool:
        """Check for calendar conflicts"""
        end = start + timedelta(minutes=duration)

        for person in participants:
            person_key = person.upper()
            if person_key in self._calendar_cache:
                for event_start, event_end in self._calendar_cache[person_key]:
                    # Check overlap
                    if start < event_end and end > event_start:
                        return True

        return False

    def _handle_schedule(self, params: Dict, requester: str) -> Dict:
        """Schedule an event"""
        event_type = EventType[params.get("type", "MEETING").upper()]
        title = params.get("title", "Meeting")
        participants = params.get("participants", [])
        start_time = params.get("start_time")
        duration = params.get("duration", 30)

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        request_id = hashlib.sha256(f"{title}{start_time}{datetime.now()}".encode()).hexdigest()[:8]

        request = ScheduleRequest(
            id=request_id,
            type=event_type,
            title=title,
            participants=tuple(participants),
            duration=duration,
            requester=requester,
            created=datetime.now(),
            status="scheduled",
            proposed_times=[start_time] if start_time else []
        )

        self._schedule_requests[request_id] = request

        # Add to calendar cache
        if start_time:
            end_time = start_time + timedelta(minutes=duration)
            for p in participants:
                p_key = p.upper()
                if p_key in self._calendar_cache:
                    self._calendar_cache[p_key].append((start_time, end_time))

        return {
            "status": "success",
            "request_id": request_id,
            "event": {
                "type": event_type.name,
                "title": title,
                "start": start_time.strftime("%A, %B %d at %I:%M %p") if start_time else "TBD",
                "duration": duration,
                "participants": participants
            },
            "anna_says": f"Done! '{title}' is on the calendar. "
                        f"I'll send invites to {len(participants)} people and set reminders. "
                        f"Anything else I can help with? 📆"
        }

    def _handle_coordinate(self, params: Dict, requester: str) -> Dict:
        """Coordinate multi-party scheduling"""
        title = params.get("title", "")
        participants = params.get("participants", [])
        urgency = params.get("urgency", "normal")  # urgent, normal, flexible

        # Generate coordination request
        request_id = hashlib.sha256(f"{title}{datetime.now()}".encode()).hexdigest()[:8]

        # Find best times
        slots = self._find_available_slots(
            participants=tuple(participants),
            duration=params.get("duration", 30),
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=14)
        )

        return {
            "status": "success",
            "request_id": request_id,
            "coordination": {
                "title": title,
                "participants": participants,
                "urgency": urgency,
                "proposed_times": [s.isoformat() for s in slots[:5]],
                "next_steps": "Awaiting participant responses"
            },
            "anna_says": f"I've found {len(slots)} potential times for '{title}'. "
                        f"I'll poll {len(participants)} participants and get back to you. "
                        f"Leave the herding of cats to me! 🐱"
        }

    def _handle_admin_task(self, params: Dict, requester: str) -> Dict:
        """Handle administrative task"""
        task_type = AdminTaskType[params.get("type", "DOCUMENTATION").upper()]
        description = params.get("description", "")
        priority = params.get("priority", 2)

        task_id = hashlib.sha256(f"{description}{datetime.now()}".encode()).hexdigest()[:8]

        task = AdminTask(
            id=task_id,
            type=task_type,
            description=description,
            requester=requester,
            created=datetime.now(),
            priority=priority
        )

        self._admin_tasks[task_id] = task

        return {
            "status": "success",
            "task_id": task_id,
            "task": {
                "type": task_type.name,
                "description": description,
                "priority": ["Critical", "High", "Medium", "Low"][priority]
            },
            "anna_says": f"Got it! {task_type.name} task logged. "
                        f"I'll take care of '{description}'. Consider it handled! ✅"
        }

    def _handle_travel(self, params: Dict, requester: str) -> Dict:
        """Handle travel arrangements"""
        traveler = params.get("traveler", "")
        destination = params.get("destination", "")
        departure = params.get("departure", "")
        return_date = params.get("return", "")
        purpose = params.get("purpose", "business")

        return {
            "status": "pending_approval",
            "travel_request": {
                "traveler": traveler,
                "destination": destination,
                "departure": departure,
                "return": return_date,
                "purpose": purpose
            },
            "anna_says": f"Travel to {destination} - got it! I'll research flight options, "
                        f"hotels near your meetings, and ground transportation. "
                        f"I'll have 3 itinerary options ready for review. ✈️"
        }

    def _handle_reminder(self, params: Dict, requester: str) -> Dict:
        """Set reminder"""
        title = params.get("title", "")
        when = params.get("when", "")
        for_whom = params.get("for", requester)

        return {
            "status": "success",
            "reminder": {
                "title": title,
                "when": when,
                "for": for_whom
            },
            "anna_says": f"Reminder set! I'll make sure '{title}' doesn't slip "
                        f"through the cracks. I've got your back! ⏰"
        }

    def _handle_queue(self, params: Dict, requester: str) -> Dict:
        """View pending requests"""
        pending_schedules = [
            {"id": r.id, "title": r.title, "status": r.status}
            for r in self._schedule_requests.values() if r.status == "pending"
        ]

        pending_admin = [
            {"id": t.id, "type": t.type.name, "description": t.description[:50]}
            for t in self._admin_tasks.values() if not t.completed
        ]

        return {
            "status": "success",
            "pending_schedules": pending_schedules[:10],
            "pending_admin": pending_admin[:10],
            "anna_says": f"{len(pending_schedules)} scheduling requests, "
                        f"{len(pending_admin)} admin tasks in my queue. Cranking through them! 💪"
        }

    def _handle_unknown(self, params: Dict, requester: str) -> Dict:
        return {
            "status": "clarification_needed",
            "anna_says": "What do you need? Scheduling? Travel? Admin stuff? "
                        "I can coordinate anything - just point me in the right direction! 😊"
        }


# Singleton
_anna_instance: Optional[AnnaKendrickAgent] = None


def get_anna() -> AnnaKendrickAgent:
    global _anna_instance
    if _anna_instance is None:
        _anna_instance = AnnaKendrickAgent()
    return _anna_instance

