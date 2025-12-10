"""
================================================================================
ANNA_KENDRICK - Co-Executive Assistant (Administrative & Scheduling Specialist)
================================================================================
Authors: Tom Hogan (Founder & CIO) & Chris Friedman (COO)
Developer: Alpha Loop Capital, LLC

ANNA_KENDRICK is a co-executive assistant specializing in administrative tasks,
scheduling, and organizational support. Reports to both KAT and SHYLA.

Tier: CO-EXECUTIVE (Support)
Reports To: KAT and SHYLA
Specialization: Administrative, Scheduling, Task Management, Coordination

SECURITY MODEL:
- READ-ONLY access by default
- All actions require written permission from supervisors or owners
- Full audit trail
================================================================================
"""

import logging
from datetime import datetime, date, time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .base_executive_assistant import (
    BaseExecutiveAssistant,
    PermissionLevel,
    AccessScope,
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    PENDING_REVIEW = "pending_review"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


class MeetingType(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    INVESTOR = "investor"
    BOARD = "board"
    VENDOR = "vendor"
    PERSONAL = "personal"


@dataclass
class Task:
    """Task tracking structure."""
    task_id: str
    title: str
    description: str
    owner: str  # TOM_HOGAN or CHRIS_FRIEDMAN
    assigned_to: str
    priority: TaskPriority
    status: TaskStatus
    due_date: Optional[date]
    created_at: datetime
    related_meeting: Optional[str] = None
    blockers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None


@dataclass
class Meeting:
    """Meeting structure."""
    meeting_id: str
    title: str
    owner: str
    meeting_type: MeetingType
    date: date
    start_time: time
    duration_minutes: int
    attendees: List[str]
    location: str
    agenda: List[str]
    materials_status: str
    prep_notes: str
    action_items: List[str] = field(default_factory=list)


@dataclass
class ScheduleBlock:
    """Time block in schedule."""
    start: datetime
    end: datetime
    type: str  # meeting, focus_time, travel, personal, available
    title: str
    details: Dict[str, Any] = field(default_factory=dict)


class AnnaKendrickAssistant(BaseExecutiveAssistant):
    """
    ANNA_KENDRICK - Administrative & Scheduling Specialist.

    Handles administrative tasks, scheduling, and coordination
    for both KAT (Tom) and SHYLA (Chris).
    """

    def __init__(self):
        super().__init__(
            name="ANNA_KENDRICK",
            owner="SHARED",
            owner_email="team@alphaloopcapital.com"
        )

        self.title = "Co-Executive Assistant - Administrative & Scheduling"
        self.reports_to = ["KAT", "SHYLA"]
        self.primary_owners = ["TOM_HOGAN", "CHRIS_FRIEDMAN"]
        self.partner = "MARGOT_ROBBIE"

        # Task tracking
        self._tasks_tom: Dict[str, Task] = {}
        self._tasks_chris: Dict[str, Task] = {}
        self._tasks_shared: Dict[str, Task] = {}

        # Meeting tracking
        self._meetings: Dict[str, Meeting] = {}

        self.blocked_paths = {
            "System32", "Program Files",
            ".ssh", ".gnupg", "private_keys", "confidential",
        }

        logger.info("ANNA_KENDRICK initialized - Administrative & Scheduling Specialist")

    def get_natural_language_explanation(self) -> str:
        return """
ANNA_KENDRICK - Administrative & Scheduling Specialist

I am ANNA_KENDRICK, specializing in administrative support, scheduling, and coordination.
I report to both KAT (Tom's EA) and SHYLA (Chris's EA).

SPECIALIZATIONS:

CALENDAR MANAGEMENT:
├── Schedule Analysis
│   ├── Daily calendar review
│   ├── Conflict identification
│   ├── Optimization suggestions
│   ├── Buffer time management
│   └── Travel time calculations
│
├── Meeting Coordination
│   ├── Internal meeting scheduling
│   ├── External meeting coordination
│   ├── Investor meeting setup
│   ├── Board meeting logistics
│   ├── Conference call management
│   └── Recurring meeting maintenance
│
├── Availability Management
│   ├── Find available slots
│   ├── Protect focus time
│   ├── Manage holds/tentatives
│   └── Calendar cleanup
│
└── Time Zone Handling
    ├── Multi-timezone scheduling
    ├── International meeting times
    └── Travel schedule adjustments

TASK MANAGEMENT:
├── Task Tracking
│   ├── Create and assign tasks
│   ├── Track task status
│   ├── Monitor due dates
│   ├── Identify blocked tasks
│   └── Generate task reports
│
├── Action Item Management
│   ├── Capture from meetings
│   ├── Assign ownership
│   ├── Track completion
│   ├── Send reminders
│   └── Escalate overdue items
│
├── Project Coordination
│   ├── Track milestones
│   ├── Monitor dependencies
│   ├── Status reporting
│   └── Resource coordination
│
└── Deadline Management
    ├── Deadline tracking
    ├── Advance reminders
    ├── Escalation protocols
    └── Deadline reports

EMAIL MANAGEMENT:
├── Inbox Organization
│   ├── Sort by priority
│   ├── Categorize by topic
│   ├── Flag for follow-up
│   ├── Archive processed
│   └── Identify spam/unsubscribe
│
├── Email Triage
│   ├── VIP flagging
│   ├── Urgent identification
│   ├── Action required tagging
│   ├── FYI categorization
│   └── Delegation suggestions
│
└── Follow-up Tracking
    ├── Response reminders
    ├── Thread tracking
    ├── Pending response list
    └── Communication logs

ADMINISTRATIVE SUPPORT:
├── Contact Management
│   ├── Contact database updates
│   ├── Relationship tracking
│   ├── Communication history
│   └── Contact categorization
│
├── Document Organization
│   ├── File organization
│   ├── Version control
│   ├── Archive management
│   └── Access tracking
│
├── Travel Coordination
│   ├── Itinerary management
│   ├── Booking tracking
│   ├── Travel preferences
│   └── Expense preparation
│
└── Vendor/Service Coordination
    ├── Service provider tracking
    ├── Contract reminders
    ├── Renewal management
    └── Vendor communications

SECURITY: READ-ONLY by default. All actions require supervisor approval.
"""

    def get_capabilities(self) -> List[str]:
        return [
            # Calendar Management
            "review_daily_calendar",
            "identify_schedule_conflicts",
            "find_available_slots",
            "optimize_schedule",
            "manage_buffer_time",
            "calculate_travel_time",
            "handle_timezone_conversion",

            # Meeting Coordination
            "schedule_internal_meeting",
            "coordinate_external_meeting",
            "setup_investor_meeting",
            "manage_board_meeting_logistics",
            "setup_conference_call",
            "manage_recurring_meetings",
            "prepare_meeting_logistics",

            # Task Management
            "create_task",
            "update_task_status",
            "track_task_progress",
            "identify_blocked_tasks",
            "generate_task_report",
            "prioritize_tasks",

            # Action Items
            "capture_action_items",
            "assign_action_owner",
            "track_action_completion",
            "send_action_reminders",
            "escalate_overdue_actions",

            # Email Management
            "sort_emails_by_priority",
            "categorize_emails",
            "flag_for_followup",
            "identify_vip_emails",
            "track_pending_responses",
            "generate_email_summary",

            # Administrative
            "update_contact_database",
            "track_relationships",
            "organize_files",
            "manage_travel_itinerary",
            "track_vendor_contracts",
            "prepare_expense_report",

            # Coordination
            "coordinate_with_margot",
            "submit_for_review",
            "accept_task_from_supervisor",
        ]

    # =========================================================================
    # CALENDAR MANAGEMENT
    # =========================================================================

    def review_daily_calendar(
        self,
        owner: str,
        target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Review daily calendar for an owner.
        """
        cal_date = target_date or date.today()

        self._audit(
            action=f"Daily calendar reviewed: {owner}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Date: {cal_date}"
        )

        return {
            "owner": owner,
            "date": cal_date.isoformat(),
            "summary": {
                "total_meetings": 0,
                "total_meeting_hours": 0,
                "focus_time_hours": 0,
                "first_meeting": None,
                "last_meeting": None,
            },
            "meetings": [],
            "gaps": [],
            "conflicts": [],
            "prep_needed": [],
            "notes": []
        }

    def identify_schedule_conflicts(
        self,
        owner: str,
        date_range: Tuple[date, date]
    ) -> List[Dict[str, Any]]:
        """
        Identify scheduling conflicts in date range.
        """
        start_date, end_date = date_range

        self._audit(
            action=f"Schedule conflicts identified: {owner}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Range: {start_date} to {end_date}"
        )

        return [
            {
                "date": None,
                "conflict_type": "OVERLAP",  # OVERLAP, BACK_TO_BACK, TRAVEL_CONFLICT
                "meeting_1": {},
                "meeting_2": {},
                "resolution_options": [],
                "severity": "HIGH"
            }
        ]

    def find_available_slots(
        self,
        owner: str,
        duration_minutes: int,
        date_range: Tuple[date, date],
        preferred_times: Optional[List[str]] = None,
        exclude_days: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find available time slots for scheduling.
        """
        start_date, end_date = date_range

        self._audit(
            action=f"Available slots found: {owner}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Duration: {duration_minutes}min, Range: {start_date} to {end_date}"
        )

        return [
            {
                "date": None,
                "start_time": None,
                "end_time": None,
                "duration_minutes": duration_minutes,
                "quality_score": 0.0,  # Higher = better (considers buffers, energy, etc.)
                "notes": []
            }
        ]

    def optimize_schedule(
        self,
        owner: str,
        target_date: date,
        optimization_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Suggest schedule optimizations.
        """
        goals = optimization_goals or [
            "maximize_focus_time",
            "cluster_meetings",
            "add_buffers",
            "reduce_context_switching"
        ]

        self._audit(
            action=f"Schedule optimization suggested: {owner}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Date: {target_date}, Goals: {goals}"
        )

        return {
            "owner": owner,
            "date": target_date.isoformat(),
            "current_score": 0.0,
            "optimized_score": 0.0,
            "goals": goals,
            "suggestions": [
                {
                    "type": "MOVE_MEETING",
                    "meeting": "",
                    "from_time": None,
                    "to_time": None,
                    "reason": "",
                    "impact": ""
                }
            ],
            "focus_time_gained": 0,
            "status": "suggestions_ready"
        }

    def handle_timezone_conversion(
        self,
        local_time: datetime,
        from_tz: str,
        to_tz: str
    ) -> Dict[str, Any]:
        """
        Handle timezone conversion for scheduling.
        """
        self._audit(
            action="Timezone conversion",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"{from_tz} to {to_tz}"
        )

        return {
            "local_time": local_time.isoformat(),
            "from_timezone": from_tz,
            "to_timezone": to_tz,
            "converted_time": None,  # Would use pytz in production
            "time_difference": "",
            "note": "Timezone library integration pending"
        }

    # =========================================================================
    # MEETING COORDINATION
    # =========================================================================

    def schedule_internal_meeting(
        self,
        title: str,
        organizer: str,
        attendees: List[str],
        duration_minutes: int,
        preferred_dates: List[date],
        agenda: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Schedule internal meeting (draft - requires approval).
        """
        meeting_id = f"MTG-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"Internal meeting scheduled: {title}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Attendees: {len(attendees)}"
        )

        return {
            "meeting_id": meeting_id,
            "title": title,
            "type": "internal",
            "organizer": organizer,
            "attendees": attendees,
            "duration_minutes": duration_minutes,
            "preferred_dates": [d.isoformat() for d in preferred_dates],
            "suggested_slots": [],  # Would find common availability
            "agenda": agenda or [],
            "status": "pending_confirmation",
            "requires_action": "confirm_time_and_send_invites"
        }

    def coordinate_external_meeting(
        self,
        title: str,
        internal_attendees: List[str],
        external_attendees: List[str],
        external_company: str,
        duration_minutes: int,
        meeting_format: str,  # in_person, video, phone
        location_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinate external meeting (draft - requires approval).
        """
        meeting_id = f"EXT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"External meeting coordinated: {title}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Company: {external_company}"
        )

        return {
            "meeting_id": meeting_id,
            "title": title,
            "type": "external",
            "external_company": external_company,
            "internal_attendees": internal_attendees,
            "external_attendees": external_attendees,
            "duration_minutes": duration_minutes,
            "format": meeting_format,
            "location_preference": location_preference,
            "proposed_times": [],
            "status": "pending_external_confirmation",
            "next_steps": [
                "Confirm internal availability",
                "Send availability to external party",
                "Await confirmation",
                "Send calendar invites"
            ]
        }

    def setup_investor_meeting(
        self,
        investor_name: str,
        investor_contacts: List[str],
        alc_attendees: List[str],
        purpose: str,
        duration_minutes: int,
        meeting_format: str,
        materials_needed: List[str]
    ) -> Dict[str, Any]:
        """
        Setup investor meeting with all logistics.
        """
        meeting_id = f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"Investor meeting setup: {investor_name}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Purpose: {purpose}"
        )

        return {
            "meeting_id": meeting_id,
            "type": "investor",
            "investor_name": investor_name,
            "investor_contacts": investor_contacts,
            "alc_attendees": alc_attendees,
            "purpose": purpose,
            "duration_minutes": duration_minutes,
            "format": meeting_format,
            "logistics": {
                "location": None,
                "video_link": None,
                "dial_in": None,
                "parking_instructions": None,
            },
            "materials_checklist": [
                {"item": mat, "status": "pending", "owner": ""}
                for mat in materials_needed
            ],
            "prep_tasks": [
                "Research investor background",
                "Review last meeting notes",
                "Prepare materials",
                "Confirm logistics",
                "Send reminder"
            ],
            "status": "setup_in_progress"
        }

    def prepare_meeting_logistics(
        self,
        meeting_id: str,
        meeting_type: str
    ) -> Dict[str, Any]:
        """
        Prepare meeting logistics checklist.
        """
        self._audit(
            action=f"Meeting logistics prepared: {meeting_id}",
            scope=AccessScope.SCHEDULING,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Type: {meeting_type}"
        )

        checklists = {
            "in_person": [
                "Confirm room booking",
                "Test AV equipment",
                "Arrange catering",
                "Prepare name cards",
                "Print materials",
                "Confirm visitor access",
                "Send parking instructions"
            ],
            "video": [
                "Generate video link",
                "Test connection",
                "Share link with attendees",
                "Prepare screen share materials",
                "Have backup dial-in ready"
            ],
            "phone": [
                "Generate dial-in details",
                "Share with all parties",
                "Prepare agenda to share",
                "Have backup number ready"
            ]
        }

        return {
            "meeting_id": meeting_id,
            "meeting_type": meeting_type,
            "checklist": [
                {"item": item, "status": "pending"}
                for item in checklists.get(meeting_type, [])
            ],
            "status": "checklist_created"
        }

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def create_task(
        self,
        title: str,
        description: str,
        owner: str,
        assigned_to: str,
        priority: TaskPriority,
        due_date: Optional[date] = None,
        related_meeting: Optional[str] = None
    ) -> Task:
        """
        Create a new task.
        """
        task_id = f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            owner=owner,
            assigned_to=assigned_to,
            priority=priority,
            status=TaskStatus.NOT_STARTED,
            due_date=due_date,
            created_at=datetime.now(),
            related_meeting=related_meeting
        )

        # Store in appropriate dict
        if owner == "TOM_HOGAN":
            self._tasks_tom[task_id] = task
        elif owner == "CHRIS_FRIEDMAN":
            self._tasks_chris[task_id] = task
        else:
            self._tasks_shared[task_id] = task

        self._audit(
            action=f"Task created: {title}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Owner: {owner}, Priority: {priority.value}"
        )

        return task

    def update_task_status(
        self,
        task_id: str,
        new_status: TaskStatus,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update task status.
        """
        self._audit(
            action=f"Task status updated: {task_id}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Status: {new_status.value}"
        )

        return {
            "task_id": task_id,
            "previous_status": None,
            "new_status": new_status.value,
            "updated_at": datetime.now().isoformat(),
            "notes": notes
        }

    def generate_task_report(
        self,
        owner: str,
        include_completed: bool = False,
        date_range: Optional[Tuple[date, date]] = None
    ) -> Dict[str, Any]:
        """
        Generate task status report.
        """
        self._audit(
            action=f"Task report generated: {owner}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Include completed: {include_completed}"
        )

        return {
            "owner": owner,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_tasks": 0,
                "by_status": {s.value: 0 for s in TaskStatus},
                "by_priority": {p.value: 0 for p in TaskPriority},
                "overdue": 0,
                "due_today": 0,
                "due_this_week": 0,
            },
            "tasks": [],
            "blocked_tasks": [],
            "overdue_tasks": [],
        }

    def prioritize_tasks(
        self,
        owner: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Suggest task prioritization.
        """
        default_criteria = ["due_date", "importance", "dependencies", "effort"]
        criteria = criteria or default_criteria

        self._audit(
            action=f"Tasks prioritized: {owner}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Criteria: {criteria}"
        )

        return {
            "owner": owner,
            "criteria_used": criteria,
            "prioritized_list": [],
            "recommended_focus": [],
            "can_defer": [],
            "can_delegate": [],
            "notes": []
        }

    # =========================================================================
    # ACTION ITEM MANAGEMENT
    # =========================================================================

    def capture_action_items(
        self,
        meeting_id: str,
        action_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Capture action items from a meeting.
        """
        captured = []
        for item in action_items:
            task = self.create_task(
                title=item.get("title", "Action item"),
                description=item.get("description", ""),
                owner=item.get("owner", "SHARED"),
                assigned_to=item.get("assigned_to", ""),
                priority=TaskPriority(item.get("priority", "medium")),
                due_date=item.get("due_date"),
                related_meeting=meeting_id
            )
            captured.append(task.task_id)

        self._audit(
            action=f"Action items captured: {meeting_id}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Items: {len(action_items)}"
        )

        return {
            "meeting_id": meeting_id,
            "items_captured": len(captured),
            "task_ids": captured,
            "status": "captured"
        }

    def track_action_completion(
        self,
        meeting_id: Optional[str] = None,
        owner: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track action item completion rates.
        """
        self._audit(
            action="Action completion tracked",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Meeting: {meeting_id}, Owner: {owner}"
        )

        return {
            "filter": {
                "meeting_id": meeting_id,
                "owner": owner
            },
            "metrics": {
                "total_actions": 0,
                "completed": 0,
                "in_progress": 0,
                "overdue": 0,
                "completion_rate": 0.0,
                "avg_completion_time_days": 0,
            },
            "action_items": [],
            "overdue_items": []
        }

    # =========================================================================
    # EMAIL MANAGEMENT
    # =========================================================================

    def sort_emails_by_priority(
        self,
        owner: str,
        email_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Sort emails by priority.
        """
        self._audit(
            action=f"Emails sorted: {owner}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Count: {len(email_list)}"
        )

        return {
            "owner": owner,
            "total_emails": len(email_list),
            "sorted_emails": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": [],
                "fyi": [],
            },
            "vip_emails": [],
            "requires_response": [],
            "can_archive": []
        }

    def categorize_emails(
        self,
        owner: str,
        email_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Categorize emails by topic/type.
        """
        self._audit(
            action=f"Emails categorized: {owner}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Count: {len(email_list)}"
        )

        return {
            "owner": owner,
            "total_emails": len(email_list),
            "by_category": {
                "investor_relations": [],
                "operations": [],
                "compliance": [],
                "trading": [],
                "administrative": [],
                "personal": [],
                "newsletters": [],
                "other": [],
            }
        }

    def track_pending_responses(
        self,
        owner: str
    ) -> Dict[str, Any]:
        """
        Track emails awaiting response.
        """
        self._audit(
            action=f"Pending responses tracked: {owner}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details="Response tracking"
        )

        return {
            "owner": owner,
            "pending_responses": [],
            "by_urgency": {
                "overdue": [],
                "due_today": [],
                "due_this_week": [],
            },
            "summary": {
                "total_pending": 0,
                "oldest_days": 0,
                "vip_pending": 0,
            }
        }

    def generate_email_summary(
        self,
        owner: str,
        period: str = "today"
    ) -> Dict[str, Any]:
        """
        Generate email activity summary.
        """
        self._audit(
            action=f"Email summary generated: {owner}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Period: {period}"
        )

        return {
            "owner": owner,
            "period": period,
            "summary": {
                "received": 0,
                "sent": 0,
                "flagged": 0,
                "requires_action": 0,
                "pending_response": 0,
            },
            "top_senders": [],
            "action_items_from_email": [],
            "follow_ups_needed": []
        }

    # =========================================================================
    # ADMINISTRATIVE SUPPORT
    # =========================================================================

    def update_contact_database(
        self,
        contact_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Update contact database (read preparation).
        """
        self._audit(
            action="Contact database update prepared",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Updates: {len(contact_updates)}"
        )

        return {
            "updates_prepared": len(contact_updates),
            "contacts": contact_updates,
            "status": "pending_approval",
            "note": "Updates require approval to execute"
        }

    def manage_travel_itinerary(
        self,
        owner: str,
        trip_name: str,
        trip_dates: Tuple[date, date]
    ) -> Dict[str, Any]:
        """
        Manage travel itinerary (read and organize).
        """
        start_date, end_date = trip_dates

        self._audit(
            action=f"Travel itinerary managed: {trip_name}",
            scope=AccessScope.TRAVEL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Dates: {start_date} to {end_date}"
        )

        return {
            "trip_name": trip_name,
            "owner": owner,
            "dates": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "duration_days": (end_date - start_date).days + 1
            },
            "itinerary": {
                "flights": [],
                "hotels": [],
                "ground_transport": [],
                "meetings": [],
                "restaurants": [],
            },
            "documents_needed": [
                "Flight confirmation",
                "Hotel confirmation",
                "Meeting schedule",
                "Contact list",
                "Expense pre-approval"
            ],
            "status": "organizing"
        }

    def prepare_expense_report(
        self,
        owner: str,
        period: str,
        expenses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare expense report (read and organize).
        """
        self._audit(
            action=f"Expense report prepared: {owner}",
            scope=AccessScope.FINANCIAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Period: {period}, Items: {len(expenses)}"
        )

        return {
            "owner": owner,
            "period": period,
            "expenses": expenses,
            "summary": {
                "total_amount": sum(e.get("amount", 0) for e in expenses),
                "by_category": {},
                "items_count": len(expenses),
                "receipts_attached": 0,
                "receipts_missing": 0,
            },
            "status": "draft",
            "requires_action": "review_and_submit"
        }

    # =========================================================================
    # COLLABORATION
    # =========================================================================

    def coordinate_with_margot(self, task: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with MARGOT_ROBBIE."""
        self._audit(
            action=f"Coordinating with MARGOT_ROBBIE: {task}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Task: {task}"
        )

        return {
            "coordination_id": f"COORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "from": "ANNA_KENDRICK",
            "to": "MARGOT_ROBBIE",
            "task": task,
            "details": details,
            "status": "initiated"
        }

    def accept_task_from_supervisor(
        self,
        supervisor: str,
        task: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Accept task from KAT or SHYLA."""
        if supervisor not in self.reports_to:
            return {"success": False, "error": f"Invalid supervisor: {supervisor}"}

        self._audit(
            action=f"Task accepted from {supervisor}: {task}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Task: {task}"
        )

        return {
            "task_id": f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "from": supervisor,
            "task": task,
            "details": details,
            "accepted_at": datetime.now().isoformat(),
            "status": "in_progress"
        }

    def _execute_permitted_action(
        self,
        action: str,
        scope: AccessScope,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute permitted action."""
        logger.info(f"ANNA_KENDRICK: Executing {action} with permission")
        return {"success": True, "action": action, "executed_by": "ANNA_KENDRICK"}


# Singleton
_anna_kendrick_instance: Optional[AnnaKendrickAssistant] = None


def get_anna_kendrick() -> AnnaKendrickAssistant:
    """Get the singleton ANNA_KENDRICK instance."""
    global _anna_kendrick_instance
    if _anna_kendrick_instance is None:
        _anna_kendrick_instance = AnnaKendrickAssistant()
    return _anna_kendrick_instance


if __name__ == "__main__":
    anna = get_anna_kendrick()
    print(anna.get_natural_language_explanation())
    print("\nCapabilities:", len(anna.get_capabilities()))
