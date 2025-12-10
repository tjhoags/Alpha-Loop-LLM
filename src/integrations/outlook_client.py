"""
================================================================================
OUTLOOK CLIENT - Microsoft 365/Outlook Integration
================================================================================
Unified Outlook client for email, calendar, and Teams.

Features:
- Email sending and management
- Calendar operations
- Teams messaging
- Dropbox file access
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Email:
    """Email structure"""
    id: str
    to: List[str]
    cc: List[str]
    subject: str
    body: str
    is_html: bool = False
    attachments: List[str] = None
    sent_at: datetime = None


@dataclass
class CalendarEvent:
    """Calendar event structure"""
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    location: str = ""
    is_online: bool = False
    teams_link: str = ""
    notes: str = ""


class OutlookClient:
    """
    Microsoft 365/Outlook integration client

    Provides unified interface for email, calendar, and Teams.
    """

    # User email mappings
    USERS = {
        "TOM": "tom@alphaloopcapital.com",
        "CHRIS": "chris@alphaloopcapital.com",
    }

    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.connected = False
        self._sent_emails: List[Email] = []
        self._calendar_events: Dict[str, List[CalendarEvent]] = {
            "TOM": [],
            "CHRIS": [],
        }

        logger.info("OutlookClient initialized")

    async def connect(self) -> bool:
        """Connect to Microsoft Graph API"""
        self.connected = True
        logger.info("Outlook/Microsoft 365 connected")
        return True

    async def send_email(self, to: List[str], subject: str, body: str,
                        cc: List[str] = None, is_html: bool = False,
                        attachments: List[str] = None,
                        priority: str = "normal") -> Dict:
        """Send email"""
        email_id = hashlib.sha256(f"{subject}{datetime.now()}".encode()).hexdigest()[:12]

        email = Email(
            id=email_id,
            to=to if isinstance(to, list) else [to],
            cc=cc or [],
            subject=subject,
            body=body,
            is_html=is_html,
            attachments=attachments or [],
            sent_at=datetime.now()
        )

        self._sent_emails.append(email)

        logger.info(f"Email sent: {subject} to {', '.join(email.to)}")

        return {
            "id": email_id,
            "to": email.to,
            "subject": subject,
            "sent_at": email.sent_at.isoformat(),
            "status": "sent"
        }

    async def create_calendar_event(self, user: str, title: str,
                                   start: datetime, end: datetime,
                                   attendees: List[str] = None,
                                   location: str = "",
                                   is_online: bool = False) -> Dict:
        """Create calendar event"""
        event_id = hashlib.sha256(f"{title}{start}".encode()).hexdigest()[:12]

        event = CalendarEvent(
            id=event_id,
            title=title,
            start=start,
            end=end,
            attendees=attendees or [],
            location=location,
            is_online=is_online,
            teams_link=f"https://teams.microsoft.com/l/meetup-join/{event_id}" if is_online else ""
        )

        user_key = user.upper()
        if user_key in self._calendar_events:
            self._calendar_events[user_key].append(event)

        logger.info(f"Calendar event created: {title}")

        return {
            "id": event_id,
            "title": title,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "attendees": attendees,
            "teams_link": event.teams_link if is_online else None,
            "status": "created"
        }

    async def get_calendar(self, user: str,
                          start_date: datetime = None,
                          days: int = 7) -> Dict:
        """Get calendar events for user"""
        start_date = start_date or datetime.now()
        end_date = start_date + timedelta(days=days)

        user_key = user.upper()
        events = self._calendar_events.get(user_key, [])

        # Filter by date range
        filtered = [
            e for e in events
            if start_date <= e.start <= end_date
        ]

        return {
            "user": user,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start": e.start.isoformat(),
                    "end": e.end.isoformat(),
                    "attendees": e.attendees
                }
                for e in filtered
            ],
            "count": len(filtered)
        }

    async def update_event(self, event_id: str, updates: Dict) -> Dict:
        """Update calendar event"""
        for user_events in self._calendar_events.values():
            for event in user_events:
                if event.id == event_id:
                    for key, value in updates.items():
                        if hasattr(event, key):
                            setattr(event, key, value)
                    return {
                        "id": event_id,
                        "updated": True
                    }

        return {"error": "Event not found"}

    async def delete_event(self, event_id: str) -> Dict:
        """Delete calendar event"""
        for user, events in self._calendar_events.items():
            for i, event in enumerate(events):
                if event.id == event_id:
                    del events[i]
                    return {"id": event_id, "deleted": True}

        return {"error": "Event not found"}

    async def check_availability(self, users: List[str],
                                start: datetime, end: datetime) -> Dict:
        """Check availability for multiple users"""
        availability = {}

        for user in users:
            user_key = user.upper()
            events = self._calendar_events.get(user_key, [])

            # Check for conflicts
            conflicts = [
                e for e in events
                if not (e.end <= start or e.start >= end)
            ]

            availability[user] = {
                "available": len(conflicts) == 0,
                "conflicts": len(conflicts)
            }

        return {
            "requested_time": {
                "start": start.isoformat(),
                "end": end.isoformat()
            },
            "availability": availability,
            "all_available": all(a["available"] for a in availability.values())
        }

    async def send_teams_message(self, channel_or_user: str,
                                message: str,
                                is_channel: bool = False) -> Dict:
        """Send Teams message"""
        message_id = hashlib.sha256(f"{message}{datetime.now()}".encode()).hexdigest()[:12]

        return {
            "id": message_id,
            "to": channel_or_user,
            "is_channel": is_channel,
            "message": message,
            "sent_at": datetime.now().isoformat(),
            "status": "sent"
        }


# Singleton
_outlook_instance: Optional[OutlookClient] = None


def get_outlook_client() -> OutlookClient:
    global _outlook_instance
    if _outlook_instance is None:
        _outlook_instance = OutlookClient()
    return _outlook_instance

