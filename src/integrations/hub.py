"""
================================================================================
INTEGRATION HUB - Unified Integration Controller
================================================================================
Central hub for managing all external integrations.

Provides:
- Unified messaging across platforms
- Calendar synchronization
- Document management
- Real-time notifications
- Learning from interactions
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from src.integrations.slack_client import get_slack_client
from src.integrations.notion_client import get_notion_client
from src.integrations.outlook_client import get_outlook_client

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported platforms"""
    SLACK = auto()
    NOTION = auto()
    OUTLOOK = auto()
    TEAMS = auto()
    APP = auto()  # Internal Alpha Loop Hub


class MessageType(Enum):
    """Types of messages"""
    NOTIFICATION = auto()
    CHAT = auto()
    EMAIL = auto()
    CALENDAR = auto()
    DOCUMENT = auto()
    ALERT = auto()


@dataclass
class UnifiedMessage:
    """Platform-agnostic message"""
    id: str
    type: MessageType
    sender: str
    recipient: str
    content: str
    platform: Platform
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    delivered: bool = False


@dataclass
class InteractionLog:
    """Log of user interactions for learning"""
    timestamp: datetime
    user: str
    platform: Platform
    action: str
    context: Dict[str, Any]
    outcome: str
    feedback: Optional[str] = None


class IntegrationHub:
    """
    Central hub for all integrations

    Provides unified interface for:
    - Multi-platform messaging
    - Calendar synchronization
    - Document management
    - Learning from interactions
    """

    def __init__(self):
        # Initialize clients
        self.slack = get_slack_client()
        self.notion = get_notion_client()
        self.outlook = get_outlook_client()

        # Connection status
        self._connected: Dict[Platform, bool] = {
            Platform.SLACK: False,
            Platform.NOTION: False,
            Platform.OUTLOOK: False,
            Platform.TEAMS: False,
            Platform.APP: True,  # Internal app always available
        }

        # Message queue
        self._message_queue: List[UnifiedMessage] = []
        self._sent_messages: List[UnifiedMessage] = []

        # Learning data
        self._interactions: List[InteractionLog] = []
        self._user_preferences: Dict[str, Dict] = {
            "TOM": {
                "preferred_platform": Platform.SLACK,
                "notification_frequency": "realtime",
                "email_digest": False,
            },
            "CHRIS": {
                "preferred_platform": Platform.SLACK,
                "notification_frequency": "realtime",
                "email_digest": False,
            }
        }

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        logger.info("IntegrationHub initialized")

    async def connect_all(self) -> Dict[Platform, bool]:
        """Connect to all platforms"""
        results = {}

        try:
            results[Platform.SLACK] = await self.slack.connect()
            self._connected[Platform.SLACK] = True
        except Exception as e:
            logger.error(f"Slack connection failed: {e}")
            results[Platform.SLACK] = False

        try:
            results[Platform.NOTION] = await self.notion.connect()
            self._connected[Platform.NOTION] = True
        except Exception as e:
            logger.error(f"Notion connection failed: {e}")
            results[Platform.NOTION] = False

        try:
            results[Platform.OUTLOOK] = await self.outlook.connect()
            self._connected[Platform.OUTLOOK] = True
            self._connected[Platform.TEAMS] = True
        except Exception as e:
            logger.error(f"Outlook connection failed: {e}")
            results[Platform.OUTLOOK] = False

        logger.info(f"Integration connections: {results}")
        return results

    def get_status(self) -> Dict:
        """Get hub status"""
        return {
            "connected": {p.name: c for p, c in self._connected.items()},
            "message_queue_size": len(self._message_queue),
            "messages_sent": len(self._sent_messages),
            "interactions_logged": len(self._interactions),
            "users_tracked": list(self._user_preferences.keys())
        }

    # =========================================================================
    # MESSAGING
    # =========================================================================

    async def send_message(self, recipient: str, content: str,
                          platform: Platform = None,
                          message_type: MessageType = MessageType.CHAT,
                          metadata: Dict = None) -> Dict:
        """Send message to any platform"""
        # Determine platform based on user preference or explicit choice
        if platform is None:
            user_prefs = self._user_preferences.get(recipient.upper(), {})
            platform = user_prefs.get("preferred_platform", Platform.SLACK)

        message = UnifiedMessage(
            id=f"msg_{datetime.now().timestamp()}",
            type=message_type,
            sender="SYSTEM",
            recipient=recipient,
            content=content,
            platform=platform,
            metadata=metadata or {}
        )

        # Route to appropriate platform
        result = await self._route_message(message)

        if result.get("success"):
            message.delivered = True
            self._sent_messages.append(message)
        else:
            self._message_queue.append(message)

        return result

    async def _route_message(self, message: UnifiedMessage) -> Dict:
        """Route message to appropriate platform"""
        try:
            if message.platform == Platform.SLACK:
                result = await self.slack.send_message(
                    channel=f"@{message.recipient.lower()}",
                    text=message.content
                )
                return {"success": True, "platform": "slack", **result}

            elif message.platform == Platform.OUTLOOK:
                result = await self.outlook.send_email(
                    to=[message.metadata.get("email", f"{message.recipient.lower()}@alphaloopcapital.com")],
                    subject=message.metadata.get("subject", "Message from Alpha Loop"),
                    body=message.content
                )
                return {"success": True, "platform": "outlook", **result}

            elif message.platform == Platform.TEAMS:
                result = await self.outlook.send_teams_message(
                    channel_or_user=message.recipient,
                    message=message.content
                )
                return {"success": True, "platform": "teams", **result}

            elif message.platform == Platform.NOTION:
                result = await self.notion.create_page(
                    title=message.metadata.get("title", "Message"),
                    content={"body": message.content}
                )
                return {"success": True, "platform": "notion", **result}

            else:
                # Internal app
                return {
                    "success": True,
                    "platform": "app",
                    "message_id": message.id,
                    "delivered": True
                }

        except Exception as e:
            logger.error(f"Message routing failed: {e}")
            return {"success": False, "error": str(e)}

    async def send_notification(self, user: str, title: str, body: str,
                               priority: str = "normal",
                               platforms: List[Platform] = None) -> Dict:
        """Send notification across multiple platforms"""
        if platforms is None:
            user_prefs = self._user_preferences.get(user.upper(), {})
            platforms = [user_prefs.get("preferred_platform", Platform.SLACK)]

        results = {}

        for platform in platforms:
            if platform == Platform.SLACK:
                results["slack"] = await self.slack.send_notification(
                    user=user, title=title, body=body, priority=priority
                )
            elif platform == Platform.OUTLOOK:
                results["email"] = await self.outlook.send_email(
                    to=[f"{user.lower()}@alphaloopcapital.com"],
                    subject=f"[{priority.upper()}] {title}",
                    body=body
                )

        return {
            "user": user,
            "title": title,
            "platforms": list(results.keys()),
            "results": results
        }

    async def broadcast(self, content: str,
                       recipients: List[str] = None,
                       platform: Platform = Platform.SLACK) -> Dict:
        """Broadcast message to multiple recipients"""
        if recipients is None:
            recipients = list(self._user_preferences.keys())

        results = []
        for recipient in recipients:
            result = await self.send_message(
                recipient=recipient,
                content=content,
                platform=platform
            )
            results.append({"recipient": recipient, **result})

        return {
            "broadcast": True,
            "recipients_count": len(recipients),
            "results": results
        }

    # =========================================================================
    # CALENDAR
    # =========================================================================

    async def sync_calendars(self, user: str) -> Dict:
        """Sync calendars across platforms"""
        # Get Outlook calendar
        outlook_cal = await self.outlook.get_calendar(user)

        # Would sync with other platforms in production

        return {
            "user": user,
            "synced": True,
            "outlook_events": outlook_cal.get("count", 0),
            "last_sync": datetime.now().isoformat()
        }

    async def create_meeting(self, title: str, organizer: str,
                            attendees: List[str], start: datetime,
                            duration_minutes: int = 30,
                            is_online: bool = True) -> Dict:
        """Create meeting across platforms"""
        end = start + timedelta(minutes=duration_minutes) if duration_minutes else start + timedelta(hours=1)

        # Create Outlook event
        outlook_event = await self.outlook.create_calendar_event(
            user=organizer,
            title=title,
            start=start,
            end=end,
            attendees=attendees,
            is_online=is_online
        )

        # Notify via Slack
        for attendee in attendees:
            await self.slack.send_notification(
                user=attendee,
                title="New Meeting",
                body=f"*{title}*\n{start.strftime('%B %d at %I:%M %p')}\nOrganized by {organizer}"
            )

        return {
            "meeting_created": True,
            "title": title,
            "start": start.isoformat(),
            "attendees": attendees,
            "outlook_event": outlook_event,
            "notifications_sent": len(attendees)
        }

    # =========================================================================
    # DOCUMENTS
    # =========================================================================

    async def create_document(self, title: str, content: Dict,
                             database: str = "docs") -> Dict:
        """Create document in Notion"""
        result = await self.notion.create_page(
            title=title,
            content=content,
            parent_database=database
        )

        return {
            "document_created": True,
            **result
        }

    # =========================================================================
    # LEARNING
    # =========================================================================

    def log_interaction(self, user: str, platform: Platform,
                       action: str, context: Dict,
                       outcome: str, feedback: str = None):
        """Log interaction for learning"""
        log = InteractionLog(
            timestamp=datetime.now(),
            user=user,
            platform=platform,
            action=action,
            context=context,
            outcome=outcome,
            feedback=feedback
        )

        self._interactions.append(log)

        # Cap for memory
        if len(self._interactions) > 10000:
            self._interactions = self._interactions[-5000:]

        # Trigger learning event
        self._emit_event("interaction_logged", log)

    def get_user_preferences(self, user: str) -> Dict:
        """Get user preferences"""
        return self._user_preferences.get(user.upper(), {})

    def update_user_preferences(self, user: str, preferences: Dict):
        """Update user preferences"""
        user_key = user.upper()
        if user_key not in self._user_preferences:
            self._user_preferences[user_key] = {}

        self._user_preferences[user_key].update(preferences)

    def analyze_interactions(self, user: str = None,
                            days: int = 7) -> Dict:
        """Analyze user interactions for insights"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)

        relevant = [
            i for i in self._interactions
            if i.timestamp > cutoff and (user is None or i.user == user.upper())
        ]

        # Analyze patterns
        by_platform = {}
        by_action = {}

        for interaction in relevant:
            platform = interaction.platform.name
            by_platform[platform] = by_platform.get(platform, 0) + 1

            action = interaction.action
            by_action[action] = by_action.get(action, 0) + 1

        return {
            "period_days": days,
            "total_interactions": len(relevant),
            "by_platform": by_platform,
            "by_action": by_action,
            "insights": self._generate_insights(relevant)
        }

    def _generate_insights(self, interactions: List[InteractionLog]) -> List[str]:
        """Generate insights from interactions"""
        insights = []

        if len(interactions) > 100:
            insights.append("High engagement - consider automated workflows")

        # Would have more sophisticated analysis in production

        return insights

    # =========================================================================
    # EVENTS
    # =========================================================================

    def on_event(self, event_name: str, handler: Callable):
        """Register event handler"""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    def _emit_event(self, event_name: str, data: Any):
        """Emit event to handlers"""
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")


# Import timedelta for the file
from datetime import timedelta

# Singleton
_hub_instance: Optional[IntegrationHub] = None


def get_integration_hub() -> IntegrationHub:
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = IntegrationHub()
    return _hub_instance

