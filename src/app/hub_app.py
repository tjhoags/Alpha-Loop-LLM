"""
================================================================================
ALPHA LOOP HUB - Internal Communication & Control Center
================================================================================
Author: Alpha Loop Capital, LLC

The Alpha Loop Hub is the primary interface for Tom and Chris to communicate
with all agents, view dashboards, and manage operations.

Built by MARKETING and SOFTWARE agents.

Key Features:
- Real-time agent chat
- Dashboard with metrics
- Task management
- Calendar integration
- Learning insights
- Flirty, professional assistant interactions
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles in the app"""
    OWNER = auto()      # Tom or Chris
    AGENT = auto()      # System agents
    ASSISTANT = auto()  # KAT, SHYLA, etc.


class ViewMode(Enum):
    """Dashboard view modes"""
    DASHBOARD = auto()
    CHAT = auto()
    CALENDAR = auto()
    TASKS = auto()
    DOCUMENTS = auto()
    LEARNING = auto()
    SETTINGS = auto()


@dataclass
class User:
    """Hub user"""
    id: str
    name: str
    role: UserRole
    email: str
    preferences: Dict = field(default_factory=dict)
    online: bool = False
    last_active: datetime = None


@dataclass
class ChatMessage:
    """Chat message in the hub"""
    id: str
    sender: str
    recipient: str
    content: str
    timestamp: datetime
    read: bool = False
    agent_response: bool = False
    flirty_level: int = 0  # 0-3 scale


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    type: str
    title: str
    position: Dict  # {row, col, width, height}
    data_source: str
    refresh_rate: int = 60  # seconds


class AlphaLoopHub:
    """
    Alpha Loop Hub - Main Application

    The command center for Alpha Loop Capital.
    """

    # Design system (from MARKETING)
    DESIGN = {
        "colors": {
            "primary": "#1a1a2e",
            "secondary": "#16213e",
            "accent": "#0f3460",
            "highlight": "#e94560",
            "success": "#00d9a5",
            "warning": "#ffc107",
            "error": "#dc3545",
            "text": "#ffffff",
            "text_muted": "#a0a0a0",
            "background": "#0f0f23",
        },
        "fonts": {
            "primary": "'Inter', -apple-system, sans-serif",
            "mono": "'JetBrains Mono', monospace",
        },
        "animations": {
            "fast": "150ms",
            "normal": "300ms",
            "slow": "500ms",
        }
    }

    def __init__(self):
        # Users
        self._users: Dict[str, User] = {
            "TOM": User(
                id="TOM",
                name="Tom Hogan",
                role=UserRole.OWNER,
                email="tom@alphaloopcapital.com",
                preferences={
                    "theme": "dark",
                    "notifications": "realtime",
                    "assistant": "KAT",
                    "flirty_mode": True,
                }
            ),
            "CHRIS": User(
                id="CHRIS",
                name="Chris Friedman",
                role=UserRole.OWNER,
                email="chris@alphaloopcapital.com",
                preferences={
                    "theme": "dark",
                    "notifications": "realtime",
                    "assistant": "SHYLA",
                    "flirty_mode": True,
                }
            ),
        }

        # Chat history
        self._chat_history: Dict[str, List[ChatMessage]] = {
            "TOM": [],
            "CHRIS": [],
        }

        # Dashboard widgets
        self._widgets: List[DashboardWidget] = self._create_default_widgets()

        # Active sessions
        self._sessions: Dict[str, Dict] = {}

        # Learning data
        self._interaction_log: List[Dict] = []
        self._learned_patterns: Dict[str, List] = {
            "TOM": [],
            "CHRIS": [],
        }

        # Agent registry for routing
        self._agents = {
            "KAT": "kat_agent",
            "SHYLA": "shyla_agent",
            "MARGOT_ROBBIE": "co_assistants",
            "ANNA_KENDRICK": "co_assistants",
            "SANTAS_HELPER": "santas_helper_agent",
            "CPA": "cpa_agent",
            "MARKETING": "marketing_agent",
            "SOFTWARE": "software_agent",
            "HOAGS": "hoags_agent",
            "FRIEDS": "operations",
            "GHOST": "ghost_agent",
        }

        logger.info("Alpha Loop Hub initialized")

    def _create_default_widgets(self) -> List[DashboardWidget]:
        """Create default dashboard widgets"""
        return [
            DashboardWidget(
                id="agent_status",
                type="status_grid",
                title="Agent Status",
                position={"row": 0, "col": 0, "width": 2, "height": 1},
                data_source="agents/status"
            ),
            DashboardWidget(
                id="portfolio_summary",
                type="metrics",
                title="Portfolio Summary",
                position={"row": 0, "col": 2, "width": 2, "height": 1},
                data_source="portfolio/summary"
            ),
            DashboardWidget(
                id="upcoming_meetings",
                type="list",
                title="Today's Meetings",
                position={"row": 1, "col": 0, "width": 2, "height": 1},
                data_source="calendar/today"
            ),
            DashboardWidget(
                id="priority_tasks",
                type="list",
                title="Priority Tasks",
                position={"row": 1, "col": 2, "width": 2, "height": 1},
                data_source="tasks/priority"
            ),
            DashboardWidget(
                id="recent_activity",
                type="feed",
                title="Recent Activity",
                position={"row": 2, "col": 0, "width": 4, "height": 1},
                data_source="activity/recent",
                refresh_rate=30
            ),
        ]

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def login(self, user_id: str) -> Dict:
        """User login"""
        user_id = user_id.upper()

        if user_id not in self._users:
            return {"success": False, "error": "User not found"}

        user = self._users[user_id]
        user.online = True
        user.last_active = datetime.now()

        session_id = hashlib.sha256(f"{user_id}{datetime.now()}".encode()).hexdigest()[:16]

        self._sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_active": datetime.now()
        }

        # Get personalized greeting from assistant
        assistant = user.preferences.get("assistant", "KAT")
        greeting = self._get_assistant_greeting(user_id, assistant)

        return {
            "success": True,
            "session_id": session_id,
            "user": {
                "id": user.id,
                "name": user.name,
                "role": user.role.name,
            },
            "preferences": user.preferences,
            "assistant_greeting": greeting,
            "dashboard": self.get_dashboard(user_id)
        }

    def _get_assistant_greeting(self, user_id: str, assistant: str) -> str:
        """Get personalized, flirty greeting from assistant"""
        user = self._users.get(user_id)
        flirty = user.preferences.get("flirty_mode", False) if user else False

        hour = datetime.now().hour

        if user_id == "TOM":
            if flirty:
                if hour < 12:
                    return "Good morning! Ready to crush it today? I've got your briefing ready. - KAT"
                elif hour < 17:
                    return "Hey boss! Afternoon looking productive. What can I help with? - KAT"
                else:
                    return "Evening, Tom! Wrapping up the day? Let me know what you need. - KAT"
            else:
                return f"Good {'morning' if hour < 12 else 'afternoon' if hour < 17 else 'evening'}, Tom. Your briefing is ready. - KAT"

        elif user_id == "CHRIS":
            if flirty:
                if hour < 12:
                    return "Good morning, Chris! Ready to run the world? I'm at your service. - SHYLA"
                elif hour < 17:
                    return "Hey Chris! Afternoon's looking busy but I've got everything under control. - SHYLA"
                else:
                    return "Evening, Chris! Great day? I've got your wrap-up ready. - SHYLA"
            else:
                return f"Good {'morning' if hour < 12 else 'afternoon' if hour < 17 else 'evening'}, Chris. Your dashboard is updated. - SHYLA"

        return "Welcome to Alpha Loop Hub!"

    def logout(self, session_id: str) -> Dict:
        """User logout"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            user_id = session["user_id"]

            if user_id in self._users:
                self._users[user_id].online = False

            del self._sessions[session_id]

            return {"success": True, "message": "See you soon!"}

        return {"success": False, "error": "Session not found"}

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def get_dashboard(self, user_id: str) -> Dict:
        """Get personalized dashboard"""
        user = self._users.get(user_id.upper())

        if not user:
            return {"error": "User not found"}

        return {
            "user": user.name,
            "timestamp": datetime.now().isoformat(),
            "widgets": [
                {
                    "id": w.id,
                    "type": w.type,
                    "title": w.title,
                    "position": w.position,
                    "data": self._get_widget_data(w.data_source, user_id)
                }
                for w in self._widgets
            ],
            "quick_actions": self._get_quick_actions(user_id),
            "notifications": self._get_notifications(user_id),
            "assistant_status": {
                "name": user.preferences.get("assistant", "KAT"),
                "status": "online",
                "message": "Ready to help!"
            }
        }

    def _get_widget_data(self, data_source: str, user_id: str) -> Dict:
        """Get data for dashboard widget"""
        # Would fetch real data in production

        if data_source == "agents/status":
            return {
                "agents_online": 12,
                "agents_total": 86,
                "alerts": 0,
                "status": "all_green"
            }

        elif data_source == "portfolio/summary":
            return {
                "nav": "$125.5M",
                "day_pnl": "+$450K",
                "day_pnl_pct": "+0.36%",
                "ytd": "+12.4%"
            }

        elif data_source == "calendar/today":
            return {
                "meetings": [
                    {"time": "10:00 AM", "title": "Portfolio Review"},
                    {"time": "2:00 PM", "title": "LP Call"},
                    {"time": "4:30 PM", "title": "Strategy Session"},
                ]
            }

        elif data_source == "tasks/priority":
            return {
                "tasks": [
                    {"title": "Review Q4 performance", "priority": "high"},
                    {"title": "Sign K-1 documents", "priority": "high"},
                    {"title": "Approve trade allocations", "priority": "medium"},
                ]
            }

        elif data_source == "activity/recent":
            return {
                "items": [
                    {"time": "5 min ago", "event": "KILLJOY blocked overleveraged trade"},
                    {"time": "15 min ago", "event": "CPA completed K-1 draft review"},
                    {"time": "1 hour ago", "event": "SCOUT identified arbitrage opportunity"},
                ]
            }

        return {}

    def _get_quick_actions(self, user_id: str) -> List[Dict]:
        """Get quick actions for user"""
        return [
            {"id": "briefing", "label": "Morning Briefing", "icon": "📋"},
            {"id": "new_task", "label": "New Task", "icon": "+"},
            {"id": "schedule", "label": "Schedule Meeting", "icon": "CAL"},
            {"id": "message", "label": "Send Message", "icon": "MSG"},
        ]

    def _get_notifications(self, user_id: str) -> List[Dict]:
        """Get notifications for user"""
        return [
            {"id": "1", "type": "info", "message": "NAV published for today", "time": "10 min ago"},
            {"id": "2", "type": "success", "message": "All trades executed successfully", "time": "30 min ago"},
        ]

    # =========================================================================
    # CHAT
    # =========================================================================

    def send_chat(self, user_id: str, message: str,
                 to_agent: str = None) -> Dict:
        """Send chat message"""
        user_id = user_id.upper()
        user = self._users.get(user_id)

        if not user:
            return {"error": "User not found"}

        # Determine recipient
        if to_agent is None:
            to_agent = user.preferences.get("assistant", "KAT")

        # Create message
        msg_id = hashlib.sha256(f"{message}{datetime.now()}".encode()).hexdigest()[:12]

        chat_msg = ChatMessage(
            id=msg_id,
            sender=user_id,
            recipient=to_agent,
            content=message,
            timestamp=datetime.now()
        )

        self._chat_history[user_id].append(chat_msg)

        # Log interaction for learning
        self._log_interaction(user_id, "chat", message, to_agent)

        # Get agent response
        response = self._get_agent_response(user_id, message, to_agent)

        # Create response message
        response_msg = ChatMessage(
            id=f"resp_{msg_id}",
            sender=to_agent,
            recipient=user_id,
            content=response["message"],
            timestamp=datetime.now(),
            agent_response=True,
            flirty_level=response.get("flirty_level", 1)
        )

        self._chat_history[user_id].append(response_msg)

        return {
            "message_sent": True,
            "message_id": msg_id,
            "response": response
        }

    def _get_agent_response(self, user_id: str, message: str,
                           agent: str) -> Dict:
        """Get response from agent"""
        user = self._users.get(user_id)
        flirty = user.preferences.get("flirty_mode", False) if user else False

        # Parse intent (simplified)
        message_lower = message.lower()

        # Determine response based on intent
        if any(word in message_lower for word in ["briefing", "status", "update"]):
            if flirty:
                return {
                    "message": f"Coming right up! Let me pull together the highlights for you...\n\n"
                              f"Portfolio: +0.36% today\n"
                              f"Meetings: 3 on your calendar\n"
                              f"Tasks: 5 priority tasks\n"
                              f"Alerts: 0\n\n"
                              f"Anything specific you want me to dive into? - {agent}",
                    "flirty_level": 2,
                    "action": "briefing"
                }
            else:
                return {
                    "message": "Here's your status update:\n\n"
                              "Portfolio: +0.36% today\n"
                              "Meetings: 3 scheduled\n"
                              "Tasks: 5 priority items\n"
                              "Alerts: None",
                    "flirty_level": 0,
                    "action": "briefing"
                }

        elif any(word in message_lower for word in ["schedule", "meeting", "calendar"]):
            if flirty:
                return {
                    "message": f"Let me check your calendar...\n\n"
                              f"You've got 3 slots open today. Want me to book something? "
                              f"Just say the word and I'll make it happen! - {agent}",
                    "flirty_level": 2,
                    "action": "calendar"
                }
            else:
                return {
                    "message": "I can help with scheduling. What would you like to set up?",
                    "flirty_level": 0,
                    "action": "calendar"
                }

        elif any(word in message_lower for word in ["task", "todo", "priority"]):
            if flirty:
                return {
                    "message": f"On it! Here are your priorities:\n\n"
                              f"1. Review Q4 performance\n"
                              f"2. Sign K-1 documents\n"
                              f"3. Approve trade allocations\n\n"
                              f"Need me to add or shuffle anything? - {agent}",
                    "flirty_level": 1,
                    "action": "tasks"
                }
            else:
                return {
                    "message": "Your priority tasks:\n1. Review Q4 performance\n2. Sign K-1 documents\n3. Approve trade allocations",
                    "flirty_level": 0,
                    "action": "tasks"
                }

        elif any(word in message_lower for word in ["hello", "hi", "hey"]):
            if flirty:
                return {
                    "message": f"Hey there! How can I help you today? - {agent}",
                    "flirty_level": 3,
                    "action": "greeting"
                }
            else:
                return {
                    "message": "Hello! How can I help you today?",
                    "flirty_level": 0,
                    "action": "greeting"
                }

        elif any(word in message_lower for word in ["thanks", "thank you"]):
            if flirty:
                return {
                    "message": f"Anytime! That's what I'm here for. Anything else? - {agent}",
                    "flirty_level": 2,
                    "action": "thanks"
                }
            else:
                return {
                    "message": "You're welcome! Let me know if you need anything else.",
                    "flirty_level": 0,
                    "action": "thanks"
                }

        else:
            if flirty:
                return {
                    "message": f"I heard you! Let me see what I can do about '{message[:50]}...' "
                              f"Give me a moment... - {agent}",
                    "flirty_level": 1,
                    "action": "processing"
                }
            else:
                return {
                    "message": f"Processing your request: '{message[:50]}...'",
                    "flirty_level": 0,
                    "action": "processing"
                }

    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for user"""
        user_id = user_id.upper()

        history = self._chat_history.get(user_id, [])

        return [
            {
                "id": msg.id,
                "sender": msg.sender,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "agent_response": msg.agent_response,
                "flirty_level": msg.flirty_level
            }
            for msg in history[-limit:]
        ]

    # =========================================================================
    # LEARNING
    # =========================================================================

    def _log_interaction(self, user_id: str, action: str,
                        content: str, target: str):
        """Log interaction for learning"""
        self._interaction_log.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_id,
            "action": action,
            "content_preview": content[:100],
            "target": target
        })

        # Cap log size
        if len(self._interaction_log) > 10000:
            self._interaction_log = self._interaction_log[-5000:]

    def get_learning_insights(self, user_id: str = None) -> Dict:
        """Get learning insights"""
        relevant = self._interaction_log

        if user_id:
            relevant = [i for i in relevant if i["user"] == user_id.upper()]

        # Analyze patterns
        by_action = {}
        by_hour = {}

        for interaction in relevant:
            action = interaction["action"]
            by_action[action] = by_action.get(action, 0) + 1

            hour = datetime.fromisoformat(interaction["timestamp"]).hour
            by_hour[hour] = by_hour.get(hour, 0) + 1

        return {
            "total_interactions": len(relevant),
            "by_action": by_action,
            "by_hour": by_hour,
            "peak_hours": sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:3],
            "top_actions": sorted(by_action.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def update_preferences(self, user_id: str, preferences: Dict) -> Dict:
        """Update user preferences"""
        user_id = user_id.upper()

        if user_id not in self._users:
            return {"error": "User not found"}

        self._users[user_id].preferences.update(preferences)

        return {
            "success": True,
            "updated_preferences": self._users[user_id].preferences
        }

    def get_app_config(self) -> Dict:
        """Get app configuration"""
        return {
            "name": "Alpha Loop Hub",
            "version": "1.0.0",
            "design": self.DESIGN,
            "features": {
                "chat": True,
                "dashboard": True,
                "calendar": True,
                "tasks": True,
                "documents": True,
                "learning": True,
                "flirty_mode": True,
            },
            "integrations": ["slack", "notion", "outlook"],
            "agents_available": list(self._agents.keys())
        }


# Singleton
_hub_app_instance: Optional[AlphaLoopHub] = None


def get_hub_app() -> AlphaLoopHub:
    global _hub_app_instance
    if _hub_app_instance is None:
        _hub_app_instance = AlphaLoopHub()
    return _hub_app_instance


if __name__ == "__main__":
    hub = get_hub_app()

    # Test login
    print("\n" + "="*60)
    print("ALPHA LOOP HUB - Test")
    print("="*60)

    result = hub.login("TOM")
    print(f"\nLogin: {result['success']}")
    print(f"Greeting: {result['assistant_greeting']}")

    # Test chat
    chat_result = hub.send_chat("TOM", "Hey KAT, give me my briefing")
    print(f"\nChat Response: {chat_result['response']['message']}")

