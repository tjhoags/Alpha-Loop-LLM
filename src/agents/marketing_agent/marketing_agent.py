"""
================================================================================
MARKETING AGENT - Brand, Content & Communications
================================================================================
Author: Alpha Loop Capital, LLC

MARKETING handles all brand strategy, content creation, and communications.
Works with SOFTWARE to create engaging UI/UX experiences.

Tier: SENIOR (2)
Division: Operations
Reports To: FRIEDS

Core Functions:
- Brand strategy and messaging
- Content creation and curation
- Internal communications
- UI/UX design direction
- User experience optimization
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


class ContentType(Enum):
    """Types of content MARKETING creates"""
    UI_COPY = auto()
    NOTIFICATION = auto()
    EMAIL = auto()
    REPORT = auto()
    DASHBOARD = auto()
    ONBOARDING = auto()
    HELP_TEXT = auto()
    ERROR_MESSAGE = auto()


class UIStyle(Enum):
    """UI aesthetic styles"""
    PROFESSIONAL = auto()  # Clean, corporate
    MODERN = auto()        # Sleek, minimal
    FRIENDLY = auto()      # Warm, approachable
    PREMIUM = auto()       # Luxurious, sophisticated
    TECH = auto()          # Technical, data-driven


@dataclass
class ContentPiece:
    """A piece of content created by MARKETING"""
    id: str
    type: ContentType
    title: str
    content: str
    style: UIStyle
    created: datetime
    approved: bool = False


@dataclass
class UIDesignSpec:
    """UI/UX design specification"""
    id: str
    component: str
    description: str
    style: UIStyle
    copy: Dict[str, str]
    interactions: List[str]
    accessibility: List[str]


class MarketingAgent(BaseAgent):
    """
    MARKETING - Brand, Content & Communications Specialist

    Creates engaging content and UI/UX experiences for Alpha Loop Capital.
    """

    # Brand voice guidelines
    BRAND_VOICE = {
        "tone": "Confident, sophisticated, approachable",
        "values": ["Excellence", "Innovation", "Trust", "Partnership"],
        "tagline": "By end of 2026, they will know Alpha Loop Capital.",
        "personality": "Smart, witty, professional with a touch of charm"
    }

    # Color palette
    COLORS = {
        "primary": "#1a1a2e",      # Deep navy
        "secondary": "#16213e",    # Dark blue
        "accent": "#0f3460",       # Ocean blue
        "highlight": "#e94560",    # Coral red
        "success": "#00d9a5",      # Emerald
        "warning": "#ffc107",      # Gold
        "error": "#dc3545",        # Red
        "text": "#ffffff",         # White
        "text_muted": "#a0a0a0",   # Gray
        "background": "#0f0f23",   # Near black
    }

    def __init__(self):
        super().__init__(
            name="MARKETING",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Brand
                "brand_strategy", "messaging", "voice_guidelines",

                # Content
                "copy_writing", "content_creation", "notification_design",
                "error_messaging", "onboarding_flows", "help_documentation",

                # UI/UX
                "ui_design_direction", "ux_optimization", "user_journey_mapping",
                "accessibility_guidelines", "interaction_design",

                # Communications
                "internal_comms", "team_messaging", "status_updates",

                # Learning
                "user_feedback_analysis", "engagement_metrics", "a_b_testing",
            ],
            user_id="OPERATIONS",
            thinking_modes=[
                ThinkingMode.CREATIVE,
                ThinkingMode.STRUCTURAL,
                ThinkingMode.LATERAL,
            ],
            learning_methods=[
                LearningMethod.REINFORCEMENT,
                LearningMethod.CONTEXTUAL,
                LearningMethod.TRANSFER,
            ],
        )

        self._content_library: Dict[str, ContentPiece] = {}
        self._ui_specs: Dict[str, UIDesignSpec] = {}
        self._feedback_log: List[Dict] = []

        logger.info("MARKETING initialized - Brand & Content specialist ready")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process marketing/content request"""
        action = inputs.get("action", "status")

        handlers = {
            "status": self._handle_status,
            "create_content": self._handle_create_content,
            "ui_spec": self._handle_ui_spec,
            "notification": self._handle_notification,
            "design_system": self._handle_design_system,
            "onboarding": self._handle_onboarding,
            "error_message": self._handle_error_message,
            "feedback": self._handle_feedback,
        }

        handler = handlers.get(action, self._handle_unknown)
        return handler(inputs)

    def _handle_status(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "agent": "MARKETING",
            "role": "Brand & Content Specialist",
            "content_library_size": len(self._content_library),
            "ui_specs": len(self._ui_specs),
            "brand_voice": self.BRAND_VOICE,
            "marketing_says": "Ready to create compelling experiences! "
                             "Brand on point, content flowing. What do you need?"
        }

    def _handle_create_content(self, params: Dict) -> Dict:
        """Create content piece"""
        content_type = ContentType[params.get("type", "UI_COPY").upper()]
        title = params.get("title", "")
        context = params.get("context", "")
        style = UIStyle[params.get("style", "PROFESSIONAL").upper()]

        # Generate content based on type
        content = self._generate_content(content_type, title, context, style)

        content_id = hashlib.sha256(f"{title}{datetime.now()}".encode()).hexdigest()[:8]

        piece = ContentPiece(
            id=content_id,
            type=content_type,
            title=title,
            content=content,
            style=style,
            created=datetime.now()
        )

        self._content_library[content_id] = piece

        return {
            "status": "success",
            "content_id": content_id,
            "type": content_type.name,
            "content": content,
            "style": style.name,
            "marketing_says": "Content created! Polished and on-brand. "
                             "Let me know if you want tweaks."
        }

    def _generate_content(self, ctype: ContentType, title: str,
                         context: str, style: UIStyle) -> str:
        """Generate content based on type and style"""

        if ctype == ContentType.UI_COPY:
            return self._generate_ui_copy(title, context, style)
        elif ctype == ContentType.NOTIFICATION:
            return self._generate_notification(title, context, style)
        elif ctype == ContentType.ERROR_MESSAGE:
            return self._generate_error_message(title, context)
        elif ctype == ContentType.ONBOARDING:
            return self._generate_onboarding(title, context)
        else:
            return f"[{ctype.name}] {title}: {context}"

    def _generate_ui_copy(self, title: str, context: str, style: UIStyle) -> str:
        """Generate UI copy"""
        style_modifiers = {
            UIStyle.PROFESSIONAL: ("Welcome to", "View", "Manage"),
            UIStyle.MODERN: ("Hey!", "Check out", "Handle"),
            UIStyle.FRIENDLY: ("Hi there!", "See", "Take care of"),
            UIStyle.PREMIUM: ("Welcome", "Review", "Oversee"),
            UIStyle.TECH: ("Dashboard:", "Analytics:", "Execute"),
        }

        prefix = style_modifiers.get(style, style_modifiers[UIStyle.PROFESSIONAL])[0]
        return f"{prefix} {title}\n{context}"

    def _generate_notification(self, title: str, context: str, style: UIStyle) -> str:
        """Generate notification copy"""
        return f"📢 {title}\n{context}"

    def _generate_error_message(self, title: str, context: str) -> str:
        """Generate user-friendly error message"""
        return f"Oops! {title}\n{context}\n\nNeed help? We're here for you."

    def _generate_onboarding(self, title: str, context: str) -> str:
        """Generate onboarding copy"""
        return f"{title}\n\nLet's get you set up!\n\n{context}"

    def _handle_ui_spec(self, params: Dict) -> Dict:
        """Create UI/UX specification"""
        component = params.get("component", "")
        description = params.get("description", "")
        style = UIStyle[params.get("style", "PROFESSIONAL").upper()]

        spec_id = hashlib.sha256(f"{component}{datetime.now()}".encode()).hexdigest()[:8]

        # Generate comprehensive UI spec
        spec = UIDesignSpec(
            id=spec_id,
            component=component,
            description=description,
            style=style,
            copy={
                "title": f"{component} - Alpha Loop Capital",
                "subtitle": description,
                "cta_primary": "Continue",
                "cta_secondary": "Learn More",
            },
            interactions=[
                "Hover: Subtle elevation + highlight",
                "Click: Ripple effect",
                "Focus: Outline ring",
                "Loading: Skeleton + shimmer",
            ],
            accessibility=[
                "ARIA labels on all interactive elements",
                "Keyboard navigation support",
                "Color contrast ratio >= 4.5:1",
                "Focus indicators visible",
            ]
        )

        self._ui_specs[spec_id] = spec

        return {
            "status": "success",
            "spec_id": spec_id,
            "component": component,
            "style": style.name,
            "colors": self.COLORS,
            "copy": spec.copy,
            "interactions": spec.interactions,
            "accessibility": spec.accessibility,
            "marketing_says": "UI spec ready! Clean, accessible, and on-brand. "
                             "SOFTWARE can run with this."
        }

    def _handle_notification(self, params: Dict) -> Dict:
        """Design notification"""
        event_type = params.get("event_type", "info")  # info, success, warning, error
        message = params.get("message", "")
        action = params.get("action", "")

        icons = {
            "info": "ℹ️",
            "success": "[OK]",
            "warning": "[WARN]",
            "error": "[ERROR]",
        }

        return {
            "status": "success",
            "notification": {
                "icon": icons.get(event_type, "📢"),
                "type": event_type,
                "message": message,
                "action": action,
                "style": {
                    "background": self.COLORS.get(event_type, self.COLORS["primary"]),
                    "border": f"1px solid {self.COLORS['highlight']}",
                    "animation": "slideIn 0.3s ease-out",
                }
            },
            "marketing_says": "Notification designed! Eye-catching but not intrusive."
        }

    def _handle_design_system(self, params: Dict) -> Dict:
        """Get design system specs"""
        return {
            "status": "success",
            "design_system": {
                "colors": self.COLORS,
                "typography": {
                    "font_primary": "'Inter', -apple-system, sans-serif",
                    "font_mono": "'JetBrains Mono', monospace",
                    "scale": {
                        "xs": "0.75rem",
                        "sm": "0.875rem",
                        "base": "1rem",
                        "lg": "1.125rem",
                        "xl": "1.25rem",
                        "2xl": "1.5rem",
                        "3xl": "1.875rem",
                        "4xl": "2.25rem",
                    }
                },
                "spacing": {
                    "xs": "0.25rem",
                    "sm": "0.5rem",
                    "md": "1rem",
                    "lg": "1.5rem",
                    "xl": "2rem",
                    "2xl": "3rem",
                },
                "borders": {
                    "radius_sm": "4px",
                    "radius_md": "8px",
                    "radius_lg": "12px",
                    "radius_full": "9999px",
                },
                "shadows": {
                    "sm": "0 1px 2px rgba(0, 0, 0, 0.3)",
                    "md": "0 4px 6px rgba(0, 0, 0, 0.4)",
                    "lg": "0 10px 15px rgba(0, 0, 0, 0.5)",
                    "glow": f"0 0 20px {self.COLORS['highlight']}40",
                },
                "animations": {
                    "duration_fast": "150ms",
                    "duration_normal": "300ms",
                    "duration_slow": "500ms",
                    "easing": "cubic-bezier(0.4, 0, 0.2, 1)",
                }
            },
            "brand": self.BRAND_VOICE,
            "marketing_says": "Full design system ready! Consistent, beautiful, Alpha Loop."
        }

    def _handle_onboarding(self, params: Dict) -> Dict:
        """Design onboarding flow"""
        user_type = params.get("user_type", "standard")  # tom, chris, agent

        flows = {
            "tom": [
                {"step": 1, "title": "Welcome, Tom",
                 "content": "Your command center is ready. Let's review the highlights."},
                {"step": 2, "title": "Quick Overview",
                 "content": "KAT and your agents are standing by. Here's what's new."},
                {"step": 3, "title": "Ready to Go",
                 "content": "All systems operational. Let's crush it."},
            ],
            "chris": [
                {"step": 1, "title": "Welcome, Chris",
                 "content": "Operations HQ at your fingertips. Let's get started."},
                {"step": 2, "title": "Your Team",
                 "content": "SHYLA and the crew are ready. Here's your dashboard."},
                {"step": 3, "title": "You're Set",
                 "content": "Everything's in order. Time to make things happen."},
            ],
        }

        return {
            "status": "success",
            "user_type": user_type,
            "flow": flows.get(user_type, flows["tom"]),
            "marketing_says": "Onboarding flow designed! Smooth, engaging, effective."
        }

    def _handle_error_message(self, params: Dict) -> Dict:
        """Design user-friendly error message"""
        error_code = params.get("error_code", "UNKNOWN")
        technical_message = params.get("technical", "")

        friendly_messages = {
            "AUTH_FAILED": ("Access Denied", "Hmm, that didn't work. Check your credentials and try again."),
            "NOT_FOUND": ("Not Found", "We couldn't find what you're looking for. Try searching again."),
            "PERMISSION": ("Permission Needed", "You'll need approval for this. Reach out to KAT or SHYLA."),
            "NETWORK": ("Connection Issue", "Having trouble connecting. Check your network and retry."),
            "TIMEOUT": ("Taking Too Long", "This is taking longer than expected. We're on it."),
            "UNKNOWN": ("Something Went Wrong", "We hit a snag. Try again or contact support."),
        }

        title, message = friendly_messages.get(error_code, friendly_messages["UNKNOWN"])

        return {
            "status": "success",
            "error_display": {
                "title": title,
                "message": message,
                "action": "Try Again",
                "help_link": "Get Help",
                "icon": "[!]",
                "style": {
                    "background": f"{self.COLORS['error']}20",
                    "border": f"1px solid {self.COLORS['error']}",
                }
            },
            "marketing_says": "Error message humanized! Friendly, helpful, not scary."
        }

    def _handle_feedback(self, params: Dict) -> Dict:
        """Log and analyze user feedback"""
        feedback_type = params.get("type", "general")
        content = params.get("content", "")
        user = params.get("user", "")
        rating = params.get("rating", 0)

        self._feedback_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "content": content,
            "user": user,
            "rating": rating
        })

        return {
            "status": "success",
            "feedback_logged": True,
            "total_feedback": len(self._feedback_log),
            "marketing_says": "Feedback captured! This helps us improve. Thank you!"
        }

    def _handle_unknown(self, params: Dict) -> Dict:
        return {
            "status": "clarification_needed",
            "marketing_says": "What do you need? Content? UI specs? Notifications? "
                             "I can make anything look and sound amazing!"
        }


# Singleton
_marketing_instance: Optional[MarketingAgent] = None


def get_marketing() -> MarketingAgent:
    global _marketing_instance
    if _marketing_instance is None:
        _marketing_instance = MarketingAgent()
    return _marketing_instance

