"""
================================================================================
SLACK CLIENT - Slack Integration
================================================================================
Unified Slack client for messaging, notifications, and workflows.

Features:
- Direct messaging to users
- Channel posting
- Rich notifications with blocks
- Workflow automation
- File sharing
================================================================================
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SlackChannelType(Enum):
    DIRECT = "direct"
    CHANNEL = "channel"
    GROUP = "group"


@dataclass
class SlackMessage:
    """Slack message structure"""
    channel: str
    text: str
    blocks: Optional[List[Dict]] = None
    thread_ts: Optional[str] = None
    reply_broadcast: bool = False


class SlackClient:
    """
    Slack integration client

    Provides unified interface for all Slack operations.
    """

    # Default channels
    CHANNELS = {
        "general": "#alc-general",
        "alerts": "#alc-alerts",
        "trading": "#alc-trading",
        "ops": "#alc-ops",
    }

    # User mappings
    USERS = {
        "TOM": "U_TOM_HOGAN",
        "CHRIS": "U_CHRIS_FRIEDMAN",
    }

    def __init__(self, api_token: str = None, bot_token: str = None):
        self.api_token = api_token
        self.bot_token = bot_token
        self.connected = False
        self._message_history: List[Dict] = []

        logger.info("SlackClient initialized")

    async def connect(self) -> bool:
        """Connect to Slack"""
        # In production, would initialize Slack SDK
        self.connected = True
        logger.info("Slack connected")
        return True

    async def send_message(self, channel: str, text: str,
                          blocks: List[Dict] = None) -> Dict:
        """Send message to channel or user"""
        message_id = f"msg_{datetime.now().timestamp()}"

        result = {
            "ok": True,
            "channel": channel,
            "ts": str(datetime.now().timestamp()),
            "message": {
                "text": text,
                "blocks": blocks
            }
        }

        self._message_history.append({
            "id": message_id,
            "channel": channel,
            "text": text,
            "sent_at": datetime.now().isoformat()
        })

        logger.info(f"Slack message sent to {channel}")
        return result

    async def send_dm(self, user_id: str, text: str,
                     blocks: List[Dict] = None) -> Dict:
        """Send direct message to user"""
        return await self.send_message(f"@{user_id}", text, blocks)

    async def send_notification(self, user: str, title: str, body: str,
                               priority: str = "normal",
                               actions: List[Dict] = None) -> Dict:
        """Send rich notification"""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title, "emoji": True}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": body}
            }
        ]

        if priority == "urgent":
            blocks.insert(0, {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*URGENT*"}
            })

        if actions:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": a["text"]},
                        "action_id": a["action_id"],
                        "style": a.get("style", "primary")
                    }
                    for a in actions
                ]
            })

        user_id = self.USERS.get(user.upper(), user)
        return await self.send_dm(user_id, title, blocks)

    async def post_to_channel(self, channel_name: str, text: str,
                             attachments: List[Dict] = None) -> Dict:
        """Post to a channel"""
        channel = self.CHANNELS.get(channel_name, f"#{channel_name}")
        return await self.send_message(channel, text)

    def create_message_blocks(self, sections: List[Dict]) -> List[Dict]:
        """Create Slack Block Kit message"""
        blocks = []

        for section in sections:
            block_type = section.get("type", "section")

            if block_type == "header":
                blocks.append({
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": section.get("text", ""),
                        "emoji": True
                    }
                })
            elif block_type == "section":
                block = {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": section.get("text", "")
                    }
                }
                if "accessory" in section:
                    block["accessory"] = section["accessory"]
                blocks.append(block)
            elif block_type == "divider":
                blocks.append({"type": "divider"})
            elif block_type == "context":
                blocks.append({
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": section.get("text", "")}
                    ]
                })

        return blocks

    def get_message_history(self, limit: int = 50) -> List[Dict]:
        """Get message history"""
        return self._message_history[-limit:]


# Singleton
_slack_instance: Optional[SlackClient] = None


def get_slack_client() -> SlackClient:
    global _slack_instance
    if _slack_instance is None:
        _slack_instance = SlackClient()
    return _slack_instance

