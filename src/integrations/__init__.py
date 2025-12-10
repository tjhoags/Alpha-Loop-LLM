"""
================================================================================
INTEGRATIONS - External Service Connectors
================================================================================
Author: Alpha Loop Capital, LLC

Unified integration layer for connecting to external services:
- Slack (messaging, notifications, workflows)
- Notion (documentation, databases)
- Microsoft 365/Outlook (email, calendar, teams)

All integrations follow a consistent interface for easy use by agents.
================================================================================
"""

from src.integrations.slack_client import SlackClient, get_slack_client
from src.integrations.notion_client import NotionClient, get_notion_client
from src.integrations.outlook_client import OutlookClient, get_outlook_client
from src.integrations.hub import IntegrationHub, get_integration_hub

__all__ = [
    "SlackClient",
    "get_slack_client",
    "NotionClient",
    "get_notion_client",
    "OutlookClient",
    "get_outlook_client",
    "IntegrationHub",
    "get_integration_hub",
]

