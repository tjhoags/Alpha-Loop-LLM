"""
Slack Notification Client
=========================
Send alerts and reports to Slack channels.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Slack notification client for trading alerts.
    
    Setup:
    1. Create Slack App at https://api.slack.com/apps
    2. Add Bot Token Scopes: chat:write, channels:read
    3. Install to workspace
    4. Copy Bot User OAuth Token to SLACK_BOT_TOKEN env var
    """
    
    def __init__(
        self,
        token: str = None,
        default_channel: str = None
    ):
        self.token = token or os.getenv("SLACK_BOT_TOKEN")
        self.default_channel = default_channel or os.getenv("SLACK_CHANNEL_ALERTS", "#trading-alerts")
        self.available = False
        self.client = None
        
        if self.token:
            try:
                from slack_sdk import WebClient
                self.client = WebClient(token=self.token)
                self.available = True
                logger.info("Slack client initialized")
            except ImportError:
                logger.warning("slack_sdk not installed: pip install slack_sdk")
            except Exception as e:
                logger.warning(f"Slack initialization failed: {e}")
    
    def send_message(
        self,
        message: str,
        channel: str = None
    ) -> bool:
        """Send a simple text message"""
        if not self.available:
            logger.warning("Slack not available")
            return False
        
        try:
            response = self.client.chat_postMessage(
                channel=channel or self.default_channel,
                text=message
            )
            return response["ok"]
        except Exception as e:
            logger.error(f"Slack message failed: {e}")
            return False
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        channel: str = None
    ) -> bool:
        """
        Send a formatted alert.
        
        Severities: info, warning, critical, success
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
            "success": "✅"
        }
        
        color_map = {
            "info": "#3498db",
            "warning": "#f39c12",
            "critical": "#e74c3c",
            "success": "#2ecc71"
        }
        
        emoji = emoji_map.get(severity, "📢")
        color = color_map.get(severity, "#808080")
        
        if not self.available:
            # Print locally if Slack unavailable
            print(f"\n{emoji} [{severity.upper()}] {title}")
            print(f"   {message}\n")
            return False
        
        try:
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{title}*"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"ALC-Algo | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        }
                    ]
                }
            ]
            
            response = self.client.chat_postMessage(
                channel=channel or self.default_channel,
                text=f"{emoji} {title}: {message}",
                blocks=blocks,
                attachments=[{"color": color, "blocks": []}]
            )
            return response["ok"]
            
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return False
    
    def send_trade_alert(
        self,
        ticker: str,
        action: str,
        reason: str,
        price: float = None,
        target: float = None,
        stop: float = None
    ) -> bool:
        """Send a trade alert"""
        action_emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "WATCH": "👀"
        }
        
        emoji = action_emoji.get(action.upper(), "📊")
        
        message_parts = [f"*Reason:* {reason}"]
        
        if price:
            message_parts.append(f"*Price:* ${price:.2f}")
        if target:
            message_parts.append(f"*Target:* ${target:.2f}")
        if stop:
            message_parts.append(f"*Stop:* ${stop:.2f}")
        
        message = "\n".join(message_parts)
        
        return self.send_alert(
            title=f"{emoji} {action.upper()} {ticker}",
            message=message,
            severity="info" if action.upper() in ["BUY", "WATCH"] else "warning"
        )
    
    def send_daily_report(
        self,
        portfolio_value: float,
        daily_pnl: float,
        top_actions: List[Dict]
    ) -> bool:
        """Send daily portfolio report"""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        actions_text = "\n".join([
            f"• [{a.get('category', 'N/A')}] {a.get('ticker', 'N/A')}: {a.get('action', 'N/A')}"
            for a in top_actions[:5]
        ])
        
        message = f"""
*Portfolio Value:* ${portfolio_value:,.0f}
*Daily P&L:* {pnl_emoji} ${daily_pnl:+,.0f}

*Today's Actions:*
{actions_text}
"""
        
        return self.send_alert(
            title="📊 Daily Report",
            message=message,
            severity="success" if daily_pnl >= 0 else "warning"
        )


if __name__ == "__main__":
    # Demo
    notifier = SlackNotifier()
    
    # Test alert (will print locally if Slack not configured)
    notifier.send_alert(
        title="Test Alert",
        message="This is a test alert from ALC-Algo",
        severity="info"
    )
    
    notifier.send_trade_alert(
        ticker="NVDA",
        action="BUY",
        reason="Order 2 beneficiary - AI power demand",
        price=140.50,
        target=180.00,
        stop=125.00
    )

