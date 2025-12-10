"""
================================================================================
AGENT CHAT INTERFACE - Chris & Tom Communication with Agents
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Web-based chat interface for Chris Friedman and Tom Hogan to communicate
directly with SANTAS_HELPER, CPA, and other agents. Built for Azure deployment
in a closed environment.

Features:
- Real-time chat with fund operations agents
- Report generation and viewing
- Conversation history for training data
- Anonymized data export for agent training

Usage:
    python -m src.ui.agent_chat --port 8080

Or with Flask dev server:
    cd src/ui && flask run --port 8080

================================================================================
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from flask import Flask, render_template_string, request, jsonify, session
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not installed. Run: pip install flask")

from src.agents.fund_operations.santas_helper_agent import get_santas_helper
from src.agents.fund_operations.cpa_agent import get_cpa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONVERSATION STORAGE (For Training Data)
# =============================================================================

class ConversationStore:
    """Store conversations for training data (anonymized)"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or "data/conversations"
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.conversations: Dict[str, List[Dict]] = {}

    def new_conversation(self) -> str:
        """Start a new conversation"""
        conv_id = str(uuid.uuid4())[:8]
        self.conversations[conv_id] = []
        return conv_id

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        agent: str = None,
        metadata: Dict = None
    ):
        """Add a message to conversation"""
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []

        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user" or "agent"
            "content": content,
            "agent": agent,
            "metadata": metadata or {},
        }
        self.conversations[conv_id].append(message)

    def get_conversation(self, conv_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversations.get(conv_id, [])

    def export_for_training(self, anonymize: bool = True) -> List[Dict]:
        """Export conversations for training (anonymized)"""
        exported = []

        for conv_id, messages in self.conversations.items():
            conv_data = {
                "conversation_id": conv_id,
                "messages": [],
                "exported_at": datetime.now().isoformat(),
            }

            for msg in messages:
                export_msg = msg.copy()

                if anonymize:
                    # Remove any identifying information
                    export_msg["content"] = self._anonymize_content(msg["content"])
                    export_msg.pop("metadata", None)

                conv_data["messages"].append(export_msg)

            exported.append(conv_data)

        return exported

    def _anonymize_content(self, content: str) -> str:
        """Anonymize content for training"""
        # Replace specific amounts with placeholders
        import re
        content = re.sub(r'\$[\d,]+', '[AMOUNT]', content)
        content = re.sub(r'LP\d+', '[INVESTOR_ID]', content)
        return content

    def save_to_file(self, filename: str = None):
        """Save conversations to file"""
        filename = filename or f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path(self.storage_path) / filename

        with open(filepath, "w") as f:
            json.dump(self.export_for_training(), f, indent=2)

        logger.info(f"Conversations saved to {filepath}")
        return str(filepath)


# =============================================================================
# CHAT ENGINE
# =============================================================================

class AgentChatEngine:
    """Engine for chat interactions with agents"""

    def __init__(self):
        self.santas_helper = get_santas_helper()
        self.cpa = get_cpa()
        self.conversation_store = ConversationStore()

        self.agents = {
            "SANTAS_HELPER": self.santas_helper,
            "CPA": self.cpa,
        }

        # Intent patterns for natural language understanding
        self.intent_patterns = {
            "nav": ["nav", "net asset value", "valuation", "fund value"],
            "fees": ["fee", "management fee", "performance", "carry", "incentive"],
            "tax": ["tax", "k-1", "k1", "1065", "estimated tax"],
            "audit": ["audit", "auditor", "pbc", "finding"],
            "report": ["report", "lp report", "quarterly", "statement"],
            "pnl": ["p&l", "pnl", "profit", "loss", "income"],
            "status": ["status", "how are you", "update"],
            "filing": ["filing", "form pf", "13f", "adv", "regulatory"],
            "team": ["team", "junior", "staff"],
        }

    def process_message(
        self,
        message: str,
        user: str = "Chris",
        conv_id: str = None,
        target_agent: str = None
    ) -> Dict[str, Any]:
        """Process a user message and return agent response"""

        conv_id = conv_id or self.conversation_store.new_conversation()

        # Log user message
        self.conversation_store.add_message(
            conv_id=conv_id,
            role="user",
            content=message,
            metadata={"user": user}
        )

        # Determine target agent if not specified
        if not target_agent:
            target_agent = self._determine_agent(message)

        agent = self.agents.get(target_agent, self.santas_helper)

        # Determine intent and build task
        intent = self._determine_intent(message)
        task = self._build_task(intent, message)

        # Get agent response
        try:
            result = agent.process(task)

            # Format response
            response = self._format_response(result, target_agent, intent)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response = f"I apologize, but I encountered an error processing your request. Error: {str(e)}"

        # Log agent response
        self.conversation_store.add_message(
            conv_id=conv_id,
            role="agent",
            content=response,
            agent=target_agent,
            metadata={"intent": intent}
        )

        return {
            "response": response,
            "agent": target_agent,
            "intent": intent,
            "conversation_id": conv_id,
        }

    def _determine_agent(self, message: str) -> str:
        """Determine which agent should handle the message"""
        message_lower = message.lower()

        # CPA handles tax, audit, regulatory
        cpa_keywords = ["tax", "k-1", "k1", "audit", "filing", "13f", "form pf", "adv", "irs"]
        if any(kw in message_lower for kw in cpa_keywords):
            return "CPA"

        # SANTAS_HELPER handles NAV, fees, reporting, P&L
        return "SANTAS_HELPER"

    def _determine_intent(self, message: str) -> str:
        """Determine intent from message"""
        message_lower = message.lower()

        for intent, patterns in self.intent_patterns.items():
            if any(p in message_lower for p in patterns):
                return intent

        return "general"

    def _build_task(self, intent: str, message: str) -> Dict[str, Any]:
        """Build task dictionary from intent"""
        task_mapping = {
            "nav": {"action": "generate_nav_pack"},
            "fees": {"action": "calculate_fees", "period": "Q4-2024"},
            "tax": {"action": "calculate_fund_taxes", "tax_year": 2024},
            "audit": {"action": "prepare_audit", "audit_firm": "Auditors LLP", "period": "FY2024"},
            "report": {"action": "prepare_lp_report", "report_type": "lp_quarterly", "period": "Q4-2024"},
            "pnl": {"action": "calculate_pnl"},
            "status": {"action": "brief_chris", "type": "daily"},
            "filing": {"action": "get_filing_calendar"},
            "team": {"action": "get_team_status"},
        }

        task = task_mapping.get(intent, {"action": "brief_chris"})
        task["original_message"] = message

        return task

    def _format_response(self, result: Dict, agent: str, intent: str) -> str:
        """Format agent result into natural language response"""

        if result.get("status") == "error":
            return f"I apologize, but I couldn't complete that request. {result.get('message', '')}"

        # Format based on intent
        if intent == "status" and "brief" in result:
            brief = result["brief"]
            return (
                f"{brief.get('greeting', 'Hello Chris,')}\n\n"
                f"Here's your update:\n"
                f"• {brief.get('summary', {}).get('nav_status', 'NAV current')}\n"
                f"• Upcoming deadlines: {', '.join(str(d.get('item')) for d in brief.get('summary', {}).get('upcoming_deadlines', []))}\n\n"
                f"Action items:\n"
                f"{''.join('• ' + item + chr(10) for item in brief.get('action_items', []))}\n"
                f"{brief.get('closing', '')}"
            )

        if intent == "nav" and "nav_pack" in result:
            nav = result["nav_pack"]
            return (
                f"Here's the NAV package for {nav.get('fund_name')} as of {nav.get('as_of_date', 'today')}:\n\n"
                f"**Net Asset Value:** ${nav.get('net_nav', 0):,.2f}\n"
                f"**NAV per Share:** ${nav.get('nav_per_share', 0):.4f}\n\n"
                f"**Returns:**\n"
                f"• MTD: {nav.get('returns', {}).get('mtd', 0):.2f}%\n"
                f"• QTD: {nav.get('returns', {}).get('qtd', 0):.2f}%\n"
                f"• YTD: {nav.get('returns', {}).get('ytd', 0):.2f}%\n\n"
                f"Would you like me to prepare the full NAV pack for distribution?"
            )

        if intent == "fees":
            return (
                f"I've calculated the fees for the requested period:\n\n"
                f"**Management Fee:** ${result.get('management_fee', {}).get('calculated_amount', 0):,.2f}\n"
                f"**Performance Fee:** ${result.get('performance_fee', {}).get('calculated_amount', 0):,.2f}\n"
                f"**Total Fees:** ${result.get('total_fees', 0):,.2f}\n\n"
                f"These calculations consider the high water mark and hurdle rate. "
                f"Let me know if you need a detailed breakdown."
            )

        if intent == "tax" and "calculation" in result:
            calc = result["calculation"]
            return (
                f"Here's the tax calculation summary for TY{calc.get('tax_year')}:\n\n"
                f"**Income Components:**\n"
                f"• Ordinary Income: ${calc.get('income', {}).get('ordinary', 0):,.0f}\n"
                f"• Short-term Gains: ${calc.get('income', {}).get('st_gains', 0):,.0f}\n"
                f"• Long-term Gains: ${calc.get('income', {}).get('lt_gains', 0):,.0f}\n\n"
                f"This is a partnership, so taxes flow through to investors. "
                f"I can prepare K-1 data when you're ready."
            )

        if intent == "filing":
            calendar = result.get("calendar", [])
            lines = ["Here's the regulatory filing calendar:\n"]
            for filing in calendar[:5]:
                lines.append(f"• {filing.get('filing')}: {filing.get('deadline')} ({filing.get('status')})")
            return "\n".join(lines)

        # Generic response
        return (
            f"I've processed your request. Here's a summary:\n\n"
            f"{json.dumps(result, indent=2, default=str)[:500]}...\n\n"
            f"Would you like more details on any specific aspect?"
        )

    def get_daily_brief(self, user: str = "Chris") -> str:
        """Generate combined daily brief from both agents"""

        santa_brief = self.santas_helper.brief_chris()
        cpa_brief = self.cpa.brief_chris()

        return f"""
Good morning {user},

Here's your daily operations brief from your fund accounting team:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FROM SANTAS_HELPER (Fund Operations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• NAV Status: {santa_brief.get('summary', {}).get('nav_status', 'Current')}
• MTD Return: {santa_brief.get('key_metrics', {}).get('mtd_return', 'N/A')}
• YTD Return: {santa_brief.get('key_metrics', {}).get('ytd_return', 'N/A')}

Action Items:
{''.join('  • ' + item + chr(10) for item in santa_brief.get('action_items', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FROM CPA (Tax & Audit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Estimated Taxes: {cpa_brief.get('tax_summary', {}).get('estimated_taxes_current', 'Current')}
• Next Payment Due: {cpa_brief.get('tax_summary', {}).get('next_payment_due', 'N/A')}
• Audit Status: {cpa_brief.get('audit_status', {}).get('status', 'N/A')}

Action Items:
{''.join('  • ' + item + chr(10) for item in cpa_brief.get('action_items', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both agents are coordinated and ready to assist with any requests.

"""


# =============================================================================
# FLASK WEB APPLICATION
# =============================================================================

if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET", "alc-agent-chat-secret-2024")

    chat_engine = AgentChatEngine()

    # HTML Template
    CHAT_HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Alpha Loop Capital - Agent Chat</title>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #0a0a0f;
                --bg-secondary: #12121a;
                --bg-tertiary: #1a1a24;
                --accent-primary: #6366f1;
                --accent-secondary: #8b5cf6;
                --accent-success: #10b981;
                --accent-warning: #f59e0b;
                --text-primary: #f0f0f5;
                --text-secondary: #a0a0b0;
                --text-muted: #606070;
                --border-color: #2a2a3a;
                --santa-color: #10b981;
                --cpa-color: #6366f1;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            .header {
                background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
                border-bottom: 1px solid var(--border-color);
                padding: 1.5rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .header h1 {
                font-size: 1.5rem;
                font-weight: 600;
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .header .subtitle {
                font-size: 0.85rem;
                color: var(--text-secondary);
                margin-top: 0.25rem;
            }

            .user-badge {
                background: var(--bg-tertiary);
                padding: 0.5rem 1rem;
                border-radius: 8px;
                font-size: 0.9rem;
                color: var(--text-secondary);
            }

            .main-container {
                display: flex;
                flex: 1;
                overflow: hidden;
            }

            .sidebar {
                width: 280px;
                background: var(--bg-secondary);
                border-right: 1px solid var(--border-color);
                padding: 1.5rem;
                display: flex;
                flex-direction: column;
            }

            .agent-selector {
                margin-bottom: 1.5rem;
            }

            .agent-selector h3 {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
                margin-bottom: 0.75rem;
            }

            .agent-btn {
                width: 100%;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                color: var(--text-secondary);
                cursor: pointer;
                text-align: left;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .agent-btn:hover {
                background: var(--bg-primary);
                border-color: var(--accent-primary);
                color: var(--text-primary);
            }

            .agent-btn.active {
                border-color: var(--accent-primary);
                background: rgba(99, 102, 241, 0.1);
            }

            .agent-btn.santa { border-left: 3px solid var(--santa-color); }
            .agent-btn.cpa { border-left: 3px solid var(--cpa-color); }

            .agent-indicator {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--accent-success);
            }

            .quick-actions {
                margin-top: auto;
            }

            .quick-actions h3 {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
                margin-bottom: 0.75rem;
            }

            .quick-btn {
                width: 100%;
                padding: 0.5rem 0.75rem;
                margin-bottom: 0.5rem;
                background: transparent;
                border: 1px solid var(--border-color);
                border-radius: 6px;
                color: var(--text-secondary);
                cursor: pointer;
                font-size: 0.85rem;
                text-align: left;
                transition: all 0.2s;
            }

            .quick-btn:hover {
                border-color: var(--accent-primary);
                color: var(--text-primary);
            }

            .chat-area {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: var(--bg-primary);
            }

            .messages {
                flex: 1;
                padding: 2rem;
                overflow-y: auto;
            }

            .message {
                max-width: 80%;
                margin-bottom: 1.5rem;
                animation: fadeIn 0.3s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .message.user {
                margin-left: auto;
            }

            .message-header {
                font-size: 0.75rem;
                color: var(--text-muted);
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .message-content {
                padding: 1rem 1.25rem;
                border-radius: 12px;
                line-height: 1.6;
            }

            .message.user .message-content {
                background: var(--accent-primary);
                color: white;
                border-bottom-right-radius: 4px;
            }

            .message.agent .message-content {
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-bottom-left-radius: 4px;
            }

            .message.agent.santa .message-header { color: var(--santa-color); }
            .message.agent.cpa .message-header { color: var(--cpa-color); }

            .input-area {
                padding: 1.5rem 2rem;
                background: var(--bg-secondary);
                border-top: 1px solid var(--border-color);
            }

            .input-container {
                display: flex;
                gap: 1rem;
                align-items: flex-end;
            }

            .input-wrapper {
                flex: 1;
                position: relative;
            }

            #message-input {
                width: 100%;
                padding: 1rem 1.25rem;
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                color: var(--text-primary);
                font-family: inherit;
                font-size: 0.95rem;
                resize: none;
                transition: border-color 0.2s;
            }

            #message-input:focus {
                outline: none;
                border-color: var(--accent-primary);
            }

            #message-input::placeholder {
                color: var(--text-muted);
            }

            .send-btn {
                padding: 1rem 1.5rem;
                background: var(--accent-primary);
                border: none;
                border-radius: 12px;
                color: white;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.2s;
            }

            .send-btn:hover {
                background: var(--accent-secondary);
                transform: translateY(-1px);
            }

            .typing-indicator {
                display: none;
                padding: 0.5rem 0;
                color: var(--text-muted);
                font-size: 0.85rem;
            }

            .typing-indicator.active {
                display: block;
            }

            pre {
                background: var(--bg-tertiary);
                padding: 1rem;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                margin: 0.5rem 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>Alpha Loop Capital</h1>
                <div class="subtitle">Fund Operations Agent Interface</div>
            </div>
            <div class="user-badge" id="user-badge">Chris Friedman</div>
        </div>

        <div class="main-container">
            <div class="sidebar">
                <div class="agent-selector">
                    <h3>Active Agents</h3>
                    <button class="agent-btn santa active" onclick="selectAgent('SANTAS_HELPER')">
                        <span class="agent-indicator"></span>
                        SANTAS_HELPER
                    </button>
                    <button class="agent-btn cpa" onclick="selectAgent('CPA')">
                        <span class="agent-indicator"></span>
                        CPA
                    </button>
                </div>

                <div class="quick-actions">
                    <h3>Quick Actions</h3>
                    <button class="quick-btn" onclick="sendQuickMessage('Give me today\'s NAV')">Today's NAV</button>
                    <button class="quick-btn" onclick="sendQuickMessage('What are the Q4 fees?')">Q4 Fees</button>
                    <button class="quick-btn" onclick="sendQuickMessage('Tax status update')">Tax Status</button>
                    <button class="quick-btn" onclick="sendQuickMessage('Filing calendar')">Filing Calendar</button>
                    <button class="quick-btn" onclick="sendQuickMessage('Daily brief')">Daily Brief</button>
                </div>
            </div>

            <div class="chat-area">
                <div class="messages" id="messages">
                    <div class="message agent santa">
                        <div class="message-header">
                            SANTAS_HELPER
                        </div>
                        <div class="message-content">
                            Good morning Chris! I'm SANTAS_HELPER, your fund operations lead.
                            I'm here to help with NAV calculations, fee computations, LP reporting,
                            and all fund accounting matters. How can I assist you today?
                        </div>
                    </div>
                </div>

                <div class="input-area">
                    <div class="typing-indicator" id="typing">Agent is typing...</div>
                    <div class="input-container">
                        <div class="input-wrapper">
                            <textarea
                                id="message-input"
                                placeholder="Ask about NAV, fees, taxes, audits, or reports..."
                                rows="1"
                                onkeydown="handleKeyDown(event)"
                            ></textarea>
                        </div>
                        <button class="send-btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentAgent = 'SANTAS_HELPER';
            let conversationId = null;

            function selectAgent(agent) {
                currentAgent = agent;
                document.querySelectorAll('.agent-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                event.target.closest('.agent-btn').classList.add('active');

                const greeting = agent === 'SANTAS_HELPER'
                    ? "Switched to SANTAS_HELPER. I'm ready to help with fund operations, NAV, and reporting."
                    : "Switched to CPA. I'm ready to help with taxes, audits, and regulatory filings.";

                addMessage('agent', greeting, agent);
            }

            function handleKeyDown(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }

            function sendQuickMessage(message) {
                document.getElementById('message-input').value = message;
                sendMessage();
            }

            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();

                if (!message) return;

                input.value = '';
                addMessage('user', message);

                document.getElementById('typing').classList.add('active');

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            agent: currentAgent,
                            conversation_id: conversationId
                        })
                    });

                    const data = await response.json();
                    conversationId = data.conversation_id;

                    document.getElementById('typing').classList.remove('active');
                    addMessage('agent', data.response, data.agent);

                } catch (error) {
                    document.getElementById('typing').classList.remove('active');
                    addMessage('agent', 'Sorry, I encountered an error. Please try again.', currentAgent);
                }
            }

            function addMessage(role, content, agent) {
                const messages = document.getElementById('messages');
                const div = document.createElement('div');

                const agentClass = agent === 'CPA' ? 'cpa' : 'santa';
                div.className = `message ${role} ${role === 'agent' ? agentClass : ''}`;

                const header = role === 'agent'
                    ? `<div class="message-header">${agent}</div>`
                    : '<div class="message-header">You</div>';

                const formattedContent = content.replace(/\\n/g, '<br>').replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');

                div.innerHTML = `
                    ${header}
                    <div class="message-content">${formattedContent}</div>
                `;

                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        return render_template_string(CHAT_HTML)

    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        message = data.get('message', '')
        agent = data.get('agent', 'SANTAS_HELPER')
        conv_id = data.get('conversation_id')

        result = chat_engine.process_message(
            message=message,
            user="Chris",
            conv_id=conv_id,
            target_agent=agent
        )

        return jsonify(result)

    @app.route('/api/daily-brief')
    def daily_brief():
        brief = chat_engine.get_daily_brief()
        return jsonify({"brief": brief})

    @app.route('/api/export-training-data')
    def export_training():
        filepath = chat_engine.conversation_store.save_to_file()
        return jsonify({"status": "success", "file": filepath})


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Chat Interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    if not FLASK_AVAILABLE:
        print("Flask is required. Install with: pip install flask")
        return

    logger.info(f"Starting Agent Chat Interface on http://{args.host}:{args.port}")
    logger.info("Agents available: SANTAS_HELPER, CPA")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

