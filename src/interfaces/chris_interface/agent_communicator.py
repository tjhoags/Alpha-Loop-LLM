"""
================================================================================
AGENT COMMUNICATOR - Direct Communication Interface for Chris Friedman
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

This module provides the backend communication layer for Chris Friedman (and Tom)
to interact directly with SANTAS_HELPER and CPA agents.

Features:
- Natural language conversation with agents
- Report requests and generation
- Task delegation and tracking
- Conversation history and context
- Anonymized data collection for training

================================================================================
"""

import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in conversations"""
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    REPORT_REQUEST = "report_request"
    REPORT_DELIVERY = "report_delivery"
    TASK_DELEGATION = "task_delegation"
    STATUS_UPDATE = "status_update"


class UserRole(Enum):
    """User roles with different access levels"""
    PRINCIPAL = "principal"  # Chris Friedman - full access
    OWNER = "owner"          # Tom Hogan - full access
    ANALYST = "analyst"      # Limited access
    VIEWER = "viewer"        # Read-only


@dataclass
class Message:
    """A message in a conversation"""
    message_id: str
    conversation_id: str
    timestamp: datetime
    sender: str  # user_id or agent_name
    recipient: str
    message_type: MessageType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "attachments": self.attachments
        }


@dataclass
class Conversation:
    """A conversation thread between user and agents"""
    conversation_id: str
    user_id: str
    user_role: UserRole
    started_at: datetime
    agents_involved: List[str] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "user_role": self.user_role.value,
            "started_at": self.started_at.isoformat(),
            "agents_involved": self.agents_involved,
            "message_count": len(self.messages),
            "is_active": self.is_active
        }


class AgentCommunicator:
    """
    Main communication interface for principals (Chris/Tom) to interact with
    SANTAS_HELPER and CPA agents.

    Features:
    - Natural language processing of requests
    - Routing to appropriate agent
    - Context management
    - Report generation
    - Training data collection
    """

    # Known user mappings
    KNOWN_USERS = {
        "CF": {"name": "Chris Friedman", "role": UserRole.PRINCIPAL},
        "TJH": {"name": "Tom Hogan", "role": UserRole.OWNER},
        "TOM": {"name": "Tom Hogan", "role": UserRole.OWNER},  # Alias
    }

    # Agent capabilities for routing
    AGENT_CAPABILITIES = {
        "SANTAS_HELPER": [
            "nav", "net asset value", "performance fee", "management fee",
            "gl", "general ledger", "financial statement", "lp report",
            "investor", "capital account", "allocation", "reconciliation",
            "daily operations", "fund accounting", "fee calculation"
        ],
        "CPA": [
            "tax", "k-1", "k1", "audit", "form pf", "adv", "13f",
            "regulatory", "compliance", "firm p&l", "pnl",
            "estimated tax", "provision", "irs", "fbar", "fatca",
            # IBKR Tax Optimization (for Tom)
            "ibkr", "tax loss", "harvest", "wash sale", "straddle",
            "section 1256", "1256", "60/40", "options tax", "holding period",
            "year end tax", "capital gains", "short term", "long term",
            "substantially identical", "loss deferral", "covered call"
        ]
    }

    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.message_count = 0

        # Agent references
        self.santas_helper = None
        self.cpa = None
        self.orchestrator = None

        # Training data collector
        self.training_collector = None

        logger.info("AgentCommunicator initialized")

    def initialize_agents(self) -> bool:
        """Initialize agent connections"""
        try:
            from src.agents.santas_helper_agent import get_santas_helper
            from src.agents.cpa_agent import get_cpa
            from src.agents.orchestrator_agent import get_orchestrator

            self.santas_helper = get_santas_helper()
            self.cpa = get_cpa()
            self.orchestrator = get_orchestrator()

            # Initialize training collector
            from src.training.fund_ops_training import TrainingDataCollector
            self.training_collector = TrainingDataCollector()

            logger.info("Agents connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            return False

    def start_conversation(
        self,
        user_id: str,
        initial_context: Dict[str, Any] = None
    ) -> Conversation:
        """Start a new conversation"""
        # Validate user
        user_info = self.KNOWN_USERS.get(user_id)
        if not user_info:
            raise ValueError(f"Unknown user: {user_id}")

        conv_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:6]}"

        conversation = Conversation(
            conversation_id=conv_id,
            user_id=user_id,
            user_role=user_info["role"],
            started_at=datetime.now(),
            context=initial_context or {}
        )

        self.conversations[conv_id] = conversation

        # Send welcome message
        welcome = self._generate_welcome_message(user_id, user_info["name"])
        self._add_system_message(conversation, welcome)

        return conversation

    def _generate_welcome_message(self, user_id: str, name: str) -> str:
        """Generate personalized welcome message"""
        if user_id == "CF":
            return (
                f"Good morning, {name}. SANTAS_HELPER and CPA are ready to assist you.\n\n"
                f"You can ask about:\n"
                f"• NAV and fund performance\n"
                f"• Fee calculations and investor reports\n"
                f"• Tax status and K-1 progress\n"
                f"• Audit coordination\n"
                f"• Daily operations status\n\n"
                f"How can we help you today?"
            )
        else:
            return (
                f"Hello, {name}. Fund operations agents are online.\n"
                f"SANTAS_HELPER and CPA are available for your requests."
            )

    def send_message(
        self,
        conversation_id: str,
        content: str,
        message_type: MessageType = MessageType.USER_MESSAGE
    ) -> Tuple[Message, Message]:
        """
        Send a message and get response from appropriate agent.

        Returns:
            Tuple of (user_message, agent_response)
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Unknown conversation: {conversation_id}")

        conversation = self.conversations[conversation_id]

        # Create user message
        self.message_count += 1
        user_msg = Message(
            message_id=f"msg_{self.message_count:06d}",
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            sender=conversation.user_id,
            recipient="AGENTS",
            message_type=message_type,
            content=content
        )
        conversation.messages.append(user_msg)

        # Route to appropriate agent
        agent_name, action = self._route_message(content)

        if agent_name not in conversation.agents_involved:
            conversation.agents_involved.append(agent_name)

        # Get agent response
        agent = self.santas_helper if agent_name == "SANTAS_HELPER" else self.cpa

        try:
            response_data = agent.process({
                "action": action,
                "user_message": content,
                "user_id": conversation.user_id,
                "context": conversation.context
            })

            response_content = self._format_agent_response(agent_name, response_data)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            response_content = f"I apologize, I encountered an issue processing your request: {str(e)}"

        # Create agent response message
        self.message_count += 1
        agent_msg = Message(
            message_id=f"msg_{self.message_count:06d}",
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            sender=agent_name,
            recipient=conversation.user_id,
            message_type=MessageType.AGENT_RESPONSE,
            content=response_content,
            metadata={"action": action, "raw_response": response_data}
        )
        conversation.messages.append(agent_msg)

        # Collect for training (anonymized)
        if self.training_collector:
            self.training_collector.collect_conversation(
                user=conversation.user_id,
                agent=agent_name,
                request={"message": content, "action": action},
                response=response_data
            )

        return user_msg, agent_msg

    def _route_message(self, content: str) -> Tuple[str, str]:
        """
        Route message to appropriate agent and determine action.

        Returns:
            (agent_name, action)
        """
        content_lower = content.lower()

        # Check SANTAS_HELPER keywords
        sh_score = sum(1 for kw in self.AGENT_CAPABILITIES["SANTAS_HELPER"] if kw in content_lower)

        # Check CPA keywords
        cpa_score = sum(1 for kw in self.AGENT_CAPABILITIES["CPA"] if kw in content_lower)

        # Route based on scores
        if cpa_score > sh_score:
            agent = "CPA"
        else:
            agent = "SANTAS_HELPER"  # Default to SANTAS_HELPER

        # Determine action
        action = self._determine_action(content_lower, agent)

        return agent, action

    def _determine_action(self, content: str, agent: str) -> str:
        """Determine the action to execute based on message content"""

        if agent == "SANTAS_HELPER":
            if "nav" in content or "net asset" in content:
                return "calculate_nav" if "calculate" in content else "get_status"
            elif "performance fee" in content or "incentive fee" in content:
                return "calculate_performance_fee"
            elif "management fee" in content:
                return "calculate_management_fee"
            elif "report" in content or "lp" in content or "investor" in content:
                return "generate_lp_report"
            elif "reconcil" in content:
                return "reconcile_nav"
            elif "status" in content or "update" in content:
                return "report_to_chris"
            elif "daily" in content or "operation" in content:
                return "run_daily_operations"
            else:
                return "report_to_chris"

        else:  # CPA
            # IBKR Tax Optimization actions (for Tom)
            if "ibkr" in content:
                if "tax loss" in content or "harvest" in content:
                    return "tax_loss_harvest_report"
                elif "wash sale" in content:
                    return "wash_sale_analysis"
                elif "straddle" in content:
                    return "straddle_analysis"
                elif "1256" in content or "60/40" in content:
                    return "section_1256_report"
                elif "year end" in content or "year-end" in content:
                    return "year_end_tax_plan"
                elif "holding period" in content:
                    return "holding_period_report"
                elif "option" in content:
                    return "options_tax_report"
                else:
                    return "analyze_ibkr_tax"
            elif "tax loss" in content or "harvest" in content:
                return "tax_loss_harvest_report"
            elif "wash sale" in content:
                return "wash_sale_analysis"
            elif "straddle" in content:
                return "straddle_analysis"
            elif "1256" in content or "60/40" in content or "section 1256" in content:
                return "section_1256_report"
            elif "year end tax" in content or "year-end tax" in content:
                return "year_end_tax_plan"
            elif "holding period" in content:
                return "holding_period_report"
            elif "options tax" in content or "option tax" in content:
                return "options_tax_report"
            # Standard CPA actions (for Chris)
            elif "k-1" in content or "k1" in content:
                return "generate_k1s"
            elif "audit" in content:
                return "coordinate_audit"
            elif "form pf" in content or "pf" in content:
                return "file_form_pf"
            elif "adv" in content:
                return "file_form_adv"
            elif "13f" in content:
                return "file_form_13f"
            elif "tax" in content:
                if "estimated" in content:
                    return "calculate_estimated_taxes"
                elif "return" in content:
                    return "prepare_fund_tax_return"
                else:
                    return "report_to_chris"
            elif "p&l" in content or "pnl" in content or "firm" in content:
                return "calculate_firm_pnl"
            elif "deadline" in content:
                return "get_deadlines"
            elif "status" in content or "update" in content:
                return "report_to_chris"
            else:
                return "report_to_chris"

    def _format_agent_response(self, agent_name: str, response: Dict) -> str:
        """Format agent response for display"""
        if response.get("status") != "success":
            return f"I encountered an issue: {response.get('message', 'Unknown error')}"

        # Format based on response type
        formatted = f"**{agent_name}**\n\n"

        # Check for different response types
        if "summary" in response:
            formatted += response["summary"] + "\n\n"

        if "key_metrics" in response:
            formatted += "**Key Metrics:**\n"
            for key, value in response["key_metrics"].items():
                formatted += f"• {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"

        if "action_items" in response and response["action_items"]:
            formatted += "**Action Items:**\n"
            for item in response["action_items"]:
                formatted += f"• {item}\n"
            formatted += "\n"

        if "items_for_awareness" in response and response["items_for_awareness"]:
            formatted += "**For Your Awareness:**\n"
            for item in response["items_for_awareness"]:
                formatted += f"• {item}\n"
            formatted += "\n"

        if "nav" in response:
            nav_data = response["nav"]
            formatted += "**NAV Details:**\n"
            formatted += f"• Net Asset Value: ${float(nav_data.get('net_asset_value', 0)):,.2f}\n"
            formatted += f"• NAV per Share: ${float(nav_data.get('nav_per_share', 0)):,.4f}\n"

        if "deadlines" in response:
            formatted += "**Upcoming Deadlines:**\n"
            for item in response.get("upcoming", []):
                formatted += f"• {item['item']}: {item['date']} ({item['status']})\n"

        # Add next update info
        if "next_update" in response:
            formatted += f"\n*{response['next_update']}*"

        return formatted.strip()

    def _add_system_message(self, conversation: Conversation, content: str):
        """Add a system message to conversation"""
        self.message_count += 1
        msg = Message(
            message_id=f"msg_{self.message_count:06d}",
            conversation_id=conversation.conversation_id,
            timestamp=datetime.now(),
            sender="SYSTEM",
            recipient=conversation.user_id,
            message_type=MessageType.SYSTEM_MESSAGE,
            content=content
        )
        conversation.messages.append(msg)

    def request_report(
        self,
        conversation_id: str,
        report_type: str,
        parameters: Dict[str, Any] = None
    ) -> Message:
        """Request a specific report from agents"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Unknown conversation: {conversation_id}")

        conversation = self.conversations[conversation_id]
        parameters = parameters or {}

        # Route report request
        report_agents = {
            "nav_pack": "SANTAS_HELPER",
            "performance_report": "SANTAS_HELPER",
            "lp_statement": "SANTAS_HELPER",
            "financial_statements": "SANTAS_HELPER",
            "k1_status": "CPA",
            "tax_status": "CPA",
            "audit_status": "CPA",
            "compliance_status": "CPA",
            "firm_pnl": "CPA"
        }

        agent_name = report_agents.get(report_type, "SANTAS_HELPER")
        agent = self.santas_helper if agent_name == "SANTAS_HELPER" else self.cpa

        # Generate report
        response = agent.process({
            "action": f"generate_{report_type}" if "generate" not in report_type else report_type,
            "parameters": parameters
        })

        # Create report message
        self.message_count += 1
        report_msg = Message(
            message_id=f"msg_{self.message_count:06d}",
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            sender=agent_name,
            recipient=conversation.user_id,
            message_type=MessageType.REPORT_DELIVERY,
            content=self._format_agent_response(agent_name, response),
            metadata={"report_type": report_type, "parameters": parameters},
            attachments=response.get("attachments", [])
        )
        conversation.messages.append(report_msg)

        return report_msg

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get conversation history"""
        if conversation_id not in self.conversations:
            return []

        conversation = self.conversations[conversation_id]
        messages = conversation.messages[-limit:]

        return [msg.to_dict() for msg in messages]

    def end_conversation(self, conversation_id: str) -> Dict:
        """End a conversation and save training data"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Unknown conversation: {conversation_id}")

        conversation = self.conversations[conversation_id]
        conversation.is_active = False

        # Save training data
        if self.training_collector:
            self.training_collector.save_to_azure()

        return {
            "status": "ended",
            "conversation_id": conversation_id,
            "message_count": len(conversation.messages),
            "agents_involved": conversation.agents_involved
        }


# Singleton
_communicator_instance: Optional[AgentCommunicator] = None


def get_communicator() -> AgentCommunicator:
    """Get singleton communicator instance"""
    global _communicator_instance
    if _communicator_instance is None:
        _communicator_instance = AgentCommunicator()
        _communicator_instance.initialize_agents()
    return _communicator_instance


if __name__ == "__main__":
    # Quick test
    comm = get_communicator()

    print("\n" + "="*60)
    print("AGENT COMMUNICATOR - Interactive Test")
    print("="*60)

    # Start conversation as Chris
    conv = comm.start_conversation("CF")
    print(f"\nConversation started: {conv.conversation_id}")

    # Test messages
    test_messages = [
        "What's the current NAV status?",
        "How are we doing on K-1 preparation?",
        "Give me a daily operations update"
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        user_msg, agent_msg = comm.send_message(conv.conversation_id, msg)
        print(f"\n{agent_msg.content}")

    # End conversation
    result = comm.end_conversation(conv.conversation_id)
    print(f"\nConversation ended: {result}")

