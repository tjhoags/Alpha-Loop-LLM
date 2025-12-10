"""
Chris Friedman Agent Communication Interface
Provides direct communication channel to SANTAS_HELPER and CPA agents.
"""

from .agent_communicator import AgentCommunicator, get_communicator
from .conversation_handler import ConversationHandler

__all__ = ["AgentCommunicator", "get_communicator", "ConversationHandler"]

