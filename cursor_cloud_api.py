"""
Cursor Cloud Agent API Client
API Documentation: https://cursor.com/docs/cloud-agent/api/endpoints

This module provides helper functions for interacting with the Cursor Cloud Agent API.
"""

import os
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class AgentStatus(Enum):
    """Possible agent status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class AgentConfig:
    """Configuration for launching a new agent"""
    prompt_text: str
    repository: str
    model: Optional[str] = None  # e.g., "claude-3-5-sonnet", "gpt-4o"
    target_branch: Optional[str] = None
    webhook_url: Optional[str] = None


class CursorCloudAPI:
    """
    Client for the Cursor Cloud Agent API.

    Usage:
        api = CursorCloudAPI(api_key="your_api_key")
        # or use CURSOR_API_KEY environment variable
        api = CursorCloudAPI()

        # List agents
        agents = api.list_agents()

        # Launch new agent
        agent = api.launch_agent(
            prompt_text="Fix the bug in auth.py",
            repository="owner/repo"
        )

        # Stop an agent
        api.stop_agent(agent_id)
    """

    BASE_URL = "https://api.cursor.com"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            api_key: Cursor API key. If not provided, reads from CURSOR_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("CURSOR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide api_key parameter or set CURSOR_API_KEY environment variable. "
                "Get your API key from https://cursor.com/settings"
            )
        self.session = requests.Session()
        self.session.auth = (self.api_key, "")
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to the API"""
        url = f"{self.BASE_URL}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    # ==================== Agent Management ====================

    def list_agents(self, limit: int = 100, cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        List all cloud agents.

        Args:
            limit: Maximum number of agents to return (max 100)
            cursor: Pagination cursor from previous response

        Returns:
            Dict with 'agents' list and optional 'next_cursor'
        """
        params = {"limit": min(limit, 100)}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/v0/agents", params=params)

    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get status and details of a specific agent.

        Args:
            agent_id: The agent ID

        Returns:
            Agent details including status, created_at, etc.
        """
        return self._request("GET", f"/v0/agents/{agent_id}")

    def get_conversation(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the full message history for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            Dict with 'messages' list containing the conversation history
        """
        return self._request("GET", f"/v0/agents/{agent_id}/conversation")

    def launch_agent(
        self,
        prompt_text: str,
        repository: str,
        model: Optional[str] = None,
        target_branch: Optional[str] = None,
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Launch a new cloud agent.

        Args:
            prompt_text: The task/prompt for the agent
            repository: GitHub repository in "owner/repo" format
            model: Optional model to use (e.g., "claude-3-5-sonnet", "gpt-4o")
            target_branch: Optional branch to target
            webhook_url: Optional webhook URL for status updates

        Returns:
            Dict with agent details including 'id'
        """
        payload = {
            "prompt": {"text": prompt_text},
            "source": {"repository": repository}
        }
        if model:
            payload["model"] = model
        if target_branch:
            payload["target"] = {"branch": target_branch}
        if webhook_url:
            payload["webhook"] = {"url": webhook_url}

        return self._request("POST", "/v0/agents", json=payload)

    def send_followup(
        self,
        agent_id: str,
        prompt_text: str,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send a follow-up instruction to a running agent.

        Args:
            agent_id: The agent ID
            prompt_text: The follow-up instruction
            images: Optional list of image URLs or base64 data

        Returns:
            Updated agent details
        """
        payload = {"prompt": {"text": prompt_text}}
        if images:
            payload["prompt"]["images"] = images
        return self._request("POST", f"/v0/agents/{agent_id}/followup", json=payload)

    def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Stop a running agent.

        Args:
            agent_id: The agent ID

        Returns:
            Confirmation of stop request
        """
        return self._request("POST", f"/v0/agents/{agent_id}/stop")

    def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Permanently delete an agent and its data.

        Args:
            agent_id: The agent ID

        Returns:
            Confirmation of deletion
        """
        return self._request("DELETE", f"/v0/agents/{agent_id}")

    # ==================== Account & Info ====================

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get information about the API key.

        Returns:
            Account details associated with the API key
        """
        return self._request("GET", "/v0/me")

    def list_models(self) -> Dict[str, Any]:
        """
        List available models for cloud agents.

        Returns:
            Dict with 'models' list
        """
        return self._request("GET", "/v0/models")

    def list_repositories(self) -> Dict[str, Any]:
        """
        List available GitHub repositories.

        Note: Rate limited to 1 request/minute, 30 requests/hour per user.

        Returns:
            Dict with 'repositories' list
        """
        return self._request("GET", "/v0/repositories")


# ==================== Convenience Functions ====================

def create_client(api_key: Optional[str] = None) -> CursorCloudAPI:
    """Create a new Cursor Cloud API client"""
    return CursorCloudAPI(api_key)


def quick_launch(
    prompt: str,
    repo: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quickly launch an agent with minimal configuration.

    Args:
        prompt: The task for the agent
        repo: GitHub repository (owner/repo format)
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Agent details including ID
    """
    client = CursorCloudAPI(api_key)
    return client.launch_agent(prompt_text=prompt, repository=repo)


if __name__ == "__main__":
    # Example usage / test
    import json

    print("Cursor Cloud Agent API Client")
    print("=" * 40)

    try:
        api = CursorCloudAPI()

        # Get account info
        print("\nAccount Info:")
        info = api.get_account_info()
        print(json.dumps(info, indent=2))

        # List available models
        print("\nAvailable Models:")
        models = api.list_models()
        print(json.dumps(models, indent=2))

        # List agents
        print("\nYour Agents:")
        agents = api.list_agents(limit=5)
        print(json.dumps(agents, indent=2))

    except ValueError as e:
        print(f"\nSetup required: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"\nAPI Error: {e}")
