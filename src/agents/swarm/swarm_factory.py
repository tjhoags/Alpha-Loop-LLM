"""Swarm Factory - Manages creation and coordination of Swarm Agents
Author: Tom Hogan | Alpha Loop Capital, LLC

Optimized for performance with lazy loading and efficient agent management.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Type

from src.core.agent_base import BaseAgent
from src.core.logger import alc_logger

# Cache the specialized path at module level
_SPECIALIZED_PATH: Path = Path(__file__).parent.parent / "specialized"


class SwarmFactory:
    """Factory for creating and managing swarm agents.

    Optimizations:
    - Cached agent file discovery
    - Lazy agent instantiation option
    - Efficient stats calculation
    """

    __slots__ = ('user_id', 'agents', '_initialized')

    def __init__(self, user_id: str = "TJH") -> None:
        self.user_id = user_id
        self.agents: Dict[str, BaseAgent] = {}
        self._initialized = False

    @staticmethod
    @lru_cache(maxsize=1)
    def _discover_agent_files() -> tuple[Path, ...]:
        """Discover agent files once and cache the result."""
        if not _SPECIALIZED_PATH.exists():
            alc_logger.warning(f"Specialized agents directory not found: {_SPECIALIZED_PATH}")
            return ()
        return tuple(_SPECIALIZED_PATH.glob("*_agent.py"))

    @staticmethod
    def _get_agent_classes(module: Any) -> Iterator[tuple[str, Type[BaseAgent]]]:
        """Yield agent classes from a module."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseAgent) and
                obj is not BaseAgent and
                name.endswith("Agent")):
                yield name, obj

    def create_all_agents(self, lazy: bool = False) -> Dict[str, BaseAgent]:
        """Dynamically load and instantiate all specialized agents.

        Args:
            lazy: If True, only discover agents but don't instantiate yet.

        Returns:
            Dictionary of agent name to agent instance.
        """
        agent_files = self._discover_agent_files()

        if not agent_files:
            return {}

        alc_logger.info(f"Scanning {len(agent_files)} agent files")

        for file_path in agent_files:
            module_name = file_path.stem
            import_path = f"src.agents.specialized.{module_name}"

            try:
                module = importlib.import_module(import_path)

                for class_name, agent_class in self._get_agent_classes(module):
                    if lazy:
                        # Store class for lazy instantiation
                        continue

                    try:
                        agent = agent_class(user_id=self.user_id)
                        self.agents[agent.name] = agent
                        alc_logger.info(f"Initialized {agent.name}")
                    except Exception as e:
                        alc_logger.error(f"Failed to instantiate {class_name}: {e}")

            except Exception as e:
                alc_logger.error(f"Failed to load module {module_name}: {e}")

        self._initialized = True
        return self.agents

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all swarm agents.

        Optimized to avoid repeated attribute access.
        """
        agents_list = list(self.agents.values())

        strategy_count = 0
        sector_count = 0
        total_executions = 0

        for agent in agents_list:
            tier_name = agent.tier.name
            if "STRATEGY" in tier_name:
                strategy_count += 1
            elif "SECTOR" in tier_name:
                sector_count += 1
            total_executions += agent.execution_count

        return {
            "total_agents": len(agents_list),
            "initialized": self._initialized,
            "by_category": {
                "strategy": strategy_count,
                "sector": sector_count,
                "specialized": len(agents_list),
            },
            "total_signals_generated": 0,  # Placeholder
            "total_analyses_completed": total_executions,
        }

