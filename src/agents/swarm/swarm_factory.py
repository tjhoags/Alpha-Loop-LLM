"""
Swarm Factory - Manages creation and coordination of Swarm Agents
Author: Tom Hogan | Alpha Loop Capital, LLC
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List
from src.core.agent_base import BaseAgent
from src.core.logger import alc_logger

class SwarmFactory:
    """Factory for creating and managing swarm agents."""
    
    def __init__(self, user_id: str = "TJH"):
        self.user_id = user_id
        self.agents: Dict[str, BaseAgent] = {}
        
    def create_all_agents(self) -> Dict[str, BaseAgent]:
        """Dynamically load and instantiate all specialized agents."""
        # Path to specialized agents
        specialized_path = Path(__file__).parent.parent / "specialized"
        
        alc_logger.info(f"Scanning for specialized agents in {specialized_path}")
        
        if not specialized_path.exists():
            alc_logger.warning(f"Specialized agents directory not found: {specialized_path}")
            return {}

        # Iterate over files
        for file_path in specialized_path.glob("*_agent.py"):
            module_name = file_path.stem
            try:
                # Import module
                # We assume the project structure allows this import path
                import_path = f"src.agents.specialized.{module_name}"
                module = importlib.import_module(import_path)
                
                # Find agent classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseAgent) and 
                        obj is not BaseAgent and
                        name.endswith("Agent")):
                        
                        # Instantiate
                        try:
                            agent = obj(user_id=self.user_id)
                            self.agents[agent.name] = agent
                            alc_logger.info(f"Initialized {agent.name}")
                        except Exception as e:
                            alc_logger.error(f"Failed to instantiate {name}: {e}")
                            
            except Exception as e:
                alc_logger.error(f"Failed to load module {module_name}: {e}")
                
        return self.agents

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all swarm agents."""
        return {
            'total_agents': len(self.agents),
            'by_category': {
                'strategy': sum(1 for a in self.agents.values() if 'Strategy' in a.tier.name),
                'sector': sum(1 for a in self.agents.values() if 'Sector' in a.tier.name),
                'specialized': len(self.agents)
            }, 
            'total_signals_generated': 0, # Placeholder
            'total_analyses_completed': sum(a.execution_count for a in self.agents.values())
        }

