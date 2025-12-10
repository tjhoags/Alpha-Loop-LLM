"""
================================================================================
AGENT TRAINING FRAMEWORK
================================================================================
Alpha Loop Capital, LLC

Central training framework for all ALC agents. Provides:
- Individual agent training
- Multi-agent collaborative training
- Cross-agent skill transfer
- Training via external scripts

Usage:
    # Train single agent
    python -m src.agents.training.train_agent --agent GHOST
    
    # Train specific set
    python -m src.agents.training.train_agent --agents GHOST,SCOUT,AUTHOR
    
    # Random agent combination
    python -m src.agents.training.train_agent --random 3
    
    # Cross-agent training (GHOST and SCOUT inform AUTHOR using capital_agent script)
    python -m src.agents.training.train_agent --cross GHOST,SCOUT --target AUTHOR --via capital_agent
================================================================================
"""

from .train_agent import AgentTrainer, train_single_agent, train_agent_set, train_random_agents
from .cross_agent_trainer import CrossAgentTrainer, cross_train_agents

__all__ = [
    'AgentTrainer',
    'train_single_agent', 
    'train_agent_set',
    'train_random_agents',
    'CrossAgentTrainer',
    'cross_train_agents'
]

