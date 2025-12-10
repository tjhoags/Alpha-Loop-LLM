"""ALC-Algo Core Module
Author: Tom Hogan | Alpha Loop Capital, LLC

Core components for the ACA (Agent Coordination Architecture) system:
- BaseAgent: Base class for all agents with ACA capability
- AgentTier: Hierarchy levels for agents
- ACAEngine: Agent Creating Agents orchestration
- LearningOptimizer: Advanced learning optimization for all agents
- GenericAgent: Dynamically created agents
"""

from .aca_engine import (
    ACAEngine,
    get_aca_engine,
)
from .agent_base import (
    AgentProposal,
    AgentStatus,
    AgentTier,
    AgentToughness,
    BaseAgent,
    CapabilityGap,
    CreativeInsight,
    LearningMethod,
    LearningOutcome,
    ThinkingMode,
)
from .event_bus import EventBus
from .learning_optimizer import (
    AdaptiveLearningRate,
    CrossMachineLearningSync,
    LearningOptimizer,
    MetaLearningCoordinator,
    PrioritizedExperienceReplay,
    learning_optimizer,
)
from .logger import ALCLogger

# from .generic_agent import (
#     GenericAgent,
#     StrategyGenericAgent,
#     SectorGenericAgent
# )

__all__ = [
    # Agent Base
    "BaseAgent",
    "AgentTier",
    "AgentStatus",
    "AgentProposal",
    "CapabilityGap",
    "ThinkingMode",
    "LearningMethod",
    "AgentToughness",
    "LearningOutcome",
    "CreativeInsight",

    # ACA Engine
    "ACAEngine",
    "get_aca_engine",

    # Learning Optimization
    "LearningOptimizer",
    "learning_optimizer",
    "PrioritizedExperienceReplay",
    "AdaptiveLearningRate",
    "MetaLearningCoordinator",
    "CrossMachineLearningSync",

    # Generic Agents
    # 'GenericAgent',
    # 'StrategyGenericAgent',
    # 'SectorGenericAgent',

    # Utilities
    "ALCLogger",
    "EventBus",
]
