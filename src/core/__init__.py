"""ALC-Algo Core Module
Author: Tom Hogan | Alpha Loop Capital, LLC

Core components for the ACA (Agent Coordination Architecture) system:
- BaseAgent: Base class for all agents with ACA capability
- AgentTier: Hierarchy levels for agents
- ACAEngine: Agent Creating Agents orchestration
- LearningOptimizer: Advanced learning optimization for all agents
- GenericAgent: Dynamically created agents
- Performance: High-performance utilities (caching, handlers, async)
- AgentMixin: Reusable patterns for agent implementation
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
from .agent_mixin import (
    AgentMixin,
    AsyncMixin,
    CachingMixin,
    LoggingMixin,
    ProcessMixin,
    ValidationMixin,
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
from .async_utils import (
    AsyncDataLoader,
    AsyncDataResult,
    AsyncTaskQueue,
    async_timer,
    async_to_sync,
    gather_with_timeout,
    get_async_loader,
    map_async,
    run_async,
    run_with_retry,
)
from .performance import (
    ExecutionTimer,
    HandlerRegistry,
    MetricAccumulator,
    SlidingWindow,
    TTLCache,
    async_batch_process,
    batch_process,
    gather_with_semaphore,
    memoize,
    memoize_method,
    run_sync,
    timed,
    ttl_cache,
)

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

    # Agent Mixins
    "AgentMixin",
    "ProcessMixin",
    "CachingMixin",
    "AsyncMixin",
    "LoggingMixin",
    "ValidationMixin",

    # Performance Utilities
    "TTLCache",
    "ttl_cache",
    "memoize",
    "memoize_method",
    "HandlerRegistry",
    "SlidingWindow",
    "MetricAccumulator",
    "gather_with_semaphore",
    "run_sync",
    "ExecutionTimer",
    "timed",
    "batch_process",
    "async_batch_process",

    # Async Utilities
    "AsyncDataLoader",
    "AsyncDataResult",
    "AsyncTaskQueue",
    "async_timer",
    "async_to_sync",
    "gather_with_timeout",
    "get_async_loader",
    "map_async",
    "run_async",
    "run_with_retry",

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
