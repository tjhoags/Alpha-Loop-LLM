"""
Alpha Loop LLM - Institutional-Grade Algorithmic Trading System
================================================================

Alpha Loop Capital, LLC

Main Components:
- agents: AI agent system (83+ specialized agents)
- core: ACA engine, base classes, utilities
- config: Configuration management
- data_ingestion: Multi-source data collection
- database: Azure SQL Server integration
- ml: Machine learning models & training
- trading: Execution engine & order management
- risk: Risk management & position sizing
- signals: Signal generation modules
- training: Agent training utilities

Quick Start:
    from src.config.settings import get_settings
    from src.data_ingestion.collector import run_collection_cycle
    from src.ml.train_models import main as train_models
    from src.trading.execution_engine import ExecutionEngine

Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Tom Hogan"
__email__ = "tom@alphaloopcapital.com"

# Lazy imports to avoid circular dependencies
def get_settings():
    from src.config.settings import get_settings
    return get_settings()

def get_hoags():
    from src.agents import get_hoags
    return get_hoags()

def get_ghost():
    from src.agents import get_ghost
    return get_ghost()

__all__ = [
    "__version__",
    "__author__",
    "get_settings",
    "get_hoags",
    "get_ghost",
]

