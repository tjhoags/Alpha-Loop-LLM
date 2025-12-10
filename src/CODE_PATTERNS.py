"""================================================================================
CODE PATTERNS - Standardized Patterns for ALC-Algo Development
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

This file defines the CANONICAL patterns used throughout the codebase.
AI assistants and developers should reference these patterns to ensure
consistency and reduce errors.

USAGE:
    When creating new agents, data ingestion, or ML components, follow
    these patterns exactly. Do NOT deviate without explicit approval.

================================================================================
"""

# =============================================================================
# PATTERN 1: AGENT CREATION
# =============================================================================
#
# All agents MUST inherit from BaseAgent and use the standard structure:
#
# class MyNewAgent(AgentMixin, BaseAgent):
#     """
#     AGENT_NAME - One-line description
#
#     Tier: STRATEGY (5) / OPERATIONAL (4) / SENIOR (3) / MASTER (1-2)
#     Reports To: PARENT_AGENT
#     Cluster: cluster_name
#     """
#
#     def __init__(self):
#         super().__init__(
#             name="AGENT_NAME",
#             tier=AgentTier.STRATEGY,
#             thinking_mode=ThinkingMode.ANALYTICAL,
#             learning_methods=[LearningMethod.SUPERVISED, ...],
#         )
#         self._setup_capabilities()
#
#     def _setup_capabilities(self) -> None:
#         self.capabilities = ["capability_1", "capability_2", ...]
#
#     def setup_handlers(self) -> None:
#         """Register task handlers using ProcessMixin pattern."""
#         self.register_handlers({
#             "action_1": self._handle_action_1,
#             "action_2": self._handle_action_2,
#         })
#         self.set_default_handler(self._handle_unknown)
#
#     def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
#         """Process task using handler registry."""
#         return self.process_with_registry(task)
#
#     def _handle_action_1(self, task: Dict[str, Any]) -> Dict[str, Any]:
#         """Handle action_1 tasks."""
#         return {"success": True, "result": ...}
#


# =============================================================================
# PATTERN 2: SETTINGS/CONFIG ACCESS
# =============================================================================
#
# Always use get_settings() from src.config.settings:
#
# from src.config.settings import get_settings
#
# settings = get_settings()
#
# # Access API keys (use new names, aliases exist for backward compatibility):
# massive_key = settings.massive_api_key       # Primary (Massive.com)
# polygon_key = settings.polygon_api_key       # Alias for massive_api_key
# alpha_key = settings.alpha_vantage_api_key
# fred_key = settings.fred_api_key
#
# # Access paths:
# data_path = settings.data_dir
# model_path = settings.models_dir
# log_path = settings.logs_dir
#


# =============================================================================
# PATTERN 3: DATA INGESTION
# =============================================================================
#
# For fetching market data, use the standardized collectors:
#
# from src.data_ingestion.sources.polygon import fetch_aggregates as massive_fetch
# from src.data_ingestion.sources.alpha_vantage import fetch_intraday as av_fetch
# from src.data_ingestion.sources.fred import FredClient
# from src.data_ingestion.sources.coinbase import fetch_candles as cb_fetch
#
# # Massive.com (formerly Polygon.io):
# df = massive_fetch(symbol="AAPL", timespan="minute", multiplier=5)
#
# # Alpha Vantage:
# df = av_fetch(symbol="AAPL", interval="5min")
#
# # FRED:
# fred = FredClient()
# data = fred.get_series("VIXCLS")
#


# =============================================================================
# PATTERN 4: DATABASE ACCESS
# =============================================================================
#
# Always use the connection module:
#
# from src.database.connection import get_engine
#
# engine = get_engine()
#
# # For reading:
# df = pd.read_sql("SELECT * FROM price_bars WHERE symbol = 'AAPL'", engine)
#
# # For writing:
# df.to_sql("price_bars", engine, if_exists="append", index=False)
#


# =============================================================================
# PATTERN 5: LOGGING
# =============================================================================
#
# Use loguru for all logging:
#
# from loguru import logger
#
# logger.info("Informational message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.debug("Debug message")
#
# # With structured data:
# logger.info(f"Fetched {len(df)} rows for {symbol}")
#


# =============================================================================
# PATTERN 6: ERROR HANDLING
# =============================================================================
#
# ALWAYS use specific exception types, NEVER bare except:
#
# try:
#     result = risky_operation()
# except ValueError as e:
#     logger.error(f"Value error: {e}")
#     return {"success": False, "error": str(e)}
# except requests.HTTPError as e:
#     logger.error(f"HTTP error: {e}")
#     return {"success": False, "error": str(e)}
# except Exception as e:  # Only as last resort
#     logger.exception(f"Unexpected error: {e}")
#     raise
#


# =============================================================================
# PATTERN 7: CACHING
# =============================================================================
#
# Use TTLCache for time-sensitive caching:
#
# from src.core.performance import TTLCache, ttl_cache
#
# # Class-level cache:
# cache = TTLCache(ttl_seconds=300, max_size=1000)
# cache.set("key", value)
# result = cache.get("key")
#
# # Function decorator:
# @ttl_cache(ttl_seconds=300)
# def expensive_function(arg):
#     return compute(arg)
#
# # For agents using CachingMixin:
# class MyAgent(CachingMixin, BaseAgent):
#     def get_data(self, key):
#         return self.cached_get(
#             f"data_{key}",
#             lambda: self._fetch_data(key),
#             ttl_seconds=300
#         )
#


# =============================================================================
# PATTERN 8: ASYNC OPERATIONS
# =============================================================================
#
# Use async utilities from core:
#
# from src.core.async_utils import run_async, gather_with_timeout
#
# # Run async from sync context:
# result = run_async(async_function())
#
# # Gather with timeout:
# results = await gather_with_timeout(
#     [coro1, coro2, coro3],
#     timeout_seconds=30,
#     return_exceptions=True
# )
#


# =============================================================================
# PATTERN 9: DATACLASSES
# =============================================================================
#
# Use Python 3.10+ dataclass with slots for performance:
#
# from dataclasses import dataclass
#
# @dataclass(slots=True)  # Use slots=True, NOT __slots__ manual definition
# class MyDataClass:
#     field1: str
#     field2: int
#     field3: Optional[float] = None
#
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "field1": self.field1,
#             "field2": self.field2,
#             "field3": self.field3,
#         }
#


# =============================================================================
# PATTERN 10: TYPE HINTS
# =============================================================================
#
# Always use type hints. For complex types, use TYPE_CHECKING:
#
# from typing import TYPE_CHECKING, Any, Dict, List, Optional
#
# if TYPE_CHECKING:
#     import pandas as pd
#     from src.core.agent_base import BaseAgent
#
# def my_function(
#     data: "pd.DataFrame",
#     agent: Optional["BaseAgent"] = None
# ) -> Dict[str, Any]:
#     ...
#


# =============================================================================
# PATTERN 11: SINGLETON PATTERN
# =============================================================================
#
# For agents that should only have one instance:
#
# _instance: Optional[MyAgent] = None
#
# def get_my_agent() -> MyAgent:
#     global _instance
#     if _instance is None:
#         _instance = MyAgent()
#     return _instance
#


# =============================================================================
# PATTERN 12: API ENDPOINTS
# =============================================================================
#
# Standard API base URLs (use these, NOT hardcoded):
#
# MASSIVE_API_URL = "https://api.massive.com"  # Formerly Polygon.io
# ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
# FRED_URL = "https://api.stlouisfed.org/fred"
# COINBASE_URL = "https://api.coinbase.com"
# OPENAI_URL = "https://api.openai.com/v1"
#


# =============================================================================
# ANTI-PATTERNS (DO NOT USE)
# =============================================================================
#
# 1. NEVER use bare except:
#    BAD:  except:
#    GOOD: except Exception:
#
# 2. NEVER use manual __slots__ in dataclasses:
#    BAD:  __slots__ = ['field1', 'field2']
#    GOOD: @dataclass(slots=True)
#
# 3. NEVER hardcode API URLs:
#    BAD:  url = "https://api.polygon.io/..."
#    GOOD: url = f"{MASSIVE_API_URL}/..."
#
# 4. NEVER use polygon.io references in new code:
#    BAD:  "Polygon.io", "polygon_api_key"
#    GOOD: "Massive.com", "massive_api_key"
#
# 5. NEVER create duplicate handler dicts:
#    BAD:  handlers = {"action": self._handle}; handlers[action](task)
#    GOOD: Use ProcessMixin.process_with_registry(task)
#
# 6. NEVER use print() for logging:
#    BAD:  print("message")
#    GOOD: logger.info("message")
#


# =============================================================================
# IMPORTS - STANDARD ORDER
# =============================================================================
#
# Always organize imports in this order:
#
# 1. Standard library
# import os
# import sys
# from datetime import datetime
# from typing import Any, Dict, List, Optional
#
# 2. Third-party
# import pandas as pd
# import numpy as np
# from loguru import logger
#
# 3. Local imports (absolute)
# from src.config.settings import get_settings
# from src.core import BaseAgent, AgentTier
# from src.core.agent_mixin import AgentMixin
#


# This module is documentation only - no exports needed

