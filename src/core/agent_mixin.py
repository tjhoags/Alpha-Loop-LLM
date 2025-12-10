"""================================================================================
AGENT MIXIN - High-Performance Patterns for Agent Implementation
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Reusable mixins that provide:
- Handler registry pattern for O(1) dispatch
- Optimized process() implementation
- Caching utilities
- Async support

Usage:
    from src.core.agent_mixin import ProcessMixin, CachingMixin

    class MyAgent(ProcessMixin, BaseAgent):

        def setup_handlers(self):
            self.register_handler("action_name", self._handle_action)

        def _handle_action(self, task):
            return {"success": True, ...}
================================================================================
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List

from src.core.performance import (
    HandlerRegistry,
    TTLCache,
    gather_with_semaphore,
)


# =============================================================================
# PROCESS MIXIN - Handler Registry Pattern
# =============================================================================


class ProcessMixin:
    """Mixin providing optimized handler registry pattern.

    Replaces repetitive handler dict creation in process() methods
    with O(1) dispatch.

    Usage:
        class MyAgent(ProcessMixin, BaseAgent):

            def setup_handlers(self):
                self.register_handler("scan", self._handle_scan)
                self.register_handler("analyze", self._handle_analyze)

            def _handle_scan(self, task: Dict[str, Any]) -> Dict[str, Any]:
                ...
    """

    _handler_registry: HandlerRegistry = None

    def __init_subclass__(cls, **kwargs):
        """Initialize handler registry for subclass."""
        super().__init_subclass__(**kwargs)

    def _ensure_registry(self) -> HandlerRegistry:
        """Ensure handler registry exists."""
        if self._handler_registry is None:
            self._handler_registry = HandlerRegistry(name=getattr(self, "name", "agent"))
            # Call setup_handlers if defined
            if hasattr(self, "setup_handlers"):
                self.setup_handlers()
        return self._handler_registry

    def register_handler(
        self,
        action: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """Register a handler for an action type."""
        registry = self._ensure_registry()
        registry.register_handler(action, handler)

    def register_handlers(
        self,
        handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]
    ) -> None:
        """Register multiple handlers at once."""
        registry = self._ensure_registry()
        registry.register_handlers(handlers)

    def set_default_handler(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """Set default handler for unknown actions."""
        registry = self._ensure_registry()
        registry.set_default(handler)

    def dispatch(self, action: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch task to appropriate handler - O(1) lookup."""
        registry = self._ensure_registry()
        return registry.dispatch(action, task)

    def get_supported_actions(self) -> List[str]:
        """Get list of supported actions."""
        registry = self._ensure_registry()
        return registry.get_actions()

    def process_with_registry(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task using handler registry.

        Call this from your process() method:
            def process(self, task):
                return self.process_with_registry(task)
        """
        action = task.get("action") or task.get("type", "unknown")
        return self.dispatch(action, task)


# =============================================================================
# CACHING MIXIN - TTL and Memoization Support
# =============================================================================


class CachingMixin:
    """Mixin providing caching utilities.

    Usage:
        class MyAgent(CachingMixin, BaseAgent):

            def expensive_calculation(self, key: str):
                return self.cached_get(
                    f"calc_{key}",
                    lambda: self._do_expensive_calculation(key),
                    ttl_seconds=300
                )
    """

    _cache: TTLCache = None
    _result_cache: TTLCache = None

    def _ensure_cache(self) -> TTLCache:
        """Ensure cache exists."""
        if self._cache is None:
            self._cache = TTLCache(ttl_seconds=300.0, max_size=1000)
        return self._cache

    def _ensure_result_cache(self) -> TTLCache:
        """Ensure result cache exists (longer TTL for expensive computations)."""
        if self._result_cache is None:
            self._result_cache = TTLCache(ttl_seconds=3600.0, max_size=500)
        return self._result_cache

    def cached_get(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: float = 300.0
    ) -> Any:
        """Get value from cache or compute and store.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl_seconds: Time-to-live (uses short cache if < 600, else long)
        """
        cache = self._ensure_cache() if ttl_seconds < 600 else self._ensure_result_cache()

        result = cache.get(key)
        if result is not None:
            return result

        result = compute_fn()
        cache.set(key, result)
        return result

    def invalidate_cache(self, key: str = None) -> None:
        """Invalidate cache entry or entire cache."""
        if key is None:
            if self._cache:
                self._cache.clear()
            if self._result_cache:
                self._result_cache.clear()
        else:
            # Individual key invalidation not supported by TTLCache
            # Clear entire cache
            if self._cache:
                self._cache.clear()


# =============================================================================
# ASYNC MIXIN - Async Processing Support
# =============================================================================


class AsyncMixin:
    """Mixin providing async processing support.

    Usage:
        class MyAgent(AsyncMixin, BaseAgent):

            async def async_process(self, task):
                results = await self.gather_tasks([
                    self.async_fetch(url1),
                    self.async_fetch(url2),
                ])
                return {"results": results}
    """

    _semaphore: asyncio.Semaphore = None
    _max_concurrent: int = 10

    def _ensure_semaphore(self) -> asyncio.Semaphore:
        """Ensure semaphore exists for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def gather_tasks(
        self,
        coros,
        max_concurrent: int = None,
        return_exceptions: bool = True
    ) -> List[Any]:
        """Run multiple coroutines with concurrency limit."""
        max_conc = max_concurrent or self._max_concurrent
        return await gather_with_semaphore(
            *coros,
            max_concurrent=max_conc,
            return_exceptions=return_exceptions
        )

    async def run_with_timeout(
        self,
        coro,
        timeout_seconds: float = 30.0,
        default: Any = None
    ) -> Any:
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return default

    def run_sync(self, coro) -> Any:
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


# =============================================================================
# LOGGING MIXIN - Structured Logging
# =============================================================================


class LoggingMixin:
    """Mixin providing structured logging utilities."""

    def log_action(self, action: str, message: str, **kwargs) -> None:
        """Log an action with structured data."""
        logger = getattr(self, "logger", None)
        if logger:
            extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info(f"[{getattr(self, 'name', 'AGENT')}] {action}: {message} {extra}".strip())

    def log_metric(self, metric_name: str, value: float, **tags) -> None:
        """Log a metric value."""
        logger = getattr(self, "logger", None)
        if logger:
            tag_str = " ".join(f"{k}={v}" for k, v in tags.items())
            logger.info(f"METRIC {metric_name}={value} {tag_str}".strip())

    def log_timing(self, operation: str, duration_ms: float) -> None:
        """Log operation timing."""
        self.log_metric(f"{operation}_duration_ms", duration_ms)


# =============================================================================
# VALIDATION MIXIN - Input Validation
# =============================================================================


class ValidationMixin:
    """Mixin providing task validation utilities."""

    def validate_task(
        self,
        task: Dict[str, Any],
        required_fields: List[str] = None,
        optional_fields: List[str] = None
    ) -> Dict[str, Any]:
        """Validate task and return validation result.

        Returns:
            {"valid": True/False, "errors": [...], "task": task_with_defaults}
        """
        errors = []

        # Validate type
        if not isinstance(task, dict):
            return {
                "valid": False,
                "errors": [f"Task must be dict, got {type(task).__name__}"],
                "task": {}
            }

        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in task:
                    errors.append(f"Missing required field: {field}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "task": task
        }

    def require_fields(self, task: Dict[str, Any], *fields: str) -> None:
        """Raise ValueError if required fields missing."""
        missing = [f for f in fields if f not in task]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")


# =============================================================================
# COMBINED AGENT MIXIN
# =============================================================================


class AgentMixin(ProcessMixin, CachingMixin, LoggingMixin, ValidationMixin):
    """Combined mixin with all agent utilities.

    Usage:
        class MyAgent(AgentMixin, BaseAgent):
            def setup_handlers(self):
                self.register_handlers({
                    "scan": self._handle_scan,
                    "analyze": self._handle_analyze,
                })
                self.set_default_handler(self._handle_unknown)

            def process(self, task):
                return self.process_with_registry(task)
    """
    pass


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "ProcessMixin",
    "CachingMixin",
    "AsyncMixin",
    "LoggingMixin",
    "ValidationMixin",
    "AgentMixin",
]

