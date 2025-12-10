"""================================================================================
PERFORMANCE UTILITIES - Optimized Decorators and Mixins for Agent Speed
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

High-performance utilities for agent execution:
- Caching decorators with TTL support
- Memoization for expensive computations
- Handler registry pattern for O(1) dispatch
- Optimized data structures
- Async utilities

PHILOSOPHY: Every millisecond matters. No compute wasted.
================================================================================
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import time
import weakref
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# TTL CACHE - Time-based expiration cache
# =============================================================================


class TTLCache(Generic[T]):
    """Fast TTL-based cache with O(1) access and automatic cleanup."""

    __slots__ = ("_cache", "_ttl_seconds", "_max_size", "_timestamps")

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self._cache: Dict[Hashable, T] = {}
        self._timestamps: Dict[Hashable, float] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size

    def get(self, key: Hashable, default: T = None) -> Optional[T]:
        """Get value if exists and not expired."""
        if key not in self._cache:
            return default

        if time.monotonic() - self._timestamps[key] > self._ttl_seconds:
            del self._cache[key]
            del self._timestamps[key]
            return default

        return self._cache[key]

    def set(self, key: Hashable, value: T) -> None:
        """Set value with current timestamp."""
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]

        self._cache[key] = value
        self._timestamps[key] = time.monotonic()

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._timestamps.clear()

    def __contains__(self, key: Hashable) -> bool:
        return self.get(key) is not None


def ttl_cache(ttl_seconds: float = 300.0, max_size: int = 128):
    """Decorator for TTL-based caching of function results.

    Usage:
        @ttl_cache(ttl_seconds=60)
        def expensive_calculation(x, y):
            ...
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        cache = TTLCache[R](ttl_seconds=ttl_seconds, max_size=max_size)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            # Create hashable key from args/kwargs
            key = _make_cache_key(args, kwargs)

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache_clear = cache.clear
        wrapper.cache = cache
        return wrapper

    return decorator


def _make_cache_key(args: tuple, kwargs: dict) -> str:
    """Create hashable cache key from function arguments."""
    key_parts = [repr(arg) for arg in args]
    key_parts.extend(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()


# =============================================================================
# MEMOIZE - Persistent result caching
# =============================================================================


def memoize(func: Callable[..., R]) -> Callable[..., R]:
    """Simple memoization decorator using functools.lru_cache.

    For methods that don't change frequently.
    """
    return functools.lru_cache(maxsize=256)(func)


def memoize_method(func: Callable[..., R]) -> Callable[..., R]:
    """Memoize instance methods without memory leaks.

    Uses weak references to prevent keeping instances alive.
    """
    cache: Dict[int, functools._lru_cache_wrapper] = {}

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> R:
        instance_id = id(self)
        if instance_id not in cache:
            # Create per-instance cache
            @functools.lru_cache(maxsize=128)
            def cached_method(*a, **kw):
                return func(self, *a, **kw)

            cache[instance_id] = cached_method
            # Register cleanup when instance is garbage collected
            weakref.finalize(self, cache.pop, instance_id, None)

        return cache[instance_id](*args, **kwargs)

    return wrapper


# =============================================================================
# HANDLER REGISTRY - O(1) Task Dispatch
# =============================================================================


@runtime_checkable
class TaskHandler(Protocol):
    """Protocol for task handlers."""

    def __call__(self, task: Dict[str, Any]) -> Dict[str, Any]: ...


class HandlerRegistry:
    """Fast handler dispatch registry with O(1) lookup.

    Replaces repeated if/elif chains and dict lookups in process() methods.

    Usage:
        registry = HandlerRegistry()

        @registry.register("scan")
        def handle_scan(task):
            ...

        # Or register manually
        registry.register_handler("analyze", handle_analyze)

        # Dispatch
        result = registry.dispatch("scan", task_data)
    """

    __slots__ = ("_handlers", "_default_handler", "_name")

    def __init__(self, name: str = "default"):
        self._handlers: Dict[str, TaskHandler] = {}
        self._default_handler: Optional[TaskHandler] = None
        self._name = name

    def register(self, action: str) -> Callable[[TaskHandler], TaskHandler]:
        """Decorator to register a handler for an action."""
        def decorator(handler: TaskHandler) -> TaskHandler:
            self._handlers[action] = handler
            return handler
        return decorator

    def register_handler(self, action: str, handler: TaskHandler) -> None:
        """Register a handler directly."""
        self._handlers[action] = handler

    def register_handlers(self, handlers: Dict[str, TaskHandler]) -> None:
        """Register multiple handlers at once."""
        self._handlers.update(handlers)

    def set_default(self, handler: TaskHandler) -> None:
        """Set default handler for unknown actions."""
        self._default_handler = handler

    def dispatch(
        self,
        action: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch task to appropriate handler."""
        handler = self._handlers.get(action)
        if handler is None:
            if self._default_handler is not None:
                return self._default_handler(task)
            return {"success": False, "error": f"Unknown action: {action}"}
        return handler(task)

    def get_actions(self) -> List[str]:
        """Get list of registered actions."""
        return list(self._handlers.keys())


# =============================================================================
# OPTIMIZED DATA STRUCTURES
# =============================================================================


class SlidingWindow(Generic[T]):
    """Fixed-size sliding window with O(1) operations.

    More memory-efficient than deque for our use case.
    """

    __slots__ = ("_data", "_size", "_index", "_count")

    def __init__(self, size: int):
        self._size = size
        self._data: List[Optional[T]] = [None] * size
        self._index = 0
        self._count = 0

    def append(self, value: T) -> None:
        """Add value to window."""
        self._data[self._index] = value
        self._index = (self._index + 1) % self._size
        if self._count < self._size:
            self._count += 1

    def get_recent(self, n: int = None) -> List[T]:
        """Get most recent n values (or all if n is None)."""
        n = min(n or self._count, self._count)
        if n == 0:
            return []

        result = []
        idx = (self._index - 1) % self._size
        for _ in range(n):
            if self._data[idx] is not None:
                result.append(self._data[idx])
            idx = (idx - 1) % self._size
        return result

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        """Iterate from oldest to newest."""
        if self._count < self._size:
            # Not full yet, start from 0
            for i in range(self._count):
                yield self._data[i]
        else:
            # Full, start from current index (oldest)
            for i in range(self._size):
                yield self._data[(self._index + i) % self._size]


@dataclass(slots=True)
class MetricAccumulator:
    """Efficient metric accumulation with running statistics.

    Computes mean, variance, min, max in O(1) per update using
    Welford's online algorithm.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, value: float) -> None:
        """Update running statistics with new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    @property
    def variance(self) -> float:
        """Population variance."""
        return self.m2 / self.count if self.count > 0 else 0.0

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return self.variance ** 0.5


# =============================================================================
# ASYNC UTILITIES
# =============================================================================


async def gather_with_semaphore(
    *coros,
    max_concurrent: int = 10,
    return_exceptions: bool = True
) -> List[Any]:
    """Run coroutines with concurrency limit.

    Prevents overwhelming external APIs or databases.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[limited(c) for c in coros],
        return_exceptions=return_exceptions
    )


def run_sync(coro) -> Any:
    """Run async coroutine synchronously.

    Safe to call from sync code, handles nested event loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)

    # Running loop exists, use nest_asyncio pattern
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================


@dataclass(slots=True)
class ExecutionTimer:
    """Lightweight execution timer with minimal overhead."""

    start_time: float = 0.0
    end_time: float = 0.0

    def start(self) -> "ExecutionTimer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        end = self.end_time if self.end_time else time.perf_counter()
        return (end - self.start_time) * 1000


def timed(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to track function execution time.

    Adds _last_execution_ms attribute to function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> R:
        timer = ExecutionTimer().start()
        try:
            return func(*args, **kwargs)
        finally:
            wrapper._last_execution_ms = timer.stop()

    wrapper._last_execution_ms = 0.0
    return wrapper


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def batch_process(
    items: List[T],
    processor: Callable[[List[T]], List[R]],
    batch_size: int = 100
) -> List[R]:
    """Process items in batches for memory efficiency.

    Useful for large data processing tasks.
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results.extend(processor(batch))
    return results


async def async_batch_process(
    items: List[T],
    processor: Callable[[List[T]], Any],
    batch_size: int = 100,
    max_concurrent_batches: int = 5
) -> List[Any]:
    """Async batch processing with concurrency control."""
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    async def process_batch(batch):
        if asyncio.iscoroutinefunction(processor):
            return await processor(batch)
        return processor(batch)

    return await gather_with_semaphore(
        *[process_batch(b) for b in batches],
        max_concurrent=max_concurrent_batches
    )


# =============================================================================
# LAZY LOADING
# =============================================================================


class LazyProperty:
    """Descriptor for lazy-loaded properties with caching.

    Usage:
        class MyClass:
            @LazyProperty
            def expensive_attr(self):
                return compute_expensive_value()
    """

    def __init__(self, func: Callable):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.func(obj))
        return getattr(obj, self.attr_name)

    def __set__(self, obj, value):
        setattr(obj, self.attr_name, value)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Caching
    "TTLCache",
    "ttl_cache",
    "memoize",
    "memoize_method",
    # Handler dispatch
    "HandlerRegistry",
    "TaskHandler",
    # Data structures
    "SlidingWindow",
    "MetricAccumulator",
    # Async utilities
    "gather_with_semaphore",
    "run_sync",
    # Performance tracking
    "ExecutionTimer",
    "timed",
    # Batch processing
    "batch_process",
    "async_batch_process",
    # Lazy loading
    "LazyProperty",
]

