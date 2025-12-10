"""================================================================================
ASYNC UTILITIES - High-Performance Async Operations for Agents
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Async utilities for I/O-bound operations:
- Async data loading from databases
- Concurrent API calls
- Batch async processing
- Connection pooling

PHILOSOPHY: Never block. Never wait. Always compute.
================================================================================
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


# =============================================================================
# ASYNC EXECUTION UTILITIES
# =============================================================================


async def gather_with_timeout(
    *coros,
    timeout: float = 30.0,
    return_exceptions: bool = True
) -> List[Any]:
    """Gather coroutines with a timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=return_exceptions),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return [TimeoutError(f"Operation timed out after {timeout}s")] * len(coros)


async def run_with_retry(
    coro_func: Callable[[], Any],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """Run a coroutine with exponential backoff retry."""
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception


async def map_async(
    func: Callable[[T], Any],
    items: List[T],
    max_concurrent: int = 10
) -> List[Any]:
    """Map a function over items with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_func(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            return func(item)

    return await asyncio.gather(*[limited_func(item) for item in items])


# =============================================================================
# ASYNC DATA LOADER
# =============================================================================


@dataclass
class AsyncDataResult:
    """Result from async data loading."""
    data: Any
    success: bool
    duration_ms: float
    error: Optional[str] = None
    source: str = "unknown"


class AsyncDataLoader:
    """Async data loader with connection pooling and caching.

    Usage:
        loader = AsyncDataLoader(max_connections=10)

        # Load single item
        result = await loader.load("price_data", {"symbol": "AAPL"})

        # Load batch
        results = await loader.load_batch([
            ("price_data", {"symbol": "AAPL"}),
            ("price_data", {"symbol": "MSFT"}),
        ])
    """

    def __init__(
        self,
        max_connections: int = 10,
        default_timeout: float = 30.0
    ):
        self._semaphore = asyncio.Semaphore(max_connections)
        self._default_timeout = default_timeout
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time_ms": 0.0,
        }

    async def load(
        self,
        loader_type: str,
        params: Dict[str, Any],
        timeout: float = None
    ) -> AsyncDataResult:
        """Load data asynchronously."""
        timeout = timeout or self._default_timeout
        start_time = time.perf_counter()

        async with self._semaphore:
            self._stats["total_requests"] += 1

            try:
                # Route to appropriate loader
                if loader_type == "price_data":
                    data = await self._load_price_data(params)
                elif loader_type == "market_data":
                    data = await self._load_market_data(params)
                elif loader_type == "fundamentals":
                    data = await self._load_fundamentals(params)
                else:
                    data = await self._load_generic(loader_type, params)

                duration_ms = (time.perf_counter() - start_time) * 1000
                self._stats["successful_requests"] += 1
                self._stats["total_time_ms"] += duration_ms

                return AsyncDataResult(
                    data=data,
                    success=True,
                    duration_ms=duration_ms,
                    source=loader_type
                )

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._stats["failed_requests"] += 1
                self._stats["total_time_ms"] += duration_ms

                return AsyncDataResult(
                    data=None,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e),
                    source=loader_type
                )

    async def load_batch(
        self,
        requests: List[tuple],
        max_concurrent: int = None
    ) -> List[AsyncDataResult]:
        """Load multiple data items concurrently.

        Args:
            requests: List of (loader_type, params) tuples
            max_concurrent: Override default concurrency limit
        """
        if max_concurrent:
            original_semaphore = self._semaphore
            self._semaphore = asyncio.Semaphore(max_concurrent)

        try:
            results = await asyncio.gather(*[
                self.load(loader_type, params)
                for loader_type, params in requests
            ])
            return list(results)
        finally:
            if max_concurrent:
                self._semaphore = original_semaphore

    async def _load_price_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load price data - placeholder for actual implementation."""
        symbol = params.get("symbol", "UNKNOWN")
        # Simulate async I/O
        await asyncio.sleep(0.01)
        return {
            "symbol": symbol,
            "price": 100.0,
            "timestamp": datetime.now().isoformat()
        }

    async def _load_market_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load market data - placeholder."""
        await asyncio.sleep(0.01)
        return {
            "vix": 20.0,
            "spy_price": 450.0,
            "timestamp": datetime.now().isoformat()
        }

    async def _load_fundamentals(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load fundamental data - placeholder."""
        symbol = params.get("symbol", "UNKNOWN")
        await asyncio.sleep(0.01)
        return {
            "symbol": symbol,
            "pe_ratio": 25.0,
            "market_cap": 1e11,
            "timestamp": datetime.now().isoformat()
        }

    async def _load_generic(
        self,
        loader_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic loader - placeholder."""
        await asyncio.sleep(0.01)
        return {
            "type": loader_type,
            "params": params,
            "timestamp": datetime.now().isoformat()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        avg_time = (
            self._stats["total_time_ms"] / self._stats["total_requests"]
            if self._stats["total_requests"] > 0 else 0
        )
        success_rate = (
            self._stats["successful_requests"] / self._stats["total_requests"]
            if self._stats["total_requests"] > 0 else 0
        )
        return {
            **self._stats,
            "avg_time_ms": round(avg_time, 2),
            "success_rate": round(success_rate, 4)
        }


# =============================================================================
# ASYNC TASK QUEUE
# =============================================================================


class AsyncTaskQueue:
    """Priority queue for async task processing.

    Usage:
        queue = AsyncTaskQueue()

        # Add tasks
        await queue.put({"type": "analyze", "ticker": "AAPL"}, priority=1)
        await queue.put({"type": "scan"}, priority=5)

        # Process tasks
        async for task in queue:
            result = await process_task(task)
    """

    def __init__(self, max_size: int = 1000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._counter = 0
        self._running = True

    async def put(
        self,
        task: Dict[str, Any],
        priority: int = 5
    ) -> None:
        """Add task to queue with priority (lower = higher priority)."""
        self._counter += 1
        await self._queue.put((priority, self._counter, task))

    async def get(self) -> Dict[str, Any]:
        """Get highest priority task."""
        _, _, task = await self._queue.get()
        return task

    def __aiter__(self):
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if not self._running and self._queue.empty():
            raise StopAsyncIteration
        return await self.get()

    def stop(self) -> None:
        """Stop the queue."""
        self._running = False

    @property
    def size(self) -> int:
        return self._queue.qsize()


# =============================================================================
# ASYNC CONTEXT MANAGERS
# =============================================================================


@asynccontextmanager
async def async_timer(name: str = "operation"):
    """Context manager for timing async operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * 1000
        # Could log this if needed


@asynccontextmanager
async def async_semaphore_context(
    semaphore: asyncio.Semaphore,
    timeout: float = None
):
    """Context manager for semaphore with optional timeout."""
    if timeout:
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Semaphore acquisition timed out after {timeout}s")
    else:
        await semaphore.acquire()

    try:
        yield
    finally:
        semaphore.release()


# =============================================================================
# SYNC WRAPPER
# =============================================================================


def run_async(coro) -> Any:
    """Run an async function synchronously.

    Safe to call from synchronous code.
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop, we need to run in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro)


def async_to_sync(func: Callable) -> Callable:
    """Decorator to make async function callable from sync code."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return run_async(func(*args, **kwargs))
    return wrapper


# =============================================================================
# SINGLETON DATA LOADER
# =============================================================================


_data_loader: Optional[AsyncDataLoader] = None


def get_async_loader() -> AsyncDataLoader:
    """Get singleton async data loader."""
    global _data_loader
    if _data_loader is None:
        _data_loader = AsyncDataLoader()
    return _data_loader


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Execution utilities
    "gather_with_timeout",
    "run_with_retry",
    "map_async",
    # Data loading
    "AsyncDataResult",
    "AsyncDataLoader",
    "get_async_loader",
    # Task queue
    "AsyncTaskQueue",
    # Context managers
    "async_timer",
    "async_semaphore_context",
    # Sync wrappers
    "run_async",
    "async_to_sync",
]

