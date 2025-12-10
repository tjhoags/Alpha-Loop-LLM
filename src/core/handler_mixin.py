"""================================================================================
HANDLER DISPATCH MIXIN - Optimized Task Routing for All Agents
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Provides a reusable mixin class for efficient handler dispatch pattern used
across all agents. Uses cached_property for O(1) handler lookup and provides
common utilities for task processing.

Usage:
    class MyAgent(BaseAgent, HandlerDispatchMixin):
        def _build_handlers(self) -> Dict[str, Callable]:
            return {
                'task_type_1': self._handle_task_1,
                'task_type_2': self._handle_task_2,
            }

        def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
            return self.dispatch(task, 'type', self._default_handler)

================================================================================
"""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

# Type definitions
T = TypeVar('T')
HandlerFunc = Callable[[Dict[str, Any]], Dict[str, Any]]
HandlerDict = Dict[str, HandlerFunc]


class HasLogger(Protocol):
    """Protocol for objects with a logger attribute."""
    logger: Any


class HandlerDispatchMixin:
    """Mixin providing efficient cached handler dispatch.

    This mixin implements the common pattern of routing tasks to handlers
    based on a task type field. It caches the handler dictionary to avoid
    repeated dictionary creation on each process() call.

    Benefits:
    - O(1) handler lookup via cached dict
    - Single point of handler registration
    - Type-safe with proper type hints
    - Reduced memory allocation per call

    Implementing classes must define:
    - _build_handlers(): Returns the handler dispatch dictionary
    - Optionally override _default_handler() for unknown task types
    """

    @abstractmethod
    def _build_handlers(self) -> HandlerDict:
        """Build and return the handler dispatch dictionary.

        Returns:
            Dict mapping task type strings to handler callables.

        Example:
            return {
                'analyze': self._handle_analyze,
                'generate': self._handle_generate,
            }
        """
        raise NotImplementedError

    @cached_property
    def _handlers(self) -> HandlerDict:
        """Cached handler dispatch table for O(1) lookup.

        Built once on first access and cached for the lifetime of the instance.
        """
        return self._build_handlers()

    def dispatch(
        self,
        task: Dict[str, Any],
        type_key: str = 'type',
        default_handler: Optional[HandlerFunc] = None,
    ) -> Dict[str, Any]:
        """Dispatch a task to the appropriate handler.

        Args:
            task: The task dictionary to process.
            type_key: The key in task dict containing the task type.
            default_handler: Handler to use if task type not found.
                           If None, uses self._default_handler.

        Returns:
            Result dictionary from the handler.
        """
        task_type = task.get(type_key, 'unknown')
        fallback = default_handler or self._default_handler
        handler = self._handlers.get(task_type, fallback)
        return handler(task)

    def _default_handler(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler for unknown task types.

        Override this method to customize behavior for unrecognized tasks.
        """
        return {
            'success': False,
            'error': f"Unknown task type: {task.get('type', 'unknown')}",
        }

    def has_handler(self, task_type: str) -> bool:
        """Check if a handler exists for the given task type."""
        return task_type in self._handlers

    def get_supported_types(self) -> tuple[str, ...]:
        """Get tuple of all supported task types."""
        return tuple(self._handlers.keys())


class BatchHandlerMixin(HandlerDispatchMixin):
    """Extended mixin supporting batch processing of tasks.

    Useful for agents that process multiple items in parallel.
    """

    def dispatch_batch(
        self,
        tasks: list[Dict[str, Any]],
        type_key: str = 'type',
        default_handler: Optional[HandlerFunc] = None,
    ) -> list[Dict[str, Any]]:
        """Dispatch multiple tasks efficiently.

        Args:
            tasks: List of task dictionaries to process.
            type_key: The key in task dict containing the task type.
            default_handler: Handler to use if task type not found.

        Returns:
            List of result dictionaries from handlers.
        """
        fallback = default_handler or self._default_handler
        results = []

        # Pre-fetch handlers for common case where many tasks have same type
        handlers_cache: Dict[str, HandlerFunc] = {}

        for task in tasks:
            task_type = task.get(type_key, 'unknown')

            if task_type not in handlers_cache:
                handlers_cache[task_type] = self._handlers.get(task_type, fallback)

            handler = handlers_cache[task_type]
            results.append(handler(task))

        return results

