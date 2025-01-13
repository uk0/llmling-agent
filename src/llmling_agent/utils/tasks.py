from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Coroutine


T = TypeVar("T")
logger = get_logger(__name__)


class TaskManagerMixin:
    """Mixin for managing async tasks.

    Provides utilities for:
    - Creating and tracking tasks
    - Fire-and-forget task execution
    - Running coroutines in sync context
    - Cleanup of pending tasks
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    def create_task(
        self, coro: Coroutine[Any, Any, T], *, name: str | None = None
    ) -> asyncio.Task[T]:
        """Create and track a new task with logging.

        Args:
            coro: Coroutine to run
            name: Optional name for the task
        """
        task = asyncio.create_task(coro, name=name)
        logger.debug("Created task: %s", task.get_name())

        def _done_callback(t: asyncio.Task[T]) -> None:
            logger.debug("Task completed: %s", t.get_name())
            self._pending_tasks.discard(t)
            if t.exception():
                logger.error("Task failed with error: %s", t.exception())

        task.add_done_callback(_done_callback)
        self._pending_tasks.add(task)
        return task

    def fire_and_forget(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run coroutine without waiting for result."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
        except RuntimeError:
            # No running loop - use new loop
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()

    def run_task_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            return loop.run_until_complete(task)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    def run_background(
        self,
        coro: Coroutine[Any, Any, Any],
        name: str | None = None,
    ) -> None:
        """Run a coroutine in the background and track it."""
        try:
            self.create_task(coro, name=name)

        except RuntimeError:
            # No running loop - use fire_and_forget
            self.fire_and_forget(coro)

    async def cleanup_tasks(self) -> None:
        """Wait for all pending tasks to complete."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

    async def complete_tasks(self, cancel: bool = False):
        """Wait for all pending tasks to complete."""
        if cancel:
            for task in self._pending_tasks:
                task.cancel()
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks)
