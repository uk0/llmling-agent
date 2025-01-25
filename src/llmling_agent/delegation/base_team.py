from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from psygnal import Signal
from psygnal.containers import EventedList

from llmling_agent.agent.connection import ConnectionManager
from llmling_agent.models.messages import ChatMessage
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Iterator, Sequence
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.teamrun import ExtendedTeamTalk
    from llmling_agent.models.messages import TeamResponse


class BaseTeam[TDeps, TResult](TaskManagerMixin):
    """Base class for Team and TeamRun."""

    outbox = Signal(ChatMessage)

    def __init__(
        self,
        agents: Sequence[AnyAgent[TDeps, TResult]],
        *,
        name: str | None = None,
        shared_prompt: str | None = None,
    ):
        """Common variables only for typing."""
        from llmling_agent.delegation.teamrun import ExtendedTeamTalk

        super().__init__()
        self.agents = EventedList(list(agents))
        self.connections: ConnectionManager
        self._team_talk = ExtendedTeamTalk()
        self.shared_prompt = shared_prompt
        self.name = name or " & ".join([i.name for i in agents])
        self.connections = ConnectionManager(self)
        self._main_task: asyncio.Task[TeamResponse] | None = None

    def __repr__(self) -> str:
        """Create readable representation."""
        members = ", ".join(agent.name for agent in self.agents)
        name = f" ({self.name})" if self.name else ""
        return f"{self.__class__.__name__}[{len(self.agents)}]{name}: {members}"

    def __len__(self) -> int:
        """Get number of team members."""
        return len(self.agents)

    def __iter__(self) -> Iterator[AnyAgent[TDeps, TResult]]:
        """Iterate over team members."""
        return iter(self.agents)

    def __getitem__(self, index_or_name: int | str) -> AnyAgent[TDeps, TResult]:
        """Get team member by index or name."""
        if isinstance(index_or_name, str):
            return next(agent for agent in self.agents if agent.name == index_or_name)
        return self.agents[index_or_name]

    @property
    def is_running(self) -> bool:
        """Whether execution is currently running."""
        return bool(self._main_task and not self._main_task.done())

    async def wait(self) -> TeamResponse:
        if not self._main_task:
            msg = "No execution running"
            raise RuntimeError(msg)
        try:
            return await self._main_task
        finally:
            await self.cleanup_tasks()

    def run_in_background(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> ExtendedTeamTalk:
        if self._main_task:
            msg = "Execution already running"
            raise RuntimeError(msg)
        coro = self.execute(*prompts, **kwargs)
        self._main_task = self.create_task(coro, name="main_execution")
        return self._team_talk

    @property
    def stats(self) -> ExtendedTeamTalk:
        """Get current execution statistics."""
        return self._team_talk

    async def cancel(self) -> None:
        """Cancel execution and cleanup."""
        if self._main_task:
            self._main_task.cancel()
        await self.cleanup_tasks()

    async def distribute(
        self,
        content: str,
        *,
        tools: list[str] | None = None,
        resources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Distribute content and capabilities to all team members."""
        for agent in self.agents:
            # Add context message
            agent.conversation.add_context_message(
                content, source="distribution", metadata=metadata
            )

            # Register tools if provided
            if tools:
                for tool in tools:
                    agent.tools.register_tool(tool)

            # Load resources if provided
            if resources:
                for resource in resources:
                    await agent.conversation.load_context_source(resource)

    @abstractmethod
    async def execute(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
    ) -> TeamResponse: ...

    @abstractmethod
    async def run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
    ) -> ChatMessage: ...
