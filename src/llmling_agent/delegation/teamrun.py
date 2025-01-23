"""Team execution management and monitoring."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.log import get_logger
from llmling_agent.models.messages import AgentResponse, ChatMessage, TeamResponse
from llmling_agent.talk.talk import Talk, TeamTalk
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.agent.agent import Agent
    from llmling_agent.delegation import Team


logger = get_logger(__name__)

ExecutionMode = Literal["parallel", "sequential"]
"""The execution mode for a TeamRun."""


@dataclass(frozen=True, kw_only=True)
class ExtendedTeamTalk(TeamTalk):
    """TeamTalk that also provides TeamRunStats interface."""

    errors: list[tuple[str, str, datetime]] = field(default_factory=list)

    def add_error(self, agent: str, error: str):
        """Track errors from AgentResponses."""
        self.errors.append((agent, error, datetime.now()))

    @property
    def error_log(self) -> list[tuple[str, str, datetime]]:
        """Errors from failed responses."""
        return self.errors


class TeamRun[TDeps](TaskManagerMixin):
    """Handles team operations with monitoring."""

    def __init__(
        self,
        team: Team[TDeps],
        mode: ExecutionMode,
    ):
        super().__init__()
        self.team = team
        self.mode = mode
        self._team_talk = ExtendedTeamTalk()
        self._main_task: asyncio.Task[TeamResponse] | None = None

    def __or__(self, other: Agent | Callable | Team | TeamRun) -> TeamRun:
        from llmling_agent import Agent, Team
        from llmling_agent_providers.callback import CallbackProvider

        match other:
            case Agent():
                self.team.agents.append(other)
            case Callable():
                provider = CallbackProvider(other)
                position = len(self.team.agents) + 1
                name = f"{other.__name__}_{position}"
                new_agent = Agent(provider=provider, name=name)
                self.team.agents.append(new_agent)
            case Team():
                # Flatten team
                self.team.agents.extend(other.agents)
            case TeamRun():
                # Merge executions
                self.team.agents.extend(other.team.agents)
        return self

    async def run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Start execution with optional monitoring."""
        self._team_talk = ExtendedTeamTalk()
        try:
            match self.mode:
                case "parallel":
                    return await self._run_parallel(*prompts)
                case "sequential":
                    return await self._run_sequential(*prompts)
                case _:
                    msg = f"Invalid mode: {self.mode}"
                    raise ValueError(msg)
        finally:
            pass

    def run_in_background(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> ExtendedTeamTalk:
        if self._main_task:
            msg = "Execution already running"
            raise RuntimeError(msg)
        coro = self.run(*prompts, **kwargs)
        self._main_task = self.create_task(coro, name="main_execution")
        return self._team_talk

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

    async def cancel(self) -> None:
        """Cancel execution and cleanup."""
        if self._main_task:
            self._main_task.cancel()
        await self.cleanup_tasks()

    @property
    def stats(self) -> ExtendedTeamTalk:
        """Get current execution statistics."""
        return self._team_talk

    async def _run_parallel(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    ) -> TeamResponse:
        """Execute in parallel mode."""
        start_time = datetime.now()
        final_prompt = list(prompt)
        if self.team.shared_prompt:
            final_prompt.insert(0, self.team.shared_prompt)

        async def run_agent(agent: AnyAgent[TDeps, Any]) -> AgentResponse[Any]:
            try:
                start = perf_counter()
                message = await agent.run(*final_prompt)
                timing = perf_counter() - start
                return AgentResponse(agent.name, message=message, timing=timing)
            except Exception as e:  # noqa: BLE001
                msg = ChatMessage(content="", role="assistant")
                self._team_talk.add_error(agent.name, str(e))
                return AgentResponse(agent_name=agent.name, message=msg, error=str(e))

        responses = await asyncio.gather(*[run_agent(a) for a in self.team.agents])
        return TeamResponse(responses, start_time)

    async def run_iter(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    ) -> AsyncIterator[Talk[Any] | AgentResponse[Any]]:
        try:
            first = self.team[0]
            connections = [
                source.pass_results_to(target, queued=True)
                for source, target in pairwise(self.team)
            ]
            for conn in connections:
                self._team_talk.append(conn)

            start = perf_counter()
            message = await first.run(*prompt)
            timing = perf_counter() - start
            response = AgentResponse[Any](
                agent_name=first.name, message=message, timing=timing
            )
            yield response
            for connection in connections:
                target = connection.targets[0]
                target_name = target.name
                yield connection  # pyright: ignore
                try:
                    start = perf_counter()
                    messages = await connection.trigger()
                    # If this is the last agent
                    if target == self.team.agents[-1]:
                        # Create Talk for stats collection only
                        last_talk = Talk[Any](target, [], connection_type="run")
                        # Add its message to the Talk's stats
                        if response.message:
                            last_talk.stats.messages.append(response.message)
                        # Add Talk to TeamTalk
                        self._team_talk.append(last_talk)
                    timing = perf_counter() - start
                    response = AgentResponse[Any](
                        agent_name=target_name, message=messages[0], timing=timing
                    )
                    yield response

                except Exception as e:  # noqa: BLE001
                    self._team_talk.add_error(connection.targets[0].name, str(e))
                    msg = ChatMessage(content="", role="assistant")
                    response = AgentResponse[Any](
                        agent_name=target_name, message=msg, error=str(e)
                    )
                    yield response
        finally:
            for connection in connections:
                connection.disconnect()

    async def _run_sequential(
        self, *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str]
    ) -> TeamResponse:
        """Execute in sequential mode."""
        start_time = datetime.now()
        final_prompt = list(prompt)
        if self.team.shared_prompt:
            final_prompt.insert(0, self.team.shared_prompt)

        responses = [
            i async for i in self.run_iter(*final_prompt) if isinstance(i, AgentResponse)
        ]
        return TeamResponse(responses, start_time)


if __name__ == "__main__":
    import asyncio

    from llmling_agent import AgentPool

    async def main():
        async with AgentPool[None]() as pool:
            # Create three agents with different roles
            agent1 = await pool.add_agent(
                "analyzer",
                system_prompt="You analyze text and find key points.",
                model="openai:gpt-4o-mini",
            )
            agent2 = await pool.add_agent(
                "summarizer",
                system_prompt="You create concise summaries.",
                model="openai:gpt-4o-mini",
            )
            agent3 = await pool.add_agent(
                "critic",
                system_prompt="You evaluate and critique summaries.",
                model="openai:gpt-4o-mini",
            )

            # Create team and get monitored execution
            team = agent1 & agent2 & agent3
            run = team.monitored(mode="sequential")

            text = "The quick brown fox jumps over the lazy dog."
            print(f"\nProcessing text: {text}\n")

            # Start run and get stats object (ExtendedTeamTalk)
            stats = run.run_in_background(text)

            # Poll stats while running
            while run.is_running:
                print("\nCurrent status:")
                print(f"Number of active connections: {len(stats)}")
                print("Errors:", len(stats.errors))
                await asyncio.sleep(0.5)

            # Wait for completion and get results
            result = await run.wait()
            print("\nFinal Results:")
            for resp in result:
                print(f"\n{resp.agent_name}:")
                print("-" * 40)
                print(resp.message)

    asyncio.run(main())
