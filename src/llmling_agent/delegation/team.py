from __future__ import annotations

import asyncio
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.log import get_logger
from llmling_agent.models.messages import AgentResponse, ChatMessage, TeamResponse


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.models.providers import ProcessorCallback
    from llmling_agent.models.task import Job
    from llmling_agent.talk import Talk


TDeps = TypeVar("TDeps", default=None)


class Team[TDeps](BaseTeam[TDeps, Any]):
    """Group of agents that can execute together."""

    def __rshift__(
        self, other: AnyAgent[Any, Any] | Team[Any] | ProcessorCallback[Any]
    ) -> Talk:
        """Connect group to target agent(s).

        Returns:
            - list[Talk] when connecting to single agent (one Talk per source)
            - list[TeamTalk] when connecting to team (one TeamTalk per source)
        """
        return self.connect_to(other)

    async def execute(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Run all agents in parallel with monitoring."""
        from llmling_agent.talk.talk import Talk

        start_time = datetime.now()
        final_prompt = list(prompts)
        responses: list[AgentResponse[Any]] = []
        errors: dict[str, Exception] = {}

        if self.shared_prompt:
            final_prompt.insert(0, self.shared_prompt)

        # Create Talk connections for monitoring this execution
        execution_talks: list[Talk[Any]] = []
        for agent in self.agents:
            talk = Talk[Any](
                agent,
                [],  # No actual forwarding, just for tracking
                connection_type="run",
                queued=True,
                queue_strategy="latest",
            )
            execution_talks.append(talk)
            self._team_talk.append(talk)  # Add to base class's TeamTalk

        async def _run(agent: AnyAgent[TDeps, Any] | BaseTeam[TDeps, Any]) -> None:
            try:
                start = perf_counter()
                message = await agent.run(*final_prompt, **kwargs)
                timing = perf_counter() - start
                r = AgentResponse(agent_name=agent.name, message=message, timing=timing)
                responses.append(r)

                # Update talk stats for this agent
                talk = next(t for t in execution_talks if t.source == agent)
                talk._stats.messages.append(message)

            except Exception as e:  # noqa: BLE001
                errors[agent.name] = e

        # Run all agents in parallel
        await asyncio.gather(*[_run(agent) for agent in self.agents])

        return TeamResponse(responses=responses, start_time=start_time, errors=errors)

    async def run_iter(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages as they arrive from parallel execution."""
        # Create queue for collecting results
        queue: asyncio.Queue[ChatMessage[Any]] = asyncio.Queue()
        errors: dict[str, Exception] = {}

        async def _run(agent: AnyAgent[TDeps, Any] | BaseTeam[TDeps, Any]) -> None:
            try:
                message = await agent.run(*prompts, **kwargs)
                await queue.put(message)
            except Exception as e:  # noqa: BLE001
                errors[agent.name] = e

        # Start all agents
        tasks = [
            asyncio.create_task(_run(agent), name=f"run_{agent.name}")
            for agent in self.agents
        ]

        # Yield messages as they arrive
        completed = 0
        while completed < len(self.agents):
            message = await queue.get()
            yield message
            completed += 1

        # Wait for all tasks to complete (for error handling)
        await asyncio.gather(*tasks, return_exceptions=True)

        if errors:
            # Maybe raise an exception with all errors?
            first_error = next(iter(errors.values()))
            raise first_error

    async def _run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        wait_for_connections: bool | None = None,
        **kwargs: Any,
    ) -> ChatMessage[list[Any]]:
        """Run all agents in parallel and return combined message."""
        result = await self.execute(*prompts, **kwargs)

        return ChatMessage(
            content=[r.message.content for r in result if r.message],
            role="assistant",
            name=self.name,
            metadata={
                "agent_names": [r.agent_name for r in result],
                "errors": {name: str(error) for name, error in result.errors.items()},
                "start_time": result.start_time.isoformat(),
            },
        )

    async def run_job[TResult](
        self,
        job: Job[TDeps, TResult],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> list[AgentResponse[TResult]]:
        """Execute a job across all team members in parallel.

        Args:
            job: Job configuration to execute
            store_history: Whether to add job execution to conversation history
            include_agent_tools: Whether to include agent's tools alongside job tools

        Returns:
            List of responses from all agents

        Raises:
            JobError: If job execution fails for any agent
            ValueError: If job configuration is invalid
        """
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.tasks import JobError

        responses: list[AgentResponse[TResult]] = []
        errors: dict[str, Exception] = {}
        start_time = datetime.now()

        # Validate dependencies for all agents
        if job.required_dependency is not None:
            invalid_agents = [
                agent.name
                for agent in self.iter_agents()
                if not isinstance(agent.context.data, job.required_dependency)
            ]
            if invalid_agents:
                msg = (
                    f"Agents {', '.join(invalid_agents)} don't have required "
                    f"dependency type: {job.required_dependency}"
                )
                raise JobError(msg)

        try:
            # Load knowledge for all agents if provided
            if job.knowledge:
                # TODO: resources
                await self.distribute(
                    content="",  # Knowledge loaded through resources
                    tools=[t.name for t in job.get_tools()],
                    # resources=job.knowledge.get_resources(),
                )

            # Get prompt
            prompt = await job.get_prompt()

            # Run job in parallel on all agents
            async def _run(
                agent: AnyAgent[TDeps, TResult] | BaseTeam[TDeps, TResult],
            ) -> None:
                assert isinstance(agent, Agent | StructuredAgent)
                try:
                    with agent.tools.temporary_tools(
                        job.get_tools(), exclusive=not include_agent_tools
                    ):
                        start = perf_counter()
                        resp = AgentResponse(
                            agent_name=agent.name,
                            message=await agent.run(prompt, store_history=store_history),
                            timing=perf_counter() - start,
                        )
                        responses.append(resp)
                except Exception as e:  # noqa: BLE001
                    errors[agent.name] = e

            await asyncio.gather(*[_run(agent) for agent in self.agents])

            return TeamResponse(responses=responses, start_time=start_time, errors=errors)

        except Exception as e:
            msg = "Job execution failed"
            logger.exception(msg)
            raise JobError(msg) from e
