from __future__ import annotations

import asyncio
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from toprompt import to_prompt
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

    from llmling_agent.messaging.messagenode import MessageNode
    from llmling_agent.models.task import Job
    from llmling_agent.talk import Talk


TDeps = TypeVar("TDeps", default=None)


class Team[TDeps](BaseTeam[TDeps, Any]):
    """Group of agents that can execute together."""

    async def execute(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Run all agents in parallel with monitoring."""
        from llmling_agent.talk.talk import Talk

        self._team_talk.clear()

        start_time = datetime.now()
        responses: list[AgentResponse[Any]] = []
        errors: dict[str, Exception] = {}
        final_prompt = list(prompts)
        if self.shared_prompt:
            final_prompt.insert(0, self.shared_prompt)
        combined_prompt = "\n".join([await to_prompt(p) for p in final_prompt])
        all_nodes = list(await self.pick_agents(combined_prompt))
        # Create Talk connections for monitoring this execution
        execution_talks: list[Talk[Any]] = []
        for node in all_nodes:
            talk = Talk[Any](
                node,
                [],  # No actual forwarding, just for tracking
                connection_type="run",
                queued=True,
                queue_strategy="latest",
            )
            execution_talks.append(talk)
            self._team_talk.append(talk)  # Add to base class's TeamTalk

        async def _run(node: MessageNode[TDeps, Any]) -> None:
            try:
                start = perf_counter()
                message = await node.run(*final_prompt, **kwargs)
                timing = perf_counter() - start
                r = AgentResponse(agent_name=node.name, message=message, timing=timing)
                responses.append(r)

                # Update talk stats for this agent
                talk = next(t for t in execution_talks if t.source == node)
                talk._stats.messages.append(message)

            except Exception as e:  # noqa: BLE001
                errors[node.name] = e

        # Run all agents in parallel
        await asyncio.gather(*[_run(node) for node in all_nodes])

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

        final_prompt = list(prompts)
        combined_prompt = "\n".join([await to_prompt(p) for p in final_prompt])
        all_nodes = list(await self.pick_agents(combined_prompt))

        async def _run(node: MessageNode[TDeps, Any]) -> None:
            try:
                message = await node.run(*prompts, **kwargs)
                await queue.put(message)
            except Exception as e:  # noqa: BLE001
                errors[node.name] = e

        # Start all agents
        tasks = [asyncio.create_task(_run(n), name=f"run_{n.name}") for n in all_nodes]
        for _ in all_nodes:
            yield await queue.get()

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
        result: TeamResponse = await self.execute(*prompts, **kwargs)

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

    async def run_job[TJobResult](
        self,
        job: Job[TDeps, TJobResult],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> list[AgentResponse[TJobResult]]:
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

        responses: list[AgentResponse[TJobResult]] = []
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
                tools = [t.name for t in job.get_tools()]
                await self.distribute(content="", tools=tools)

            prompt = await job.get_prompt()

            async def _run(agent: MessageNode[TDeps, TJobResult]) -> None:
                assert isinstance(agent, Agent | StructuredAgent)
                try:
                    with agent.tools.temporary_tools(
                        job.get_tools(), exclusive=not include_agent_tools
                    ):
                        start = perf_counter()
                        resp = AgentResponse(
                            agent_name=agent.name,
                            message=await agent.run(prompt, store_history=store_history),  # pyright: ignore
                            timing=perf_counter() - start,
                        )
                        responses.append(resp)
                except Exception as e:  # noqa: BLE001
                    errors[agent.name] = e

            # Run job in parallel on all agents
            await asyncio.gather(*[_run(node) for node in self.agents])

            return TeamResponse(responses=responses, start_time=start_time, errors=errors)

        except Exception as e:
            msg = "Job execution failed"
            logger.exception(msg)
            raise JobError(msg) from e
