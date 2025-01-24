from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, overload

from typing_extensions import TypeVar

from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.log import get_logger
from llmling_agent.models.messages import AgentResponse, ChatMessage, TeamResponse
from llmling_agent.utils.inspection import has_return_type


logger = get_logger(__name__)

if TYPE_CHECKING:
    from datetime import timedelta
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import AnyTransformFn, AsyncFilterFn
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.providers import ProcessorCallback
    from llmling_agent.talk import QueueStrategy, TeamTalk


TDeps = TypeVar("TDeps", default=None)


class Team[TDeps](BaseTeam[TDeps, Any]):
    """Group of agents that can execute together."""

    @overload
    def __and__(self, other: Team[None]) -> Team[None]: ...

    @overload
    def __and__(self, other: Team[TDeps]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: Team[Any]) -> Team[Any]: ...

    @overload
    def __and__(self, other: AnyAgent[TDeps, Any]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: AnyAgent[Any, Any]) -> Team[Any]: ...

    def __and__(self, other: Team[Any] | AnyAgent[Any, Any]) -> Team[Any]:
        """Combine teams, preserving type safety for same types."""
        # Combine agents

        match other:
            case Team():
                combined_agents = [*self.agents, *other.agents]
                combined_prompts = []
                if self.shared_prompt:
                    combined_prompts.append(self.shared_prompt)
                if other.shared_prompt:
                    combined_prompts.append(other.shared_prompt)
                shared = "\n".join(combined_prompts) if combined_prompts else None
                return Team(agents=combined_agents, shared_prompt=shared)
            case _:  # AnyAgent case
                # Keep same deps if types match
                agents = [*self.agents, other]
                return Team(agents=agents, shared_prompt=self.shared_prompt)

    def __or__(
        self, other: AnyAgent[Any, Any] | ProcessorCallback[Any] | Team[Any]
    ) -> Team[TDeps]:
        """Create a pipeline using | operator.

        Example:
            pipeline = team | transform | other_team  # Sequential processing
        """
        from llmling_agent.agent import Agent, StructuredAgent

        match other:
            case Team():
                # Create sequential execution with all agents
                execution = Team([*self.agents, *other.agents])
            case Callable():
                # Convert callable to agent and add to pipeline
                from llmling_agent import Agent
                from llmling_agent_providers.callback import CallbackProvider

                provider = CallbackProvider(other)
                new_agent = Agent(provider=provider, name=other.__name__)
                execution = Team([*self.agents, new_agent])
            case Agent() | StructuredAgent():  # Agent case
                execution = Team([*self.agents, other])
            case _:
                msg = f"Invalid pipeline element: {other}"
                raise ValueError(msg)

        # Setup connections for sequential processing
        for i in range(len(execution.agents) - 1):
            current = execution.agents[i]
            next_agent = execution.agents[i + 1]
            current.pass_results_to(next_agent)

        return execution

    def __rshift__(
        self, other: AnyAgent[Any, Any] | Team[Any] | ProcessorCallback[Any]
    ) -> TeamTalk:
        """Connect group to target agent(s).

        Returns:
            - list[Talk] when connecting to single agent (one Talk per source)
            - list[TeamTalk] when connecting to team (one TeamTalk per source)
        """
        return self.pass_results_to(other)

    def pass_results_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | ProcessorCallback[Any],
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk:
        """Forward results to another agent or all agents in a team."""
        from llmling_agent.agent import Agent, StructuredAgent

        if callable(other):
            if has_return_type(other, str):
                other = Agent.from_callback(other)
            else:
                other = StructuredAgent.from_callback(other)
        return self.connections.connect_group_to(
            other,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    async def run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
    ) -> TeamResponse:
        """Run all agents in parallel."""
        start_time = datetime.now()
        final_prompt = list(prompts)
        if self.shared_prompt:
            final_prompt.insert(0, self.shared_prompt)

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

        responses = await asyncio.gather(*[run_agent(a) for a in self.agents])
        return TeamResponse(responses, start_time)
