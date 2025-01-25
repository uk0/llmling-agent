"""Team execution management and monitoring."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.log import get_logger
from llmling_agent.models.messages import AgentResponse, ChatMessage, TeamResponse
from llmling_agent.talk.talk import Talk, TeamTalk
from llmling_agent_providers.base import StreamingResponseProtocol


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    import os

    import PIL.Image
    from tokonomics.pydanticai_cost import Usage
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent


logger = get_logger(__name__)

ExecutionMode = Literal["parallel", "sequential"]
"""The execution mode for a TeamRun."""


@dataclass(frozen=True, kw_only=True)
class ExtendedTeamTalk(TeamTalk):
    """TeamTalk that also provides TeamRunStats interface."""

    errors: list[tuple[str, str, datetime]] = field(default_factory=list)

    def clear(self):
        """Reset all tracking data."""
        super().clear()  # Clear base TeamTalk
        self.errors.clear()

    def add_error(self, agent: str, error: str):
        """Track errors from AgentResponses."""
        self.errors.append((agent, error, datetime.now()))

    @property
    def error_log(self) -> list[tuple[str, str, datetime]]:
        """Errors from failed responses."""
        return self.errors


class TeamRun[TDeps, TResult](BaseTeam[TDeps, TResult]):
    """Handles team operations with monitoring."""

    def __init__(
        self,
        agents: Sequence[AnyAgent[TDeps, Any] | BaseTeam[TDeps, Any]],
        *,
        name: str | None = None,
        shared_prompt: str | None = None,
        validator: AnyAgent[Any, TResult] | None = None,
    ):
        super().__init__(agents, name=name, shared_prompt=shared_prompt)
        self.validator = validator

    async def _run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        wait_for_connections: bool | None = None,
        **kwargs: Any,
    ) -> ChatMessage[Any]:
        """Run agents sequentially and return combined message.

        This message wraps execute and extracts the ChatMessage in order to fulfill
        the "message protocol".
        """
        result = await self.execute(*prompts, **kwargs)

        return ChatMessage(
            content=[r.message.content for r in result if r.message],
            role="assistant",
            name=self.name,
            metadata={
                "agent_names": [r.agent_name for r in result],
                "errors": {name: str(error) for name, error in result.errors.items()},
                "start_time": result.start_time.isoformat(),
                "execution_order": [r.agent_name for r in result],
            },
        )

    async def execute(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> TeamResponse[TResult]:
        """Start execution with optional monitoring."""
        self._team_talk.clear()
        start_time = datetime.now()
        final_prompt = list(prompts)
        if self.shared_prompt:
            final_prompt.insert(0, self.shared_prompt)

        responses = [
            i
            async for i in self.execute_iter(*final_prompt)
            if isinstance(i, AgentResponse)
        ]
        return TeamResponse(responses, start_time)

    async def run_iter(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages from the execution chain."""
        async for item in self.execute_iter(*prompts, **kwargs):
            match item:
                case AgentResponse():
                    if item.message:
                        yield item.message
                case Talk():
                    pass

    async def execute_iter(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        **kwargs: Any,
    ) -> AsyncIterator[Talk[Any] | AgentResponse[Any]]:
        connections: list[Talk[Any]] = []
        try:
            first = self.agents[0]
            connections = [
                source.connect_to(target, queued=True)  # pyright: ignore
                for source, target in pairwise(self.agents)
            ]
            for conn in connections:
                self._team_talk.append(conn)

            # First agent
            start = perf_counter()
            message = await first.run(*prompt, **kwargs)
            timing = perf_counter() - start
            response = AgentResponse[Any](
                agent_name=first.name, message=message, timing=timing
            )
            yield response

            # Process through chain
            for connection in connections:
                target = connection.targets[0]
                target_name = target.name
                yield connection

                # Let errors propagate - they break the chain
                start = perf_counter()
                messages = await connection.trigger()

                # If this is the last agent
                if target == self.agents[-1]:
                    last_talk = Talk[Any](target, [], connection_type="run")
                    if response.message:
                        last_talk.stats.messages.append(response.message)
                    self._team_talk.append(last_talk)

                timing = perf_counter() - start
                response = AgentResponse[Any](
                    agent_name=target_name, message=messages[0], timing=timing
                )
                yield response

        finally:
            # Always clean up connections
            for connection in connections:
                connection.disconnect()

    async def chain(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        require_all: bool = True,
        **kwargs: Any,
    ) -> ChatMessage:
        """Pass message through the chain of team members.

        Each agent processes the result of the previous one.

        Args:
            prompts: Initial messages to process
            require_all: If True, all agents must succeed
            kwargs: Additional arguments for agents

        Returns:
            Final processed message

        Raises:
            ValueError: If chain breaks and require_all=True
        """
        current_message = prompts

        for agent in self.agents:
            try:
                result = await agent.run(*current_message, **kwargs)
                current_message = (result.content,)
            except Exception as e:
                if require_all:
                    msg = f"Chain broken at {agent.name}: {e}"
                    raise ValueError(msg) from e
                logger.warning("Chain handler %s failed: %s", agent.name, e)

        return result

    @asynccontextmanager
    async def chain_stream(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        require_all: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol]:
        """Stream results through chain of team members."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation import TeamRun

        async with AsyncExitStack() as stack:
            streams: list[StreamingResponseProtocol[str]] = []
            current_message = prompts

            # Set up all streams
            for agent in self.agents:
                try:
                    assert isinstance(agent, TeamRun | Agent | StructuredAgent), (
                        "Cannot stream teams!"
                    )
                    stream = await stack.enter_async_context(
                        agent.run_stream(*current_message, **kwargs)
                    )
                    streams.append(stream)  # type: ignore
                    # Wait for complete response for next agent
                    async for chunk in stream.stream():
                        current_message = chunk
                        if stream.is_complete:
                            current_message = (stream.formatted_content,)  # type: ignore
                            break
                except Exception as e:
                    if require_all:
                        msg = f"Chain broken at {agent.name}: {e}"
                        raise ValueError(msg) from e
                    logger.warning("Chain handler %s failed: %s", agent.name, e)

            # Create a stream-like interface for the chain
            class ChainStream(StreamingResponseProtocol[str]):
                def __init__(self):
                    self.streams = streams
                    self.current_stream_idx = 0
                    self.is_complete = False
                    self.model_name = None

                def usage(self) -> Usage:
                    @dataclass
                    class Usage:
                        total_tokens: int | None
                        request_tokens: int | None
                        response_tokens: int | None

                    return Usage(0, 0, 0)

                async def stream(self) -> AsyncIterator[str]:  # type: ignore
                    for idx, stream in enumerate(self.streams):
                        self.current_stream_idx = idx
                        async for chunk in stream.stream():
                            yield chunk
                            if idx == len(self.streams) - 1 and stream.is_complete:
                                self.is_complete = True

            yield ChainStream()

    @asynccontextmanager
    async def run_stream(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol[TResult]]:
        """Stream responses through the chain.

        Provides same interface as Agent.run_stream.
        """
        async with self.chain_stream(*prompts, **kwargs) as stream:
            yield stream


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
            run = agent1 | agent2 | agent3

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
