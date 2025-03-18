"""Sequential, ordered group of agents / nodes."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import AgentResponse, ChatMessage, TeamResponse
from llmling_agent.talk.talk import Talk, TeamTalk
from llmling_agent.utils.now import get_now


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence
    from datetime import datetime
    import os

    import PIL.Image
    from tokonomics.pydanticai_cost import Usage
    from toprompt import AnyPromptType

    from llmling_agent import MessageNode
    from llmling_agent.agent import AnyAgent
    from llmling_agent_providers.base import StreamingResponseProtocol


logger = get_logger(__name__)

ResultMode = Literal["last", "concat"]


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
        self.errors.append((agent, error, get_now()))

    @property
    def error_log(self) -> list[tuple[str, str, datetime]]:
        """Errors from failed responses."""
        return self.errors


class TeamRun[TDeps, TResult](BaseTeam[TDeps, TResult]):
    """Handles team operations with monitoring."""

    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        shared_prompt: str | None = None,
        validator: MessageNode[Any, TResult] | None = None,
        picker: AnyAgent[Any, Any] | None = None,
        num_picks: int | None = None,
        pick_prompt: str | None = None,
        # result_mode: ResultMode = "last",
    ):
        super().__init__(
            agents,
            name=name,
            description=description,
            shared_prompt=shared_prompt,
            picker=picker,
            num_picks=num_picks,
            pick_prompt=pick_prompt,
        )
        self.validator = validator
        self.result_mode = "last"

    def __prompt__(self) -> str:
        """Format team info for prompts."""
        members = " -> ".join(a.name for a in self.agents)
        desc = f" - {self.description}" if self.description else ""
        return f"Sequential Team '{self.name}'{desc}\nPipeline: {members}"

    async def _run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        wait_for_connections: bool | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Run agents sequentially and return combined message.

        This message wraps execute and extracts the ChatMessage in order to fulfill
        the "message protocol".
        """
        message_id = message_id or str(uuid4())

        result = await self.execute(*prompts, **kwargs)
        all_messages = [r.message for r in result if r.message]
        assert all_messages, "Error during execution, returned None for TeamRun"
        # Determine content based on mode
        match self.result_mode:
            case "last":
                content = all_messages[-1].content
            # case "concat":
            #     content = "\n".join(msg.format() for msg in all_messages)
            case _:
                msg = f"Invalid result mode: {self.result_mode}"
                raise ValueError(msg)

        return ChatMessage(
            content=content,
            role="assistant",
            name=self.name,
            associated_messages=all_messages,
            message_id=message_id,
            conversation_id=conversation_id,
            metadata={
                "execution_order": [r.agent_name for r in result],
                "start_time": result.start_time.isoformat(),
                "errors": {name: str(error) for name, error in result.errors.items()},
            },
        )

    async def execute(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        **kwargs: Any,
    ) -> TeamResponse[TResult]:
        """Start execution with optional monitoring."""
        self._team_talk.clear()
        start_time = get_now()
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
        from toprompt import to_prompt

        connections: list[Talk[Any]] = []
        try:
            combined_prompt = "\n".join([await to_prompt(p) for p in prompt])
            all_nodes = list(await self.pick_agents(combined_prompt))
            if self.validator:
                all_nodes.append(self.validator)
            first = all_nodes[0]
            connections = [
                source.connect_to(target, queued=True)
                for source, target in pairwise(all_nodes)
            ]
            for conn in connections:
                self._team_talk.append(conn)

            # First agent
            start = perf_counter()
            message = await first.run(*prompt, **kwargs)
            timing = perf_counter() - start
            response = AgentResponse[Any](first.name, message=message, timing=timing)
            yield response

            # Process through chain
            for connection in connections:
                target = connection.targets[0]
                target_name = target.name
                yield connection

                # Let errors propagate - they break the chain
                start = perf_counter()
                messages = await connection.trigger()

                # If this is the last node
                if target == all_nodes[-1]:
                    last_talk = Talk[Any](target, [], connection_type="run")
                    if response.message:
                        last_talk.stats.messages.append(response.message)
                    self._team_talk.append(last_talk)

                timing = perf_counter() - start
                msg = messages[0]
                response = AgentResponse[Any](target_name, message=msg, timing=timing)
                yield response

        finally:
            # Always clean up connections
            for connection in connections:
                connection.disconnect()

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
        from llmling_agent_providers.base import StreamingResponseProtocol

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

                async def stream(self) -> AsyncGenerator[str, None]:  # type: ignore
                    for idx, stream in enumerate(self.streams):
                        self.current_stream_idx = idx
                        async for chunk in stream.stream():
                            yield chunk
                            if idx == len(self.streams) - 1 and stream.is_complete:
                                self.is_complete = True

                async def stream_text(
                    self,
                    delta: bool = False,
                ) -> AsyncGenerator[str, None]:
                    for idx, stream in enumerate(self.streams):
                        self.current_stream_idx = idx
                        async for chunk in stream.stream_text(delta=delta):
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

            # Create team and get monitored execution
            run = agent1 | agent2

            text = "The quick brown fox jumps over the lazy dog."
            print(f"\nProcessing text: {text}\n")

            # Start run and get stats object (ExtendedTeamTalk)
            stats = await run.run_in_background(text)

            # Poll stats while running
            while run.is_running:
                print("\nCurrent status:")
                print(f"Number of active connections: {len(stats)}")
                print("Errors:", len(stats.errors))
                await asyncio.sleep(0.5)

            # Wait for completion and get results
            result = await run.wait()
            print("\nFinal Results:")
            print(result)

    asyncio.run(main())
