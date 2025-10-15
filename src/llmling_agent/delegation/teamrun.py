"""Sequential, ordered group of agents / nodes."""

from __future__ import annotations

import asyncio
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
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime
    import os

    import PIL.Image
    from pydantic_ai.messages import AgentStreamEvent
    from toprompt import AnyPromptType

    from llmling_agent import MessageNode
    from llmling_agent.agent import AnyAgent
    from llmling_agent.agent.agent import StreamCompleteEvent


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
            connections = [s.connect_to(t, queued=True) for s, t in pairwise(all_nodes)]
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

                if target == all_nodes[-1]:
                    last_talk = Talk[Any](target, [], connection_type="run")
                    if response.message:
                        last_talk.stats.messages.append(response.message)
                    self._team_talk.append(last_talk)

                timing = perf_counter() - start
                msg = messages[0]
                response = AgentResponse[Any](target_name, message=msg, timing=timing)
                yield response

        finally:  # Always clean up connections
            for connection in connections:
                connection.disconnect()

    async def run_stream(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        require_all: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[AnyAgent, AgentStreamEvent | StreamCompleteEvent]]:
        """Stream responses through the chain of team members.

        Args:
            prompts: Input prompts to process through the chain
            require_all: If True, fail if any agent fails. If False,
                         continue with remaining agents.
            kwargs: Additional arguments passed to each agent

        Yields:
            Tuples of (agent, event) where agent is the Agent instance
            and event is the streaming event.
        """
        from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.agent.agent import StreamCompleteEvent
        from llmling_agent.delegation import TeamRun

        current_message = prompts
        collected_content = []

        for agent in self.agents:
            try:
                assert isinstance(agent, TeamRun | Agent | StructuredAgent), (
                    "Cannot stream teams!"
                )

                agent_content = []
                async for event in agent.run_stream(*current_message, **kwargs):
                    match event:
                        case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                            agent_content.append(delta)
                            collected_content.append(delta)
                            yield (agent, event)  # Yield tuple with agent context
                        case StreamCompleteEvent(message=message):
                            # Use complete response as input for next agent
                            current_message = (message.content,)
                            yield (agent, event)  # Yield tuple with agent context
                        case _:
                            yield (agent, event)  # Yield tuple with agent context

            except Exception as e:
                if require_all:
                    msg = f"Chain broken at {agent.name}: {e}"
                    logger.exception(msg)
                    raise ValueError(msg) from e
                logger.warning("Chain handler %s failed: %s", agent.name, e)


if __name__ == "__main__":
    import asyncio

    from llmling_agent import AgentPool

    async def main():
        async with AgentPool[None]() as pool:
            # Create three agents with different roles
            agent1 = await pool.add_agent(
                "analyzer",
                system_prompt="You analyze text and find key points.",
                model="openai:gpt-5-nano",
            )
            agent2 = await pool.add_agent(
                "summarizer",
                system_prompt="You create concise summaries.",
                model="openai:gpt-5-nano",
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
