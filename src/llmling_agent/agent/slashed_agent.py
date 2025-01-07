from __future__ import annotations

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, overload

from slashed import (
    CommandStore,
    DefaultOutputWriter,
    ExitCommandError,
    OutputWriter,
)
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.result import StreamedRunResult

    from llmling_agent.agent import AnyAgent
    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.agent.providers.base import AgentProvider
    from llmling_agent.common_types import ModelType
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.context import AgentContext
    from llmling_agent.prompts.convert import AnyPromptType
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)

TContext = TypeVar("TContext")
TDeps = TypeVar("TDeps")
TResult = TypeVar("TResult", default=str)


class SlashedAgent[TDeps, TContext]:
    """Wraps an agent with slash command support."""

    def __init__(
        self,
        agent: AnyAgent[TDeps, Any],
        *,
        command_context: TContext | None = None,
        command_history_path: str | None = None,
        output: DefaultOutputWriter | None = None,
    ):
        self.agent = agent
        assert self.agent.context, "Agent must have a context!"
        assert self.agent.context.pool, "Agent must have a pool!"

        self.commands = CommandStore(
            history_file=command_history_path,
            enable_system_commands=True,
        )
        self.command_context: TContext = command_context or self  # type: ignore
        self.output = output or DefaultOutputWriter()

    @overload
    async def run[TMethodResult](
        self,
        *prompt: AnyPromptType,
        result_type: type[TMethodResult],
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[TMethodResult]: ...

    @overload
    async def run(
        self,
        *prompt: AnyPromptType,
        result_type: None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]: ...

    async def run(
        self,
        *prompt: AnyPromptType,
        result_type: type[Any] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[Any]:
        """Run with slash command support."""
        # First execute all commands sequentially
        remaining_prompts = []
        for p in prompt:
            if isinstance(p, str) and p.startswith("/"):
                await self.handle_command(
                    p[1:],
                    output=output or self.output,
                    metadata=metadata,
                )
            else:
                remaining_prompts.append(p)

        # Then pass remaining prompts to agent
        return await self.agent.run(
            *remaining_prompts, result_type=result_type, deps=deps, model=model
        )

    @overload
    def run_stream[TMethodResult](
        self,
        *prompt: AnyPromptType,
        result_type: type[TMethodResult],
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[
        StreamedRunResult[AgentContext[TDeps], TMethodResult]
    ]: ...

    @overload
    def run_stream(
        self,
        *prompt: AnyPromptType,
        result_type: None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentContext[TDeps], str]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        *prompt: AnyPromptType,
        result_type: type[Any] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], Any]]:
        """Stream responses with slash command support."""
        # First execute all commands sequentially
        remaining_prompts: list[AnyPromptType] = []
        for p in prompt:
            if isinstance(p, str) and p.startswith("/"):
                await self.handle_command(
                    p[1:],
                    output=output or self.output,
                    metadata=metadata,
                )
            else:
                remaining_prompts.append(p)

        # Then yield from agent's stream
        async with self.agent.run_stream(
            *remaining_prompts, result_type=result_type, deps=deps, model=model
        ) as stream:
            yield stream

    async def handle_command(
        self,
        command: str,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]:
        """Handle a slash command."""
        try:
            await self.commands.execute_command_with_context(
                command,
                context=self.command_context,
                output_writer=output or self.output,
                metadata=metadata,
            )
            return ChatMessage(content="", role="system")
        except ExitCommandError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"Command error: {e}"
            return ChatMessage(content=msg, role="system")

    @property
    def tools(self) -> ToolManager:
        """Access to tool management."""
        return self.agent.tools

    @property
    def conversation(self) -> ConversationManager:
        """Access to conversation management."""
        return self.agent.conversation

    @property
    def provider(self) -> AgentProvider[TDeps]:
        """Access to the underlying provider."""
        return self.agent._provider

    @property
    def pool(self) -> AgentPool:
        """Get agent's pool from context."""
        assert self.agent.context.pool
        return self.agent.context.pool

    @property
    def model_name(self) -> str | None:
        """Get current model name."""
        return self.agent.model_name

    @property
    def context(self) -> AgentContext[TDeps]:
        """Access to agent context."""
        return self.agent.context
