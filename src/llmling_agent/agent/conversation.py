"""Conversation management for LLMling agent."""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
import tempfile
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from llmling.prompts import BasePrompt, PromptMessage, StaticPrompt
from pydantic_ai.messages import ModelRequest, UserPromptPart
from upath import UPath

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent.models.sources import (
    ContextSource,
    FileContextSource,
    PromptContextSource,
    ResourceContextSource,
)
from llmling_agent.pydantic_ai_utils import convert_model_message, format_response


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from datetime import datetime
    import os

    from pydantic_ai.messages import ModelMessage

    from llmling_agent.agent.agent import LLMlingAgent

logger = get_logger(__name__)

OverrideMode = Literal["replace", "append"]
type PromptInput = str | BasePrompt


def _to_base_prompt(prompt: PromptInput) -> BasePrompt:
    """Convert input to BasePrompt instance."""
    if isinstance(prompt, str):
        msg = PromptMessage(role="system", content=prompt)
        return StaticPrompt(
            name="System prompt", description="System prompt", messages=[msg]
        )
    return prompt


class ConversationManager:
    """Manages conversation state and system prompts."""

    def __init__(
        self,
        agent: LLMlingAgent[Any, Any],
        session_id: str | UUID | None = None,
        initial_prompts: str | Sequence[str] | None = None,
        *,
        context_sources: Sequence[ContextSource | str] = (),
    ):
        """Initialize conversation manager.

        Args:
            agent: instance to manage
            session_id: Optional session ID to load and continue conversation
            initial_prompts: Initial system prompts that start each conversation
            context_sources: Optional paths to load as context
        """
        self._agent = agent
        self._initial_prompts: list[BasePrompt] = []
        self._current_history: list[ModelMessage] = []
        self._last_messages: list[ModelMessage] = []
        self._pending_messages: deque[ModelRequest] = deque()
        if session_id is not None:
            from llmling_agent.storage.models import Message

            # Use provided session ID and load its history
            self.id = str(session_id)
            messages = Message.to_pydantic_ai_messages(self.id)
            self._current_history = messages
        else:
            # Start new conversation with UUID
            self.id = str(uuid4())
            # Add initial prompts
            if not initial_prompts:
                return
            prompts_list = (
                [initial_prompts] if isinstance(initial_prompts, str) else initial_prompts
            )
            for prompt in prompts_list:
                obj = StaticPrompt(
                    name="Initial system prompt",
                    description="Initial system prompt",
                    messages=[PromptMessage(role="system", content=prompt)],
                )
                self._initial_prompts.append(obj)
        # Add context loading tasks to agent
        for source in context_sources:
            task = asyncio.create_task(self.load_context_source(source))
            self._agent._pending_tasks.add(task)
            task.add_done_callback(self._agent._pending_tasks.discard)

    def __repr__(self) -> str:
        return f"ConversationManager(id={self.id!r})"

    async def load_context_source(self, source: ContextSource | str) -> None:
        """Load context from a single source."""
        try:
            match source:
                case str():
                    await self.add_context_from_path(source)
                case FileContextSource():
                    await self.add_context_from_path(
                        source.path, convert_to_md=source.convert_to_md, **source.metadata
                    )
                case ResourceContextSource():
                    await self.add_context_from_resource(
                        source.name, **source.arguments | source.metadata
                    )
                case PromptContextSource():
                    await self.add_context_from_prompt(
                        source.name, source.arguments, **source.metadata
                    )
        except Exception:
            msg = "Failed to load context from %s"
            logger.exception(msg, "file" if isinstance(source, str) else source.type)

    async def load_history_from_database(
        self,
        session_id: str | None = None,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        roles: set[Literal["user", "assistant", "system"]] | None = None,
        limit: int | None = None,
    ) -> None:
        """Load and set conversation history from database.

        Args:
            session_id: ID of conversation to load
            since: Only include messages after this time
            until: Only include messages before this time
            roles: Only include messages with these roles
            limit: Maximum number of messages to return

        Example:
            # Load last hour of user/assistant messages
            await conversation.load_history_from_database(
                "conv-123",
                since=datetime.now() - timedelta(hours=1),
                roles={"user", "assistant"}
            )
        """
        from llmling_agent.storage.models import Message

        conversation_id = session_id if session_id is not None else self.id
        messages = Message.to_pydantic_ai_messages(
            conversation_id,
            since=since,
            until=until,
            roles=roles,
            limit=limit,
        )
        self.set_history(messages)
        if session_id is not None:
            self.id = session_id

    @asynccontextmanager
    async def temporary(
        self,
        *,
        sys_prompts: PromptInput | Sequence[PromptInput] | None = None,
        mode: OverrideMode = "append",
    ) -> AsyncIterator[None]:
        """Start temporary conversation with different system prompts."""
        # Store original state
        original_prompts = list(self._initial_prompts)
        original_system_prompts = (
            self._agent._pydantic_agent._system_prompts
        )  # Store pydantic-ai prompts
        original_history = self._current_history

        try:
            if sys_prompts is not None:
                # Convert to list of BasePrompt
                new_prompts: list[BasePrompt] = []
                if isinstance(sys_prompts, str | BasePrompt):
                    new_prompts = [_to_base_prompt(sys_prompts)]
                else:
                    new_prompts = [_to_base_prompt(prompt) for prompt in sys_prompts]

                self._initial_prompts = (
                    original_prompts + new_prompts if mode == "append" else new_prompts
                )

                # Update pydantic-ai's system prompts
                formatted_prompts = await self.get_all_prompts()
                self._agent._pydantic_agent._system_prompts = tuple(formatted_prompts)

            # Force new conversation
            self._current_history = []
            yield
        finally:
            # Restore complete original state
            self._initial_prompts = original_prompts
            self._agent._pydantic_agent._system_prompts = original_system_prompts
            self._current_history = original_history

    def add_prompt(self, prompt: PromptInput):
        """Add a system prompt.

        Args:
            prompt: String content or BasePrompt instance to add
        """
        self._initial_prompts.append(_to_base_prompt(prompt))

    async def get_all_prompts(self) -> list[str]:
        """Get all formatted system prompts in order."""
        result: list[str] = []

        for prompt in self._initial_prompts:
            try:
                messages = await prompt.format()
                result.extend(
                    msg.get_text_content() for msg in messages if msg.role == "system"
                )
            except Exception:
                logger.exception("Error formatting prompt")

        return result

    def get_history(self, include_pending: bool = True) -> list[ModelMessage]:
        """Get current conversation history.

        Args:
            include_pending: Whether to include pending messages in the history.
                             If True, pending messages are moved to main history.

        Returns:
            List of messages in chronological order
        """
        if include_pending and self._pending_messages:
            self._current_history.extend(self._pending_messages)
            self._pending_messages.clear()

        return self._current_history

    def get_pending_messages(self) -> list[ModelMessage]:
        """Get messages that will be included in next interaction."""
        return list(self._pending_messages)

    def clear_pending(self):
        """Clear pending messages without adding them to history."""
        self._pending_messages.clear()

    def set_history(self, history: list[ModelMessage]):
        """Update conversation history after run."""
        self._current_history = history

    def clear(self):
        """Clear conversation history and prompts."""
        self._initial_prompts.clear()
        self._current_history = []
        self._last_messages = []

    @property
    def last_run_messages(self) -> list[ChatMessage]:
        """Get messages from the last run converted to our format."""
        return [convert_model_message(msg) for msg in self._last_messages]

    async def add_context_message(
        self,
        content: str,
        source: str | None = None,
        **metadata: Any,
    ):
        """Add a context message.

        Args:
            content: Text content to add
            source: Description of content source
            **metadata: Additional metadata to include with the message
        """
        meta_str = ""
        if metadata:
            meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            meta_str = f"\nMetadata:\n{meta_str}\n"

        header = f"Content from {source}:" if source else "Additional context:"
        formatted = f"{header}{meta_str}\n{content}\n"
        message = ModelRequest(parts=[UserPromptPart(content=formatted)])
        self._pending_messages.append(message)
        # Emit as user message - will trigger logging through existing flow

        chat_message = ChatMessage[str](
            content=formatted,
            role="user",
            name="user",
            model=self._agent.model_name,
            metadata=metadata,
        )
        self._agent.message_received.emit(chat_message)

    async def add_context_from_path(
        self,
        path: str | os.PathLike[str],
        *,
        convert_to_md: bool = False,
        **metadata: Any,
    ):
        """Add file or URL content as context message.

        Args:
            path: Any UPath-supported path
            convert_to_md: Whether to convert content to markdown
            **metadata: Additional metadata to include with the message

        Raises:
            ValueError: If content cannot be loaded or converted
        """
        path_obj = UPath(path)

        if convert_to_md:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()

                # Direct handling for local paths and http(s) URLs
                if path_obj.protocol in ("", "file", "http", "https"):
                    result = md.convert(path_obj.path)
                else:
                    with tempfile.NamedTemporaryFile(suffix=path_obj.suffix) as tmp:
                        tmp.write(path_obj.read_bytes())
                        tmp.flush()
                        result = md.convert(tmp.name)

                content = result.text_content
                source = f"markdown:{path_obj.name}"

            except Exception as e:
                msg = f"Failed to convert {path_obj} to markdown: {e}"
                raise ValueError(msg) from e
        else:
            try:
                content = path_obj.read_text()
                source = f"{path_obj.protocol}:{path_obj.name}"
            except Exception as e:
                msg = f"Failed to read {path_obj}: {e}"
                raise ValueError(msg) from e

        await self.add_context_message(content, source=source, **metadata)

    async def add_context_from_resource(
        self,
        resource_name: str,
        **params: Any,
    ):
        """Add content from runtime resource as context message.

        Args:
            resource_name: Name of the resource to load
            **params: Parameters to pass to resource loader

        Raises:
            RuntimeError: If no runtime is available
            ValueError: If resource loading fails
        """
        if not self._agent.runtime:
            msg = "No runtime available to load resources"
            raise RuntimeError(msg)

        content = await self._agent.runtime.load_resource(resource_name, **params)
        await self.add_context_message(str(content), source=f"resource:{resource_name}")

    async def add_context_from_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
        **metadata: Any,
    ):
        """Add rendered prompt content as context message.

        Args:
            prompt_name: Name of the prompt to render
            arguments: Optional arguments for prompt formatting
            **metadata: Additional metadata to include with the message

        Raises:
            RuntimeError: If no runtime is available
            ValueError: If prompt loading or rendering fails
        """
        if not self._agent.runtime:
            msg = "No runtime available to load prompts"
            raise RuntimeError(msg)

        messages = await self._agent.runtime.render_prompt(prompt_name, arguments)
        content = "\n\n".join(msg.get_text_content() for msg in messages)

        await self.add_context_message(
            content,
            source=f"prompt:{prompt_name}",
            prompt_args=arguments,
            **metadata,
        )

    def get_history_tokens(self) -> int:
        """Get token count for current history."""
        import tiktoken

        encoding = tiktoken.encoding_for_model(self._agent.model_name or "gpt-3.5-turbo")
        return sum(
            len(encoding.encode(format_response(part)))
            for msg in self._current_history
            for part in msg.parts
        )

    def get_pending_tokens(self) -> int:
        """Get token count for pending messages."""
        import tiktoken

        encoding = tiktoken.encoding_for_model(self._agent.model_name or "gpt-3.5-turbo")
        return sum(
            len(encoding.encode(format_response(part)))
            for msg in self._pending_messages
            for part in msg.parts
        )


if __name__ == "__main__":
    from llmling_agent import LLMlingAgent

    async def main():
        async with LLMlingAgent[Any, Any].open() as agent:
            convo = ConversationManager(agent, session_id="test")
            await convo.add_context_from_path("E:/mcp_zed.yml")
            print(convo._current_history)

    import asyncio

    asyncio.run(main())
