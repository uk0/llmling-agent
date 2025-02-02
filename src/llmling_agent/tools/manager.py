"""Tool management for LLMling agents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from llmling import BaseRegistry, LLMCallableTool, ToolError
from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.resource_providers.callable_provider import (
    CallableResourceProvider,
    ResourceCallable,
)
from llmling_agent.tools.base import ToolInfo


if TYPE_CHECKING:
    from collections.abc import Iterator

    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import AnyCallable, ToolSource, ToolType
    from llmling_agent.resource_providers.base import ResourceProvider


logger = get_logger(__name__)

MAX_LEN_DESCRIPTION = 1000


class ToolManager(BaseRegistry[str, ToolInfo]):
    """Manages tool registration, enabling/disabling and access.

    Inherits from BaseRegistry providing:
    - Dict-like access: manager["tool_name"] -> ToolInfo
    - Async startup/shutdown: await manager.startup()
    - Event observation: manager.add_observer(observer)
    - Registration: manager.register("tool_name", tool)
    - Listing: manager.list_items()
    - State check: manager.is_empty, manager.has_item()
    - Async iteration: async for name, tool in manager: ...
    """

    @dataclass(frozen=True)
    class ToolStateReset:
        """Emitted when tool states are reset."""

        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=datetime.now)

    tool_states_reset = Signal(ToolStateReset)

    def __init__(
        self,
        tools: Sequence[ToolInfo | ToolType | dict[str, Any]] | None = None,
    ):
        """Initialize tool manager.

        Args:
            tools: Initial tools to register
        """
        super().__init__()
        self._resource_providers: list[ResourceProvider] = []

        # Register initial tools
        for tool in tools or []:
            t = self._validate_item(tool)
            self.register(t.name, t)

    def __prompt__(self) -> str:
        enabled_tools = [t.name for t in self.values() if t.enabled]
        if not enabled_tools:
            return "No tools available"
        return f"Available tools: {', '.join(enabled_tools)}"

    def add_provider(self, provider: ResourceProvider | ResourceCallable) -> None:
        """Add a resource provider or tool callable.

        Args:
            provider: Either a ResourceProvider instance or a callable
                     returning tools. Callables are automatically wrapped.
        """
        from llmling_agent.resource_providers.base import ResourceProvider

        if isinstance(provider, ResourceProvider):
            self._resource_providers.append(provider)
        else:
            # Wrap old-style callable in ResourceProvider
            wrapped = CallableResourceProvider(provider)
            self._resource_providers.append(wrapped)

    def remove_provider(self, provider: ResourceProvider | ResourceCallable) -> None:
        """Remove a resource provider."""
        from llmling_agent.resource_providers.base import ResourceProvider

        if isinstance(provider, ResourceProvider):
            self._resource_providers.remove(provider)
        else:
            # Find and remove wrapped callable
            for p in self._resource_providers:
                if (
                    isinstance(p, CallableResourceProvider)
                    and p.tool_callable == provider
                ):
                    self._resource_providers.remove(p)
                    break

    def reset_states(self):
        """Reset all tools to their default enabled states."""
        for info in self.values():
            info.enabled = True

    @property
    def _error_class(self) -> type[ToolError]:
        """Error class for tool operations."""
        return ToolError

    def _validate_item(self, item: ToolInfo | ToolType | dict[str, Any]) -> ToolInfo:
        """Validate and convert items before registration."""
        match item:
            case ToolInfo():
                return item
            case LLMCallableTool():
                return ToolInfo(callable=item)
            case str():
                item = LLMCallableTool.from_callable(item)
                return ToolInfo(callable=item)
            case _ if callable(item):
                tool = LLMCallableTool.from_callable(item)
                return ToolInfo(callable=tool)
            case {"callable": callable_item, **config} if callable(
                callable_item
            ) or isinstance(callable_item, LLMCallableTool):
                # First convert callable to LLMCallableTool if needed
                tool = (
                    callable_item
                    if isinstance(callable_item, LLMCallableTool)
                    else LLMCallableTool.from_callable(callable_item)
                )

                # Get valid fields from ToolInfo dataclass (excluding 'callable')
                valid_keys = {f.name for f in fields(ToolInfo)} - {"callable"}
                tool_config = {k: v for k, v in config.items() if k in valid_keys}

                return ToolInfo(callable=tool, **tool_config)  # type: ignore

            case _:
                typ = type(item)
                msg = f"Item must be ToolInfo, LLMCallableTool, or callable. Got {typ}"
                raise ToolError(msg)

    def enable_tool(self, tool_name: str):
        """Enable a previously disabled tool."""
        if tool_name not in self:
            msg = f"Tool not found: {tool_name}"
            raise ToolError(msg)
        tool_info = self[tool_name]
        tool_info.enabled = True
        self.events.changed(tool_name, tool_info)
        logger.debug("Enabled tool: %s", tool_name)

    def disable_tool(self, tool_name: str):
        """Disable a tool."""
        if tool_name not in self:
            msg = f"Tool not found: {tool_name}"
            raise ToolError(msg)
        tool_info = self[tool_name]
        tool_info.enabled = False
        self.events.changed(tool_name, tool_info)
        logger.debug("Disabled tool: %s", tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is currently enabled."""
        return self[tool_name].enabled if tool_name in self else False

    async def list_tools(self) -> dict[str, bool]:
        """Get a mapping of all tools and their enabled status."""
        return {tool.name: tool.enabled for tool in await self.get_tools()}

    async def get_tools(
        self,
        state: Literal["all", "enabled", "disabled"] = "all",
    ) -> list[ToolInfo]:
        """Get tool objects based on filters."""
        tools: list[ToolInfo] = []

        # Get tools from registry
        tools.extend(t for t in self.values() if t.matches_filter(state))

        # Get tools from providers
        for provider in self._resource_providers:
            try:
                provider_tools = await provider.get_tools()
                tools.extend(t for t in provider_tools if t.matches_filter(state))
            except Exception:
                logger.exception("Failed to get tools from provider: %r", provider)
                continue
        # Sort by priority if any have non-default priority
        if any(t.priority != 100 for t in tools):  # noqa: PLR2004
            tools.sort(key=lambda t: t.priority)

        return tools

    async def get_tool_names(
        self, state: Literal["all", "enabled", "disabled"] = "all"
    ) -> set[str]:
        """Get tool names based on state."""
        return {t.name for t in await self.get_tools() if t.matches_filter(state)}

    def register_tool(
        self,
        tool: LLMCallableTool | AnyCallable | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        enabled: bool = True,
        source: ToolSource = "runtime",
        priority: int = 100,
        requires_confirmation: bool = False,
        requires_capability: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ToolInfo:
        """Register a new tool with custom settings.

        Args:
            tool: Tool to register (callable, LLMCallableTool, or import path)
            enabled: Whether tool is initially enabled
            name_override: Optional name override for the tool
            description_override: Optional description override for the tool
            source: Tool source (runtime/agent/builtin/dynamic)
            priority: Execution priority (lower = higher priority)
            requires_confirmation: Whether tool needs confirmation
            requires_capability: Optional capability needed to use tool
            metadata: Additional tool metadata

        Returns:
            Created ToolInfo instance
        """
        # First convert to basic ToolInfo
        match tool:
            case LLMCallableTool():
                llm_tool = tool
                llm_tool.name = name_override or llm_tool.name
                llm_tool.description = description_override or llm_tool.description
            case _:
                llm_tool = LLMCallableTool.from_callable(
                    tool,
                    name_override=name_override,
                    description_override=description_override,
                )

        if llm_tool.description and len(llm_tool.description) > MAX_LEN_DESCRIPTION:
            msg = f"Too long description for {tool}"
            raise ToolError(msg)
        tool_info = ToolInfo(
            llm_tool,
            enabled=enabled,
            source=source,
            priority=priority,
            requires_confirmation=requires_confirmation,
            requires_capability=requires_capability,
            metadata=metadata or {},
        )
        # Register the tool
        self.register(tool_info.name, tool_info)
        return tool_info

    def register_worker(
        self,
        worker: AnyAgent[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
        parent: AnyAgent[Any, Any] | None = None,
    ) -> ToolInfo:
        """Register an agent as a worker tool.

        Args:
            worker: Agent to register as worker
            name: Optional name override for the worker tool
            reset_history_on_run: Whether to clear history before each run
            pass_message_history: Whether to pass parent's message history
            share_context: Whether to pass parent's context/deps
            parent: Optional parent agent for history/context sharing
        """
        tool = worker.to_agent_tool(
            parent=parent,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            share_context=share_context,
        )
        msg = "Registering worker %s as tool %s"
        logger.debug(msg, worker.name, tool.name)
        return self.register_tool(tool, source="agent", metadata={"agent": worker.name})

    def reset(self):
        """Reset tool states."""
        old_tools = {i.name: i.enabled for i in self._items.values()}
        self.reset_states()
        new_tools = {i.name: i.enabled for i in self._items.values()}

        event = self.ToolStateReset(old_tools, new_tools)
        self.tool_states_reset.emit(event)

    @contextmanager
    def temporary_tools(
        self,
        tools: ToolType | Sequence[ToolType],
        *,
        exclusive: bool = False,
    ) -> Iterator[list[ToolInfo]]:
        """Temporarily register tools.

        Args:
            tools: Tool(s) to register
            exclusive: Whether to temporarily disable all other tools

        Yields:
            List of registered tool infos

        Example:
            ```python
            with tool_manager.temporary_tools([tool1, tool2], exclusive=True) as tools:
                # Only tool1 and tool2 are available
                await agent.run(prompt)
            # Original tool states are restored
            ```
        """
        # Normalize inputs to lists
        tools_list: list[ToolType] = (
            [tools] if not isinstance(tools, Sequence) else list(tools)
        )

        # Store original tool states if exclusive
        original_states: dict[str, bool] = {}
        if exclusive:
            original_states = {name: tool.enabled for name, tool in self.items()}
            # Disable all existing tools
            for t in self.values():
                t.enabled = False

        # Register all tools
        registered_tools: list[ToolInfo] = []
        try:
            for tool in tools_list:
                tool_info = self.register_tool(tool)
                registered_tools.append(tool_info)
            yield registered_tools

        finally:
            # Remove temporary tools
            for tool_info in registered_tools:
                del self[tool_info.name]

            # Restore original tool states if exclusive
            if exclusive:
                for name_, was_enabled in original_states.items():
                    if t := self.get(name_):
                        t.enabled = was_enabled

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        enabled: bool = True,
        source: ToolSource = "runtime",
        priority: int = 100,
        requires_confirmation: bool = False,
        requires_capability: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Callable[[AnyCallable], AnyCallable]:
        """Decorator to register a function as a tool.

        Args:
            name: Optional override for tool name (defaults to function name)
            description: Optional description override
            enabled: Whether tool is initially enabled
            source: Tool source type
            priority: Execution priority (lower = higher)
            requires_confirmation: Whether tool needs confirmation
            requires_capability: Optional required capability
            metadata: Additional tool metadata

        Returns:
            Decorator function that registers the tool

        Example:
            @tool_manager.register(
                name="search_docs",
                description="Search documentation",
                requires_confirmation=True
            )
            async def search(query: str) -> str:
                '''Search the docs.'''
                return "Results..."
        """

        def decorator(func: AnyCallable) -> AnyCallable:
            self.register_tool(
                func,
                name_override=name,
                description_override=description,
                enabled=enabled,
                source=source,
                priority=priority,
                requires_confirmation=requires_confirmation,
                requires_capability=requires_capability,
                metadata=metadata,
            )
            return func

        return decorator
