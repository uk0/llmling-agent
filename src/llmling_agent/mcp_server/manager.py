"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self

from llmling.prompts import PromptMessage, StaticPrompt
from mcp.types import TextResourceContents

from llmling_agent.log import get_logger
from llmling_agent.mcp_server.client import MCPClient
from llmling_agent.models.content import AudioBase64Content, ImageBase64Content
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)
from llmling_agent_config.resources import ResourceInfo
from llmling_agent_providers.base import UsageLimits


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp import types
    from mcp.client.session import RequestContext
    from mcp.types import Prompt as MCPPrompt, Resource as MCPResource

    from llmling_agent.mcp_server.progress import ProgressHandler
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.content import BaseContent
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


async def convert_mcp_prompt(client: MCPClient, prompt: MCPPrompt) -> StaticPrompt:
    """Convert MCP prompt to StaticPrompt."""
    from mcp.types import TextContent

    result = await client.get_prompt(prompt.name)
    messages = [
        PromptMessage(role="system", content=message.content.text)
        for message in result.messages
        if isinstance(message.content, TextContent | TextResourceContents)
    ]
    desc = prompt.description or "No description provided"
    return StaticPrompt(name=prompt.name, description=desc, messages=messages)


async def convert_mcp_resource(resource: MCPResource) -> ResourceInfo:
    """Convert MCP resource to ResourceInfo."""
    return ResourceInfo(
        name=resource.name, uri=str(resource.uri), description=resource.description
    )


class MCPManager(ResourceProvider):
    """Manages MCP server connections and tools."""

    def __init__(
        self,
        name: str = "mcp",
        owner: str | None = None,
        servers: Sequence[MCPServerConfig | str] | None = None,
        context: NodeContext | None = None,
        progress_handler: ProgressHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        super().__init__(name, owner=owner)
        self.servers: list[MCPServerConfig] = []
        for server in servers or []:
            self.add_server_config(server)
        self.context = context
        self.clients: dict[str, MCPClient] = {}
        self.exit_stack = AsyncExitStack()
        self._progress_handler = progress_handler
        self._accessible_roots = accessible_roots

    @property
    def requires_async(self) -> bool:
        return True

    def add_server_config(self, server: MCPServerConfig | str):
        """Add a new MCP server to the manager."""
        server = (
            StdioMCPServerConfig.from_string(server)
            if isinstance(server, str)
            else server
        )
        self.servers.append(server)

    def __repr__(self) -> str:
        return f"MCPManager({self.servers!r})"

    async def __aenter__(self) -> Self:
        try:
            # Setup directly provided servers
            for server in self.servers:
                await self.setup_server(server)

            # Setup servers from context if available
            if self.context and self.context.config and self.context.config.mcp_servers:
                for server in self.context.config.get_mcp_servers():
                    await self.setup_server(server)

        except Exception as e:
            # Clean up in case of error
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize MCP manager"
            raise RuntimeError(msg) from e

        return self

    async def __aexit__(self, *exc):
        await self.cleanup()

    async def _elicitation_callback(
        self,
        context: RequestContext,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Handle elicitation requests from MCP server."""
        from mcp import types

        from llmling_agent.agent.context import AgentContext

        if self.context and isinstance(self.context, AgentContext):
            return await self.context.handle_elicitation(params)
        return types.ErrorData(
            code=types.INVALID_REQUEST,
            message="Elicitation not supported - no agent context available",
        )

    async def _sampling_callback(
        self,
        context: RequestContext,
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.ErrorData:
        """Handle MCP sampling by creating a new agent with specified preferences."""
        from mcp import types

        from llmling_agent.agent import Agent

        try:
            # Convert MCP messages to prompts for the agent
            prompts: list[BaseContent | str] = []
            for mcp_msg in params.messages:
                match mcp_msg.content:
                    case types.TextContent() as text_content:
                        prompts.append(text_content.text)
                    case types.ImageContent() as image_content:
                        # Convert to our ImageBase64Content for actual processing
                        our_image = ImageBase64Content(
                            data=image_content.data,
                            mime_type=image_content.mimeType,
                        )
                        prompts.append(our_image)
                    case types.AudioContent() as audio_content:
                        # Convert to our AudioBase64Content for actual processing
                        our_audio = AudioBase64Content(
                            data=audio_content.data,
                            format=audio_content.mimeType.removeprefix("audio/"),
                        )
                        prompts.append(our_audio)

            # Extract model from preferences
            model = None
            if (
                params.modelPreferences
                and params.modelPreferences.hints
                and params.modelPreferences.hints[0].name
            ):
                model = params.modelPreferences.hints[0].name

            # Create usage limits from sampling parameters
            usage_limits = UsageLimits(
                output_tokens_limit=params.maxTokens,
                request_limit=1,  # Single sampling request
            )

            # TODO: Apply temperature from params.temperature
            # Currently no direct way to pass temperature to Agent constructor
            # May need provider-level configuration or runtime model settings

            # Create agent with sampling parameters
            agent: Agent[Any] = Agent(
                name="mcp-sampling-agent",
                model=model,
                system_prompt=params.systemPrompt or "",
                session=False,  # Don't store history for sampling
            )

            async with agent:
                # Pass all prompts directly to the agent
                result = await agent.run(
                    *prompts,
                    store_history=False,
                    usage_limits=usage_limits,
                )

                return types.CreateMessageResult(
                    role="assistant",
                    content=types.TextContent(type="text", text=str(result.content)),
                    model=result.model or "unknown",
                    stopReason="endTurn",  # Could detect actual stop reason from result
                )

        except Exception as e:
            logger.exception("Sampling failed")
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Sampling failed: {e!s}",
            )

    async def setup_server(self, config: MCPServerConfig):
        """Set up a single MCP server connection."""
        if not config.enabled:
            return
        env = config.get_env_vars()
        match config:
            case StdioMCPServerConfig():
                client = MCPClient(
                    transport_mode="stdio",
                    elicitation_callback=self._elicitation_callback,
                    sampling_callback=self._sampling_callback,
                    progress_handler=self._progress_handler,
                    accessible_roots=self._accessible_roots,
                )
                client = await self.exit_stack.enter_async_context(client)
                await client.connect(config.command, args=config.args, env=env)
                client_id = f"{config.command}_{' '.join(config.args)}"
            case SSEMCPServerConfig():
                client = MCPClient(
                    transport_mode="sse",
                    elicitation_callback=self._elicitation_callback,
                    sampling_callback=self._sampling_callback,
                    progress_handler=self._progress_handler,
                    accessible_roots=self._accessible_roots,
                )
                client = await self.exit_stack.enter_async_context(client)
                await client.connect("", [], url=config.url, env=env)
                client_id = f"sse_{config.url}"
            case StreamableHTTPMCPServerConfig():
                client = MCPClient(
                    transport_mode="streamable-http",
                    elicitation_callback=self._elicitation_callback,
                    sampling_callback=self._sampling_callback,
                    progress_handler=self._progress_handler,
                    accessible_roots=self._accessible_roots,
                )
                client = await self.exit_stack.enter_async_context(client)
                await client.connect("", [], url=config.url, env=env)
                client_id = f"streamable_http_{config.url}"
        self.clients[client_id] = client

    async def get_tools(self) -> list[Tool]:
        """Get all tools from all connected servers."""
        from llmling_agent.tools.base import Tool

        tools: list[Tool] = []
        for client in self.clients.values():
            for tool in client._available_tools:
                try:
                    fn = client.create_tool_callable(tool)
                    meta = {"mcp_tool": tool.name}
                    tool_info = Tool.from_callable(fn, source="mcp", metadata=meta)
                    tools.append(tool_info)
                    logger.debug("Registered MCP tool: %s", tool.name)
                except Exception:
                    msg = "Failed to create tool from MCP tool: %s"
                    logger.exception(msg, tool.name)
                    continue

        return tools

    async def list_prompts(self) -> list[StaticPrompt]:
        """Get all available prompts from MCP servers."""
        prompts = []
        for client in self.clients.values():
            try:
                result = await client.list_prompts()
            except Exception:
                logger.exception("Failed to get prompts from MCP server")
            for prompt in result.prompts:
                try:
                    converted = await convert_mcp_prompt(client, prompt)
                    prompts.append(converted)
                except Exception:
                    logger.exception("Failed to convert prompt: %s", prompt.name)
        return prompts

    async def list_resources(self) -> list[ResourceInfo]:
        """Get all available resources from MCP servers."""
        resources = []
        for client in self.clients.values():
            try:
                result = await client.list_resources()
            except Exception:
                logger.exception("Failed to get resources from MCP server")
            for resource in result.resources:
                try:
                    converted = await convert_mcp_resource(resource)
                    resources.append(converted)
                except Exception:
                    logger.exception("Failed to convert resource: %s", resource.name)
        return resources

    async def cleanup(self) -> None:
        """Clean up all MCP connections."""
        try:
            try:
                # Clean up exit stack (which includes MCP clients)
                await self.exit_stack.aclose()
            except RuntimeError as e:
                if "different task" in str(e):
                    # Handle task context mismatch
                    current_task = asyncio.current_task()
                    if current_task:
                        loop = asyncio.get_running_loop()
                        await loop.create_task(self.exit_stack.aclose())
                else:
                    raise

            self.clients.clear()

        except Exception as e:
            msg = "Error during MCP manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    @property
    def active_servers(self) -> list[str]:
        """Get IDs of active servers."""
        return list(self.clients)
